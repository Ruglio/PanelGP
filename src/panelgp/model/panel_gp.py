"""The PanelGP model: sparse variational bilinear GP for return panels.

Notation follows the paper. ``beta`` are smooth loadings at characteristics,
``factor`` realized factors at dates. The variational state is whitened
(Appendix C.3) and stacks H*M inducing loadings and H*T training factors.
Returns are scaled by their training root mean square (Appendix C.1); all
public predictions are in original return units.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any

import torch
from torch.nn import functional

from .components import (
    FullGaussian,
    LoadingInnovationLikelihood,
    affine_kernel,
    inverse_softplus,
    rbf_kernel,
    stable_cholesky,
)

# Fixed numerical choices shared by every experiment (Appendices C.3 and C.4).
JITTER = 1.0e-7  # initial diagonal term of covariance factorizations
MIN_VARIANCE = 1.0e-8  # floor added to positive parameters
OBSERVATION_FRACTION = 0.1  # initial sigma_r^2 in scaled units
LOADING_NOISE_RATIO = 0.1  # initial tau_beta / eta
FACTOR_CONSTANT_FRACTION = 0.1  # initial b_h
FACTOR_INNOVATION_FRACTION = 0.1  # initial rho_h
POSTERIOR_MEAN_SCALE = 1.0e-3  # s.d. of the initial whitened mean
POSTERIOR_INIT_VARIANCE = 1.0  # initial whitened covariance is the identity


@dataclass(frozen=True)
class ModelSettings:
    """Specification choices that differ across the paper's experiments."""

    hidden_dim: int = 2
    loading_kernel: str = "rbf"  # "rbf" or "affine"
    factor_kernel: str = "rbf"  # "rbf" (RBF + constant) or "constant"
    inducing_points: int = 128
    loading_lengthscale: float = 1.0  # initial ARD lengthscale, all coordinates
    factor_lengthscale: float = 1.0
    shared_innovations: bool = False  # isotropic D_beta and D_phi (simulation)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> ModelSettings:
        values = dict(config.get("model", config))
        unknown = set(values) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown model settings: {sorted(unknown)}")
        settings = cls(**values)
        if settings.loading_kernel not in {"rbf", "affine"}:
            raise ValueError("loading_kernel must be 'rbf' or 'affine'.")
        if settings.factor_kernel not in {"rbf", "constant"}:
            raise ValueError("factor_kernel must be 'rbf' or 'constant'.")
        return settings


@dataclass(frozen=True)
class RowMoments:
    """Joint q of each row's smooth loading and realized factor (A.9), scaled units."""

    beta_mean: torch.Tensor
    factor_mean: torch.Tensor
    beta_covariance: torch.Tensor
    factor_covariance: torch.Tensor
    beta_factor_covariance: torch.Tensor


@dataclass(frozen=True)
class PredictiveMoments:
    mean: torch.Tensor
    variance: torch.Tensor


@dataclass(frozen=True)
class PredictiveCumulants:
    mean: torch.Tensor
    variance: torch.Tensor
    third: torch.Tensor
    fourth: torch.Tensor
    skewness: torch.Tensor
    excess_kurtosis: torch.Tensor


@dataclass(frozen=True)
class CrossSectionMoments:
    """Predictive mean and covariance of one cross section, eqs. (24) and (B.9).

    The innovation terms are the diagonal components of eq. (27):
    ``factor_innovation`` is V^phi, ``loading_innovation`` is V^beta,
    ``innovation_interaction`` is tr(D_beta D_phi).
    """

    mean: torch.Tensor
    covariance: torch.Tensor
    factor_innovation: torch.Tensor
    loading_innovation: torch.Tensor
    innovation_interaction: torch.Tensor
    observation: torch.Tensor


def _positive(
    shape: Sequence[int], value: float, like: torch.Tensor
) -> torch.nn.Parameter:
    initial = torch.full(
        tuple(shape), float(value), device=like.device, dtype=like.dtype
    )
    return torch.nn.Parameter(inverse_softplus(initial))


class PanelGP(torch.nn.Module):
    """Variational bilinear GP for panel-return distributions."""

    def __init__(
        self,
        *,
        input_dim: int,
        time_features: torch.Tensor,
        settings: ModelSettings,
        feature_names: Sequence[str] | None = None,
    ) -> None:
        super().__init__()
        hidden, inducing = settings.hidden_dim, settings.inducing_points
        self.settings = settings
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden)
        self.total_inducing = int(inducing)
        self.time_dim = int(time_features.shape[1])
        self.num_train_dates = int(time_features.shape[0])
        self.feature_names = tuple(
            feature_names or [f"x{i + 1}" for i in range(input_dim)]
        )
        self.register_buffer("time_features", time_features.detach().clone())
        self.register_buffer("target_std", time_features.new_ones(()))

        # Initial loading output scale from the variance budget (C.6) with a unit
        # kernel diagonal; initialize_inducing rescales it for the affine kernel.
        smooth_budget = (1.0 - OBSERVATION_FRACTION) / (1.0 + LOADING_NOISE_RATIO)
        loading_scale = smooth_budget / hidden
        self.mean_loading_kernel_diagonal_init = 1.0
        self.likelihood = LoadingInnovationLikelihood(
            hidden_dim=hidden,
            observation_variance=OBSERVATION_FRACTION,
            loading_variances=torch.full(
                (hidden,),
                LOADING_NOISE_RATIO * loading_scale,
                device=time_features.device,
                dtype=time_features.dtype,
            ),
            shared=settings.shared_innovations,
            jitter=MIN_VARIANCE,
        )

        # Factor-kernel shares (a_h, b_h, rho_h) are a softmax with the RBF
        # logit fixed at zero (C.4). A constant kernel removes the RBF share;
        # shared innovations use one sigmoid for rho and a two-way softmax.
        macro_fraction = 1.0 - FACTOR_CONSTANT_FRACTION - FACTOR_INNOVATION_FRACTION
        if settings.factor_kernel == "constant" or settings.shared_innovations:
            innovation_logit = math.log(
                FACTOR_INNOVATION_FRACTION / (1.0 - FACTOR_INNOVATION_FRACTION)
            )
        else:
            innovation_logit = math.log(FACTOR_INNOVATION_FRACTION / macro_fraction)
        rbf_factors = settings.factor_kernel == "rbf"

        # Registration order fixes the parameter order, and hence the summation
        # order of gradient-norm clipping: keep it for bitwise reproducibility.
        self.raw_loading_outputscales = _positive(
            (hidden,), loading_scale, time_features
        )
        self.raw_factor_lengthscales = _positive(
            (hidden, self.time_dim), settings.factor_lengthscale, time_features
        )
        self.raw_factor_lengthscales.requires_grad_(rbf_factors)
        self.raw_factor_innovation_fraction = torch.nn.Parameter(
            time_features.new_full(
                (1 if settings.shared_innovations else hidden,), innovation_logit
            )
        )
        self.raw_factor_constant_fraction = torch.nn.Parameter(
            time_features.new_full(
                (hidden,), math.log(FACTOR_CONSTANT_FRACTION / macro_fraction)
            ),
            requires_grad=rbf_factors,
        )
        self.inducing_locations = torch.nn.Parameter(
            time_features.new_zeros(hidden, inducing, input_dim)
        )
        self.raw_loading_lengthscales = _positive(
            (hidden, input_dim), settings.loading_lengthscale, time_features
        )
        self.raw_loading_lengthscales.requires_grad_(settings.loading_kernel == "rbf")
        self.posterior = FullGaussian(
            hidden * (inducing + self.num_train_dates),
            init_variance=POSTERIOR_INIT_VARIANCE,
            mean_init_scale=POSTERIOR_MEAN_SCALE,
            min_variance=MIN_VARIANCE,
            device=time_features.device,
            dtype=time_features.dtype,
        )

    # ------------------------------------------------------------------
    # Constrained parameters
    # ------------------------------------------------------------------
    @property
    def loading_outputscales(self) -> torch.Tensor:
        return functional.softplus(self.raw_loading_outputscales) + MIN_VARIANCE

    @property
    def loading_lengthscales(self) -> torch.Tensor:
        return functional.softplus(self.raw_loading_lengthscales) + MIN_VARIANCE

    @property
    def factor_lengthscales(self) -> torch.Tensor:
        return functional.softplus(self.raw_factor_lengthscales) + MIN_VARIANCE

    @property
    def factor_kernel_fractions(self) -> torch.Tensor:
        """Shares (a_h, b_h, rho_h) of the factor prior variance; rows sum to one."""
        innovation_logit = self.raw_factor_innovation_fraction
        if innovation_logit.numel() == 1:
            innovation_logit = innovation_logit.expand(self.hidden_dim)
        if self.settings.factor_kernel == "constant":
            macro_logit = innovation_logit.new_full((self.hidden_dim,), -math.inf)
            constant_logit = torch.zeros_like(innovation_logit)
        else:
            constant_logit = self.raw_factor_constant_fraction
            macro_logit = torch.zeros_like(innovation_logit)
        if self.settings.shared_innovations:
            innovation = torch.sigmoid(innovation_logit)[:, None]
            smooth = torch.softmax(
                torch.stack((macro_logit, constant_logit), dim=-1), dim=-1
            )
            return torch.cat(((1.0 - innovation) * smooth, innovation), dim=-1)
        logits = torch.stack((macro_logit, constant_logit, innovation_logit), dim=-1)
        return torch.softmax(logits, dim=-1)

    # ------------------------------------------------------------------
    # Initialization from training data
    # ------------------------------------------------------------------
    @torch.no_grad()
    def fit_target_scaler(self, target: torch.Tensor) -> None:
        """Set s_r^2 to the mean squared training return, eq. (C.1)."""
        self.target_std.copy_(target.square().mean().sqrt().clamp_min(MIN_VARIANCE))

    @torch.no_grad()
    def initialize_inducing(self, x: torch.Tensor) -> None:
        """Draw distinct training inputs per component (repeating if too few).

        For the affine kernel, also rescale the initial output scale and loading
        innovation so the average prior return variance matches the RBF case.
        """
        if self.settings.loading_kernel == "affine":
            diagonal = x.square().sum(dim=-1).mean() + 1.0
            self.mean_loading_kernel_diagonal_init = float(diagonal.item())
            scale = (1.0 - OBSERVATION_FRACTION) / (
                self.hidden_dim * (diagonal + LOADING_NOISE_RATIO)
            )
            self.raw_loading_outputscales.copy_(
                inverse_softplus(scale.expand_as(self.raw_loading_outputscales))
            )
            self.likelihood.raw_loading_variance.copy_(
                inverse_softplus(
                    (LOADING_NOISE_RATIO * scale).expand_as(
                        self.likelihood.raw_loading_variance
                    )
                )
            )
        candidates = torch.unique(x, dim=0)
        available = candidates.shape[0]
        for hidden in range(self.hidden_dim):
            if available < self.total_inducing:
                repeats = math.ceil(self.total_inducing / available)
                selected = candidates.repeat(repeats, 1)[: self.total_inducing]
            else:
                order = torch.randperm(available, device=candidates.device)
                selected = candidates[order[: self.total_inducing]]
            self.inducing_locations[hidden].copy_(selected)

    # ------------------------------------------------------------------
    # Kernel maps
    # ------------------------------------------------------------------
    def _split_posterior(self) -> tuple[torch.Tensor, ...]:
        """Mean and Cholesky rows of the inducing-loading and training-factor blocks."""
        loading_size = self.hidden_dim * self.total_inducing
        mean = self.posterior.mean
        cholesky = self.posterior.factor
        return (
            mean[:loading_size].reshape(self.hidden_dim, self.total_inducing),
            mean[loading_size:].reshape(self.hidden_dim, self.num_train_dates),
            cholesky[:loading_size].reshape(self.hidden_dim, self.total_inducing, -1),
            cholesky[loading_size:].reshape(self.hidden_dim, self.num_train_dates, -1),
        )

    def _loading_kernel(self, a, b, *, lengthscale, outputscale) -> torch.Tensor:
        if self.settings.loading_kernel == "affine":
            return affine_kernel(a, b, outputscale=outputscale)
        return rbf_kernel(a, b, lengthscale=lengthscale, outputscale=outputscale)

    def _loading_maps(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Whitened maps L_uu^{-1} K_uz and residual prior variances, eq. (A.5)."""
        values = x.unsqueeze(0).expand(self.hidden_dim, -1, -1)
        inducing, lengthscale = self.inducing_locations, self.loading_lengthscales
        scale = self.loading_outputscales
        kzz = self._loading_kernel(
            inducing, inducing, lengthscale=lengthscale, outputscale=scale
        )
        chol, _ = stable_cholesky(kzz, jitter=JITTER)
        kzx = self._loading_kernel(
            inducing, values, lengthscale=lengthscale, outputscale=scale
        )
        maps = torch.linalg.solve_triangular(chol, kzx, upper=False).transpose(1, 2)
        diagonal = scale[:, None]
        if self.settings.loading_kernel == "affine":
            diagonal = diagonal * (values.square().sum(dim=-1) + 1)
        residual = diagonal - maps.square().sum(dim=-1)
        return maps, residual.clamp_min(0.0)

    def _weighted_loading_within_covariance(
        self, x: torch.Tensor, maps: torch.Tensor, hidden_weights: torch.Tensor
    ) -> torch.Tensor:
        """Cross-asset covariance retained by the conditional loading GP."""
        weights = hidden_weights.clamp_min(0.0)
        covariance = x.new_zeros(x.shape[0], x.shape[0])
        lengthscales, scales = self.loading_lengthscales, self.loading_outputscales
        for hidden in range(self.hidden_dim):
            kxx = self._loading_kernel(
                x, x, lengthscale=lengthscales[hidden], outputscale=scales[hidden]
            )
            row_map = maps[hidden]
            kxx.addmm_(row_map, row_map.T, beta=1.0, alpha=-1.0)
            covariance.add_(kxx.mul_(weights[hidden]))
        return 0.5 * (covariance + covariance.T)

    def _factor_prior(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Cholesky factor of the realized-factor prior, eq. (11), per component."""
        fractions = self.factor_kernel_fractions
        dates = self.time_features[None].expand(self.hidden_dim, -1, -1)
        macro = rbf_kernel(
            dates,
            dates,
            lengthscale=self.factor_lengthscales,
            outputscale=fractions[:, 0],
        )
        identity = torch.eye(
            self.num_train_dates, device=macro.device, dtype=macro.dtype
        )
        covariance = macro + fractions[:, 1, None, None] * torch.ones_like(macro)
        covariance = covariance + fractions[:, 2, None, None] * identity[None]
        return stable_cholesky(covariance, jitter=JITTER)

    def _factor_maps(
        self, context: torch.Tensor, time_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Factor maps for rows: training dates select their own row of the prior
        Cholesky factor; new dates (index -1) solve the conditional map and keep a
        residual that includes a fresh factor innovation."""
        chol, _ = self._factor_prior()
        rows = context.shape[0]
        maps = context.new_zeros(self.hidden_dim, rows, self.num_train_dates)
        residual = context.new_zeros(self.hidden_dim, rows)
        known = (time_index >= 0) & (time_index < self.num_train_dates)
        if bool(known.any()):
            maps[:, known] = chol[:, time_index[known].long(), :]
        unseen = ~known
        if bool(unseen.any()):
            query = context[unseen]
            fractions = self.factor_kernel_fractions
            cross = (
                rbf_kernel(
                    query[None].expand(self.hidden_dim, -1, -1),
                    self.time_features[None].expand(self.hidden_dim, -1, -1),
                    lengthscale=self.factor_lengthscales,
                    outputscale=fractions[:, 0],
                )
                + fractions[:, 1, None, None]
            )
            solve = torch.linalg.solve_triangular(
                chol, cross.transpose(1, 2), upper=False
            )
            query_map = solve.transpose(1, 2)
            maps[:, unseen] = query_map
            residual[:, unseen] = (1.0 - query_map.square().sum(dim=-1)).clamp_min(0.0)
        return maps, residual

    def row_moments(
        self, x: torch.Tensor, context: torch.Tensor, time_index: torch.Tensor
    ) -> RowMoments:
        """Joint q of (smooth loading, realized factor) for each row, before the
        loading innovation; the building block of training and prediction."""
        beta_map, beta_residual = self._loading_maps(x)
        factor_map, factor_residual = self._factor_maps(context, time_index)
        mean_b, mean_f, beta_root, factor_root = self._split_posterior()
        beta_mean = torch.einsum("hnm,hm->nh", beta_map, mean_b)
        factor_mean = torch.einsum("hnt,ht->nh", factor_map, mean_f)
        # Per-row square roots: beta_covariance[n] = root[n] @ root[n].T + residual.
        beta_row_root = torch.einsum("hnm,hmr->nhr", beta_map, beta_root)
        factor_row_root = torch.einsum("hnt,htr->nhr", factor_map, factor_root)
        beta_covariance = torch.einsum(
            "nhr,nkr->nhk", beta_row_root, beta_row_root
        ) + torch.diag_embed(beta_residual.T)
        factor_covariance = torch.einsum(
            "nhr,nkr->nhk", factor_row_root, factor_row_root
        ) + torch.diag_embed(factor_residual.T)
        cross = torch.einsum("nhr,nkr->nhk", beta_row_root, factor_row_root)
        return RowMoments(
            beta_mean, factor_mean, beta_covariance, factor_covariance, cross
        )

    # ------------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------------
    def expected_log_likelihood(
        self,
        *,
        x: torch.Tensor,
        context: torch.Tensor,
        time_index: torch.Tensor,
        target: torch.Tensor,
        samples: int,
        antithetic: bool = False,
    ) -> torch.Tensor:
        """Monte Carlo estimate of the likelihood term of the ELBO, eq. (A.11).

        One factor draw per distinct date is shared by all of its rows; the
        loading is integrated exactly given the draw. Antithetic sampling uses
        samples/2 Gaussian draws and their negatives.
        """
        if samples <= 0 or (antithetic and samples % 2):
            raise ValueError("Factor samples must be positive and even when paired.")
        moments = self.row_moments(x, context, time_index)
        unique, inverse = torch.unique(time_index, sorted=True, return_inverse=True)
        first = torch.full_like(unique, len(time_index))
        first.scatter_reduce_(
            0, inverse, torch.arange(len(time_index), device=x.device), reduce="amin"
        )
        factor_mean = moments.factor_mean[first]
        factor_covariance = moments.factor_covariance[first]
        chol, _ = stable_cholesky(factor_covariance, jitter=0.0)
        noise = torch.randn(
            int(samples) // 2 if antithetic else int(samples),
            unique.numel(),
            self.hidden_dim,
            device=x.device,
            dtype=x.dtype,
        )
        if antithetic:
            noise = torch.cat((noise, -noise), dim=0)
        factor_samples = factor_mean[None] + torch.einsum("uih,suh->sui", chol, noise)
        factor_samples = factor_samples[:, inverse]

        cross = moments.beta_factor_covariance
        gain_t = torch.cholesky_solve(cross.transpose(1, 2), chol[inverse])
        gain = gain_t.transpose(1, 2)
        centered = factor_samples - moments.factor_mean[None]
        conditional_beta_mean = moments.beta_mean[None] + torch.einsum(
            "nhk,snk->snh", gain, centered
        )
        conditional_beta_covariance = moments.beta_covariance - gain @ cross.transpose(
            1, 2
        )
        conditional_beta_covariance = 0.5 * (
            conditional_beta_covariance + conditional_beta_covariance.transpose(1, 2)
        )
        mean_return = torch.einsum("snh,snh->sn", conditional_beta_mean, factor_samples)
        beta_variance = torch.einsum(
            "snh,nhk,snk->sn",
            factor_samples,
            conditional_beta_covariance,
            factor_samples,
        ).clamp_min(0.0)
        log_prob = self.likelihood.integrated_log_prob(
            target=target / self.target_std,
            factor_samples=factor_samples,
            conditional_mean_return=mean_return,
            conditional_beta_variance=beta_variance,
        )
        return log_prob.mean(dim=0).sum()

    def kl_divergence(self) -> torch.Tensor:
        return self.posterior.kl_standard_normal()

    def parameter_groups(
        self,
    ) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
        """Variational parameters and model parameters (separate learning rates)."""
        variational = [p for p in self.posterior.parameters() if p.requires_grad]
        ids = {id(p) for p in variational}
        model = [p for p in self.parameters() if p.requires_grad and id(p) not in ids]
        return variational, model

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def _innovation_variances(self, b, f, beta_diagonal, factor_diagonal, new_date):
        """V^phi, V^beta and tr(D_beta D_phi) of eq. (27). Training dates are
        realized posterior states and receive no fresh factor innovation."""
        rho = (
            self.factor_kernel_fractions[:, 2]
            * torch.as_tensor(new_date, device=b.device)[..., None]
        )
        loading = self.likelihood.loading_variances
        factor_term = (rho * (b.square() + beta_diagonal)).sum(dim=-1)
        interaction = (loading * rho).sum(dim=-1).expand_as(factor_term)
        loading_total = (loading * (f.square() + factor_diagonal)).sum(dim=-1)
        return (
            factor_term,
            loading_total.expand_as(factor_term) - interaction,
            interaction,
        )

    @torch.no_grad()
    def predictive_moments(
        self, x: torch.Tensor, context: torch.Tensor, time_index: torch.Tensor
    ) -> PredictiveMoments:
        """Predictive mean and variance of each row; rows may span several dates."""
        moments = self.row_moments(x, context, time_index)
        b, f = moments.beta_mean, moments.factor_mean
        beta_cov, factor_cov = moments.beta_covariance, moments.factor_covariance
        cross = moments.beta_factor_covariance
        mean = torch.einsum("nh,nh->n", b, f) + torch.diagonal(
            cross, dim1=-2, dim2=-1
        ).sum(dim=-1)
        factor_term = torch.einsum("nh,nhk,nk->n", b, factor_cov, b)
        loading_term = torch.einsum("nh,nhk,nk->n", f, beta_cov, f)
        interaction = 2.0 * torch.einsum("nh,nkh,nk->n", b, cross, f)
        interaction = interaction + torch.einsum("nhk,nkh->n", beta_cov, factor_cov)
        interaction = interaction + torch.einsum("nhk,nkh->n", cross, cross)
        _, loading_noise, innovation_interaction = self._innovation_variances(
            b,
            f,
            beta_cov.diagonal(dim1=-2, dim2=-1),
            factor_cov.diagonal(dim1=-2, dim2=-1),
            (time_index < 0) | (time_index >= self.num_train_dates),
        )
        residual = self.likelihood.observation_variance + (
            loading_noise + innovation_interaction
        )
        variance = (factor_term + loading_term + interaction + residual).clamp_min(
            MIN_VARIANCE
        )
        return PredictiveMoments(
            mean=self.target_std * mean, variance=variance * self.target_std.square()
        )

    @torch.no_grad()
    def predictive_cumulants(
        self, x: torch.Tensor, context: torch.Tensor, time_index: torch.Tensor
    ) -> PredictiveCumulants:
        """First four predictive cumulants of each row, Proposition 2.

        With z = [beta~, phi~] ~ N(m, S) and A the matrix with z'Az = beta~'phi~,
        kappa_k = 2^(k-1) (k-1)! [tr((A S)^k) + k m'(A S)^(k-1) A m], plus the
        observation variance in kappa_2.
        """
        moments = self.row_moments(x, context, time_index)
        rows, hidden = x.shape[0], self.hidden_dim
        joint_mean = torch.cat((moments.beta_mean, moments.factor_mean), dim=1)
        beta_covariance = moments.beta_covariance + torch.diag_embed(
            self.likelihood.loading_variances.expand(rows, -1)
        )
        cross = moments.beta_factor_covariance
        joint_covariance = torch.cat(
            (
                torch.cat((beta_covariance, cross), dim=2),
                torch.cat((cross.transpose(1, 2), moments.factor_covariance), dim=2),
            ),
            dim=1,
        )
        identity = torch.eye(2 * hidden, device=x.device, dtype=x.dtype)
        quadratic = torch.zeros_like(identity)
        quadratic[:hidden, hidden:] = 0.5 * torch.eye(
            hidden, device=x.device, dtype=x.dtype
        )
        quadratic[hidden:, :hidden] = quadratic[:hidden, hidden:].T
        matrix = quadratic @ joint_covariance
        right = torch.einsum("ij,nj->ni", quadratic, joint_mean)
        power = identity.expand(rows, -1, -1)
        cumulants = []
        for order in range(1, 5):
            next_power = power @ matrix
            trace = torch.diagonal(next_power, dim1=-2, dim2=-1).sum(dim=-1)
            noncentral = torch.einsum("ni,nij,nj->n", joint_mean, power, right)
            coefficient = float(2 ** (order - 1) * math.factorial(order - 1))
            cumulants.append(coefficient * (trace + order * noncentral))
            power = next_power
        scale = self.target_std
        variance = scale.square() * (
            cumulants[1] + self.likelihood.observation_variance
        )
        third, fourth = scale.pow(3) * cumulants[2], scale.pow(4) * cumulants[3]
        return PredictiveCumulants(
            mean=scale * cumulants[0],
            variance=variance,
            third=third,
            fourth=fourth,
            skewness=third / variance.pow(1.5),
            excess_kurtosis=fourth / variance.square(),
        )

    @torch.no_grad()
    def cross_section_moments(
        self, x: torch.Tensor, context: torch.Tensor, time_index: int = -1
    ) -> CrossSectionMoments:
        """Full predictive covariance of the assets of one date, eq. (B.9).

        ``context`` is the date's context row; ``time_index`` is -1 for a new
        date and a training index to reconstruct a realized training date.
        """
        context = context.reshape(1, self.time_dim)
        beta_map, beta_residual = self._loading_maps(x)
        factor_map, factor_residual = self._factor_maps(
            context, torch.tensor([time_index], device=x.device, dtype=torch.long)
        )
        mean_b, mean_f, beta_root, factor_root = self._split_posterior()
        b = torch.einsum("hnm,hm->nh", beta_map, mean_b)
        f = torch.einsum("hnt,ht->nh", factor_map, mean_f)[0]
        beta_row_root = torch.einsum("hnm,hmr->nhr", beta_map, beta_root)
        factor_row_root = torch.einsum("hnt,htr->nhr", factor_map, factor_root)[0]
        beta_within_diagonal = beta_residual.T
        factor_cov = factor_row_root @ factor_row_root.T + torch.diag(
            factor_residual[:, 0]
        )
        cross = torch.einsum("nhr,kr->nhk", beta_row_root, factor_row_root)
        factor_diagonal = torch.diagonal(factor_cov)

        mean = b @ f + torch.diagonal(cross, dim1=-2, dim2=-1).sum(dim=-1)
        # Terms of eq. (B.9): factor uncertainty, loading uncertainty, interactions.
        covariance = b @ factor_cov @ b.T
        factor_variance = covariance.diagonal().clone()
        loading_root = torch.einsum("h,nhr->nr", f, beta_row_root)
        loading_component = loading_root @ loading_root.T
        loading_variance = loading_component.diagonal().clone() + torch.einsum(
            "h,nh->n", f.square(), beta_within_diagonal
        )
        covariance = covariance + loading_component
        mean_interaction = b @ torch.einsum("h,nhk->nk", f, cross).T
        interaction = mean_interaction + mean_interaction.T
        interaction = interaction + torch.einsum(
            "nhr,hk,mkr->nm", beta_row_root, factor_cov, beta_row_root
        )
        interaction = interaction + torch.einsum("nhk,mkh->nm", cross, cross)
        interaction_variance = interaction.diagonal().clone() + torch.einsum(
            "h,nh->n", factor_diagonal, beta_within_diagonal
        )
        covariance = covariance + interaction
        covariance = covariance + self._weighted_loading_within_covariance(
            x, beta_map, f.square() + factor_diagonal
        )
        covariance = 0.5 * (covariance + covariance.T)

        observation = self.likelihood.observation_variance.expand(x.shape[0]).clone()
        factor_innovation, loading_innovation, innovation_interaction = (
            self._innovation_variances(
                b,
                f,
                beta_row_root.square().sum(dim=-1) + beta_within_diagonal,
                factor_diagonal,
                not 0 <= time_index < self.num_train_dates,
            )
        )
        residual = observation + (loading_innovation + innovation_interaction)
        covariance.diagonal().copy_(
            (
                factor_variance + loading_variance + interaction_variance + residual
            ).clamp_min(MIN_VARIANCE)
        )
        scale2 = self.target_std.square()
        return CrossSectionMoments(
            mean=self.target_std * mean,
            covariance=covariance * scale2,
            factor_innovation=factor_innovation * scale2,
            loading_innovation=loading_innovation * scale2,
            innovation_interaction=innovation_interaction * scale2,
            observation=observation * scale2,
        )

    @torch.no_grad()
    def diagnostics(self) -> dict[str, Any]:
        """Fitted hyperparameters in scaled units (returns divided by target_std)."""

        def values(tensor: torch.Tensor) -> list:
            return tensor.detach().cpu().tolist()

        return {
            "feature_names": list(self.feature_names),
            "target_std": float(self.target_std),
            "observation_variance": float(self.likelihood.observation_variance),
            "loading_outputscales": values(self.loading_outputscales),
            "loading_innovation_variances": values(self.likelihood.loading_variances),
            "loading_lengthscales": values(self.loading_lengthscales),
            "factor_kernel_fractions": values(self.factor_kernel_fractions),
            "factor_lengthscales": values(self.factor_lengthscales),
            "mean_loading_kernel_diagonal_init": self.mean_loading_kernel_diagonal_init,
        }
