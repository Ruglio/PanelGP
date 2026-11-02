"""Kernels, the dense Gaussian variational family, and the integrated likelihood."""

from __future__ import annotations

import math

import torch
from torch.nn import functional


def inverse_softplus(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp_min(1.0e-12)
    return value + torch.log(-torch.expm1(-value))


def _scale(outputscale: torch.Tensor | float, like: torch.Tensor) -> torch.Tensor:
    """Reshape a per-component output scale to broadcast against kernel matrices."""
    scale = torch.as_tensor(outputscale, device=like.device, dtype=like.dtype)
    return scale.reshape(scale.shape + (1,) * (like.ndim - 1))


def rbf_kernel(
    x1: torch.Tensor,
    x2: torch.Tensor,
    *,
    lengthscale: torch.Tensor,
    outputscale: torch.Tensor | float,
) -> torch.Tensor:
    """ARD RBF kernel for inputs (N,D),(M,D) or component batches (H,N,D),(H,M,D)."""
    scale = lengthscale.unsqueeze(-2)
    squared_distance = torch.cdist(x1 / scale, x2 / scale).square()
    return _scale(outputscale, x1) * torch.exp(-0.5 * squared_distance)


def affine_kernel(
    x1: torch.Tensor, x2: torch.Tensor, *, outputscale: torch.Tensor | float
) -> torch.Tensor:
    """Affine kernel outputscale * (1 + x1'x2), with the same batching as rbf_kernel."""
    scale = _scale(outputscale, x1)
    return scale * (x1 @ x2.transpose(-2, -1)) + scale


def stable_cholesky(
    matrix: torch.Tensor, *, jitter: float, max_tries: int = 8
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cholesky factor of a (batch of) matrix plus a diagonal term.

    The term starts at ``jitter`` and grows tenfold until the factorization
    succeeds. Returns the factor and the stabilized matrix.
    """
    symmetric = 0.5 * (matrix + matrix.transpose(-2, -1))
    identity = torch.eye(matrix.shape[-1], device=matrix.device, dtype=matrix.dtype)
    current = float(jitter)
    for _ in range(max_tries):
        effective = symmetric + current * identity
        factor, info = torch.linalg.cholesky_ex(effective)
        if bool((info == 0).all()):
            return factor, effective
        current = 10.0 * max(current, float(torch.finfo(matrix.dtype).eps))
    raise RuntimeError("Cholesky factorization failed after adaptive jitter.")


class FullGaussian(torch.nn.Module):
    """Whitened variational Gaussian N(mean, L L') with a trainable dense Cholesky factor."""

    def __init__(
        self,
        dimension: int,
        *,
        init_variance: float,
        mean_init_scale: float,
        min_variance: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.dimension = int(dimension)
        self.min_std = math.sqrt(float(min_variance))
        self.mean = torch.nn.Parameter(
            mean_init_scale * torch.randn(dimension, device=device, dtype=dtype)
        )
        diagonal_std = math.sqrt(max(float(init_variance), float(min_variance)))
        diagonal = inverse_softplus(
            torch.full(
                (dimension,),
                max(diagonal_std - self.min_std, 1.0e-10),
                device=device,
                dtype=dtype,
            )
        )
        self.raw_factor = torch.nn.Parameter(torch.diag(diagonal))

    @property
    def factor(self) -> torch.Tensor:
        diagonal = functional.softplus(torch.diagonal(self.raw_factor)) + self.min_std
        return torch.tril(self.raw_factor, diagonal=-1) + torch.diag(diagonal)

    def kl_standard_normal(self) -> torch.Tensor:
        """KL(q || N(0, I)) of the whitened state."""
        factor = self.factor
        trace = factor.square().sum()
        logdet = 2.0 * torch.log(torch.diagonal(factor)).sum()
        return 0.5 * (trace + self.mean.square().sum() - self.dimension - logdet)


class LoadingInnovationLikelihood(torch.nn.Module):
    """Return likelihood after integrating the loading innovation, eq. (7)."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        observation_variance: float,
        loading_variances: torch.Tensor,
        shared: bool,
        jitter: float,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.jitter = float(jitter)
        values = loading_variances.mean().reshape(1) if shared else loading_variances
        self.raw_observation_variance = torch.nn.Parameter(
            inverse_softplus(
                torch.tensor(
                    max(observation_variance, 1.0e-12),
                    device=values.device,
                    dtype=values.dtype,
                )
            )
        )
        self.raw_loading_variance = torch.nn.Parameter(
            inverse_softplus(values.clamp_min(1.0e-12))
        )

    @property
    def shared(self) -> bool:
        return self.raw_loading_variance.numel() == 1

    @property
    def observation_variance(self) -> torch.Tensor:
        return functional.softplus(self.raw_observation_variance) + self.jitter

    @property
    def loading_variances(self) -> torch.Tensor:
        values = functional.softplus(self.raw_loading_variance)
        return values.expand(self.hidden_dim) if self.shared else values

    def integrated_log_prob(
        self,
        *,
        target: torch.Tensor,
        factor_samples: torch.Tensor,
        conditional_mean_return: torch.Tensor,
        conditional_beta_variance: torch.Tensor,
    ) -> torch.Tensor:
        """Expected log likelihood (A.11), exact in the loading given each factor draw."""
        variance = self.observation_variance + torch.einsum(
            "...h,h->...", factor_samples.square(), self.loading_variances
        )
        variance = variance.clamp_min(self.jitter)
        squared_error = (
            target[None] - conditional_mean_return
        ).square() + conditional_beta_variance.clamp_min(0.0)
        return -0.5 * (
            math.log(2.0 * math.pi) + torch.log(variance) + squared_error / variance
        )
