"""Predictive density of a single return by Fourier inversion (Appendix B.3)."""

from __future__ import annotations

import numpy as np
import torch
from scipy.integrate import quad_vec
from scipy.optimize import brentq


def quadratic_form_log_density(
    target, mean, covariance, observation_variance, *, tolerance=1e-9
):
    """Log density at ``target`` of z'Qz + e, z ~ N(mean, covariance), e ~ N(0, s2).

    Q has I/2 in its off-diagonal blocks, so z'Qz = beta'phi for z = [beta, phi].
    For each observation, the CGF is tilted at the root s of K'(s) = y and the
    tilted characteristic function is integrated adaptively. This is exact
    numerical inversion, not a saddlepoint approximation.
    """
    target, mean, covariance = (
        np.asarray(v, dtype=float) for v in (target, mean, covariance)
    )
    count, dim = mean.shape
    h = dim // 2
    q = np.block([[np.zeros((h, h)), np.eye(h) / 2], [np.eye(h) / 2, np.zeros((h, h))]])
    # Diagonalize the quadratic form in whitened coordinates.
    values, basis = np.linalg.eigh(0.5 * (covariance + covariance.swapaxes(1, 2)))
    root = basis * np.sqrt(np.maximum(values, 0))[:, None, :]
    eigenvalues, vectors = np.linalg.eigh(root.swapaxes(1, 2) @ q @ root)
    b = 2 * np.einsum("nji,jk,nk->ni", root, q, mean)
    delta2 = np.einsum("nji,nj->ni", vectors, b) ** 2
    c = np.einsum("ni,ij,nj->n", mean, q, mean)

    def cgf(s):
        d = 1 - 2 * eigenvalues * s[:, None]
        return (
            c * s
            - 0.5 * np.log(d).sum(1)
            + 0.5 * s * s * (delta2 / d).sum(1)
            + 0.5 * observation_variance * s * s
        )

    def derivative(x, lam, d2, offset):
        d = 1 - 2 * lam * x
        return (
            offset
            + (lam / d).sum()
            + (d2 * (x / d + lam * x * x / (d * d))).sum()
            + observation_variance * x
        )

    s = np.zeros(count)
    for i in range(count):
        lam, d2 = eigenvalues[i], delta2[i]
        arguments = (lam, d2, c[i] - target[i])
        lower = 1 / (2 * lam.min()) if lam.min() < 0 else -np.inf
        upper = 1 / (2 * lam.max()) if lam.max() > 0 else np.inf
        lower = np.nextafter(lower, 0) if np.isfinite(lower) else lower
        upper = np.nextafter(upper, 0) if np.isfinite(upper) else upper
        variance = (2 * lam * lam + d2).sum() + observation_variance
        width = max(
            1 / np.sqrt(variance), 2 * abs(target[i] - c[i] - lam.sum()) / variance
        )
        lo, hi = max(-width, lower), min(width, upper)
        while derivative(lo, *arguments) > 0:
            lo = max(2 * lo, lower)
        while derivative(hi, *arguments) < 0:
            hi = min(2 * hi, upper)
        s[i] = brentq(derivative, lo, hi, args=arguments, xtol=1e-12, rtol=1e-12)
    d = 1 - 2 * eigenvalues * s[:, None]
    sd = np.sqrt(
        (2 * eigenvalues**2 / d**2 + delta2 / d**3).sum(1) + observation_variance
    )
    ks = cgf(s)

    def integrand(w):
        t = w / sd
        return np.exp(cgf(s + 1j * t) - ks - 1j * t * target).real

    integral, _, info = quad_vec(
        integrand,
        0,
        np.inf,
        epsabs=tolerance,
        epsrel=tolerance,
        norm="max",
        full_output=True,
    )
    if not info.success or (integral <= 0).any() or not np.isfinite(integral).all():
        raise ArithmeticError("Density inversion failed")
    return ks - s * target + np.log(integral / (np.pi * sd))


@torch.no_grad()
def predictive_log_scores(model, x, context, target) -> np.ndarray:
    """Bilinear predictive log density of each realized return at a new date."""
    index = torch.full((x.shape[0],), -1, dtype=torch.long, device=x.device)
    latent = model.row_moments(x, context.expand(x.shape[0], -1), index)
    beta = latent.beta_covariance + torch.diag(model.likelihood.loading_variances)[None]
    cross = latent.beta_factor_covariance
    covariance = torch.cat(
        (
            torch.cat((beta, cross), dim=2),
            torch.cat((cross.mT, latent.factor_covariance), dim=2),
        ),
        dim=1,
    )
    mean = torch.cat((latent.beta_mean, latent.factor_mean), dim=1)
    scale = float(model.target_std)
    return quadratic_form_log_density(
        target.cpu().numpy() / scale,
        mean.cpu().numpy(),
        covariance.cpu().numpy(),
        float(model.likelihood.observation_variance),
    ) - np.log(scale)
