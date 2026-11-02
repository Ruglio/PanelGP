"""Data-generating process of the controlled simulation (Section 5.1)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .data import PanelSchema

SCHEMA = PanelSchema(("x1", "x2"), ("c1",), "excess_return", "asset")


def true_factors(context: np.ndarray) -> np.ndarray:
    """Smooth factor maps phi_1, phi_2 of eq. (28) on c in [0, 1]."""
    c = np.asarray(context, dtype=float).reshape(-1)
    return np.column_stack(
        (
            0.5
            + np.sin(2 * np.pi * c)
            + 0.5 * np.cos(4 * np.pi * c)
            + np.sin(6 * np.pi * c) / 3,
            -0.5
            + np.cos(2 * np.pi * c)
            - 0.5 * np.sin(4 * np.pi * c)
            + np.cos(6 * np.pi * c) / 3,
        )
    )


def true_loadings(x: np.ndarray) -> np.ndarray:
    """Smooth loading maps beta_1, beta_2 of eq. (28) on x in [0, 1]^2."""
    x1, x2 = np.asarray(x, dtype=float).T
    return np.column_stack(
        (
            np.sin(2 * np.pi * x1) * np.cos(np.pi * x2)
            + 0.5 * np.cos(2 * np.pi * (x1 + x2))
            + np.sin(4 * np.pi * x1) * np.sin(2 * np.pi * x2) / 3,
            np.cos(2 * np.pi * x2) * np.sin(np.pi * x1)
            - 0.5 * np.sin(2 * np.pi * (x1 - x2))
            + np.cos(2 * np.pi * x1) * np.cos(4 * np.pi * x2) / 3,
        )
    )


def asset_grid(count: int) -> np.ndarray:
    """First ``count`` points of the smallest square grid on [0, 1]^2, row by row."""
    columns = math.ceil(math.sqrt(count))
    rows = math.ceil(count / columns)
    grid = [
        (column / max(columns - 1, 1), row / max(rows - 1, 1))
        for row in range(rows)
        for column in range(columns)
    ]
    return np.array(grid[:count])


def simulate_panel(
    seed: int,
    *,
    assets: int = 80,
    months: int = 50,
    loading_innovation: float = 0.05,
    factor_innovation: float = 0.05,
    noise: float = 0.05,
) -> pd.DataFrame:
    """Balanced panel from eq. (3) with isotropic innovations, plus the truth.

    Columns ``true_mean`` and ``true_variance`` are the conditional moments of
    eq. (6) given the smooth maps; ``true_realized_phi*`` are the realized factors.
    """
    rng = np.random.default_rng(seed)
    dates = pd.period_range("2000-01", periods=months, freq="M")
    context = np.linspace(0.0, 1.0, months)
    phi = true_factors(context)
    realized_phi = phi + rng.normal(0.0, math.sqrt(factor_innovation), phi.shape)
    x = asset_grid(assets)
    beta = true_loadings(x)
    hidden = beta.shape[1]
    returns, mean, variance = (np.empty((months, assets)) for _ in range(3))
    for t in range(months):
        xi = rng.normal(0.0, math.sqrt(loading_innovation), beta.shape)
        returns[t] = (beta + xi) @ realized_phi[t]
        returns[t] += rng.normal(0.0, math.sqrt(noise), assets)
        mean[t] = beta @ phi[t]
        variance[t] = (
            factor_innovation * np.sum(beta**2, axis=1)
            + loading_innovation * float(phi[t] @ phi[t])
            + hidden * loading_innovation * factor_innovation
            + noise
        )
    frame = pd.DataFrame(
        {
            "asset": np.tile(np.arange(1, assets + 1), months),
            "year": np.repeat(dates.year, assets),
            "month": np.repeat(dates.month, assets),
            "excess_return": returns.reshape(-1),
            "x1": np.tile(x[:, 0], months),
            "x2": np.tile(x[:, 1], months),
            "c1": np.repeat(context, assets),
            "true_mean": mean.reshape(-1),
            "true_variance": variance.reshape(-1),
        }
    )
    for h in range(hidden):
        frame[f"true_beta{h + 1}"] = np.tile(beta[:, h], months)
        frame[f"true_phi{h + 1}"] = np.repeat(phi[:, h], assets)
        frame[f"true_realized_phi{h + 1}"] = np.repeat(realized_phi[:, h], assets)
    return frame
