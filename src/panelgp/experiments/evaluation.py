"""Forecast scores and portfolios of Sections 6.3 and 6.6."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.linalg import solve, solve_triangular
from scipy.stats import spearmanr

STRATEGIES = ("gmv", "tangency", "long_short")


def score_cross_section(
    target: np.ndarray, mean: np.ndarray, covariance: np.ndarray
) -> dict[str, float]:
    """Monthly ingredients of the pooled R^2 (30), rank correlation and log score (31)."""
    y, mu = np.asarray(target, float), np.asarray(mean, float)
    error = y - mu
    chol = np.linalg.cholesky(0.5 * (covariance + covariance.T))
    whitened = solve_triangular(chol, error, lower=True, check_finite=False)
    logdet = float(2.0 * np.log(np.diag(chol)).sum())
    nll = 0.5 * (y.size * math.log(2.0 * math.pi) + logdet + float(whitened @ whitened))
    return {
        "n": y.size,
        "sum_squared_error": float(np.sum(error**2)),
        "sum_target_squared": float(y @ y),
        "spearman": float(spearmanr(y, mu).statistic),
        "log_score": -nll / y.size,
    }


def long_short_weights(mean: np.ndarray, fraction: float = 0.1) -> np.ndarray:
    """Equal-weight top minus bottom ``fraction`` of predicted returns, half gross each."""
    count = max(1, min(mean.size // 2, int(np.ceil(mean.size * fraction))))
    order = np.argsort(mean, kind="stable")
    weights = np.zeros(mean.size)
    weights[order[:count]] = -0.5 / count
    weights[order[-count:]] = 0.5 / count
    return weights


def portfolio_weights(
    mean: np.ndarray, covariance: np.ndarray
) -> dict[str, np.ndarray]:
    """GMV and unit-gross tangency weights of eq. (32) and the decile long-short."""
    matrix = 0.5 * (covariance + covariance.T)
    directions = solve(
        matrix, np.column_stack((np.ones(mean.size), mean)), assume_a="pos"
    )
    directions = directions / np.linalg.norm(directions, ord=1, axis=0)
    return {
        "gmv": directions[:, 0] / directions[:, 0].sum(),
        "tangency": directions[:, 1],
        "long_short": long_short_weights(mean),
    }


def volatility(returns) -> float:
    """Annualized volatility of monthly returns."""
    return float(np.asarray(returns, float).std(ddof=1) * np.sqrt(12))


def sharpe(returns) -> float:
    """Annualized Sharpe ratio of monthly excess returns."""
    return float(np.asarray(returns, float).mean() * 12) / volatility(returns)


def portfolio_turnover(
    weights: pd.DataFrame, asset_returns: pd.DataFrame, risk_free: pd.Series
) -> pd.Series:
    """Monthly turnover TO_t of eq. (33) after drifting the previous weights.

    ``weights`` and ``asset_returns`` have columns year, month, permno and
    ``weight`` or ``asset_return`` (excess returns realized over the next
    month). ``risk_free`` is indexed by input (year, month) and holds the
    risk-free return of the following month. The cash account holds 1 - sum(w).
    The first month has zero turnover.
    """
    keys = ["year", "month"]
    matrix = (
        weights.pivot(index=keys, columns="permno", values="weight")
        .fillna(0.0)
        .sort_index()
    )
    previous = matrix.shift(1)
    returns = asset_returns.pivot(index=keys, columns="permno", values="asset_return")
    previous_returns = returns.reindex(
        index=matrix.index, columns=matrix.columns
    ).shift(1)
    if (previous.ne(0) & previous.notna() & previous_returns.isna()).any(axis=None):
        raise ValueError("Missing realized returns for held assets.")
    previous_returns = previous_returns.fillna(0.0)
    rate = risk_free.reindex(matrix.index)
    if rate.iloc[:-1].isna().any():
        raise ValueError("The risk-free rate does not cover every portfolio month.")
    previous_rate = rate.shift(1)
    wealth = 1.0 + previous_rate + (previous * previous_returns).sum(axis=1)
    previous = previous * (1.0 + previous_rate.to_numpy()[:, None] + previous_returns)
    previous = previous.div(wealth, axis=0)
    turnover = (matrix - previous).abs().sum(axis=1)
    turnover.iloc[0] = 0.0
    return turnover
