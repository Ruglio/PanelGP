"""IPCA benchmarks of Section 6.2 (Kelly, Pruitt and Su, 2019)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..model.data import PanelSchema

EPS = 1.0e-12


def _instruments(frame: pd.DataFrame, schema: PanelSchema) -> np.ndarray:
    """Characteristics with a constant column, X_t^+ = [X_t, 1]."""
    x = frame[list(schema.feature_columns)].to_numpy(float)
    return np.column_stack((x, np.ones(len(x))))


@dataclass(frozen=True)
class FittedIPCA:
    gamma: np.ndarray  # (P+1) x (H + intercept)
    factor_mean: np.ndarray  # sample mean of the training factors (1 for the intercept)
    factor_covariance: np.ndarray  # sample covariance; zero for the intercept factor
    residual_variance: float  # pooled in-sample residual variance

    def predict_mean(self, frame: pd.DataFrame, schema: PanelSchema) -> np.ndarray:
        return (_instruments(frame, schema) @ self.gamma) @ self.factor_mean

    def covariance(self, frame: pd.DataFrame, schema: PanelSchema) -> np.ndarray:
        """Predictive covariance of eq. (29)."""
        loadings = _instruments(frame, schema) @ self.gamma
        covariance = loadings @ self.factor_covariance @ loadings.T
        covariance += max(self.residual_variance, EPS) * np.eye(len(frame))
        return 0.5 * (covariance + covariance.T)


def fit_ipca(
    train: pd.DataFrame, schema: PanelSchema, *, hidden_dim: int, intercept: bool
) -> FittedIPCA:
    """Estimate IPCA by alternating least squares with the ``ipca`` package."""
    from ipca import InstrumentedPCA

    x = _instruments(train, schema)
    target = train[schema.target_column].to_numpy(float)
    dates = pd.MultiIndex.from_frame(train[["year", "month"]].astype(int))
    indices = np.column_stack(
        (
            pd.factorize(train[schema.asset_column], sort=True)[0],
            pd.factorize(dates, sort=True)[0],
        )
    ).astype(np.int64)
    estimator = InstrumentedPCA(
        n_factors=hidden_dim,
        intercept=intercept,
        max_iter=10000,
        iter_tol=1.0e-5,
        alpha=0.0,
        l1_ratio=1.0,
        n_jobs=1,
        backend="loky",
    )
    estimator.fit(X=x, y=target, indices=indices, data_type="panel")
    gamma, factors = estimator.get_factors()
    gamma, factors = np.asarray(gamma, float), np.atleast_2d(np.asarray(factors, float))

    latent = np.atleast_2d(np.cov(factors[:hidden_dim], bias=False))
    latent = 0.5 * (latent + latent.T)
    smallest = float(np.linalg.eigvalsh(latent).min())
    if smallest < EPS:
        latent += (EPS - smallest) * np.eye(hidden_dim)
    covariance = np.zeros((factors.shape[0], factors.shape[0]))
    covariance[:hidden_dim, :hidden_dim] = latent

    fitted = estimator.predict(
        X=x, indices=indices, mean_factor=False, data_type="panel"
    )
    residual = float(np.mean((target - np.asarray(fitted, float).reshape(-1)) ** 2))
    return FittedIPCA(gamma, factors.mean(axis=1), covariance, max(residual, EPS))
