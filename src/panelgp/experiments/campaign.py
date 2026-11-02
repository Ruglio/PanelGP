"""The empirical campaign of Section 6: its 386 cases and the runner for one case.

Cases are numbered as in the paper's campaign:

* 0-159    PanelGP, fixed training: 4 kernel pairs x H = 1..8 x seeds 1..5,
           index = 40 * pair + 5 * (H - 1) + (seed - 1);
* 160-309  PanelGP, annual re-estimation, RBF/RBF, H = 5: seeds 1..5 x 1995..2024;
* 310-385  IPCA without (310-347) and with (348-385) an intercept: fixed training
           for H = 1..8, then annual re-estimation with H = 5 for 1995..2024.

Each case writes ``metrics.csv`` and ``portfolios.csv`` (one row per month);
PanelGP cases add ``diagnostics.json``, ``history.csv`` and ``model.pt``.
Annual re-estimation cases also write the security-level ``weights.parquet``
used for trading costs, and PanelGP ones ``predictions.parquet`` with the
variance terms and bilinear log scores of Sections 6.7-6.8.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from ..model.data import PanelSchema, to_tensors
from ..model.training import fit_model
from .config import merge, write_config
from .data import load_panel, month_number, standardize_context
from .density import predictive_log_scores
from .evaluation import STRATEGIES, portfolio_weights, score_cross_section
from .ipca import fit_ipca

KERNEL_PAIRS = (
    ("affine", "constant"),
    ("affine", "rbf"),
    ("rbf", "constant"),
    ("rbf", "rbf"),
)
HIDDEN_DIMS = range(1, 9)
SEEDS = range(1, 6)
YEARS = range(1995, 2025)


@dataclass(frozen=True)
class Case:
    index: int
    method: str  # "panelgp" or "ipca"
    protocol: str  # "fixed" or "rolling"
    hidden_dim: int
    loading_kernel: str = "rbf"
    factor_kernel: str = "rbf"
    seed: int = 1
    year: int | None = None  # evaluation year of an annual re-estimation
    intercept: bool = False  # IPCA with a return intercept

    @property
    def name(self) -> str:
        return f"case_{self.index:03d}"

    def apply(self, config: Mapping[str, Any]) -> dict[str, Any]:
        """The preset with this case's specification and evaluation period."""
        update: dict[str, Any] = {
            "model": {
                "hidden_dim": self.hidden_dim,
                "loading_kernel": self.loading_kernel,
                "factor_kernel": self.factor_kernel,
            },
            "runtime": {"seed": self.seed},
            "run": {"protocol": self.protocol},
        }
        if self.year is not None:
            update["run"].update(
                test_start=f"{self.year}-01", test_end=f"{self.year}-12"
            )
        return merge(config, update)


def cases() -> list[Case]:
    result: list[Case] = []

    def add(**fields) -> None:
        result.append(Case(index=len(result), **fields))

    for loading, factor in KERNEL_PAIRS:
        for hidden in HIDDEN_DIMS:
            for seed in SEEDS:
                add(
                    method="panelgp",
                    protocol="fixed",
                    hidden_dim=hidden,
                    loading_kernel=loading,
                    factor_kernel=factor,
                    seed=seed,
                )
    for seed in SEEDS:
        for year in YEARS:
            add(
                method="panelgp", protocol="rolling", hidden_dim=5, seed=seed, year=year
            )
    for intercept in (False, True):
        for hidden in HIDDEN_DIMS:
            add(method="ipca", protocol="fixed", hidden_dim=hidden, intercept=intercept)
        for year in YEARS:
            add(
                method="ipca",
                protocol="rolling",
                hidden_dim=5,
                year=year,
                intercept=intercept,
            )
    return result


def _month(value: str) -> int:
    year, month = str(value).split("-")
    return int(year) * 12 + int(month) - 1


def split_months(frame: pd.DataFrame, run: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Training, validation and evaluation input months (as year*12 + month - 1).

    Fixed training uses ``train_start`` up to ``validation_start`` and holds
    out the validation period. Annual re-estimation trains on the latest
    ``training_months`` before the evaluation year, without validation.
    """
    months = np.sort(month_number(frame).unique())
    test = months[
        (months >= _month(run["test_start"])) & (months <= _month(run["test_end"]))
    ]
    if run["protocol"] == "fixed":
        validation_start = _month(run["validation_start"])
        train = months[
            (months >= _month(run["train_start"])) & (months < validation_start)
        ]
        validation = months[(months >= validation_start) & (months < test[0])]
    else:
        train = months[months < test[0]][-int(run["training_months"]) :]
        validation = months[:0]
    if len(train) == 0 or len(test) == 0:
        raise ValueError("Empty training or evaluation period.")
    return {"train": train, "validation": validation, "test": test}


def panelgp_forecaster(model, schema: PanelSchema):
    """Predictive mean and covariance of a cross section at a new date."""

    def forecast(rows: pd.DataFrame):
        data = to_tensors(rows, schema, device=model.time_features.device)
        moments = model.cross_section_moments(data.x, data.context[0], -1)
        mean, covariance = moments.mean.cpu().numpy(), moments.covariance.cpu().numpy()
        return mean, covariance, (model, data, moments)

    return forecast


def ipca_forecaster(model, schema: PanelSchema):
    def forecast(rows: pd.DataFrame):
        return model.predict_mean(rows, schema), model.covariance(rows, schema), None

    return forecast


def _predictions(target, mean, covariance, extra) -> pd.DataFrame:
    """Row-level forecasts, variance terms (27) and bilinear log scores (35)."""
    model, data, moments = extra
    return pd.DataFrame(
        {
            "target": target,
            "mean": mean,
            "variance": np.diag(covariance),
            "factor_innovation": moments.factor_innovation.cpu().numpy(),
            "loading_innovation": moments.loading_innovation.cpu().numpy(),
            "innovation_interaction": moments.innovation_interaction.cpu().numpy(),
            "bilinear_log_score": predictive_log_scores(
                model, data.x, data.context, data.target
            ),
        }
    )


def evaluate(
    forecast, frames: dict[str, pd.DataFrame], schema: PanelSchema, *, detail: bool
):
    """Monthly scores and portfolio returns; with ``detail``, also the
    security-level weights and predictions of the evaluation period."""
    metrics, portfolios, weights, predictions = [], [], [], []
    for period, frame in frames.items():
        for (year, month), rows in frame.groupby(["year", "month"], sort=True):
            rows = rows.reset_index(drop=True)
            mean, covariance, extra = forecast(rows)
            date = {"period": period, "year": int(year), "month": int(month)}
            target = rows[schema.target_column].to_numpy(float)
            metrics.append({**date, **score_cross_section(target, mean, covariance)})
            strategies = portfolio_weights(mean, covariance)
            for strategy in STRATEGIES:
                realized = float(strategies[strategy] @ target)
                portfolios.append({**date, "strategy": strategy, "return": realized})
            if not detail or period != "test":
                continue
            ids = pd.DataFrame(
                {
                    "year": int(year),
                    "month": int(month),
                    "permno": rows[schema.asset_column],
                }
            )
            for strategy in STRATEGIES:
                weights.append(
                    ids.assign(
                        strategy=strategy,
                        weight=strategies[strategy],
                        asset_return=target,
                    )
                )
            if extra is not None:
                predictions.append(
                    pd.concat(
                        [ids, _predictions(target, mean, covariance, extra)],
                        axis=1,
                    )
                )
    return (
        pd.DataFrame(metrics),
        pd.DataFrame(portfolios),
        pd.concat(weights, ignore_index=True) if weights else None,
        pd.concat(predictions, ignore_index=True) if predictions else None,
    )


def run_case(config: Mapping[str, Any], case: Case, output: str | Path) -> Path:
    """Estimate and evaluate one case of the campaign; returns its output directory."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config = case.apply(config)
    panel, schema = load_panel(config)
    if config["model"].get("loading_lengthscale") == "auto":
        # sqrt(P/12): ranks are approximately uniform on [-0.5, 0.5] (Appendix C.4).
        lengthscale = math.sqrt(len(schema.feature_columns) / 12)
        config = merge(config, {"model": {"loading_lengthscale": lengthscale}})
    write_config(config, output / "config.yaml")
    split = split_months(panel, config["run"])
    codes = month_number(panel)
    frames = {
        name: panel.loc[codes.isin(months)].reset_index(drop=True)
        for name, months in split.items()
    }
    data = config["data"]
    train, validation, test = standardize_context(
        frames["train"],
        [frames["train"], frames["validation"], frames["test"]],
        list(schema.context_columns),
        list(data.get("log_context", [])),
    )
    print(
        f"{case.name}: train={len(train):,} validation={len(validation):,} test={len(test):,}",
        flush=True,
    )
    if case.method == "ipca":
        ipca = fit_ipca(
            train, schema, hidden_dim=case.hidden_dim, intercept=case.intercept
        )
        forecast = ipca_forecaster(ipca, schema)
    else:
        fit = fit_model(train, schema, config)
        model = fit.model.eval()
        fit.history.to_csv(output / "history.csv", index=False)
        torch.save(model.state_dict(), output / "model.pt")
        (output / "diagnostics.json").write_text(
            json.dumps(model.diagnostics(), indent=2)
        )
        forecast = panelgp_forecaster(model, schema)
    # The validation period serves PanelGP's choice of H (Figure S2).
    periods = {"test": test}
    if case.method == "panelgp" and len(validation):
        periods = {"validation": validation, "test": test}
    metrics, portfolios, weights, predictions = evaluate(
        forecast, periods, schema, detail=case.protocol == "rolling"
    )
    metrics.to_csv(output / "metrics.csv", index=False)
    portfolios.to_csv(output / "portfolios.csv", index=False)
    if weights is not None:
        weights.to_parquet(output / "weights.parquet", index=False)
    if predictions is not None:
        predictions.to_parquet(output / "predictions.parquet", index=False)
    (output / "complete.json").write_text(json.dumps({"case": case.index}) + "\n")
    return output
