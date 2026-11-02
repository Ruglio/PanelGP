"""Collect completed fits into the compact numerical record behind the paper.

    python scripts/collect.py --runs results/runs --output results/paper/data

Reads ``<runs>/case_NNN`` (scripts/run_case.py) and ``<runs>/synthetic/seedS``
(scripts/run_synthetic.py, location set by ``--synthetic``); ``--part``
collects only one of the two. Writes:

* summary.csv           one row per fitted model and period (validation/test)
* monthly_metrics.csv   monthly scores of the H = 5 models (evaluation period)
* monthly_portfolios.csv  monthly portfolio returns of the same models
* lengthscales.csv      inverse ARD lengthscales, fixed RBF/RBF, H = 5
* trading_costs.csv     turnover and net Sharpe ratios, annual re-estimation
* risk_*.csv, density_gain.csv  variance-term and density diagnostics
* r2_split.csv          pooled R^2 of the H = 5 RBF/RBF forecasts, common and
                        cross-sectional parts (needs the fitted models and data)
* synthetic*.csv        simulation recovery and the representative replication
* market_benchmark.csv  Sharpe ratio of the S&P 500 over the same months and halves
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from panelgp.experiments.campaign import Case, cases
from panelgp.experiments.config import load_config
from panelgp.experiments.evaluation import (
    STRATEGIES,
    portfolio_turnover,
    sharpe,
    volatility,
)

KEYS = ["model", "protocol", "loading", "factor", "hidden_dim", "seed"]
COST = 10 / 10_000  # proportional trading cost of 10 basis points
PERIODS = ((1995, 2009), (2010, 2024))  # input-month halves


def labels(case: Case) -> dict:
    model = (
        case.method
        if case.method == "panelgp"
        else ("ipca_intercept" if case.intercept else "ipca")
    )
    panelgp = case.method == "panelgp"
    return {
        "model": model,
        "protocol": case.protocol,
        "loading": case.loading_kernel if panelgp else "affine",
        "factor": case.factor_kernel if panelgp else "",
        "hidden_dim": case.hidden_dim,
        "seed": case.seed if panelgp else 0,
    }


def read_cases(runs: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics, portfolios = [], []
    for case in cases():
        directory = runs / case.name
        if not (directory / "complete.json").exists():
            raise FileNotFoundError(f"Incomplete case {case.index}: {directory}")
        tags = labels(case)
        metrics.append(pd.read_csv(directory / "metrics.csv").assign(**tags))
        portfolios.append(pd.read_csv(directory / "portfolios.csv").assign(**tags))
    return pd.concat(metrics, ignore_index=True), pd.concat(
        portfolios, ignore_index=True
    )


def summarize(metrics: pd.DataFrame, portfolios: pd.DataFrame) -> pd.DataFrame:
    """Pooled R^2, mean rank correlation and log score, Sharpe ratios, GMV volatility."""
    keys = [*KEYS, "period"]
    returns = dict(tuple(portfolios.groupby([*keys, "strategy"])))
    rows = []
    for key, group in metrics.groupby(keys):
        strategy = {name: returns[(*key, name)]["return"] for name in STRATEGIES}
        rows.append(
            {
                **dict(zip(keys, key)),
                "months": len(group),
                "r2_percent": 100
                * (1 - group.sum_squared_error.sum() / group.sum_target_squared.sum()),
                "rank_percent": 100 * np.mean(group.spearman.to_numpy()),
                "log_score": np.mean(group.log_score.to_numpy()),
                "tangency_sharpe": sharpe(strategy["tangency"]),
                "long_short_sharpe": sharpe(strategy["long_short"]),
                "gmv_percent": 100 * volatility(strategy["gmv"]),
            }
        )
    summary = pd.DataFrame(rows)
    if not summary.query("period == 'test'").months.eq(360).all():
        raise ValueError("Every evaluation path must cover the 360 months 1995-2024.")
    return summary


def lengthscales(runs: Path, context: list[str]) -> pd.DataFrame:
    rows = []
    for case in cases():
        if (
            case.method,
            case.protocol,
            case.hidden_dim,
            case.loading_kernel,
            case.factor_kernel,
        ) != ("panelgp", "fixed", 5, "rbf", "rbf"):
            continue
        fitted = json.loads(
            (runs / case.name / "diagnostics.json").read_text(encoding="utf-8")
        )
        for kind, names, values in (
            ("loading", fitted["feature_names"], fitted["loading_lengthscales"]),
            ("factor", context, fitted["factor_lengthscales"]),
        ):
            for component, row in enumerate(values, 1):
                rows += [
                    {
                        "kind": kind,
                        "feature": name,
                        "seed": case.seed,
                        "component": component,
                        "inverse_lengthscale": 1 / value,
                    }
                    for name, value in zip(names, row)
                ]
    return pd.DataFrame(rows)


def lead_risk_free(macro_file: str) -> pd.Series:
    """Risk-free return of month t+1 indexed by input month t (Goyal ``Rfree``)."""
    macro = pd.read_csv(macro_file, usecols=["yyyymm", "Rfree"])
    month = pd.to_datetime(
        macro.yyyymm.astype(str), format="%Y%m"
    ) - pd.offsets.MonthBegin(1)
    index = pd.MultiIndex.from_arrays(
        [month.dt.year, month.dt.month], names=["year", "month"]
    )
    return pd.Series(macro.Rfree.to_numpy(), index=index)


def trading_costs(
    runs: Path, portfolios: pd.DataFrame, risk_free: pd.Series
) -> pd.DataFrame:
    """Drift-adjusted turnover (33) and net Sharpe ratios (34) of the rolling paths."""
    weights = []
    for case in cases():
        if case.protocol == "rolling":
            frame = pd.read_parquet(runs / case.name / "weights.parquet")
            weights.append(frame.assign(**labels(case)))
    weights = pd.concat(weights, ignore_index=True)
    asset_returns = weights.drop_duplicates(["year", "month", "permno"])
    portfolios = portfolios.query("protocol == 'rolling' and period == 'test'")
    rows = []
    for (model, seed, strategy), group in weights.groupby(
        ["model", "seed", "strategy"]
    ):
        if strategy == "gmv":
            continue
        turnover = portfolio_turnover(group, asset_returns, risk_free)
        gross = (
            portfolios.query(
                "model == @model and seed == @seed and strategy == @strategy"
            )
            .set_index(["year", "month"])["return"]
            .sort_index()
        )
        held = (
            (group.weight * group.asset_return).groupby([group.year, group.month]).sum()
        )
        if len(gross) != 360 or not np.allclose(held, gross, rtol=0, atol=1e-12):
            raise ValueError(
                f"Weights do not reproduce returns for {model}/{seed}/{strategy}."
            )
        rows.append(
            {
                "model": model,
                "seed": seed,
                "strategy": strategy,
                "mean_monthly_turnover": turnover.iloc[1:].mean(),
                "gross_sharpe": sharpe(gross),
                "net_sharpe_10bp": sharpe(gross - COST * turnover),
                "break_even_bp": 10_000 * gross.mean() / turnover.mean(),
            }
        )
    return (
        pd.DataFrame(rows)
        .groupby(["model", "strategy"], as_index=False)
        .mean(numeric_only=True)
        .drop(columns="seed")
    )


def rolling_diagnostics(runs: Path) -> dict[str, pd.DataFrame]:
    """Density gains (Figure S3) and variance terms against forecast errors (Figure 4)."""
    predictions = []
    for case in cases():
        if case.method == "panelgp" and case.protocol == "rolling":
            frame = pd.read_parquet(runs / case.name / "predictions.parquet")
            predictions.append(frame.assign(seed=case.seed))
    frame = pd.concat(predictions, ignore_index=True)
    error = frame.target - frame["mean"]
    variance = frame.variance.clip(lower=1.0e-12)
    z = error / np.sqrt(variance)
    gaussian = -0.5 * (np.log(2.0 * np.pi * variance) + np.square(z))
    frame = frame.assign(
        error2=np.square(error),
        gain=frame.bilinear_log_score - gaussian,
        bucket=np.searchsorted([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], z, side="right")
        + 1,
    )
    density = frame.groupby(["seed", "year", "month", "bucket"], as_index=False).agg(
        n=("gain", "size"), gain_sum=("gain", "sum")
    )
    monthly, within, ranks = [], [], []
    for (seed, year, month), group in frame.groupby(
        ["seed", "year", "month"], sort=True
    ):
        date = {"seed": seed, "year": year, "month": month}
        monthly.append(
            {
                **date,
                "error2": group.error2.mean(),
                "total": group.variance.mean(),
                "factor_innovation": group.factor_innovation.mean(),
                "loading_innovation": group.loading_innovation.mean(),
                "innovation_interaction": group.innovation_interaction.mean(),
            }
        )
        ranks.append(
            {
                **date,
                "comparison": "factor_across_stocks",
                "spearman": group[["factor_innovation", "error2"]]
                .corr("spearman")
                .iloc[0, 1],
            }
        )
        order = np.argsort(group.factor_innovation.to_numpy())
        for bucket, rows in enumerate(np.array_split(order, 5), 1):
            selected = group.iloc[rows]
            within.append(
                {
                    **date,
                    "bucket": bucket,
                    "n": len(selected),
                    "error2_sum": selected.error2.sum(),
                    "total_sum": selected.variance.sum(),
                    "factor_sum": selected.factor_innovation.sum(),
                }
            )
    monthly = pd.DataFrame(monthly)
    temporal = []
    for seed, group in monthly.groupby("seed"):
        bucket = (
            pd.qcut(group.loading_innovation.rank(method="first"), 5, labels=False) + 1
        )
        for value, selected in group.groupby(bucket):
            temporal.append(
                {
                    "seed": seed,
                    "bucket": int(value),
                    "n": len(selected),
                    "error2": selected.error2.mean(),
                    "total": selected.total.mean(),
                    "loading_innovation": selected.loading_innovation.mean(),
                }
            )
        ranks.append(
            {
                "seed": seed,
                "comparison": "loading_across_months",
                "spearman": group[["loading_innovation", "error2"]]
                .corr("spearman")
                .iloc[0, 1],
            }
        )
    return {
        "density_gain.csv": density,
        "risk_monthly.csv": monthly,
        "risk_stock_quintiles.csv": pd.DataFrame(within),
        "risk_month_quintiles.csv": pd.DataFrame(temporal),
        "risk_rank_correlations.csv": pd.DataFrame(ranks),
    }


def synthetic(runs: Path, output: Path) -> None:
    """Five controlled replications; the median factor recovery is shown in figures."""
    frame = pd.concat(
        [pd.read_csv(runs / f"seed{seed}/summary.csv") for seed in range(1, 6)],
        ignore_index=True,
    )
    frame["average_factor_r2"] = frame[["factor_probe_1_r2", "factor_probe_2_r2"]].mean(
        axis=1
    )
    frame.to_csv(output / "synthetic.csv", index=False)
    representative = int(frame.sort_values("average_factor_r2").iloc[2].seed)
    for name in ("factor_dates", "factor_probe", "loading_probe"):
        shutil.copy2(
            runs / f"seed{representative}/{name}.csv", output / f"synthetic_{name}.csv"
        )


def r2_parts(frame: pd.DataFrame) -> dict:
    """Pooled R^2 split into the part due to the monthly average forecast
    (common) and the part due to the deviations from it (cross-section)."""
    error = frame.target - frame["mean"]
    keys = [frame.year, frame.month]
    count = error.groupby(keys).size()
    common_error = (count * error.groupby(keys).mean() ** 2).sum()
    common_target = (count * frame.target.groupby(keys).mean() ** 2).sum()
    total, squared = (frame.target**2).sum(), (error**2).sum()
    return {
        "r2_percent": 100 * (1 - squared / total),
        "common_percent": 100 * (common_target - common_error) / total,
        "cross_percent": 100
        * ((total - common_target) - (squared - common_error))
        / total,
    }


def r2_split(runs: Path, metrics: pd.DataFrame) -> pd.DataFrame:
    """R^2 parts of the H = 5 RBF/RBF forecasts (Section 6.6). Annual
    re-estimation saved its forecasts; fixed-training forecasts are recomputed
    from the fitted models and must reproduce the saved pooled R^2."""
    import torch

    from panelgp.experiments.campaign import split_months
    from panelgp.experiments.data import load_panel, month_number, standardize_context
    from panelgp.model import ModelSettings, PanelGP, predict_rows

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    forecasts, test = {}, None
    for case in cases():
        if (case.method, case.hidden_dim, case.loading_kernel, case.factor_kernel) != (
            "panelgp",
            5,
            "rbf",
            "rbf",
        ):
            continue
        folder = runs / case.name
        if case.protocol == "rolling":
            columns = ["year", "month", "target", "mean"]
            frame = pd.read_parquet(folder / "predictions.parquet", columns=columns)
        else:
            config = load_config(folder / "config.yaml")
            if test is None:
                panel, schema = load_panel(config)
                codes, months = month_number(panel), split_months(panel, config["run"])
                train, test = (
                    panel.loc[codes.isin(months[k])].reset_index(drop=True)
                    for k in ("train", "test")
                )
                _, test = standardize_context(
                    train,
                    [train, test],
                    list(schema.context_columns),
                    list(config["data"].get("log_context", [])),
                )
            state = torch.load(folder / "model.pt", map_location=device)
            model = PanelGP(
                input_dim=len(schema.feature_columns),
                time_features=state["time_features"],
                settings=ModelSettings.from_config(config),
                feature_names=schema.feature_columns,
            ).to(device=device, dtype=torch.float64)
            model.load_state_dict(state)
            frame = test[["year", "month"]].assign(
                target=test[schema.target_column],
                mean=predict_rows(model, test, schema).mean.numpy(),
            )
        forecasts.setdefault((case.protocol, case.seed), []).append(frame)
    rows = []
    for (protocol, seed), frames in forecasts.items():
        parts = r2_parts(pd.concat(frames, ignore_index=True))
        saved = metrics.query(
            "model == 'panelgp' and protocol == @protocol and seed == @seed "
            "and hidden_dim == 5 and loading == 'rbf' and factor == 'rbf' "
            "and period == 'test'"
        )
        pooled = 100 * (
            1 - saved.sum_squared_error.sum() / saved.sum_target_squared.sum()
        )
        if abs(parts["r2_percent"] - pooled) > 1e-6:
            raise ValueError(f"Forecasts do not reproduce R^2 of {protocol} {seed}.")
        rows.append({"protocol": protocol, "seed": seed, **parts})
    return pd.DataFrame(rows)


def market_benchmark(goyal_file: str) -> pd.DataFrame:
    """Sharpe ratio of the S&P 500 total return minus Rfree over the realization
    months of the forecasts, for the evaluation period and its two halves."""
    macro = pd.read_csv(goyal_file, usecols=["yyyymm", "ret", "Rfree"])
    rows = []
    for first, last in ((1995, 2024), *PERIODS):
        market = macro.loc[macro.yyyymm.between(first * 100 + 2, last * 100 + 101)]
        months = 12 * (last - first + 1)
        if len(market) != months or market.isna().any(axis=None):
            raise ValueError(f"The market reference misses months of {first}-{last}.")
        rows.append(
            {
                "label": "S&P 500",
                "period": f"{first}-{last}",
                "months": months,
                "sharpe": sharpe(market.ret - market.Rfree),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--runs", type=Path, default=Path("results/runs"))
    parser.add_argument("--synthetic", type=Path, help="default: <runs>/synthetic")
    parser.add_argument("--output", type=Path, default=Path("results/paper/data"))
    parser.add_argument("--config", default="configs/empirical.yaml")
    parser.add_argument(
        "--part",
        choices=("all", "empirical", "synthetic"),
        default="all",
        help="collect only the empirical cases or only the simulation",
    )
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    if args.part in ("all", "synthetic"):
        synthetic(args.synthetic or args.runs / "synthetic", output)
    if args.part == "synthetic":
        return
    config = load_config(args.config)
    metrics, portfolios = read_cases(args.runs)
    summarize(metrics, portfolios).to_csv(output / "summary.csv", index=False)
    test = metrics.period.eq("test") & metrics.hidden_dim.eq(5)
    metrics.loc[test].drop(columns="period").to_csv(
        output / "monthly_metrics.csv", index=False
    )
    test = portfolios.period.eq("test") & portfolios.hidden_dim.eq(5)
    portfolios.loc[test].drop(columns="period").to_csv(
        output / "monthly_portfolios.csv", index=False
    )
    lengthscales(args.runs, config["data"]["context_columns"]).to_csv(
        output / "lengthscales.csv", index=False
    )
    risk_free = lead_risk_free(config["data"]["macro_file"])
    trading_costs(args.runs, portfolios, risk_free).to_csv(
        output / "trading_costs.csv", index=False
    )
    for name, frame in rolling_diagnostics(args.runs).items():
        frame.to_csv(output / name, index=False)
    market_benchmark(config["data"]["macro_file"]).to_csv(
        output / "market_benchmark.csv", index=False
    )
    r2_split(args.runs, metrics).to_csv(output / "r2_split.csv", index=False)
    print(f"Collected the numerical record in {output}")


if __name__ == "__main__":
    main()
