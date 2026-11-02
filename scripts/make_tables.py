"""Tables 1-4, S1-S3 and the numbers quoted in the text, from the numerical record.

    python scripts/make_tables.py --data results/paper/data --output results/paper/tables

Writes LaTeX fragments ``tab_*.tex`` and ``numbers.tex``, and the full set of
pairwise comparisons with Holm-adjusted p-values to ``pairwise_tests.csv``,
all in ``<output>``.
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.sandwich_covariance import S_hac_simple

LAGS = 12  # Bartlett lags of the Newey-West long-run variances (Appendix D.3)
PERIODS = ((1995, 2009), (2010, 2024))  # halves of input months
KERNELS = [
    ("affine", "constant", "Affine", "const."),
    ("affine", "rbf", "Affine", "RBF"),
    ("rbf", "constant", "RBF", "const."),
    ("rbf", "rbf", "RBF", "RBF"),
]
IPCA = [("ipca", "IPCA"), ("ipca_intercept", "IPCA+int.")]
FIELDS = [
    "r2_percent",
    "rank_percent",
    "log_score",
    "tangency_sharpe",
    "long_short_sharpe",
    "gmv_percent",
]
DIGITS = [3, 2, 3, 2, 2, 1]
HEADERS = [
    r"$R^2$ (\%)",
    r"Rank (\%)",
    "Log score",
    "Tangency",
    "Long--short",
    r"GMV vol.\ (\%)",
]
CRITERIA = {
    "r2": r"$R^2$",
    "rank": "Rank",
    "joint": "Log score",
    "tangency": "Tangency",
    "long_short": "Long--short",
    "gmv": "GMV vol.",
}


# ----------------------------------------------------------------------------
# Formatting
# ----------------------------------------------------------------------------
def sharpe(returns: pd.Series) -> float:
    """Annualized Sharpe ratio of monthly excess returns."""
    return float(np.sqrt(12) * returns.mean() / returns.std(ddof=1))


def number(value: float, digits: int) -> str:
    if abs(value) < 0.5 * 10 ** (-digits):
        value = 0.0
    return f"{value:.{digits}f}"


def count(value: float) -> str:
    """An integer with thousands separators that also work in math mode."""
    return f"{int(value):,}".replace(",", "{,}")


def signed(value: float, digits: int) -> str:
    """Like number(), with a typeset minus sign for negative entries."""
    text = number(value, digits)
    return f"${text}$" if text.startswith("-") else text


def bold(value: str) -> str:
    if value.startswith("$") and value.endswith("$"):
        return r"$\boldsymbol{" + value[1:-1] + "}$"
    return r"\textbf{" + value + "}"


def latex_table(
    headers: list[str],
    rows: list[list[str]],
    alignment: str,
    *,
    sep: int = 4,
    width: int = 3,
    groups: tuple[str, ...] = ("Return forecasts", "Portfolio performance"),
) -> str:
    """Booktabs table; ``groups`` label consecutive blocks of ``width`` columns
    after the first two (only for tables with that layout)."""
    lines = [
        r"\begingroup\small",
        rf"\setlength{{\tabcolsep}}{{{sep}pt}}\renewcommand{{\arraystretch}}{{1.15}}",
        rf"\begin{{tabular*}}{{\textwidth}}{{@{{\extracolsep{{\fill}}}}{alignment}@{{}}}}",
        r"\toprule",
    ]
    if len(headers) == 2 + width * len(groups):
        spans = [rf"\multicolumn{{{width}}}{{c}}{{{group}}}" for group in groups]
        rules = [
            rf"\cmidrule(lr){{{3 + width * i}-{2 + width * (i + 1)}}}"
            for i in range(len(groups))
        ]
        lines += [" & & " + " & ".join(spans) + r" \\", "".join(rules)]
    lines += [" & ".join(headers) + r" \\", r"\midrule"]
    for index, row in enumerate(rows):
        if index and row[0].startswith("IPCA") and row[0] != rows[index - 1][0]:
            lines.append(r"\addlinespace[0.5ex]")
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular*}", r"\endgroup"]
    return "\n".join(lines) + "\n"


def results_table(
    labelled: list[tuple[str, str, pd.Series]], first: str, second: str
) -> str:
    """Rows of the six criteria; bold marks the best point estimate per column."""
    values = np.array([[row[field] for field in FIELDS] for *_, row in labelled])
    best = (values * [1, 1, 1, 1, 1, -1]).argmax(axis=0)  # lowest GMV volatility
    rows = []
    for index, (a, b, row) in enumerate(labelled):
        cells = [number(row[field], digits) for field, digits in zip(FIELDS, DIGITS)]
        rows.append(
            [a, b, *(bold(c) if best[k] == index else c for k, c in enumerate(cells))]
        )
    return latex_table([first, second, *HEADERS], rows, "llrrrrrr")


# ----------------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------------
class Record:
    def __init__(self, data: Path) -> None:
        summary = pd.read_csv(data / "summary.csv").fillna({"factor": ""})
        self.test = summary.query("period == 'test'")
        self.metrics = pd.read_csv(data / "monthly_metrics.csv").fillna({"factor": ""})
        self.portfolios = pd.read_csv(data / "monthly_portfolios.csv").fillna(
            {"factor": ""}
        )

    def panelgp(
        self, protocol: str, loading="rbf", factor="rbf", hidden_dim=5
    ) -> pd.Series:
        """Five-seed average of the PanelGP evaluation statistics."""
        rows = self.test.query(
            "model == 'panelgp' and protocol == @protocol and loading == @loading "
            "and factor == @factor and hidden_dim == @hidden_dim"
        )
        if len(rows) != 5:
            raise ValueError(
                f"Expected five seeds for {protocol} {loading}/{factor} H={hidden_dim}."
            )
        return rows[FIELDS].mean()

    def ipca(self, model: str, protocol: str) -> pd.Series:
        return self.test.query(
            "model == @model and protocol == @protocol and hidden_dim == 5"
        )[FIELDS].iloc[0]

    def by_half(self, model: str, protocol: str) -> dict[str, list[float]]:
        """Seed-averaged tangency and long-short Sharpe ratios and rank
        correlations (%) of the H = 5 model in each half (RBF/RBF PanelGP)."""
        query = "model == @model and protocol == @protocol"
        if model == "panelgp":
            query += " and loading == 'rbf' and factor == 'rbf'"
        metrics, portfolios = self.metrics.query(query), self.portfolios.query(query)
        result = {"tangency": [], "long_short": [], "rank": []}
        for first, last in PERIODS:
            returns = portfolios.query("@first <= year <= @last")
            for strategy in ("tangency", "long_short"):
                chosen = returns.query("strategy == @strategy")
                result[strategy].append(
                    chosen.groupby("seed")["return"].apply(sharpe).mean()
                )
            ranks = metrics.query("@first <= year <= @last").spearman
            result["rank"].append(100 * ranks.mean())
        return result

    def monthly(self, model: str, protocol: str, loading="rbf", factor="rbf"):
        """Seed-averaged monthly scores and seed-level monthly portfolio returns."""
        query = "model == @model and protocol == @protocol"
        if model == "panelgp":
            query += " and loading == @loading and factor == @factor"
        metrics = self.metrics.query(query)
        scores = metrics.groupby(["year", "month"]).mean(numeric_only=True)
        scores = scores.rename(
            columns={
                "sum_squared_error": "sse",
                "sum_target_squared": "target_ss",
                "spearman": "rank",
                "log_score": "joint",
            }
        )
        portfolios = self.portfolios.query(query)
        returns = {
            strategy: portfolios.query("strategy == @strategy")
            .pivot(index=["year", "month"], columns="seed", values="return")
            .sort_index()
            for strategy in ("tangency", "long_short", "gmv")
        }
        return scores, returns


# ----------------------------------------------------------------------------
# Paired comparisons (Appendix D.3)
# ----------------------------------------------------------------------------
def long_run_variance(values: np.ndarray) -> float:
    centered = values - values.mean()
    return float(S_hac_simple(centered[:, None], nlags=LAGS)[0, 0] / len(values))


def mean_test(series: pd.Series) -> tuple[float, float, float]:
    """Mean, Newey-West standard error and t-statistic of a monthly difference."""
    values = series.to_numpy(float)
    error = np.sqrt(max(long_run_variance(values), 0.0) / len(values))
    return float(values.mean()), float(error), float(values.mean() / error)


def portfolio_test(first: pd.DataFrame, second: pd.DataFrame, statistic: str):
    """Paired test of seed-averaged Sharpe ratios or volatilities (Ledoit-Wolf delta
    method). Each column is one optimization seed; positive favors ``first``."""

    def estimate(frame):
        returns = frame.to_numpy(float)
        mean, second_moment = returns.mean(axis=0), np.square(returns).mean(axis=0)
        variance = returns.var(axis=0, ddof=0)
        correction = np.sqrt(len(frame) / (len(frame) - 1))
        if statistic == "sharpe":
            value = mean / np.sqrt(variance)
            influence = second_moment / variance**1.5 * (returns - mean)
            influence -= 0.5 * mean / variance**1.5 * (returns**2 - second_moment)
            scale = np.sqrt(12) / correction
        else:
            value = np.sqrt(variance)
            influence = ((returns - mean) ** 2 - variance) / (2 * np.sqrt(variance))
            scale = np.sqrt(12) * correction
        return scale * value.mean(), scale * influence.mean(axis=1)

    if not first.index.equals(second.index):
        raise ValueError("Portfolio comparisons require identical months.")
    (a, a_influence), (b, b_influence) = estimate(first), estimate(second)
    sign = -1 if statistic == "volatility" else 1
    influence = sign * (a_influence - b_influence)
    error = np.sqrt(max(long_run_variance(influence), 0.0) / len(influence))
    return float(sign * (a - b)), float(error), float(sign * (a - b) / error)


def pairwise_tests(models: dict) -> pd.DataFrame:
    """All pairs of a table; Holm adjustment within each criterion."""
    rows = []
    for first, second in combinations(models, 2):
        (a, a_returns), (b, b_returns) = models[first], models[second]
        if not a.index.equals(b.index) or not np.allclose(
            a.target_ss, b.target_ss, rtol=1e-12
        ):
            raise ValueError(
                "Pairwise comparisons require identical targets and months."
            )
        estimates = {
            "r2": mean_test(100 * len(a) * (b.sse - a.sse) / a.target_ss.sum()),
            "rank": mean_test(100 * (a["rank"] - b["rank"])),
            "joint": mean_test(a.joint - b.joint),
        }
        for strategy in ("tangency", "long_short", "gmv"):
            kind = "volatility" if strategy == "gmv" else "sharpe"
            difference, error, t = portfolio_test(
                a_returns[strategy], b_returns[strategy], kind
            )
            scale = 100 if strategy == "gmv" else 1
            estimates[strategy] = scale * difference, scale * error, t
        rows += [
            {
                "first": first,
                "second": second,
                "criterion": criterion,
                "difference": d,
                "standard_error": e,
                "t_statistic": t,
                "p_value": 2 * norm.sf(abs(t)),
            }
            for criterion, (d, e, t) in estimates.items()
        ]
    tests = pd.DataFrame(rows)
    for _, group in tests.groupby("criterion"):
        reject, adjusted, _, _ = multipletests(group.p_value, alpha=0.05, method="holm")
        tests.loc[group.index, "p_holm"] = adjusted
        tests.loc[group.index, "reject"] = reject
    return tests


def test_table(tests: pd.DataFrame) -> str:
    rows = []
    for (first, second), group in tests.groupby(["first", "second"], sort=False):
        scores = group.set_index("criterion")
        cells = []
        for key in CRITERIA:
            cell = f"${number(scores.loc[key, 't_statistic'], 2)}$"
            cells.append(bold(cell) if scores.loc[key, "reject"] else cell)
        rows.append([first, second, *cells])
    return latex_table(["First", "Second", *CRITERIA.values()], rows, "llrrrrrr", sep=3)


# ----------------------------------------------------------------------------
# Tables and macros
# ----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", type=Path, default=Path("results/paper/data"))
    parser.add_argument("--output", type=Path, default=Path("results/paper/tables"))
    args = parser.parse_args()
    data, output = args.data, args.output
    output.mkdir(parents=True, exist_ok=True)
    record = Record(data)
    tables = {}

    # Table 2: kernel pairs and IPCA, fixed training, H = 5.
    labelled = [(a, b, record.panelgp("fixed", lo, fa)) for lo, fa, a, b in KERNELS]
    labelled += [(label, "--", record.ipca(model, "fixed")) for model, label in IPCA]
    tables["tab_kernels"] = results_table(labelled, "Loadings", "Factors")

    # Table 3: fixed versus annual re-estimation.
    labelled = [
        ("PanelGP", "Fixed", record.panelgp("fixed")),
        ("PanelGP", "Rolling", record.panelgp("rolling")),
    ]
    labelled += [
        (label, protocol.capitalize(), record.ipca(model, protocol))
        for model, label in IPCA
        for protocol in ("fixed", "rolling")
    ]
    tables["tab_rolling"] = results_table(labelled, "Model", "Training")

    # Table 4: trading costs under annual re-estimation.
    costs = pd.read_csv(data / "trading_costs.csv").set_index(["model", "strategy"])
    names = {
        "panelgp": "PanelGP",
        "ipca": "IPCA",
        "ipca_intercept": "IPCA+int.",
        "tangency": "Tangency",
        "long_short": "Long--short",
    }
    rows = [
        [
            names[model],
            names[strategy],
            *(
                number(costs.loc[(model, strategy), column], digits)
                for column, digits in (
                    ("mean_monthly_turnover", 2),
                    ("gross_sharpe", 2),
                    ("net_sharpe_10bp", 2),
                    ("break_even_bp", 1),
                )
            ),
        ]
        for model in ("panelgp", "ipca", "ipca_intercept")
        for strategy in ("tangency", "long_short")
    ]
    tables["tab_trading_costs"] = latex_table(
        [
            "Model",
            "Portfolio",
            "Turnover",
            "Gross SR",
            "Net SR, 10 bp",
            "Break-even (bp)",
        ],
        rows,
        "llrrrr",
    )

    # Table S3: Table 3 by half.
    halves = {
        (label, protocol.capitalize()): record.by_half(model, protocol)
        for model, label in [("panelgp", "PanelGP"), *IPCA]
        for protocol in ("fixed", "rolling")
    }
    years = [f"{first % 100:02d}--{last % 100:02d}" for first, last in PERIODS]
    rows = [
        [
            label,
            protocol,
            *(signed(v, 2) for v in values["tangency"]),
            *(signed(v, 2) for v in values["long_short"]),
            *(signed(v, 2) for v in values["rank"]),
        ]
        for (label, protocol), values in halves.items()
    ]
    tables["tab_subperiods"] = latex_table(
        ["Model", "Training", *years, *years, *years],
        rows,
        "llrrrrrr",
        width=2,
        groups=("Tangency Sharpe", "Long--short Sharpe", r"Rank (\%)"),
    )

    # Tables S1 and S2: paired comparisons within Tables 2 and 3.
    kernels = {
        f"{a}/{b}": record.monthly("panelgp", "fixed", lo, fa)
        for lo, fa, a, b in KERNELS
    }
    protocols = {
        "PanelGP fixed": kernels["RBF/RBF"],
        "PanelGP rolling": record.monthly("panelgp", "rolling"),
    }
    for model, label in IPCA:
        for protocol in ("fixed", "rolling"):
            protocols[f"{label} {protocol}"] = record.monthly(model, protocol)
        kernels[label] = protocols[f"{label} fixed"]
    tests = []
    for name, models in (
        ("tab_test_statistics", kernels),
        ("tab_rolling_tests", protocols),
    ):
        result = pairwise_tests(models)
        tables[name] = test_table(result)
        tests.append(result.assign(table=name))
    pd.concat(tests, ignore_index=True).to_csv(
        output / "pairwise_tests.csv", index=False
    )

    # Table 1: the controlled simulation.
    synthetic = pd.read_csv(data / "synthetic.csv")
    factor = 100 * synthetic[["factor_probe_1_r2", "factor_probe_2_r2"]].mean(axis=1)
    loading = 100 * synthetic[["loading_probe_1_r2", "loading_probe_2_r2"]].mean(axis=1)
    quantities = [
        (r"Factor function $R^2$ (\%)", 100.0, factor, 2),
        (r"Loading function $R^2$ (\%)", 100.0, loading, 2),
        (
            r"Loading innovation $\sigma_\beta^2$",
            0.05,
            synthetic.loading_innovation_average,
            3,
        ),
        (
            r"Factor innovation $\sigma_\phi^2$",
            0.05,
            synthetic.factor_innovation_average,
            3,
        ),
        (
            r"Observation noise $\sigma_r^2$",
            0.05,
            synthetic.observation_variance_estimate,
            3,
        ),
    ]
    rows = [
        [
            label,
            number(truth, d),
            number(values.mean(), d),
            number(values.std(ddof=1), d),
        ]
        for label, truth, values, d in quantities
    ]
    tables["tab_synthetic"] = latex_table(
        ["Quantity", "True", "Estimated mean", "Replication s.d."], rows, "lrrr"
    )
    for name, text in tables.items():
        (output / f"{name}.tex").write_text(text, encoding="utf-8")

    # Numbers quoted in the text.
    fixed, rolling = record.panelgp("fixed"), record.panelgp("rolling")
    density = pd.read_csv(data / "density_gain.csv")
    ranks = (
        pd.read_csv(data / "risk_rank_correlations.csv")
        .groupby("comparison")
        .spearman.mean()
    )
    design = synthetic[["assets", "months", "observations", "epochs"]].drop_duplicates()
    if len(design) != 1:
        raise ValueError("Simulation replications differ in design.")
    design = design.iloc[0]
    # Sample of Section 6.1: the selected characteristics and the stocks per month.
    features = pd.read_csv(data / "lengthscales.csv").query("kind == 'loading'")
    characteristics = features.feature.nunique()
    stocks = (
        pd.read_csv(data / "monthly_metrics.csv")
        .query("model == 'panelgp' and protocol == 'fixed' and seed == 1")
        .query("loading == 'rbf' and factor == 'rbf'")
        .n
    )
    if len(stocks) != 360:
        raise ValueError("The evaluation panel must cover 360 months.")
    macros = {
        "Characteristics": str(characteristics),
        "AffineFunctions": str(characteristics + 1),
        "LoadingLengthscale": number(np.sqrt(characteristics / 12), 2),
        "EvaluationRows": count(stocks.sum()),
        "MinStocks": str(int(stocks.min())),
        "MaxStocks": str(int(stocks.max())),
        "DensityGain": number(density.gain_sum.sum() / density.n.sum(), 3),
        "RbfThreeTangency": number(
            record.panelgp("fixed", hidden_dim=3).tangency_sharpe, 2
        ),
        "SynRepresentative": str(
            int(synthetic.sort_values("average_factor_r2").iloc[2].seed)
        ),
        "SynAssets": count(design.assets),
        "SynMonths": count(design.months),
        "SynRows": count(design.observations),
        "SynEpochs": count(design.epochs),
        "SynFactorRtwo": number(factor.mean(), 2),
        "SynLoadingRtwo": number(loading.mean(), 2),
        "SynMeanRtwo": number(100 * synthetic.mean_r2.mean(), 2),
        "SynVarianceCorrelation": number(synthetic.variance_correlation.mean(), 2),
        "SynVarianceRatio": number(synthetic.variance_ratio.mean(), 2),
        "FixedRank": number(fixed.rank_percent, 2),
        "FixedJoint": number(fixed.log_score, 3),
        "FixedTangency": number(fixed.tangency_sharpe, 2),
        "FixedGMV": number(fixed.gmv_percent, 1),
        "RollingRank": number(rolling.rank_percent, 2),
        "RollingJoint": number(rolling.log_score, 3),
        "RollingTangency": number(rolling.tangency_sharpe, 2),
        "RollingGMV": number(rolling.gmv_percent, 1),
        "PanelNetTangencyTen": number(
            costs.loc[("panelgp", "tangency"), "net_sharpe_10bp"], 2
        ),
        "PanelNetLongShortTen": number(
            costs.loc[("panelgp", "long_short"), "net_sharpe_10bp"], 2
        ),
        "PanelBreakEvenTangency": number(
            costs.loc[("panelgp", "tangency"), "break_even_bp"], 1
        ),
        "PanelBreakEvenLongShort": number(
            costs.loc[("panelgp", "long_short"), "break_even_bp"], 1
        ),
        "IpcaInterceptNetTangencyTen": number(
            costs.loc[("ipca_intercept", "tangency"), "net_sharpe_10bp"], 2
        ),
        "IpcaInterceptNetLongShortTen": number(
            costs.loc[("ipca_intercept", "long_short"), "net_sharpe_10bp"], 2
        ),
        "LoadingTimeRank": number(100 * ranks["loading_across_months"], 1),
        "FactorCrossRank": number(100 * ranks["factor_across_stocks"], 1),
        "IpcaFixedTangency": number(record.ipca("ipca", "fixed").tangency_sharpe, 2),
        "IpcaInterceptFixedTangency": number(
            record.ipca("ipca_intercept", "fixed").tangency_sharpe, 2
        ),
        "IpcaInterceptRollingTangency": number(
            record.ipca("ipca_intercept", "rolling").tangency_sharpe, 2
        ),
    }
    # Section 5.2: innovation recovery in the representative replication (Fig. 1).
    dates = pd.read_csv(data / "synthetic_factor_dates.csv")
    for hidden, name in ((1, "One"), (2, "Two")):
        true = dates[f"true_realized_phi{hidden}"] - dates[f"true_phi{hidden}"]
        recovered = dates[f"variational_phi{hidden}"] - dates[f"aligned_phi{hidden}"]
        r2 = 1 - np.mean((true - recovered) ** 2) / np.var(true)
        macros[f"SynInnovationRtwo{name}"] = number(100 * r2, 1)
    # Section 6.6: the fixed-rolling R^2 difference and its common and
    # cross-sectional parts.
    rolling_tests = tests[1].set_index(["first", "second", "criterion"])
    macros["FixedRollingRtwoStat"] = number(
        rolling_tests.loc[("PanelGP fixed", "PanelGP rolling", "r2"), "t_statistic"], 2
    )
    split = pd.read_csv(data / "r2_split.csv").groupby("protocol").mean()
    for protocol in ("fixed", "rolling"):
        for part in ("common", "cross"):
            name = f"{protocol.capitalize()}Rtwo{part.capitalize()}"
            macros[name] = number(split.loc[protocol, f"{part}_percent"], 3)
    # Section 6.6 and Appendix D.4: halves of Table S3.
    market = pd.read_csv(data / "market_benchmark.csv").set_index("period").sharpe
    rolling = halves[("PanelGP", "Rolling")]["tangency"]
    for position, (first, last), value in zip(("First", "Second"), PERIODS, rolling):
        macros[f"HalfRollingTangency{position}"] = number(value, 2)
        macros[f"HalfMarket{position}"] = number(market[f"{first}-{last}"], 2)
    (output / "numbers.tex").write_text(
        "".join(
            f"\\newcommand{{\\{key}}}{{{value}}}\n" for key, value in macros.items()
        ),
        encoding="utf-8",
    )
    print(f"Wrote tables and numbers to {output}")


if __name__ == "__main__":
    main()
