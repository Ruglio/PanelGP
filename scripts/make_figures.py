"""Figures 1-4 and S1-S3 from the numerical record.

    python scripts/make_figures.py --data results/paper/data --output results/paper/figures

Red marks PanelGP with RBF loadings and RBF factors. Figures are drawn at
their printed width, so type sizes are final.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter, MaxNLocator, NullFormatter

DATA = Path("results/paper/data")
FIGURES = Path("results/paper/figures")

WIDTH = 6.5  # inches, the text width of the manuscript
FONTS = ["Nimbus Sans", "Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"]
INK = "#1a1a1a"  # text, baselines and the realized quantities
MUTED = "#6e6e6e"  # secondary text and benchmarks
LIGHT = "#a6a6a6"
GUIDE = "#d9d9d9"  # hairline guides
TINT = "#ececec"  # recession blocks
BAR = "#c8c8c8"
RED = "#d7261e"  # PanelGP with RBF loadings and RBF factors
BLUE = "#2d5da8"
# Kernel pairs in a fixed order; the four hues pass color-vision-deficiency
# separation checks on all pairs, and every series also has its own marker.
KERNELS = [
    ("affine", "constant", "Affine / constant", BLUE, "o"),
    ("affine", "rbf", "Affine / RBF", "#c79a1c", "s"),
    ("rbf", "constant", "RBF / constant", "#00897b", "D"),
    ("rbf", "rbf", "RBF / RBF", RED, "^"),
]
BENCHMARKS = [
    ("ipca", "IPCA", MUTED, "x"),
    ("ipca_intercept", "IPCA + intercept", INK, "+"),
]
DASHED = (0, (4, 2))
CRISES = [
    ("2001-03-01", "2001-12-01"),
    ("2007-12-01", "2009-07-01"),
    ("2020-02-01", "2020-05-01"),
]


def style() -> None:
    available = {font.name for font in font_manager.fontManager.ttflist}
    family = next(name for name in FONTS if name in available or name == FONTS[-1])
    plt.rcParams.update(
        {
            "font.family": family,
            "mathtext.fontset": "custom",
            "mathtext.rm": family,
            "mathtext.it": f"{family}:italic",
            "mathtext.bf": f"{family}:bold",
            "mathtext.cal": family,
            "mathtext.sf": family,
            "font.size": 8,
            "text.color": INK,
            "axes.titlesize": 8.5,
            "axes.titlelocation": "left",
            "axes.titlepad": 7,
            "axes.labelsize": 8,
            "axes.labelcolor": INK,
            "axes.edgecolor": INK,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "axes.axisbelow": True,
            "grid.color": GUIDE,
            "grid.linewidth": 0.45,
            "xtick.major.size": 2.5,
            "xtick.major.width": 0.6,
            "xtick.labelcolor": MUTED,
            "ytick.major.size": 0,
            "xtick.minor.size": 0,
            "ytick.minor.size": 0,
            "ytick.major.pad": 2,
            "ytick.labelcolor": MUTED,
            "legend.fontsize": 7.5,
            "legend.frameon": False,
            "legend.handlelength": 2.4,
            "lines.linewidth": 1.15,
            "lines.markersize": 3.4,
            "lines.markeredgewidth": 0.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    """Prefix each panel title with a bold letter and write the PDF."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    axes = [ax for ax in fig.axes if ax.axison and ax.get_label() != "<colorbar>"]
    for letter, ax in zip("abcdefghijklmnopqrstuvwxyz", axes):
        title = ax.get_title(loc="left") or ax.get_title()
        ax.set_title("")
        ax.set_title(rf"$\mathbf{{{letter}}}$   {title}".rstrip(), loc="left")
    fig.savefig(
        FIGURES / f"{name}.pdf",
        bbox_inches="tight",
        pad_inches=0.02,
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(fig)


def shade_crises(ax: plt.Axes) -> None:
    for start, end in CRISES:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color=TINT, lw=0, zorder=0)


def synthetic_factor_figure() -> None:
    """Separate smooth-factor prediction from training-innovation recovery."""
    probe = pd.read_csv(DATA / "synthetic_factor_probe.csv")
    dates = pd.read_csv(DATA / "synthetic_factor_dates.csv")
    innovations = [
        (
            dates[f"true_realized_phi{h}"] - dates[f"true_phi{h}"],
            dates[f"variational_phi{h}"] - dates[f"aligned_phi{h}"],
        )
        for h in (1, 2)
    ]
    innovation_bound = 1.12 * np.abs(np.asarray(innovations)).max()
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(WIDTH, 4.6),
        sharex="col",
        layout="constrained",
        gridspec_kw={"width_ratios": [1.45, 1]},
    )
    for hidden, (ax, innovation_ax) in enumerate(axes, 1):
        x = probe.c1.to_numpy(float)
        truth = probe[f"true_phi{hidden}"].to_numpy(float)
        estimate = probe[f"aligned_phi{hidden}"].to_numpy(float)
        estimate_sd = probe[f"aligned_phi{hidden}_sd"].to_numpy(float)
        ax.fill_between(
            x,
            estimate - 1.96 * estimate_sd,
            estimate + 1.96 * estimate_sd,
            color=RED,
            alpha=0.12,
            lw=0,
            label="New-date 95% predictive band",
        )
        ax.plot(x, truth, color=INK, lw=1.2, label="True smooth response")
        ax.plot(x, estimate, color=RED, lw=1.3, ls="-.", label="Fitted smooth response")
        ax.scatter(
            dates.c1,
            dates[f"true_realized_phi{hidden}"],
            marker="x",
            s=16,
            linewidths=0.8,
            color=MUTED,
            label="True realized factor",
            zorder=4,
        )
        ax.scatter(
            dates.c1,
            dates[f"variational_phi{hidden}"],
            s=11,
            color=BLUE,
            linewidths=0,
            label="Variational training mean",
            zorder=5,
        )
        ax.set(title=f"Factor {hidden}: response and realizations")
        ax.set_ylabel("Factor value")
        ax.set_xticks(np.linspace(0, 1, 5))

        true_innovation, inferred_innovation = innovations[hidden - 1]
        r2 = 1 - np.mean((true_innovation - inferred_innovation) ** 2) / np.var(
            true_innovation
        )
        innovation_ax.axline((0, 0), slope=1, color=INK, lw=0.6, ls=DASHED)
        innovation_ax.scatter(
            true_innovation, inferred_innovation, s=11, color=BLUE, lw=0, zorder=3
        )
        innovation_ax.set(
            title=f"Factor {hidden}: innovations",
            ylabel="Recovered innovation",
            xlim=(-innovation_bound, innovation_bound),
            ylim=(-innovation_bound, innovation_bound),
        )
        innovation_ax.set_aspect("equal", adjustable="box")
        innovation_ax.xaxis.set_major_locator(MaxNLocator(4))
        innovation_ax.yaxis.set_major_locator(MaxNLocator(4))
        innovation_ax.text(
            0.04,
            0.96,
            f"$R^2 = {100 * r2:.1f}\\%$",
            transform=innovation_ax.transAxes,
            va="top",
        )
    axes[-1, 0].set_xlabel("Context")
    axes[-1, 1].set_xlabel("True innovation")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower left", ncol=3, columnspacing=1.6)
    save(fig, "factor_recovery")


def synthetic_loading_figure() -> None:
    """Compare true and fitted nonlinear loading surfaces on common scales."""
    frame = pd.read_csv(DATA / "synthetic_loading_probe.csv")
    x = np.sort(frame.x1.unique())
    y = np.sort(frame.x2.unique())
    diverging = LinearSegmentedColormap.from_list("swiss", [BLUE, "#ffffff", RED])
    fig, axes = plt.subplots(
        2, 2, figsize=(0.9 * WIDTH, 4.6), sharex=True, sharey=True, layout="constrained"
    )
    for hidden in (1, 2):
        truth = (
            frame.pivot(index="x2", columns="x1", values=f"true_beta{hidden}")
            .loc[y, x]
            .to_numpy()
        )
        estimate = (
            frame.pivot(index="x2", columns="x1", values=f"aligned_beta{hidden}")
            .loc[y, x]
            .to_numpy()
        )
        # Scale from the truth only: it is a fixed closed-form function, so the
        # color scale (and the truth panel's appearance) stays identical across
        # replications and reruns. The fitted estimate is clipped by extend="both"
        # when it overshoots.
        bound = abs(truth).max()
        levels = np.linspace(-bound, bound, 19)
        for column, (values, title) in enumerate(
            [
                (truth, f"DGP loading {hidden}"),
                (estimate, f"PanelGP loading {hidden}"),
            ]
        ):
            ax = axes[hidden - 1, column]
            contour = ax.contourf(
                x, y, values, levels=levels, cmap=diverging, extend="both"
            )
            ax.set_title(title)
            ax.set_aspect("equal")
            ax.grid(False)
            ax.spines["left"].set_visible(True)
            ax.tick_params(axis="y", length=2.5, width=0.6)
        colorbar = fig.colorbar(
            contour, ax=axes[hidden - 1, :], shrink=0.88, pad=0.03, format="%.1f"
        )
        colorbar.outline.set_visible(False)
        colorbar.ax.tick_params(length=0, labelcolor=MUTED)
    for ax in axes[-1, :]:
        ax.set_xlabel(r"$x_1$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$x_2$")
    save(fig, "loading_recovery")


def summary(period: str) -> pd.DataFrame:
    frame = pd.read_csv(DATA / "summary.csv").fillna({"factor": ""})
    return frame.query("protocol == 'fixed' and period == @period")


def kernel_curves(ax: plt.Axes, data: pd.DataFrame, column: str) -> None:
    """Five-seed means of the four PanelGP kernel pairs, with ±1 seed SD bands."""
    for loading, factor, label, color, marker in KERNELS:
        group = data.query("loading == @loading and factor == @factor")
        stats = group.groupby("hidden_dim")[column].agg(["mean", "std"])
        ax.fill_between(
            stats.index,
            stats["mean"] - stats["std"],
            stats["mean"] + stats["std"],
            color=color,
            alpha=0.12,
            lw=0,
        )
        ax.plot(
            stats.index,
            stats["mean"],
            color=color,
            marker=marker,
            mfc=color,
            lw=1.6 if color == RED else 1.0,
            zorder=3 if color == RED else 2,
            label=label,
        )
    ax.axvline(5, color=LIGHT, lw=0.6, ls=DASHED, zorder=0)
    ax.set_xticks(range(1, 9))


def dimension_figure() -> None:
    """Plot seed means with ± one sample SD (not a confidence interval)."""
    metrics = [
        ("r2_percent", r"Predictive $R^2$ (%)"),
        ("rank_percent", "Monthly rank correlation (%)"),
        ("log_score", "Gaussian log score"),
        ("tangency_sharpe", "Tangency Sharpe ratio"),
        ("long_short_sharpe", "Long–short Sharpe ratio"),
    ]
    dimension = summary("test").query("model == 'panelgp'")
    ipca = summary("test").query("model != 'panelgp'")
    market = pd.read_csv(DATA / "market_benchmark.csv").query("period == '1995-2024'")
    market = float(market.sharpe.iloc[0])
    fig, axes = plt.subplots(
        2, 3, figsize=(WIDTH, 4.3), sharex=True, layout="constrained"
    )
    for ax, (column, title) in zip(axes.flat, metrics):
        kernel_curves(ax, dimension, column)
        for name, label, color, marker in BENCHMARKS:
            group = ipca.query("model == @name").sort_values("hidden_dim")
            ax.plot(
                group.hidden_dim,
                group[column],
                color=color,
                marker=marker,
                lw=0.8,
                ls=DASHED,
                label=label,
            )
        if column.endswith("sharpe"):
            ax.axhline(
                market, color=INK, lw=0.8, ls=(0, (1, 1.5)), zorder=1, label="S&P 500"
            )
        ax.set_title(title)
    for ax in axes[1]:
        ax.set_xlabel(r"Number of factors $H$")
    key = axes[1, 2]
    key.axis("off")
    handles, labels = axes[1, 0].get_legend_handles_labels()
    groups = [
        ("PanelGP, mean ± 1 seed SD", slice(0, 4)),
        ("Benchmarks", slice(4, None)),
    ]
    top = 1.0
    for title, rows in groups:
        legend = key.legend(
            handles[rows],
            labels[rows],
            loc="upper left",
            bbox_to_anchor=(0, top),
            title=title,
            alignment="left",
            borderaxespad=0,
        )
        legend.get_title().set_fontweight("bold")
        key.add_artist(legend)
        # Stack the next group below the measured extent of this one.
        fig.canvas.draw()
        extent = legend.get_window_extent().transformed(key.transAxes.inverted())
        top = extent.y0 - 0.08
    save(fig, "hidden_dimension")


def validation_figure() -> None:
    """Display the validation grid without pooling optimization seeds."""
    data = summary("validation").query("model == 'panelgp'")
    counts = data.groupby(["loading", "factor", "hidden_dim"]).seed.nunique()
    if len(data) != 160 or len(counts) != 32 or not counts.eq(5).all():
        raise ValueError("Validation curves require all 32 designs and five seeds.")
    metrics = [
        ("r2_percent", r"Predictive $R^2$ (%)"),
        ("rank_percent", "Monthly rank correlation (%)"),
        ("log_score", "Gaussian log score"),
        ("tangency_sharpe", "Tangency Sharpe ratio"),
    ]
    fig, axes = plt.subplots(
        2, 2, figsize=(WIDTH, 4.0), sharex=True, layout="constrained"
    )
    for ax, (column, title) in zip(axes.flat, metrics):
        kernel_curves(ax, data, column)
        ax.set_title(title)
    for ax in axes[1]:
        ax.set_xlabel(r"Number of factors $H$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper left", ncol=4, columnspacing=1.6)
    save(fig, "validation_dimension")


def ard_lengthscale_figure() -> None:
    """Show inverse RBF lengthscales and component-level dispersion."""
    data = pd.read_csv(DATA / "lengthscales.csv")
    display_names = {  # as named in the text
        "retx": "Market return",
        "avgcor": "Average correlation",
        "dfy": "Default-yield spread",
        "svar": "Market variance",
        "infl": "Inflation",
        "tms": "Term spread",
        "div12m_me": "Dividend yield",
        "ret_1_0": "One-month return",
        "prc": "Price",
        "dolvol_126d": "Dollar volume (126 days)",
        "prc_highprc_252d": "Price / 52-week high",
        "betabab_1260d": "BAB beta (5 years)",
        "zero_trades_21d": "Zero-trading days (1 month)",
        "ivol_capm_252d": "CAPM residual volatility (1 year)",
        "turnover_126d": "Turnover (126 days)",
        "beta_60m": "Market beta (5 years)",
        "market_equity": "Market capitalization",
        "ebit_sale": "Operating profit / sales",
        "ni_ivol": "Earnings volatility",
        "sale_me": "Sales-to-market",
    }
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(WIDTH, 2.7),
        layout="constrained",
        gridspec_kw={"width_ratios": [1.15, 1]},
    )
    panels = [
        ("loading", 12, "Loading inputs", "Characteristic"),
        ("factor", 6, "Factor inputs", "Economic context"),
    ]
    rng = np.random.default_rng(1947)
    for ax, (kind, keep, title, ylabel) in zip(axes, panels):
        subset = data.query("kind == @kind").copy()
        order = (
            subset.groupby("feature")
            .inverse_lengthscale.mean()
            .nlargest(keep)
            .sort_values()
            .index
        )
        subset = subset[subset.feature.isin(order)]
        means = subset.groupby("feature").inverse_lengthscale.mean().reindex(order)
        positions = np.arange(len(order))
        ax.barh(positions, means, color=BAR, height=0.66, label="Average", zorder=1)
        mapping = {feature: position for position, feature in enumerate(order)}
        y = subset.feature.map(mapping).to_numpy(float)
        y += rng.uniform(-0.22, 0.22, len(y))
        ax.scatter(
            subset.inverse_lengthscale,
            y,
            s=8,
            color=RED,
            lw=0,
            label="Latent component and seed",
        )
        ax.set_yticks(
            positions,
            [display_names.get(name, name.replace("_", " ")) for name in order],
        )
        ax.tick_params(axis="y", labelcolor=INK)
        ax.set_xlabel(r"Inverse lengthscale $1/\ell$")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
    handles, labels = axes[0].get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: labels[i] != "Average")
    fig.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        loc="outside upper left",
        ncol=2,
        columnspacing=1.6,
    )
    save(fig, "ard_lengthscales")


def density_gain_figure() -> None:
    """Locate the bilinear density gain by tail depth and calendar time."""
    data = pd.read_csv(DATA / "density_gain.csv")
    absolute_bucket = {4: 1, 5: 1, 3: 2, 6: 2, 2: 3, 7: 3, 1: 4, 8: 4}
    data["absolute_bucket"] = data.bucket.map(absolute_bucket)
    pooled = data.groupby("absolute_bucket").agg(
        n=("n", "sum"), total=("gain_sum", "sum")
    )
    # Each bar is the bin's contribution to the full-sample average gain, so
    # the bars add up to the average gain.
    pooled["gain"] = pooled.total / pooled.n.sum()
    monthly = (
        data.groupby(["year", "month"])
        .agg(n=("n", "sum"), total=("gain_sum", "sum"))
        .reset_index()
    )
    monthly["gain"] = monthly.total / monthly.n
    # Stored dates index the inputs; the return and its error realize in t+1.
    monthly["date"] = pd.to_datetime(
        {"year": monthly.year, "month": monthly.month, "day": 1}
    ) + pd.offsets.MonthBegin(1)
    monthly = monthly.set_index("date").sort_index()
    smooth = monthly.gain.rolling(12, min_periods=6).mean()

    fig, (left, right) = plt.subplots(
        1,
        2,
        figsize=(WIDTH, 2.3),
        gridspec_kw={"width_ratios": [1, 2.3]},
        layout="constrained",
    )
    x = np.arange(1, 5)
    colors = np.where(pooled.gain.to_numpy() >= 0, INK, RED)
    left.bar(x, pooled.gain, color=colors, width=0.62)
    left.axhline(0, color=INK, lw=0.6)
    left.set_xticks(x, [r"$<1$", r"$1$–$2$", r"$2$–$3$", r"$\geq3$"])
    left.set_xlabel("Absolute standardized error")
    left.set_ylabel("Contribution to average gain")
    left.set_title("By forecast error")
    left.text(
        0.04,
        0.96,
        f"Total {pooled.gain.sum():+.3f}",
        transform=left.transAxes,
        va="top",
    )

    shade_crises(right)
    right.plot(monthly.index, monthly.gain, color=LIGHT, lw=0.5, label="Monthly")
    right.plot(smooth.index, smooth, color=RED, lw=1.3, label="12-month average")
    right.axhline(0, color=INK, lw=0.6)
    right.set_ylabel("Average log-score gain")
    right.set_title("Over time")
    right.legend(loc="upper left", ncol=2, columnspacing=1.6)
    save(fig, "density_gain")


def variance_evidence_figure() -> None:
    """Show exact innovation components against subsequent forecast errors."""
    monthly = pd.read_csv(DATA / "risk_monthly.csv")
    # Stored dates index the inputs; the return and its error realize in t+1.
    monthly["date"] = pd.to_datetime(
        {"year": monthly.year, "month": monthly.month, "day": 1}
    ) + pd.offsets.MonthBegin(1)
    time_series = monthly.groupby("date").mean(numeric_only=True)
    within = pd.read_csv(DATA / "risk_stock_quintiles.csv")
    within = within.groupby("bucket")[
        ["error2_sum", "total_sum", "factor_sum", "n"]
    ].sum()
    stock_rows = np.column_stack(
        (
            100 * np.sqrt(within.error2_sum / within.n),
            100 * np.sqrt(within.total_sum / within.n),
            100 * np.sqrt(within.factor_sum / within.n),
        )
    )
    temporal = (
        pd.read_csv(DATA / "risk_month_quintiles.csv")
        .groupby("bucket")[["error2", "total", "loading_innovation"]]
        .mean()
    )
    month_rows = 100 * np.sqrt(temporal.to_numpy())

    fig = plt.figure(figsize=(0.88 * WIDTH, 3.4), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[1.15, 1])
    top = fig.add_subplot(grid[0, :])
    shade_crises(top)
    top.plot(
        time_series.index,
        100 * np.sqrt(time_series.error2),
        color=INK,
        lw=0.9,
        label="Realized RMS forecast error",
    )
    top.plot(
        time_series.index,
        100 * np.sqrt(time_series.total),
        color=RED,
        lw=1.3,
        label="Total predicted risk",
    )
    top.plot(
        time_series.index,
        100 * np.sqrt(time_series.loading_innovation),
        color=BLUE,
        lw=0.9,
        ls=DASHED,
        label=r"Loading innovation $V^\beta$",
    )
    top.set_yscale("log")
    top.set_ylabel("Monthly risk (%)")
    top.set_title("Risk over time")
    top.set_yticks([4, 8, 16])
    top.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    top.yaxis.set_minor_formatter(NullFormatter())

    left = fig.add_subplot(grid[1, 0])
    right = fig.add_subplot(grid[1, 1])
    for ax, values, title, xlabel, component, component_color in [
        (
            left,
            month_rows,
            "Across months",
            "Loading-innovation quintile",
            r"Loading innovation $V^\beta$",
            BLUE,
        ),
        (
            right,
            stock_rows,
            "Across stocks",
            "Factor-innovation quintile",
            r"Factor innovation $V^\phi$",
            MUTED,
        ),
    ]:
        ax.plot(
            range(1, 6),
            values[:, 0],
            color=INK,
            marker="o",
            label="Realized RMS forecast error",
        )
        ax.plot(
            range(1, 6),
            values[:, 1],
            color=RED,
            marker="s",
            lw=1.3,
            label="Total predicted risk",
        )
        ax.plot(
            range(1, 6),
            values[:, 2],
            color=component_color,
            marker="^",
            ls=DASHED,
            label=component,
        )
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_xticks(range(1, 6), [f"Q{i}" for i in range(1, 6)])
        ax.set_yscale("log")
        ax.set_yticks([3, 6, 12])
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
    left.set_ylabel("Risk (%)")
    handles, labels = [], []
    for ax in (top, left, right):
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.legend(handles, labels, loc="outside upper left", ncol=2, columnspacing=1.6)
    save(fig, "variance_evidence")


def main() -> None:
    global DATA, FIGURES
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=FIGURES)
    args = parser.parse_args()
    DATA, FIGURES = args.data, args.output
    style()
    synthetic_factor_figure()
    synthetic_loading_figure()
    dimension_figure()
    validation_figure()
    ard_lengthscale_figure()
    density_gain_figure()
    variance_evidence_figure()
    print(f"Wrote figures to {FIGURES}")


if __name__ == "__main__":
    main()
