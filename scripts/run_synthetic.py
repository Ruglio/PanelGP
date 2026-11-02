"""One replication of the controlled simulation (Section 5, Table 1, Figures 1, S1).

    python scripts/run_synthetic.py --seed 4 --output results/runs/synthetic/seed4

The panel seed equals the training seed. The bilinear form is identified only
up to an invertible map, so estimated factors are aligned to the true smooth
factors by least squares and loadings and innovation covariances are
transformed accordingly (Section 5.1). Writes ``summary.csv`` and the factor
and loading recovery data behind the figures.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr

from panelgp.experiments.config import load_config, write_config
from panelgp.model import build_time_grid, fit_model, to_tensors
from panelgp.model.synthetic import SCHEMA, simulate_panel, true_factors, true_loadings


def numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def r2(estimate: np.ndarray, truth: np.ndarray) -> float:
    return float(
        1 - np.sum((estimate - truth) ** 2) / np.sum((truth - truth.mean()) ** 2)
    )


def latent(model, x: np.ndarray, context: np.ndarray, time_index=None):
    frame = pd.DataFrame(
        {"x1": x[:, 0], "x2": x[:, 1], "c1": context, SCHEMA.target_column: 0.0}
    )
    data = to_tensors(
        frame, SCHEMA, device=model.time_features.device, time_index=time_index
    )
    with torch.no_grad():
        return model.row_moments(data.x, data.context, data.time_index)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/synthetic.yaml")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    config = load_config(args.config, [f"runtime.seed={args.seed}", *args.set])
    design = config["simulation"]
    frame = simulate_panel(args.seed, **design)
    model = fit_model(frame, SCHEMA, config).model.eval()
    scale = float(model.target_std)

    # Predictions at every date treated as new, and variational training factors.
    _, index_map = build_time_grid(frame, SCHEMA)
    rows = []
    for (year, month), block in frame.groupby(["year", "month"], sort=True):
        block = block.reset_index(drop=True)
        data = to_tensors(block, SCHEMA, device=model.time_features.device)
        moments = model.cross_section_moments(data.x, data.context[0], -1)
        index = index_map.query("year == @year and month == @month").time_index.iloc[0]
        new = latent(model, block[["x1", "x2"]].to_numpy(), block.c1.to_numpy())
        trained = latent(
            model,
            block[["x1", "x2"]].to_numpy(),
            block.c1.to_numpy(),
            np.full(len(block), index),
        )
        block["mean"] = numpy(moments.mean)
        block["variance"] = np.diag(numpy(moments.covariance))
        for h in (0, 1):
            block[f"estimated_phi{h + 1}"] = numpy(new.factor_mean[:, h])
            block[f"variational_phi{h + 1}"] = numpy(trained.factor_mean[:, h])
        rows.append(block)
    predictions = pd.concat(rows, ignore_index=True)

    # Align estimated to true smooth factors: estimated @ A ~ true.
    dates = predictions.drop_duplicates(["year", "month"]).sort_values(
        ["year", "month"]
    )
    phi_columns = ["estimated_phi1", "estimated_phi2"]
    transform, *_ = np.linalg.lstsq(
        dates[phi_columns].to_numpy(float),
        dates[["true_phi1", "true_phi2"]].to_numpy(float),
        rcond=None,
    )
    loading_transform = np.linalg.inv(transform).T

    factor_dates = dates[
        [
            "year",
            "month",
            "c1",
            "true_phi1",
            "true_phi2",
            "true_realized_phi1",
            "true_realized_phi2",
        ]
    ].copy()
    factor_dates[["aligned_phi1", "aligned_phi2"]] = (
        dates[phi_columns].to_numpy() @ transform
    )
    factor_dates[["variational_phi1", "variational_phi2"]] = (
        dates[["variational_phi1", "variational_phi2"]].to_numpy() @ transform
    )

    # Smooth factor maps on a fine context grid, with new-date predictive s.d.
    context = np.linspace(0.0, 1.0, 481)
    probe = latent(model, np.full((len(context), 2), 0.5), context)
    covariance = np.einsum(
        "ji,njk,kl->nil", transform, numpy(probe.factor_covariance), transform
    )
    factor_probe = pd.DataFrame({"c1": context})
    aligned = numpy(probe.factor_mean) @ transform
    for h, truth in enumerate(true_factors(context).T):
        factor_probe[f"true_phi{h + 1}"] = truth
        factor_probe[f"aligned_phi{h + 1}"] = aligned[:, h]
        factor_probe[f"aligned_phi{h + 1}_sd"] = np.sqrt(
            np.maximum(covariance[:, h, h], 0.0)
        )

    # Smooth loading maps on a 61 x 61 characteristic grid.
    axis = np.linspace(0.0, 1.0, 61)
    points = np.column_stack(
        [v.ravel() for v in np.meshgrid(axis, axis, indexing="ij")]
    )
    aligned = (
        numpy(latent(model, points, np.zeros(len(points))).beta_mean)
        @ loading_transform
    )
    aligned = aligned * scale  # loadings carry the return scale
    loading_probe = pd.DataFrame(points, columns=["x1", "x2"])
    for h, truth in enumerate(true_loadings(points).T):
        loading_probe[f"true_beta{h + 1}"] = truth
        loading_probe[f"aligned_beta{h + 1}"] = aligned[:, h]

    # Innovation covariances transformed by the same alignment (Table 1).
    alignment = loading_transform * scale
    loading = (
        alignment.T @ np.diag(numpy(model.likelihood.loading_variances)) @ alignment
    )
    factor = (
        transform.T @ np.diag(numpy(model.factor_kernel_fractions[:, 2])) @ transform
    )
    variance = np.maximum(predictions.variance.to_numpy(float), 1.0e-12)
    true_variance = predictions.true_variance.to_numpy(float)
    summary = {
        "seed": args.seed,
        "assets": design["assets"],
        "months": design["months"],
        "observations": len(frame),
        "epochs": config["training"]["epochs"],
        "mean_r2": r2(
            predictions["mean"].to_numpy(float), predictions.true_mean.to_numpy(float)
        ),
        "variance_correlation": float(pearsonr(variance, true_variance).statistic),
        "variance_ratio": float(np.mean(variance) / np.mean(true_variance)),
        "loading_innovation_average": float(np.diag(loading).mean()),
        "factor_innovation_average": float(np.diag(factor).mean()),
        "observation_variance_estimate": float(model.likelihood.observation_variance)
        * scale**2,
    }
    for h in (1, 2):
        summary[f"factor_probe_{h}_r2"] = r2(
            factor_probe[f"aligned_phi{h}"].to_numpy(),
            factor_probe[f"true_phi{h}"].to_numpy(),
        )
        summary[f"loading_probe_{h}_r2"] = r2(
            loading_probe[f"aligned_beta{h}"].to_numpy(),
            loading_probe[f"true_beta{h}"].to_numpy(),
        )

    args.output.mkdir(parents=True, exist_ok=True)
    write_config(config, args.output / "config.yaml")
    pd.DataFrame([summary]).to_csv(args.output / "summary.csv", index=False)
    factor_dates.to_csv(args.output / "factor_dates.csv", index=False)
    factor_probe.to_csv(args.output / "factor_probe.csv", index=False)
    loading_probe.to_csv(args.output / "loading_probe.csv", index=False)
    print(pd.Series(summary).to_string(), flush=True)


if __name__ == "__main__":
    main()
