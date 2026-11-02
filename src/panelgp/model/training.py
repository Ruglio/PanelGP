"""ELBO training with Adam and a fractional multistep learning-rate schedule."""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch

from .data import PanelSchema, build_time_grid, to_tensors
from .panel_gp import ModelSettings, PanelGP, PredictiveMoments

# Optimization settings of Appendix C.4; presets override the number of epochs.
TRAINING_DEFAULTS = {
    "epochs": 500,
    "batch_size": 2048,
    "elbo_samples": 64,
    "factor_sampling": "antithetic",
    "variational_lr": 0.01,
    "hyperparameter_lr": 0.005,
    "gradient_clip": 1.0,
    "milestone_fractions": [0.3, 0.5, 0.6, 0.7, 0.8, 0.9],
    "scheduler_factor": 0.5,
    "log_interval": 10,
}


@dataclass(frozen=True)
class FitResult:
    model: PanelGP
    history: pd.DataFrame
    time_grid: pd.DataFrame


def resolve_device(value: str = "auto") -> torch.device:
    if str(value).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fit_model(
    train: pd.DataFrame, schema: PanelSchema, config: Mapping[str, Any]
) -> FitResult:
    """Maximize the ELBO on ``train`` (one row per asset and date).

    ``config`` has ``model`` (see :class:`ModelSettings`), ``training`` (keys
    of ``TRAINING_DEFAULTS``) and ``runtime`` (``seed``, ``device``) sections.
    Training runs in float64.
    """
    runtime = config.get("runtime", {})
    training = {**TRAINING_DEFAULTS, **config.get("training", {})}
    unknown = set(training) - TRAINING_DEFAULTS.keys()
    if unknown:
        raise ValueError(f"Unsupported training settings: {sorted(unknown)}")
    seed = int(runtime.get("seed", 1))
    seed_everything(seed)
    device, dtype = resolve_device(runtime.get("device", "auto")), torch.float64

    grid, index_map = build_time_grid(train, schema)
    indexed = train.merge(
        index_map, on=["year", "month"], how="left", validate="many_to_one"
    )
    data = to_tensors(
        indexed, schema, device=device, dtype=dtype, time_index=indexed["time_index"]
    )
    time_features = torch.as_tensor(
        np.ascontiguousarray(grid[list(schema.context_columns)], dtype=np.float64),
        device=device,
        dtype=dtype,
    )
    model = PanelGP(
        input_dim=len(schema.feature_columns),
        time_features=time_features,
        settings=ModelSettings.from_config(config),
        feature_names=schema.feature_columns,
    ).to(device=device, dtype=dtype)
    model.fit_target_scaler(data.target)
    generator = torch.Generator().manual_seed(seed + 1701)
    model.initialize_inducing(data.x)

    variational, hyperparameters = model.parameter_groups()
    optimizer = torch.optim.Adam(
        [
            {"params": variational, "lr": float(training["variational_lr"])},
            {"params": hyperparameters, "lr": float(training["hyperparameter_lr"])},
        ]
    )
    epochs = int(training["epochs"])
    fractions = training["milestone_fractions"]
    milestones = sorted({max(1, min(epochs - 1, round(epochs * f))) for f in fractions})
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=float(training["scheduler_factor"])
    )
    antithetic = training["factor_sampling"] == "antithetic"
    num_rows, batch_size = data.x.shape[0], int(training["batch_size"])
    rows = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        permutation = torch.randperm(num_rows, generator=generator).to(device)
        for start in range(0, num_rows, batch_size):
            batch = permutation[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            likelihood = model.expected_log_likelihood(
                x=data.x[batch],
                context=data.context[batch],
                time_index=data.time_index[batch],
                target=data.target[batch],
                samples=int(training["elbo_samples"]),
                antithetic=antithetic,
            )
            loss = -likelihood / batch.numel() + model.kl_divergence() / num_rows
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Non-finite ELBO objective at epoch {epoch}.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training["gradient_clip"])
            )
            optimizer.step()
            total_loss += float(loss.detach().item()) * batch.numel()
        scheduler.step()
        rows.append(
            {
                "epoch": epoch,
                "loss_per_observation": total_loss / num_rows,
                "kl": float(model.kl_divergence().detach().item()),
                "variational_lr": optimizer.param_groups[0]["lr"],
                "model_lr": optimizer.param_groups[1]["lr"],
            }
        )
        if epoch in (1, epochs) or epoch % int(training["log_interval"]) == 0:
            print(
                f"epoch={epoch:04d}/{epochs} loss={total_loss / num_rows:.6f}",
                flush=True,
            )
    return FitResult(model=model, history=pd.DataFrame(rows), time_grid=grid)


@torch.no_grad()
def predict_rows(
    model: PanelGP, frame: pd.DataFrame, schema: PanelSchema, *, batch_size: int = 4096
) -> PredictiveMoments:
    """Row-wise predictive means and variances at new dates."""
    device, dtype = model.time_features.device, model.time_features.dtype
    model.eval()
    parts = []
    for start in range(0, len(frame), batch_size):
        data = to_tensors(
            frame.iloc[start : start + batch_size], schema, device=device, dtype=dtype
        )
        parts.append(model.predictive_moments(data.x, data.context, data.time_index))
    return PredictiveMoments(
        mean=torch.cat([p.mean for p in parts]).cpu(),
        variance=torch.cat([p.variance for p in parts]).cpu(),
    )
