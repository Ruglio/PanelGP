"""The contract between a tidy panel DataFrame and the model."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch


@dataclass(frozen=True)
class PanelSchema:
    """Column roles. Frames also need integer ``year`` and ``month`` columns;
    context columns must be constant within each date."""

    feature_columns: tuple[str, ...]
    context_columns: tuple[str, ...]
    target_column: str
    asset_column: str


@dataclass(frozen=True)
class PanelTensors:
    x: torch.Tensor
    context: torch.Tensor
    target: torch.Tensor
    time_index: torch.Tensor


def build_time_grid(
    train: pd.DataFrame, schema: PanelSchema
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sorted training dates with their contexts, and the date-to-index map."""
    context = list(schema.context_columns)
    if (train.groupby(["year", "month"])[context].nunique() > 1).any().any():
        raise ValueError("Context must be constant within every monthly cross section.")
    grid = (
        train[["year", "month", *context]]
        .drop_duplicates(["year", "month"])
        .sort_values(["year", "month"], kind="stable")
        .reset_index(drop=True)
    )
    grid["time_index"] = np.arange(len(grid), dtype=np.int64)
    return grid, grid[["year", "month", "time_index"]]


def to_tensors(
    frame: pd.DataFrame,
    schema: PanelSchema,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
    time_index: Sequence[int] | None = None,
) -> PanelTensors:
    """Model tensors for the rows of ``frame``; time index -1 marks new dates."""

    def array(columns) -> np.ndarray:
        return np.array(frame[columns], dtype=np.float64, order="C", copy=True)

    x, context = (
        array(list(schema.feature_columns)),
        array(list(schema.context_columns)),
    )
    target = array(schema.target_column)
    if not (
        np.isfinite(x).all()
        and np.isfinite(context).all()
        and np.isfinite(target).all()
    ):
        raise ValueError("Model tensors contain non-finite values.")
    if time_index is None:
        index = np.full(len(frame), -1, dtype=np.int64)
    else:
        index = np.array(time_index, dtype=np.int64, order="C", copy=True)
    return PanelTensors(
        torch.as_tensor(x, device=device, dtype=dtype),
        torch.as_tensor(context, device=device, dtype=dtype),
        torch.as_tensor(target, device=device, dtype=dtype),
        torch.as_tensor(index, device=device),
    )
