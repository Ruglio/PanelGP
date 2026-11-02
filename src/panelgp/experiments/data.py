"""The S&P 500 panel of Section 6.1: inputs, timing, and preprocessing.

Input month t pairs characteristics observed at month-end t and the context
of month t with the excess return realized in month t+1 (JKP
``ret_exc_lead1m``). Rows are historical S&P 500 members at t. The
characteristics are selected from the data: every column of the JKP extract
other than the identifiers is a candidate (the 153 JKP characteristics), the
model uses those observed often enough in the screening period, and
stock-months need enough of the selected ones.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ..model.data import PanelSchema

ASSET, DATE, TARGET = "permno", "eom", "ret_exc_lead1m"
IDENTIFIERS = (ASSET, DATE, "size_grp", TARGET)


def in_membership(frame: pd.DataFrame, spells: pd.DataFrame) -> pd.DataFrame:
    """Rows whose month-end lies in one of the asset's index-membership spells.

    ``spells`` has one row per spell (``permno, mbrstartdt, mbrenddt``), so an
    asset that left and later rejoined the index has several rows. Matching on
    the row's own date is free of look-ahead and survivorship bias.
    """
    intervals = pd.DataFrame(
        {
            ASSET: spells["permno"].astype(np.int64),
            "start": pd.to_datetime(spells["mbrstartdt"]),
            "end": pd.to_datetime(spells["mbrenddt"]),
        }
    )
    dated = pd.DataFrame(
        {ASSET: frame[ASSET].astype(np.int64), DATE: frame[DATE], "row": frame.index}
    ).merge(intervals, on=ASSET)
    member = dated.loc[dated[DATE].between(dated.start, dated.end), "row"].unique()
    return frame.loc[frame.index.isin(member)]


def read_context(
    path: str, columns: list[str], lags: Mapping[str, int]
) -> pd.DataFrame:
    """Monthly context from Goyal's predictor file, with publication lags.

    A lag of k months replaces the value of month t by that of calendar month
    t-k; a missing source month stays missing rather than borrowing a neighbor.
    """
    macro = pd.read_csv(path)
    macro["year"], macro["month"] = macro.yyyymm // 100, macro.yyyymm % 100
    macro = macro[["year", "month", *columns]].drop_duplicates(
        ["year", "month"], keep="last"
    )
    dates = pd.to_datetime(macro[["year", "month"]].assign(day=1)).dt.to_period("M")
    for column, lag in lags.items():
        values = pd.Series(macro[column].to_numpy(), index=dates)
        macro[column] = values.reindex(dates - int(lag)).to_numpy()
    return macro


def rank_characteristics(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Cross-sectional ranks mapped to [-0.5, 0.5] within each month; missing -> 0."""
    grouped = frame.groupby(["year", "month"], sort=False)[features]
    ranks = grouped.rank(method="average")
    counts = grouped.transform("count")
    values = (ranks - 1.0) / (counts - 1.0).replace(0.0, np.nan) - 0.5
    frame[features] = values.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return frame


def screen_characteristics(
    path: str,
    years: tuple[int, int],
    screening_years: tuple[int, int],
    row_share: float,
    characteristic_share: float,
    spells: pd.DataFrame | None,
) -> tuple[pd.DataFrame, list[str]]:
    """Select the characteristics, then the stock-months, of the S&P 500 panel.

    Every characteristic of the extract is a candidate. The selected ones are
    observed for at least ``characteristic_share`` of the index-member
    stock-months of the screening years. Member stock-months then need at least
    ``row_share`` of the selected characteristics. Returns these rows (all
    rows if ``spells`` is None) and the selected characteristics, in the order
    of the extract.
    """
    source = pq.ParquetFile(path)
    candidates = [c for c in source.schema_arrow.names if c not in IDENTIFIERS]
    parts = []
    for group in range(source.num_row_groups):  # bounded memory
        frame = source.read_row_group(group, use_threads=False).to_pandas()
        frame = frame.loc[frame[DATE].dt.year.between(*years)]
        if spells is not None:
            frame = in_membership(frame.loc[frame[ASSET].isin(spells.permno)], spells)
        parts.append(frame[[ASSET, DATE, TARGET, *candidates]])
    frame = pd.concat(parts, ignore_index=True)
    early = frame.loc[frame[DATE].dt.year.between(*screening_years), candidates]
    available = early.notna().sum()
    selected = [
        c for c in candidates if available[c] >= characteristic_share * len(early)
    ]
    required = math.ceil(row_share * len(selected))
    keep = frame[selected].notna().sum(axis=1) >= required
    print(
        f"Screens: {len(selected)} of {len(candidates)} characteristics observed "
        f"for at least {characteristic_share:.0%} of the {len(early):,} member "
        f"stock-months of {screening_years[0]}-{screening_years[1]}; "
        f"{int(keep.sum()):,} of {len(frame):,} member stock-months have at least "
        f"{required} of them.",
        flush=True,
    )
    return frame.loc[keep, [ASSET, DATE, TARGET, *selected]], selected


def load_panel(config: Mapping[str, Any]) -> tuple[pd.DataFrame, PanelSchema]:
    """Load, screen, and rank the empirical panel described by ``config['data']``."""
    data = config["data"]
    context = [str(name) for name in data["context_columns"]]
    spells = None
    if data.get("membership_file"):
        spells = pd.read_csv(data["membership_file"]).dropna()
    frame, features = screen_characteristics(
        data["panel_file"],
        (int(data["start_year"]), int(data["end_year"])),
        tuple(int(year) for year in data["screening_years"]),
        float(data["minimum_row_share"]),
        float(data["minimum_characteristic_share"]),
        spells,
    )
    frame = frame.assign(
        year=frame[DATE].dt.year.astype(int), month=frame[DATE].dt.month.astype(int)
    )
    macro = read_context(data["macro_file"], context, data.get("context_lags", {}))
    frame = frame.merge(macro, on=["year", "month"], how="left", validate="many_to_one")
    if frame[context].isna().any(axis=None):
        missing = frame.loc[frame[context].isna().any(axis=1), ["year", "month"]]
        raise ValueError(f"Context is missing for months:\n{missing.drop_duplicates()}")
    frame = rank_characteristics(frame.copy(), features)
    frame = (
        frame.dropna(subset=[TARGET])
        .sort_values(["year", "month", ASSET], kind="stable")
        .reset_index(drop=True)
    )
    return frame, PanelSchema(tuple(features), tuple(context), TARGET, ASSET)


def standardize_context(
    train: pd.DataFrame, frames: list[pd.DataFrame], columns: list[str], log: list[str]
) -> list[pd.DataFrame]:
    """Log-transform ``log`` columns, then z-score with training-month moments."""

    def transform(frame: pd.DataFrame) -> pd.DataFrame:
        values = frame[columns].copy()
        values[log] = np.log(values[log])
        return values

    monthly = transform(train.drop_duplicates(["year", "month"]))
    mean, scale = monthly.mean(axis=0), monthly.std(axis=0, ddof=0).replace(0.0, 1.0)
    result = []
    for frame in frames:
        frame = frame.copy()
        frame[columns] = ((transform(frame) - mean) / scale).to_numpy()
        result.append(frame)
    return result


def month_number(frame: pd.DataFrame) -> pd.Series:
    return frame["year"].astype(int) * 12 + frame["month"].astype(int) - 1
