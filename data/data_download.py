"""Download and prepare the three inputs of the empirical application.

    python data/data_download.py goyal        # public, no account needed
    python data/data_download.py membership   # WRDS (CRSP subscription)
    python data/data_download.py jkp          # WRDS (Global Factor Data)
    python data/data_download.py check        # completeness of the JKP extract

Run from the repository root; files are written to ``data/``. The script
needs pandas, pyarrow, PyYAML, openpyxl and the WRDS client. ``wrds`` pins its
own pandas version, so use a separate environment, for example

    uv run --no-project --with wrds --with pyarrow --with pyyaml --with openpyxl \
        python data/data_download.py all

The first WRDS connection asks for the username and password and offers to
store them in ``~/.pgpass``. WRDS also requires two-factor authentication for
each new IP address: approve the Duo push sent when the script connects, or
first log in once from the same machine with
``ssh <username>@wrds-cloud.wharton.upenn.edu`` (valid for 30 days). Without
it, WRDS closes the connection ("SSL connection has been closed unexpectedly");
avoid repeated failed attempts, which can lock the account.

Outputs
-------
goyal_monthly.csv      Monthly sheet of Amit Goyal's predictor data: the context
                       variables, the S&P 500 return and the risk-free rate.
sp500_constituents.csv CRSP S&P 500 membership spells (permno, start, end).
jkp_usa.parquet        The 153 Jensen-Kelly-Pedersen characteristics and the
                       next-month excess return of U.S. common stocks, with
                       the standard screens and without nano caps. The
                       characteristic screens are applied when the panel is
                       loaded (panelgp.experiments.data).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent
ROOT = DATA.parent
IDENTIFIERS = ("permno", "eom", "size_grp", "ret_exc_lead1m")

# Goyal's "all data up to 2025" workbook (combined Welch-Goyal 2008 and
# Goyal-Welch-Zafirov 2024 predictors), https://sites.google.com/view/agoyal145.
GOYAL_URL = "https://docs.google.com/spreadsheets/d/17mw_IpaiLFDrGnrPRQ2o1ugV5nJsZuD1/export?format=xlsx"

# Standard JKP screens (jkpfactors.com/jkp-wrds-guide): common stocks, main
# exchanges, the primary security of each firm, and the CRSP observation when
# CRSP and Compustat overlap. JKP size groups follow NYSE market-capitalization
# breakpoints; nano caps (below the 1st percentile) are excluded. The target
# ret_exc_lead1m is the excess return of the following month.
JKP_QUERY = """
    SELECT permno, eom, size_grp, ret_exc_lead1m, {characteristics}
    FROM contrib.global_factor
    WHERE excntry = 'USA'
      AND common = 1 AND exch_main = 1 AND primary_sec = 1 AND obs_main = 1
      AND size_grp <> 'nano'
      AND permno IS NOT NULL AND ret_exc_lead1m IS NOT NULL
      AND eom BETWEEN '{year}-01-01' AND '{year}-12-31'
    ORDER BY eom, permno
"""
# The 153 characteristics of Jensen, Kelly and Pedersen (2023), as listed by JKP.
FACTOR_DETAILS = (
    "https://github.com/bkelly-lab/jkp-data/raw/main/src/jkp/data/resources/"
    "factor_details.xlsx"
)

MEMBERSHIP_QUERY = """
    SELECT permno, mbrstartdt, mbrenddt
    FROM crsp_a_indexes.dsp500list_v2
    ORDER BY permno, mbrstartdt
"""


def config() -> dict:
    import yaml

    return yaml.safe_load(
        (ROOT / "configs/empirical.yaml").read_text(encoding="utf-8")
    )["data"]


def goyal() -> None:
    monthly = pd.read_excel(GOYAL_URL, sheet_name="Monthly")
    monthly.to_csv(DATA / "goyal_monthly.csv", index=False)
    print(f"goyal_monthly.csv: {monthly.yyyymm.min()}-{monthly.yyyymm.max()}")


def membership() -> None:
    import wrds

    with wrds.Connection() as connection:
        spells = connection.raw_sql(MEMBERSHIP_QUERY)
    spells.to_csv(DATA / "sp500_constituents.csv", index=False)
    print(f"sp500_constituents.csv: {len(spells)} membership spells")


def jkp_characteristics() -> list[str]:
    details = pd.read_excel(FACTOR_DETAILS)
    return details["abr_jkp"].dropna().astype(str).str.strip().tolist()


def jkp() -> None:
    """Download one year per query: the WRDS client holds a whole result in memory."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    import wrds

    data, characteristics = config(), jkp_characteristics()
    numeric = ["permno", "ret_exc_lead1m", *characteristics]
    target = DATA / "jkp_usa.parquet"
    writer, rows = None, 0
    with wrds.Connection() as connection:
        for year in range(int(data["start_year"]), int(data["end_year"]) + 1):
            query = JKP_QUERY.format(
                characteristics=", ".join(characteristics), year=year
            )
            frame = connection.raw_sql(query, date_cols=["eom"], chunksize=None)
            frame[numeric] = frame[numeric].astype("float64")  # also all-missing ones
            frame["size_grp"] = frame["size_grp"].astype(str)
            frame["eom"] = frame["eom"].astype("datetime64[ms]")
            table = pa.Table.from_pandas(frame, preserve_index=False)
            writer = writer or pq.ParquetWriter(target, table.schema)
            writer.write_table(table.cast(writer.schema))
            rows += len(frame)
            print(f"{year}: {len(frame):,} stock-months", flush=True)
    writer.close()
    print(
        f"jkp_usa.parquet: {rows:,} stock-months, {len(characteristics)} characteristics"
    )
    check()


def check() -> None:
    """Verify the extract: all 153 characteristics, no nano caps, every target present."""
    import pyarrow.parquet as pq

    data = config()
    source = pq.ParquetFile(ROOT / data["panel_file"])
    columns = source.schema_arrow.names
    characteristics = [c for c in columns if c not in IDENTIFIERS]
    if len(characteristics) != 153 or any(c not in columns for c in IDENTIFIERS):
        raise ValueError(f"Expected the identifiers and 153 characteristics: {columns}")
    rows, years = 0, set()
    for group in range(source.num_row_groups):  # one at a time: bounded memory
        frame = source.read_row_group(
            group, columns=list(IDENTIFIERS), use_threads=False
        ).to_pandas()
        if frame.size_grp.eq("nano").any() or frame.ret_exc_lead1m.isna().any():
            raise ValueError("Nano caps or missing next-month returns in the extract.")
        rows += len(frame)
        years.update(frame.eom.dt.year.unique())
    expected = set(range(int(data["start_year"]), int(data["end_year"]) + 1))
    if not expected <= years:
        raise ValueError(f"Missing years: {sorted(expected - years)}")
    print(
        f"Extract complete: {rows:,} stock-months, {min(years)}-{max(years)}, "
        f"153 characteristics."
    )


if __name__ == "__main__":
    steps = {"goyal": goyal, "membership": membership, "jkp": jkp, "check": check}
    requested = sys.argv[1:] or ["goyal", "membership", "jkp"]
    if "all" in requested:
        requested = ["goyal", "membership", "jkp"]
    for name in requested:
        steps[name]()
