# PanelGP: A Probabilistic Factor Model for Asset-Return Panels

Replication package of Ruglioni, A., and Filipovic, D. (2026), *PanelGP: A
Probabilistic Factor Model for Asset-Return Panels*, working paper, EPFL.
The model is the Python package `panelgp` in `src/`.

## Overview

This package reproduces every table, figure and number of the paper and its
online supplement. It can be used at three levels:

1. **Tables and figures from the included results** (no data, CPU, minutes):
   `results/paper/data/` holds the numerical record of all fits; two scripts
   turn it into the paper's tables and figures.
2. **Simulation, Section 5** (no data): five replications from scratch.
3. **Empirical application, Section 6** (WRDS access, GPU): 310 PanelGP and
   76 IPCA fits, then the collection step that produces `results/paper/data/`.

## Data availability and provenance

| Data | Source | Access | File in `data/` |
|---|---|---|---|
| Stock characteristics and returns | Jensen, Kelly and Pedersen (2023), Global Factor Data, WRDS table `contrib.global_factor` | WRDS subscription | `jkp_usa.parquet` |
| S&P 500 membership spells | CRSP, WRDS table `crsp_a_indexes.dsp500list_v2` | WRDS with a CRSP subscription | `sp500_constituents.csv` |
| Monthly predictors, S&P 500 return and risk-free rate | Goyal, Welch and Zafirov (2024), https://sites.google.com/view/agoyal145 | Public | `goyal_monthly.csv` |

None of the inputs is included, and the licensed data cannot be
redistributed. `data/data_download.py` downloads all three (Level 3 below).
WRDS asks for two-factor authentication from each new IP address: approve the
Duo push when the script connects, or log in once from the same machine with
`ssh <username>@wrds-cloud.wharton.upenn.edu` beforehand. Levels 1 and 2 do
not use these data.

**Characteristics.** The extract contains the 153 characteristics of Jensen,
Kelly and Pedersen (2023), as listed in JKP's `factor_details.xlsx`, for U.S.
common stocks on the main exchanges, the primary security of each firm and the
CRSP observation where CRSP and Compustat overlap
(`common = exch_main = primary_sec = obs_main = 1`), excluding nano caps
(`size_grp = 'nano'`, below the 1st NYSE size percentile) and rows without a
next-month return, for January 1960 to December 2024. Accounting variables
carry the JKP four-month reporting lag.

**Sample.** `panelgp.experiments.data.load_panel` keeps stock-months whose
month-end lies in an S&P 500 membership spell and selects the characteristics
with the shares set in `configs/empirical.yaml`: it uses the characteristics
observed for at least 70% of these stock-months in 1960–1989 (137 of 153 in
the October 2026 download) and keeps the stock-months with at least 70% of
them observed. It then ranks each characteristic within the month onto
[-0.5, 0.5] (missing values set to zero).

**Timing.** Characteristics and context of input month t forecast the excess
return realized in month t + 1 (`ret_exc_lead1m`). The context is Goyal's
`retx`, `tms`, `dfy`, `log(svar)`, `avgcor` and `infl`; inflation is lagged one
further month for its publication delay (Welch and Goyal, 2008). Goyal's
`Rfree` of month t + 1 is the risk-free rate used for portfolio turnover.

**Revisions.** The paper's estimates use the inputs downloaded in October
2026. JKP and Goyal revise their historical data, so models refitted on later
downloads can differ slightly from the paper.

## Computational requirements

* Python 3.12 (3.10–3.12 supported) with the dependencies pinned in
  `uv.lock`, installed with [uv](https://docs.astral.sh/uv/):
  `uv sync --frozen --extra all`. Main packages: PyTorch 2.9.1, NumPy, pandas,
  SciPy, statsmodels, ipca 0.6.7, matplotlib.
* Level 1: any computer; under five minutes.
* Level 2: about 2.5 hours per replication on eight CPU threads (5,000
  epochs on 4,000 observations); a GPU is faster.
* Level 3: an NVIDIA GPU with float64 support for PanelGP. The paper's fits ran
  on NVIDIA H100 GPUs: about 12 minutes for a five-factor fit and about 65
  GPU-hours for all 310 PanelGP fits. The 76 IPCA fits run on two CPU cores
  in 1 to about 40 minutes each (median 14). Loading the panel needs about
  4 GB of memory; the JKP extract takes about 2 GB of disk and the fitted
  models about 15 GB.
* Randomness is controlled by the seeds in the case table and the configs,
  and runs on the same hardware are reproducible. On other hardware,
  floating-point differences accumulate over training: a CPU refit of
  simulation seed 4 matches the H100 fit's function R² to about three digits,
  while its factor-innovation estimate differs by 4% (0.0556 against 0.0533).
  Such differences can also change the composition of a decile portfolio.

## Description of programs

| File | Purpose |
|---|---|
| `src/panelgp/model/` | The model (Sections 3–4): kernels and variational family (`components.py`), PanelGP and its predictive moments (`panel_gp.py`), ELBO training (`training.py`), simulation design (`synthetic.py`) |
| `src/panelgp/experiments/` | The application (Section 6): data (`data.py`), IPCA (`ipca.py`), scores and portfolios (`evaluation.py`), predictive densities (`density.py`), the case table and runner (`campaign.py`) |
| `configs/empirical.yaml`, `configs/synthetic.yaml` | All settings of the application and the simulation |
| `data/data_download.py` | Downloads and screens the inputs |
| `scripts/run_case.py` | Fits and evaluates one or more of the 386 empirical cases |
| `scripts/run_synthetic.py` | One replication of the simulation |
| `scripts/collect.py` | Collects all fits into `results/paper/data/` |
| `scripts/make_tables.py` | Tables and the numbers quoted in the text |
| `scripts/make_figures.py` | Figures |

The 386 cases (`python scripts/run_case.py --list`): 0–159 PanelGP with
fixed training (four kernel pairs × H = 1–8 × seeds 1–5, case
40·pair + 5·(H − 1) + seed − 1); 160–309 PanelGP with annual re-estimation
(RBF/RBF, H = 5, seeds 1–5 × evaluation years 1995–2024); 310–385 IPCA
without (310–347) and with (348–385) an intercept, fixed training for
H = 1–8 followed by annual re-estimation for 1995–2024 with H = 5.

## Instructions

Run all commands from the package root.

```bash
uv sync --frozen --extra all && source .venv/bin/activate
```

**Level 1.** Tables (LaTeX) and figures (PDF) from the included results:

```bash
python scripts/make_tables.py    # results/paper/tables/, with all pairwise tests
python scripts/make_figures.py   # results/paper/figures/
```

**Level 2.** The simulation (Table 1, Figures 1 and S1), then level 1:

```bash
for s in 1 2 3 4 5; do
  python scripts/run_synthetic.py --seed $s --output results/runs/synthetic/seed$s
done
python scripts/collect.py --part synthetic
```

**Level 3.** The empirical application:

```bash
uv run --no-project --with wrds --with pyarrow --with pyyaml --with openpyxl \
    python data/data_download.py all      # WRDS credentials are requested
python data/data_download.py check
python scripts/run_case.py 0-385 --output results/runs   # or in parallel, by index
python scripts/collect.py --runs results/runs --output results/paper/data
python scripts/make_tables.py && python scripts/make_figures.py
```

Cases are independent and can run in parallel, for example as an array job
with one index per task. `collect.py` needs all 386 cases, the fitted models
and the input data (and, without `--part empirical`, the five simulation
replications).

## Tables, figures and numbers

| Exhibit | Content | Produced by | From |
|---|---|---|---|
| Table 1 | Simulation recovery | `make_tables.py` (`tab_synthetic.tex`) | `synthetic.csv` |
| Table 2 | Kernel pairs and IPCA, H = 5 | `make_tables.py` (`tab_kernels.tex`) | `summary.csv` |
| Table 3 | Fixed and annual re-estimation | `make_tables.py` (`tab_rolling.tex`) | `summary.csv` |
| Table 4 | Trading costs | `make_tables.py` (`tab_trading_costs.tex`) | `trading_costs.csv` |
| Tables S1–S2 | Paired comparisons | `make_tables.py` (`tab_test_statistics.tex`, `tab_rolling_tests.tex`) | `monthly_*.csv` |
| Table S3 | Results by subperiod | `make_tables.py` (`tab_subperiods.tex`) | `monthly_*.csv`, `market_benchmark.csv` |
| Figure 1 | Factor recovery in the simulation | `make_figures.py` (`factor_recovery.pdf`) | `synthetic_factor_*.csv` |
| Figure 2 | ARD lengthscales | `make_figures.py` (`ard_lengthscales.pdf`) | `lengthscales.csv` |
| Figure 3 | Results by number of factors | `make_figures.py` (`hidden_dimension.pdf`) | `summary.csv`, `market_benchmark.csv` |
| Figure 4 | Predicted variance terms | `make_figures.py` (`variance_evidence.pdf`) | `risk_*.csv` |
| Figure S1 | Loading recovery in the simulation | `make_figures.py` (`loading_recovery.pdf`) | `synthetic_loading_probe.csv` |
| Figure S2 | Validation by number of factors | `make_figures.py` (`validation_dimension.pdf`) | `summary.csv` |
| Figure S3 | Log-score gain of the bilinear density | `make_figures.py` (`density_gain.pdf`) | `density_gain.csv` |
| Numbers in the text | | `make_tables.py` (`numbers.tex`) | as above, `r2_split.csv`, `risk_rank_correlations.csv` |

Files in the last column are in `results/paper/data/`.

## References

Goyal, A., Welch, I., and Zafirov, A. (2024). A comprehensive 2022 look at the
empirical performance of equity premium prediction. *Review of Financial
Studies* 37, 3490–3557.

Jensen, T. I., Kelly, B., and Pedersen, L. H. (2023). Is there a replication
crisis in finance? *Journal of Finance* 78, 2465–2518.

Welch, I., and Goyal, A. (2008). A comprehensive look at the empirical
performance of equity premium prediction. *Review of Financial Studies* 21,
1455–1508.
