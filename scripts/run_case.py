"""Estimate and evaluate cases of the empirical campaign (Section 6).

    python scripts/run_case.py --list                 # the 386 cases
    python scripts/run_case.py 140                    # RBF/RBF, H=5, seed 1, fixed
    python scripts/run_case.py 160-309 --output results/runs

Each case writes ``<output>/case_NNN/``. Cases are independent and can run in
parallel; a PanelGP case needs a GPU for practical run times (about twelve
minutes on an H100 for H = 5).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from panelgp.experiments.campaign import cases, run_case
from panelgp.experiments.config import load_config


def indices(values: list[str]) -> list[int]:
    result = []
    for value in values:
        first, _, last = value.partition("-")
        result.extend(range(int(first), int(last or first) + 1))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "cases", nargs="*", help="case indices or ranges such as 160-189"
    )
    parser.add_argument("--config", default="configs/empirical.yaml")
    parser.add_argument("--output", type=Path, default=Path("results/runs"))
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--list", action="store_true", help="print the case table")
    parser.add_argument("--skip-complete", action="store_true")
    args = parser.parse_args()
    table = cases()
    if args.list:
        for case in table:
            if case.method == "panelgp":
                kind = f"PanelGP {case.loading_kernel}/{case.factor_kernel} seed {case.seed}"
            else:
                kind = "IPCA with intercept" if case.intercept else "IPCA"
            period = f"year {case.year}" if case.year else "1995-2024"
            print(
                f"{case.index:3d}  {case.protocol:7s}  H={case.hidden_dim}  {kind:30s} {period}"
            )
        return
    config = load_config(args.config, args.set)
    for index in indices(args.cases):
        case = table[index]
        destination = args.output / case.name
        if args.skip_complete and (destination / "complete.json").exists():
            continue
        run_case(config, case, destination)


if __name__ == "__main__":
    main()
