"""Build LFS-estimated labour-transition parameters for PolicyEngine FRS.

The LFS file is licensed UKDS microdata and must be downloaded by a registered
user. It is read locally and is never copied into the repository. Public BRES
employment totals constrain sector mass. The output contains imputed
parameters for FRS records, not linked or observed ASHE outcomes.

Usage:
    .venv/bin/python analysis/impute_lfs_to_frs.py \
      --lfs-tab /path/to/lgwt25_5q_*.tab
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from policyengine_uk import Microsimulation

from uk_trade_shock_study.exposure import simulation_sic_division
from uk_trade_shock_study.lfs_imputation import (
    align_lfs_to_bres,
    bres_sector_targets,
    calibrate_receiver_transitions,
    impute_frs_transition_parameters,
    prepare_lfs_transitions,
    transition_cells,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BRES = (
    ROOT / "data" / "public" / "bres_manufacturing_employment_2015_2024.csv"
)
DEFAULT_FRS = ROOT / "data" / "frs_2024_25.h5"
DEFAULT_OUT = ROOT / "results" / "lfs_imputed_frs_transition_parameters.csv.gz"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lfs-tab", type=Path, required=True)
    parser.add_argument("--bres", type=Path, default=DEFAULT_BRES)
    parser.add_argument("--frs", type=Path, default=DEFAULT_FRS)
    parser.add_argument("--period", type=int, default=2025)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--shrinkage-weight", type=float, default=50)
    args = parser.parse_args()

    if not args.lfs_tab.exists():
        raise SystemExit(f"LFS tab file not found: {args.lfs_tab}")
    if not args.bres.exists():
        raise SystemExit("BRES file missing; first run `make public-bres`.")

    raw_lfs = pd.read_csv(args.lfs_tab, sep="\t", low_memory=False)
    lfs = prepare_lfs_transitions(raw_lfs)
    targets = bres_sector_targets(pd.read_csv(args.bres))
    calibrated_lfs, calibration = align_lfs_to_bres(lfs, targets)
    cells = transition_cells(
        calibrated_lfs, shrinkage_weight=args.shrinkage_weight
    )

    simulation = Microsimulation(dataset=str(args.frs))
    income = simulation.calculate(
        "employment_income", args.period, map_to="person"
    )
    frs = pd.DataFrame(
        {
            "person_id": simulation.calculate(
                "person_id", args.period, map_to="person"
            ).values,
            "employment_income": income.values,
            "sic_division": simulation_sic_division(simulation, args.period),
            "gender": simulation.calculate(
                "gender", args.period, map_to="person"
            ).values,
            "age": simulation.calculate(
                "age", args.period, map_to="person"
            ).values,
            "weight": simulation.calculate(
                "household_weight", args.period, map_to="person"
            ).values,
        }
    )
    frs["gender"] = frs["gender"].map({"MALE": 1, "FEMALE": 2})
    frs = frs[
        frs["employment_income"].gt(0)
        & frs["sic_division"].between(10, 33)
    ].copy()
    imputed, coverage = impute_frs_transition_parameters(frs, cells)
    imputed, level_calibration = calibrate_receiver_transitions(
        imputed, calibrated_lfs
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    imputed.to_csv(args.output, index=False)
    diagnostics_path = args.output.with_suffix("").with_suffix(
        ".diagnostics.json"
    )
    diagnostics_path.write_text(
        json.dumps(
            {
                "method": (
                    "Five-quarter LFS donor cells (SIC division × sex × age "
                    "band), credibility-shrunk to sector means and sector-mass "
                    "aligned to open BRES 2024 employee totals; receiver "
                    "levels calibrated to direct weighted LFS targets, with "
                    "an income-tercile sensitivity estimate"
                ),
                "lfs_source": str(args.lfs_tab),
                "lfs_employed_wave1_records": len(lfs),
                "lfs_calibrated_manufacturing_records": len(calibrated_lfs),
                "transition_cell_count": len(cells),
                "frs_employee_records": len(frs),
                "coverage": coverage.to_dict(orient="records"),
                "level_calibration": level_calibration,
                "unsupported_bres_sectors": calibration.index[
                    ~calibration["supported"]
                ].astype(int).tolist(),
                "caveat": (
                    "Imputed parameters are not linked outcomes and do not "
                    "identify a causal tariff effect."
                ),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Wrote {len(imputed):,} FRS records to {args.output}")
    print(coverage.to_string(index=False))


if __name__ == "__main__":
    main()
