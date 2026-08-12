"""Fiscal sensitivity to calibrated LFS-based within-SIC worker selection."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd

from uk_trade_shock_study.runner import (
    _baseline_and_persons,
    json_value,
    run_monte_carlo_prepared,
)
from uk_trade_shock_study.shocks import PRESETS


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FRS = ROOT / "data/frs_2024_25.h5"
DEFAULT_RISK = ROOT / "results/lfs_qrf_benchmark.csv.gz"
DEFAULT_OUT = ROOT / "results/lfs_selection_sensitivity.json"
RISK_COLUMNS = {
    "cells": "job_exit_probability",
    "income_terciles": "job_exit_probability_banded",
    "qrf": "qrf_job_exit_calibrated",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frs", type=Path, default=DEFAULT_FRS)
    parser.add_argument("--risk", type=Path, default=DEFAULT_RISK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--period", type=int, default=2026)
    parser.add_argument("--n-draws", type=int, default=100)
    args = parser.parse_args()

    dataset, baseline, persons = _baseline_and_persons(
        args.frs, None, args.period
    )
    risk = pd.read_csv(args.risk)[["person_id", *RISK_COLUMNS.values()]]
    if risk["person_id"].duplicated().any():
        raise ValueError("LFS receiver benchmark has duplicate person IDs.")
    persons = persons.merge(risk, on="person_id", how="left", validate="one_to_one")
    exposed = (
        persons["employment_income"].gt(0)
        & persons["sic_division"].between(10, 33)
    )
    coverage = persons.loc[exposed, list(RISK_COLUMNS.values())].notna().mean()
    if coverage.min() < 0.99:
        raise RuntimeError(
            "LFS-risk join coverage is below 99% for manufacturing employees: "
            f"{coverage.to_dict()}"
        )

    results = {}
    for label, column in RISK_COLUMNS.items():
        scenario = replace(
            PRESETS["full_tariff_displacement"],
            name=f"full_tariff_displacement_{label}",
            selection_risk_column=column,
        )
        result = run_monte_carlo_prepared(
            dataset,
            baseline,
            persons,
            scenario,
            period=args.period,
            n_draws=args.n_draws,
        )
        results[label] = json_value(asdict(result))
        print(
            f"{label}: gross £{result.draws[0]['gross_earnings_loss']/1e6:.0f}m "
            f"(first draw), cushioning {result.cushioning_rate_mean*100:.1f}%"
        )

    payload = {
        "description": (
            "Within-SIC displacement risks vary with calibrated longitudinal "
            "LFS imputations; each SIC's expected weighted wage-bill loss is "
            "held at the central sector shock."
        ),
        "n_draws": args.n_draws,
        "models": results,
        "caveat": (
            "Baseline transition risk is predictive and is not identified as "
            "tariff-specific worker selection."
        ),
    }
    args.output.write_text(
        json.dumps(json_value(payload), indent=2, allow_nan=False) + "\n"
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
