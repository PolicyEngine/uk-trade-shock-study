"""Leave each exposed SIC division out of the unit 12-month primary contrast."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from uk_trade_shock_study.runner import (
    _baseline_and_persons,
    run_monte_carlo_prepared,
)
from uk_trade_shock_study.shocks import TradeShockScenario, _person_shock


def clean(value):
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/frs_2024_25.h5"))
    parser.add_argument("--output", type=Path, default=Path("results/leave_one_sector_out.json"))
    parser.add_argument("--period", type=int, default=2026)
    parser.add_argument("--n-draws", type=int, default=20)
    args = parser.parse_args()

    dataset, baseline, persons = _baseline_and_persons(
        args.dataset, None, args.period
    )
    reference = TradeShockScenario(
        "reference", "full_tariff", "wage_cut"
    )
    shock = _person_shock(persons, reference)
    earnings_weight = (
        persons["employment_income"].to_numpy(float)
        * persons["weight"].to_numpy(float)
    )
    division = persons["sic_division"].to_numpy(float)
    contributions = {
        int(value): float((shock[division == value] * earnings_weight[division == value]).sum())
        for value in np.unique(division[np.isfinite(division)])
        if float((shock[division == value] * earnings_weight[division == value]).sum()) > 0
    }
    output = {
        "n_draws": args.n_draws,
        "estimand": "unit 12-month duration-equivalent stress",
        "excluded_divisions": {},
    }
    for value, contribution in sorted(
        contributions.items(), key=lambda item: item[1], reverse=True
    ):
        items = {}
        for margin in ("displacement", "wage_cut"):
            scenario = TradeShockScenario(
                f"exclude_sic_{value}_{margin}",
                "full_tariff",
                margin,
                selection_method="balanced",
                excluded_divisions=(value,),
            )
            result = run_monte_carlo_prepared(
                dataset,
                baseline,
                persons,
                scenario,
                args.period,
                args.n_draws,
            )
            items[margin] = clean(asdict(result))
        output["excluded_divisions"][str(value)] = {
            "excluded_expected_gross_loss_million": contribution / 1e6,
            "displacement_exchequer_mean_million": items["displacement"][
                "exchequer_cost_mean"
            ]
            / 1e6,
            "wage_cut_exchequer_mean_million": items["wage_cut"][
                "exchequer_cost_mean"
            ]
            / 1e6,
            "displacement_cushioning_percent": items["displacement"][
                "cushioning_rate_mean"
            ]
            * 100,
            "wage_cut_cushioning_percent": items["wage_cut"][
                "cushioning_rate_mean"
            ]
            * 100,
            "poverty_change_displacement_pp": items["displacement"][
                "poverty_rate_change_bhc_mean"
            ]
            * 100,
        }
        print(
            f"excluded SIC {value}: £{contribution / 1e6:.0f}m target removed",
            flush=True,
        )
    args.output.write_text(json.dumps(clean(output), indent=2) + "\n")


if __name__ == "__main__":
    main()
