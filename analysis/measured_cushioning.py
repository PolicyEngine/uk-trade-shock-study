"""Cushioning rates for the MEASURED family (seed 0), mirroring
results/cushioning_seed0.json for the calibrated families.

Writes results/measured_cushioning_seed0.json with gross earnings loss,
net disposable loss and cushioning rate for measured_displacement (seed 0)
and the deterministic measured_wage_cut.

Usage: .venv/bin/python analysis/measured_cushioning.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np  # noqa: F401 (kept for parity with runner helpers)

from uk_trade_shock_study.runner import _baseline_and_persons
from uk_trade_shock_study.shocks import TradeShockScenario, apply_shocks, build_shocked_simulation

PERIOD = 2026
DATASET = Path("data/frs_2024_25.h5")
OUT = Path("results/measured_cushioning_seed0.json")


def _hni_total(sim) -> float:
    hh_w = sim.calculate("household_weight", period=PERIOD, map_to="household").values
    hni = sim.calculate("hbai_household_net_income", period=PERIOD, map_to="household").values
    return float((hni * hh_w).sum())


def main() -> None:
    dataset, baseline, persons = _baseline_and_persons(DATASET, None, PERIOD)
    base_total = _hni_total(baseline)
    weight = persons["weight"].to_numpy()
    out = {}
    for margin in ("displacement", "wage_cut"):
        scenario = TradeShockScenario(f"measured_{margin}", "measured", margin)
        shocked_table = apply_shocks(persons, scenario, seed=0)
        shocked = build_shocked_simulation(dataset, baseline, shocked_table, PERIOD)
        gross = float(
            (
                (persons["employment_income"].to_numpy() - shocked_table["employment_income"].to_numpy())
                * weight
            ).sum()
        )
        net = base_total - _hni_total(shocked)
        out[scenario.name] = {
            "gross_earnings_loss": gross,
            "net_disposable_loss": net,
            "cushioning_rate": 1.0 - net / gross,
        }
        print(scenario.name, out[scenario.name])
    OUT.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
