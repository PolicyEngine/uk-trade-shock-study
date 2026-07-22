"""Seed-0 cushioning accounting and fixed-line absolute poverty, all families.

Regenerates two seed-0 diagnostic artefacts:
  results/cushioning_seed0.json               gross/net loss and 1 - net/gross
  results/absolute_poverty_fixed_line_seed0.json   BHC absolute poverty change

Both are computed at seed 0 under the current package behaviour, which
includes the post-shock Universal Credit take-up re-draw
(shocks.DEFAULT_UC_TAKEUP).

Usage: .venv/bin/python analysis/seed0_cushioning.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from uk_trade_shock_study.runner import _baseline_and_persons
from uk_trade_shock_study.shocks import PRESETS, apply_shocks, build_shocked_simulation

PERIOD = 2026
DATASET = Path("data/frs_2024_25.h5")
RESULTS = Path("results")
FAMILIES = (
    "full_tariff_displacement",
    "full_tariff_wage_cut",
    "full_tariff_inactivity",
    "full_tariff_reallocation",
    "epd_displacement",
    "epd_wage_cut",
    "epd_inactivity",
    "epd_reallocation",
)


def _hni_total(sim) -> float:
    hh_w = sim.calculate("household_weight", period=PERIOD, map_to="household").values
    hni = sim.calculate("hbai_household_net_income", period=PERIOD, map_to="household").values
    return float((hni * hh_w).sum())


def _abs_poverty(sim) -> float:
    """Fixed-2010-11-line absolute BHC poverty rate.

    Not available in every policyengine-uk version (the installed 2.89.x does
    not expose in_absolute_poverty_bhc); returns NaN there, in which case
    results/absolute_poverty_fixed_line_seed0.json is left untouched rather
    than overwritten with NaNs.
    """
    p_w = sim.calculate("person_weight", period=PERIOD, map_to="person").values
    try:
        values = sim.calculate(
            "in_absolute_poverty_bhc", period=PERIOD, map_to="person"
        ).values
    except Exception:
        return float("nan")
    return float(np.average(values, weights=p_w))


def main() -> None:
    dataset, baseline, persons = _baseline_and_persons(DATASET, None, PERIOD)
    base_total = _hni_total(baseline)
    base_abs = _abs_poverty(baseline)
    weight = persons["weight"].to_numpy(float)
    cushioning, abs_pov = {}, {}
    for name in FAMILIES:
        table = apply_shocks(persons, PRESETS[name], seed=0)
        sim = build_shocked_simulation(dataset, baseline, table, PERIOD)
        gross = float(
            (
                (
                    persons["employment_income"].to_numpy(float)
                    - table["employment_income"].to_numpy(float)
                )
                * weight
            ).sum()
        )
        net = base_total - _hni_total(sim)
        cushioning[name] = {
            "gross_earnings_loss": gross,
            "net_disposable_loss": net,
            "cushioning_rate": 1.0 - net / gross,
        }
        abs_pov[name] = _abs_poverty(sim) - base_abs
        print(name, cushioning[name], flush=True)
        del sim
    (RESULTS / "cushioning_seed0.json").write_text(json.dumps(cushioning, indent=2))
    if not np.isnan(base_abs):
        (RESULTS / "absolute_poverty_fixed_line_seed0.json").write_text(
            json.dumps(abs_pov, indent=2)
        )
    else:
        print(
            "[skip] in_absolute_poverty_bhc unavailable in this "
            "policyengine-uk version; absolute_poverty_fixed_line_seed0.json "
            "left unchanged."
        )


if __name__ == "__main__":
    main()
