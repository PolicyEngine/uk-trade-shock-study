"""Duration sensitivity, full_tariff displacement margin.

Conventions mirror the sister study's analysis/sensitivity_duration_takeup.py.
10 Monte Carlo displacement draws (seeds 0-9) per variant.

(a) DURATION: the central results annualise a full-year (12-month) out-of-work
    spell. Under d = 6 months, displaced workers keep 50% of baseline annual
    employment income, evaluated IN-MODEL so taxes and means-tested benefits
    respond to the actual annual income. Documented hybrid: the annual model
    has no intra-year timing, so displaced persons carry the full displacement
    transition (hours zeroed, employment_status = UNEMPLOYED) while receiving
    half a year's earnings. d = 12 months is the central case, recomputed on
    the same seeds for comparability.

(b) UC TAKE-UP: superseded. The take-up sensitivity is now reported by the
    main-text grid (analysis/takeup_sensitivity.py); the variants that once
    lived here have been removed. All runs below use the corrected default
    post-shock re-draw at uc_takeup = 0.80 (shocks.DEFAULT_UC_TAKEUP).

Writes results/sensitivity_duration_takeup.json.
Usage: .venv/bin/python analysis/sensitivity_duration_takeup.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from uk_trade_shock_study.runner import _baseline_and_persons
from uk_trade_shock_study.shocks import (
    DEFAULT_UC_TAKEUP,
    PRESETS,
    apply_shocks,
    build_shocked_simulation,
)

PERIOD = 2026
DATASET = Path("data/frs_2024_25.h5")
RESULTS = Path("results")
N_DRAWS = 10
# The take-up half of this script is SUPERSEDED by the main-text take-up grid
# (analysis/takeup_sensitivity.py -> Table tab:takeupgrid). Only the duration
# variants are run and written here.
VARIANTS = (
    "duration_12m_central",
    "duration_6m",
)


def sim_metrics(sim):
    hh_w = sim.calculate("household_weight", period=PERIOD, map_to="household").values
    p_w = sim.calculate("person_weight", period=PERIOD, map_to="person").values
    return {
        "gov": float((sim.calculate("gov_balance", period=PERIOD, map_to="household").values * hh_w).sum()),
        "pov_bhc": float(np.average(
            sim.calculate("in_poverty_bhc", period=PERIOD, map_to="person").values, weights=p_w
        )),
        "hni": float((sim.calculate("hbai_household_net_income", period=PERIOD, map_to="household").values * hh_w).sum()),
        "uc": float((sim.calculate("universal_credit", period=PERIOD, map_to="household").values * hh_w).sum()),
    }


def main() -> None:
    dataset, baseline, persons = _baseline_and_persons(DATASET, None, PERIOD)
    w = persons["weight"].to_numpy(float)
    base_emp = persons["employment_income"].to_numpy(float)
    scen = PRESETS["full_tariff_displacement"]
    base = sim_metrics(baseline)

    modelled_takeup = float(np.average(
        baseline.calculate("would_claim_uc", period=PERIOD, map_to="person").values,
        weights=w,
    ))
    rows = {v: [] for v in VARIANTS}
    for seed in range(N_DRAWS):
        table = apply_shocks(persons, scen, seed=seed)
        displaced = table["displaced"].to_numpy()

        for variant in VARIANTS:
            t = table.copy()
            t.attrs = dict(table.attrs)
            base_row = base
            if variant == "duration_6m":
                t["employment_income"] = np.where(
                    displaced, 0.5 * base_emp, t["employment_income"].to_numpy(float)
                )
            sim = build_shocked_simulation(dataset, baseline, t, PERIOD)
            m = sim_metrics(sim)
            del sim
            gross = float(((base_emp - t["employment_income"].to_numpy(float)) * w).sum())
            net = base_row["hni"] - m["hni"]
            rows[variant].append({
                "gross_earnings_loss_bn": gross / 1e9,
                "exchequer_cost_bn": (base_row["gov"] - m["gov"]) / 1e9,
                "cushioning_rate": 1.0 - net / gross,
                "uc_change_bn": (m["uc"] - base_row["uc"]) / 1e9,
                "poverty_change_bhc_pp": 100 * (m["pov_bhc"] - base_row["pov_bhc"]),
                "displaced_weighted": float(w[displaced].sum()),
            })
            print(f"[{variant}] seed {seed}: {json.dumps(rows[variant][-1])}", flush=True)

    out = {
        "scenario": "full_tariff_displacement",
        "n_draws": N_DRAWS,
        "modelled_baseline_uc_takeup_person_weighted": modelled_takeup,
        "central_post_shock_uc_takeup": DEFAULT_UC_TAKEUP,
        "notes": {
            "duration_6m": "displaced keep 50% of baseline annual earnings "
                           "(6-month spell), in-model hybrid: transition "
                           "(hours/status) unchanged.",
        },
    }
    for variant in VARIANTS:
        keys = rows[variant][0].keys()
        out[variant] = {
            k: {
                "mean": float(np.mean([r[k] for r in rows[variant]])),
                "sd": float(np.std([r[k] for r in rows[variant]], ddof=1)),
            }
            for k in keys
        }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "sensitivity_duration_takeup.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({v: out[v] for v in VARIANTS}, indent=2))


if __name__ == "__main__":
    main()
