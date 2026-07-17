"""Aggregate poverty gap and poverty among affected households.

The national headcount change (+0.055pp under full displacement) is near-null;
this script asks whether the intensive margin moves: the aggregate relative
BHC poverty gap (policyengine-uk's poverty_gap_bhc, household shortfall below
the relative BHC line, weighted to £bn/yr), baseline vs shocked, nationally
and among AFFECTED households only (households containing a displaced
worker), plus the poverty-rate change among people in affected households.

Scenarios: full_tariff and epd displacement (20 Monte Carlo draws each,
seeds 0-19; affected-household baseline values vary by draw because the drawn
households differ), plus the deterministic full_tariff wage cut (affected =
households with any earnings cut).

Note: the shocked gap uses each simulation's own relative line; the shock is
far too small to move the median (Appendix on absolute poverty), so this is
indistinguishable from a fixed-line gap.

Writes results/poverty_gap.json.
Usage: .venv/bin/python analysis/poverty_gap.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from uk_trade_shock_study.runner import _baseline_and_persons
from uk_trade_shock_study.shocks import PRESETS, apply_shocks, build_shocked_simulation

PERIOD = 2026
DATASET = Path("data/frs_2024_25.h5")
RESULTS = Path("results")
N_DRAWS = 20


def gap_arrays(sim):
    return {
        "id": sim.calculate("household_id", period=PERIOD, map_to="household").values.astype(int),
        "weight": sim.calculate("household_weight", period=PERIOD, map_to="household").values.astype(float),
        "gap": sim.calculate("poverty_gap_bhc", period=PERIOD, map_to="household").values.astype(float),
        "n_people": sim.calculate("household_count_people", period=PERIOD, map_to="household").values.astype(float),
        "in_pov_person": sim.calculate("in_poverty_bhc", period=PERIOD, map_to="person").values.astype(float),
    }


def draw_gap(base, shock, p_w, p_hh_row, affected_person):
    aff_hh = np.zeros(len(base["id"]), dtype=bool)
    aff_hh[np.unique(p_hh_row[affected_person])] = True
    p_aff = aff_hh[p_hh_row]

    def gap_bn(a, mask=None):
        m = np.ones(len(a["gap"]), bool) if mask is None else mask
        return float((a["gap"][m] * a["weight"][m]).sum() / 1e9)

    def pov_rate(a, pmask):
        return float(np.average(a["in_pov_person"][pmask], weights=p_w[pmask]))

    return {
        "national_gap_baseline_bn": gap_bn(base),
        "national_gap_shocked_bn": gap_bn(shock),
        "affected_hh_gap_baseline_bn": gap_bn(base, aff_hh),
        "affected_hh_gap_shocked_bn": gap_bn(shock, aff_hh),
        "affected_people_weighted": float(p_w[p_aff].sum()),
        "affected_pov_rate_baseline": pov_rate(base, p_aff),
        "affected_pov_rate_shocked": pov_rate(shock, p_aff),
        "affected_pov_rate_change_pp": 100 * (pov_rate(shock, p_aff) - pov_rate(base, p_aff)),
        "national_gap_change_bn": gap_bn(shock) - gap_bn(base),
        "affected_hh_gap_change_bn": gap_bn(shock, aff_hh) - gap_bn(base, aff_hh),
    }


def main() -> None:
    dataset, baseline, persons = _baseline_and_persons(DATASET, None, PERIOD)
    base = gap_arrays(baseline)
    p_w = persons["weight"].to_numpy(float)
    p_hh = baseline.calculate("household_id", period=PERIOD, map_to="person").values.astype(int)
    hh_row = pd.Series(np.arange(len(base["id"])), index=base["id"])
    p_hh_row = hh_row[p_hh].to_numpy()
    base_emp = persons["employment_income"].to_numpy(float)

    out = {"n_draws": N_DRAWS}
    for name in ("full_tariff_displacement", "epd_displacement"):
        draws = []
        for seed in range(N_DRAWS):
            table = apply_shocks(persons, PRESETS[name], seed=seed)
            shocked = build_shocked_simulation(dataset, baseline, table, PERIOD)
            draws.append(draw_gap(
                base, gap_arrays(shocked), p_w, p_hh_row, table["displaced"].to_numpy()
            ))
            del shocked
            print(f"[poverty_gap] {name} seed {seed}: {json.dumps(draws[-1])}", flush=True)
        out[name] = {
            k: {
                "mean": float(np.mean([d[k] for d in draws])),
                "sd": float(np.std([d[k] for d in draws], ddof=1)),
            }
            for k in draws[0]
        }
        out[name + "_per_draw"] = draws

    wc_table = apply_shocks(persons, PRESETS["full_tariff_wage_cut"], seed=0)
    wc_shocked = build_shocked_simulation(dataset, baseline, wc_table, PERIOD)
    out["full_tariff_wage_cut"] = draw_gap(
        base, gap_arrays(wc_shocked), p_w, p_hh_row,
        base_emp > wc_table["employment_income"].to_numpy(float),
    )
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "poverty_gap.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if not k.endswith("_per_draw")}, indent=2))


if __name__ == "__main__":
    main()
