"""Aggregate poverty gap and poverty among affected households.

The national headcount change (+0.055pp under full displacement) is near-null;
this script asks whether the intensive margin moves: the aggregate relative
BHC poverty gap (policyengine-uk's poverty_gap_bhc, household shortfall below
the relative BHC line, weighted to £bn/yr), baseline vs shocked, nationally
and among AFFECTED households only (households containing a displaced
worker), plus the poverty-rate change among people in affected households.

Scenarios: full_tariff and epd displacement (``--n-draws`` Monte Carlo draws
each, seeds 0..n-1; affected-household baseline values vary by draw because
the drawn households differ), plus the deterministic full_tariff wage cut
(affected = households with any earnings cut).

DRAW COUNT. The default (5) is an exploratory setting: the affected-household
poverty rate has an assignment SD of roughly 30 points at that sample size,
which is not reportable. Run ``make poverty-gap-production`` to rebuild the
artifact at the submission design's 50 draws;
analysis/write_welfare_results.py refuses to emit macros below that.
The wage-cut margin stays a single deterministic run: it has no assignment
randomness, so extra seeds vary only through the post-shock UC take-up
re-draw, whose new-entitlement set is empty at the paper's calibration.

Note: the shocked gap uses each simulation's own relative line; the shock is
far too small to move the median (Appendix on absolute poverty), so this is
indistinguishable from a fixed-line gap.

Writes results/poverty_gap.json.
Usage: .venv/bin/python analysis/poverty_gap.py [--n-draws 50]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from uk_trade_shock_study.runner import _baseline_and_persons
from uk_trade_shock_study.shocks import PRESETS, apply_shocks, build_shocked_simulation

PERIOD = 2026
DATASET = Path("data/frs_2024_25.h5")
RESULTS = Path("results")
# Exploratory default; override with --n-draws. The affected-household
# poverty rate is unreportable at this sample size (SD ~30pp), so the
# production artifact is built at the submission design's 50 draws via
# `make poverty-gap-production`.
N_DRAWS = 5
#: Draws for the wage-cut margin: deterministic, so one run (see the module
#: docstring). Recorded in the artifact as ``wage_cut_n_draws``.
WAGE_CUT_N_DRAWS = 1


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


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--n-draws",
        type=int,
        default=N_DRAWS,
        help="Monte Carlo draws per displacement scenario (seeds 0..n-1)",
    )
    parser.add_argument(
        "--output", type=Path, default=RESULTS / "poverty_gap.json"
    )
    args = parser.parse_args(argv)
    if args.n_draws < 1:
        parser.error("--n-draws must be at least 1")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    n_draws = args.n_draws
    dataset, baseline, persons = _baseline_and_persons(DATASET, None, PERIOD)
    base = gap_arrays(baseline)
    p_w = persons["weight"].to_numpy(float)
    p_hh = baseline.calculate("household_id", period=PERIOD, map_to="person").values.astype(int)
    hh_row = pd.Series(np.arange(len(base["id"])), index=base["id"])
    p_hh_row = hh_row[p_hh].to_numpy()
    base_emp = persons["employment_income"].to_numpy(float)

    out = {"n_draws": n_draws, "wage_cut_n_draws": WAGE_CUT_N_DRAWS}
    for name in ("full_tariff_displacement", "epd_displacement"):
        draws = []
        for seed in range(n_draws):
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if not k.endswith("_per_draw")}, indent=2))


if __name__ == "__main__":
    main()
