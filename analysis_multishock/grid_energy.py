"""Two-axis sensitivity grid for the energy second stage.

Axes (referee round, August 2026):
  1. Rebase factor: the paper's computed pre-crisis rebase (~0.606), an
     intermediate 0.8, and 1.0 (no rebase).  The flat-cash discretionary
     instruments (EBSS, cost-of-living payments) do not scale with the
     rebase while the shock denominator does, so the cushioning rates
     depend on it -- the dependence the manuscript now discloses.
  2. Discretionary stack: none / EPG only / EPG+EBSS / full (+CoL).

One PolicyEngine run; every cell is arithmetic on the same household
arrays (the measured automatic response is structurally zero, verified
by second_stage_energy.py, so no re-simulation per cell is needed).

Run:  python grid_energy.py --dataset <path to enhanced_frs.h5>
Emits out/grid_energy.json, out/generated_grid.tex, out/table_grid.tex.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from second_stage_energy import P, weighted, wmean

REBASES = [None, 0.8, 1.0]          # None = paper's computed rebase
STACKS = ["none", "epg", "epg_ebss", "full"]
STACK_LABEL = {"none": "None", "epg": "EPG only",
               "epg_ebss": "EPG + EBSS", "full": "Full stack"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--outdir", default="out")
    args = ap.parse_args()

    from policyengine_uk import Microsimulation

    yr = P["sim_year"]
    sim = Microsimulation(dataset=args.dataset)

    def hh(name):
        return np.asarray(sim.calculate(name, yr, map_to="household"))

    w = np.asarray(sim.calculate("household_weight", yr))
    energy_raw = np.asarray(sim.calculate("domestic_energy_consumption", yr))
    equiv_income = np.asarray(sim.calculate("equiv_hbai_household_net_income", yr))

    # Deciles: identical construction to second_stage_energy.py.
    order = np.argsort(equiv_income)
    cw = np.cumsum(w[order])
    dec = np.zeros(len(equiv_income), dtype=int)
    dec[order] = np.minimum((cw / cw[-1] * 10).astype(int), 9)

    # Fixed-cash discretionary components (rebase-invariant).
    ebss = np.full_like(energy_raw, P["ebss_per_household"])
    means_tested_amt = sum(hh(v) for v in
                           ["universal_credit", "pension_credit", "tax_credits",
                            "income_support", "housing_benefit", "jsa_income",
                            "esa_income"])
    col = ((means_tested_amt > 0) * P["col_means_tested"]
           + (hh("state_pension") > 0) * P["col_pensioner"]
           + ((hh("pip") + hh("dla") + hh("attendance_allowance")) > 0)
           * P["col_disability"])

    base_cap_fy = (P["cap_apr_2021"] + P["cap_oct_2021"]) / 2.0
    paper_rebase = base_cap_fy / wmean(energy_raw, w)
    counter_rise = (P["cap_apr_2022"] + P["cap_oct_2022"]) / 2.0 / base_cap_fy - 1.0
    realised_rise = (P["cap_apr_2022"] + P["epg_level"]) / 2.0 / base_cap_fy - 1.0

    cells = []
    for rb in REBASES:
        rebase = paper_rebase if rb is None else rb
        energy = energy_raw * rebase
        dE_counter = energy * counter_rise
        epg_cushion = energy * (counter_rise - realised_rise)
        G = weighted(dE_counter, w)
        for stack in STACKS:
            disc = np.zeros_like(energy)
            if stack in ("epg", "epg_ebss", "full"):
                disc = disc + epg_cushion
            if stack in ("epg_ebss", "full"):
                disc = disc + ebss
            if stack == "full":
                disc = disc + col
            cell = {
                "rebase": round(rebase, 4),
                "rebase_label": "paper" if rb is None else f"{rb:.1f}",
                "stack": stack,
                "counter_agg_gbp_bn": round(G / 1e9, 1),
                "discretionary_gbp_bn": round(weighted(disc, w) / 1e9, 1),
                "rate_pct": round(100 * weighted(disc, w) / G, 1),
            }
            for d, key in ((0, "d1"), (9, "d10")):
                m = dec == d
                cell[f"cushion_pct_{key}"] = round(
                    100 * weighted(disc[m], w[m]) / weighted(dE_counter[m], w[m]), 1)
            cells.append(cell)

    out = Path(args.outdir)
    out.mkdir(exist_ok=True)
    res = {"meta": {"dataset": Path(args.dataset).name,
                    "paper_rebase": round(paper_rebase, 4),
                    "records": int(len(w))},
           "cells": cells}
    (out / "grid_energy.json").write_text(json.dumps(res, indent=1))

    # Macros for the four corners quoted in prose.
    def cell(rb_label, stack):
        return next(c for c in cells
                    if c["rebase_label"] == rb_label and c["stack"] == stack)

    macros = {
        "GridPaperFullRatePct": cell("paper", "full")["rate_pct"],
        "GridUnityFullRatePct": cell("1.0", "full")["rate_pct"],
        "GridPaperEpgRatePct": cell("paper", "epg")["rate_pct"],
        "GridUnityEpgRatePct": cell("1.0", "epg")["rate_pct"],
        "GridPaperFullDOne": cell("paper", "full")["cushion_pct_d1"],
        "GridUnityFullDOne": cell("1.0", "full")["cushion_pct_d1"],
        "GridPaperEpgDOne": cell("paper", "epg")["cushion_pct_d1"],
        "GridPaperEpgDTen": cell("paper", "epg")["cushion_pct_d10"],
    }
    with open(out / "generated_grid.tex", "w") as f:
        f.write("% generated by grid_energy.py -- do not edit by hand.\n")
        for k, v in macros.items():
            f.write(f"\\newcommand{{\\Ss{k}}}{{{v:.1f}}}\n")

    # The 12-cell table body.
    with open(out / "table_grid.tex", "w") as f:
        f.write("% generated by grid_energy.py -- do not edit by hand.\n")
        f.write("\\begin{tabular}{llrrrr}\n\\toprule\n")
        f.write("Rebase & Stack & Shock (\\pounds bn) & Rate (\\%)"
                " & D1 (\\%) & D10 (\\%) \\\\\n\\midrule\n")
        last_rb = None
        for c in cells:
            rb = (f"{c['rebase']:.3f}" + (" (paper)" if c["rebase_label"] == "paper" else ""))
            show_rb = rb if rb != last_rb else ""
            last_rb = rb
            f.write(f"{show_rb} & {STACK_LABEL[c['stack']]} & "
                    f"{c['counter_agg_gbp_bn']:.1f} & {c['rate_pct']:.1f} & "
                    f"{c['cushion_pct_d1']:.1f} & {c['cushion_pct_d10']:.1f} \\\\\n")
            if c["stack"] == "full" and c is not cells[-1]:
                f.write("\\addlinespace\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    print(json.dumps(res["meta"]))
    for c in cells:
        print(c)


if __name__ == "__main__":
    main()
