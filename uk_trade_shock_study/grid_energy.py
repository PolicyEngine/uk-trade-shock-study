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
from second_stage_energy import RESULTS_DIR, PAPER_DIR

REBASES = [None, 0.8, 1.0]          # None = paper's computed rebase (table)
# Fine rebase sweep for the heatmap: 0.55 to 1.05 in steps of 0.05.
REBASES_FINE = [round(0.55 + 0.05 * i, 2) for i in range(11)]
STACKS = ["none", "epg", "epg_ebss", "full"]
STACK_LABEL = {"none": "None", "epg": "EPG only",
               "epg_ebss": "EPG + EBSS", "full": "Full stack"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--outdir", default=str(RESULTS_DIR))
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

    # Fine sweep (figure only; the table keeps the three canonical columns).
    # The cost-of-living payments are decomposed into their three statutory
    # components, added cumulatively, so the stack axis shows each
    # instrument's marginal contribution.
    col_mt = (means_tested_amt > 0) * P["col_means_tested"]
    col_pen = (hh("state_pension") > 0) * P["col_pensioner"]
    col_dis = (((hh("pip") + hh("dla") + hh("attendance_allowance")) > 0)
               * P["col_disability"])
    STACKS_FINE = ["epg", "epg_ebss", "col_mt", "col_pen", "col_dis"]
    cells_fine = []
    for rb in REBASES_FINE:
        energy = energy_raw * rb
        dE_counter = energy * counter_rise
        epg_cushion = energy * (counter_rise - realised_rise)
        G = weighted(dE_counter, w)
        disc = np.zeros_like(energy)
        adds = {"epg": epg_cushion, "epg_ebss": ebss, "col_mt": col_mt,
                "col_pen": col_pen, "col_dis": col_dis}
        for stack in STACKS_FINE:
            disc = disc + adds[stack]
            cells_fine.append({"rebase": rb, "stack": stack,
                               "rate_pct": round(100 * weighted(disc, w) / G, 1)})

    out = Path(args.outdir)
    out.mkdir(exist_ok=True)
    res = {"meta": {"dataset": Path(args.dataset).name,
                    "paper_rebase": round(paper_rebase, 4),
                    "records": int(len(w))},
           "cells": cells, "cells_fine": cells_fine}
    (out / "grid_energy.json").write_text(json.dumps(res, indent=1))

    # Macros for the four corners quoted in prose.
    def cell(rb_label, stack):
        return next(c for c in cells
                    if c["rebase_label"] == rb_label and c["stack"] == stack)

    # Announced-path counterfactual (Ofgem's November 2022 announcement of
    # the Jan-Mar 2023 cap): FY mean (2 x Apr-2022 + Oct-2022 + Jan-2023)/4.
    ann_fy = (2 * P["cap_apr_2022"] + P["cap_oct_2022"] + P["cap_jan_2023"]) / 4.0
    ann_rise = ann_fy / base_cap_fy - 1.0
    energy_p = energy_raw * paper_rebase
    dE_ann = energy_p * ann_rise
    epg_ann = energy_p * (ann_rise - realised_rise)
    G_ann = weighted(dE_ann, w)
    disc_full_ann = epg_ann + ebss + col
    ann = {
        "AnnCounterFyCap": f"{ann_fy:,.0f}",
        "AnnCounterAggBn": f"{G_ann/1e9:,.1f}",
        "AnnEpgRatePct": f"{100*weighted(epg_ann,w)/G_ann:.1f}",
        "AnnFullRatePct": f"{100*weighted(disc_full_ann,w)/G_ann:.1f}",
    }
    # Outturn-calibrated stack (referee variant): EBSS scaled to its ~GBP 11bn
    # scheme outturn, removing the weight-excess overshoot from the numerator.
    ebss_cal = ebss * (11.0e9 / weighted(ebss, w))
    dE_paper = energy_raw * paper_rebase * counter_rise
    epg_paper = energy_raw * paper_rebase * (counter_rise - realised_rise)
    G_paper = weighted(dE_paper, w)
    disc_cal = epg_paper + ebss_cal + col
    ann["CalFullRatePct"] = f"{100*weighted(disc_cal,w)/G_paper:.1f}"
    ann["CalFullBn"] = f"{weighted(disc_cal,w)/1e9:,.1f}"

    macros = {
        "GridPaperFullRatePct": cell("paper", "full")["rate_pct"],
        "GridUnityFullRatePct": cell("1.0", "full")["rate_pct"],
        "GridPaperEpgRatePct": cell("paper", "epg")["rate_pct"],
        "GridUnityEpgRatePct": cell("1.0", "epg")["rate_pct"],
        "GridPaperFullDOne": cell("paper", "full")["cushion_pct_d1"],
        "GridUnityFullDOne": cell("1.0", "full")["cushion_pct_d1"],
        "GridPaperEpgDOne": cell("paper", "epg")["cushion_pct_d1"],
        "GridPaperEpgDTen": cell("paper", "epg")["cushion_pct_d10"],
        "GridPaperEbssRatePct": cell("paper", "epg_ebss")["rate_pct"],
        "GridPaperEbssDOne": cell("paper", "epg_ebss")["cushion_pct_d1"],
        "GridPaperEbssDTen": cell("paper", "epg_ebss")["cushion_pct_d10"],
        "GridUnityEbssRatePct": cell("1.0", "epg_ebss")["rate_pct"],
        "GridUnityEpgDTen": cell("1.0", "epg")["cushion_pct_d10"],
        "GridUnityFullDTen": cell("1.0", "full")["cushion_pct_d10"],
    }
    with open(out / "generated_grid.tex", "w") as f:
        f.write("% generated by grid_energy.py -- do not edit by hand.\n")
        for k, v in macros.items():
            f.write(f"\\newcommand{{\\Ss{k}}}{{{v:.1f}}}\n")
        for k, v in ann.items():
            f.write(f"\\newcommand{{\\Ss{k}}}{{{v}}}\n")

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

    import shutil
    for f in ("generated_grid.tex", "table_grid.tex"):
        shutil.copy(RESULTS_DIR / f, PAPER_DIR / f)

    print(json.dumps(res["meta"]))
    for c in cells:
        print(c)


if __name__ == "__main__":
    main()
