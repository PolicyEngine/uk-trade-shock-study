"""Two-axis sensitivity grid for the energy second stage.

Axes (referee round, August 2026):
  1. Rebase factor: the paper's computed pre-crisis rebase (~0.606), an
     intermediate 0.8, and 1.0 (no rebase).  The flat-cash discretionary
     instruments (EBSS, cost-of-living payments) do not scale with the
     rebase while the shock denominator does, so the cushioning rates
     depend on it -- the dependence the manuscript now discloses.
  2. Discretionary stack: none / EPG only / EPG+EBSS / energy-specific
     (EPG+EBSS+Council Tax Rebate) / full (+CoL).

One PolicyEngine run; every cell is arithmetic on the same household
arrays (the measured automatic response is structurally zero, verified
by second_stage_energy.py, so no re-simulation per cell is needed).

Run:  python grid_energy.py   (certified Microcosm UK via policyengine.py)
Emits results/grid_energy.json, results/seasonal_energy.json (actual-cap
path and seasonal weighting), results/energy_referee_checks.json (a rebase
below the paper's with the maximum decile rate, the Council Tax Rebate
variant and the Cost of Living Payments outturn check; see
energy_referee_checks.py), results/generated_grid.tex and
results/table_grid.tex.  The printed table is built from the referee-check
cells; grid_energy.json keeps its three canonical rebases unchanged (it is
a hash-pinned quoted result).
"""

import argparse
import json
from pathlib import Path

import numpy as np

from second_stage_energy import P, load_sim, col_components, ctr_rebate, weighted, wmean
from second_stage_energy import RESULTS_DIR, PAPER_DIR
from energy_referee_checks import referee_checks

# None = paper's computed rebase.  The pinned grid_energy.json keeps these
# three; the printed table adds one below the paper's
# (energy_referee_checks.TABLE_REBASES).
REBASES = [None, 0.8, 1.0]
# Fine rebase sweep for the heatmap: 0.55 to 1.05 in steps of 0.05.
REBASES_FINE = [round(0.55 + 0.05 * i, 2) for i in range(11)]
# "energy" is the energy-specific stack: Guarantee + Bills Support (GBP 400)
# + Council Tax Rebate (GBP 150).  "epg_ebss" is the intermediate rung
# without the rebate.  The paper-facing macros named ...Ebss... report the
# "energy" stack (see the macro block below).
STACKS = ["none", "epg", "epg_ebss", "energy", "full"]
STACK_LABEL = {"none": "None", "epg": "EPG only",
               "epg_ebss": "EPG + EBSS",
               "energy": "+ CT rebate (energy-specific)",
               "full": "Full stack"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None,
                    help="ignored; data come from the certified Microcosm UK build")
    ap.add_argument("--outdir", default=str(RESULTS_DIR))
    args = ap.parse_args()

    yr = P["sim_year"]
    sim = load_sim()

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
    col = sum(col_components(sim))
    ctr = ctr_rebate(sim)

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
            if stack in ("epg", "epg_ebss", "energy", "full"):
                disc = disc + epg_cushion
            if stack in ("epg_ebss", "energy", "full"):
                disc = disc + ebss
            if stack in ("energy", "full"):
                disc = disc + ctr
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

    # Fine sweep (figure only; the table prints four rebases).
    # The cost-of-living payments are decomposed into their three statutory
    # components, added cumulatively, so the stack axis shows each
    # instrument's marginal contribution.
    col_mt, col_pen, col_dis = col_components(sim)
    STACKS_FINE = ["epg", "epg_ebss", "ctr", "col_mt", "col_pen", "col_dis"]
    cells_fine = []
    for rb in REBASES_FINE:
        energy = energy_raw * rb
        dE_counter = energy * counter_rise
        epg_cushion = energy * (counter_rise - realised_rise)
        G = weighted(dE_counter, w)
        disc = np.zeros_like(energy)
        adds = {"epg": epg_cushion, "epg_ebss": ebss, "ctr": ctr,
                "col_mt": col_mt,
                "col_pen": col_pen, "col_dis": col_dis}
        for stack in STACKS_FINE:
            disc = disc + adds[stack]
            cells_fine.append({"rebase": rb, "stack": stack,
                               "rate_pct": round(100 * weighted(disc, w) / G, 1)})

    out = Path(args.outdir)
    out.mkdir(exist_ok=True)
    res = {"meta": {"dataset": "microcosm_uk_2024_25",
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
    disc_full_ann = epg_ann + ebss + ctr + col
    ann = {
        "AnnCounterFyCap": f"{ann_fy:,.0f}",
        "AnnCounterAggBn": f"{G_ann/1e9:,.1f}",
        "AnnEpgRatePct": f"{100*weighted(epg_ann,w)/G_ann:.1f}",
        "AnnFullRatePct": f"{100*weighted(disc_full_ann,w)/G_ann:.1f}",
    }
    # Energy-specific stack (EPG + EBSS + Council Tax Rebate) on the
    # announced path.
    ann["AnnEbssRatePct"] = (
        f"{100*(weighted(epg_ann,w)+weighted(ebss+ctr,w))/G_ann:.1f}")
    # Outturn-calibrated stack (referee variant): EBSS scaled to its NAO
    # scheme outturn (GBP 11.69bn, all UK), removing the weight-excess
    # overshoot from the numerator.
    ebss_cal = ebss * (P["ebss_outturn_gbp_bn"] * 1e9 / weighted(ebss, w))
    dE_paper = energy_raw * paper_rebase * counter_rise
    epg_paper = energy_raw * paper_rebase * (counter_rise - realised_rise)
    G_paper = weighted(dE_paper, w)
    disc_cal = epg_paper + ebss_cal + ctr + col
    ann["CalFullRatePct"] = f"{100*weighted(disc_cal,w)/G_paper:.1f}"
    ann["CalFullBn"] = f"{weighted(disc_cal,w)/1e9:,.1f}"

    # --- Actual-cap path and seasonal weighting (referee round) -------------
    # Month weights: PolicyEngine UK's own seasonal profile, read off the
    # model at run time (monthly_domestic_energy_consumption, which the
    # model's Energy Price Guarantee variable epg_subsidy uses; the model
    # sources it to Resolution Foundation, "A chilling crisis", 2022, p.11).
    # Nothing is hard-coded here: the share of each calendar month is the
    # weighted monthly variable over the weighted annual variable.
    annual = weighted(energy_raw, w)
    month_share = [
        weighted(np.asarray(sim.calculate(
            "monthly_domestic_energy_consumption", f"{yr}-{m:02d}")), w) / annual
        for m in range(1, 13)]
    cw = {"apr_sep": sum(month_share[3:9]), "oct_dec": sum(month_share[9:12]),
          "jan_mar": sum(month_share[0:3])}
    eq = {"apr_sep": 0.5, "oct_dec": 0.25, "jan_mar": 0.25}

    def fy_mean(wt, apr_sep, oct_dec, jan_mar):
        tot = wt["apr_sep"] + wt["oct_dec"] + wt["jan_mar"]
        return (wt["apr_sep"] * apr_sep + wt["oct_dec"] * oct_dec
                + wt["jan_mar"] * jan_mar) / tot

    def seasonal_variant(wt, actual_cap, rebase_target=None):
        """Cushioning on one month-weighting and one counterfactual path.
        The same weights price the base, counterfactual and realised years;
        the energy rebase target is the base-year mean on those weights
        (a household's FY2021-22 bill is its consumption-weighted cap)."""
        base = fy_mean(wt, P["cap_apr_2021"], P["cap_oct_2021"], P["cap_oct_2021"])
        counter = fy_mean(wt, P["cap_apr_2022"], P["cap_oct_2022"],
                          P["cap_jan_2023"] if actual_cap else P["cap_oct_2022"])
        realised = fy_mean(wt, P["cap_apr_2022"], P["epg_level"], P["epg_level"])
        energy = energy_raw * ((rebase_target or base) / wmean(energy_raw, w))
        dE = energy * (counter / base - 1.0)
        epg = energy * (counter - realised) / base
        G = weighted(dE, w)
        stacks = {"epg": epg, "epg_ebss": epg + ebss,
                  "energy": epg + ebss + ctr,
                  "full": epg + ebss + ctr + col}
        out = {"base_fy_mean_cap": round(base, 1),
               "counterfactual_fy_mean_cap": round(counter, 1),
               "realised_fy_mean_cap": round(realised, 1),
               "counterfactual_rise_pct": round(100 * (counter / base - 1), 1),
               "realised_rise_pct": round(100 * (realised / base - 1), 1),
               "counter_agg_gbp_bn": round(G / 1e9, 2),
               "realised_agg_gbp_bn": round(
                   weighted(energy * (realised / base - 1.0), w) / 1e9, 2),
               "epg_gbp_bn": round(weighted(epg, w) / 1e9, 2)}
        for key, disc in stacks.items():
            out[f"{key}_rate_pct"] = round(100 * weighted(disc, w) / G, 1)
            for d, dk in ((0, "d1"), (9, "d10")):
                m = dec == d
                out[f"{key}_cushion_pct_{dk}"] = round(
                    100 * weighted(disc[m], w[m]) / weighted(dE[m], w[m]), 1)
        return out

    equal_base = fy_mean(eq, P["cap_apr_2021"], P["cap_oct_2021"], P["cap_oct_2021"])
    seasonal = {
        "month_share_source": "PolicyEngine UK monthly_domestic_energy_consumption "
                              "(Resolution Foundation, A chilling crisis, 2022, p.11)",
        "month_share_jan_to_dec": [round(x, 4) for x in month_share],
        "fy_weights_consumption": {k: round(v, 4) for k, v in cw.items()},
        "equal_held": seasonal_variant(eq, False),       # current headline
        "equal_actual": seasonal_variant(eq, True),      # = announced path
        "consumption_held": seasonal_variant(cw, False),
        "consumption_actual": seasonal_variant(cw, True),
        # Sensitivity: consumption-weighted prices, but the energy rebase
        # target left at the equal-weight FY2021-22 mean.
        "consumption_actual_rebase_held": seasonal_variant(cw, True, equal_base),
    }
    # Own file: grid_energy.json is a frozen, hash-pinned quoted result.
    (out / "seasonal_energy.json").write_text(json.dumps(seasonal, indent=1))
    seas_macros = {
        "SeasShareAprSepPct": f"{100*cw['apr_sep']:.1f}",
        "SeasShareOctDecPct": f"{100*cw['oct_dec']:.1f}",
        "SeasShareJanMarPct": f"{100*cw['jan_mar']:.1f}",
    }
    for tag, key in (("EqHeld", "equal_held"), ("EqActual", "equal_actual"),
                     ("CwHeld", "consumption_held"),
                     ("CwActual", "consumption_actual")):
        v = seasonal[key]
        seas_macros.update({
            f"Seas{tag}BaseCap": f"{v['base_fy_mean_cap']:,.0f}",
            f"Seas{tag}CounterCap": f"{v['counterfactual_fy_mean_cap']:,.0f}",
            f"Seas{tag}RealisedCap": f"{v['realised_fy_mean_cap']:,.0f}",
            f"Seas{tag}CounterRisePct": f"{v['counterfactual_rise_pct']:.1f}",
            f"Seas{tag}RealisedRisePct": f"{v['realised_rise_pct']:.1f}",
            f"Seas{tag}CounterAggBn": f"{v['counter_agg_gbp_bn']:,.1f}",
            f"Seas{tag}EpgBn": f"{v['epg_gbp_bn']:,.1f}",
            f"Seas{tag}EpgRatePct": f"{v['epg_rate_pct']:.1f}",
            f"Seas{tag}EbssRatePct": f"{v['energy_rate_pct']:.1f}",
            f"Seas{tag}FullRatePct": f"{v['full_rate_pct']:.1f}",
            f"Seas{tag}FullDOne": f"{v['full_cushion_pct_d1']:.1f}",
            f"Seas{tag}FullDTen": f"{v['full_cushion_pct_d10']:.1f}",
            f"Seas{tag}EbssDOne": f"{v['energy_cushion_pct_d1']:.1f}",
            f"Seas{tag}EbssDTen": f"{v['energy_cushion_pct_d10']:.1f}",
        })
    seas_macros["SeasCwActualRebaseHeldFullRatePct"] = (
        f"{seasonal['consumption_actual_rebase_held']['full_rate_pct']:.1f}")
    seas_macros["SeasCwActualRebaseHeldEpgBn"] = (
        f"{seasonal['consumption_actual_rebase_held']['epg_gbp_bn']:,.1f}")

    # --- Referee checks: rebase below the paper's, maximum decile rate, the
    # Council Tax Rebate variant and the Cost of Living outturn comparison.
    # Own file: grid_energy.json is a frozen, hash-pinned quoted result.
    ref, ref_macros, table_cells = referee_checks(
        sim, w, dec, energy_raw, paper_rebase, counter_rise, realised_rise,
        ebss, ctr, (col_mt, col_pen, col_dis))
    (out / "energy_referee_checks.json").write_text(json.dumps(ref, indent=1))
    # The table's cells must agree with the pinned grid where they overlap.
    for c in cells:
        t = next(x for x in table_cells if x["rebase_label"] == c["rebase_label"]
                 and x["stack"] == c["stack"])
        for k in ("counter_agg_gbp_bn", "rate_pct", "cushion_pct_d1",
                  "cushion_pct_d10"):
            assert abs(t[k] - c[k]) < 0.051, (c["rebase_label"], c["stack"], k)

    # MACRO SEMANTICS.  ...Ebss... = the ENERGY-SPECIFIC stack (Guarantee +
    # Bills Support + Council Tax Rebate); ...Full... = that plus the Cost
    # of Living Payments.  The same numbers without the rebate are emitted
    # as ...NoCtr... by energy_referee_checks.py.
    macros = {
        "GridPaperFullRatePct": cell("paper", "full")["rate_pct"],
        "GridUnityFullRatePct": cell("1.0", "full")["rate_pct"],
        "GridPaperEpgRatePct": cell("paper", "epg")["rate_pct"],
        "GridUnityEpgRatePct": cell("1.0", "epg")["rate_pct"],
        "GridPaperFullDOne": cell("paper", "full")["cushion_pct_d1"],
        "GridUnityFullDOne": cell("1.0", "full")["cushion_pct_d1"],
        "GridPaperEpgDOne": cell("paper", "epg")["cushion_pct_d1"],
        "GridPaperEpgDTen": cell("paper", "epg")["cushion_pct_d10"],
        "GridPaperEbssRatePct": cell("paper", "energy")["rate_pct"],
        "GridPaperEbssDOne": cell("paper", "energy")["cushion_pct_d1"],
        "GridPaperEbssDTen": cell("paper", "energy")["cushion_pct_d10"],
        "GridUnityEbssRatePct": cell("1.0", "energy")["rate_pct"],
        "GridUnityEpgDTen": cell("1.0", "epg")["cushion_pct_d10"],
        "GridUnityFullDTen": cell("1.0", "full")["cushion_pct_d10"],
    }
    with open(out / "generated_grid.tex", "w") as f:
        f.write("% generated by grid_energy.py -- do not edit by hand.\n")
        for k, v in macros.items():
            f.write(f"\\newcommand{{\\Ss{k}}}{{{v:.1f}}}\n")
        for k, v in ann.items():
            f.write(f"\\newcommand{{\\Ss{k}}}{{{v}}}\n")
        for k, v in seas_macros.items():
            f.write(f"\\newcommand{{\\Ss{k}}}{{{v}}}\n")
        for k, v in ref_macros.items():
            f.write(f"\\newcommand{{\\Ss{k}}}{{{v}}}\n")

    # The table body: four rebases (one below the paper's) x five stacks,
    # with the highest decile rate and the decile it belongs to.
    with open(out / "table_grid.tex", "w") as f:
        f.write("% generated by grid_energy.py -- do not edit by hand.\n")
        f.write("\\begin{tabular}{llrrrrl}\n\\toprule\n")
        f.write("Rebase & Stack & Shock (\\pounds bn) & Rate (\\%)"
                " & D1 (\\%) & D10 (\\%) & Max decile (\\%) \\\\\n\\midrule\n")
        last_rb = None
        for c in table_cells:
            rb = (f"{c['rebase']:.3f}" + (" (paper)" if c["rebase_label"] == "paper" else ""))
            show_rb = rb if rb != last_rb else ""
            last_rb = rb
            which = "all" if c["max_decile"] is None else f"D{c['max_decile']}"
            f.write(f"{show_rb} & {STACK_LABEL[c['stack']]} & "
                    f"{c['counter_agg_gbp_bn']:.1f} & {c['rate_pct']:.1f} & "
                    f"{c['cushion_pct_d1']:.1f} & {c['cushion_pct_d10']:.1f} & "
                    f"{c['max_decile_pct']:.1f} ({which}) \\\\\n")
            if c["stack"] == "full" and c is not table_cells[-1]:
                f.write("\\addlinespace\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    import shutil
    for f in ("generated_grid.tex", "table_grid.tex"):
        shutil.copy(RESULTS_DIR / f, PAPER_DIR / f)

    print(json.dumps(res["meta"]))
    for c in cells:
        print(c)
    print(json.dumps(seasonal, indent=1))
    print(json.dumps({k: v for k, v in ref.items() if k != "grid"}, indent=1))
    print(json.dumps(ref["grid"]["thresholds"], indent=1))


if __name__ == "__main__":
    main()
