"""Second stage: the 2022-23 energy episode on enhanced-FRS households.

Computes the cushioning of a trade-transmitted price shock through the
statutory tax-benefit system, decomposed into automatic and
discretionary components, and the dispersion of exposure within income
deciles.

Design notes, in response to referee comments on the first version:

* The fourteen cash benefits condition on nominal income, which the
  price shock does not move, so the calculator re-run (baseline vs
  shocked energy inputs) returns zero for them.  The model's own Energy
  Price Guarantee variable (epg_subsidy) does read energy consumption
  and does respond; the re-run records that response separately
  (NATIVE_INSTRUMENTS) and the paper classifies it as discretionary.
* Both price paths are on ONE basis: financial-year mean caps.  The
  counterfactual FY2022-23 mean is (1,971 + 3,549)/2 with no Energy
  Price Guarantee; the realised FY mean is (1,971 + 2,500)/2; the
  baseline FY2021-22 mean is (1,138 + 1,277)/2.  Mixing a point-to-point
  cap ratio with a financial-year mean ratio -- as the first version did
  -- made the cushioning rate incomparable across paths.
* Deciles are equivalised (HBAI before-housing-costs concept).
* The uprating shortfall is reported against the base it belongs to
  (benefit income), not against a single commodity's shock.
* The variance decomposition is reported on four bases, because a ratio
  with income in the denominator is dominated by very low incomes.
* Missing variables raise; the PolicyEngine version is recorded.

Run:  python second_stage_energy.py   (certified Microcosm UK via policyengine.py)
"""

import argparse
import json
from importlib.metadata import version as _pkg_version
from pathlib import Path

import numpy as np
from pathlib import Path as _Path

REPO = _Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO / "results"
PAPER_DIR = REPO / "paper"

# ---------------------------------------------------------------------------
# Declared parameters.  DATA unless marked ASSUMPTION.
# ---------------------------------------------------------------------------
P = {
    # Ofgem default tariff cap, typical dual-fuel direct debit (DATA:
    # Ofgem cap letters 6 Aug 2021, 26 Aug 2022, Figure 1).
    "cap_apr_2021": 1138.0,
    "cap_oct_2021": 1277.0,
    "cap_apr_2022": 1971.0,
    "cap_oct_2022": 3549.0,
    "epg_level": 2500.0,
    # Wholesale component, CfD-in-wholesale restatement (model v1.13).
    "wholesale_oct_2021": 550.0,
    "wholesale_oct_2022": 2468.0,
    # Discretionary support, 2022-23 (DATA: gov.uk scheme parameters).
    "ebss_per_household": 400.0,
    "col_means_tested": 650.0,
    "col_pensioner": 300.0,
    "col_disability": 150.0,
    # Uprating (DATA: DWP April 2022 order; ONS D7BT).  The 10.1 per cent is
    # the September 2022 twelve-month CPI rate, the figure the April 2023
    # uprating applied; the FY2022-23 mean of monthly twelve-month rates is
    # about 10.0 per cent (ONS D7BT), so the September convention is used
    # and named as such.
    "uprating_applied_apr_2022": 0.031,
    "cpi_sep_2022_yoy": 0.101,
    # Energy Bills Support Scheme outturn, all UK (DATA: NAO, Energy bills
    # support: an update, November 2024, GBP 11,692m; GB alone 11,364m).
    "ebss_outturn_gbp_bn": 11.69,
    # ASSUMPTION: simulated year.
    "sim_year": 2023,
    # ONS UK household count, for the weight reconciliation (DATA).
    "ons_uk_households_m": 28.4,
    # Ofgem cap announced November 2022 for Jan-Mar 2023 (DATA); used by the
    # grid's announced-path counterfactual variant.
    "cap_jan_2023": 4279.0,
    # ONS Family Spending FYE2022 mean total expenditure per household,
    # GBP/yr (DATA: derived from the pinned first-pass artifact; 528.9/wk).
    # The second stage rebases its expenditure denominator to this base so
    # shares of spending are level-comparable with the first pass.
    "ons_total_spend_yr": 27503.0,
}

MEANS_TESTED = ["universal_credit", "pension_credit", "tax_credits",
                "income_support", "housing_benefit", "jsa_income", "esa_income"]
# Fourteen cash benefits.  Thirteen were uprated by 10.1 per cent in April
# 2023; the winter fuel payment is a fixed cash amount (GBP 200/300, no
# uprating step in the model's parameters), so the list is "cash benefits",
# not "uprated cash benefits".
UPRATED_CASH = MEANS_TESTED + ["state_pension", "child_benefit", "pip", "dla",
                               "attendance_allowance", "carers_allowance",
                               "winter_fuel_allowance"]
# The model's own 2022-23 instruments.  PolicyEngine UK 2.95.0 encodes the
# Energy Price Guarantee (epg_subsidy, which reads domestic energy
# consumption), the Energy Bills Rebate and the cost-of-living payments as
# variables inside household_benefits.  The re-run below measures their
# response to the shocked energy inputs alongside the fourteen benefits.
NATIVE_INSTRUMENTS = ["epg_subsidy", "cost_of_living_support_payment",
                      "energy_bills_rebate", "household_benefits",
                      "household_net_income"]


# Weighted primitives, delegated to microdf (the survey-weighting library
# PolicyEngine itself returns results in), so no weight arithmetic is done
# by hand anywhere in the pipeline.  Signatures unchanged; verified to
# reproduce the previous hand-rolled results exactly.
from microdf import MicroSeries

from microcosm import BASE_YEAR, copy_inputs, microsimulation, provenance


#: Rulebook year of the 2022-23 Cost of Living Payments (GBP 650 means-tested
#: per qualifying benefit unit, GBP 300 pensioner per Winter Fuel household,
#: GBP 150 per disabled person), taken from the model rather than rebuilt.
COL_YEAR = 2022


def col_components(sim):
    """2022-23 Cost of Living Payments per household: (means-tested,
    pensioner, disability), from PolicyEngine UK's own eligibility rules."""
    def h(name):
        return np.asarray(sim.calculate(name, COL_YEAR, map_to="household"))
    total = h("cost_of_living_support_payment")
    mt = h("cost_of_living_means_tested_payment")
    dis = h("cost_of_living_disability_payment")
    return mt, total - mt - dis, dis


def ctr_rebate(sim):
    """The GBP 150 Council Tax Rebate per household (April 2022; part of the
    Energy Bills Rebate package, so classed energy-specific), from
    PolicyEngine UK's own rule at period 2022: council tax bands A-D in
    England, Scotland and Wales, none in Northern Ireland.  The component
    variable alone is read -- NOT energy_bills_rebate, which at calendar
    2022 also carries the October-December part of the GBP 400 scheme that
    the stack already counts in full as a flat GBP 400."""
    return np.asarray(sim.calculate("ebr_council_tax_rebate", COL_YEAR,
                                    map_to="household"))


#: Referee robustness switch (set by ``--deflate``): scale the copied inputs
#: from 2024-25 levels to each back-dated rulebook year with PolicyEngine
#: UK's own projection indices, inverted (microcosm.copy_inputs).  Off in
#: the central run.
DEFLATE = False


def load_sim(years=None, reform=None, deflate=None):
    """A managed Microcosm UK simulation with inputs present in ``years``.

    Microcosm UK carries survey inputs only for its base year; any other
    period a script calculates in (the episode's rulebook year, or the
    vintage run's previous year) gets every input column copied in first.
    """
    sim = microsimulation(reform)
    target = sorted(set(years or [P["sim_year"], COL_YEAR]) - {BASE_YEAR})
    if target:
        copy_inputs(sim, target,
                    deflate=DEFLATE if deflate is None else deflate)
    return sim


def weighted(x, w):
    """Weighted total of x."""
    return float(MicroSeries(np.asarray(x, dtype=float), weights=w).sum())


def wmean(x, w):
    """Weighted mean of x."""
    return float(MicroSeries(np.asarray(x, dtype=float), weights=w).mean())


def wquantile(x, w, q):
    """Weighted q-quantile of x."""
    return float(MicroSeries(np.asarray(x, dtype=float),
                             weights=w).quantile(q))


def variance_split(burden, w, dec):
    """Between/within-decile shares of the weighted variance of `burden`."""
    b = MicroSeries(np.asarray(burden, dtype=float), weights=w)
    gm = float(b.mean())
    total = float(MicroSeries((np.asarray(burden, dtype=float) - gm) ** 2,
                              weights=w).mean())
    between = 0.0
    wsum = float(np.sum(w))
    for d in np.unique(dec):
        m = dec == d
        between += float(np.sum(w[m])) * (wmean(burden[m], w[m]) - gm) ** 2
    between /= wsum
    return (round(100 * between / total, 1),
            round(100 * (1 - between / total), 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None,
                    help="ignored; data come from the certified Microcosm UK build")
    ap.add_argument("--out", default=str(RESULTS_DIR / "second_stage_energy.json"))
    ap.add_argument("--deflate", action="store_true",
                    help="robustness: deflate copied inputs to the rulebook "
                         "years with the model's own projection indices")
    ap.add_argument("--sim-year", type=int, default=None,
                    help="robustness: rulebook year for the whole stage "
                         "(default 2023, i.e. 2023-24 rules)")
    args = ap.parse_args()

    global DEFLATE
    DEFLATE = args.deflate
    if args.sim_year is not None:
        P["sim_year"] = args.sim_year
    yr = P["sim_year"]
    sim = load_sim()

    def hh(name, s=None):
        """Household-level total of a variable.  Raises if unavailable."""
        s = s or sim
        if name not in s.tax_benefit_system.variables:
            raise KeyError(f"variable not in this PolicyEngine UK build: {name}")
        return np.asarray(s.calculate(name, yr, map_to="household"))

    w = np.asarray(sim.calculate("household_weight", yr))
    persons_m = float(sim.calculate("people", yr).sum()) / 1e6  # microdf-native
    energy_raw = np.asarray(sim.calculate("domestic_energy_consumption", yr))
    elec_raw = np.asarray(sim.calculate("electricity_consumption", yr))
    gas_raw = np.asarray(sim.calculate("gas_consumption", yr))
    equiv_income = np.asarray(sim.calculate("equiv_hbai_household_net_income", yr))
    net_income = np.asarray(sim.calculate("household_net_income", yr))
    total_spend_raw = np.asarray(sim.calculate("consumption", yr))
    spend_rebase = P["ons_total_spend_yr"] / wmean(total_spend_raw, w)
    total_spend = total_spend_raw * spend_rebase   # FYE2022 price/level base
    # Uprating-shortfall base: benefit income under the 2022-23 rules the
    # lagged uprating applied to, excluding the (non-uprated) winter fuel
    # payment.
    benefit_income = sum(
        np.asarray(sim.calculate(v, COL_YEAR, map_to="household"))
        for v in UPRATED_CASH if v != "winter_fuel_allowance")

    # --- Rebase energy to pre-crisis price levels (ASSUMPTION, disclosed) ---
    base_cap_fy = (P["cap_apr_2021"] + P["cap_oct_2021"]) / 2.0
    rebase = base_cap_fy / wmean(energy_raw, w)
    energy = energy_raw * rebase

    # --- Both paths on ONE basis: financial-year mean caps ------------------
    counter_fy = (P["cap_apr_2022"] + P["cap_oct_2022"]) / 2.0   # no EPG
    realised_fy = (P["cap_apr_2022"] + P["epg_level"]) / 2.0     # EPG in force
    counter_rise = counter_fy / base_cap_fy - 1.0
    realised_rise = realised_fy / base_cap_fy - 1.0

    dE_counter = energy * counter_rise
    dE_realised = energy * realised_rise
    epg_cushion = dE_counter - dE_realised

    # --- MEASURED automatic response ---------------------------------------
    shocked = load_sim()
    for var, raw in (("electricity_consumption", elec_raw),
                     ("gas_consumption", gas_raw),
                     ("domestic_energy_consumption", energy_raw)):
        if var in shocked.tax_benefit_system.variables:
            shocked.set_input(var, yr, raw * rebase * (1.0 + realised_rise))

    automatic, auto_change = {}, 0.0
    for v in UPRATED_CASH:
        base_v = weighted(hh(v), w)
        shock_v = weighted(hh(v, shocked), w)
        automatic[v] = {
            "baseline_gbp_bn": round(base_v / 1e9, 3),
            "shocked_gbp_bn": round(shock_v / 1e9, 3),
            "change_gbp_bn": round((shock_v - base_v) / 1e9, 4),
        }
        auto_change += (shock_v - base_v) / 1e9

    # --- The model's own instruments respond to the shocked input ----------
    native = {}
    for v in NATIVE_INSTRUMENTS:
        base_v = weighted(hh(v), w)  # hh() raises if the variable is absent
        shock_v = weighted(hh(v, shocked), w)
        native[v] = {
            "baseline_gbp_bn": round(base_v / 1e9, 3),
            "shocked_gbp_bn": round(shock_v / 1e9, 3),
            "change_gbp_bn": round((shock_v - base_v) / 1e9, 3),
        }

    # --- Discretionary instruments -----------------------------------------
    ebss = np.full_like(energy, P["ebss_per_household"])
    means_tested_amt = sum(hh(v) for v in MEANS_TESTED)
    col = sum(col_components(sim))
    ctr = ctr_rebate(sim)
    energy_specific = epg_cushion + ebss + ctr     # legislated against energy
    discretionary = energy_specific + col
    discretionary_no_ctr = epg_cushion + ebss + col
    energy_specific_no_ctr = epg_cushion + ebss

    gap = P["cpi_sep_2022_yoy"] - P["uprating_applied_apr_2022"]
    shortfall = benefit_income * gap

    G = weighted(dE_counter, w)
    res = {
        "meta": {
            "dataset": provenance(sim)["default_dataset"],
            "provenance": provenance(sim),
            "policyengine_uk_version": _pkg_version("policyengine-uk"),
            "policyengine_version": _pkg_version("policyengine"),
            "sim_year": yr,
            "records": int(len(w)),
            "weighted_persons_m": round(persons_m, 1),
            "weighted_households_m": round(float(np.sum(w)) / 1e6, 2),
            "ons_uk_households_m": P["ons_uk_households_m"],
            "weight_excess_pct": round(
                100 * (float(np.sum(w)) / 1e6 / P["ons_uk_households_m"] - 1), 1),
            "energy_rebase_factor": round(rebase, 4),
            "spend_rebase_factor": round(spend_rebase, 4),
            "energy_mean_before_rebase": round(wmean(energy_raw, w), 1),
            "energy_mean_after_rebase": round(wmean(energy, w), 1),
        },
        "parameters": P,
        "shock": {
            "basis": "financial-year mean caps throughout",
            "base_fy_mean_cap": base_cap_fy,
            "counterfactual_fy_mean_cap": counter_fy,
            "realised_fy_mean_cap": realised_fy,
            "counterfactual_rise": round(counter_rise, 4),
            "realised_rise": round(realised_rise, 4),
            "counterfactual_aggregate_gbp_bn": round(G / 1e9, 2),
            "realised_aggregate_gbp_bn": round(weighted(dE_realised, w) / 1e9, 2),
        },
        "automatic_response": {
            "measured_change_gbp_bn": round(auto_change, 3),
            "measured_rate_pct": round(100 * auto_change * 1e9 / G, 2),
            "by_instrument": automatic,
            "native_instruments": native,
            "note": "calculator re-run with shocked energy inputs. The fourteen "
                    "cash benefits condition on nominal income, which the price "
                    "shock does not move, so none responds. The model's own "
                    "Energy Price Guarantee variable (epg_subsidy) reads energy "
                    "consumption and does respond; its change is reported "
                    "under native_instruments and classified as discretionary "
                    "in the paper's taxonomy.",
        },
        "uprating_shortfall": {
            "benefit_income_base_gbp_bn": round(weighted(benefit_income, w) / 1e9, 1),
            "uprating_gap_pp": round(100 * gap, 1),
            "shortfall_gbp_bn": round(weighted(shortfall, w) / 1e9, 2),
            "shortfall_pct_of_benefit_income": round(100 * gap, 1),
            "note": "real erosion of benefit income by ALL inflation over the "
                    "year; not attributable to the energy episode alone, and "
                    "deliberately not expressed as a share of the energy shock",
        },
        "discretionary": {
            "epg_gbp_bn": round(weighted(epg_cushion, w) / 1e9, 2),
            "ebss_gbp_bn": round(weighted(ebss, w) / 1e9, 2),
            "ctr_gbp_bn": round(weighted(ctr, w) / 1e9, 2),
            "ctr_households_m": round(
                weighted((ctr > 0).astype(float), w) / 1e6, 2),
            "energy_specific_gbp_bn": round(weighted(energy_specific, w) / 1e9, 2),
            "energy_specific_rate_pct": round(
                100 * weighted(energy_specific, w) / G, 1),
            "col_payments_gbp_bn": round(weighted(col, w) / 1e9, 2),
            "total_gbp_bn": round(weighted(discretionary, w) / 1e9, 2),
            "rate_pct_of_counterfactual": round(100 * weighted(discretionary, w) / G, 1),
            # What the Council Tax Rebate adds: the same stack without it.
            "without_ctr": {
                "total_gbp_bn": round(weighted(discretionary_no_ctr, w) / 1e9, 2),
                "rate_pct_of_counterfactual": round(
                    100 * weighted(discretionary_no_ctr, w) / G, 1),
                "energy_specific_gbp_bn": round(
                    weighted(energy_specific_no_ctr, w) / 1e9, 2),
                "energy_specific_rate_pct": round(
                    100 * weighted(energy_specific_no_ctr, w) / G, 1),
            },
        },
    }

    # Robustness variants only (the central JSON is unchanged): what was
    # deflated, and the income-conditioned quantities the deflation moves.
    if DEFLATE:
        res["meta"]["inputs_deflated"] = True
        res["meta"]["deflation"] = getattr(sim, "_deflation", None)
    if DEFLATE or args.sim_year is not None:
        res["checks"] = {
            "equiv_income_mean": round(wmean(equiv_income, w), 0),
            "net_income_mean": round(wmean(net_income, w), 0),
            "means_tested_households_m": round(
                weighted((means_tested_amt > 0).astype(float), w) / 1e6, 2),
            "col_components_gbp_bn": [round(weighted(c, w) / 1e9, 2)
                                      for c in col_components(sim)],
        }

    # --- Equivalised deciles -----------------------------------------------
    order = np.argsort(equiv_income)
    cw = np.cumsum(w[order])
    dec = np.zeros(len(equiv_income), dtype=int)
    dec[order] = np.minimum((cw / cw[-1] * 10).astype(int), 9)

    rows, within = [], []
    for d in range(10):
        m = dec == d
        wd = w[m]
        rows.append({
            "decile": d + 1,
            "realised_gbp_per_year": round(wmean(dE_realised[m], wd), 1),
            "counterfactual_gbp_per_year": round(wmean(dE_counter[m], wd), 1),
            "discretionary_gbp_per_year": round(wmean(discretionary[m], wd), 1),
            "cushioning_rate_pct": round(
                100 * weighted(discretionary[m], wd) / weighted(dE_counter[m], wd), 1),
            "energy_specific_rate_pct": round(
                100 * weighted(energy_specific[m], wd) / weighted(dE_counter[m], wd), 1),
            "cushioning_rate_pct_without_ctr": round(
                100 * weighted(discretionary_no_ctr[m], wd)
                / weighted(dE_counter[m], wd), 1),
            "energy_specific_rate_pct_without_ctr": round(
                100 * weighted(energy_specific_no_ctr[m], wd)
                / weighted(dE_counter[m], wd), 1),
            "burden_pct_of_spending": round(
                100 * weighted(dE_realised[m], wd) / weighted(total_spend[m], wd), 2),
        })
        bs = dE_realised[m] / np.maximum(total_spend[m], 1)
        ok = np.isfinite(bs)
        p10, p50, p90 = (wquantile(bs[ok], wd[ok], q) for q in (0.10, 0.50, 0.90))
        within.append({
            "decile": d + 1,
            "p10_burden_pct_spending": round(100 * p10, 2),
            "p50_burden_pct_spending": round(100 * p50, 2),
            "p90_burden_pct_spending": round(100 * p90, 2),
            "p90_p10_ratio": round(p90 / p10, 1) if p10 > 0 else None,
        })
    res["by_decile"] = rows
    res["within_decile_dispersion"] = within

    # --- Variance decomposition on four bases -------------------------------
    spend_share = dE_realised / np.maximum(total_spend, 1)
    b_sp, wb_sp = variance_split(spend_share, w, dec)
    lo, hi = wquantile(spend_share, w, 0.01), wquantile(spend_share, w, 0.99)
    trim = (spend_share >= lo) & (spend_share <= hi)
    b_tr, wb_tr = variance_split(spend_share[trim], w[trim], dec[trim])
    b_lg, wb_lg = variance_split(np.log(np.maximum(spend_share, 1e-6)), w, dec)
    fin = net_income > 0
    inc_share = dE_realised[fin] / net_income[fin]
    b_in, wb_in = variance_split(inc_share, w[fin], dec[fin])

    res["variance_decomposition"] = {
        "preferred_basis": "share of household expenditure",
        "expenditure_share": {"between_pct": b_sp, "within_pct": wb_sp},
        "expenditure_share_trimmed_1_99": {"between_pct": b_tr, "within_pct": wb_tr},
        "expenditure_share_log": {"between_pct": b_lg, "within_pct": wb_lg},
        "income_share_untrimmed": {
            "between_pct": b_in, "within_pct": wb_in,
            "caveat": "ratio with income in the denominator; dominated by very "
                      "low incomes and reported only for comparison"},
        "caveat": "computed on IMPUTED consumption (QRF from LCFS donors), so part "
                  "of the within-decile dispersion is donor-model variation; this "
                  "is a bound, not a measurement of true exposure dispersion",
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=float))
    s, a = res["shock"], res["automatic_response"]
    u, dsc = res["uprating_shortfall"], res["discretionary"]
    print(f"basis: FY mean caps | counterfactual +{100*s['counterfactual_rise']:.1f}% "
          f"(GBP {s['counterfactual_aggregate_gbp_bn']}bn), realised "
          f"+{100*s['realised_rise']:.1f}% (GBP {s['realised_aggregate_gbp_bn']}bn)")
    print(f"automatic response MEASURED: GBP {a['measured_change_gbp_bn']}bn "
          f"= {a['measured_rate_pct']}% of the shock")
    print(f"uprating shortfall GBP {u['shortfall_gbp_bn']}bn "
          f"= {u['shortfall_pct_of_benefit_income']}% of benefit income")
    print(f"discretionary GBP {dsc['total_gbp_bn']}bn = "
          f"{dsc['rate_pct_of_counterfactual']}% of the counterfactual shock")
    v = res["variance_decomposition"]
    print(f"variance (expenditure share): {v['expenditure_share']['between_pct']}/"
          f"{v['expenditure_share']['within_pct']} | trimmed "
          f"{v['expenditure_share_trimmed_1_99']['between_pct']}/"
          f"{v['expenditure_share_trimmed_1_99']['within_pct']} | log "
          f"{v['expenditure_share_log']['between_pct']}/"
          f"{v['expenditure_share_log']['within_pct']}")
    print("wrote", out)


if __name__ == "__main__":
    main()


def emit_macros(res, path=None):
    path = path or RESULTS_DIR / "generated_secondstage.tex"
    """LaTeX macros for the manuscript, prefixed Ss."""
    m = ["% generated by second_stage_energy.py -- do not edit by hand."]

    def add(k, v):
        m.append(f"\\newcommand{{\\Ss{k}}}{{{v}}}")

    s, a = res["shock"], res["automatic_response"]
    u, d, v = res["uprating_shortfall"], res["discretionary"], res["variance_decomposition"]
    mt = res["meta"]
    add("CounterRisePct", f"{100*s['counterfactual_rise']:.1f}")
    add("RealisedRisePct", f"{100*s['realised_rise']:.1f}")
    add("CounterAggBn", f"{s['counterfactual_aggregate_gbp_bn']:,.1f}")
    add("RealisedAggBn", f"{s['realised_aggregate_gbp_bn']:,.1f}")
    add("CounterFyCap", f"{s['counterfactual_fy_mean_cap']:,.0f}")
    add("RealisedFyCap", f"{s['realised_fy_mean_cap']:,.0f}")
    add("BaseFyCap", f"{s['base_fy_mean_cap']:,.0f}")
    add("AutomaticBn", f"{a['measured_change_gbp_bn']:,.2f}")
    add("AutomaticRatePct", f"{a['measured_rate_pct']:.1f}")
    nat = a.get("native_instruments", {})
    if "epg_subsidy" in nat:
        add("NativeEpgBn", f"{nat['epg_subsidy']['baseline_gbp_bn']:,.1f}")
        add("NativeEpgDeltaBn", f"{nat['epg_subsidy']['change_gbp_bn']:,.1f}")
    if "cost_of_living_support_payment" in nat:
        add("NativeColBn",
            f"{nat['cost_of_living_support_payment']['baseline_gbp_bn']:,.1f}")
    if "energy_bills_rebate" in nat:
        add("NativeEbrBn", f"{nat['energy_bills_rebate']['baseline_gbp_bn']:,.1f}")
    if "household_benefits" in nat:
        add("HouseholdBenefitsDeltaBn",
            f"{nat['household_benefits']['change_gbp_bn']:,.1f}")
    add("EbssOutturnBn", f"{P['ebss_outturn_gbp_bn']:,.1f}")
    add("EbssExcessPct",
        f"{100 * (d['ebss_gbp_bn'] / P['ebss_outturn_gbp_bn'] - 1):.0f}")
    add("BenefitIncomeBn", f"{u['benefit_income_base_gbp_bn']:,.0f}")
    add("UpratingGapPp", f"{u['uprating_gap_pp']:.1f}")
    add("UpratingShortfallBn", f"{u['shortfall_gbp_bn']:,.1f}")
    add("UpratingShortfallPctOfBenefitIncome", f"{u['shortfall_pct_of_benefit_income']:.1f}")
    add("EpgBn", f"{d['epg_gbp_bn']:,.1f}")
    add("EbssBn", f"{d['ebss_gbp_bn']:,.1f}")
    add("ColBn", f"{d['col_payments_gbp_bn']:,.1f}")
    # Council Tax Rebate (in the central stack, energy-specific) and the
    # same headline numbers without it.
    nc = d["without_ctr"]
    add("CtrBn", f"{d['ctr_gbp_bn']:,.1f}")
    add("CtrHouseholdsM", f"{d['ctr_households_m']:.1f}")
    add("EnergySpecificBn", f"{d['energy_specific_gbp_bn']:,.1f}")
    add("EnergySpecificRatePct", f"{d['energy_specific_rate_pct']:.1f}")
    add("NoCtrDiscretionaryBn", f"{nc['total_gbp_bn']:,.1f}")
    add("NoCtrDiscretionaryRatePct", f"{nc['rate_pct_of_counterfactual']:.1f}")
    add("NoCtrEnergySpecificBn", f"{nc['energy_specific_gbp_bn']:,.1f}")
    add("NoCtrEnergySpecificRatePct", f"{nc['energy_specific_rate_pct']:.1f}")
    add("CtrRateAddPp", f"{d['rate_pct_of_counterfactual'] - nc['rate_pct_of_counterfactual']:.1f}")
    add("DiscretionaryBn", f"{d['total_gbp_bn']:,.1f}")
    add("DiscretionaryRatePct", f"{d['rate_pct_of_counterfactual']:.1f}")
    add("BetweenPct", f"{v['expenditure_share_trimmed_1_99']['between_pct']:.1f}")
    add("WithinPct", f"{v['expenditure_share_trimmed_1_99']['within_pct']:.1f}")
    add("BetweenPctUntrimmed", f"{v['expenditure_share']['between_pct']:.1f}")
    add("WithinPctUntrimmed", f"{v['expenditure_share']['within_pct']:.1f}")
    add("BetweenPctLog", f"{v['expenditure_share_log']['between_pct']:.1f}")
    add("WithinPctLog", f"{v['expenditure_share_log']['within_pct']:.1f}")
    add("HouseholdsRecords", f"{mt['records']:,}")
    add("WeightedHouseholdsM", f"{mt['weighted_households_m']:.1f}")
    add("OnsHouseholdsM", f"{mt['ons_uk_households_m']:.1f}")
    add("WeightExcessPct", f"{mt['weight_excess_pct']:.1f}")
    add("RebaseFactor", f"{mt['energy_rebase_factor']:.3f}")
    add("SpendRebase", f"{mt['spend_rebase_factor']:.3f}")
    add("PersonsM", f"{res['meta']['weighted_persons_m']:.1f}")
    add("PolicyEngineVersion", mt["policyengine_uk_version"])
    words = ("One","Two","Three","Four","Five","Six","Seven","Eight","Nine","Ten")
    for row, wd in zip(res["by_decile"], words):
        add(f"CushionPctDec{wd}", f"{row['cushioning_rate_pct']:.0f}")
        add(f"EnergySpecificPctDec{wd}", f"{row['energy_specific_rate_pct']:.0f}")
        add(f"NoCtrCushionPctDec{wd}",
            f"{row['cushioning_rate_pct_without_ctr']:.0f}")
        add(f"NoCtrEnergySpecificPctDec{wd}",
            f"{row['energy_specific_rate_pct_without_ctr']:.0f}")
        add(f"BurdenPctSpendDec{wd}", f"{row['burden_pct_of_spending']:.2f}")
    for row, wd in zip(res["within_decile_dispersion"], words):
        add(f"PTenDec{wd}", f"{row['p10_burden_pct_spending']:.2f}")
        add(f"PNinetyDec{wd}", f"{row['p90_burden_pct_spending']:.2f}")
        add(f"RatioDec{wd}", f"{row['p90_p10_ratio']:.1f}")
    Path(path).write_text("\n".join(m) + "\n")
    print("wrote", path)


if __name__ == "__main__":
    emit_macros(json.loads((RESULTS_DIR / "second_stage_energy.json").read_text()))
    import shutil
    shutil.copy(RESULTS_DIR / "generated_secondstage.tex",
                PAPER_DIR / "generated_secondstage.tex")
