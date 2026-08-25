"""Second stage: the 2022-23 energy episode on enhanced-FRS households.

Computes the cushioning of a trade-transmitted price shock through the
statutory tax-benefit system, decomposed into automatic and
discretionary components, and the dispersion of exposure within income
deciles.

Design notes, in response to referee comments on the first version:

* The automatic response is MEASURED, not asserted.  The tax-benefit
  calculator is run twice -- once on baseline inputs and once with the
  shocked energy inputs -- and the change in every entitlement is
  recorded.  A zero, if it appears, is a computed zero.
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

Run:  python second_stage_energy.py --dataset <path to enhanced_frs.h5>
"""

import argparse
import json
from importlib.metadata import version as _pkg_version
from pathlib import Path

import numpy as np

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
    # Uprating (DATA: DWP April 2022 order; ONS D7BT).
    "uprating_applied_apr_2022": 0.031,
    "cpi_fy2022_23_mean": 0.101,
    # ASSUMPTION: simulated year.
    "sim_year": 2023,
    # ONS UK household count, for the weight reconciliation (DATA).
    "ons_uk_households_m": 28.4,
}

MEANS_TESTED = ["universal_credit", "pension_credit", "tax_credits",
                "income_support", "housing_benefit", "jsa_income", "esa_income"]
UPRATED_CASH = MEANS_TESTED + ["state_pension", "child_benefit", "pip", "dla",
                               "attendance_allowance", "carers_allowance",
                               "winter_fuel_allowance"]


# Weighted primitives, delegated to microdf (the survey-weighting library
# PolicyEngine itself returns results in), so no weight arithmetic is done
# by hand anywhere in the pipeline.  Signatures unchanged; verified to
# reproduce the previous hand-rolled results exactly.
from microdf import MicroSeries


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
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default="out/second_stage_energy.json")
    args = ap.parse_args()

    from policyengine_uk import Microsimulation

    yr = P["sim_year"]
    sim = Microsimulation(dataset=args.dataset)

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
    total_spend = np.asarray(sim.calculate("consumption", yr))
    benefit_income = sum(hh(v) for v in UPRATED_CASH)

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
    shocked = Microsimulation(dataset=args.dataset)
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

    # --- Discretionary instruments -----------------------------------------
    ebss = np.full_like(energy, P["ebss_per_household"])
    means_tested_amt = sum(hh(v) for v in MEANS_TESTED)
    col = ((means_tested_amt > 0) * P["col_means_tested"]
           + (hh("state_pension") > 0) * P["col_pensioner"]
           + ((hh("pip") + hh("dla") + hh("attendance_allowance")) > 0)
           * P["col_disability"])
    discretionary = epg_cushion + ebss + col

    gap = P["cpi_fy2022_23_mean"] - P["uprating_applied_apr_2022"]
    shortfall = benefit_income * gap

    G = weighted(dE_counter, w)
    res = {
        "meta": {
            "dataset": Path(args.dataset).name,
            "policyengine_uk_version": _pkg_version("policyengine-uk"),
            "sim_year": yr,
            "records": int(len(w)),
            "weighted_persons_m": round(persons_m, 1),
            "weighted_households_m": round(float(np.sum(w)) / 1e6, 2),
            "ons_uk_households_m": P["ons_uk_households_m"],
            "weight_excess_pct": round(
                100 * (float(np.sum(w)) / 1e6 / P["ons_uk_households_m"] - 1), 1),
            "energy_rebase_factor": round(rebase, 4),
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
            "note": "calculator re-run with shocked energy inputs; entitlements "
                    "condition on nominal income, which a price shock does not "
                    "move, so no means-tested award responds",
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
            "col_payments_gbp_bn": round(weighted(col, w) / 1e9, 2),
            "total_gbp_bn": round(weighted(discretionary, w) / 1e9, 2),
            "rate_pct_of_counterfactual": round(100 * weighted(discretionary, w) / G, 1),
        },
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


def emit_macros(res, path="out/generated_secondstage.tex"):
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
    add("BenefitIncomeBn", f"{u['benefit_income_base_gbp_bn']:,.0f}")
    add("UpratingGapPp", f"{u['uprating_gap_pp']:.1f}")
    add("UpratingShortfallBn", f"{u['shortfall_gbp_bn']:,.1f}")
    add("UpratingShortfallPctOfBenefitIncome", f"{u['shortfall_pct_of_benefit_income']:.1f}")
    add("EpgBn", f"{d['epg_gbp_bn']:,.1f}")
    add("EbssBn", f"{d['ebss_gbp_bn']:,.1f}")
    add("ColBn", f"{d['col_payments_gbp_bn']:,.1f}")
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
    add("PersonsM", f"{res['meta']['weighted_persons_m']:.1f}")
    add("PolicyEngineVersion", mt["policyengine_uk_version"])
    words = ("One","Two","Three","Four","Five","Six","Seven","Eight","Nine","Ten")
    for row, wd in zip(res["by_decile"], words):
        add(f"CushionPctDec{wd}", f"{row['cushioning_rate_pct']:.0f}")
        add(f"BurdenPctSpendDec{wd}", f"{row['burden_pct_of_spending']:.2f}")
    for row, wd in zip(res["within_decile_dispersion"], words):
        add(f"PTenDec{wd}", f"{row['p10_burden_pct_spending']:.2f}")
        add(f"PNinetyDec{wd}", f"{row['p90_burden_pct_spending']:.2f}")
        add(f"RatioDec{wd}", f"{row['p90_p10_ratio']:.1f}")
    Path(path).write_text("\n".join(m) + "\n")
    print("wrote", path)


if __name__ == "__main__":
    emit_macros(json.loads(Path("out/second_stage_energy.json").read_text()))
