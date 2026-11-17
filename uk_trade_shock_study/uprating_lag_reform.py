"""The April-2022 uprating lag priced as a parameters-only counterfactual.

Benefits were uprated in April 2022 by 3.1 per cent (September 2021 CPI)
while prices rose 10.1 per cent over the year to September 2022.  The
vintage run (vintage_and_tca.py) prices the withheld 7.0 points by scaling
the whole 2022-23 -> 2023-24 rulebook difference, which bundles every
parameter change between the two years.  This script isolates the lag:

* same households, 2022-23 rules in both runs (inputs copied into 2022);
* counterfactual: every benefit amount that the April 2022 order raised by
  3.1 per cent is set, for 2022 only, to the value a 10.1 per cent rise
  would have given -- its 2022-23 value times 1.101/1.031, which is its
  2021-22 value times 1.101 up to the order's rounding (and, for the
  Universal Credit standard allowance, excludes the temporary GBP 20
  uplift that sat in the 2021-22 parameter);
* everything else (tapers, thresholds not in the order, taxes, the
  cost-of-living payments) stays at 2022-23 law;
* cost = change in the aggregate of the cash benefits, by benefit.

The parameter list is built from the model, not typed in: every currency
parameter under gov.dwp and gov.hmrc.child_benefit whose 2022-23 value is
its 2021-22 value raised by 3.1 per cent (ratio within rounding tolerance),
plus the four Universal Credit standard allowances (whose 2021-22 value
carries the uplift, so the ratio test cannot see their 3.1 per cent rise).
The model's own uprating metadata (gov.benefit_uprating_cpi) is used as a
cross-check and the comparison is reported.  Parameters that rose 3.1 per
cent but are earnings rules or charge schedules rather than benefit amounts
are excluded by name and listed.

Run:  python uprating_lag_reform.py [--deflate]
Emits results/uprating_lag_reform.json and results/generated_lagreform.tex
(copied to paper/).
"""

import argparse
import json
import shutil

import numpy as np

from second_stage_energy import (P, PAPER_DIR, RESULTS_DIR, UPRATED_CASH,
                                 load_sim, weighted)

RULE_YEAR = 2022
APPLIED = P["uprating_applied_apr_2022"]      # 3.1%
COUNTERFACTUAL = P["cpi_sep_2022_yoy"]        # 10.1%
FACTOR = (1 + COUNTERFACTUAL) / (1 + APPLIED)
RATIO_BAND = (1.024, 1.038)                   # 3.1% up to weekly-rate rounding
ROOTS = ("gov.dwp.", "gov.hmrc.child_benefit.")
UC_STANDARD_ALLOWANCE = "gov.dwp.universal_credit.standard_allowance.amount."
# Rose 3.1 per cent but are not benefit amounts.
EXCLUDED = ("gov.dwp.carers_allowance.earnings_limit",       # earnings rule
            "gov.dwp.housing_benefit.ineligible_charges.",   # meal charges
            ".threshold")                                    # income bands
PENSIONER = ["state_pension", "pension_credit", "attendance_allowance"]
# Components the model carries as reported amounts rather than as rules, so
# a parameter reform cannot move them.  They are never folded silently into
# the rules-priced figure: where the reform leaves one unchanged, a
# separately labelled mechanical add-on (its 2022-23 aggregate times the
# same 1.101/1.031 - 1) is reported beside it.
REPORTED_HELD = ["esa_income", "jsa_income", "additional_state_pension"]
# Additional State Pension (SERPS/S2P and protected payments) is uprated by
# September CPI through this index rather than through a rate parameter
# (it rose 3.1 per cent in April 2022 with everything else).  Its 2022
# value is raised by the same factor; no other variable reads the index.
SEPTEMBER_CPI_INDEX = "gov.economic_assumptions.indices.september_cpi_uprating"
STATE_PENSION_PARTS = ["basic_state_pension", "new_state_pension",
                       "additional_state_pension"]


def lag_parameters():
    """(selected, excluded, metadata_only) parameter lists from the model."""
    from policyengine_core.parameters import Parameter
    from policyengine_uk.system import system

    def uprated_by(p):
        u = p.metadata.get("uprating")
        return u if isinstance(u, str) else (u or {}).get("parameter") \
            if isinstance(u, dict) else None

    selected, excluded, metadata_only, seen = {}, [], [], set()
    for p in system.parameters.gov.get_descendants():
        if not isinstance(p, Parameter) or p.name in seen:
            continue
        seen.add(p.name)
        if not p.name.startswith(ROOTS) or p.metadata.get("unit") != "currency-GBP":
            continue
        try:
            v21, v22, jan = (p(f"{RULE_YEAR - 1}-12-31"), p(f"{RULE_YEAR}-12-31"),
                             p(f"{RULE_YEAR}-01-01"))
        except Exception:  # noqa: BLE001 - parameter undefined in these years
            continue
        if not all(isinstance(x, (int, float)) and not isinstance(x, bool)
                   for x in (v21, v22, jan)) or not v21:
            continue
        is_sa = p.name.startswith(UC_STANDARD_ALLOWANCE)
        in_band = RATIO_BAND[0] <= v22 / v21 <= RATIO_BAND[1]
        flagged = uprated_by(p) == "gov.benefit_uprating_cpi"
        if not (in_band or is_sa):
            if flagged:
                metadata_only.append({"parameter": p.name, "value_2021": v21,
                                      "value_2022": v22})
            continue
        if any(p.name == e or (e.endswith(".") and p.name.startswith(e))
               or (e.startswith(".") and p.name.endswith(e)) for e in EXCLUDED):
            excluded.append(p.name)
            continue
        if jan != v22:
            raise ValueError(f"{p.name} changes within {RULE_YEAR}; the "
                             "calendar-year reform would misdate it")
        selected[p.name] = {"value_2021": v21, "value_2022": v22,
                            "counterfactual_2022": v22 * FACTOR,
                            "model_metadata_cpi_uprated": flagged,
                            "selected_by": "uc_standard_allowance" if is_sa
                            else "3.1% ratio"}
    return selected, excluded, metadata_only


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deflate", action="store_true",
                    help="sensitivity: inputs deflated to 2022-23 levels")
    args = ap.parse_args()

    selected, excluded, metadata_only = lag_parameters()
    period = f"{RULE_YEAR}-01-01.{RULE_YEAR}-12-31"
    reform = {name: {period: v["counterfactual_2022"]}
              for name, v in selected.items()}
    from policyengine_uk.system import system
    index_2022 = float(system.parameters.get_child(SEPTEMBER_CPI_INDEX)(
        f"{RULE_YEAR}-06-01"))
    reform[SEPTEMBER_CPI_INDEX] = {period: index_2022 * FACTOR}

    def totals(reform_dict):
        sim = load_sim([RULE_YEAR], reform=reform_dict, deflate=args.deflate)
        w = np.asarray(sim.calculate("household_weight", RULE_YEAR))
        out = {v: weighted(np.asarray(sim.calculate(v, RULE_YEAR,
                                                    map_to="household")), w) / 1e9
               for v in UPRATED_CASH + STATE_PENSION_PARTS
               + ["household_benefits", "household_net_income"]}
        # The reform must have reached the rulebook the simulation uses.
        probe = "gov.dwp.universal_credit.standard_allowance.amount.SINGLE_OLD"
        out["_probe"] = float(sim.tax_benefit_system.parameters.get_child(probe)(
            f"{RULE_YEAR}-06-01"))
        return out

    base = totals(None)
    cf = totals(reform)
    probe_ratio = cf["_probe"] / base["_probe"]
    if abs(probe_ratio - FACTOR) > 1e-6:
        raise RuntimeError(f"reform not applied: probe ratio {probe_ratio}")

    by = {}
    for v in UPRATED_CASH:
        d = cf[v] - base[v]
        by[v] = {"baseline_gbp_bn": round(base[v], 3),
                 "counterfactual_gbp_bn": round(cf[v], 3),
                 "change_gbp_bn": round(d, 3),
                 "change_pct": round(100 * d / base[v], 2) if base[v] else None}
    total = sum(cf[v] - base[v] for v in UPRATED_CASH)
    base_total = sum(base[v] for v in UPRATED_CASH)
    sp = cf["state_pension"] - base["state_pension"]
    pens = sum(cf[v] - base[v] for v in PENSIONER)
    unmoved = [v for v in UPRATED_CASH
               if base[v] > 0.05 and abs(cf[v] - base[v]) < 1e-3]
    addon = {v: round(base[v] * (FACTOR - 1), 3) for v in REPORTED_HELD
             if abs(cf[v] - base[v]) < 1e-3 * max(base[v], 1.0)}
    addon_total = sum(addon.values())
    addon_sp = addon.get("additional_state_pension", 0.0)
    res = {
        "rule_year": RULE_YEAR,
        "inputs_deflated": bool(args.deflate),
        "applied_uprating_pct": 100 * APPLIED,
        "counterfactual_uprating_pct": 100 * COUNTERFACTUAL,
        "parameter_factor": round(FACTOR, 6),
        "parameters_changed": len(selected),
        "index_changed": {"parameter": SEPTEMBER_CPI_INDEX,
                          "value_2022": index_2022,
                          "counterfactual_2022": index_2022 * FACTOR,
                          "moves": "additional_state_pension"},
        "parameters": selected,
        "excluded_by_name": excluded,
        "model_cpi_metadata_but_not_3_1pct_in_2022": metadata_only,
        "selected_without_model_cpi_metadata": sorted(
            k for k, v in selected.items() if not v["model_metadata_cpi_uprated"]),
        "benefit_base_gbp_bn": round(base_total, 2),
        "lag_cost_gbp_bn": round(total, 2),
        "lag_cost_pct_of_base": round(100 * total / base_total, 2),
        "state_pension_gbp_bn": round(sp, 2),
        "non_state_pension_gbp_bn": round(total - sp, 2),
        "pensioner_benefits_gbp_bn": round(pens, 2),
        "working_age_and_other_gbp_bn": round(total - pens, 2),
        "household_benefits_change_gbp_bn": round(
            cf["household_benefits"] - base["household_benefits"], 2),
        "household_net_income_change_gbp_bn": round(
            cf["household_net_income"] - base["household_net_income"], 2),
        "benefits_that_did_not_move": unmoved,
        "state_pension_parts": {
            v: {"baseline_gbp_bn": round(base[v], 3),
                "change_gbp_bn": round(cf[v] - base[v], 3)}
            for v in STATE_PENSION_PARTS},
        "reported_amount_add_on": {
            "components_gbp_bn": addon,
            "total_gbp_bn": round(addon_total, 2),
            "lag_cost_with_add_on_gbp_bn": round(total + addon_total, 2),
            "state_pension_with_add_on_gbp_bn": round(sp + addon_sp, 2),
            "non_state_pension_with_add_on_gbp_bn": round(
                total - sp + addon_total - addon_sp, 2),
            "note": "not a rules result: reported-amount components the "
                    "parameter reform cannot reach, scaled mechanically by "
                    f"{100 * (FACTOR - 1):.2f}% with no taper or eligibility "
                    "response",
        },
        "by_benefit": by,
        "note": "pensioner = state pension + pension credit + attendance "
                "allowance; the remainder is working-age and child/disability "
                "benefits (housing benefit and disability benefits include "
                "some pensioner recipients).  The gross mechanical rise is "
                f"{100 * (FACTOR - 1):.2f}% of each uprated amount; "
                "means-tested awards rise by more than that in proportion "
                "because income is unchanged, and unmodelled-as-rules reported "
                "amounts do not move.",
    }
    suffix = "_deflated" if args.deflate else ""
    (RESULTS_DIR / f"uprating_lag_reform{suffix}.json").write_text(
        json.dumps(res, indent=1))
    for k in ("parameters_changed", "benefit_base_gbp_bn", "lag_cost_gbp_bn",
              "lag_cost_pct_of_base", "state_pension_gbp_bn",
              "non_state_pension_gbp_bn", "pensioner_benefits_gbp_bn",
              "working_age_and_other_gbp_bn", "household_benefits_change_gbp_bn",
              "household_net_income_change_gbp_bn", "benefits_that_did_not_move",
              "state_pension_parts", "reported_amount_add_on"):
        print(k, res[k])
    for v, r in by.items():
        print(f"  {v:24s} {r['baseline_gbp_bn']:8.2f} -> "
              f"{r['counterfactual_gbp_bn']:8.2f}  {r['change_pct']}%")
    emit()


def emit() -> None:
    """Macros from the central (and, if present, deflated) JSON."""
    central = RESULTS_DIR / "uprating_lag_reform.json"
    if not central.exists():
        return
    r = json.loads(central.read_text())
    m = ["% generated by uprating_lag_reform.py -- do not edit by hand."]

    def add(k, v):
        m.append(f"\\newcommand{{\\SsLagReform{k}}}{{{v}}}")

    add("Bn", f"{r['lag_cost_gbp_bn']:,.1f}")
    add("PctOfBase", f"{r['lag_cost_pct_of_base']:.1f}")
    add("BaseBn", f"{r['benefit_base_gbp_bn']:,.1f}")
    add("StatePensionBn", f"{r['state_pension_gbp_bn']:,.1f}")
    add("NonStatePensionBn", f"{r['non_state_pension_gbp_bn']:,.1f}")
    add("PensionerBn", f"{r['pensioner_benefits_gbp_bn']:,.1f}")
    add("WorkingAgeBn", f"{r['working_age_and_other_gbp_bn']:,.1f}")
    add("NetIncomeBn", f"{r['household_net_income_change_gbp_bn']:,.1f}")
    add("Parameters", f"{r['parameters_changed']}")
    add("ParameterFactorPct", f"{100 * (r['parameter_factor'] - 1):.1f}")
    by = r["by_benefit"]
    for key, tag in (("universal_credit", "UcBn"), ("pension_credit", "PcBn"),
                     ("child_benefit", "ChbBn"), ("housing_benefit", "HbBn"),
                     ("tax_credits", "TcBn")):
        add(tag, f"{round(by[key]['change_gbp_bn'], 1) + 0.0:,.1f}")
    add("DisabilityCarerBn", f"{sum(by[k]['change_gbp_bn'] for k in ('pip', 'dla', 'attendance_allowance', 'carers_allowance')):,.1f}")
    ao = r["reported_amount_add_on"]
    add("ReportedAddOnBn", f"{ao['total_gbp_bn']:,.1f}")
    add("WithReportedBn", f"{ao['lag_cost_with_add_on_gbp_bn']:,.1f}")
    add("WithReportedStatePensionBn", f"{ao['state_pension_with_add_on_gbp_bn']:,.1f}")
    add("WithReportedNonStatePensionBn",
        f"{ao['non_state_pension_with_add_on_gbp_bn']:,.1f}")
    vin = RESULTS_DIR / "generated_vintage.tex"
    if vin.exists():
        import re
        hit = re.search(r"\\SsVinLagCostBn\}\{([\d.,]+)\}", vin.read_text())
        if hit:
            old = float(hit.group(1).replace(",", ""))
            add("VsVintagePct", f"{100 * (r['lag_cost_gbp_bn'] / old - 1):.0f}")
            add("WithReportedVsVintagePct",
                f"{100 * (ao['lag_cost_with_add_on_gbp_bn'] / old - 1):.0f}")
    defl = RESULTS_DIR / "uprating_lag_reform_deflated.json"
    if defl.exists():
        d = json.loads(defl.read_text())
        add("DeflBn", f"{d['lag_cost_gbp_bn']:,.1f}")
        add("DeflWithReportedBn",
            f"{d['reported_amount_add_on']['lag_cost_with_add_on_gbp_bn']:,.1f}")
        add("DeflBaseBn", f"{d['benefit_base_gbp_bn']:,.1f}")
    path = RESULTS_DIR / "generated_lagreform.tex"
    path.write_text("\n".join(m) + "\n")
    shutil.copy(path, PAPER_DIR / "generated_lagreform.tex")
    print("wrote", path)


if __name__ == "__main__":
    main()
