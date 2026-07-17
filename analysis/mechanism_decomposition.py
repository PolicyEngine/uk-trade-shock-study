"""Demonstrate the cushioning mechanism instead of asserting it.

Three exercises, all on the full_tariff schedule, period 2026:

1. ZERO-UC DECOMPOSITION (seed 0, stability over seeds 0-4): among displaced
   workers post-shock, who gets zero Universal Credit and which constraint
   binds — the £16k capital limit, the joint income taper (partner earnings /
   unearned income exhausting the maximum award), modelled non-take-up, or
   none (positive UC).

2. COMPONENT DECOMPOSITION of the cushioning rate for both margins
   (displacement seed 0 and the deterministic wage cut): the cushioned share
   of the gross earnings loss split into income tax, employee NI, UC,
   other benefits (JSA/ESA/HB/tax credits/pension credit/...), and a residual
   (council tax, pension contributions, statutory pay zeroed in transition,
   student loans, ...). Components sum to the cushioning rate by construction
   because HBAI net income is an exact add/subtract list.

3. CAPITAL-RULES COUNTERFACTUAL: rerun displacement (seed 0) under reforms
   that neutralise UC capital rules — (a) the £16k hard limit only,
   (b) the limit plus tariff income from capital (£6k-£16k) — with taper and
   work allowances untouched. The paper's causal claim predicts the
   displacement cushioning rate rises toward the wage-cut 41%.

Writes results/mechanism_seed0.json, results/cushioning_decomposition.json,
results/mechanism_capital_reform.json.

Usage: .venv/bin/python analysis/mechanism_decomposition.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from policyengine_uk.variables.household.income.hbai_household_net_income import (
    HBAI_HOUSEHOLD_NET_INCOME_ADDS,
    HBAI_HOUSEHOLD_NET_INCOME_SUBTRACTS,
)
from uk_trade_shock_study.runner import _baseline_and_persons
from uk_trade_shock_study.shocks import (
    PRESETS,
    TRANSITION_ZEROED_VARIABLES,
    apply_shocks,
)

PERIOD = 2026
DATASET = Path("data/frs_2024_25.h5")
RESULTS = Path("results")
FAR = "2013-01-01.2100-12-31"
BIG = 1_000_000_000

#: Reform A: remove the £16k eligibility cliff only; tariff income (the
#: £6k-£16k taper) still accrues, work allowance/55% taper untouched.
REFORM_LIMIT_OFF = {
    "gov.dwp.universal_credit.means_test.capital.limit": {FAR: BIG},
}
#: Reform B: all capital rules off (no cliff, no tariff income).
REFORM_CAPITAL_OFF = {
    "gov.dwp.universal_credit.means_test.capital.limit": {FAR: BIG},
    "gov.dwp.universal_credit.means_test.capital.tariff_income.threshold": {FAR: BIG},
}

TAX_COMPONENTS = ("income_tax", "national_insurance")
UC_COMPONENT = "universal_credit"
#: Benefit variables in the HBAI add list other than UC.
OTHER_BENEFITS = tuple(
    v
    for v in HBAI_HOUSEHOLD_NET_INCOME_ADDS
    if v
    not in (
        "employment_income",
        "self_employment_income",
        "savings_interest_income",
        "dividend_income",
        "miscellaneous_income",
        "property_income",
        "private_pension_income",
        "private_transfer_income",
        "maintenance_income",
        UC_COMPONENT,
        # statutory pay is zeroed as part of the displacement transition
        # (part of the shock, not a cushion) -> residual
        "statutory_sick_pay",
        "statutory_maternity_pay",
    )
)


def build_shocked_sim(dataset, baseline_sim, shocked_table, period, reform=None):
    """build_shocked_simulation with an optional parameter reform (same
    contract, including the hard-error transition check)."""
    from policyengine_uk import Microsimulation

    sim = Microsimulation(dataset=dataset, reform=reform)
    sim.set_input(
        "employment_income", period, shocked_table["employment_income"].to_numpy(float)
    )
    displaced = shocked_table["displaced"].to_numpy()
    for var in TRANSITION_ZEROED_VARIABLES:
        values = baseline_sim.calculate(var, period=period, map_to="person").values.astype(float)
        values[displaced] = 0.0
        sim.set_input(var, period, values)
    status = baseline_sim.calculate(
        "employment_status", period=period, map_to="person"
    ).values.astype(object)
    status[displaced] = "UNEMPLOYED"
    sim.set_input("employment_status", period, status)
    applied = sim.calculate("employment_status", period=period, map_to="person").values.astype(str)
    if not (applied[displaced] == "UNEMPLOYED").all():
        raise RuntimeError("employment_status transition not applied")
    return sim


def person_benunit_vars(sim, period):
    """Benunit-level UC machinery mapped to persons."""
    out = {}
    for var in (
        "universal_credit",
        "uc_maximum_amount",
        "uc_assessable_capital",
        "uc_earned_income",
        "uc_unearned_income",
        "would_claim_uc",
        "is_uc_eligible",
        "jsa_contrib",
        "jsa_income",
    ):
        out[var] = sim.calculate(var, period=period, map_to="person").values.astype(float)
    return out


def zero_uc_decomposition(sim, displaced, weight, period, taper_rate=0.55, limit=16_000.0):
    """Weighted shares of displaced workers by binding UC constraint."""
    v = person_benunit_vars(sim, period)
    w = weight[displaced]
    tot = float(w.sum())
    uc = v["universal_credit"][displaced]
    cap = v["uc_assessable_capital"][displaced]
    claim = v["would_claim_uc"][displaced] > 0
    ucmax = v["uc_maximum_amount"][displaced]
    # counterfactual reduction ignoring the min() clamp at uc_maximum_amount
    reduction = taper_rate * v["uc_earned_income"][displaced] + v["uc_unearned_income"][displaced]

    positive = uc > 0
    zero = ~positive
    cap_binding = zero & (cap > limit)
    # would the joint income taper alone (partner earnings + unearned income)
    # exhaust the maximum award? evaluable only where a maximum exists;
    # for capital-disqualified benunits uc_maximum_amount is zeroed by
    # eligibility, so income-tapered is measured on the capital-eligible zero-UC set
    income_tapered = zero & ~cap_binding & claim & (ucmax > 0) & (reduction >= ucmax)
    non_takeup = zero & ~cap_binding & ~claim
    other_zero = zero & ~cap_binding & ~income_tapered & ~non_takeup

    def share(mask):
        return float(w[mask].sum() / tot) if tot else 0.0

    return {
        "displaced_weighted": tot,
        "share_positive_uc": share(positive),
        "share_zero_uc": share(zero),
        "share_zero_uc_capital_over_16k": share(cap_binding),
        "share_zero_uc_income_tapered": share(income_tapered),
        "share_zero_uc_non_takeup": share(non_takeup),
        "share_zero_uc_other": share(other_zero),
        "mean_uc_award_displaced": float(np.average(uc, weights=w)) if tot else 0.0,
        "mean_uc_award_if_positive": (
            float(np.average(uc[positive], weights=w[positive])) if positive.any() else 0.0
        ),
        "mean_jsa_contrib_displaced": float(
            np.average(v["jsa_contrib"][displaced], weights=w)
        ),
        "mean_jsa_income_displaced": float(
            np.average(v["jsa_income"][displaced], weights=w)
        ),
    }


def hh_totals(sim, period, variables):
    hh_w = sim.calculate("household_weight", period=period, map_to="household").values
    return {
        v: float(
            (sim.calculate(v, period=period, map_to="household").values * hh_w).sum()
        )
        for v in variables
    }


def cushioning_components(baseline, shocked, persons, shocked_table, period):
    """Split the cushioning rate into named components of HBAI net income."""
    w = persons["weight"].to_numpy(float)
    gross_loss = float(
        (
            (persons["employment_income"].to_numpy(float) - shocked_table["employment_income"].to_numpy(float))
            * w
        ).sum()
    )
    variables = ("hbai_household_net_income",) + TAX_COMPONENTS + (UC_COMPONENT,) + OTHER_BENEFITS
    base = hh_totals(baseline, period, variables)
    shock = hh_totals(shocked, period, variables)
    net_loss = base["hbai_household_net_income"] - shock["hbai_household_net_income"]
    c_it = base["income_tax"] - shock["income_tax"]
    c_ni = base["national_insurance"] - shock["national_insurance"]
    c_uc = shock[UC_COMPONENT] - base[UC_COMPONENT]
    c_ob = sum(shock[v] - base[v] for v in OTHER_BENEFITS)
    cushion = gross_loss - net_loss
    c_other = cushion - (c_it + c_ni + c_uc + c_ob)
    return {
        "gross_earnings_loss": gross_loss,
        "net_disposable_loss": net_loss,
        "cushioning_rate": cushion / gross_loss,
        "components_share_of_gross_loss": {
            "income_tax": c_it / gross_loss,
            "employee_national_insurance": c_ni / gross_loss,
            "universal_credit": c_uc / gross_loss,
            "other_benefits": c_ob / gross_loss,
            "other_residual": c_other / gross_loss,
        },
        "components_gbp": {
            "income_tax": c_it,
            "employee_national_insurance": c_ni,
            "universal_credit": c_uc,
            "other_benefits": c_ob,
            "other_residual": c_other,
        },
        "other_benefits_detail_gbp": {
            v: shock[v] - base[v]
            for v in OTHER_BENEFITS
            if abs(shock[v] - base[v]) > 1e6
        },
    }


def main() -> None:
    dataset, baseline, persons = _baseline_and_persons(DATASET, None, PERIOD)
    weight = persons["weight"].to_numpy(float)
    scen = PRESETS["full_tariff_displacement"]

    # ---- 1. zero-UC decomposition, seeds 0-4 ----------------------------
    per_seed = {}
    shocked0 = None
    table0 = None
    for seed in range(5):
        table = apply_shocks(persons, scen, seed=seed)
        shocked = build_shocked_sim(dataset, baseline, table, PERIOD)
        per_seed[seed] = zero_uc_decomposition(
            shocked, table["displaced"].to_numpy(), weight, PERIOD
        )
        if seed == 0:
            shocked0, table0 = shocked, table
        print(f"[mechanism] seed {seed}: {json.dumps(per_seed[seed])}")

    keys = [k for k in per_seed[0] if k.startswith("share_")]
    summary = {
        k: {
            "mean": float(np.mean([per_seed[s][k] for s in per_seed])),
            "sd": float(np.std([per_seed[s][k] for s in per_seed], ddof=1)),
        }
        for k in keys
    }
    (RESULTS / "mechanism_seed0.json").write_text(
        json.dumps({"seed0": per_seed[0], "seeds_0_4_summary": summary, "per_seed": per_seed}, indent=2)
    )

    # ---- 2. component decomposition, both margins -----------------------
    decomp = {
        "full_tariff_displacement_seed0": cushioning_components(
            baseline, shocked0, persons, table0, PERIOD
        )
    }
    wc_table = apply_shocks(persons, PRESETS["full_tariff_wage_cut"], seed=0)
    wc_shocked = build_shocked_sim(dataset, baseline, wc_table, PERIOD)
    decomp["full_tariff_wage_cut"] = cushioning_components(
        baseline, wc_shocked, persons, wc_table, PERIOD
    )
    (RESULTS / "cushioning_decomposition.json").write_text(json.dumps(decomp, indent=2))
    print(json.dumps(decomp, indent=2))

    # ---- 3. capital-rules counterfactual (seed 0) -----------------------
    from policyengine_uk import Microsimulation

    reform_out = {"baseline_rules": decomp["full_tariff_displacement_seed0"]["cushioning_rate"]}
    for name, reform in (
        ("capital_limit_off", REFORM_LIMIT_OFF),
        ("capital_rules_off", REFORM_CAPITAL_OFF),
    ):
        base_r = Microsimulation(dataset=dataset, reform=reform)
        shock_r = build_shocked_sim(dataset, base_r, table0, PERIOD, reform=reform)
        comp = cushioning_components(base_r, shock_r, persons, table0, PERIOD)
        mech = zero_uc_decomposition(
            shock_r, table0["displaced"].to_numpy(), weight, PERIOD
        )
        reform_out[name] = {
            "cushioning_rate": comp["cushioning_rate"],
            "components_share_of_gross_loss": comp["components_share_of_gross_loss"],
            "zero_uc_decomposition": mech,
        }
        print(f"[reform {name}] cushioning {comp['cushioning_rate']:.4f}")
    (RESULTS / "mechanism_capital_reform.json").write_text(json.dumps(reform_out, indent=2))


if __name__ == "__main__":
    main()
