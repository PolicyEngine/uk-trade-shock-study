"""E3: what is tariff-specific? A uniform benchmark and decile income shares.

Two outputs, built from one baseline so that they are on the same footing.

1. UNIFORM BENCHMARK (results/e3_uniform.json). The tariff scenario's gross
   earnings loss (sum_i s_i x earnings_i x weight_i over the exposed
   divisions) is spread instead over EVERY employee in the economy:

   * uniform wage cut -- every employee loses the same share k of earnings,
     k = tariff gross loss / total employee earnings;
   * uniform displacement -- every employee faces the same job-loss
     probability k for the year (so the expected gross loss is the same).

   Both use the conventions of the tariff runs: the exact-expectation design
   for displacement (all records, no draws, New Style JSA approximation, 26
   weeks) and the exact 0.8 / 0.2 mix of the all-claim and none-claim
   simulations for newly entitled benefit units on both margins. The
   difference between a tariff-pattern rate and its uniform counterpart is
   what the tariff's sectoral pattern -- who works in the exposed divisions,
   what they earn and what households they live in -- contributes.

2. INCOME SHARES BY DECILE (results/e3_income_shares.json). For the tariff
   scenario and the uniform benchmark, both margins: the gross earnings loss
   and the post-cushion loss of HBAI net income in each decile, as a share
   of the decile's aggregate baseline HBAI household net income and of its
   aggregate consumption. Deciles are HOUSEHOLD-weighted deciles of
   equivalised HBAI net income (before housing costs) -- the construction of
   the price episodes (second_stage_energy.py) -- not the person-weighted
   deciles behind the ``\\EThree...DecGbp...`` macros. Consumption is the
   model's ``consumption`` variable at the simulation year (imputed from
   LCFS donors, NOT rebased to an ONS control total as the energy episode
   does), so the spending shares are indicative.

Displacement passes are pickled under .cache/e3_passes/ (gitignored). The
tariff displacement passes are shared with sensitivity_duration_takeup.py.

Usage: PYTHONPATH=. .venv/bin/python analysis/e3_uniform_and_income_shares.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from analysis.sensitivity_duration_takeup import CENTRAL, PERIOD, _cached
from tariff_pipeline.expectation import (
    expectation_result,
    expected_household_changes,
    household_deciles,
    loss_shares_by_decile,
    run_passes,
    wage_cut_expectation_result,
)
from tariff_pipeline.runner import _baseline_and_persons, _metrics, json_value
from tariff_pipeline.shocks import PRESETS, TradeShockScenario, _person_shock

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
SPLIT = ("income_tax", "national_insurance", "universal_credit", "jsa_contrib",
         "pension_contributions")


def summary(result: dict) -> dict:
    shares = result["cushion_component_shares"]["share_of_gross_loss"]
    rate = result["cushioning_rate"]
    return {
        "scenario": result["scenario"],
        "design": result["design"],
        "uc_takeup": result["uc_takeup"],
        "exposed_records": result["exposed_records"],
        "exposed_households": result["exposed_households"],
        "gross_earnings_loss": result["gross_earnings_loss"],
        "target_gross_earnings_loss": result["target_gross_earnings_loss"],
        "net_disposable_loss": result["net_disposable_loss"],
        "cushioning_rate": rate,
        "tax_benefit_cushioning_rate": rate - shares["pension_contributions"],
        "cushioning_rate_sampling_se": result["cushioning_rate_sampling_se"],
        "share_of_gross_loss": {line: shares[line] for line in SPLIT},
        "share_of_gross_loss_all_lines": shares,
        "exchequer_cost": result["exchequer_cost_mean"],
        "poverty_rate_change_bhc": result["poverty_rate_change_bhc_mean"],
        "extra": {
            k: result[k]
            for k in ("redrawn_benefit_units", "households_with_two_redrawn_units",
                      "omitted_gross_share", "displaced_weighted_mean",
                      "n_simulation_passes", "effective_loss_records")
            if k in result
        },
    }


def main() -> None:
    dataset, baseline, persons = _baseline_and_persons(None, None, PERIOD)
    base_metrics = _metrics(baseline, PERIOD)
    equiv_person = baseline.calculate(
        "equiv_hbai_household_net_income", period=PERIOD, map_to="person"
    ).values
    region = baseline.calculate("region", period=PERIOD, map_to="person").values.astype(str)

    def hh(name):
        return np.asarray(
            baseline.calculate(name, period=PERIOD, map_to="household").values, dtype=float
        )

    w = hh("household_weight")
    net_income = hh("hbai_household_net_income")
    consumption = hh("consumption")
    decile = household_deciles(hh("equiv_hbai_household_net_income"), w)

    weight = persons["weight"].to_numpy(dtype=float)
    earnings = persons["employment_income"].to_numpy(dtype=float)
    tariff_shock = _person_shock(persons, CENTRAL)
    target = float((tariff_shock * earnings * weight).sum())
    employee_earnings = float((earnings * weight)[earnings > 0].sum())
    k = target / employee_earnings

    wage_scenarios = {
        "tariff": PRESETS["full_tariff_wage_cut"],
        "uniform": TradeShockScenario(
            "uniform_wage_cut", "full_tariff", "wage_cut", uniform_shock=k
        ),
    }
    displacement_scenarios = {
        "tariff": CENTRAL,
        "uniform": TradeShockScenario(
            "uniform_displacement", "full_tariff", "displacement",
            new_style_jsa=True, uniform_shock=k,
        ),
    }

    rows, shares = {}, {}
    for pattern, scenario in wage_scenarios.items():
        result, expected = wage_cut_expectation_result(
            dataset, baseline, persons, scenario, PERIOD, base_metrics,
            equiv_person, region, return_household_changes=True,
        )
        rows[f"{pattern}_wage_cut"] = summary(result)
        shares[f"{pattern}_wage_cut"] = loss_shares_by_decile(
            expected, w, decile, net_income, consumption
        )
    for pattern, scenario in displacement_scenarios.items():
        key = "jsa26" if pattern == "tariff" else "uniform_displacement"
        passes = _cached(
            key,
            lambda scenario=scenario: run_passes(
                dataset, baseline, persons, scenario, PERIOD, base_metrics
            ),
        )
        result = expectation_result(
            passes, persons, scenario, base_metrics, equiv_person, region
        )
        expected = expected_household_changes(
            passes, _person_shock(persons, scenario), scenario.uc_takeup
        )[0]
        rows[f"{pattern}_displacement"] = summary(result)
        shares[f"{pattern}_displacement"] = loss_shares_by_decile(
            expected, w, decile, net_income, consumption
        )
        del passes

    # the tariff rows must be the paper's central estimates
    for name, key in (("full_tariff_wage_cut", "tariff_wage_cut"),
                      ("full_tariff_displacement", "tariff_displacement")):
        stored = json.loads((RESULTS / f"{name}.json").read_text())["cushioning_rate"]
        if abs(rows[key]["cushioning_rate"] - stored) > 1e-6:
            raise RuntimeError(f"{key} does not reproduce results/{name}.json")
    for key in ("uniform_wage_cut", "uniform_displacement"):
        if abs(rows[key]["target_gross_earnings_loss"] / target - 1.0) > 1e-9:
            raise RuntimeError(f"{key}: imposed loss is not the tariff scenario's")

    def gap(margin, field):
        return rows[f"tariff_{margin}"][field] - rows[f"uniform_{margin}"][field]

    uniform = {
        "period": PERIOD,
        "gross_earnings_loss": target,
        "employee_earnings": employee_earnings,
        "uniform_shock": k,
        "employees": float(weight[earnings > 0].sum()),
        "employee_records": int((earnings > 0).sum()),
        "rows": rows,
        "tariff_minus_uniform": {
            margin: {
                "cushioning_rate": gap(margin, "cushioning_rate"),
                "tax_benefit_cushioning_rate": gap(margin, "tax_benefit_cushioning_rate"),
                "share_of_gross_loss": {
                    line: rows[f"tariff_{margin}"]["share_of_gross_loss"][line]
                    - rows[f"uniform_{margin}"]["share_of_gross_loss"][line]
                    for line in SPLIT
                },
            }
            for margin in ("wage_cut", "displacement")
        },
        "note": (
            "Same gross earnings loss as the tariff scenario, spread over every "
            "employee: an equal proportional cut (wage cut) or an equal job-loss "
            "probability (displacement). Same New Style JSA and UC take-up "
            "conventions as the tariff runs."
        ),
    }
    income = {
        "period": PERIOD,
        "decile_definition": (
            "Household-weighted deciles of equivalised HBAI household net income "
            "(before housing costs) at baseline, as in the price episodes."
        ),
        "income_concept": "hbai_household_net_income (baseline, decile aggregate)",
        "consumption_concept": (
            "model variable `consumption` at the simulation year (LCFS-imputed, "
            "not rebased to an ONS control total)"
        ),
        "decile_baseline": {
            str(d): {
                "households": float(w[decile == d].sum()),
                "net_income": float((w * net_income)[decile == d].sum()),
                "consumption": float((w * consumption)[decile == d].sum()),
            }
            for d in range(1, 11)
        },
        "shares": shares,
    }
    for name, payload in (("e3_uniform", uniform), ("e3_income_shares", income)):
        (RESULTS / f"{name}.json").write_text(
            json.dumps(json_value(payload), indent=2, allow_nan=False) + "\n"
        )

    print(f"uniform shock k = {100 * k:.4f}% of earnings / job-loss probability")
    for key, row in rows.items():
        s = row["share_of_gross_loss"]
        print(f"{key:22s} rate {100 * row['cushioning_rate']:.2f} tax+ben "
              f"{100 * row['tax_benefit_cushioning_rate']:.2f} | IT {100 * s['income_tax']:.1f} "
              f"NI {100 * s['national_insurance']:.1f} UC {100 * s['universal_credit']:.1f} "
              f"JSA {100 * s['jsa_contrib']:.1f} pension {100 * s['pension_contributions']:.1f} "
              f"SE {100 * row['cushioning_rate_sampling_se']:.2f} {row['extra']}")
    for key, table in shares.items():
        print(key)
        for field in ("share_of_total_gross_loss", "gross_loss_share_of_income",
                      "net_loss_share_of_income", "net_loss_share_of_consumption",
                      "cushioning_rate"):
            print(f"  {field:32s}", " ".join(
                f"{100 * table[str(d)][field]:7.3f}" for d in [*range(1, 11), "all"]))


if __name__ == "__main__":
    main()
