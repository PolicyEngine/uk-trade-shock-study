"""Referee-revision computations: pension channel, take-up headline grid,
and the monthly-versus-annual UC bounding exercise.

Three artefacts requested by the referee report:

A. Pension-contribution channel. The HBAI disposable-income concept treats
   employee pension contributions as an outflow, so part of the wage-cut
   margin's measured cushioning is reduced retirement saving rather than
   tax--benefit insurance. This block recomputes the unit 12-month wage-cut
   and displacement cushioning rates under a pension-gross income concept
   (disposable income plus employee pension and salary-sacrifice
   contributions), so the pure statutory component of the wage-minus-
   displacement contrast can be reported. Runs on the primary submission
   estimator (balanced assignment, unit 12-month stress, 50 paired draws).

B. Take-up as a headline sensitivity. Full_tariff displacement at post-shock
   new-entitlement UC take-up 0.55/0.80/1.00 over 25 balanced-comparator
   (Bernoulli) seeds, and wage_cut at the same rates over 5 seeds (its
   dispersion is claiming-draw only).

C. Monthly-versus-annual UC bounding. Parameter-based accounting for a
   representative single displaced worker at the exposed-sector FRS mean wage:
   compares the paper's duration-equivalent annual stress with a correct
   monthly UC assessment and partial-year PAYE for 3-, 6- and 12-month spells.

Writes results/referee_fixes.json.
Usage: uv run python analysis/referee_fixes.py [--only pension takeup monthly]
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from uk_trade_shock_study.runner import _baseline_and_persons
from uk_trade_shock_study.shocks import (
    TradeShockScenario,
    _baseline_flag_values_and_rate,
    apply_shocks,
    build_shocked_simulation,
)

PERIOD = 2026
DATASET = Path("data/frs_2024_25.h5")
RESULTS = Path("results")
PENSION_VARS = ("employee_pension_contributions", "pension_contributions_via_salary_sacrifice")
#: Pension channel now runs on the primary submission estimator: balanced
#: assignment at the unit 12-month stress, 50 paired draws — the same design
#: behind every headline number (referee presentation point on estimator
#: consistency).
PENSION_SEEDS = 50
TAKEUPS = (0.55, 0.80, 1.00)
TAKEUP_SEEDS_DISPLACEMENT = 25
TAKEUP_SEEDS_WAGE_CUT = 5
EXPOSED_MEAN_EARNINGS = 48_272.0  # FRS exposed-goods-division weighted mean


def _hni_total(sim) -> float:
    hh_w = sim.calculate("household_weight", period=PERIOD, map_to="household").values
    hni = sim.calculate("hbai_household_net_income", period=PERIOD, map_to="household").values
    return float((hni * hh_w).sum())


def _pension_total(sim, weights) -> float:
    total = 0.0
    for v in PENSION_VARS:
        total += float(
            (sim.calculate(v, period=PERIOD, map_to="person").values * weights).sum()
        )
    return total


def pension_block(dataset, baseline, persons) -> dict:
    w = persons["weight"].to_numpy(float)
    base_hni = _hni_total(baseline)
    base_pen = _pension_total(baseline, w)
    scenarios = {
        "full_tariff_wage_cut": (
            TradeShockScenario(
                "pension_unit_12m_wage_cut",
                "full_tariff",
                "wage_cut",
                elasticity=1.0,
                selection_method="balanced",
            ),
            1,
        ),
        "full_tariff_displacement": (
            TradeShockScenario(
                "pension_unit_12m_displacement",
                "full_tariff",
                "displacement",
                elasticity=1.0,
                selection_method="balanced",
            ),
            PENSION_SEEDS,
        ),
    }
    out = {"n_seeds": PENSION_SEEDS, "selection_method": "balanced"}
    for name, (scenario, seeds) in scenarios.items():
        rows = []
        for seed in range(seeds):
            table = apply_shocks(persons, scenario, seed=seed)
            sim = build_shocked_simulation(dataset, baseline, table, PERIOD)
            gross = float(
                (
                    (
                        persons["employment_income"].to_numpy(float)
                        - table["employment_income"].to_numpy(float)
                    )
                    * w
                ).sum()
            )
            d_hni = base_hni - _hni_total(sim)
            d_pen = base_pen - _pension_total(sim, w)
            rows.append(
                {
                    "gross_earnings_loss": gross,
                    "disposable_loss_hbai": d_hni,
                    "pension_outflow_fall": d_pen,
                    "cushioning_hbai": 1.0 - d_hni / gross,
                    # pension-gross income = disposable + pension outflows;
                    # its loss is d_hni + d_pen (contributions fall -> the
                    # pension-gross loss exceeds the HBAI loss).
                    "cushioning_pension_gross": 1.0 - (d_hni + d_pen) / gross,
                    "pension_share_of_gross_loss": d_pen / gross,
                }
            )
            print(f"[pension] {name} seed {seed}: {rows[-1]}", flush=True)
            del sim
        out[name] = {
            k: {
                "mean": float(np.mean([r[k] for r in rows])),
                "sd": float(np.std([r[k] for r in rows], ddof=1)) if len(rows) > 1 else 0.0,
            }
            for k in rows[0]
        }
    hbai_gap = (
        out["full_tariff_wage_cut"]["cushioning_hbai"]["mean"]
        - out["full_tariff_displacement"]["cushioning_hbai"]["mean"]
    )
    pg_gap = (
        out["full_tariff_wage_cut"]["cushioning_pension_gross"]["mean"]
        - out["full_tariff_displacement"]["cushioning_pension_gross"]["mean"]
    )
    out["contrast"] = {
        "cushioning_gap_hbai_pp": 100 * hbai_gap,
        "cushioning_gap_pension_gross_pp": 100 * pg_gap,
        "pension_channel_share_of_gap": 1.0 - pg_gap / hbai_gap if hbai_gap else float("nan"),
    }
    return out


def takeup_block(dataset, baseline, persons) -> dict:
    w = persons["weight"].to_numpy(float)
    base_hni = _hni_total(baseline)
    hh_w = baseline.calculate("household_weight", period=PERIOD, map_to="household").values
    base_gov = float(
        (baseline.calculate("gov_balance", period=PERIOD, map_to="household").values * hh_w).sum()
    )
    out = {}
    for margin, n_seeds in (
        ("displacement", TAKEUP_SEEDS_DISPLACEMENT),
        ("wage_cut", TAKEUP_SEEDS_WAGE_CUT),
    ):
        block = {}
        for takeup in TAKEUPS:
            scen = TradeShockScenario(
                f"full_tariff_{margin}", "full_tariff", margin, uc_takeup=takeup
            )
            rows = []
            for seed in range(n_seeds):
                table = apply_shocks(persons, scen, seed=seed)
                sim = build_shocked_simulation(dataset, baseline, table, PERIOD)
                gross = float(
                    (
                        (
                            persons["employment_income"].to_numpy(float)
                            - table["employment_income"].to_numpy(float)
                        )
                        * w
                    ).sum()
                )
                d_hni = base_hni - _hni_total(sim)
                gov = float(
                    (
                        sim.calculate("gov_balance", period=PERIOD, map_to="household").values
                        * hh_w
                    ).sum()
                )
                rows.append(
                    {
                        "cushioning_rate": 1.0 - d_hni / gross,
                        "exchequer_cost_m": (base_gov - gov) / 1e6,
                    }
                )
                del sim
            block[f"{takeup:.2f}"] = {
                k: {
                    "mean": float(np.mean([r[k] for r in rows])),
                    "sd": float(np.std([r[k] for r in rows], ddof=1))
                    if len(rows) > 1
                    else 0.0,
                }
                for k in rows[0]
            }
            print(f"[takeup] {margin} {takeup}: {block[f'{takeup:.2f}']}", flush=True)
        out[margin] = {"n_seeds": n_seeds, **block}

    # Fourth convention: the stale pre-shock claiming flag (pre-fix
    # behaviour), displacement only.
    baseline_flag, _ = _baseline_flag_values_and_rate(baseline, PERIOD)
    scen = TradeShockScenario(
        "full_tariff_displacement", "full_tariff", "displacement"
    )
    rows = []
    for seed in range(TAKEUP_SEEDS_DISPLACEMENT):
        table = apply_shocks(persons, scen, seed=seed)
        sim = build_shocked_simulation(dataset, baseline, table, PERIOD)
        sim.set_input("would_claim_uc", PERIOD, baseline_flag)
        sim._invalidate_all_caches()
        gross = float(
            (
                (
                    persons["employment_income"].to_numpy(float)
                    - table["employment_income"].to_numpy(float)
                )
                * w
            ).sum()
        )
        d_hni = base_hni - _hni_total(sim)
        gov = float(
            (
                sim.calculate("gov_balance", period=PERIOD, map_to="household").values
                * hh_w
            ).sum()
        )
        rows.append(
            {
                "cushioning_rate": 1.0 - d_hni / gross,
                "exchequer_cost_m": (base_gov - gov) / 1e6,
            }
        )
        del sim
    out["displacement"]["stale_baseline_flag"] = {
        k: {
            "mean": float(np.mean([r[k] for r in rows])),
            "sd": float(np.std([r[k] for r in rows], ddof=1)),
        }
        for k in rows[0]
    }
    return out


def monthly_uc_block() -> dict:
    """Parameter-based bounding of the annual model against a monthly UC
    assessment for a representative single displaced worker (aged 25+,
    no children, no housing element, capital below the lower limit).
    """
    from policyengine_uk.system import system

    inst = f"{PERIOD}-01-01"
    gov = system.parameters.gov
    sa_month = float(gov.dwp.universal_credit.standard_allowance.amount.SINGLE_OLD(inst))
    pa = float(gov.hmrc.income_tax.allowances.personal_allowance.amount(inst))
    brackets = gov.hmrc.income_tax.rates.uk.brackets
    basic = float(brackets[0].rate(inst))
    higher = float(brackets[1].rate(inst))
    higher_threshold = float(brackets[1].threshold(inst))
    ni_main = float(gov.hmrc.national_insurance.class_1.rates.employee.main(inst))
    ni_pt_week = float(gov.hmrc.national_insurance.class_1.thresholds.primary_threshold(inst))
    e = EXPOSED_MEAN_EARNINGS

    def income_tax(y: float) -> float:
        taxable = max(0.0, y - pa)
        return basic * min(taxable, higher_threshold) + higher * max(
            0.0, taxable - higher_threshold
        )

    def ni_annualised(y: float) -> float:
        return ni_main * max(0.0, y - ni_pt_week * 52)

    rows = {}
    for m in (3, 6, 12):
        f = m / 12
        # Monthly-correct: m months out of work at full standard allowance
        # (zero earned income), (12-m) months at earnings that taper UC to
        # zero; PAYE reconciles to annual tax on partial-year earnings.
        uc_monthly = m * sa_month
        tax_relief_monthly = income_tax(e) - income_tax((1 - f) * e)
        ni_relief_monthly = ni_main * max(0.0, e / 12 - ni_pt_week * 52 / 12) * m
        # Annual duration-equivalent stress: probability f of a coherent
        # full-year displaced state; expectations per exposed worker.
        uc_annual = f * 12 * sa_month
        tax_relief_annual = f * income_tax(e)
        ni_relief_annual = f * ni_annualised(e)
        rows[f"{m}m"] = {
            "uc_monthly_correct": uc_monthly,
            "uc_annual_equivalent": uc_annual,
            "tax_relief_monthly_correct": tax_relief_monthly,
            "tax_relief_annual_equivalent": tax_relief_annual,
            "ni_relief_monthly_correct": ni_relief_monthly,
            "ni_relief_annual_equivalent": ni_relief_annual,
            "gross_loss": f * e,
            "cushion_share_monthly_correct": (
                uc_monthly + tax_relief_monthly + ni_relief_monthly
            )
            / (f * e),
            "cushion_share_annual_equivalent": (
                uc_annual + tax_relief_annual + ni_relief_annual
            )
            / (f * e),
        }
    return {
        "parameters": {
            "standard_allowance_single_25_plus_month": sa_month,
            "personal_allowance": pa,
            "basic_rate": basic,
            "higher_rate": higher,
            "ni_employee_main": ni_main,
            "representative_earnings": e,
        },
        "spells": rows,
        "notes": (
            "UC entitlement per worker-month is identical by construction: "
            "m months at the full standard allowance equals probability m/12 "
            "of 12 months. Differences arise in income tax (convexity: the "
            "annual model overstates relief for partial-year spells at this "
            "earnings level, where zeroing a full year forgoes higher-rate "
            "relief while a partial-year earner loses basic/higher-rate "
            "slices) and in the 5-week payment wait, which shifts timing, "
            "not annual entitlement."
        ),
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        nargs="*",
        choices=("monthly", "pension", "takeup"),
        help=(
            "recompute only the named blocks, merging into the existing "
            "results/referee_fixes.json (all blocks when omitted)"
        ),
    )
    args = parser.parse_args()
    blocks = set(args.only or ("monthly", "pension", "takeup"))

    artifact = RESULTS / "referee_fixes.json"
    out = json.loads(artifact.read_text()) if args.only and artifact.exists() else {}
    dataset = baseline = persons = None
    if blocks & {"pension", "takeup"}:
        dataset, baseline, persons = _baseline_and_persons(DATASET, None, PERIOD)
    if "monthly" in blocks:
        out["monthly_uc_bounding"] = monthly_uc_block()
    if "pension" in blocks:
        out["pension_channel"] = pension_block(dataset, baseline, persons)
    if "takeup" in blocks:
        out["takeup_headline"] = takeup_block(dataset, baseline, persons)
    RESULTS.mkdir(exist_ok=True)
    artifact.write_text(json.dumps(out, indent=2))
    print("[written] results/referee_fixes.json")


if __name__ == "__main__":
    main()
