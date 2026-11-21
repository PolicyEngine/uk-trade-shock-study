"""E3 sensitivity: spell duration, UC take-up and New Style JSA weeks.

Displacement margin of the full-tariff scenario, exact-expectation design
(tariff_pipeline/expectation.py): every figure below is an expectation over
the Bernoulli job-loss assignment computed from all exposed records, so
none of them carries Monte Carlo error. Writes results/e3_sensitivity.json;
analysis/emit_e3_macros.py turns it into the ``\\EThreeSens...`` macros.

(a) DURATION. The central case is a job loss lasting the whole year. The
    model is annual, so a shorter spell cannot be simulated month by month;
    two approximations are reported, which err in opposite directions on the
    benefit side, for spells of 6 and 3 months (share f of the year):

    * ``annual_averaging`` -- the worker's annual earnings, hours and
      earnings-linked deductions are scaled by 1 - f, employment status is
      unchanged, New Style JSA is paid for min(26, 52 f) weeks, and the
      annual model is run on those annual totals
      (shocks.apply_partial_year_spell). Income tax, which is assessed on
      annual income, is right. Universal Credit and National Insurance are
      assessed monthly / per pay period, and averaging earnings over the year
      before applying the UC taper understates the UC paid in the months out
      of work. LOWER bound on the benefit response.

    * ``monthly_state_mix`` -- the year is f in the "out of work" state and
      1 - f in the baseline state. Every monthly-assessed line (UC and other
      means-tested benefits, National Insurance, pension contributions,
      student-loan repayments, statutory pay, JSA) is f times its change in a
      full-year out-of-work simulation in which New Style JSA is in payment
      every week (26 weeks of a 6-month spell = every week of it); income
      tax alone is taken from the annual-averaging pass. This assumes UC
      starts with the spell and stops with it (no five-week wait, no
      assessment-period timing, no surplus-earnings rule) and that the
      full-year out-of-work UC award applies pro rata, so it is an UPPER
      bound on the benefit response. It is NOT the central rate by
      construction: income tax is annual (the worker keeps the whole
      personal allowance against part-year earnings) and JSA covers a larger
      share of a shorter spell.

    The former hybrid (half annual earnings with a full-year UNEMPLOYED
    status) stays removed: it is not coherent.

(b) UC TAKE-UP of newly entitled benefit units at 0.6 and 1.0 (central
    0.8). The design mixes the all-claim and none-claim passes linearly, so
    these are re-weightings of the central passes, not new simulations.

(c) NEW STYLE JSA WEEKS: 13, against 26 (central) and 0 (the stored
    results/full_tariff_displacement_no_jsa.json).

Passes are pickled under .cache/e3_passes/ (gitignored: they are
household-level derivatives of licensed microdata) so a rerun only redoes
the arithmetic. Delete the folder to force new simulations.

Usage: PYTHONPATH=. .venv/bin/python analysis/sensitivity_duration_takeup.py
"""

from __future__ import annotations

import json
import pickle
from dataclasses import replace
from pathlib import Path

import numpy as np

from tariff_pipeline.expectation import (
    _clustered_ratio_se,
    expected_household_changes,
    run_passes,
)
from tariff_pipeline.runner import (
    _baseline_and_persons,
    _metrics,
    component_shares,
    cushion_components,
    json_value,
)
from tariff_pipeline.shocks import (
    TradeShockScenario,
    _person_shock,
    apply_fixed_displacement,
    apply_partial_year_spell,
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
CACHE = ROOT / ".cache" / "e3_passes"
PERIOD = 2026
TAKEUP_GRID = (0.6, 0.8, 1.0)
SPELLS = {"6m": 0.5, "3m": 0.25}
NET = "hbai_household_net_income"
NET_BEFORE_FLOOR = "hbai_household_net_income_before_reset_to_zero"
#: Lines reported in every row (share of the gross earnings loss).
REPORTED_LINES = (
    "income_tax",
    "national_insurance",
    "universal_credit",
    "jsa_contrib",
    "pension_contributions",
)

CENTRAL = TradeShockScenario(
    "full_tariff_displacement", "full_tariff", "displacement", new_style_jsa=True
)


def _cached(key: str, build):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{key}_{PERIOD}.pkl"
    if path.exists():
        print(f"[sensitivity] {key}: passes read from {path}")
        with open(path, "rb") as f:
            return pickle.load(f)
    passes = build()
    with open(path, "wb") as f:
        pickle.dump(passes, f)
    return passes


def summarise(expected: dict, weight_h: np.ndarray, gross_target: float) -> dict:
    """Cushioning rate and split from expected household changes."""
    totals = {name: float((weight_h * values).sum()) for name, values in expected.items()}
    components = cushion_components(totals)
    gross = components["gross_loss"]
    net = -totals[NET]
    shares = component_shares(components)["share_of_gross_loss"]
    rate = 1.0 - net / gross
    return {
        "gross_earnings_loss": gross,
        "gross_vs_target": gross / gross_target,
        "net_disposable_loss": net,
        "cushioning_rate": rate,
        "tax_benefit_cushioning_rate": rate - shares["pension_contributions"],
        "cushioning_rate_sampling_se": _clustered_ratio_se(
            -weight_h * expected[NET], -weight_h * expected["employment_income"]
        ),
        "share_of_gross_loss": {line: shares[line] for line in REPORTED_LINES},
        "share_of_gross_loss_all_lines": shares,
    }


def monthly_state_mix(out_of_work: dict, annual: dict, fraction: float) -> dict:
    """Expected household changes under the monthly-state mix (see module
    docstring): ``fraction`` of the out-of-work state, with income tax taken
    from the annual-averaging pass."""
    mixed = {name: fraction * values for name, values in out_of_work.items()}
    tax_correction = annual["income_tax"] - mixed["income_tax"]
    mixed["income_tax"] = annual["income_tax"].copy()
    # income tax is subtracted in HBAI net income and adds to the exchequer
    mixed[NET] = mixed[NET] - tax_correction
    mixed[NET_BEFORE_FLOOR] = mixed[NET_BEFORE_FLOOR] - tax_correction
    mixed["gov_balance"] = mixed["gov_balance"] + tax_correction
    return mixed


def main() -> None:
    dataset, baseline, persons = _baseline_and_persons(None, None, PERIOD)
    base_metrics = _metrics(baseline, PERIOD)
    hours = np.asarray(
        baseline.calculate("hours_worked", period=PERIOD, map_to="person").values,
        dtype=float,
    )
    shock = _person_shock(persons, CENTRAL)
    weight = persons["weight"].to_numpy(dtype=float)
    earnings = persons["employment_income"].to_numpy(dtype=float)
    target = float((shock * earnings * weight).sum())

    def passes_for(key, scenario=CENTRAL, build_table=None):
        return _cached(
            key,
            lambda: run_passes(
                dataset, baseline, persons, scenario, PERIOD, base_metrics,
                build_table=build_table,
            ),
        )

    def expected(passes, uc_takeup=CENTRAL.uc_takeup):
        return expected_household_changes(passes, shock, uc_takeup)[0]

    # ---- central passes: take-up grid -------------------------------------
    central = passes_for("jsa26")
    w = central.household_weight
    takeup = {
        f"{t:.1f}": summarise(expected(central, t), w, target) for t in TAKEUP_GRID
    }
    stored = json.loads((RESULTS / "full_tariff_displacement.json").read_text())
    central_row = takeup[f"{CENTRAL.uc_takeup:.1f}"]
    reproduces = abs(central_row["cushioning_rate"] - stored["cushioning_rate"])
    if reproduces > 1e-6:
        raise RuntimeError(
            "central passes do not reproduce results/full_tariff_displacement.json: "
            f"{central_row['cushioning_rate']:.6f} vs {stored['cushioning_rate']:.6f}"
        )

    # ---- New Style JSA weeks ----------------------------------------------
    no_jsa = json.loads((RESULTS / "full_tariff_displacement_no_jsa.json").read_text())
    jsa13 = passes_for("jsa13", replace(CENTRAL, new_style_jsa_weeks=13.0))
    no_jsa_shares = no_jsa["cushion_component_shares"]["share_of_gross_loss"]
    jsa_weeks = {
        "0": {
            "source": "results/full_tariff_displacement_no_jsa.json",
            "cushioning_rate": no_jsa["cushioning_rate"],
            "tax_benefit_cushioning_rate": no_jsa["cushioning_rate"]
            - no_jsa_shares["pension_contributions"],
            "cushioning_rate_sampling_se": no_jsa["cushioning_rate_sampling_se"],
            "share_of_gross_loss": {line: no_jsa_shares[line] for line in REPORTED_LINES},
        },
        "13": summarise(expected(jsa13), w, target),
        "26": central_row,
    }

    # ---- duration ---------------------------------------------------------
    def out_of_work_table(displaced, node_scenario):
        table = apply_fixed_displacement(persons, node_scenario, displaced, seed=0)
        # Annualised state: JSA in payment every week. Multiplied by the
        # share of the year out of work below; never a 52-week entitlement.
        table.attrs["new_style_jsa_weeks"] = 52.0
        table.attrs["new_style_jsa_max_weeks"] = 52.0
        return table

    out_of_work = expected(
        passes_for("out_of_work_jsa_every_week", build_table=out_of_work_table)
    )
    duration = {"12m": {"full_year_job_loss": central_row}}
    for label, fraction in SPELLS.items():
        def spell_table(affected, node_scenario, fraction=fraction):
            return apply_partial_year_spell(
                persons, node_scenario, affected, fraction, hours, seed=0
            )

        annual = expected(passes_for(f"annual_averaging_{label}", build_table=spell_table))
        annual_row = summarise(annual, w, fraction * target)
        mix_row = summarise(
            monthly_state_mix(out_of_work, annual, fraction), w, fraction * target
        )
        for row in (annual_row, mix_row):
            if abs(row["gross_vs_target"] - 1.0) > 1e-3:
                raise RuntimeError(f"{label}: gross loss is not {fraction} x the full-year loss")
        duration[label] = {
            "spell_share_of_year": fraction,
            "new_style_jsa_weeks": min(26.0, 52.0 * fraction),
            "annual_averaging": annual_row,
            "monthly_state_mix": mix_row,
        }

    out = {
        "scenario": CENTRAL.name,
        "period": PERIOD,
        "design": "exact_expectation",
        "central": {
            "uc_takeup": CENTRAL.uc_takeup,
            "new_style_jsa_weeks": CENTRAL.new_style_jsa_weeks,
            "spell": "12m",
            "cushioning_rate": central_row["cushioning_rate"],
            "reproduces_stored_central_within": reproduces,
        },
        "uc_takeup": takeup,
        "new_style_jsa_weeks": jsa_weeks,
        "duration": duration,
        "notes": {
            "uc_takeup": (
                "Claiming probability of benefit units NEWLY entitled to UC; "
                "exact linear re-weighting of the central all-claim and "
                "none-claim passes."
            ),
            "duration_annual_averaging": (
                "Annual earnings, hours and earnings-linked deductions scaled "
                "by 1 - f, status unchanged, JSA for min(26, 52f) weeks, annual "
                "model. Understates UC (assessed monthly): lower bound on the "
                "benefit response."
            ),
            "duration_monthly_state_mix": (
                "f x the change in a full-year out-of-work simulation with JSA "
                "in payment every week, for every line except income tax, "
                "which is taken from the annual-averaging pass. Assumes UC "
                "runs exactly for the spell: upper bound on the benefit "
                "response."
            ),
            "gross": (
                "A spell of share f removes f x the full-year gross earnings "
                "loss; rates are per pound of that smaller loss."
            ),
        },
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "e3_sensitivity.json").write_text(
        json.dumps(json_value(out), indent=2, allow_nan=False) + "\n"
    )

    def line(name, row):
        s = row["share_of_gross_loss"]
        print(
            f"{name:38s} rate {100 * row['cushioning_rate']:.2f}  tax+ben "
            f"{100 * row['tax_benefit_cushioning_rate']:.2f}  | IT {100 * s['income_tax']:.1f} "
            f"NI {100 * s['national_insurance']:.1f} UC {100 * s['universal_credit']:.1f} "
            f"JSA {100 * s['jsa_contrib']:.1f}"
        )

    for t, row in takeup.items():
        line(f"take-up {t}", row)
    for k, row in jsa_weeks.items():
        line(f"JSA weeks {k}", row)
    for label in SPELLS:
        line(f"{label} annual averaging", duration[label]["annual_averaging"])
        line(f"{label} monthly state mix", duration[label]["monthly_state_mix"])


if __name__ == "__main__":
    main()
