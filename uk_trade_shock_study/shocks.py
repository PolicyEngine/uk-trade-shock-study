"""Adjustment-margin families for the tariff shock on FRS workers.

The derived per-SIC earnings-shock sizes (exposure.sector_earnings_shocks)
are realised on employees in exposed divisions through three families — the
paper's central axis, because Universal Credit replaces unemployment far more
generously than in-work wage cuts:

- ``displacement`` (ADH short-run): the sector earnings shock is delivered as
  JOB LOSS. Within each exposed SIC division a weighted head-count quota of
  ``shock_j x division employee weight`` is drawn with UNIFORM ordering keys
  (the survey weight enters only through quota consumption, mirroring
  uk-ai-study: a represented person's inclusion probability does not depend
  on their record's grossing weight). Displaced workers move to
  employment_status UNEMPLOYED with earnings, hours, pension contributions
  and statutory pay zeroed.

- ``wage_cut`` (Dauth/Traiberman long-run reallocation): the SAME aggregate
  earnings loss is delivered as proportional wage cuts with no job loss.
  Formula: worker i in division j takes cut_i = k * shock_j, with the single
  calibration constant k chosen so that the weighted aggregate earnings
  removed equals the displacement family's expected removal,
  sum_j shock_j x (employee wage bill of j) — earnings-equivalence by
  construction (see apply_wage_cut docstring).

- ``inactivity`` (Beatty-Fothergill UK history): the displacement draw, but
  displaced workers aged >= ``inactivity_age_threshold`` exit to economic
  INACTIVITY (UC health element route) rather than unemployment.

Hard-error contract: build_shocked_simulation verifies that every displaced
person's employment_status actually changed in the shocked simulation and
raises RuntimeError otherwise (the uk-ai-study template had a silent-failure
risk here; we fail hard).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from uk_trade_shock_study.exposure import (
    DEFAULT_ELASTICITY,
    DEFAULT_PASSTHROUGH,
    person_earnings_shock,
)

MARGINS = ("displacement", "wage_cut", "inactivity")


@dataclass(frozen=True)
class TradeShockScenario:
    name: str
    tariff_scenario: str  # "full_tariff" | "epd"
    margin: str  # "displacement" | "wage_cut" | "inactivity"
    elasticity: float = DEFAULT_ELASTICITY
    passthrough: float = DEFAULT_PASSTHROUGH
    inactivity_age_threshold: int = 50

    def __post_init__(self):
        if self.margin not in MARGINS:
            raise ValueError(f"unknown margin {self.margin!r}; use one of {MARGINS}")


#: Scenario presets: {full_tariff, epd} x {displacement, wage_cut, inactivity}.
PRESETS = {
    f"{tariff}_{margin}": TradeShockScenario(f"{tariff}_{margin}", tariff, margin)
    for tariff in ("full_tariff", "epd")
    for margin in MARGINS
}


def _person_shock(persons: pd.DataFrame, scenario: TradeShockScenario) -> np.ndarray:
    return person_earnings_shock(
        persons["sic_division"],
        scenario.tariff_scenario,
        elasticity=scenario.elasticity,
        passthrough=scenario.passthrough,
    )


def draw_displaced(
    persons: pd.DataFrame,
    scenario: TradeShockScenario,
    seed: int = 0,
) -> np.ndarray:
    """Boolean displaced mask: per-division weighted quotas, uniform ordering.

    Quota for division j = shock_j x (weighted employee count of j). Within
    the division, members are drawn without replacement with UNIFORM ordering
    keys; weights are consumed against the quota, and the quota-crossing
    person is included with probability equal to the remaining quota
    fraction, so the expected displaced weight equals the quota exactly.
    """
    rng = np.random.default_rng(seed)
    employed = persons["employment_income"].to_numpy(dtype=float) > 0
    weight = persons["weight"].to_numpy(dtype=float)
    shock = _person_shock(persons, scenario)
    division = pd.to_numeric(persons["sic_division"], errors="coerce").to_numpy(dtype=float)

    displaced = np.zeros(len(persons), dtype=bool)
    for d in np.unique(division[employed & (shock > 0)]):
        members = np.flatnonzero(employed & (division == d))
        quota = float(shock[members[0]]) * float(weight[members].sum())
        if quota <= 0:
            continue
        chosen = rng.permutation(members)
        cum = np.cumsum(weight[chosen])
        displaced[chosen[cum <= quota]] = True
        crossing = np.searchsorted(cum, quota)
        if crossing < len(chosen) and cum[crossing] > quota:
            shortfall = quota - (cum[crossing - 1] if crossing else 0.0)
            if rng.random() < shortfall / weight[chosen[crossing]]:
                displaced[chosen[crossing]] = True
    return displaced


def apply_displacement(
    persons: pd.DataFrame,
    scenario: TradeShockScenario,
    seed: int = 0,
) -> pd.DataFrame:
    """Displacement (or inactivity) margin: shocked copy of the person table.

    Adds boolean columns ``displaced`` (out of work) and ``inactive`` (subset
    of displaced who flow to inactivity — empty under pure displacement).
    """
    shocked = persons.copy()
    displaced = draw_displaced(persons, scenario, seed=seed)
    shocked["displaced"] = displaced
    age = persons["age"].to_numpy(dtype=float)
    if scenario.margin == "inactivity":
        shocked["inactive"] = displaced & (age >= scenario.inactivity_age_threshold)
    else:
        shocked["inactive"] = np.zeros(len(persons), dtype=bool)
    earnings = shocked["employment_income"].to_numpy(dtype=float)
    shocked["employment_income"] = np.where(displaced, 0.0, earnings)
    return shocked


def apply_wage_cut(persons: pd.DataFrame, scenario: TradeShockScenario) -> pd.DataFrame:
    """Wage-cut margin: earnings-equivalent proportional cuts, no job loss.

    Target aggregate loss L = sum_j shock_j x B_j, where B_j is the weighted
    employee wage bill of division j — exactly the displacement family's
    expected earnings removal (the quota removes shock_j of the division
    head-count with uniform ordering, so the expected earnings removed is
    shock_j x B_j up to within-draw selection covariance).

    Each employee in division j is cut by k x shock_j where
    k = L / sum_j shock_j x B_j = 1 identically for proportional cuts, so
    cut_i = shock_j — the per-division cut IS the derived shock size, and the
    weighted aggregate earnings removed equals L exactly (conservation is
    asserted, not assumed). Non-exposed workers are untouched. Deterministic
    (no draw, no seed).
    """
    if scenario.margin != "wage_cut":
        raise ValueError("apply_wage_cut requires a wage_cut scenario")
    shocked = persons.copy()
    earnings = shocked["employment_income"].to_numpy(dtype=float)
    weight = shocked["weight"].to_numpy(dtype=float)
    employed = earnings > 0
    shock = _person_shock(persons, scenario)
    if (shock[employed] >= 1.0).any():
        raise ValueError(
            "derived sector shock >= 100% of earnings for some workers; "
            "check elasticity/passthrough calibration."
        )
    target = float((shock * earnings * weight)[employed].sum())
    new = np.where(employed, earnings * (1.0 - shock), earnings)
    realised = float(((earnings - new) * weight)[employed].sum())
    if not np.isclose(realised, target, rtol=1e-9):
        raise AssertionError("wage-bill conservation failed")
    shocked["employment_income"] = new
    shocked["displaced"] = np.zeros(len(persons), dtype=bool)
    shocked["inactive"] = np.zeros(len(persons), dtype=bool)
    return shocked


def apply_shocks(
    persons: pd.DataFrame,
    scenario: TradeShockScenario,
    seed: int = 0,
) -> pd.DataFrame:
    """Dispatch on the scenario's adjustment margin.

    Expects columns: employment_income, sic_division, age, weight.
    """
    if scenario.margin == "wage_cut":
        return apply_wage_cut(persons, scenario)
    return apply_displacement(persons, scenario, seed=seed)


#: Person-level inputs zeroed for displaced workers so they do not remain
#: in_work (hours > 0 keeps childcare elements paying), keep deducting
#: pension contributions from zero earnings, or draw statutory pay
#: (transition contract inherited from uk-ai-study).
TRANSITION_ZEROED_VARIABLES = (
    "hours_worked",
    "employee_pension_contributions",
    "pension_contributions_via_salary_sacrifice",
    "statutory_maternity_pay",
    "statutory_paternity_pay",
    "statutory_sick_pay",
)

SHOCKED_INCOME_VARIABLES = ("employment_income",)


def build_shocked_simulation(dataset, baseline_sim, shocked_table, period):
    """One shared constructor for the shocked simulation (every pipeline).

    Displaced-not-inactive workers become UNEMPLOYED; inactive workers become
    INACTIVE (the UC-health-element route is a downstream TODO: flag limited
    capability for work once the modelling decision on WCA status is taken).
    A rejected set_input would silently leave displaced workers EMPLOYED with
    zero hours, changing entitlements in every result — we verify and FAIL
    HARD (do not replicate the template's silent-failure bug).
    """
    from policyengine_uk import Microsimulation

    sim = Microsimulation(dataset=dataset)
    for column in SHOCKED_INCOME_VARIABLES:
        sim.set_input(column, period, shocked_table[column].to_numpy(dtype=float))
    displaced = shocked_table["displaced"].to_numpy()
    inactive = shocked_table["inactive"].to_numpy()
    for var in TRANSITION_ZEROED_VARIABLES:
        values = baseline_sim.calculate(var, period=period, map_to="person").values.astype(float)
        values[displaced] = 0.0
        sim.set_input(var, period, values)
    status = baseline_sim.calculate("employment_status", period=period, map_to="person").values.astype(object)
    status[displaced] = "UNEMPLOYED"
    status[inactive] = "OTHER_INACTIVE"
    sim.set_input("employment_status", period, status)
    applied = sim.calculate("employment_status", period=period, map_to="person").values.astype(str)
    if not (applied[displaced & ~inactive] == "UNEMPLOYED").all():
        raise RuntimeError(
            "employment_status transition not applied: displaced persons are "
            "not all UNEMPLOYED in the shocked simulation."
        )
    if not (applied[inactive] == "OTHER_INACTIVE").all():
        raise RuntimeError(
            "employment_status transition not applied: inactive persons are "
            "not all INACTIVE in the shocked simulation."
        )
    return sim
