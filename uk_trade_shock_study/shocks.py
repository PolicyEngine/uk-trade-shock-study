"""Adjustment-margin families for the tariff shock on FRS workers.

The derived per-SIC earnings-shock sizes (exposure.sector_earnings_shocks)
are realised on employees in exposed divisions through three families — the
paper's central axis, because Universal Credit replaces unemployment far more
generously than in-work wage cuts:

- ``displacement`` (ADH short-run): the sector earnings shock is delivered as
  JOB LOSS. Each employee in division j is independently displaced with
  probability ``shock_j``. Survey weights never enter the draw, so expected
  weighted employment and earnings losses are both exactly proportional to
  the sector shock. Displaced workers move to
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
  INACTIVITY *and* are flagged as having limited capability for
  work-related activity (uc_limited_capability_for_WRA = True), so their
  Universal Credit includes the LCWRA health element — the benefit-financed
  inactivity route. ``lcwra_takeup`` (default 1.0) thins the flag for the
  sensitivity variant in which only a share of the older displaced pass the
  Work Capability Assessment.

- ``reallocation`` (literal sectoral reallocation, ADH/Autor-Dorn-Hanson
  "workers go into services"): the SAME displacement draw (same seed, so the
  two families are paired draw-for-draw), but the drawn workers do NOT become
  unemployed. They are RE-EMPLOYED in a services division — their
  ``sic_industry_division`` is reassigned to one of REALLOCATION_DESTINATIONS
  in proportion to those divisions' FRS employee weight — at an earnings
  penalty ``reallocation_penalty`` calibrated on the FRS itself
  (DEFAULT_REALLOCATION_PENALTY). ``reallocation_lag_months`` optionally puts
  them out of work for part of the year first, scaling annual earnings and
  hours by (1 - lag/12) on top of the penalty.

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

MARGINS = ("displacement", "wage_cut", "inactivity", "reallocation", "mixed")

#: Services destination divisions for the ``reallocation`` margin (SIC 2007):
#: 47 retail trade, 86 human health activities, 49 land transport (incl.
#: delivery/couriers), 56 food and beverage service activities — the sectors
#: the China-shock literature identifies as absorbing displaced manufacturing
#: labour.
REALLOCATION_DESTINATIONS = (47, 86, 49, 56)

#: Destination mix: each division's share of employee grossing weight WITHIN
#: the four destinations, measured on the FRS 2024-25 employee population at
#: period 2026 (analysis/reallocation_calibration.py). 47: 1.538m,
#: 86: 2.389m, 49: 0.480m, 56: 0.807m of 5.214m employees.
DESTINATION_SHARES = (0.2951, 0.4581, 0.0920, 0.1548)

#: FRS-CALIBRATED services wage penalty (Jul 2026). Weighted mean annual
#: employee earnings: exposed goods-producing divisions (the intensity
#: table's SIC 10-32) £48,272 vs the four services destinations £34,594 — a
#: ratio of 0.717, i.e. a 28.3% ANNUAL EARNINGS penalty. It embeds the hours
#: fall (destination mean 1,701 vs source 2,047 annual hours), which is the
#: economically relevant quantity for household income. Controlling crudely
#: for hours and age (weighted OLS of log annual earnings on a destination
#: dummy, log hours, age and age^2 over the same population) leaves a 14.0%
#: pure hourly-wage penalty — the lower-bound sensitivity.
DEFAULT_REALLOCATION_PENALTY = 0.283
#: Hours/age-controlled lower bound (see above).
HOURLY_REALLOCATION_PENALTY = 0.140

#: POST-SHOCK UNIVERSAL CREDIT TAKE-UP among benefit units containing a newly
#: displaced / inactive / reallocated worker.
#:
#: WHY THIS PARAMETER EXISTS. ``would_claim_uc`` is a STORED BOOLEAN INPUT in
#: the FRS microdata, not a behavioural formula: policyengine-uk-data draws it
#: once at dataset build time, anchoring reported UC recipients to True and
#: filling the remainder so that the flag rate over ALL benefit units hits the
#: calibration target (0.55). It is therefore (i) a population-wide flag share,
#: NOT a take-up rate among the entitled — and so not comparable to the ~80 per
#: cent take-up-among-entitled assumed in Resolution Foundation / IFS work —
#: and (ii) conditioned entirely on PRE-SHOCK circumstances. Most exposed
#: workers were employed and non-entitled when the draw was made, so their
#: stored flag carries no information about whether their family would claim
#: once the earner loses their job. Carrying the baseline draw through the
#: shock (measured take-up among the displaced post-shock was 0.469, BELOW the
#: population flag rate) models a newly unemployed family as not claiming
#: precisely because it was not claiming while in work. We therefore RE-DRAW
#: the flag post-shock for AFFECTED benefit units at ``uc_takeup``, from an
#: independent RNG stream (seeded UC_TAKEUP_SEED_OFFSET + seed) so that the
#: displacement draw is bit-identical across take-up values. Unaffected benefit
#: units keep their baseline draw untouched.
DEFAULT_UC_TAKEUP = 0.80
#: RNG stream offset for the post-shock take-up re-draw (independent of the
#: displacement, destination and LCWRA streams).
UC_TAKEUP_SEED_OFFSET = 900_000


@dataclass(frozen=True)
class TradeShockScenario:
    name: str
    tariff_scenario: str  # "full_tariff" | "epd"
    margin: str  # "displacement" | "wage_cut" | "inactivity"
    elasticity: float = DEFAULT_ELASTICITY
    passthrough: float = DEFAULT_PASSTHROUGH
    inactivity_age_threshold: int = 50
    #: share of inactive (older displaced) workers flagged LCWRA; 1.0 is the
    #: paper's upper-bound assumption, 0.5 the sensitivity variant.
    lcwra_takeup: float = 1.0
    #: reallocation margin: proportional cut to annual earnings on moving to
    #: services (FRS-calibrated; see DEFAULT_REALLOCATION_PENALTY).
    reallocation_penalty: float = DEFAULT_REALLOCATION_PENALTY
    #: months out of work before the services job starts (0 = instant).
    reallocation_lag_months: float = 0.0
    #: post-shock UC take-up among AFFECTED benefit units (see
    #: DEFAULT_UC_TAKEUP for why the baseline stored flag is not usable).
    uc_takeup: float = DEFAULT_UC_TAKEUP
    #: Mixed margin: share of each sector's expected earnings loss delivered
    #: through Bernoulli displacement; the remainder is a survivor wage cut.
    displacement_share: float = 0.5

    def __post_init__(self):
        if not 0.0 <= self.uc_takeup <= 1.0:
            raise ValueError("uc_takeup must lie in [0, 1]")
        if self.margin not in MARGINS:
            raise ValueError(f"unknown margin {self.margin!r}; use one of {MARGINS}")
        if not 0.0 <= self.reallocation_penalty < 1.0:
            raise ValueError("reallocation_penalty must lie in [0, 1)")
        if not 0.0 <= self.reallocation_lag_months <= 12.0:
            raise ValueError("reallocation_lag_months must lie in [0, 12]")
        if not 0.0 <= self.displacement_share <= 1.0:
            raise ValueError("displacement_share must lie in [0, 1]")


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
    """Boolean mask from independent Bernoulli sector-shock draws.

    Every employed record in division j has inclusion probability shock_j,
    making weighted headcount and wage-bill losses unbiased regardless of
    survey-weight dispersion.
    """
    rng = np.random.default_rng(seed)
    employed = persons["employment_income"].to_numpy(dtype=float) > 0
    shock = _person_shock(persons, scenario)
    return employed & (rng.random(len(persons)) < shock)


def apply_displacement(
    persons: pd.DataFrame,
    scenario: TradeShockScenario,
    seed: int = 0,
) -> pd.DataFrame:
    """Displacement (or inactivity) margin: shocked copy of the person table.

    Adds boolean columns ``displaced`` (out of work), ``inactive`` (subset
    of displaced who flow to inactivity — empty under pure displacement) and
    ``lcwra`` (subset of inactive flagged limited-capability-for-WRA; equals
    ``inactive`` when lcwra_takeup is 1.0, an independent thinning otherwise,
    drawn from a separate stream so the displacement draw is unchanged).
    """
    shocked = persons.copy()
    displaced = draw_displaced(persons, scenario, seed=seed)
    shocked["displaced"] = displaced
    age = persons["age"].to_numpy(dtype=float)
    if scenario.margin == "inactivity":
        inactive = displaced & (age >= scenario.inactivity_age_threshold)
        shocked["inactive"] = inactive
        if scenario.lcwra_takeup >= 1.0:
            shocked["lcwra"] = inactive
        else:
            thin_rng = np.random.default_rng((seed, 7_002_026))
            shocked["lcwra"] = inactive & (
                thin_rng.random(len(persons)) < scenario.lcwra_takeup
            )
    else:
        shocked["inactive"] = np.zeros(len(persons), dtype=bool)
        shocked["lcwra"] = np.zeros(len(persons), dtype=bool)
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
    shocked["lcwra"] = np.zeros(len(persons), dtype=bool)
    return shocked


def apply_mixed_margin(
    persons: pd.DataFrame,
    scenario: TradeShockScenario,
    seed: int = 0,
) -> pd.DataFrame:
    """Deliver one sector shock through a mix of job loss and wage cuts.

    Let ``s`` be a worker's sector shock and ``lambda`` the displacement
    share. Displacement probability is ``p = lambda * s``. A survivor takes
    wage cut ``c = (s - p) / (1 - p)``. Therefore the expected earnings loss
    is exactly ``p + (1-p)c = s`` for every worker, at every mixture. The
    endpoints reproduce the pure wage-cut and displacement margins.
    """
    if scenario.margin != "mixed":
        raise ValueError("apply_mixed_margin requires a mixed scenario")
    shocked = persons.copy()
    earnings = persons["employment_income"].to_numpy(dtype=float)
    employed = earnings > 0
    sector_shock = _person_shock(persons, scenario)
    lam = float(scenario.displacement_share)
    p = lam * sector_shock
    if (p[employed] >= 1.0).any():
        raise ValueError("mixed-margin displacement probability must be below one")
    rng = np.random.default_rng(seed)
    displaced = employed & (rng.random(len(persons)) < p)
    survivor_cut = np.divide(
        sector_shock - p,
        1.0 - p,
        out=np.zeros_like(sector_shock),
        where=(1.0 - p) > 0,
    )
    new_earnings = np.where(
        displaced,
        0.0,
        np.where(employed, earnings * (1.0 - survivor_cut), earnings),
    )
    shocked["employment_income"] = new_earnings
    shocked["displaced"] = displaced
    shocked["inactive"] = np.zeros(len(persons), dtype=bool)
    shocked["lcwra"] = np.zeros(len(persons), dtype=bool)
    shocked["earnings_changed"] = employed & ~np.isclose(new_earnings, earnings)
    return shocked


def _blank_reallocation(shocked: pd.DataFrame) -> pd.DataFrame:
    shocked["reallocated"] = np.zeros(len(shocked), dtype=bool)
    shocked["destination_division"] = np.full(len(shocked), np.nan)
    return shocked


def draw_destinations(
    reallocated: np.ndarray,
    seed: int = 0,
    destinations=REALLOCATION_DESTINATIONS,
    shares=DESTINATION_SHARES,
) -> np.ndarray:
    """Destination SIC division per reallocated worker (NaN otherwise).

    Multinomial over ``destinations`` with probabilities ``shares`` (the
    destinations' FRS employee-weight shares), drawn from a SEPARATE random
    stream from the displacement draw so the drawn worker set is byte-identical
    to the displacement family's under the same seed.
    """
    probs = np.asarray(shares, dtype=float)
    probs = probs / probs.sum()
    rng = np.random.default_rng((seed, 4_010_2026))
    out = np.full(len(reallocated), np.nan)
    idx = np.flatnonzero(reallocated)
    out[idx] = rng.choice(np.asarray(destinations, dtype=float), size=idx.size, p=probs)
    return out


def apply_reallocation(
    persons: pd.DataFrame,
    scenario: TradeShockScenario,
    seed: int = 0,
) -> pd.DataFrame:
    """Reallocation margin: displaced workers re-employed in services.

    The drawn set is EXACTLY the displacement family's (same ``draw_displaced``
    call, same seed) — the two families are paired draw-for-draw. Instead of
    losing their job, each drawn worker is assigned a services destination
    division (``draw_destinations``) and keeps

        earnings_new = earnings_old x (1 - penalty) x (1 - lag_months / 12)

    where ``penalty`` is the FRS-calibrated services earnings penalty and the
    lag factor represents months spent out of work before re-entry (annual
    earnings and annual hours are both scaled by it; the worker is EMPLOYED
    at the end of the year, which is what the annual static microsimulation
    scores). ``displaced``/``inactive``/``lcwra`` are all False: nobody is out
    of work at the point of measurement.
    """
    if scenario.margin != "reallocation":
        raise ValueError("apply_reallocation requires a reallocation scenario")
    shocked = persons.copy()
    reallocated = draw_displaced(persons, scenario, seed=seed)
    lag_factor = 1.0 - scenario.reallocation_lag_months / 12.0
    factor = (1.0 - scenario.reallocation_penalty) * lag_factor
    earnings = shocked["employment_income"].to_numpy(dtype=float)
    shocked["employment_income"] = np.where(reallocated, earnings * factor, earnings)
    shocked["reallocated"] = reallocated
    shocked["destination_division"] = draw_destinations(reallocated, seed=seed)
    shocked["reallocation_hours_factor"] = np.where(reallocated, lag_factor, 1.0)
    shocked["displaced"] = np.zeros(len(persons), dtype=bool)
    shocked["inactive"] = np.zeros(len(persons), dtype=bool)
    shocked["lcwra"] = np.zeros(len(persons), dtype=bool)
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
        shocked = _blank_reallocation(apply_wage_cut(persons, scenario))
    elif scenario.margin == "mixed":
        shocked = _blank_reallocation(apply_mixed_margin(persons, scenario, seed=seed))
    elif scenario.margin == "reallocation":
        shocked = apply_reallocation(persons, scenario, seed=seed)
    else:
        shocked = _blank_reallocation(apply_displacement(persons, scenario, seed=seed))
    # Carried to build_shocked_simulation, which needs the take-up rate and
    # the draw seed to re-draw would_claim_uc for affected benefit units.
    shocked.attrs["uc_takeup"] = float(scenario.uc_takeup)
    shocked.attrs["seed"] = int(seed)
    return shocked


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


def lcwra_benunit_addon(sim, lcwra, monthly: float) -> np.ndarray:
    """Annual UC health-element addon per BENEFIT UNIT (£/year).

    Universal Credit pays AT MOST ONE limited-capability-for-work-related-
    activity element per benefit unit. The addon must therefore be a
    benunit-level indicator — does this benunit contain ANY newly flagged
    member — and NOT a person-level sum, which double-counts benefit units
    with two flagged members (e.g. a displaced over-50 couple).
    """
    flagged_bu = np.asarray(
        sim.map_result(np.asarray(lcwra, dtype=float), "person", "benunit"), dtype=float
    )
    addon = (flagged_bu > 0).astype(float) * monthly * 12.0
    if addon.size and addon.max() > monthly * 12.0 + 1e-6:
        raise RuntimeError(
            "LCWRA addon exceeds one annual health element for some benefit "
            "unit (flagged members double-counted)."
        )
    return addon


def merge_lcwra_element(base_element, addon) -> np.ndarray:
    """Apply a newly triggered LCWRA element without ever stacking elements."""
    base = np.asarray(base_element, dtype=float)
    new = np.asarray(addon, dtype=float)
    if base.shape != new.shape:
        raise ValueError("baseline LCWRA element and addon must have the same shape")
    return np.maximum(base, new)


def affected_mask(shocked_table) -> np.ndarray:
    """Persons whose labour-market circumstances changed in this draw.

    This explicit marker complements the earnings comparison in
    ``redraw_uc_takeup``. It covers non-earnings changes such as a sector or
    employment-status transition.
    """
    n = len(shocked_table)
    out = np.zeros(n, dtype=bool)
    for column in ("displaced", "inactive", "reallocated", "earnings_changed"):
        if column in shocked_table:
            out |= shocked_table[column].to_numpy(dtype=bool)
    return out


def redraw_uc_takeup(
    sim,
    baseline_sim,
    shocked_table,
    period,
    uc_takeup: float | None = None,
    seed: int | None = None,
    baseline_potential_award: np.ndarray | None = None,
) -> np.ndarray:
    """Re-draw take-up for changed units with positive post-shock UC potential.

    See DEFAULT_UC_TAKEUP for the full justification: the baseline flag is a
    stored draw conditioned on PRE-SHOCK circumstances (and calibrated as a
    population-wide share, not take-up among the entitled). The same rule is
    applied to every adjustment margin: a benefit unit is redrawn only if its
    labour-market circumstances changed and its potential UC award moves from
    zero at baseline to positive post-shock. The draw uses an independent RNG
    stream, leaving labour-market draws, existing eligible units, and unchanged
    or ineligible units untouched.

    Returns the benunit-level boolean flag actually applied. Hard-error
    contract: a silently-rejected ``set_input`` would leave the stale flags in
    place and change every downstream result, so the applied array is read back
    and verified.
    """
    if uc_takeup is None:
        uc_takeup = float(shocked_table.attrs.get("uc_takeup", DEFAULT_UC_TAKEUP))
    if seed is None:
        seed = int(shocked_table.attrs.get("seed", 0))
    if not 0.0 <= uc_takeup <= 1.0:
        raise ValueError("uc_takeup must lie in [0, 1]")

    baseline_flag = np.asarray(
        baseline_sim.calculate("would_claim_uc", period=period, map_to="benunit").values,
        dtype=bool,
    )
    # Use the economic change, rather than the scenario label, to identify the
    # population whose pre-shock claiming draw is stale.  This deliberately
    # includes wage cuts and reallocation as well as job loss.
    baseline_earnings = np.asarray(
        baseline_sim.calculate("employment_income", period=period, map_to="person").values,
        dtype=float,
    )
    shocked_earnings = shocked_table["employment_income"].to_numpy(dtype=float)
    changed = ~np.isclose(shocked_earnings, baseline_earnings, rtol=0.0, atol=1e-8)
    changed |= affected_mask(shocked_table)
    if not changed.any():
        return baseline_flag
    changed_bu = (
        np.asarray(
            sim.map_result(changed.astype(float), "person", "benunit"), dtype=float
        )
        > 0
    )

    # Measure post-shock potential UC with claiming switched on.  The stored
    # would_claim_uc input gates the award, so the observed award cannot be
    # used to determine entitlement without this temporary all-True override.
    sim.set_input("would_claim_uc", period, np.ones(baseline_flag.size, dtype=bool))
    enabled = np.asarray(
        sim.calculate("would_claim_uc", period=period, map_to="benunit").values,
        dtype=bool,
    )
    if not enabled.all():
        raise RuntimeError(
            "post-shock UC take-up re-draw not applied: temporary entitlement "
            "calculation could not enable would_claim_uc."
        )
    potential_award = np.asarray(
        sim.calculate("universal_credit", period=period, map_to="benunit").values,
        dtype=float,
    )
    if baseline_potential_award is None:
        # Unit-level callers without a full dataset treat the baseline as not
        # entitled. Production callers provide the all-claim baseline award.
        baseline_potential_award = np.zeros_like(potential_award)
    baseline_potential_award = np.asarray(baseline_potential_award, dtype=float)
    if baseline_potential_award.shape != potential_award.shape:
        raise ValueError("baseline and post-shock potential UC awards must align")

    # Apply the take-up draw only to benefit units that become newly entitled.
    # Existing claimants and existing eligible non-claimants retain their
    # baseline claiming state; otherwise a small wage cut would inadvertently
    # impose a population-wide take-up reform on all already-entitled workers.
    redraw_bu = (
        changed_bu
        & (baseline_potential_award <= 1e-8)
        & (potential_award > 1e-8)
    )
    if not redraw_bu.any():
        sim.set_input("would_claim_uc", period, baseline_flag)
        sim._invalidate_all_caches()
        return baseline_flag
    rng = np.random.default_rng(UC_TAKEUP_SEED_OFFSET + int(seed))
    draw = rng.random(baseline_flag.size) < uc_takeup
    new_flag = np.where(redraw_bu, draw, baseline_flag)
    sim.set_input("would_claim_uc", period, new_flag)
    # universal_credit was evaluated under the temporary all-claim input.
    # Clear formula outputs while preserving the final user inputs so all
    # downstream metrics recompute under new_flag rather than using that cache.
    sim._invalidate_all_caches()
    applied = np.asarray(
        sim.calculate("would_claim_uc", period=period, map_to="benunit").values, dtype=bool
    )
    if not np.array_equal(applied, new_flag):
        raise RuntimeError(
            "post-shock UC take-up re-draw not applied: would_claim_uc in the "
            "shocked simulation does not match the re-drawn flag."
        )
    return new_flag


def build_shocked_simulation(dataset, baseline_sim, shocked_table, period):
    """One shared constructor for the shocked simulation (every pipeline).

    Displaced-not-inactive workers become UNEMPLOYED; inactive workers become
    OTHER_INACTIVE and (per the shocked table's ``lcwra`` column) are flagged
    uc_limited_capability_for_WRA = True on top of any baseline flag, so
    their Universal Credit includes the LCWRA health element.
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
    reallocated = (
        shocked_table["reallocated"].to_numpy()
        if "reallocated" in shocked_table
        else np.zeros(len(displaced), dtype=bool)
    )
    hours_factor = (
        shocked_table["reallocation_hours_factor"].to_numpy(dtype=float)
        if "reallocation_hours_factor" in shocked_table
        else np.ones(len(displaced), dtype=float)
    )
    for var in TRANSITION_ZEROED_VARIABLES:
        values = baseline_sim.calculate(var, period=period, map_to="person").values.astype(float)
        values[displaced] = 0.0
        if var == "hours_worked":
            values = values * hours_factor
        sim.set_input(var, period, values)
    if reallocated.any():
        # Literal sector switch: move the reallocated workers' SIC industry
        # division to their drawn services destination. Verified below (a
        # silently-rejected set_input would leave them in manufacturing and
        # make the margin indistinguishable from a plain wage cut).
        sic = baseline_sim.calculate(
            "sic_industry_division", period=period, map_to="person"
        ).values.astype(float)
        destination = shocked_table["destination_division"].to_numpy(dtype=float)
        sic = np.where(reallocated, destination, sic)
        sim.set_input("sic_industry_division", period, sic)
        applied_sic = sim.calculate(
            "sic_industry_division", period=period, map_to="person"
        ).values.astype(float)
        if not np.array_equal(applied_sic[reallocated], destination[reallocated]):
            raise RuntimeError(
                "sector reallocation not applied: reallocated persons' "
                "sic_industry_division does not match their drawn services "
                "destination in the shocked simulation."
            )
    status = baseline_sim.calculate("employment_status", period=period, map_to="person").values.astype(object)
    status[displaced] = "UNEMPLOYED"
    status[inactive] = "OTHER_INACTIVE"
    sim.set_input("employment_status", period, status)
    lcwra = (
        shocked_table["lcwra"].to_numpy()
        if "lcwra" in shocked_table
        else np.zeros(len(status), dtype=bool)
    )
    if lcwra.any():
        # uc_LCWRA_element is a STORED INPUT in the FRS h5 (imputed survey
        # receipt), so setting uc_limited_capability_for_WRA alone never
        # reaches UC: the element's formula is shadowed by the stored array.
        # We therefore (a) flag the person for consistency and (b) ensure the
        # benunit has one annual LCWRA amount, leaving everyone else's element
        # untouched (recomputing from the formula would repriced the whole
        # population off is_disabled_for_benefits and break comparability
        # with the baseline simulation).
        flag = baseline_sim.calculate(
            "uc_limited_capability_for_WRA", period=period, map_to="person"
        ).values.astype(bool)
        sim.set_input("uc_limited_capability_for_WRA", period, flag | lcwra)
        base_element = baseline_sim.calculate(
            "uc_LCWRA_element", period=period, map_to="benunit"
        ).values.astype(float)
        monthly = float(
            sim.tax_benefit_system.parameters(
                f"{period}-01-01"
            ).gov.dwp.universal_credit.elements.disabled.amount
        )
        addon = lcwra_benunit_addon(sim, lcwra, monthly)
        # UC pays at most one LCWRA element per benefit unit.  A unit already
        # receiving the stored element must not receive a second full element.
        sim.set_input("uc_LCWRA_element", period, merge_lcwra_element(base_element, addon))
        applied_element = sim.calculate(
            "uc_LCWRA_element", period=period, map_to="person"
        ).values.astype(float)
        if not (applied_element[lcwra] >= monthly * 12.0 - 1e-6).all():
            raise RuntimeError(
                "LCWRA element not applied: some inactive persons' benefit "
                "units lack the UC health element in the shocked simulation."
            )
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
    if reallocated.any():
        # Reallocated workers switch SECTOR, not employment state: their
        # employment_status must be exactly their baseline one (FT_EMPLOYED,
        # PT_EMPLOYED, ...) and in particular must never be UNEMPLOYED.
        baseline_status = np.asarray(
            baseline_sim.calculate("employment_status", period=period, map_to="person").values,
            dtype=str,
        )
        if not (applied[reallocated] == baseline_status[reallocated]).all():
            raise RuntimeError(
                "reallocated persons' employment_status changed: they switch "
                "sector, they do not lose their job."
            )
    # Post-shock UC take-up. The baseline would_claim_uc flag is a STORED draw
    # conditioned on PRE-SHOCK circumstances (and calibrated as a
    # population-wide share, not take-up among the entitled), so it is not
    # informative about whether a newly displaced family would claim; carrying
    # it through models the newly unemployed as not claiming because they were
    # not claiming while in work. Re-drawn here for affected benefit units only,
    # from an independent RNG stream, so the displacement draw is unchanged.
    # Applied LAST so that it is never shadowed by an earlier cached value; it
    # has its own hard-error check inside redraw_uc_takeup.
    baseline_potential_award = getattr(
        baseline_sim, "_trade_shock_uc_potential_award", None
    )
    if baseline_potential_award is None:
        potential_sim = Microsimulation(dataset=dataset)
        baseline_flag = np.asarray(
            baseline_sim.calculate(
                "would_claim_uc", period=period, map_to="benunit"
            ).values,
            dtype=bool,
        )
        potential_sim.set_input(
            "would_claim_uc", period, np.ones(baseline_flag.size, dtype=bool)
        )
        baseline_potential_award = np.asarray(
            potential_sim.calculate(
                "universal_credit", period=period, map_to="benunit"
            ).values,
            dtype=float,
        )
        baseline_sim._trade_shock_uc_potential_award = baseline_potential_award
    redraw_uc_takeup(
        sim,
        baseline_sim,
        shocked_table,
        period,
        baseline_potential_award=baseline_potential_award,
    )
    return sim
