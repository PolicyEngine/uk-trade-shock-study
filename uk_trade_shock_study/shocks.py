"""Adjustment-margin families for the tariff shock on FRS workers.

The derived per-SIC earnings-shock sizes (exposure.sector_earnings_shocks)
are realised on employees in exposed divisions through three families — the
paper's central axis, because Universal Credit replaces unemployment far more
generously than in-work wage cuts:

- ``displacement`` (ADH short-run): the sector earnings shock is delivered as
  JOB LOSS. The declared person risks are integrated using Bernoulli,
  systematic or balanced repeated assignments. Survey weights never define
  baseline person risk; the balanced submission design uses them only to
  condition realised aggregate losses. Displaced workers move to
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

MARGINS = (
    "displacement",
    "wage_cut",
    "concentrated_wage_cut",
    "inactivity",
    "reallocation",
    "mixed",
    "transition",
)

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
#: independent RNG stream (tuple-seeded (seed, UC_TAKEUP_SEED_OFFSET)) so the
#: displacement draw is bit-identical across take-up values. Unaffected benefit
#: units keep their baseline draw untouched.
DEFAULT_UC_TAKEUP = 0.80
#: Stream tag for the post-shock take-up re-draw, used as the SECOND element
#: of a TUPLE seed (seed, UC_TAKEUP_SEED_OFFSET). Tuple seeding keeps the
#: stream disjoint from the plain-integer displacement stream for every seed
#: (an additive offset would overlap it once seeds reach the offset) and from
#: the destination/LCWRA tuple streams, which use different second elements.
UC_TAKEUP_SEED_OFFSET = 900_000

#: WHICH CHANGED BENEFIT UNITS HAVE THEIR CLAIMING FLAG RE-DRAWN.
#:
#: ``new_entitlement`` (default, and the convention behind every stored
#: result) re-draws only benefit units whose POTENTIAL UC award moves from
#: zero at baseline to positive post-shock. It answers "what happens to
#: families the shock pushes onto UC from scratch", and deliberately leaves
#: already-entitled units on their stored flag so that a small wage cut does
#: not silently impose a population-wide take-up reform.
#:
#: ``all_entitled`` re-draws EVERY changed benefit unit with a positive
#: post-shock potential award, including those already entitled at baseline.
#: This is the scope that binds empirically: in the exposed population most
#: benefit units that end the shock with a positive potential award were
#: ALREADY potentially entitled at baseline (in-work UC), so the
#: ``new_entitlement`` set can be — and at the paper's calibration is —
#: EMPTY, which makes a take-up grid run under it exactly inert. The
#: ``all_entitled`` scope therefore bounds how much the claiming margin can
#: move the headline cushioning rate. It is an upper bound on the claiming
#: response, not a behavioural estimate: it assumes the shock re-opens the
#: claiming decision for every affected entitled family.
#:
#: Both scopes consume the SAME independent tuple-seeded RNG stream
#: (seed, UC_TAKEUP_SEED_OFFSET), drawn over all benefit units and then
#: masked, so a given benefit unit's draw is identical across scopes and the
#: labour-market draws are untouched in either case.
UC_TAKEUP_SCOPES = ("new_entitlement", "all_entitled")
DEFAULT_UC_TAKEUP_SCOPE = "new_entitlement"


def _validate_uc_takeup_scope(scope: str) -> str:
    if scope not in UC_TAKEUP_SCOPES:
        raise ValueError(
            f"uc_takeup_scope must be one of {UC_TAKEUP_SCOPES}; got {scope!r}"
        )
    return scope


@dataclass(frozen=True)
class TradeShockScenario:
    name: str
    tariff_scenario: str  # "full_tariff" | "epd"
    margin: str  # adjustment-margin family, including ``transition``
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
    #: WHICH changed benefit units are re-drawn at ``uc_takeup``: see
    #: UC_TAKEUP_SCOPES. The default preserves the published convention.
    uc_takeup_scope: str = DEFAULT_UC_TAKEUP_SCOPE
    #: Mixed margin: share of each sector's expected earnings loss delivered
    #: through Bernoulli displacement; the remainder is a survivor wage cut.
    displacement_share: float = 0.5
    #: Optional person column used to vary displacement risk within SIC.
    #: Probabilities are recalibrated within division to preserve the expected
    #: weighted wage-bill loss. None retains uniform central assignment.
    selection_risk_column: str | None = None
    #: Assignment design for job-loss margins. ``systematic`` preserves the
    #: intended first-order probabilities but fixes each industry's realised
    #: record count to floor/ceil of its expected count and spreads selections
    #: across age and earnings strata. ``balanced`` draws repeated systematic
    #: candidates and retains the one closest to industry wage-bill and
    #: weighted-headcount targets. ``bernoulli`` is retained as a transparent
    #: sensitivity comparator.
    selection_method: str = "bernoulli"
    #: Scale applied to every sector earnings shock. Values 0.25 and 0.5 are
    #: annual duration-equivalent stresses for three and six months: they do
    #: not simulate a partial-year unemployment spell.
    duration_equivalent: float = 1.0
    #: Divisions set to zero exposure for leave-one-sector-out diagnostics.
    excluded_divisions: tuple[int, ...] = ()

    @property
    def wage_bill_incidence(self) -> float:
        """Canonical name for the legacy ``passthrough`` constructor field."""
        return self.passthrough

    def __post_init__(self):
        if not 0.0 <= self.uc_takeup <= 1.0:
            raise ValueError("uc_takeup must lie in [0, 1]")
        _validate_uc_takeup_scope(self.uc_takeup_scope)
        if self.margin not in MARGINS:
            raise ValueError(f"unknown margin {self.margin!r}; use one of {MARGINS}")
        if not 0.0 <= self.reallocation_penalty < 1.0:
            raise ValueError("reallocation_penalty must lie in [0, 1)")
        if not 0.0 <= self.reallocation_lag_months <= 12.0:
            raise ValueError("reallocation_lag_months must lie in [0, 12]")
        if not 0.0 <= self.displacement_share <= 1.0:
            raise ValueError("displacement_share must lie in [0, 1]")
        if self.selection_method not in ("bernoulli", "systematic", "balanced"):
            raise ValueError(
                "selection_method must be 'bernoulli', 'systematic' or 'balanced'"
            )
        if not 0.0 < self.duration_equivalent <= 1.0:
            raise ValueError("duration_equivalent must lie in (0, 1]")


#: Scenario presets: {full_tariff, epd} x {displacement, wage_cut, inactivity}.
PRESETS = {
    f"{tariff}_{margin}": TradeShockScenario(f"{tariff}_{margin}", tariff, margin)
    for tariff in ("full_tariff", "epd")
    for margin in MARGINS
}

#: LITERATURE-DISCIPLINED SENSITIVITY for the mixed margin's split. The quasi-
#: experimental rent-sharing literature provides a plausible scale for
#: incumbent ("survivor") wage responses:
#:
#: - Garin & Silverio (ReStud 2024): incumbent-wage elasticity to firm output
#:   shocks of ~0.15;
#: - Card, Cardoso, Heining & Kline (JOLE 2018 survey): rent-sharing
#:   elasticities of 0.05-0.15;
#: - Hummels, Jorgensen, Munch & Xiang (AER 2014): within-job wage responses
#:   to trade shocks of 0.03-0.08.
#:
#: We use 0.15 to define a transparent 15/85 wage-cut/displacement scenario.
#: This is NOT an empirical calibration: an elasticity of wages to firm
#: output or rents is not algebraically the share of an aggregate wage-bill
#: loss borne by survivor wages. Identifying that share requires linked
#: employer-worker estimates of wages, hours, employment and re-employment.
RENT_SHARING_ELASTICITY = 0.15


def rent_sharing_displacement_share(
    elasticity: float = RENT_SHARING_ELASTICITY,
) -> float:
    """Construct a stylised split from a rent-sharing sensitivity value.

    In ``apply_mixed_margin``, lambda = ``displacement_share`` and a worker in
    division j with sector shock s_j faces displacement probability
    p = lambda * s_j, while survivors take cut c = (s_j - p) / (1 - p). The
    EXPECTED wage-bill loss of division j therefore splits exactly as

        displacement:   p * B_j           = lambda * s_j * B_j
        survivor cuts:  (1 - p) * c * B_j = (1 - lambda) * s_j * B_j

    so the survivor-wage-cut share of the imposed sector wage-bill loss is
    (1 - lambda). We set that scenario share numerically equal to the supplied
    sensitivity value for transparency only. Do not interpret this function
    as converting a reduced-form rent-sharing elasticity into an empirically
    identified extensive/intensive-margin decomposition.
    """
    if not 0.0 <= elasticity <= 1.0:
        raise ValueError("rent-sharing elasticity must lie in [0, 1]")
    return 1.0 - elasticity


#: Mixed-margin presets with a literature-disciplined stylised split: wage
#: cuts absorb RENT_SHARING_ELASTICITY (15%) of each sector's wage-bill loss,
#: displacement the remaining 85% (displacement_share = 0.85).
RENT_SHARING_PRESETS = {
    f"{tariff}_rentsharing": TradeShockScenario(
        f"{tariff}_rentsharing",
        tariff,
        "mixed",
        displacement_share=rent_sharing_displacement_share(),
    )
    for tariff in ("full_tariff", "epd")
}

# Named central scenarios for the paper's incidence exercise.  Keep the
# explicit ``mixed_central`` label alongside the historical ``rentsharing``
# alias: the latter describes how the 85/15 split was motivated, while this
# name makes the estimand (a mixed adjustment path conditional on a common
# earnings shock) unambiguous in result files and tables.
MIXED_CENTRAL_PRESETS = {
    f"{tariff}_mixed_central": TradeShockScenario(
        f"{tariff}_mixed_central",
        tariff,
        "mixed",
        displacement_share=rent_sharing_displacement_share(),
    )
    for tariff in ("full_tariff", "epd")
}

TRANSITION_PRESETS = {
    f"{tariff}_transition_central": TradeShockScenario(
        f"{tariff}_transition_central", tariff, "transition",
        displacement_share=0.70, reallocation_penalty=DEFAULT_REALLOCATION_PENALTY,
        reallocation_lag_months=3.0,
    ) for tariff in ("full_tariff", "epd")
}


def _person_shock(persons: pd.DataFrame, scenario: TradeShockScenario) -> np.ndarray:
    shock = person_earnings_shock(
        persons["sic_division"],
        scenario.tariff_scenario,
        elasticity=scenario.elasticity,
        wage_bill_incidence=scenario.wage_bill_incidence,
    )
    shock = np.asarray(shock, dtype=float) * scenario.duration_equivalent
    if scenario.excluded_divisions:
        division = persons["sic_division"].to_numpy(dtype=float)
        shock[np.isin(division, scenario.excluded_divisions)] = 0.0
    return shock


def systematic_probability_sample(
    probabilities: np.ndarray,
    groups: np.ndarray,
    age: np.ndarray,
    earnings: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Systematic probability sampling within groups.

    For each industry, records are ordered by broad age and within-industry
    earnings ranks, with a random tie-breaker. A random-start systematic pass
    then selects cumulative-probability interval crossings. This preserves
    every record's first-order inclusion probability while reducing realised
    industry-count variation from Bernoulli sampling.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    groups = np.asarray(groups)
    age = np.asarray(age, dtype=float)
    earnings = np.asarray(earnings, dtype=float)
    if not (
        probabilities.shape == groups.shape == age.shape == earnings.shape
    ):
        raise ValueError("systematic-sampling inputs must have equal shape")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("sampling probabilities must lie in [0, 1]")

    selected = np.zeros(probabilities.size, dtype=bool)
    active_groups = pd.unique(groups[probabilities > 0])
    for group in active_groups:
        idx = np.flatnonzero((groups == group) & (probabilities > 0))
        if not idx.size:
            continue
        # Rank rather than cut at fixed currency values so every industry is
        # spread across its own earnings distribution.
        rank = pd.Series(earnings[idx]).rank(method="average", pct=True).to_numpy()
        age_band = np.floor(np.nan_to_num(age[idx], nan=40.0) / 10.0)
        tie_break = rng.random(idx.size)
        order = np.lexsort((tie_break, rank, age_band))
        ordered = idx[order]
        p = probabilities[ordered]
        cumulative = np.cumsum(p)
        total = float(cumulative[-1])
        start = float(rng.random())
        thresholds = start + np.arange(int(np.ceil(total)))
        thresholds = thresholds[thresholds < total]
        if thresholds.size:
            positions = np.searchsorted(cumulative, thresholds, side="right")
            selected[ordered[positions]] = True
    return selected


def balanced_probability_sample(
    probabilities: np.ndarray,
    persons: pd.DataFrame,
    rng: np.random.Generator,
    n_candidates: int = 256,
) -> np.ndarray:
    """Choose a systematic candidate close to weighted industry targets.

    This is a numerical-integration design, not a survey estimator. Repeated
    candidates share the declared first-order risk model; selecting the best
    balanced candidate intentionally trades exact marginal inclusion
    probabilities for much lower dependence on arbitrary high-weight records.
    Bernoulli results are always retained as the unbiased comparator.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    sic = persons["sic_division"].to_numpy()
    age = persons["age"].to_numpy(dtype=float)
    earnings = persons["employment_income"].to_numpy(dtype=float)
    weight = persons["weight"].to_numpy(dtype=float)
    wage_contribution = earnings * weight
    target_wage = probabilities * wage_contribution
    target_headcount = probabilities * weight
    divisions = pd.unique(sic[probabilities > 0])
    total_wage_target = float(target_wage.sum())
    total_headcount_target = float(target_headcount.sum())
    employed = probabilities > 0
    earnings_rank = np.zeros(probabilities.size, dtype=float)
    earnings_rank[employed] = (
        pd.Series(earnings[employed]).rank(method="average", pct=True).to_numpy()
    )
    earnings_band = np.minimum((earnings_rank * 3).astype(int), 2)
    age_band = np.select(
        [age < 35, age < 50],
        [0, 1],
        default=2,
    )

    best = np.zeros(probabilities.size, dtype=bool)
    best_score = float("inf")
    for _ in range(n_candidates):
        candidate = systematic_probability_sample(
            probabilities, sic, age, earnings, rng
        )
        realised_wage = float(wage_contribution[candidate].sum())
        realised_headcount = float(weight[candidate].sum())
        score = 10_000.0 * (
            (realised_wage - total_wage_target) / total_wage_target
        ) ** 2
        score += 100.0 * (
            (realised_headcount - total_headcount_target)
            / total_headcount_target
        ) ** 2
        for division in divisions:
            group = sic == division
            wage_target = float(target_wage[group].sum())
            headcount_target = float(target_headcount[group].sum())
            if wage_target > 0:
                target_share = wage_target / total_wage_target
                score += target_share * (
                    (float(wage_contribution[group & candidate].sum()) - wage_target)
                    / wage_target
                ) ** 2
            if headcount_target > 0:
                target_share = headcount_target / total_headcount_target
                score += 0.25 * target_share * (
                    (float(weight[group & candidate].sum()) - headcount_target)
                    / headcount_target
                ) ** 2
        # Preserve the main observed fiscal-risk gradients as marginal
        # balances. Crossed cells would be too sparse at the roughly ten-record
        # reference assignment.
        for strata in (earnings_band, age_band):
            for value in (0, 1, 2):
                group = employed & (strata == value)
                wage_target = float(target_wage[group].sum())
                headcount_target = float(target_headcount[group].sum())
                if wage_target > 0:
                    score += 1.0 * (wage_target / total_wage_target) * (
                        (
                            float(wage_contribution[group & candidate].sum())
                            - wage_target
                        )
                        / wage_target
                    ) ** 2
                if headcount_target > 0:
                    score += 0.25 * (
                        headcount_target / total_headcount_target
                    ) * (
                        (
                            float(weight[group & candidate].sum())
                            - headcount_target
                        )
                        / headcount_target
                    ) ** 2
        if score < best_score:
            best_score = score
            best = candidate
    return best


def draw_displaced(
    persons: pd.DataFrame,
    scenario: TradeShockScenario,
    seed: int = 0,
) -> np.ndarray:
    """Boolean displacement mask under the scenario's assignment design.

    Bernoulli preserves the declared person probabilities exactly. Systematic
    sampling preserves first-order probabilities while balancing record
    counts. Balanced repeated assignment conditions on weighted aggregate and
    observed-composition targets; its Bernoulli comparator remains necessary.
    """
    employed = persons["employment_income"].to_numpy(dtype=float) > 0
    shock = _person_shock(persons, scenario)
    probabilities = shock
    if scenario.selection_risk_column is not None:
        probabilities = risk_weighted_displacement_probabilities(
            persons, shock, scenario.selection_risk_column
        )
    rng = np.random.default_rng(seed)
    if scenario.selection_method == "bernoulli":
        return employed & (rng.random(len(persons)) < probabilities)
    active_probabilities = np.where(employed, probabilities, 0.0)
    if scenario.selection_method == "systematic":
        sampled = systematic_probability_sample(
            active_probabilities,
            persons["sic_division"].to_numpy(),
            persons["age"].to_numpy(dtype=float),
            persons["employment_income"].to_numpy(dtype=float),
            rng,
        )
    else:
        sampled = balanced_probability_sample(
            active_probabilities, persons, rng
        )
    return employed & sampled


def risk_weighted_displacement_probabilities(
    persons: pd.DataFrame,
    sector_shock: np.ndarray,
    risk_column: str,
) -> np.ndarray:
    """Vary risk within SIC while preserving each division's wage-bill loss."""
    from uk_trade_shock_study.lfs_imputation import calibrate_probabilities

    if risk_column not in persons:
        raise KeyError(f"selection risk column not found: {risk_column}")
    result = np.zeros(len(persons), dtype=float)
    earnings = persons["employment_income"].to_numpy(dtype=float)
    survey_weight = persons["weight"].to_numpy(dtype=float)
    risk = persons[risk_column].to_numpy(dtype=float)
    sic = persons["sic_division"].to_numpy(dtype=float)
    sector_shock = np.asarray(sector_shock, dtype=float)
    employed = earnings > 0
    for division in np.unique(sic[np.isfinite(sic) & employed]):
        selected = employed & (sic == division)
        target = float(sector_shock[selected][0])
        if target <= 0:
            continue
        sector_risk = risk[selected].copy()
        valid_risk = np.isfinite(sector_risk) & (sector_risk >= 0)
        fallback = (
            np.average(
                sector_risk[valid_risk],
                weights=(earnings[selected] * survey_weight[selected])[valid_risk],
            )
            if valid_risk.any()
            else 1.0
        )
        sector_risk[~valid_risk] = fallback
        result[selected] = calibrate_probabilities(
            sector_risk,
            target,
            np.ones(selected.sum(), dtype=bool),
            earnings[selected] * survey_weight[selected],
        )
    return result


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
    weighted aggregate earnings removed equals L exactly. The conservation
    property (wage-cut loss = Monte Carlo mean of displacement losses across
    seeds) is verified in tests/test_shocks.py::test_wage_bill_conservation.
    Non-exposed workers are untouched. Deterministic (no draw, no seed).
    """
    if scenario.margin != "wage_cut":
        raise ValueError("apply_wage_cut requires a wage_cut scenario")
    shocked = persons.copy()
    earnings = shocked["employment_income"].to_numpy(dtype=float)
    employed = earnings > 0
    shock = _person_shock(persons, scenario)
    if (shock[employed] >= 1.0).any():
        raise ValueError(
            "derived sector shock >= 100% of earnings for some workers; "
            "check elasticity/passthrough calibration."
        )
    new = np.where(employed, earnings * (1.0 - shock), earnings)
    # Sanity checks that CAN fire (the former realised-vs-target comparison
    # recomputed the same expression twice and was tautological — referee
    # point M6; the economic conservation property, wage-cut loss equalling
    # the Monte Carlo mean of displacement losses, is asserted in
    # tests/test_shocks.py::test_wage_bill_conservation).
    if (new < 0).any() or (new > earnings + 1e-9).any():
        raise AssertionError("wage cut produced negative earnings or a raise")
    if not np.array_equal(new[~employed], earnings[~employed]):
        raise AssertionError("wage cut touched non-employed records")
    shocked["employment_income"] = new
    shocked["displaced"] = np.zeros(len(persons), dtype=bool)
    shocked["inactive"] = np.zeros(len(persons), dtype=bool)
    shocked["lcwra"] = np.zeros(len(persons), dtype=bool)
    return shocked


def apply_concentrated_wage_cut(
    persons: pd.DataFrame,
    scenario: TradeShockScenario,
    seed: int = 0,
) -> pd.DataFrame:
    """Concentrated wage cut: the displacement draw's losses without job loss.

    FACTORIAL MIDDLE CELL (referee major point 1). The drawn worker set is
    EXACTLY the displacement family's (same ``draw_displaced`` call, same
    seed, same selection method), and each drawn worker loses their entire
    annual earnings — the same worker-level loss displacement imposes. But
    nobody's employment state changes: workers remain EMPLOYED with baseline
    hours, and only the earnings-linked deductions (pensions, statutory pay)
    scale with the earnings factor in build_shocked_simulation, exactly as
    for any other in-work earnings cut.

    Comparisons this cell licenses, holding the aggregate loss fixed:

    - broad wage cut vs concentrated wage cut  = concentration + worker
      selection, holding employment state fixed at EMPLOYED;
    - concentrated wage cut vs displacement    = the incremental effect of
      the employment-state change itself (status flag, hours zeroing, and
      the benefit rules they activate), holding the loss-bearing workers and
      their worker-level losses fixed.

    Because employment is binary, "diffuse displacement" is not defined, so
    the sequential path broad -> concentrated -> displaced is the unique
    monotone decomposition of the headline gap.  It is NOT a Shapley
    allocation: that would require valuing the undefined
    diffuse-displacement coalition.
    """
    if scenario.margin != "concentrated_wage_cut":
        raise ValueError(
            "apply_concentrated_wage_cut requires a concentrated_wage_cut scenario"
        )
    shocked = persons.copy()
    drawn = draw_displaced(persons, scenario, seed=seed)
    earnings = shocked["employment_income"].to_numpy(dtype=float)
    shocked["employment_income"] = np.where(drawn, 0.0, earnings)
    shocked["displaced"] = np.zeros(len(persons), dtype=bool)
    shocked["inactive"] = np.zeros(len(persons), dtype=bool)
    shocked["lcwra"] = np.zeros(len(persons), dtype=bool)
    shocked["earnings_changed"] = drawn
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


def apply_transition_margin(persons: pd.DataFrame, scenario: TradeShockScenario, seed: int = 0) -> pd.DataFrame:
    """Annualized transition-equivalent earnings path plus survivor cuts.

    ``displacement_share`` is the share of the sector loss borne by transition
    workers (rather than their probability).  If a transition-equivalent
    worker loses fraction ``d`` of annual earnings through a re-employment
    penalty and lag, its draw probability is ``p = lambda*s/d``; survivors
    take ``(1-lambda)*s/(1-p)``.  Thus expected loss is exactly ``s`` per
    worker.  The annual PolicyEngine run represents the lag through earnings
    and hours; it does not create a within-year unemployment spell or
    activate unemployment benefits.
    """
    if scenario.margin != "transition":
        raise ValueError("apply_transition_margin requires a transition scenario")
    shocked = _blank_reallocation(persons.copy())
    earnings = persons["employment_income"].to_numpy(dtype=float)
    employed = earnings > 0
    s = _person_shock(persons, scenario)
    lag_factor = 1.0 - scenario.reallocation_lag_months / 12.0
    factor = (1.0 - scenario.reallocation_penalty) * lag_factor
    d = 1.0 - factor
    if d <= 0:
        raise ValueError("transition penalty and lag must imply a positive loss")
    p = scenario.displacement_share * s / d
    if (p[employed] >= 1.0).any():
        raise ValueError("transition probability must be below one")
    rng = np.random.default_rng(seed)
    transitioned = employed & (rng.random(len(persons)) < p)
    survivor_cut = np.divide(
        (1.0 - scenario.displacement_share) * s,
        1.0 - p,
        out=np.zeros_like(s), where=(1.0 - p) > 0,
    )
    new = np.where(transitioned, earnings * factor,
                   np.where(employed, earnings * (1.0 - survivor_cut), earnings))
    shocked["employment_income"] = new
    shocked["displaced"] = np.zeros(len(persons), dtype=bool)
    shocked["inactive"] = np.zeros(len(persons), dtype=bool)
    shocked["lcwra"] = np.zeros(len(persons), dtype=bool)
    shocked["reallocated"] = transitioned
    shocked["destination_division"] = draw_destinations(transitioned, seed=seed)
    shocked["reallocation_hours_factor"] = np.where(transitioned, lag_factor, 1.0)
    shocked["transitioned"] = transitioned
    shocked["earnings_changed"] = employed & ~np.isclose(new, earnings)
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
    elif scenario.margin == "concentrated_wage_cut":
        shocked = _blank_reallocation(
            apply_concentrated_wage_cut(persons, scenario, seed=seed)
        )
    elif scenario.margin == "mixed":
        shocked = _blank_reallocation(apply_mixed_margin(persons, scenario, seed=seed))
    elif scenario.margin == "transition":
        shocked = apply_transition_margin(persons, scenario, seed=seed)
    elif scenario.margin == "reallocation":
        shocked = apply_reallocation(persons, scenario, seed=seed)
    else:
        shocked = _blank_reallocation(apply_displacement(persons, scenario, seed=seed))
    # Carried to build_shocked_simulation, which needs the take-up rate, the
    # re-draw scope and the draw seed to re-draw would_claim_uc for affected
    # benefit units.
    shocked.attrs["uc_takeup"] = float(scenario.uc_takeup)
    shocked.attrs["uc_takeup_scope"] = _validate_uc_takeup_scope(
        scenario.uc_takeup_scope
    )
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

#: Subset of TRANSITION_ZEROED_VARIABLES that are earnings-linked deductions
#: or earnings-linked statutory pay. For NON-displaced workers whose earnings
#: change (wage-cut, mixed-survivor and reallocation margins) these are scaled
#: by the worker's earnings factor (shocked / baseline earnings) rather than
#: left at their baseline level — leaving full baseline pension contributions
#: against cut earnings overstates tax relief and inflates the measured
#: cushioning rate (referee point H2). hours_worked is deliberately NOT in
#: this set: a proportional wage cut is a price change, not an hours change,
#: and the reallocation margin already carries its own hours factor.
EARNINGS_SCALED_VARIABLES = (
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
    lcwra = np.asarray(lcwra, dtype=float)
    flagged_bu = np.asarray(
        sim.map_result(lcwra, "person", "benunit"), dtype=float
    )
    # Real self-check (the former max-vs-one-element comparison was vacuous:
    # the addon is constructed from a boolean so could never exceed one
    # element). The person->benunit mapping must CONSERVE the flagged count —
    # a lossy or duplicating map_result would silently mis-pay units.
    if not np.isclose(flagged_bu.sum(), lcwra.sum(), rtol=0.0, atol=1e-6):
        raise RuntimeError(
            "LCWRA person->benunit mapping did not conserve the flagged "
            f"count (persons {lcwra.sum():.0f} vs benunit total "
            f"{flagged_bu.sum():.0f})."
        )
    return (flagged_bu > 0).astype(float) * monthly * 12.0


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


def _baseline_flag_values_and_rate(sim, period: int) -> tuple[np.ndarray, float]:
    """Return the boolean claim flags and their weighted benefit-unit rate."""
    weighted_flags = sim.calculate("would_claim_uc", period=period, map_to="benunit")
    return np.asarray(weighted_flags.values, dtype=bool), float(weighted_flags.mean())


#: Key under which ``redraw_uc_takeup`` records its redraw-set diagnostic on
#: the shocked table's ``.attrs`` (and, as an attribute of the same name with
#: a leading underscore, on the shocked simulation object).
UC_TAKEUP_DIAGNOSTIC_KEY = "uc_takeup_redraw"


def _benunit_weights(sim, period, n_benunits: int) -> np.ndarray | None:
    """Benefit-unit weights, or None when the sim cannot supply them.

    Unit-test stubs and reduced simulations have no ``benunit_weight``
    variable; the diagnostic degrades to unweighted counts rather than
    failing, so this helper never raises.
    """
    try:
        values = np.asarray(
            sim.calculate("benunit_weight", period=period, map_to="benunit").values,
            dtype=float,
        )
    except Exception:  # noqa: BLE001 - any stub/engine failure means "no weights"
        return None
    if values.shape != (n_benunits,) or not np.isfinite(values).all():
        return None
    return values


def uc_takeup_redraw_diagnostic(shocked_table) -> dict | None:
    """Diagnostic recorded by the most recent ``redraw_uc_takeup`` call.

    Exists because a take-up grid run over an EMPTY redraw set is silently
    inert: every take-up value returns bit-identical results and the
    experiment looks like a robustness finding ("take-up does not matter")
    when it is actually a no-op. Callers should store ``n_redrawn`` alongside
    any take-up sensitivity they report.
    """
    return getattr(shocked_table, "attrs", {}).get(UC_TAKEUP_DIAGNOSTIC_KEY)


def _record_takeup_diagnostic(sim, shocked_table, diagnostic: dict) -> dict:
    try:
        shocked_table.attrs[UC_TAKEUP_DIAGNOSTIC_KEY] = diagnostic
    except (AttributeError, TypeError):  # pragma: no cover - non-pandas caller
        pass
    try:
        setattr(sim, f"_{UC_TAKEUP_DIAGNOSTIC_KEY}", diagnostic)
    except AttributeError:  # pragma: no cover - slotted stub
        pass
    return diagnostic


def redraw_uc_takeup(
    sim,
    baseline_sim,
    shocked_table,
    period,
    uc_takeup: float | None = None,
    seed: int | None = None,
    baseline_potential_award: np.ndarray | None = None,
    uc_takeup_scope: str | None = None,
) -> np.ndarray:
    """Re-draw take-up for changed units with positive post-shock UC potential.

    See DEFAULT_UC_TAKEUP for the full justification: the baseline flag is a
    stored draw conditioned on PRE-SHOCK circumstances (and calibrated as a
    population-wide share, not take-up among the entitled). The same rule is
    applied to every adjustment margin: a benefit unit is a candidate only if
    its labour-market circumstances changed and its potential UC award is
    positive post-shock. ``uc_takeup_scope`` then selects WHICH of those
    candidates are re-drawn (see UC_TAKEUP_SCOPES):

    - ``new_entitlement`` (default): only units whose potential award moves
      from zero at baseline to positive post-shock;
    - ``all_entitled``: every changed unit with a positive post-shock
      potential award, including units already entitled at baseline.

    The draw uses an independent RNG stream in both cases, leaving
    labour-market draws and unchanged or ineligible units untouched.

    Returns the benunit-level boolean flag actually applied, and records a
    redraw-set diagnostic retrievable via ``uc_takeup_redraw_diagnostic``.
    Hard-error contract: a silently-rejected ``set_input`` would leave the
    stale flags in place and change every downstream result, so the applied
    array is read back and verified.
    """
    if uc_takeup is None:
        uc_takeup = float(shocked_table.attrs.get("uc_takeup", DEFAULT_UC_TAKEUP))
    if seed is None:
        seed = int(shocked_table.attrs.get("seed", 0))
    if uc_takeup_scope is None:
        uc_takeup_scope = str(
            shocked_table.attrs.get("uc_takeup_scope", DEFAULT_UC_TAKEUP_SCOPE)
        )
    _validate_uc_takeup_scope(uc_takeup_scope)
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

    weights = _benunit_weights(sim, period, baseline_flag.size)

    def _diagnostic(**fields) -> dict:
        base = {
            "uc_takeup": float(uc_takeup),
            "uc_takeup_scope": uc_takeup_scope,
            "seed": int(seed),
            "n_benunits": int(baseline_flag.size),
            "weights_available": weights is not None,
        }
        base.update(fields)
        return base

    def _weighted(mask) -> float | None:
        if weights is None:
            return None
        return float(weights[np.asarray(mask, dtype=bool)].sum())

    if not changed.any():
        _record_takeup_diagnostic(
            sim,
            shocked_table,
            _diagnostic(
                n_changed_benunits=0,
                n_entitled_changed_benunits=0,
                n_redrawn=0,
                weighted_changed_benunits=0.0 if weights is not None else None,
                weighted_redrawn=0.0 if weights is not None else None,
            ),
        )
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

    # Candidate set common to both scopes: a changed benefit unit that would
    # actually receive something if it claimed.
    entitled_changed_bu = changed_bu & (potential_award > 1e-8)
    if uc_takeup_scope == "new_entitlement":
        # Apply the take-up draw only to benefit units that become newly
        # entitled. Existing claimants and existing eligible non-claimants
        # retain their baseline claiming state; otherwise a small wage cut
        # would inadvertently impose a population-wide take-up reform on all
        # already-entitled workers. NOTE: this set is empty whenever every
        # changed-and-entitled unit was already potentially entitled at
        # baseline, in which case a take-up grid run under this scope is inert
        # by construction — check ``n_redrawn`` in the diagnostic before
        # reporting such a grid as a robustness result.
        redraw_bu = entitled_changed_bu & (baseline_potential_award <= 1e-8)
    else:  # "all_entitled" — validated above
        # Re-open the claiming decision for every changed entitled unit,
        # including baseline-entitled ones. Bounds the claiming margin.
        redraw_bu = entitled_changed_bu

    diagnostic_fields = {
        "n_changed_benunits": int(changed_bu.sum()),
        "n_entitled_changed_benunits": int(entitled_changed_bu.sum()),
        "n_redrawn": int(redraw_bu.sum()),
        "weighted_changed_benunits": _weighted(changed_bu),
        "weighted_redrawn": _weighted(redraw_bu),
    }

    if not redraw_bu.any():
        _record_takeup_diagnostic(sim, shocked_table, _diagnostic(**diagnostic_fields))
        sim.set_input("would_claim_uc", period, baseline_flag)
        sim._invalidate_all_caches()
        _verify_uc_award_cache_flushed(sim, period, baseline_flag, potential_award)
        return baseline_flag
    # Tuple seeding gives the take-up stream its own np.random.SeedSequence
    # spawn key, so it can NEVER collide with the plain-integer displacement
    # stream or the other tuple-seeded streams (destination (seed, 4_010_2026)
    # and LCWRA (seed, 7_002_026)) — an additive integer offset shares the
    # displacement seed space and overlaps it for seeds >= the offset.
    # The draw is generated over ALL benefit units and then masked, so a given
    # benefit unit's draw is identical across take-up scopes: switching scope
    # changes WHICH units are re-drawn, never the underlying random numbers.
    rng = np.random.default_rng((int(seed), UC_TAKEUP_SEED_OFFSET))
    draw = rng.random(baseline_flag.size) < uc_takeup
    new_flag = np.where(redraw_bu, draw, baseline_flag)
    _record_takeup_diagnostic(sim, shocked_table, _diagnostic(**diagnostic_fields))
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
    _verify_uc_award_cache_flushed(sim, period, new_flag, potential_award)
    return new_flag


def _verify_uc_award_cache_flushed(sim, period, flag, potential_award) -> None:
    """Hard-error if the UC award still reflects the temporary all-claim pass.

    ``redraw_uc_takeup`` measures entitlement by computing universal_credit
    under a temporary all-True ``would_claim_uc``. Restoring the final flag
    relies on the private ``sim._invalidate_all_caches()`` to drop that cached
    award; if a policyengine upgrade turned that call into a no-op, every
    downstream metric would silently score the all-claim award. Verify the
    flush by recomputing the award: any unit with a positive potential award
    whose final flag is False must now score zero UC (referee point M5).
    """
    award = np.asarray(
        sim.calculate("universal_credit", period=period, map_to="benunit").values,
        dtype=float,
    )
    stale = (~np.asarray(flag, dtype=bool)) & (np.asarray(potential_award) > 1e-8)
    if stale.any() and (award[stale] > 1e-8).any():
        raise RuntimeError(
            "post-shock UC take-up re-draw not applied: universal_credit still "
            "reflects the temporary all-claim entitlement pass — the "
            "simulation cache was not flushed (_invalidate_all_caches no-op?)."
        )


def build_shocked_simulation(dataset, baseline_sim, shocked_table, period):
    """One shared constructor for the shocked simulation (every pipeline).

    Displaced workers have all TRANSITION_ZEROED_VARIABLES zeroed; workers
    whose earnings change without displacement have their earnings-linked
    deductions (EARNINGS_SCALED_VARIABLES) scaled by the earnings factor.
    Displaced-not-inactive workers become UNEMPLOYED; inactive workers become
    OTHER_INACTIVE and (per the shocked table's ``lcwra`` column) are flagged
    uc_limited_capability_for_WRA = True on top of any baseline flag, so
    their Universal Credit includes the LCWRA health element.
    A rejected set_input would silently leave displaced workers EMPLOYED with
    zero hours, changing entitlements in every result — we verify and FAIL
    HARD (do not replicate the template's silent-failure bug).
    """
    # Cloning the already validated baseline avoids reconstructing and deeply
    # copying the complete UK parameter tree for every integration draw.
    # set_input invalidates dependent calculations; the transition and UC
    # checks below hard-error if any stale baseline state survives.
    if hasattr(baseline_sim, "clone"):
        sim = baseline_sim.clone()
    else:  # lightweight test doubles and older PolicyEngine releases
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
    # Earnings factor per worker: shocked / baseline earnings (1.0 where the
    # baseline is zero). Displaced workers have factor 0 but are handled by
    # the explicit zeroing below; the factor scales the earnings-linked
    # deductions of NON-displaced shocked workers (wage-cut, mixed-survivor
    # and reallocation margins) — referee point H2.
    baseline_earnings = baseline_sim.calculate(
        "employment_income", period=period, map_to="person"
    ).values.astype(float)
    shocked_earnings = shocked_table["employment_income"].to_numpy(dtype=float)
    earnings_factor = np.divide(
        shocked_earnings,
        baseline_earnings,
        out=np.ones_like(shocked_earnings),
        where=baseline_earnings > 0,
    )
    for var in TRANSITION_ZEROED_VARIABLES:
        values = baseline_sim.calculate(var, period=period, map_to="person").values.astype(float)
        if var in EARNINGS_SCALED_VARIABLES:
            values = values * np.where(displaced, 1.0, earnings_factor)
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
        if hasattr(baseline_sim, "clone"):
            potential_sim = baseline_sim.clone()
        else:
            from policyengine_uk import Microsimulation

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
