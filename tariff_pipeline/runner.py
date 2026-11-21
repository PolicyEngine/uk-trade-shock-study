"""Run tariff-shock scenarios through PolicyEngine UK and summarise deltas.

Mirrors uk-ai-study's runner: HBAI cash disposable income is the income
concept throughout (Gini, deciles, changes), matching the in_poverty_*
concept; the broad household_net_income is used only inside gov_balance.
Monte Carlo support (n_draws, mean +/- SD) is built in from the start
(referee point M5 on the sister paper).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from tariff_pipeline.exposure import attach_sic_division, simulation_sic_division
from tariff_pipeline.shocks import (
    PRESETS,
    TradeShockScenario,
    apply_shocks,
    build_shocked_simulation,
)

AGE_BANDS = ((16, 24), (25, 34), (35, 44), (45, 54), (55, 64), (65, 200))

#: TAX/BENEFIT SPLIT OF THE CUSHION (referee request).
#:
#: The cushion is gross employee-earnings loss minus the loss of HBAI
#: household net income. HBAI net income is, in the model,
#: max(0, sum(ADDS) - sum(SUBTRACTS)) over the two lists in
#: policyengine_uk...hbai_household_net_income, and employment_income is one
#: of the ADDS. So, writing D(x) for the weighted change (shocked - baseline)
#: in x,
#:
#:   cushion = D(net) - D(employment_income)
#:           = sum_{a in ADDS, a != employment_income} D(a)
#:             - sum_{s in SUBTRACTS} D(s)
#:             + D(net - net_before_reset_to_zero)
#:
#: exactly. Every ADD and SUBTRACT is computed individually and assigned to
#: one named group below; anything not named falls into ``other`` and the
#: zero-floor term is its own line, so the groups sum to the cushion by
#: construction. ``cushion_components`` asserts that they do.
#: A positive entry cushions the loss (lower tax, higher benefit).
CUSHION_GROUPS = {
    "income_tax": ("income_tax",),
    "national_insurance": ("national_insurance",),
    "universal_credit": ("universal_credit",),
    "other_means_tested": (
        "housing_benefit",
        "pension_credit",
        "working_tax_credit",
        "child_tax_credit",
        "esa_income",
        "jsa_income",
        "income_support",
        "council_tax_benefit",
    ),
    "jsa_contrib": ("jsa_contrib",),
    "esa_contrib": ("esa_contrib",),
    "child_benefit": ("child_benefit",),
    "pension_contributions": (
        "employee_pension_contributions",
        "personal_pension_contributions",
    ),
    "student_loan_repayments": ("student_loan_repayments",),
    # Statutory sick/maternity/paternity pay is zeroed for displaced workers
    # (TRANSITION_ZEROED_VARIABLES), so this line is a NEGATIVE cushion.
    "statutory_pay": (
        "statutory_sick_pay",
        "statutory_maternity_pay",
        "statutory_paternity_pay",
    ),
}
#: Residual lines: every other HBAI component, and the HBAI zero floor.
CUSHION_OTHER = "other"
CUSHION_FLOOR = "negative_income_floor"
CUSHION_GROUP_NAMES = (*CUSHION_GROUPS, CUSHION_OTHER, CUSHION_FLOOR)


def _hbai_lists() -> tuple[list[str], list[str]]:
    from policyengine_uk.variables.household.income.hbai_household_net_income import (
        HBAI_HOUSEHOLD_NET_INCOME_ADDS,
        HBAI_HOUSEHOLD_NET_INCOME_SUBTRACTS,
    )

    return list(HBAI_HOUSEHOLD_NET_INCOME_ADDS), list(HBAI_HOUSEHOLD_NET_INCOME_SUBTRACTS)


def household_components(sim, period: int) -> dict[str, np.ndarray]:
    """Household-level £ arrays of every HBAI component, plus the totals.

    Keys: each variable in the model's HBAI ADDS and SUBTRACTS lists,
    ``hbai_household_net_income`` and
    ``hbai_household_net_income_before_reset_to_zero``.
    """
    adds, subtracts = _hbai_lists()
    names = [
        *adds,
        *subtracts,
        "hbai_household_net_income",
        "hbai_household_net_income_before_reset_to_zero",
    ]
    return {
        name: np.asarray(
            sim.calculate(name, period=period, map_to="household").values, dtype=float
        )
        for name in names
    }


def cushion_components(
    delta: dict[str, np.ndarray | float], rel_tol: float = 1e-6, abs_tol: float = 1.0
) -> dict[str, float]:
    """Group HBAI component CHANGES (shocked - baseline) into the cushion split.

    ``delta`` maps every key of ``household_components`` to a change -- a
    weighted total, or an array that is summed. Returns £ per group in
    CUSHION_GROUP_NAMES plus ``cushion`` (= gross loss - net loss) and
    ``gross_loss``. Raises if the model's own identity
    net_before_reset = sum(ADDS) - sum(SUBTRACTS) does not hold for the
    changes, or if the groups do not sum to the cushion.
    """
    adds, subtracts = _hbai_lists()
    total = {key: float(np.sum(value)) for key, value in delta.items()}
    sign = {name: 1.0 for name in adds}
    sign.update({name: -1.0 for name in subtracts})
    scale = max(abs(total["employment_income"]), 1.0)
    identity_gap = total["hbai_household_net_income_before_reset_to_zero"] - sum(
        sign[name] * total[name] for name in sign
    )
    if abs(identity_gap) > abs_tol + rel_tol * scale:
        raise AssertionError(
            "HBAI components do not sum to net income before the zero floor: "
            f"gap £{identity_gap:,.2f} on a gross change of £{scale:,.0f}"
        )
    out: dict[str, float] = {}
    assigned = {"employment_income"}
    for group, members in CUSHION_GROUPS.items():
        missing = [name for name in members if name not in sign]
        if missing:
            raise KeyError(f"cushion group {group!r}: not HBAI components: {missing}")
        out[group] = sum(sign[name] * total[name] for name in members)
        assigned.update(members)
    out[CUSHION_OTHER] = sum(
        sign[name] * total[name] for name in sign if name not in assigned
    )
    out[CUSHION_FLOOR] = (
        total["hbai_household_net_income"]
        - total["hbai_household_net_income_before_reset_to_zero"]
    )
    cushion = total["hbai_household_net_income"] - total["employment_income"]
    gap = sum(out.values()) - cushion
    # ``other`` absorbs unnamed components but NOT a broken identity: the
    # groups are built from the component totals, the cushion from the two
    # headline totals, so this fails if the model's net income stops being
    # the sum of its listed components.
    if abs(gap) > abs_tol + rel_tol * scale:
        raise AssertionError(
            f"cushion split does not sum to gross - net: gap £{gap:,.2f} "
            f"on a gross change of £{scale:,.0f}"
        )
    out["cushion"] = cushion
    out["gross_loss"] = -total["employment_income"]
    return out


def component_shares(components: dict[str, float]) -> dict[str, dict[str, float]]:
    """Each group's share of the gross loss and of the total cushion.

    Ratios of summed £ (pooled), not means of per-draw ratios.
    ``share_of_gross_loss`` sums to the cushioning rate;
    ``share_of_cushion`` sums to one.
    """
    gross, cushion = components["gross_loss"], components["cushion"]
    return {
        "share_of_gross_loss": {
            g: (components[g] / gross if gross else float("nan"))
            for g in CUSHION_GROUP_NAMES
        },
        "share_of_cushion": {
            g: (components[g] / cushion if cushion else float("nan"))
            for g in CUSHION_GROUP_NAMES
        },
    }

PERSON_VARIABLES = ("person_id", "age", "employment_income")


@dataclass(frozen=True)
class ScenarioResult:
    scenario: str
    tariff_scenario: str
    margin: str
    exchequer_cost: float
    poverty_rate_change_bhc: float
    poverty_rate_change_ahc: float
    absolute_poverty_rate_change_bhc: float
    gini_baseline: float
    gini_shocked: float
    displaced_weighted: float
    inactive_weighted: float
    lcwra_weighted: float = 0.0
    reallocated_weighted: float = 0.0
    gross_earnings_loss: float = float("nan")
    net_disposable_loss: float = float("nan")
    cushioning_rate: float = float("nan")
    affected_records: int = 0
    effective_loss_records: float = 0.0
    max_record_loss_share: float = 0.0
    decile_income_change: dict = field(default_factory=dict)
    region_income_change: dict = field(default_factory=dict)
    age_band_displacement_share: dict = field(default_factory=dict)
    #: £ change in each tax/benefit line behind the cushion (CUSHION_GROUP_NAMES
    #: plus ``cushion`` and ``gross_loss``); the groups sum to ``cushion``.
    cushion_components: dict = field(default_factory=dict)


@dataclass(frozen=True)
class MonteCarloResult:
    scenario: str
    n_draws: int
    exchequer_cost_mean: float
    exchequer_cost_sd: float
    poverty_rate_change_bhc_mean: float
    poverty_rate_change_bhc_sd: float
    gini_change_mean: float
    gini_change_sd: float
    displaced_weighted_mean: float
    exchequer_cost_mc_se: float = 0.0
    poverty_rate_change_bhc_mc_se: float = 0.0
    gini_change_mc_se: float = 0.0
    lcwra_weighted_mean: float = 0.0
    reallocated_weighted_mean: float = 0.0
    cushioning_rate_mean: float = float("nan")
    cushioning_rate_sd: float = 0.0
    cushioning_rate_mc_se: float = 0.0
    cushioning_valid_draws: int = 0
    #: Assignment design behind these draws. Recorded because the legacy
    #: 100-draw families run the Bernoulli comparator while the submission
    #: design is balanced, and comparing across the two without noticing is
    #: how an appendix reconciliation ends up on a different run from the
    #: table it reconciles.
    selection_method: str = ""
    #: 1 - mean(net loss) / mean(gross loss): the ratio of means, which
    #: weights draws by their realised gross loss (the mean of per-draw
    #: ratios above does not).
    cushioning_rate_pooled: float = float("nan")
    #: Mean £ per tax/benefit line across draws, and its shares of the mean
    #: gross loss / mean cushion (see ``component_shares``).
    cushion_components_mean: dict = field(default_factory=dict)
    cushion_component_shares: dict = field(default_factory=dict)
    draws: list = field(default_factory=list)


def gini(values: np.ndarray, weights: np.ndarray) -> float:
    # bottom-code at zero: negative incomes make the Gini exceed 1
    values = np.clip(np.asarray(values, float), 0.0, None)
    order = np.argsort(values)
    v, w = values[order], np.asarray(weights, float)[order]
    cw = np.cumsum(w)
    cv = np.cumsum(v * w)
    if cv[-1] == 0:
        return 0.0
    return float(1 - 2 * np.sum((cv - v * w / 2) * w) / (cv[-1] * cw[-1]))


def _person_table(sim, period: int) -> pd.DataFrame:
    table = pd.DataFrame(
        {v: sim.calculate(v, period=period, map_to="person").values for v in PERSON_VARIABLES}
    )
    table["weight"] = sim.calculate("person_weight", period=period, map_to="person").values
    return table


def _household_income_per_person(sim, period: int) -> np.ndarray:
    """Household disposable income allocated equally across its members.

    PolicyEngine broadcasts the *whole* household income when a household
    variable is mapped to persons.  Using that broadcast value directly as a
    per-person outcome counts a household's income change once for every
    member.  Divide by the correspondingly broadcast household size before
    constructing person-weighted decile or regional means.
    """
    income = np.asarray(
        sim.calculate("hbai_household_net_income", period=period, map_to="person").values,
        dtype=float,
    )
    people = np.asarray(
        sim.calculate("household_count_people", period=period, map_to="person").values,
        dtype=float,
    )
    if income.shape != people.shape:
        raise ValueError("person-mapped household income and household size differ in shape")
    if (people <= 0).any():
        raise ValueError("person-mapped household size must be positive")
    return income / people


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cw = np.cumsum(w)
    if cw[-1] <= 0:
        raise ValueError("weights must have positive total")
    return float(v[np.searchsorted(cw, 0.5 * cw[-1])])


def _metrics(sim, period: int, poverty_lines: tuple[float, float] | None = None) -> dict:
    """Summary metrics for one simulation.

    POVERTY LINES ARE FROZEN AT BASELINE (referee point H1). The headline
    ``poverty_bhc`` / ``poverty_ahc`` rates are the person-weighted shares of
    people in households whose equivalised HBAI net income (BHC / AHC) lies
    below 60% of the BASELINE weighted median — the standard relative-poverty
    definition, but with the threshold computed ONCE from the baseline
    simulation and held fixed for shocked simulations. Recomputing the
    relative threshold inside each shocked simulation (as the model's own
    ``in_relative_poverty_*`` variables do) lets a broad earnings shock drag
    the 60%-of-median line down with it, mechanically attenuating the
    measured poverty change.

    ``poverty_lines``: ``None`` for the baseline call — the (BHC, AHC)
    thresholds are derived from THIS simulation's income distribution and
    returned under ``poverty_line_bhc`` / ``poverty_line_ahc``; shocked calls
    pass the baseline pair so shocked incomes are scored against the frozen
    baseline line. The median mirrors policyengine-uk's
    ``in_relative_poverty_*`` construction (household-weighted median of
    equivalised HBAI income).

    The model's own variable-based rates remain available under
    ``poverty_bhc_model_line`` / ``poverty_ahc_model_line`` for reference.
    """
    hh_w = sim.calculate("household_weight", period=period, map_to="household").values
    equiv = sim.calculate("equiv_hbai_household_net_income", period=period, map_to="household").values
    equiv_ahc = sim.calculate(
        "equiv_hbai_household_net_income_ahc", period=period, map_to="household"
    ).values
    hh_count = sim.calculate("household_count_people", period=period, map_to="household").values
    p_w = sim.calculate("person_weight", period=period, map_to="person").values
    if poverty_lines is None:
        line_bhc = 0.6 * _weighted_median(equiv, hh_w)
        line_ahc = 0.6 * _weighted_median(equiv_ahc, hh_w)
    else:
        line_bhc, line_ahc = (float(x) for x in poverty_lines)
    equiv_p = np.asarray(
        sim.calculate("equiv_hbai_household_net_income", period=period, map_to="person").values,
        dtype=float,
    )
    equiv_ahc_p = np.asarray(
        sim.calculate(
            "equiv_hbai_household_net_income_ahc", period=period, map_to="person"
        ).values,
        dtype=float,
    )
    out = {
        "gov_balance": float((sim.calculate("gov_balance", period=period, map_to="household").values * hh_w).sum()),
        # Headline rates: fixed (baseline) relative poverty line (H1).
        "poverty_bhc": float(np.average(equiv_p < line_bhc, weights=p_w)),
        "poverty_ahc": float(np.average(equiv_ahc_p < line_ahc, weights=p_w)),
        "poverty_line_bhc": float(line_bhc),
        "poverty_line_ahc": float(line_ahc),
        # Legacy model-variable rates (threshold recomputed inside this
        # simulation by policyengine-uk); kept for reference only.
        "poverty_bhc_model_line": float(np.average(
            sim.calculate("in_poverty_bhc", period=period, map_to="person").values, weights=p_w
        )),
        "poverty_ahc_model_line": float(np.average(
            sim.calculate("in_poverty_ahc", period=period, map_to="person").values, weights=p_w
        )),
        "gini": gini(equiv, hh_w * hh_count),
        "hni": _household_income_per_person(sim, period),
        "hni_total": float(
            (sim.calculate("hbai_household_net_income", period=period, map_to="household").values * hh_w).sum()
        ),
    }
    out["component_totals"] = {
        name: float((values * hh_w).sum())
        for name, values in household_components(sim, period).items()
    }
    # absolute BHC poverty (fixed 2010-11 line, HBAI); tolerate model
    # versions that expose only the relative measures
    try:
        out["abs_poverty_bhc"] = float(np.average(
            sim.calculate("in_absolute_poverty_bhc", period=period, map_to="person").values,
            weights=p_w,
        ))
    except ValueError as exc:
        # policyengine-uk 2.89.2 reports an unavailable optional variable as a
        # ValueError.  Do not hide unrelated calculation/data failures.
        if "Variable in_absolute_poverty_bhc does not exist" not in str(exc):
            raise
        out["abs_poverty_bhc"] = float("nan")
    return out


def _baseline_and_persons(dataset_path, adult_tab_path, period):
    """adult_tab_path=None (default path) reads SIC from the h5's
    sic_industry_division variable; passing a path uses the legacy
    adult.tab join as a fallback for h5 builds without the column."""
    if dataset_path is None:
        # Default: the certified Microcosm UK build pinned by policyengine.py.
        from uk_trade_shock_study.microcosm import microsimulation

        dataset = None
        baseline = microsimulation()
    else:  # an explicit local h5 (tests, legacy replication)
        from policyengine_uk import Microsimulation
        from policyengine_uk.data import UKSingleYearDataset

        dataset = UKSingleYearDataset(file_path=str(dataset_path))
        baseline = Microsimulation(dataset=dataset)
    persons = _person_table(baseline, period)
    if adult_tab_path is None:
        persons["sic_division"] = simulation_sic_division(baseline, period)
    else:
        persons["sic_division"] = attach_sic_division(persons["person_id"], adult_tab_path)
    return dataset, baseline, persons


def _one_draw(
    dataset,
    baseline,
    persons,
    scenario,
    period,
    seed,
    baseline_metrics: dict | None = None,
    baseline_equiv: np.ndarray | None = None,
    baseline_region: np.ndarray | None = None,
) -> ScenarioResult:
    shocked_table = apply_shocks(persons, scenario, seed=seed)
    shocked = build_shocked_simulation(dataset, baseline, shocked_table, period)
    displaced = shocked_table["displaced"].to_numpy()
    inactive = shocked_table["inactive"].to_numpy()
    lcwra = (
        shocked_table["lcwra"].to_numpy()
        if "lcwra" in shocked_table
        else np.zeros(len(displaced), dtype=bool)
    )

    reallocated = (
        shocked_table["reallocated"].to_numpy()
        if "reallocated" in shocked_table
        else np.zeros(len(displaced), dtype=bool)
    )

    base = baseline_metrics if baseline_metrics is not None else _metrics(baseline, period)
    # Freeze the BASELINE relative poverty lines when scoring the shocked
    # simulation (referee point H1: an endogenous 60%-of-median line falls
    # with a broad earnings shock and understates the poverty change).
    shock = _metrics(
        shocked, period, poverty_lines=(base["poverty_line_bhc"], base["poverty_line_ahc"])
    )

    weight = persons["weight"].to_numpy()
    age = persons["age"].to_numpy()
    income_delta = shock["hni"] - base["hni"]
    base_earnings = persons["employment_income"].to_numpy(dtype=float)
    shocked_earnings = shocked_table["employment_income"].to_numpy(dtype=float)
    gross_loss = float(((base_earnings - shocked_earnings) * weight).sum())
    net_loss = base["hni_total"] - shock["hni_total"]
    components = cushion_components(
        {
            name: shock["component_totals"][name] - base["component_totals"][name]
            for name in base["component_totals"]
        }
    )
    # The split's gross loss is the household-aggregated employment_income
    # change read back from the SIMULATION; it must equal the loss imposed on
    # the person table.
    if not np.isclose(components["gross_loss"], gross_loss, rtol=1e-6, atol=1.0):
        raise AssertionError(
            "simulated employment_income change differs from the imposed "
            f"gross loss: £{components['gross_loss']:,.0f} vs £{gross_loss:,.0f}"
        )
    equiv = (
        baseline_equiv
        if baseline_equiv is not None
        else baseline.calculate(
            "equiv_hbai_household_net_income", period=period, map_to="person"
        ).values
    )
    order = np.argsort(equiv)
    cum = np.cumsum(weight[order])
    ranks = np.empty(len(equiv), dtype=float)
    ranks[order] = cum / cum[-1]
    deciles = np.clip(np.ceil(ranks * 10).astype(int), 1, 10)
    decile_change = {
        int(d): float(np.average(income_delta[deciles == d], weights=weight[deciles == d]))
        for d in range(1, 11)
    }
    region = (
        baseline_region
        if baseline_region is not None
        else baseline.calculate("region", period=period, map_to="person").values.astype(str)
    )
    region_change = {
        r: float(np.average(income_delta[region == r], weights=weight[region == r]))
        for r in sorted(set(region))
    }
    displaced_w = float(weight[displaced].sum())
    record_loss = (
        persons["employment_income"].to_numpy(dtype=float)
        - shocked_table["employment_income"].to_numpy(dtype=float)
    ) * weight
    positive_loss = record_loss > 0
    total_record_loss = float(record_loss[positive_loss].sum())
    effective_loss_records = (
        total_record_loss**2 / float(np.square(record_loss[positive_loss]).sum())
        if positive_loss.any()
        else 0.0
    )
    max_record_loss_share = (
        float(record_loss[positive_loss].max() / total_record_loss)
        if positive_loss.any() and total_record_loss > 0
        else 0.0
    )
    # age composition is reported over the AFFECTED set: displaced under the
    # job-loss margins, reallocated under the reallocation margin.
    affected = displaced | reallocated
    affected_w = float(weight[affected].sum())
    band_share = {}
    for lo, hi in AGE_BANDS:
        mask = (age >= lo) & (age <= hi)
        label = f"{lo}-{hi if hi < 200 else '+'}"
        band_share[label] = float(weight[mask & affected].sum() / affected_w) if affected_w else 0.0

    return ScenarioResult(
        scenario=scenario.name,
        tariff_scenario=scenario.tariff_scenario,
        margin=scenario.margin,
        exchequer_cost=base["gov_balance"] - shock["gov_balance"],
        poverty_rate_change_bhc=shock["poverty_bhc"] - base["poverty_bhc"],
        poverty_rate_change_ahc=shock["poverty_ahc"] - base["poverty_ahc"],
        absolute_poverty_rate_change_bhc=shock["abs_poverty_bhc"] - base["abs_poverty_bhc"],
        gini_baseline=base["gini"],
        gini_shocked=shock["gini"],
        displaced_weighted=displaced_w,
        inactive_weighted=float(weight[inactive].sum()),
        lcwra_weighted=float(weight[lcwra].sum()),
        reallocated_weighted=float(weight[reallocated].sum()),
        gross_earnings_loss=gross_loss,
        net_disposable_loss=net_loss,
        cushioning_rate=(1.0 - net_loss / gross_loss) if gross_loss else float("nan"),
        affected_records=int(positive_loss.sum()),
        effective_loss_records=effective_loss_records,
        max_record_loss_share=max_record_loss_share,
        decile_income_change=decile_change,
        region_income_change=region_change,
        age_band_displacement_share=band_share,
        cushion_components=components,
    )


def run_scenario(
    dataset_path: str | Path | None,
    scenario: TradeShockScenario | str,
    period: int = 2026,
    seed: int = 0,
    adult_tab_path: str | Path | None = None,
) -> ScenarioResult:
    if isinstance(scenario, str):
        scenario = PRESETS[scenario]
    dataset, baseline, persons = _baseline_and_persons(dataset_path, adult_tab_path, period)
    return _one_draw(dataset, baseline, persons, scenario, period, seed)


def run_monte_carlo(
    dataset_path: str | Path | None,
    scenario: TradeShockScenario | str,
    period: int = 2026,
    n_draws: int = 20,
    base_seed: int = 0,
    adult_tab_path: str | Path | None = None,
) -> MonteCarloResult:
    """Repeat a scenario and report assignment SD and numerical MC SE.

    Wage earnings cuts are deterministic, but newly entitled benefit units'
    UC claiming draws vary by seed, so wage-cut scenarios retain ``n_draws``.
    """
    if isinstance(scenario, str):
        scenario = PRESETS[scenario]
    dataset, baseline, persons = _baseline_and_persons(dataset_path, adult_tab_path, period)
    return run_monte_carlo_prepared(
        dataset, baseline, persons, scenario, period, n_draws, base_seed
    )


def run_monte_carlo_prepared(
    dataset,
    baseline,
    persons: pd.DataFrame,
    scenario: TradeShockScenario,
    period: int = 2026,
    n_draws: int = 20,
    base_seed: int = 0,
) -> MonteCarloResult:
    """Run draws using an already prepared/enriched person table."""
    baseline_metrics = _metrics(baseline, period)
    baseline_equiv = baseline.calculate(
        "equiv_hbai_household_net_income", period=period, map_to="person"
    ).values
    baseline_region = baseline.calculate(
        "region", period=period, map_to="person"
    ).values.astype(str)
    draws = [
        _one_draw(
            dataset,
            baseline,
            persons,
            scenario,
            period,
            base_seed + i,
            baseline_metrics,
            baseline_equiv,
            baseline_region,
        )
        for i in range(n_draws)
    ]
    cost = np.array([d.exchequer_cost for d in draws])
    pov = np.array([d.poverty_rate_change_bhc for d in draws])
    gini_change = np.array([d.gini_shocked - d.gini_baseline for d in draws])
    root_n = np.sqrt(n_draws)
    cost_sd = float(cost.std(ddof=1)) if n_draws > 1 else 0.0
    pov_sd = float(pov.std(ddof=1)) if n_draws > 1 else 0.0
    gini_sd = float(gini_change.std(ddof=1)) if n_draws > 1 else 0.0
    cushioning_mean, cushioning_sd, valid_cushioning = _finite_mean_sd(
        [d.cushioning_rate for d in draws]
    )
    gross_mean = float(np.mean([d.gross_earnings_loss for d in draws]))
    net_mean = float(np.mean([d.net_disposable_loss for d in draws]))
    components_mean = {
        key: float(np.mean([d.cushion_components[key] for d in draws]))
        for key in draws[0].cushion_components
    }
    return MonteCarloResult(
        scenario=scenario.name,
        n_draws=n_draws,
        exchequer_cost_mean=float(cost.mean()),
        exchequer_cost_sd=cost_sd,
        exchequer_cost_mc_se=cost_sd / root_n,
        poverty_rate_change_bhc_mean=float(pov.mean()),
        poverty_rate_change_bhc_sd=pov_sd,
        poverty_rate_change_bhc_mc_se=pov_sd / root_n,
        gini_change_mean=float(gini_change.mean()),
        gini_change_sd=gini_sd,
        gini_change_mc_se=gini_sd / root_n,
        displaced_weighted_mean=float(np.mean([d.displaced_weighted for d in draws])),
        lcwra_weighted_mean=float(np.mean([d.lcwra_weighted for d in draws])),
        reallocated_weighted_mean=float(np.mean([d.reallocated_weighted for d in draws])),
        cushioning_rate_mean=cushioning_mean,
        cushioning_rate_sd=cushioning_sd,
        cushioning_rate_mc_se=(
            cushioning_sd / np.sqrt(valid_cushioning)
            if valid_cushioning
            else float("nan")
        ),
        cushioning_valid_draws=valid_cushioning,
        selection_method=scenario.selection_method,
        cushioning_rate_pooled=(
            1.0 - net_mean / gross_mean if gross_mean else float("nan")
        ),
        cushion_components_mean=components_mean,
        cushion_component_shares=component_shares(components_mean),
        draws=[asdict(d) for d in draws],
    )


def _finite_mean_sd(values) -> tuple[float, float, int]:
    """Mean/SD for defined draws, retaining the valid-draw denominator."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return float("nan"), 0.0, 0
    standard_deviation = float(finite.std(ddof=1)) if len(finite) > 1 else 0.0
    return float(finite.mean()), standard_deviation, len(finite)


def json_value(value):
    """Return standards-compliant JSON values (RFC 8259 has no NaN).

    Every artifact writer in this package must route its payload through this
    before ``json.dumps`` — NaN and +/-Infinity are Python/JavaScript
    extensions that strict parsers reject, so an artifact carrying them is not
    readable by a conforming consumer. Non-finite floats become ``null``,
    which round-trips as "this draw had no defined value" rather than as a
    token no standard parser accepts. Pair it with ``allow_nan=False`` so a
    missed branch fails loudly at write time instead of shipping.
    """
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_value(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def write_result(result, path: str | Path) -> None:
    payload = json_value(asdict(result))
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False))
