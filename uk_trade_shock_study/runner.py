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

from uk_trade_shock_study.exposure import attach_sic_division, simulation_sic_division
from uk_trade_shock_study.shocks import (
    PRESETS,
    TradeShockScenario,
    apply_shocks,
    build_shocked_simulation,
)

AGE_BANDS = ((16, 24), (25, 34), (35, 44), (45, 54), (55, 64), (65, 200))

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
    )


def run_scenario(
    dataset_path: str | Path,
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
    dataset_path: str | Path,
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


def write_result(result, path: str | Path) -> None:
    def json_value(value):
        """Return standards-compliant JSON values (RFC 8259 has no NaN)."""
        if isinstance(value, dict):
            return {key: json_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [json_value(item) for item in value]
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            return None
        return value

    payload = json_value(asdict(result))
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False))
