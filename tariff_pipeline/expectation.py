"""Exact-expectation design for the displacement margin.

WHY. The displacement margin gives every exposed employee i an independent
job-loss probability s_i (the sector shock of their SIC division). A
Bernoulli draw realises about 50 displaced sample records out of 5,295
exposed ones, so one draw's gross loss ranges from a third of to three times
its expectation and the cushioning rate moves by ten points between draws.
The ``balanced`` assignment in shocks.py does not fix this: it still
displaces about 50 records per draw and, by its own docstring, gives up the
declared inclusion probabilities to hit the aggregate targets. Averaging
more draws only shrinks the noise at rate 1/sqrt(draws).

WHAT. This module computes the EXPECTATION over the Bernoulli assignment
directly, using every exposed record with weight s_i, instead of sampling
from it. It rests on one property of the static tax-benefit model: a
household's taxes and benefits depend only on that household's own members
(the poverty lines are frozen at baseline). So

    E[X] = sum_h w_h * sum_{S subset of E_h} P_h(S) * dX_h(S)

for any outcome X that is a weighted sum over households (gross earnings,
HBAI net income, each tax and benefit line, the exchequer balance, a
fixed-line poverty count), where E_h is the set of exposed employees in
household h, P_h(S) = prod_{i in S} s_i * prod_{i in E_h \\ S} (1 - s_i), and
dX_h(S) is the change in household h's outcome when exactly the workers in S
lose their jobs.

HOW. dX_h({i}) is read off a simulation "pass" in which the rank-k exposed
worker of EVERY household is displaced at once (k = 0, 1, 2, 3; no household
has more than four). Because households do not interact, one pass yields
dX_h({i}) for thousands of workers simultaneously; the code asserts that
households with no displaced member are unchanged in every pass, which is
the separability the design relies on. One further pass displaces ranks 0
and 1 together and gives dX_h({0, 1}) for the households with two or more
exposed workers -- the within-household interaction a per-record reweighting
would get wrong (two earners losing their jobs together draw more benefit
than the sum of each losing theirs alone).

TRUNCATION. Subsets of three or more workers, and pairs involving a rank-2
or rank-3 worker, are dropped (they occur only in the households with three
or more exposed workers, with probability of order s^2 <= 7.5e-4 each). The
dropped share of the expected gross loss is computed exactly and reported
(``omitted_gross_share``); the rate is a ratio of covered net to covered
gross, so the truncation does not bias it mechanically.

UC TAKE-UP. Benefit units newly entitled to UC claim with probability
``uc_takeup`` (0.80). Each pass is run with the re-draw forced to "all
claim" (1.0) and "none claim" (0.0) and the two household outcomes are mixed
with weights uc_takeup and 1 - uc_takeup. With one benefit unit changed per
household this IS the expectation over the claiming draw; in the joint pass
two benefit units of one household can be re-drawn, and the mix treats
their claiming as perfectly correlated -- a second-order effect on an
already second-order term.

WHAT IS NOT AVAILABLE. Non-linear statistics (the Gini) have no exact
expectation under this design and are reported as null.

WAGE-CUT MARGIN. ``wage_cut_expectation_result`` gives the wage-cut margin
the SAME take-up treatment: the cut itself is deterministic (every exposed
employee loses s_i of earnings), so the only randomness is the claiming
re-draw of the newly entitled benefit units, and one forced "all claim" and
one forced "none claim" simulation mixed with weights uc_takeup and
1 - uc_takeup give its expectation (exact when no household holds two
re-drawn benefit units; the function counts them).

PRECISION. Assignment (Monte Carlo) error is zero by construction. What
remains is that the survey holds a finite number of exposed households;
``sampling_se`` is a household-clustered linearisation SE of the ratio that
treats exposed households as independent draws with fixed weights. It is a
different quantity from a Monte Carlo SE and is labelled as such.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from tariff_pipeline.runner import (
    AGE_BANDS,
    CUSHION_GROUP_NAMES,
    component_shares,
    cushion_components,
    household_components,
)
from tariff_pipeline.shocks import (
    TradeShockScenario,
    _person_shock,
    apply_fixed_displacement,
    apply_shocks,
    build_shocked_simulation,
    uc_takeup_redraw_diagnostic,
)

#: Take-up values the passes are run at; mixed with the scenario's uc_takeup.
TAKEUP_NODES = (1.0, 0.0)
#: Household outcomes carried besides the HBAI components.
EXTRA_OUTCOMES = ("gov_balance", "poor_people_bhc", "poor_people_ahc")
#: A household with no displaced member must be unchanged in a pass (£).
SEPARABILITY_TOL = 0.5


def exposure_ranks(household: np.ndarray, exposed: np.ndarray) -> np.ndarray:
    """Within-household rank (0, 1, ...) of each exposed person; -1 otherwise."""
    household = np.asarray(household)
    exposed = np.asarray(exposed, dtype=bool)
    rank = np.full(household.size, -1, dtype=int)
    rank[exposed] = pd.Series(household[exposed]).groupby(household[exposed]).cumcount().to_numpy()
    return rank


def subset_probabilities(
    shock: np.ndarray, household: np.ndarray, rank: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Bernoulli probabilities of the subsets the passes cover.

    Returns ``(p_single, p_pair)``. ``p_single[i]`` is the probability that
    exposed person i is the ONLY exposed member of their household displaced.
    ``p_pair[h]`` (indexed like ``household`` values, i.e. by household
    position) is the probability that exactly the rank-0 and rank-1 members
    are displaced, zero for households with fewer than two exposed members.
    """
    shock = np.asarray(shock, dtype=float)
    exposed = rank >= 0
    n_households = int(household.max()) + 1
    # log of prod(1 - s) over each household's exposed members
    log_stay = np.zeros(n_households)
    np.add.at(log_stay, household[exposed], np.log1p(-shock[exposed]))
    p_single = np.zeros(shock.size)
    p_single[exposed] = (
        shock[exposed] / (1.0 - shock[exposed]) * np.exp(log_stay[household[exposed]])
    )
    odds = np.ones((2, n_households))
    present = np.zeros((2, n_households), dtype=bool)
    for k in (0, 1):
        members = rank == k
        odds[k, household[members]] = shock[members] / (1.0 - shock[members])
        present[k, household[members]] = True
    p_pair = np.where(present.all(axis=0), odds[0] * odds[1] * np.exp(log_stay), 0.0)
    return p_single, p_pair


def _household_outcomes(sim, period: int, poverty_lines) -> dict[str, np.ndarray]:
    out = household_components(sim, period)

    def _hh(name):
        return np.asarray(
            sim.calculate(name, period=period, map_to="household").values, dtype=float
        )

    people = _hh("household_count_people")
    out["gov_balance"] = _hh("gov_balance")
    out["poor_people_bhc"] = people * (
        _hh("equiv_hbai_household_net_income") < poverty_lines[0]
    )
    out["poor_people_ahc"] = people * (
        _hh("equiv_hbai_household_net_income_ahc") < poverty_lines[1]
    )
    return out


@dataclass
class ExpectationPasses:
    """Household-level changes from the displacement passes (scenario-free
    apart from the exposed set, the JSA switch and the take-up scope)."""

    period: int
    household: np.ndarray  # household position of each person
    household_weight: np.ndarray
    rank: np.ndarray  # exposure rank per person, -1 if not exposed
    #: single[k][t][name] -> household array of changes when rank-k workers
    #: are displaced, at take-up node t; pair[t][name] likewise for ranks 0+1.
    single: list
    pair: dict
    n_simulations: int
    max_unaffected_change: float
    max_single_exposed_pair_gap: float

    def mixed(self, deltas: dict, uc_takeup: float) -> dict[str, np.ndarray]:
        return {
            name: uc_takeup * deltas[1.0][name] + (1.0 - uc_takeup) * deltas[0.0][name]
            for name in deltas[1.0]
        }


def run_passes(
    dataset, baseline, persons: pd.DataFrame, scenario: TradeShockScenario, period: int,
    baseline_metrics: dict, log=print, build_table=None,
) -> ExpectationPasses:
    """Run the displacement passes for ``scenario``'s exposed set.

    ``build_table(displaced, scenario_at_node)`` builds the shocked person
    table of one pass; the default is the full-year job loss
    (``apply_fixed_displacement``). The duration sensitivity passes other
    builders (a part-year spell); everything else -- the ranks, the take-up
    nodes, the separability checks -- is shared.
    """
    if build_table is None:
        def build_table(displaced, node_scenario):
            return apply_fixed_displacement(persons, node_scenario, displaced, seed=0)
    if scenario.margin != "displacement" or scenario.selection_risk_column is not None:
        raise ValueError(
            "the exact-expectation design covers the uniform-risk displacement margin"
        )
    lines = (baseline_metrics["poverty_line_bhc"], baseline_metrics["poverty_line_ahc"])
    household_ids = np.asarray(
        baseline.calculate("household_id", period=period, map_to="household").values
    )
    person_household_id = np.asarray(
        baseline.calculate("household_id", period=period, map_to="person").values
    )
    household = pd.Index(household_ids).get_indexer(person_household_id)
    if (household < 0).any():
        raise RuntimeError("person household ids not found among households")
    household_weight = np.asarray(
        baseline.calculate("household_weight", period=period, map_to="household").values,
        dtype=float,
    )
    if not np.allclose(household_weight[household], persons["weight"].to_numpy(float)):
        raise RuntimeError("person weights differ from their household weights")
    earnings = persons["employment_income"].to_numpy(dtype=float)
    exposed = (earnings > 0) & (_person_shock(persons, scenario) > 0)
    rank = exposure_ranks(household, exposed)
    base = _household_outcomes(baseline, period, lines)
    max_unaffected = 0.0
    n_sims = 0

    def one_pass(displaced: np.ndarray, label: str) -> dict:
        nonlocal max_unaffected, n_sims
        touched = np.zeros(household_weight.size, dtype=bool)
        touched[household[displaced]] = True
        by_takeup = {}
        for node in TAKEUP_NODES:
            table = build_table(displaced, replace(scenario, uc_takeup=node))
            sim = build_shocked_simulation(dataset, baseline, table, period)
            outcome = _household_outcomes(sim, period, lines)
            del sim
            n_sims += 1
            delta = {name: outcome[name] - base[name] for name in base}
            worst = max(float(np.abs(v[~touched]).max()) for v in delta.values())
            max_unaffected = max(max_unaffected, worst)
            if worst > SEPARABILITY_TOL:
                raise RuntimeError(
                    f"pass {label}: a household with no displaced member changed "
                    f"by £{worst:,.2f}; the design's household separability fails"
                )
            for v in delta.values():
                v[~touched] = 0.0
            by_takeup[node] = delta
            log(f"[expectation] pass {label} take-up {node:.0f}: "
                f"{int(displaced.sum())} records displaced")
        return by_takeup

    single = [one_pass(rank == k, f"rank {k}") for k in range(int(rank.max()) + 1)]
    pair = one_pass((rank == 0) | (rank == 1), "ranks 0+1")
    # Free determinism check: in a household with ONE exposed worker the joint
    # pass displaces the same person as the rank-0 pass, so it must reproduce
    # that pass's household change.
    multi = np.zeros(household_weight.size, dtype=bool)
    multi[household[rank == 1]] = True
    gap = max(
        float(np.abs(pair[node][name][~multi] - single[0][node][name][~multi]).max())
        for node in TAKEUP_NODES
        for name in ("hbai_household_net_income", "universal_credit", "income_tax")
    )
    if gap > SEPARABILITY_TOL:
        raise RuntimeError(
            f"joint pass does not reproduce the rank-0 pass (gap £{gap:,.2f})"
        )
    return ExpectationPasses(
        period=period,
        household=household,
        household_weight=household_weight,
        rank=rank,
        single=single,
        pair=pair,
        n_simulations=n_sims,
        max_unaffected_change=max_unaffected,
        max_single_exposed_pair_gap=gap,
    )


def expected_household_changes(
    passes: ExpectationPasses, shock: np.ndarray, uc_takeup: float
) -> tuple[dict[str, np.ndarray], dict]:
    """E[change] per household for every carried outcome, and diagnostics."""
    household, rank = passes.household, passes.rank
    p_single, p_pair = subset_probabilities(shock, household, rank)
    n_households = passes.household_weight.size
    names = list(passes.single[0][1.0])
    expected = {name: np.zeros(n_households) for name in names}
    for k, deltas in enumerate(passes.single):
        members = rank == k
        hh = household[members]
        mixed = passes.mixed(deltas, uc_takeup)
        for name in names:
            # each household holds at most one rank-k member
            expected[name][hh] += p_single[members] * mixed[name][hh]
    mixed_pair = passes.mixed(passes.pair, uc_takeup)
    for name in names:
        expected[name] += p_pair * mixed_pair[name]
    first_order = {name: np.zeros(n_households) for name in ("employment_income", "hbai_household_net_income")}
    for k, deltas in enumerate(passes.single):
        members = rank == k
        hh = household[members]
        mixed = passes.mixed(deltas, uc_takeup)
        for name in first_order:
            first_order[name][hh] += shock[members] * mixed[name][hh]
    return expected, {"first_order": first_order, "p_single": p_single, "p_pair": p_pair}


def _incidence_by_decile_and_region(
    expected_net: np.ndarray,
    household: np.ndarray,
    household_weight: np.ndarray,
    weight: np.ndarray,
    baseline_equiv: np.ndarray,
    baseline_region: np.ndarray,
) -> tuple[dict, dict]:
    """Person-weighted mean change in household income per head, by baseline
    equivalised-income decile and by region.

    The household total change is spread over its members, so the sum over
    persons of weight * change / people equals w_h * change_h per household.
    """
    net_change_person = (expected_net * household_weight)[household]
    people = np.bincount(household, minlength=household_weight.size).astype(float)
    per_head = net_change_person / people[household]
    order = np.argsort(baseline_equiv)
    cum = np.cumsum(weight[order])
    ranks = np.empty(len(weight))
    ranks[order] = cum / cum[-1]
    deciles = np.clip(np.ceil(ranks * 10).astype(int), 1, 10)
    decile_change = {
        int(d): float(per_head[deciles == d].sum() / weight[deciles == d].sum())
        for d in range(1, 11)
    }
    region_change = {
        r: float(per_head[baseline_region == r].sum() / weight[baseline_region == r].sum())
        for r in sorted(set(baseline_region))
    }
    return decile_change, region_change


def _clustered_ratio_se(numerator_h: np.ndarray, gross_h: np.ndarray) -> float:
    """Household-clustered linearisation SE of sum(numerator) / sum(gross)."""
    active = gross_h > 0
    gross = float(gross_h.sum())
    ratio = float(numerator_h.sum()) / gross
    z = (numerator_h[active] - ratio * gross_h[active]) / gross
    m = int(active.sum())
    return float(np.sqrt(m / (m - 1) * np.square(z).sum()))


def expectation_result(
    passes: ExpectationPasses,
    persons: pd.DataFrame,
    scenario: TradeShockScenario,
    baseline_metrics: dict,
    baseline_equiv: np.ndarray,
    baseline_region: np.ndarray,
) -> dict:
    """Summarise a displacement scenario under the exact-expectation design.

    ``passes`` may be reused across scenarios that share the exposed set, the
    New Style JSA switch and the take-up scope (e.g. a different elasticity:
    the household changes do not depend on s_i, only the weights do).
    """
    shock = _person_shock(persons, scenario)
    weight = persons["weight"].to_numpy(dtype=float)
    earnings = persons["employment_income"].to_numpy(dtype=float)
    age = persons["age"].to_numpy(dtype=float)
    exposed = passes.rank >= 0
    if not np.array_equal(exposed, (earnings > 0) & (shock > 0)):
        raise ValueError("scenario's exposed set differs from the passes'")
    w = passes.household_weight
    expected, diag = expected_household_changes(passes, shock, scenario.uc_takeup)
    totals = {name: float((w * values).sum()) for name, values in expected.items()}
    components = cushion_components(totals)
    gross = components["gross_loss"]
    net = -totals["hbai_household_net_income"]
    target_gross = float((shock * earnings * weight)[exposed].sum())
    rate = 1.0 - net / gross

    # First-order (no within-household interaction) comparator.
    fo_gross = -float((w * diag["first_order"]["employment_income"]).sum())
    fo_net = -float((w * diag["first_order"]["hbai_household_net_income"]).sum())

    # Household-clustered linearisation SE of the ratio (sampling, not MC).
    g_h = -w * expected["employment_income"]
    n_h = -w * expected["hbai_household_net_income"]
    active = g_h > 0
    z = (n_h[active] - (net / gross) * g_h[active]) / gross
    m = int(active.sum())
    sampling_se = float(np.sqrt(m / (m - 1) * np.square(z).sum()))
    component_se = {}
    for group in ("income_tax", "national_insurance", "universal_credit"):
        sign = -1.0 if group in ("income_tax", "national_insurance") else 1.0
        c_h = sign * w * expected[group]
        zc = (c_h[active] - (components[group] / gross) * g_h[active]) / gross
        component_se[group] = float(np.sqrt(m / (m - 1) * np.square(zc).sum()))

    record_loss = (shock * earnings * weight)[exposed]
    person_w_total = float(weight.sum())
    decile_change, region_change = _incidence_by_decile_and_region(
        expected["hbai_household_net_income"], passes.household, w, weight,
        baseline_equiv, baseline_region,
    )
    displaced_w = float((shock * weight)[exposed].sum())
    band_share = {}
    for lo, hi in AGE_BANDS:
        mask = exposed & (age >= lo) & (age <= hi)
        band_share[f"{lo}-{hi if hi < 200 else '+'}"] = float(
            (shock * weight)[mask].sum() / displaced_w
        )
    return {
        "scenario": scenario.name,
        "tariff_scenario": scenario.tariff_scenario,
        "margin": scenario.margin,
        "design": "exact_expectation",
        "selection_method": "exact_expectation",
        "design_note": (
            "Expectation over independent Bernoulli(s_i) displacement of every "
            "exposed employee, computed from household-separable simulation "
            "passes (tariff_pipeline/expectation.py). No assignment draws: "
            "Monte Carlo error is zero by construction."
        ),
        "elasticity": scenario.elasticity,
        "uc_takeup": scenario.uc_takeup,
        "uc_takeup_scope": scenario.uc_takeup_scope,
        "new_style_jsa": scenario.new_style_jsa,
        "new_style_jsa_weeks": scenario.new_style_jsa_weeks,
        "n_draws": 0,
        "draws": [],
        "n_simulation_passes": passes.n_simulations,
        "exposed_records": int(exposed.sum()),
        "exposed_households": m,
        "households_with_two_or_more_exposed": int((np.bincount(passes.household[exposed], minlength=w.size) >= 2).sum()),
        "effective_loss_records": float(record_loss.sum() ** 2 / np.square(record_loss).sum()),
        "max_record_loss_share": float(record_loss.max() / record_loss.sum()),
        "target_gross_earnings_loss": target_gross,
        "gross_earnings_loss": gross,
        "omitted_gross_share": 1.0 - gross / target_gross,
        "net_disposable_loss": net,
        "cushioning_rate": rate,
        # legacy keys, so readers of the Monte Carlo schema keep working
        "cushioning_rate_mean": rate,
        "cushioning_rate_pooled": rate,
        "cushioning_rate_sd": 0.0,
        "cushioning_rate_mc_se": 0.0,
        "cushioning_rate_first_order": 1.0 - fo_net / fo_gross,
        "cushioning_rate_sampling_se": sampling_se,
        "sampling_se_note": (
            "Household-clustered linearisation SE of the ratio over exposed "
            "households, weights treated as fixed. Survey sampling error, "
            "NOT Monte Carlo error."
        ),
        "cushion_components_mean": components,
        "cushion_component_shares": component_shares(components),
        "cushion_component_share_of_gross_sampling_se": component_se,
        "exchequer_cost_mean": -totals["gov_balance"],
        "poverty_rate_change_bhc_mean": totals["poor_people_bhc"] / person_w_total,
        "poverty_rate_change_ahc_mean": totals["poor_people_ahc"] / person_w_total,
        "gini_change_mean": None,
        "displaced_weighted_mean": displaced_w,
        "decile_income_change": decile_change,
        "region_income_change": region_change,
        "age_band_displacement_share": band_share,
        "max_unaffected_household_change": passes.max_unaffected_change,
        "max_single_exposed_pair_gap": passes.max_single_exposed_pair_gap,
    }


def validate_against_draw(
    dataset, baseline, persons, scenario, passes: ExpectationPasses, period: int, seed: int = 0
) -> dict:
    """Check the passes against one REALISED Bernoulli draw.

    Runs one ordinary Bernoulli displacement draw with every newly entitled
    unit claiming (uc_takeup = 1) and compares each affected household's
    change in HBAI net income with the change the passes predict for the
    same displaced set: the single pass of that worker's rank, or the joint
    pass when the draw displaces exactly a household's rank-0 and rank-1
    workers. Households whose displaced set the passes do not cover are
    counted and skipped. Raises on any mismatch above SEPARABILITY_TOL.
    """
    from tariff_pipeline.shocks import apply_shocks

    draw_scenario = replace(scenario, uc_takeup=1.0, selection_method="bernoulli")
    table = apply_shocks(persons, draw_scenario, seed=seed)
    displaced = table["displaced"].to_numpy(dtype=bool)
    sim = build_shocked_simulation(dataset, baseline, table, period)

    def _net(s):
        return np.asarray(
            s.calculate("hbai_household_net_income", period=period, map_to="household").values,
            dtype=float,
        )

    realised = _net(sim) - _net(baseline)
    hh = passes.household
    predicted = np.zeros_like(realised)
    count = np.bincount(hh[displaced], minlength=realised.size)
    covered = np.zeros(realised.size, dtype=bool)
    name = "hbai_household_net_income"
    for i in np.flatnonzero(displaced):
        h = hh[i]
        if count[h] == 1:
            predicted[h] = passes.single[passes.rank[i]][1.0][name][h]
            covered[h] = True
    for h in np.flatnonzero(count == 2):
        ranks = sorted(passes.rank[displaced & (hh == h)])
        if ranks == [0, 1]:
            predicted[h] = passes.pair[1.0][name][h]
            covered[h] = True
    affected = count > 0
    gap = float(np.abs(realised[covered] - predicted[covered]).max()) if covered.any() else 0.0
    stray = float(np.abs(realised[~affected]).max())
    if gap > SEPARABILITY_TOL or stray > SEPARABILITY_TOL:
        raise RuntimeError(
            f"exact-expectation passes disagree with a realised draw: "
            f"max household gap £{gap:,.2f}, unaffected change £{stray:,.2f}"
        )
    w = passes.household_weight
    return {
        "seed": seed,
        "displaced_records": int(displaced.sum()),
        "affected_households": int(affected.sum()),
        "covered_households": int(covered.sum()),
        "max_household_gap_gbp": gap,
        "realised_net_loss": -float((w * realised).sum()),
        "predicted_net_loss_covered": -float((w * predicted).sum()),
    }


def wage_cut_expectation_result(
    dataset,
    baseline,
    persons: pd.DataFrame,
    scenario: TradeShockScenario,
    period: int,
    baseline_metrics: dict,
    baseline_equiv: np.ndarray,
    baseline_region: np.ndarray,
    log=print,
) -> dict:
    """Wage-cut margin with the claiming re-draw mixed exactly.

    Two simulations of the SAME deterministic wage cut, with the newly
    entitled benefit units forced to all claim (1.0) and to none claim (0.0);
    every household outcome is mixed with weights ``uc_takeup`` and
    ``1 - uc_takeup``. For a household with one re-drawn benefit unit that is
    the expectation over its claiming draw; the number of households holding
    two or more is reported (``households_with_two_redrawn_units``) and the
    mix treats their claiming as perfectly correlated. The keys the Monte
    Carlo schema's readers use are kept; ``n_draws`` is 0 and ``draws`` is
    empty because nothing is drawn. The Gini is non-linear, so it is
    reported at the two nodes rather than as an expectation.
    """
    from tariff_pipeline.runner import gini

    if scenario.margin != "wage_cut":
        raise ValueError("wage_cut_expectation_result requires the wage_cut margin")
    lines = (baseline_metrics["poverty_line_bhc"], baseline_metrics["poverty_line_ahc"])

    def _calc(sim, name, entity="household", dtype=float):
        return np.asarray(sim.calculate(name, period=period, map_to=entity).values, dtype=dtype)

    household_ids = _calc(baseline, "household_id", dtype=None)
    household = pd.Index(household_ids).get_indexer(
        _calc(baseline, "household_id", "person", dtype=None)
    )
    w = _calc(baseline, "household_weight")
    people = _calc(baseline, "household_count_people")
    weight = persons["weight"].to_numpy(dtype=float)
    earnings = persons["employment_income"].to_numpy(dtype=float)
    shock = _person_shock(persons, scenario)
    exposed = (earnings > 0) & (shock > 0)
    base = _household_outcomes(baseline, period, lines)
    base_flag = _calc(baseline, "would_claim_uc", "benunit", dtype=bool)
    benunit_household = pd.Index(household_ids).get_indexer(
        _calc(baseline, "household_id", "benunit", dtype=None)
    )
    delta, node_summary, flags = {}, {}, {}
    for node in TAKEUP_NODES:
        table = apply_shocks(persons, replace(scenario, uc_takeup=node), seed=0)
        sim = build_shocked_simulation(dataset, baseline, table, period)
        outcome = _household_outcomes(sim, period, lines)
        delta[node] = {name: outcome[name] - base[name] for name in base}
        flags[node] = _calc(sim, "would_claim_uc", "benunit", dtype=bool)
        gross = -float((w * delta[node]["employment_income"]).sum())
        net = -float((w * delta[node]["hbai_household_net_income"]).sum())
        node_summary[f"{node:.0f}"] = {
            "cushioning_rate": 1.0 - net / gross,
            "net_disposable_loss": net,
            "gini_change": gini(_calc(sim, "equiv_hbai_household_net_income"), w * people)
            - baseline_metrics["gini"],
            "uc_takeup_redraw": uc_takeup_redraw_diagnostic(table),
        }
        del sim
        log(f"[expectation] wage cut take-up {node:.0f}: "
            f"cushioning {100 * node_summary[f'{node:.0f}']['cushioning_rate']:.3f}%")
    # The re-drawn benefit units are exactly those whose flag differs between
    # the two forced nodes; everything else is identical across nodes.
    redrawn = flags[1.0] != flags[0.0]
    if (flags[0.0] & redrawn).any() or not np.array_equal(
        flags[1.0][~redrawn], base_flag[~redrawn]
    ):
        raise RuntimeError("forced take-up nodes are not all-claim / none-claim")
    per_household = np.bincount(benunit_household[redrawn], minlength=w.size)
    untouched = per_household == 0
    node_gap = max(
        float(np.abs(delta[1.0][name][untouched] - delta[0.0][name][untouched]).max())
        for name in base
    )
    if node_gap > SEPARABILITY_TOL:
        raise RuntimeError(
            f"a household with no re-drawn benefit unit differs between the "
            f"take-up nodes by £{node_gap:,.2f}"
        )
    mix = scenario.uc_takeup
    expected = {
        name: mix * delta[1.0][name] + (1.0 - mix) * delta[0.0][name] for name in base
    }
    totals = {name: float((w * values).sum()) for name, values in expected.items()}
    components = cushion_components(totals)
    gross = components["gross_loss"]
    net = -totals["hbai_household_net_income"]
    target_gross = float((shock * earnings * weight)[exposed].sum())
    if not np.isclose(gross, target_gross, rtol=1e-6, atol=1.0):
        raise AssertionError(
            f"simulated wage-cut loss £{gross:,.0f} differs from the imposed "
            f"£{target_gross:,.0f}"
        )
    rate = 1.0 - net / gross
    g_h = -w * expected["employment_income"]
    sampling_se = _clustered_ratio_se(-w * expected["hbai_household_net_income"], g_h)
    decile_change, region_change = _incidence_by_decile_and_region(
        expected["hbai_household_net_income"], household, w, weight,
        baseline_equiv, baseline_region,
    )
    record_loss = (shock * earnings * weight)[exposed]
    person_w_total = float(weight.sum())
    return {
        "scenario": scenario.name,
        "tariff_scenario": scenario.tariff_scenario,
        "margin": scenario.margin,
        "design": "exact_takeup_expectation",
        "selection_method": "exact_takeup_expectation",
        "design_note": (
            "Deterministic wage cut (every exposed employee loses s_i of "
            "earnings). The Universal Credit claiming re-draw of newly "
            "entitled benefit units is mixed exactly from one all-claim and "
            "one none-claim simulation with weights uc_takeup and "
            "1 - uc_takeup (tariff_pipeline/expectation.py). No draws: Monte "
            "Carlo error is zero by construction."
        ),
        "elasticity": scenario.elasticity,
        "uc_takeup": scenario.uc_takeup,
        "uc_takeup_scope": scenario.uc_takeup_scope,
        "n_draws": 0,
        "draws": [],
        "n_simulation_passes": len(TAKEUP_NODES),
        "takeup_nodes": node_summary,
        "redrawn_benefit_units": int(redrawn.sum()),
        "households_with_two_redrawn_units": int((per_household >= 2).sum()),
        "max_untouched_household_node_gap": node_gap,
        "exposed_records": int(exposed.sum()),
        "exposed_households": int((g_h > 0).sum()),
        "effective_loss_records": float(record_loss.sum() ** 2 / np.square(record_loss).sum()),
        "max_record_loss_share": float(record_loss.max() / record_loss.sum()),
        "target_gross_earnings_loss": target_gross,
        "gross_earnings_loss": gross,
        "net_disposable_loss": net,
        "cushioning_rate": rate,
        # legacy keys, so readers of the Monte Carlo schema keep working
        "cushioning_rate_mean": rate,
        "cushioning_rate_pooled": rate,
        "cushioning_rate_sd": 0.0,
        "cushioning_rate_mc_se": 0.0,
        "cushioning_rate_sampling_se": sampling_se,
        "sampling_se_note": (
            "Household-clustered linearisation SE of the ratio over exposed "
            "households, weights treated as fixed. Survey sampling error, "
            "NOT Monte Carlo error."
        ),
        "cushion_components_mean": components,
        "cushion_component_shares": component_shares(components),
        "exchequer_cost_mean": -totals["gov_balance"],
        "poverty_rate_change_bhc_mean": totals["poor_people_bhc"] / person_w_total,
        "poverty_rate_change_ahc_mean": totals["poor_people_ahc"] / person_w_total,
        "gini_change_mean": None,
        "displaced_weighted_mean": 0.0,
        "decile_income_change": decile_change,
        "region_income_change": region_change,
    }


__all__ = [
    "CUSHION_GROUP_NAMES",
    "ExpectationPasses",
    "expectation_result",
    "expected_household_changes",
    "exposure_ranks",
    "run_passes",
    "subset_probabilities",
    "validate_against_draw",
    "wage_cut_expectation_result",
]
