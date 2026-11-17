"""Tests for the E3 referee reruns: the exact-expectation displacement
design, the tax/benefit split of the cushion, and the New Style JSA
approximation. Pure functions only -- no microdata."""

import dataclasses
import itertools

import numpy as np
import pandas as pd
import pytest

from tariff_pipeline.expectation import (
    ExpectationPasses,
    expected_household_changes,
    exposure_ranks,
    subset_probabilities,
)
from tariff_pipeline.runner import (
    CUSHION_FLOOR,
    CUSHION_GROUP_NAMES,
    CUSHION_GROUPS,
    CUSHION_OTHER,
    _hbai_lists,
    component_shares,
    cushion_components,
)
from tariff_pipeline.shocks import (
    PRESETS,
    TradeShockScenario,
    apply_fixed_displacement,
    apply_shocks,
    new_style_jsa_amounts,
)


# --------------------------------------------------------------------------
# tax/benefit split
# --------------------------------------------------------------------------


def _deltas(rng, floor=0.0):
    adds, subtracts = _hbai_lists()
    delta = {name: float(rng.normal(0, 100)) for name in (*adds, *subtracts)}
    delta["employment_income"] = -10_000.0
    before = sum(delta[a] for a in adds) - sum(delta[s] for s in subtracts)
    delta["hbai_household_net_income_before_reset_to_zero"] = before
    delta["hbai_household_net_income"] = before + floor
    return delta


def test_cushion_groups_are_hbai_components_and_disjoint():
    adds, subtracts = _hbai_lists()
    members = [m for group in CUSHION_GROUPS.values() for m in group]
    assert len(members) == len(set(members))
    assert set(members) <= set(adds) | set(subtracts)
    assert "employment_income" not in members


def test_split_sums_to_gross_minus_net():
    delta = _deltas(np.random.default_rng(1), floor=37.0)
    out = cushion_components(delta)
    gross_loss = -delta["employment_income"]
    net_loss = -delta["hbai_household_net_income"]
    assert out["gross_loss"] == pytest.approx(gross_loss)
    assert out["cushion"] == pytest.approx(gross_loss - net_loss)
    assert sum(out[g] for g in CUSHION_GROUP_NAMES) == pytest.approx(out["cushion"])
    assert out[CUSHION_FLOOR] == pytest.approx(37.0)


def test_split_signs_lower_tax_and_higher_benefit_both_cushion():
    adds, subtracts = _hbai_lists()
    delta = {name: 0.0 for name in (*adds, *subtracts)}
    delta.update(employment_income=-1000.0, income_tax=-200.0, national_insurance=-80.0,
                 universal_credit=300.0, student_loan_repayments=-9.0,
                 statutory_sick_pay=-50.0, free_school_meals=20.0)
    before = sum(delta[a] for a in adds) - sum(delta[s] for s in subtracts)
    delta["hbai_household_net_income_before_reset_to_zero"] = before
    delta["hbai_household_net_income"] = before
    out = cushion_components(delta)
    assert out["income_tax"] == pytest.approx(200.0)
    assert out["national_insurance"] == pytest.approx(80.0)
    assert out["universal_credit"] == pytest.approx(300.0)
    assert out["student_loan_repayments"] == pytest.approx(9.0)
    assert out["statutory_pay"] == pytest.approx(-50.0)
    assert out[CUSHION_OTHER] == pytest.approx(20.0)
    shares = component_shares(out)
    assert sum(shares["share_of_cushion"].values()) == pytest.approx(1.0)
    assert sum(shares["share_of_gross_loss"].values()) == pytest.approx(
        out["cushion"] / out["gross_loss"]
    )


def test_split_raises_when_components_do_not_sum_to_net_income():
    delta = _deltas(np.random.default_rng(2))
    delta["hbai_household_net_income_before_reset_to_zero"] += 500.0
    delta["hbai_household_net_income"] += 500.0
    with pytest.raises(AssertionError, match="do not sum"):
        cushion_components(delta)


# --------------------------------------------------------------------------
# exact-expectation design
# --------------------------------------------------------------------------


def test_exposure_ranks_count_exposed_members_within_household():
    household = np.array([0, 0, 1, 2, 2, 2])
    exposed = np.array([True, True, True, False, True, True])
    assert exposure_ranks(household, exposed).tolist() == [0, 1, 0, -1, 0, 1]


def test_subset_probabilities_match_bernoulli_enumeration():
    household = np.array([0, 0, 1, 2, 2, 2])
    shock = np.array([0.1, 0.2, 0.3, 0.0, 0.4, 0.5])
    rank = exposure_ranks(household, shock > 0)
    p_single, p_pair = subset_probabilities(shock, household, rank)
    assert p_single == pytest.approx([0.1 * 0.8, 0.2 * 0.9, 0.3, 0.0, 0.4 * 0.5, 0.5 * 0.6])
    assert p_pair == pytest.approx([0.1 * 0.2, 0.0, 0.4 * 0.5])


def _toy_passes(household, rank, single_net, pair_net):
    """Passes whose take-up nodes agree, carrying net income and earnings."""
    def both(values):
        d = {"hbai_household_net_income": np.asarray(values, float),
             "employment_income": np.asarray(values, float)}
        return {1.0: d, 0.0: d}

    return ExpectationPasses(
        period=2026, household=household, household_weight=np.ones(int(household.max()) + 1),
        rank=rank, single=[both(v) for v in single_net], pair=both(pair_net),
        n_simulations=0, max_unaffected_change=0.0, max_single_exposed_pair_gap=0.0,
    )


def test_expectation_equals_brute_force_enumeration_with_interaction():
    # one household, two exposed workers, and a NON-additive joint outcome
    household = np.array([0, 0])
    shock = np.array([0.3, 0.6])
    rank = exposure_ranks(household, shock > 0)
    d0, d1, d01 = -100.0, -40.0, -500.0  # jointly far worse than the sum
    passes = _toy_passes(household, rank, [[d0], [d1]], [d01])
    expected, _ = expected_household_changes(passes, shock, uc_takeup=0.8)
    brute = 0.0
    for a, b in itertools.product((0, 1), repeat=2):
        p = (shock[0] if a else 1 - shock[0]) * (shock[1] if b else 1 - shock[1])
        brute += p * {(0, 0): 0.0, (1, 0): d0, (0, 1): d1, (1, 1): d01}[(a, b)]
    assert expected["hbai_household_net_income"][0] == pytest.approx(brute)
    # a per-record reweighting (no joint pass) would get this wrong
    assert shock[0] * d0 + shock[1] * d1 != pytest.approx(brute)


def test_expectation_mixes_takeup_nodes_at_the_scenario_rate():
    household = np.array([0])
    rank = np.array([0])
    claim = {"hbai_household_net_income": np.array([-100.0]), "employment_income": np.array([-1000.0])}
    no_claim = {"hbai_household_net_income": np.array([-900.0]), "employment_income": np.array([-1000.0])}
    zero = {k: np.zeros(1) for k in claim}
    passes = ExpectationPasses(
        period=2026, household=household, household_weight=np.ones(1), rank=rank,
        single=[{1.0: claim, 0.0: no_claim}], pair={1.0: zero, 0.0: zero},
        n_simulations=0, max_unaffected_change=0.0, max_single_exposed_pair_gap=0.0,
    )
    expected, _ = expected_household_changes(passes, np.array([0.5]), uc_takeup=0.8)
    assert expected["hbai_household_net_income"][0] == pytest.approx(0.5 * (0.8 * -100 + 0.2 * -900))
    assert expected["employment_income"][0] == pytest.approx(-500.0)


def _persons():
    return pd.DataFrame(
        {
            "person_id": np.arange(4),
            "age": [30.0, 45.0, 22.0, 60.0],
            "employment_income": [30_000.0, 0.0, 18_000.0, 40_000.0],
            "sic_division": [29.0, 29.0, 29.0, 86.0],
            "weight": [100.0, 100.0, 100.0, 100.0],
        }
    )


def test_fixed_displacement_table_matches_apply_shocks_shape():
    persons = _persons()
    scenario = dataclasses.replace(PRESETS["full_tariff_displacement"], new_style_jsa=True)
    mask = np.array([True, False, False, False])
    table = apply_fixed_displacement(persons, scenario, mask, seed=3)
    drawn = apply_shocks(persons, scenario, seed=3)
    assert set(table.columns) == set(drawn.columns)
    assert table.attrs == drawn.attrs
    assert table.attrs["new_style_jsa"] is True
    assert table["employment_income"].tolist() == [0.0, 0.0, 18_000.0, 40_000.0]
    assert table["displaced"].tolist() == mask.tolist()
    with pytest.raises(ValueError, match="positive earnings"):
        apply_fixed_displacement(persons, scenario, np.array([False, True, False, False]))


# --------------------------------------------------------------------------
# New Style JSA approximation
# --------------------------------------------------------------------------

RATES = dict(weekly_rate_18_24=75.65, weekly_rate_25_plus=95.55, weekly_lower_earnings_limit=125.0)


def test_new_style_jsa_pays_26_weeks_at_the_age_rate():
    award = new_style_jsa_amounts(
        displaced_unemployed=[True, True, True],
        age=[24, 25, 40],
        baseline_earnings=[20_000, 20_000, 20_000],
        state_pension_age=[False, False, False],
        **RATES,
    )
    assert award == pytest.approx([26 * 75.65, 26 * 95.55, 26 * 95.55])


def test_new_style_jsa_eligibility_gates():
    award = new_style_jsa_amounts(
        #                     not displaced, under 18, pension age, low earnings, eligible
        displaced_unemployed=[False, True, True, True, True],
        age=[40, 17, 67, 40, 40],
        baseline_earnings=[30_000, 30_000, 30_000, 50 * 125.0 - 1, 50 * 125.0],
        state_pension_age=[False, False, True, False, False],
        **RATES,
    )
    assert award[:4] == pytest.approx([0, 0, 0, 0])
    assert award[4] == pytest.approx(26 * 95.55)


def test_new_style_jsa_is_capped_at_26_weeks_and_off_by_default():
    with pytest.raises(ValueError, match="26 weeks"):
        new_style_jsa_amounts([True], [40], [30_000], [False], **RATES, weeks=27)
    with pytest.raises(ValueError):
        TradeShockScenario("x", "full_tariff", "displacement", new_style_jsa_weeks=30)
    assert PRESETS["full_tariff_displacement"].new_style_jsa is False
    assert apply_shocks(_persons(), PRESETS["full_tariff_displacement"]).attrs["new_style_jsa"] is False
