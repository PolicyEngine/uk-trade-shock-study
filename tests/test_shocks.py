"""Unit tests for the adjustment-margin mechanics (synthetic tables only)."""

import numpy as np
import pandas as pd
import pytest

from uk_trade_shock_study.exposure import DEFAULT_ELASTICITY, person_earnings_shock
from uk_trade_shock_study.shocks import (
    MARGINS,
    PRESETS,
    TradeShockScenario,
    apply_shocks,
    apply_wage_cut,
    draw_displaced,
)


def make_persons(n=3000, seed=1):
    rng = np.random.default_rng(seed)
    # mix of exposed manufacturing divisions and unexposed services
    division = rng.choice([21.0, 24.0, 28.0, 29.0, 62.0, 86.0, np.nan], n)
    return pd.DataFrame(
        {
            "age": rng.integers(18, 64, n),
            "employment_income": rng.lognormal(10, 0.6, n),
            "weight": rng.uniform(100, 2000, n),
            "sic_division": division,
        }
    )


def test_presets_cover_the_grid():
    assert set(PRESETS) == {
        f"{t}_{m}" for t in ("full_tariff", "epd") for m in MARGINS
    }


def test_bad_margin_errors():
    with pytest.raises(ValueError):
        TradeShockScenario("t", "epd", "nonsense")


def test_displacement_quota_in_expectation():
    """Expected displaced weight per division equals shock_j x employee weight."""
    persons = make_persons()
    scenario = PRESETS["full_tariff_displacement"]
    w = persons["weight"].to_numpy()
    shock = person_earnings_shock(persons["sic_division"], "full_tariff")
    employed = persons["employment_income"].to_numpy() > 0
    expected = float((shock * w)[employed].sum())
    realised = np.mean(
        [w[draw_displaced(persons, scenario, seed=s)].sum() for s in range(200)]
    )
    assert realised == pytest.approx(expected, rel=0.05)


def test_equal_inclusion_regardless_of_weight():
    """Within a division, displacement probability must not depend on the
    grossing weight (uniform ordering keys; template finding 6)."""
    persons = pd.DataFrame(
        {
            "age": [40, 40],
            "employment_income": [30000.0, 30000.0],
            "weight": [1.0, 9.0],
            "sic_division": [29.0, 29.0],
        }
    )
    # calibrate a ~50% division shock via elasticity
    shock = person_earnings_shock(np.array([29.0]), "full_tariff")[0]
    scenario = TradeShockScenario("t", "full_tariff", "displacement", elasticity=DEFAULT_ELASTICITY * 0.5 / shock)
    hits, n = np.zeros(2), 4000
    for s in range(n):
        hits += draw_displaced(persons, scenario, seed=s)
    assert hits[0] / n == pytest.approx(0.5, abs=0.03)
    assert hits[1] / n == pytest.approx(0.5, abs=0.03)


def test_displaced_earn_zero_and_unexposed_untouched():
    persons = make_persons()
    shocked = apply_shocks(persons, PRESETS["full_tariff_displacement"], seed=0)
    displaced = shocked["displaced"].to_numpy()
    assert displaced.any()
    assert (shocked["employment_income"].to_numpy()[displaced] == 0).all()
    unexposed = person_earnings_shock(persons["sic_division"], "full_tariff") == 0
    assert not displaced[unexposed].any()
    np.testing.assert_array_equal(
        shocked["employment_income"].to_numpy()[unexposed],
        persons["employment_income"].to_numpy()[unexposed],
    )
    assert not shocked["inactive"].to_numpy().any()


def test_wage_bill_conservation():
    """Wage-cut aggregate loss equals sum_j shock_j x division wage bill —
    the displacement family's expected earnings removal."""
    persons = make_persons()
    scenario = PRESETS["full_tariff_wage_cut"]
    shocked = apply_wage_cut(persons, scenario)
    base = persons["employment_income"].to_numpy()
    new = shocked["employment_income"].to_numpy()
    w = persons["weight"].to_numpy()
    shock = person_earnings_shock(persons["sic_division"], "full_tariff")
    employed = base > 0
    loss = ((base - new) * w)[employed].sum()
    target = (shock * base * w)[employed].sum()
    assert loss == pytest.approx(target, rel=1e-9)
    # no job loss, no negative incomes
    assert (new[employed] > 0).all()
    assert not shocked["displaced"].to_numpy().any()


def test_wage_cut_gradient_matches_sector_shock():
    persons = make_persons()
    shocked = apply_wage_cut(persons, PRESETS["epd_wage_cut"])
    base = persons["employment_income"].to_numpy()
    new = shocked["employment_income"].to_numpy()
    shock = person_earnings_shock(persons["sic_division"], "epd")
    employed = base > 0
    np.testing.assert_allclose(
        (base - new)[employed] / base[employed], shock[employed], rtol=1e-9
    )


def test_inactivity_margin_age_split():
    """Older displaced workers flow to inactivity; younger to unemployment."""
    persons = make_persons()
    scenario = PRESETS["full_tariff_inactivity"]
    shocked = apply_shocks(persons, scenario, seed=0)
    displaced = shocked["displaced"].to_numpy()
    inactive = shocked["inactive"].to_numpy()
    age = persons["age"].to_numpy()
    assert inactive.any()
    assert (inactive <= displaced).all()  # inactive is a subset of displaced
    assert (age[inactive] >= scenario.inactivity_age_threshold).all()
    assert (age[displaced & ~inactive] < scenario.inactivity_age_threshold).all()
    # same draw as pure displacement (same seed): identical displaced mask
    pure = apply_shocks(persons, PRESETS["full_tariff_displacement"], seed=0)
    np.testing.assert_array_equal(displaced, pure["displaced"].to_numpy())
    # under the default upper-bound assumption every inactive worker is
    # flagged LCWRA; pure displacement flags nobody
    np.testing.assert_array_equal(shocked["lcwra"].to_numpy(), inactive)
    assert not pure["lcwra"].to_numpy().any()


def test_inactivity_lcwra_takeup_thinning():
    """lcwra_takeup < 1 thins the LCWRA flag within the inactive set without
    changing the displacement draw."""
    persons = make_persons()
    full = PRESETS["full_tariff_inactivity"]
    half = TradeShockScenario(
        "t", "full_tariff", "inactivity", lcwra_takeup=0.5
    )
    shocked_full = apply_shocks(persons, full, seed=0)
    shocked_half = apply_shocks(persons, half, seed=0)
    np.testing.assert_array_equal(
        shocked_full["displaced"].to_numpy(), shocked_half["displaced"].to_numpy()
    )
    np.testing.assert_array_equal(
        shocked_full["inactive"].to_numpy(), shocked_half["inactive"].to_numpy()
    )
    lcwra = shocked_half["lcwra"].to_numpy()
    inactive = shocked_half["inactive"].to_numpy()
    assert (lcwra <= inactive).all()
    # thinned strictly below the full-takeup count, at roughly half in
    # expectation across seeds
    shares = []
    for s in range(50):
        t = apply_shocks(persons, half, seed=s)
        shares.append(t["lcwra"].to_numpy().sum() / max(t["inactive"].to_numpy().sum(), 1))
    assert np.mean(shares) == pytest.approx(0.5, abs=0.1)


def test_epd_displaces_fewer_than_full_tariff():
    persons = make_persons()
    w = persons["weight"].to_numpy()
    full = np.mean(
        [w[draw_displaced(persons, PRESETS["full_tariff_displacement"], seed=s)].sum() for s in range(50)]
    )
    epd = np.mean(
        [w[draw_displaced(persons, PRESETS["epd_displacement"], seed=s)].sum() for s in range(50)]
    )
    assert epd < full


def test_build_shocked_simulation_requires_policyengine():
    """Smoke-guard: the FRS pipeline is exercised only when policyengine-uk
    and the (licensed, gitignored) FRS h5 are available."""
    pytest.importorskip("policyengine_uk")
    from pathlib import Path

    if not Path("data/frs_2024_25.h5").exists():
        pytest.skip("FRS dataset not downloaded (run analysis/download_data.py)")
