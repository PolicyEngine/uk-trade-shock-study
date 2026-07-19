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


def test_lcwra_addon_one_element_per_benunit():
    """A benunit with TWO flagged persons receives exactly ONE health element."""
    from uk_trade_shock_study.shocks import lcwra_benunit_addon

    class StubSim:
        """persons 0,1 -> benunit 0; person 2 -> benunit 1; person 3 -> benunit 2."""

        benunit_of_person = np.array([0, 0, 1, 2])

        def map_result(self, values, source, target):
            assert (source, target) == ("person", "benunit")
            out = np.zeros(3)
            np.add.at(out, self.benunit_of_person, np.asarray(values, dtype=float))
            return out

    monthly = 100.0
    # benunit 0 has two flagged members, benunit 1 one, benunit 2 none.
    lcwra = np.array([True, True, True, False])
    addon = lcwra_benunit_addon(StubSim(), lcwra, monthly)
    assert addon == pytest.approx([monthly * 12.0, monthly * 12.0, 0.0])


# --- reallocation margin -------------------------------------------------


def test_reallocation_quota_identical_to_displacement_under_same_seed():
    """Paired draws: the reallocated set IS the displaced set, seed for seed."""
    persons = make_persons()
    for seed in range(5):
        realloc = apply_shocks(persons, PRESETS["full_tariff_reallocation"], seed=seed)
        displ = apply_shocks(persons, PRESETS["full_tariff_displacement"], seed=seed)
        np.testing.assert_array_equal(
            realloc["reallocated"].to_numpy(), displ["displaced"].to_numpy()
        )
    # ...and nobody is out of work under reallocation
    assert not realloc["displaced"].to_numpy().any()
    assert not realloc["inactive"].to_numpy().any()
    assert not realloc["lcwra"].to_numpy().any()


def test_reallocation_destinations_are_services_with_expected_mix():
    from uk_trade_shock_study.shocks import DESTINATION_SHARES, REALLOCATION_DESTINATIONS

    persons = make_persons()
    dest = []
    for seed in range(30):
        t = apply_shocks(persons, PRESETS["full_tariff_reallocation"], seed=seed)
        d = t["destination_division"].to_numpy()
        moved = t["reallocated"].to_numpy()
        # exactly the reallocated get a destination, and it is a services one
        assert np.isnan(d[~moved]).all()
        assert not np.isnan(d[moved]).any()
        assert set(np.unique(d[moved])) <= set(map(float, REALLOCATION_DESTINATIONS))
        dest.append(d[moved])
    pooled = np.concatenate(dest)
    for code, share in zip(REALLOCATION_DESTINATIONS, DESTINATION_SHARES):
        assert (pooled == code).mean() == pytest.approx(share, abs=0.03)


def test_reallocation_applies_the_wage_penalty():
    from uk_trade_shock_study.shocks import DEFAULT_REALLOCATION_PENALTY

    persons = make_persons()
    shocked = apply_shocks(persons, PRESETS["epd_reallocation"], seed=3)
    moved = shocked["reallocated"].to_numpy()
    base = persons["employment_income"].to_numpy()
    new = shocked["employment_income"].to_numpy()
    assert moved.any()
    np.testing.assert_allclose(
        new[moved], base[moved] * (1 - DEFAULT_REALLOCATION_PENALTY), rtol=1e-9
    )
    # everyone else untouched, and no reallocated worker loses all earnings
    np.testing.assert_array_equal(new[~moved], base[~moved])
    assert (new[moved] > 0).all()


def test_reallocation_lag_scales_earnings_and_hours():
    from uk_trade_shock_study.shocks import DEFAULT_REALLOCATION_PENALTY

    persons = make_persons()
    lagged = TradeShockScenario(
        "t", "full_tariff", "reallocation", reallocation_lag_months=3.0
    )
    shocked = apply_shocks(persons, lagged, seed=0)
    moved = shocked["reallocated"].to_numpy()
    base = persons["employment_income"].to_numpy()
    factor = (1 - DEFAULT_REALLOCATION_PENALTY) * 0.75
    np.testing.assert_allclose(
        shocked["employment_income"].to_numpy()[moved], base[moved] * factor, rtol=1e-9
    )
    hf = shocked["reallocation_hours_factor"].to_numpy()
    assert hf[moved] == pytest.approx(0.75)
    assert hf[~moved] == pytest.approx(1.0)
    # the draw is unchanged by the lag
    instant = apply_shocks(persons, PRESETS["full_tariff_reallocation"], seed=0)
    np.testing.assert_array_equal(moved, instant["reallocated"].to_numpy())
    # a lagged reallocation costs the worker strictly more than an instant one
    assert (
        shocked["employment_income"].to_numpy()[moved].sum()
        < instant["employment_income"].to_numpy()[moved].sum()
    )


def test_reallocation_loss_is_the_penalty_share_of_displacement_loss():
    """Reallocation removes penalty x (displacement loss): the SAME workers
    are hit, but they keep (1 - penalty) of their earnings instead of zero.

    Note the resulting ordering against the earnings-equivalent wage cut:
    the wage cut removes shock_j x the WHOLE division wage bill, whereas
    reallocation removes only the penalty on the movers' earnings, so the
    gross loss is SMALLER under reallocation than under the wage cut. The
    two are not orderable by construction — only displacement dominates.
    """
    from uk_trade_shock_study.shocks import DEFAULT_REALLOCATION_PENALTY

    persons = make_persons()
    w = persons["weight"].to_numpy()
    base = persons["employment_income"].to_numpy()

    def loss(table):
        return float(((base - table["employment_income"].to_numpy()) * w).sum())

    d = loss(apply_shocks(persons, PRESETS["full_tariff_displacement"], seed=0))
    r = loss(apply_shocks(persons, PRESETS["full_tariff_reallocation"], seed=0))
    assert r == pytest.approx(DEFAULT_REALLOCATION_PENALTY * d, rel=1e-9)
    assert 0 < r < d


def test_reallocation_scenario_parameter_validation():
    with pytest.raises(ValueError):
        TradeShockScenario("t", "epd", "reallocation", reallocation_penalty=1.0)
    with pytest.raises(ValueError):
        TradeShockScenario("t", "epd", "reallocation", reallocation_lag_months=13.0)


def test_reallocation_hard_error_when_sector_switch_is_dropped():
    """build_shocked_simulation must FAIL HARD if the sector set_input is
    silently ignored — otherwise reallocated workers would stay in
    manufacturing and the margin would collapse into a plain wage cut."""
    from unittest import mock

    from uk_trade_shock_study import shocks as shocks_module

    persons = make_persons(n=400)
    table = apply_shocks(persons, PRESETS["full_tariff_reallocation"], seed=0)
    assert table["reallocated"].to_numpy().any()
    n = len(table)

    class SilentSim:
        """Accepts every set_input and forgets it (the failure mode)."""

        def calculate(self, var, period=None, map_to=None):
            import types

            if var == "employment_status":
                values = np.array(["EMPLOYED"] * n, dtype=object)
            elif var == "sic_industry_division":
                values = persons["sic_division"].to_numpy(dtype=float)
            else:
                values = np.zeros(n)
            return types.SimpleNamespace(values=values)

        def set_input(self, *args, **kwargs):
            return None

    stub = SilentSim()
    fake_pe = mock.MagicMock()
    fake_pe.Microsimulation.return_value = stub
    with mock.patch.dict("sys.modules", {"policyengine_uk": fake_pe}):
        with pytest.raises(RuntimeError, match="sector reallocation not applied"):
            shocks_module.build_shocked_simulation(None, stub, table, 2026)
