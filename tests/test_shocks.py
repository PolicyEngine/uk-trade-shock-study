"""Unit tests for the adjustment-margin mechanics (synthetic tables only)."""

import numpy as np
import pandas as pd
import pytest

from uk_trade_shock_study.exposure import DEFAULT_ELASTICITY, person_earnings_shock
from uk_trade_shock_study.shocks import (
    MARGINS,
    PRESETS,
    RENT_SHARING_ELASTICITY,
    RENT_SHARING_PRESETS,
    TradeShockScenario,
    apply_mixed_margin,
    apply_shocks,
    apply_wage_cut,
    balanced_probability_sample,
    risk_weighted_displacement_probabilities,
    draw_displaced,
    rent_sharing_displacement_share,
    systematic_probability_sample,
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


def test_bad_selection_and_duration_errors():
    with pytest.raises(ValueError, match="selection_method"):
        TradeShockScenario(
            "t", "epd", "displacement", selection_method="convenience"
        )
    with pytest.raises(ValueError, match="duration_equivalent"):
        TradeShockScenario(
            "t", "epd", "displacement", duration_equivalent=0
        )


def test_duration_equivalent_scales_wage_loss():
    persons = make_persons()
    full = apply_wage_cut(
        persons, TradeShockScenario("full", "full_tariff", "wage_cut")
    )
    half = apply_wage_cut(
        persons,
        TradeShockScenario(
            "half", "full_tariff", "wage_cut", duration_equivalent=0.5
        ),
    )
    baseline = persons["employment_income"].to_numpy(float)
    full_loss = baseline - full["employment_income"].to_numpy(float)
    half_loss = baseline - half["employment_income"].to_numpy(float)
    np.testing.assert_allclose(half_loss, 0.5 * full_loss)


def test_systematic_sampling_preserves_marginals_and_balances_group_counts():
    n = 40
    probability = np.linspace(0.02, 0.38, n)
    group = np.repeat([21, 29], n // 2)
    age = np.tile(np.arange(20, 60), 1)
    earnings = np.linspace(10_000, 100_000, n)
    hits = np.zeros(n)
    for seed in range(3_000):
        draw = systematic_probability_sample(
            probability,
            group,
            age,
            earnings,
            np.random.default_rng(seed),
        )
        hits += draw
        for value in (21, 29):
            expected = probability[group == value].sum()
            realised = draw[group == value].sum()
            assert realised in (np.floor(expected), np.ceil(expected))
    np.testing.assert_allclose(hits / 3_000, probability, atol=0.025)


def test_balanced_sampling_reduces_weighted_target_error():
    persons = make_persons(n=500)
    probabilities = person_earnings_shock(
        persons["sic_division"], "full_tariff"
    )
    contribution = (
        persons["employment_income"].to_numpy(float)
        * persons["weight"].to_numpy(float)
    )
    target = float((probabilities * contribution).sum())
    rng = np.random.default_rng(42)
    balanced = balanced_probability_sample(
        probabilities, persons, rng, n_candidates=128
    )
    balanced_error = abs(float(contribution[balanced].sum()) - target)
    systematic_errors = []
    for seed in range(128):
        draw = systematic_probability_sample(
            probabilities,
            persons["sic_division"].to_numpy(),
            persons["age"].to_numpy(float),
            persons["employment_income"].to_numpy(float),
            np.random.default_rng(seed),
        )
        systematic_errors.append(abs(float(contribution[draw].sum()) - target))
    assert balanced_error <= np.median(systematic_errors)


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


def test_risk_weighting_preserves_sector_expected_wage_bill_loss():
    persons = pd.DataFrame(
        {
            "age": [30, 40, 50, 35],
            "employment_income": [10_000.0, 40_000.0, 80_000.0, 30_000.0],
            "weight": [2.0, 1.0, 3.0, 5.0],
            "sic_division": [29.0, 29.0, 29.0, 62.0],
            "risk": [0.02, 0.10, 0.30, np.nan],
        }
    )
    shock = np.array([0.08, 0.08, 0.08, 0.0])
    probabilities = risk_weighted_displacement_probabilities(
        persons, shock, "risk"
    )
    wage_bill_weights = persons["employment_income"] * persons["weight"]
    manufacturing = persons["sic_division"].eq(29)
    assert np.average(
        probabilities[manufacturing],
        weights=wage_bill_weights[manufacturing],
    ) == pytest.approx(0.08)
    assert probabilities[0] < probabilities[1] < probabilities[2]
    assert probabilities[3] == 0


def test_displacement_expected_wage_loss_matches_sector_shock():
    """Bernoulli sampling is unbiased for the wage bill with unequal weights
    and earnings, not merely for weighted headcount."""
    persons = pd.DataFrame(
        {
            "age": [40, 40, 40],
            "employment_income": [10_000.0, 50_000.0, 120_000.0],
            "weight": [1.0, 7.0, 30.0],
            "sic_division": [29.0, 29.0, 29.0],
        }
    )
    shock = person_earnings_shock(np.array([29.0]), "full_tariff")[0]
    scenario = PRESETS["full_tariff_displacement"]
    wage_bill = (
        persons["employment_income"].to_numpy() * persons["weight"].to_numpy()
    )
    losses = [wage_bill[draw_displaced(persons, scenario, seed=s)].sum() for s in range(5_000)]
    assert np.mean(losses) == pytest.approx(shock * wage_bill.sum(), rel=0.08)


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
    """THE economic conservation property (referee point M6): the wage cut's
    deterministic aggregate earnings loss equals the MONTE CARLO MEAN of the
    displacement family's weighted earnings losses across seeds. The former
    version compared the wage-cut loss to its own defining expression and
    could never fail."""
    persons = make_persons()
    base = persons["employment_income"].to_numpy()
    w = persons["weight"].to_numpy()
    shocked = apply_wage_cut(persons, PRESETS["full_tariff_wage_cut"])
    new = shocked["employment_income"].to_numpy()
    wage_cut_loss = float(((base - new) * w).sum())
    assert wage_cut_loss > 0

    displacement_losses = []
    for seed in range(400):
        mask = draw_displaced(persons, PRESETS["full_tariff_displacement"], seed=seed)
        displacement_losses.append(float((base * w)[mask].sum()))
    assert wage_cut_loss == pytest.approx(np.mean(displacement_losses), rel=0.05)
    # no job loss, no negative incomes
    employed = base > 0
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


@pytest.mark.parametrize("share", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_mixed_margin_preserves_expected_worker_loss(share):
    """For each worker, E[loss/base] equals the original sector shock."""
    persons = make_persons(n=1200)
    scenario = TradeShockScenario(
        "mixed", "full_tariff", "mixed", displacement_share=share
    )
    base = persons["employment_income"].to_numpy(float)
    expected = person_earnings_shock(persons["sic_division"], "full_tariff")
    realised = np.zeros(len(persons))
    for seed in range(500):
        shocked = apply_mixed_margin(persons, scenario, seed=seed)
        realised += (base - shocked["employment_income"].to_numpy(float)) / base
    np.testing.assert_allclose(realised / 500, expected, atol=0.035)


def test_mixed_margin_endpoints_match_pure_margins():
    persons = make_persons()
    wage = apply_mixed_margin(
        persons,
        TradeShockScenario("mixed_wage", "epd", "mixed", displacement_share=0.0),
        seed=3,
    )
    pure_wage = apply_wage_cut(persons, PRESETS["epd_wage_cut"])
    np.testing.assert_allclose(wage["employment_income"], pure_wage["employment_income"])
    assert not wage["displaced"].any()

    job = apply_mixed_margin(
        persons,
        TradeShockScenario("mixed_job", "epd", "mixed", displacement_share=1.0),
        seed=3,
    )
    pure_job = apply_shocks(persons, PRESETS["epd_displacement"], seed=3)
    np.testing.assert_array_equal(job["displaced"], pure_job["displaced"])
    np.testing.assert_allclose(job["employment_income"], pure_job["employment_income"])


def test_rent_sharing_lambda_mapping():
    """displacement_share = 1 - rent-sharing elasticity (survivor wage cuts
    absorb exactly the elasticity share of the sector wage-bill loss)."""
    assert RENT_SHARING_ELASTICITY == pytest.approx(0.15)
    assert rent_sharing_displacement_share() == pytest.approx(0.85)
    assert rent_sharing_displacement_share(0.05) == pytest.approx(0.95)
    assert rent_sharing_displacement_share(0.0) == 1.0
    assert rent_sharing_displacement_share(1.0) == 0.0
    with pytest.raises(ValueError):
        rent_sharing_displacement_share(-0.01)
    with pytest.raises(ValueError):
        rent_sharing_displacement_share(1.01)


def test_rent_sharing_presets_are_calibrated_mixed_scenarios():
    assert set(RENT_SHARING_PRESETS) == {"full_tariff_rentsharing", "epd_rentsharing"}
    for name, scenario in RENT_SHARING_PRESETS.items():
        assert scenario.name == name
        assert scenario.margin == "mixed"
        assert scenario.displacement_share == pytest.approx(
            1.0 - RENT_SHARING_ELASTICITY
        )
    assert RENT_SHARING_PRESETS["epd_rentsharing"].tariff_scenario == "epd"
    assert (
        RENT_SHARING_PRESETS["full_tariff_rentsharing"].tariff_scenario
        == "full_tariff"
    )


def test_rent_sharing_preset_splits_wage_bill_loss_85_15():
    """Running the rentsharing preset through apply_shocks delivers a mixed-
    margin run in which, in expectation, displacement removes 85% and
    survivor wage cuts 15% of the sector wage-bill loss."""
    persons = make_persons(n=2000)
    scenario = RENT_SHARING_PRESETS["full_tariff_rentsharing"]
    base = persons["employment_income"].to_numpy(float)
    w = persons["weight"].to_numpy(float)
    shock = person_earnings_shock(persons["sic_division"], "full_tariff")
    employed = base > 0
    total_expected = float((shock * base * w)[employed].sum())

    displacement_loss, survivor_loss = 0.0, 0.0
    n_seeds = 300
    for seed in range(n_seeds):
        shocked = apply_shocks(persons, scenario, seed=seed)
        displaced = shocked["displaced"].to_numpy(bool)
        assert displaced.any()  # mixed margin with a displacement component
        new = shocked["employment_income"].to_numpy(float)
        loss = (base - new) * w
        displacement_loss += float(loss[displaced].sum())
        survivor_loss += float(loss[~displaced].sum())
    displacement_loss /= n_seeds
    survivor_loss /= n_seeds
    total = displacement_loss + survivor_loss
    assert total == pytest.approx(total_expected, rel=0.05)
    assert displacement_loss / total == pytest.approx(0.85, abs=0.02)
    assert survivor_loss / total == pytest.approx(0.15, abs=0.02)


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


# ---------------------------------------------------------------------------
# Post-shock Universal Credit take-up re-draw
# ---------------------------------------------------------------------------


class _TakeupSim:
    """Minimal sim stub: persons map to benunits in fixed-size blocks."""

    def __init__(self, n_persons, per_benunit=2, baseline_flag=None, potential_uc=None):
        self.n = n_persons
        self.k = per_benunit
        self.n_bu = (n_persons + per_benunit - 1) // per_benunit
        self.flag = (
            np.zeros(self.n_bu, dtype=bool) if baseline_flag is None else baseline_flag
        )
        self.stored = {}
        self.potential_uc = (
            np.ones(self.n_bu) if potential_uc is None else np.asarray(potential_uc)
        )

    def calculate(self, var, period=None, map_to=None):
        import types

        if var == "would_claim_uc":
            values = self.stored.get(var, self.flag)
        elif var == "employment_income":
            values = np.ones(self.n)
        elif var == "universal_credit":
            claim = self.stored.get("would_claim_uc", self.flag)
            values = self.potential_uc * np.asarray(claim)
        else:
            values = np.zeros(self.n)
        return types.SimpleNamespace(values=np.asarray(values))

    def set_input(self, var, period, values):
        self.stored[var] = np.asarray(values)

    def _invalidate_all_caches(self):
        return None

    def map_result(self, values, source, target):
        assert (source, target) == ("person", "benunit")
        out = np.zeros(self.n_bu, dtype=float)
        for i, v in enumerate(np.asarray(values, dtype=float)):
            out[i // self.k] += v
        return out


def _takeup_table(n, affected_idx, uc_takeup=0.8, seed=0):
    table = pd.DataFrame(
        {
            "employment_income": np.ones(n),
            "displaced": np.zeros(n, dtype=bool),
            "inactive": np.zeros(n, dtype=bool),
            "reallocated": np.zeros(n, dtype=bool),
        }
    )
    table.loc[list(affected_idx), "displaced"] = True
    table.attrs["uc_takeup"] = uc_takeup
    table.attrs["seed"] = seed
    return table


def test_uc_takeup_redrawn_only_for_affected_benunits():
    """Affected benunits get a fresh draw; every other benunit is untouched."""
    from uk_trade_shock_study.shocks import redraw_uc_takeup

    n = 200
    rng = np.random.default_rng(3)
    baseline = rng.random(n // 2) < 0.55
    sim = _TakeupSim(n, per_benunit=2, baseline_flag=baseline.copy())
    base_sim = _TakeupSim(n, per_benunit=2, baseline_flag=baseline.copy())
    affected_persons = [0, 5, 6, 41]
    table = _takeup_table(n, affected_persons)

    new = redraw_uc_takeup(sim, base_sim, table, 2026)
    affected_bu = np.zeros(len(baseline), dtype=bool)
    affected_bu[[i // 2 for i in affected_persons]] = True
    # unaffected benefit units keep the baseline draw exactly
    np.testing.assert_array_equal(new[~affected_bu], baseline[~affected_bu])
    # the flag actually reached the simulation
    np.testing.assert_array_equal(sim.stored["would_claim_uc"], new)


def test_uc_takeup_rate_approximately_honoured_among_affected():
    from uk_trade_shock_study.shocks import redraw_uc_takeup

    n = 4000
    baseline = np.zeros(n // 2, dtype=bool)  # baseline all False
    affected_persons = list(range(0, n, 2))  # every benunit affected
    rates = []
    for seed in range(20):
        sim = _TakeupSim(n, 2, baseline.copy())
        base_sim = _TakeupSim(n, 2, baseline.copy())
        table = _takeup_table(n, affected_persons, uc_takeup=0.8, seed=seed)
        rates.append(redraw_uc_takeup(sim, base_sim, table, 2026).mean())
    assert np.mean(rates) == pytest.approx(0.8, abs=0.02)

    rates0 = []
    for seed in range(5):
        sim = _TakeupSim(n, 2, baseline.copy())
        base_sim = _TakeupSim(n, 2, baseline.copy())
        table = _takeup_table(n, affected_persons, uc_takeup=1.0, seed=seed)
        rates0.append(redraw_uc_takeup(sim, base_sim, table, 2026).mean())
    assert min(rates0) == 1.0


def test_uc_takeup_no_redraw_when_nothing_changed():
    """Wage-cut margin: no new claimants, baseline flags survive untouched."""
    from uk_trade_shock_study.shocks import redraw_uc_takeup

    n = 100
    baseline = np.random.default_rng(1).random(n // 2) < 0.5
    sim = _TakeupSim(n, 2, baseline.copy())
    base_sim = _TakeupSim(n, 2, baseline.copy())
    table = _takeup_table(n, [])
    out = redraw_uc_takeup(sim, base_sim, table, 2026)
    np.testing.assert_array_equal(out, baseline)
    assert "would_claim_uc" not in sim.stored


def test_wage_cut_earnings_changes_are_considered_for_takeup():
    persons = make_persons()
    table = apply_shocks(persons, PRESETS["full_tariff_wage_cut"], seed=0)
    assert not np.array_equal(
        table["employment_income"].to_numpy(), persons["employment_income"].to_numpy()
    )


def test_uc_takeup_redraw_requires_positive_postshock_entitlement():
    from uk_trade_shock_study.shocks import redraw_uc_takeup

    n = 8
    baseline = np.array([False, False, True, True])
    potential = np.array([100.0, 0.0, 100.0, 0.0])
    sim = _TakeupSim(n, 2, baseline.copy(), potential_uc=potential)
    base_sim = _TakeupSim(n, 2, baseline.copy(), potential_uc=potential)
    table = _takeup_table(n, [0, 2, 4, 6], uc_takeup=1.0)
    out = redraw_uc_takeup(sim, base_sim, table, 2026)
    np.testing.assert_array_equal(out, np.array([True, False, True, True]))


def test_uc_takeup_redraw_only_newly_entitled_units():
    from uk_trade_shock_study.shocks import redraw_uc_takeup

    n = 6
    baseline = np.array([False, False, False])
    sim = _TakeupSim(n, 2, baseline.copy(), potential_uc=[100.0, 100.0, 0.0])
    base_sim = _TakeupSim(n, 2, baseline.copy(), potential_uc=[0.0, 100.0, 0.0])
    table = _takeup_table(n, [0, 2, 4], uc_takeup=1.0)
    out = redraw_uc_takeup(
        sim,
        base_sim,
        table,
        2026,
        baseline_potential_award=np.array([0.0, 100.0, 0.0]),
    )
    np.testing.assert_array_equal(out, np.array([True, False, False]))


def test_existing_lcwra_element_is_not_double_paid():
    from uk_trade_shock_study.shocks import merge_lcwra_element

    annual = 5_000.0
    base = np.array([0.0, annual, annual * 1.1])
    addon = np.array([annual, annual, annual])
    np.testing.assert_array_equal(
        merge_lcwra_element(base, addon), np.array([annual, annual, annual * 1.1])
    )


def test_displacement_draw_invariant_to_uc_takeup():
    """uc_takeup must not perturb the displacement/reallocation draw."""
    persons = make_persons()
    for margin in ("displacement", "inactivity", "reallocation"):
        a = apply_shocks(
            persons,
            TradeShockScenario("a", "full_tariff", margin, uc_takeup=0.55),
            seed=3,
        )
        b = apply_shocks(
            persons,
            TradeShockScenario("b", "full_tariff", margin, uc_takeup=1.0),
            seed=3,
        )
        for col in ("displaced", "inactive", "lcwra", "reallocated", "employment_income"):
            np.testing.assert_array_equal(a[col].to_numpy(), b[col].to_numpy())


def test_uc_takeup_scenario_validation_and_attrs():
    with pytest.raises(ValueError):
        TradeShockScenario("t", "epd", "displacement", uc_takeup=1.5)
    persons = make_persons()
    table = apply_shocks(
        persons, TradeShockScenario("t", "epd", "displacement", uc_takeup=0.7), seed=4
    )
    assert table.attrs["uc_takeup"] == 0.7
    assert table.attrs["seed"] == 4


def test_uc_takeup_hard_error_when_flag_is_dropped():
    from uk_trade_shock_study.shocks import redraw_uc_takeup

    class SilentSim(_TakeupSim):
        def set_input(self, var, period, values):
            return None

    n = 20
    sim = SilentSim(n, 2, np.zeros(n // 2, dtype=bool))
    base_sim = _TakeupSim(n, 2, np.zeros(n // 2, dtype=bool))
    table = _takeup_table(n, [0, 1, 2], uc_takeup=1.0)
    with pytest.raises(RuntimeError, match="take-up re-draw not applied"):
        redraw_uc_takeup(sim, base_sim, table, 2026)


class _CachingTakeupSim(_TakeupSim):
    """Stub that CACHES universal_credit like the real engine (referee M5).

    The first universal_credit calculation (the temporary all-claim
    entitlement pass inside redraw_uc_takeup) is cached and served until
    ``_invalidate_all_caches`` is called. ``flush_works=False`` models a
    policyengine upgrade turning the private cache-flush into a no-op.
    """

    def __init__(self, *args, flush_works=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.flush_works = flush_works
        self._uc_cache = None

    def calculate(self, var, period=None, map_to=None):
        import types

        if var == "universal_credit":
            if self._uc_cache is None:
                self._uc_cache = super().calculate(var, period, map_to).values
            return types.SimpleNamespace(values=self._uc_cache)
        return super().calculate(var, period, map_to)

    def _invalidate_all_caches(self):
        if self.flush_works:
            self._uc_cache = None


def test_uc_award_cache_flush_is_verified_after_redraw():
    """redraw_uc_takeup must FAIL HARD if _invalidate_all_caches is a no-op:
    the UC award would otherwise silently retain the temporary all-claim
    entitlement pass for units drawn to NOT claim (referee point M5)."""
    from uk_trade_shock_study.shocks import redraw_uc_takeup

    n = 40
    baseline = np.zeros(n // 2, dtype=bool)
    affected = list(range(0, n, 2))  # every benunit affected & newly entitled

    # uc_takeup=0: every redrawn unit ends NOT claiming, so any positive
    # post-redraw award can only come from the stale all-claim cache.
    working = _CachingTakeupSim(n, 2, baseline.copy(), flush_works=True)
    base_sim = _TakeupSim(n, 2, baseline.copy())
    out = redraw_uc_takeup(
        working, base_sim, _takeup_table(n, affected, uc_takeup=0.0), 2026
    )
    assert not out.any()
    # the flushed cache means the final award differs from the all-claim pass
    assert (working.calculate("universal_credit").values == 0).all()

    broken = _CachingTakeupSim(n, 2, baseline.copy(), flush_works=False)
    base_sim = _TakeupSim(n, 2, baseline.copy())
    with pytest.raises(RuntimeError, match="cache was not flushed"):
        redraw_uc_takeup(
            broken, base_sim, _takeup_table(n, affected, uc_takeup=0.0), 2026
        )


def test_uc_takeup_stream_uses_tuple_seeding():
    """The take-up stream must never collide with the displacement stream:
    seed = UC_TAKEUP_SEED_OFFSET for the displacement RNG must not reproduce
    the take-up draw of seed 0 (the failure mode of additive offsets)."""
    from uk_trade_shock_study.shocks import UC_TAKEUP_SEED_OFFSET

    a = np.random.default_rng((0, UC_TAKEUP_SEED_OFFSET)).random(1000)
    b = np.random.default_rng(UC_TAKEUP_SEED_OFFSET).random(1000)
    assert not np.allclose(a, b)


# ---------------------------------------------------------------------------
# H2: earnings-linked deductions scale with the earnings factor
# ---------------------------------------------------------------------------


class _RecordingSim:
    """Stores set_input values and serves them back; defaults otherwise."""

    def __init__(self, persons):
        self.persons = persons
        self.n = len(persons)
        self.stored = {}

    def calculate(self, var, period=None, map_to=None):
        import types

        if var in self.stored:
            return types.SimpleNamespace(values=self.stored[var])
        if var == "employment_status":
            values = np.array(["EMPLOYED"] * self.n, dtype=object)
        elif var == "employment_income":
            values = self.persons["employment_income"].to_numpy(dtype=float)
        elif var in (
            "employee_pension_contributions",
            "pension_contributions_via_salary_sacrifice",
        ):
            values = np.full(self.n, 1_000.0)
        else:
            values = np.zeros(self.n)
        return types.SimpleNamespace(values=values)

    def set_input(self, var, period, values):
        self.stored[var] = np.asarray(values)

    def _invalidate_all_caches(self):
        return None

    def map_result(self, values, source, target):
        return np.asarray(values, dtype=float)


def test_survivor_pension_contributions_scale_with_earnings_factor():
    """Wage-cut (and mixed-survivor/reallocation) workers must not keep full
    baseline pension contributions against cut earnings (referee point H2);
    displaced workers keep the zeroing behaviour."""
    from unittest import mock

    from uk_trade_shock_study import shocks as shocks_module

    persons = make_persons(n=500)
    table = apply_shocks(persons, PRESETS["full_tariff_wage_cut"], seed=0)
    base_earn = persons["employment_income"].to_numpy(dtype=float)
    new_earn = table["employment_income"].to_numpy(dtype=float)
    cut = ~np.isclose(new_earn, base_earn)
    assert cut.any()

    baseline_sim = _RecordingSim(persons)
    shocked_sim = _RecordingSim(persons)
    fake_pe = mock.MagicMock()
    fake_pe.Microsimulation.return_value = shocked_sim
    with mock.patch.dict("sys.modules", {"policyengine_uk": fake_pe}):
        shocks_module.build_shocked_simulation(None, baseline_sim, table, 2026)

    factor = np.divide(new_earn, base_earn, out=np.ones_like(new_earn), where=base_earn > 0)
    applied = shocked_sim.stored["employee_pension_contributions"]
    np.testing.assert_allclose(applied, 1_000.0 * factor, rtol=1e-9)
    assert (applied[cut] < 1_000.0).all()
    np.testing.assert_allclose(applied[~cut], 1_000.0)


def test_displaced_pension_contributions_still_zeroed():
    from unittest import mock

    from uk_trade_shock_study import shocks as shocks_module

    persons = make_persons(n=500)
    table = apply_shocks(persons, PRESETS["full_tariff_displacement"], seed=0)
    displaced = table["displaced"].to_numpy()
    assert displaced.any()

    baseline_sim = _RecordingSim(persons)
    shocked_sim = _RecordingSim(persons)
    fake_pe = mock.MagicMock()
    fake_pe.Microsimulation.return_value = shocked_sim
    with mock.patch.dict("sys.modules", {"policyengine_uk": fake_pe}):
        shocks_module.build_shocked_simulation(None, baseline_sim, table, 2026)

    applied = shocked_sim.stored["employee_pension_contributions"]
    assert (applied[displaced] == 0.0).all()
    np.testing.assert_allclose(applied[~displaced], 1_000.0)
