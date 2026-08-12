"""Unit tests for the continuous concentration sweep (synthetic tables only).

These exercise everything in analysis/concentration_sweep.py that does not
need policyengine-uk or the licensed FRS microdata: the invariance of the
imposed aggregate loss in phi, the two clips, the equivalence of the sweep's
endpoints with the manuscript's existing diffuse and concentrated wage-cut
cells, and the concentration diagnostics.
"""

import json

import numpy as np
import pandas as pd
import pytest

import analysis.concentration_sweep as sweep
from analysis.concentration_sweep import (
    concentrated_cut_table,
    concentration_schedule,
    division_schedule,
    expected_gross_loss,
    loss_diagnostics,
    phi_grid,
    select_records,
    sweep_scenario,
)
from uk_trade_shock_study.shocks import (
    TradeShockScenario,
    _person_shock,
    apply_shocks,
    apply_wage_cut,
)

EXPOSED_DIVISIONS = [21.0, 24.0, 28.0, 29.0]
UNEXPOSED_DIVISIONS = [62.0, 86.0, np.nan]


def make_persons(n=2000, seed=1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    division = rng.choice(EXPOSED_DIVISIONS + UNEXPOSED_DIVISIONS, n)
    earnings = rng.lognormal(10, 0.6, n)
    # a fifth of records are out of work: they must never bear a loss
    earnings[rng.random(n) < 0.2] = 0.0
    return pd.DataFrame(
        {
            "age": rng.integers(18, 64, n),
            "employment_income": earnings,
            "weight": rng.uniform(100, 2000, n),
            "sic_division": division,
        }
    )


def make_uniform_persons(n=4000, seed=2) -> pd.DataFrame:
    """Constant earnings and weights: a clean Monte Carlo test population."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "age": rng.integers(18, 64, n),
            "employment_income": np.full(n, 30_000.0),
            "weight": np.full(n, 1_000.0),
            "sic_division": rng.choice(EXPOSED_DIVISIONS, n),
        }
    )


def bernoulli_scenario() -> TradeShockScenario:
    return sweep_scenario(selection_method="bernoulli")


def test_loss_fraction_times_selection_probability_is_the_sector_shock():
    """The invariance the whole design rests on, checked exactly."""
    persons = make_persons()
    shock = _person_shock(persons, bernoulli_scenario())
    for phi in (1e-4, 0.001, 0.02, 0.1, 0.5, 1.0):
        phi_effective, probability = concentration_schedule(shock, phi)
        np.testing.assert_allclose(phi_effective * probability, shock, atol=1e-15)


def test_both_clips_hold():
    persons = make_persons()
    shock = _person_shock(persons, bernoulli_scenario())
    for phi in (1e-6, 0.005, 1.0):
        phi_effective, probability = concentration_schedule(shock, phi)
        assert (phi_effective <= 1.0).all()
        assert (probability <= 1.0).all()
        assert (probability >= 0.0).all()
        # the lower clip is applied to phi, never to the probability
        assert (phi_effective >= np.minimum(shock, 1.0) - 1e-15).all()


def test_phi_outside_the_unit_interval_is_rejected():
    shock = np.array([0.01, 0.0])
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="phi"):
            concentration_schedule(shock, bad)


def test_divisions_above_phi_stay_fully_diffuse():
    """Where s_j > phi the division sits at its own diffuse endpoint."""
    persons = make_persons()
    scenario = bernoulli_scenario()
    shock = _person_shock(persons, scenario)
    exposed = shock[shock > 0]
    phi = float(np.median(exposed))
    schedule = division_schedule(persons, scenario, phi)
    assert any(entry["clipped"] for entry in schedule.values())
    for entry in schedule.values():
        if entry["sector_shock"] > phi:
            assert entry["clipped"]
            assert entry["phi_effective"] == pytest.approx(entry["sector_shock"])
            assert entry["selection_probability"] == pytest.approx(1.0)
        else:
            assert not entry["clipped"]
            assert entry["phi_effective"] == pytest.approx(phi)


def test_diffuse_endpoint_reproduces_the_broad_wage_cut():
    """phi <= min_j s_j is the manuscript's diffuse wage-cut cell."""
    persons = make_persons()
    scenario = bernoulli_scenario()
    shock = _person_shock(persons, scenario)
    grid = phi_grid(shock)
    swept = concentrated_cut_table(persons, scenario, grid[0], seed=0)
    reference = apply_wage_cut(
        persons,
        TradeShockScenario("wc", scenario.tariff_scenario, "wage_cut"),
    )
    np.testing.assert_allclose(
        swept["employment_income"].to_numpy(dtype=float),
        reference["employment_income"].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-9,
    )
    # every exposed employee bears part of the loss at the diffuse endpoint
    employed = persons["employment_income"].to_numpy(dtype=float) > 0
    assert swept["earnings_changed"].to_numpy().sum() == int((employed & (shock > 0)).sum())


def test_full_concentration_reproduces_the_concentrated_wage_cut():
    """phi = 1 is the manuscript's concentrated cell, bit for bit."""
    persons = make_persons()
    scenario = bernoulli_scenario()
    reference_scenario = TradeShockScenario(
        "cwc",
        scenario.tariff_scenario,
        "concentrated_wage_cut",
        selection_method="bernoulli",
    )
    for seed in range(3):
        swept = concentrated_cut_table(persons, scenario, 1.0, seed=seed)
        reference = apply_shocks(persons, reference_scenario, seed=seed)
        np.testing.assert_array_equal(
            swept["employment_income"].to_numpy(dtype=float),
            reference["employment_income"].to_numpy(dtype=float),
        )
        np.testing.assert_array_equal(
            swept["earnings_changed"].to_numpy(),
            reference["earnings_changed"].to_numpy(),
        )
        # and the loss-bearing workers really do lose everything
        selected = swept["earnings_changed"].to_numpy()
        assert (swept["employment_income"].to_numpy(dtype=float)[selected] == 0).all()


def test_realised_aggregate_loss_matches_the_target_in_expectation():
    persons = make_uniform_persons()
    scenario = bernoulli_scenario()
    target = expected_gross_loss(persons, scenario)
    assert target > 0
    for phi in (0.05, 0.5):
        realised = np.mean(
            [
                loss_diagnostics(
                    persons, concentrated_cut_table(persons, scenario, phi, seed=seed)
                )["gross_earnings_loss"]
                for seed in range(200)
            ]
        )
        assert realised == pytest.approx(target, rel=0.03)


def test_concentration_rises_along_the_sweep():
    """The dependent concentration measure moves monotonically in phi."""
    persons = make_persons()
    scenario = bernoulli_scenario()
    shock = _person_shock(persons, scenario)
    grid = phi_grid(shock, 5)
    effective = [
        np.mean(
            [
                loss_diagnostics(
                    persons, concentrated_cut_table(persons, scenario, phi, seed=seed)
                )["effective_loss_records"]
                for seed in range(10)
            ]
        )
        for phi in grid
    ]
    assert effective == sorted(effective, reverse=True)
    assert effective[-1] < 0.1 * effective[0]


def test_shocked_table_keeps_everyone_employed_and_carries_the_seed():
    persons = make_persons()
    scenario = bernoulli_scenario()
    table = concentrated_cut_table(persons, scenario, 0.2, seed=3)
    for column in ("displaced", "inactive", "lcwra", "reallocated"):
        assert not table[column].to_numpy().any()
    assert table.attrs["seed"] == 3
    assert table.attrs["uc_takeup"] == scenario.uc_takeup
    assert table.attrs["uc_takeup_scope"] == scenario.uc_takeup_scope
    assert table["destination_division"].isna().all()


def test_only_exposed_employees_ever_bear_a_loss():
    persons = make_persons()
    scenario = bernoulli_scenario()
    shock = _person_shock(persons, scenario)
    employed = persons["employment_income"].to_numpy(dtype=float) > 0
    eligible = employed & (shock > 0)
    for phi in (0.01, 0.3, 1.0):
        for seed in range(5):
            table = concentrated_cut_table(persons, scenario, phi, seed=seed)
            selected = table["earnings_changed"].to_numpy()
            assert not (selected & ~eligible).any()
            np.testing.assert_allclose(
                table["employment_income"].to_numpy(dtype=float)[~selected],
                persons["employment_income"].to_numpy(dtype=float)[~selected],
            )


@pytest.mark.parametrize("method", ("bernoulli", "systematic", "balanced"))
def test_every_assignment_design_hits_the_target_loss(method):
    """All three shared samplers are usable along the sweep."""
    persons = make_uniform_persons(n=400)
    scenario = sweep_scenario(selection_method=method)
    target = expected_gross_loss(persons, scenario)
    realised = np.mean(
        [
            loss_diagnostics(
                persons, concentrated_cut_table(persons, scenario, 0.2, seed=seed)
            )["gross_earnings_loss"]
            for seed in range(8)
        ]
    )
    assert realised == pytest.approx(target, rel=0.35)


def test_unknown_assignment_design_is_rejected():
    persons = make_persons(n=100)
    with pytest.raises(ValueError, match="selection_method"):
        select_records(
            persons,
            np.zeros(len(persons)),
            "convenience",
            np.random.default_rng(0),
        )


def test_sweep_runs_end_to_end_against_a_stub_microsimulation(tmp_path, monkeypatch):
    """Cover main() without policyengine-uk or the licensed FRS microdata.

    The stub replaces the microsimulation with a linear cushion: 40 per cent
    of every pound of gross earnings loss is offset, and 30 per cent of it
    lands on the Exchequer. Everything else — the grid, the per-point
    schedule, the summaries and the JSON payload — is the production path.
    """
    persons = make_persons(n=600, seed=7)
    baseline_metrics = {
        "gov_balance": 1e12,
        "hni_total": 2e12,
        "poverty_bhc": 0.17,
        "poverty_ahc": 0.22,
        "poverty_line_bhc": 15_000.0,
        "poverty_line_ahc": 12_000.0,
    }

    def fake_baseline_and_persons(dataset_path, adult_tab_path, period):
        return "dataset", "baseline", persons

    def fake_build(dataset, baseline, shocked_table, period):
        return shocked_table

    def fake_metrics(sim, period, poverty_lines=None):
        if isinstance(sim, str):
            return dict(baseline_metrics)
        gross = loss_diagnostics(persons, sim)["gross_earnings_loss"]
        assert poverty_lines == (
            baseline_metrics["poverty_line_bhc"],
            baseline_metrics["poverty_line_ahc"],
        )
        return {
            **baseline_metrics,
            "gov_balance": baseline_metrics["gov_balance"] - 0.3 * gross,
            "hni_total": baseline_metrics["hni_total"] - 0.6 * gross,
        }

    monkeypatch.setattr(sweep, "_baseline_and_persons", fake_baseline_and_persons)
    monkeypatch.setattr(sweep, "build_shocked_simulation", fake_build)
    monkeypatch.setattr(sweep, "_metrics", fake_metrics)

    output = tmp_path / "concentration_sweep.json"
    sweep.main(
        [
            "--output", str(output),
            "--phi-points", "4",
            "--n-seeds", "3",
            "--selection-method", "bernoulli",
        ]
    )
    payload = json.loads(output.read_text())

    design = payload["design"]
    assert design["seeds"] == [0, 1, 2]
    assert len(design["phi_grid"]) == 4
    assert design["phi_grid"][-1] == 1.0
    assert design["expected_gross_earnings_loss"] > 0

    points = payload["points"]
    assert len(points) == 4
    for point in points:
        assert len(point["draws"]) == 3
        # the stub cushions a fixed 40% of whatever gross loss is imposed
        assert point["summary"]["cushioning_rate"]["mean"] == pytest.approx(0.4)
        assert point["summary"]["exchequer_cost"]["mean"] == pytest.approx(
            0.3 * point["summary"]["gross_earnings_loss"]["mean"]
        )
        # the imposed aggregate is held fixed along the sweep
        assert point["realised_over_expected_loss"] == pytest.approx(1.0, rel=0.4)
        assert point["n_divisions"] == 4
    # the first point is fully diffuse; the last is fully concentrated
    assert points[0]["n_divisions_clipped"] == 3
    assert points[-1]["n_divisions_clipped"] == 0
    effective = [p["summary"]["effective_loss_records"]["mean"] for p in points]
    assert effective == sorted(effective, reverse=True)


def test_phi_grid_spans_diffuse_to_fully_concentrated():
    persons = make_persons()
    shock = _person_shock(persons, bernoulli_scenario())
    grid = phi_grid(shock, 9)
    assert len(grid) == 9
    assert grid[0] == pytest.approx(float(shock[shock > 0].min()))
    assert grid[-1] == 1.0
    assert grid == sorted(grid)
    with pytest.raises(ValueError, match="at least"):
        phi_grid(shock, 1)
    with pytest.raises(ValueError, match="no exposed records"):
        phi_grid(np.zeros(10))
