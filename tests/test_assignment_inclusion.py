"""Synthetic tests for the balanced-versus-Bernoulli inclusion diagnostic.

The script itself needs licensed FRS data, but its pure functions do not, so
the arithmetic that the manuscript will quote is testable here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.assignment_inclusion_diagnostic import (
    DATASET,
    OUTPUT,
    ROOT,
    bernoulli_noise_floor,
    binomial_absolute_deviation,
    common_seed_contrast,
    inclusion_draws,
    inclusion_frequencies,
    summarise,
)
from uk_trade_shock_study.shocks import TradeShockScenario, _person_shock


def _persons(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "person_id": np.arange(n),
            "age": rng.integers(25, 60, n),
            "employment_income": np.linspace(20_000, 60_000, n),
            "weight": np.linspace(100.0, 2_000.0, n),
            "sic_division": np.where(np.arange(n) % 2 == 0, 29.0, 24.0),
        }
    )


def _scenario(method: str) -> TradeShockScenario:
    return TradeShockScenario(
        "inclusion_test",
        "full_tariff",
        "displacement",
        elasticity=1.0,
        duration_equivalent=1.0,
        selection_method=method,
    )


def _fixture(method: str, n_draws: int):
    persons = _persons()
    scenario = _scenario(method)
    declared = np.asarray(_person_shock(persons, scenario), dtype=float)
    weight = persons["weight"].to_numpy(dtype=float)
    exposed = (declared > 0) & (persons["employment_income"].to_numpy() > 0)
    assert exposed.any()
    draws = inclusion_draws(persons, scenario, n_draws)
    return declared, draws.mean(axis=0), weight, exposed, draws


def test_frequencies_are_bounded_and_averaged_over_draws() -> None:
    freq = inclusion_frequencies(_persons(), _scenario("bernoulli"), 20)
    assert freq.shape == (60,)
    assert np.all((freq >= 0.0) & (freq <= 1.0))
    # Every value must be a multiple of 1/n_draws: it is a count over draws.
    assert np.allclose(freq * 20, np.round(freq * 20))


def test_draws_are_indicators_that_average_to_the_frequencies() -> None:
    persons, scenario = _persons(), _scenario("balanced")
    draws = inclusion_draws(persons, scenario, 12)
    assert draws.shape == (12, len(persons))
    assert draws.dtype == bool
    np.testing.assert_array_equal(
        draws.mean(axis=0), inclusion_frequencies(persons, scenario, 12)
    )


def test_bernoulli_frequencies_converge_on_the_declared_probability() -> None:
    """The comparator's whole purpose: it preserves first-order probabilities."""
    persons = _persons()
    scenario = _scenario("bernoulli")
    declared = np.asarray(_person_shock(persons, scenario), dtype=float)
    few = inclusion_frequencies(persons, scenario, 25)
    many = inclusion_frequencies(persons, scenario, 400)
    exposed = declared > 0
    assert exposed.any()
    err_few = np.abs(few - declared)[exposed].mean()
    err_many = np.abs(many - declared)[exposed].mean()
    assert err_many < err_few


def test_zero_draws_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        inclusion_draws(_persons(), _scenario("bernoulli"), 0)


def test_summarise_reports_signed_and_absolute_deviations() -> None:
    declared = np.array([0.10, 0.10, 0.10, 0.0])
    realised = np.array([0.10, 0.20, 0.05, 0.90])
    weight = np.array([1.0, 3.0, 1.0, 99.0])
    exposed = np.array([True, True, True, False])
    out = summarise(declared, realised, weight, exposed, n_draws=20)
    # Unexposed record is excluded despite its huge weight and deviation.
    assert out["n_exposed_records"] == 3
    assert out["max_absolute_deviation"] == pytest.approx(0.10)
    assert out["weighted_mean_absolute_deviation"] == pytest.approx(
        (0.0 * 1 + 0.10 * 3 + 0.05 * 1) / 5
    )
    # The signed statistic keeps the sign, so the over- and under-selection
    # partly cancel instead of accumulating.
    assert out["weighted_mean_signed_deviation"] == pytest.approx(
        (0.0 * 1 + 0.10 * 3 - 0.05 * 1) / 5
    )


def test_summarise_recovers_a_known_systematic_drift() -> None:
    """A pure drift with no noise must come back exactly, sign and all."""
    declared = np.full(50, 0.20)
    weight = np.linspace(1.0, 10.0, 50)
    exposed = np.ones(50, dtype=bool)
    for drift in (-0.05, 0.03):
        out = summarise(declared, declared + drift, weight, exposed, n_draws=500)
        assert out["weighted_mean_signed_deviation"] == pytest.approx(drift)
        assert out["weighted_mean_absolute_deviation"] == pytest.approx(abs(drift))


def test_signed_deviation_has_no_noise_floor_but_absolute_does() -> None:
    """The defect this diagnostic was rewritten for.

    The Bernoulli comparator has ZERO systematic drift by construction, so a
    statistic that measures drift must report ~0 at every draw count. The
    signed deviation does. The absolute deviation cannot: it sits on a noise
    floor that is large relative to the declared probabilities and shrinks
    only like 1/sqrt(n_draws), so the old absolute-only summary reported a
    number that looked like drift at any finite draw count.
    """
    small = summarise(*_fixture("bernoulli", 25)[:4], n_draws=25, draws=None)
    large = summarise(*_fixture("bernoulli", 400)[:4], n_draws=400, draws=None)
    for out in (small, large):
        # Signed: indistinguishable from zero against its own standard error.
        assert abs(out["weighted_mean_signed_deviation_t"]) < 3.0
        # Absolute: nowhere near zero, and dominated by its noise floor.
        assert out["weighted_mean_absolute_deviation"] > 0.05 * out[
            "mean_declared_probability"
        ]
        assert abs(
            out["weighted_mean_absolute_deviation_excess_over_noise_floor"]
        ) < 0.6 * out["bernoulli_absolute_deviation_noise_floor"]
    # The absolute statistic is dominated by 1/sqrt(n) noise, not by drift.
    assert (
        small["weighted_mean_absolute_deviation"]
        > 2.0 * large["weighted_mean_absolute_deviation"]
    )
    assert (
        small["bernoulli_absolute_deviation_noise_floor"]
        > 2.0 * large["bernoulli_absolute_deviation_noise_floor"]
    )


def test_balanced_drift_persists_while_the_comparator_stays_at_zero() -> None:
    """The contrast the manuscript wants to make, made measurable.

    The balanced design's signed deviation is a systematic quantity: it does
    not shrink as draws accumulate, and its t-statistic GROWS. The absolute
    deviation cannot show this, because at small draw counts both designs sit
    on the same noise floor.
    """
    balanced = [
        summarise(*_fixture("balanced", n)[:4], n_draws=n, draws=_fixture("balanced", n)[4])
        for n in (25, 400)
    ]
    bernoulli = [
        summarise(
            *_fixture("bernoulli", n)[:4], n_draws=n, draws=_fixture("bernoulli", n)[4]
        )
        for n in (25, 400)
    ]
    drifts = [out["weighted_mean_signed_deviation"] for out in balanced]
    # Same order of magnitude at 25 and at 400 draws: this is not noise.
    assert min(drifts) > 0 and max(drifts) < 4 * min(drifts)
    assert abs(balanced[1]["weighted_mean_signed_deviation_t"]) > abs(
        balanced[0]["weighted_mean_signed_deviation_t"]
    )
    assert abs(balanced[1]["weighted_mean_signed_deviation_t"]) > 3.0
    # The comparator stays indistinguishable from zero at both draw counts.
    for out in bernoulli:
        assert abs(out["weighted_mean_signed_deviation_t"]) < 3.0
    # ... whereas at 25 draws the two designs' ABSOLUTE deviations are the
    # same size, which is exactly why the old summary could not tell them
    # apart at the draw counts this diagnostic is run at.
    assert balanced[0]["weighted_mean_absolute_deviation"] < 2.0 * bernoulli[0][
        "weighted_mean_absolute_deviation"
    ]


def test_standard_error_across_draws_is_used_when_the_draws_are_supplied() -> None:
    declared, realised, weight, exposed, draws = _fixture("balanced", 60)
    empirical = summarise(declared, realised, weight, exposed, 60, draws=draws)
    analytic = summarise(declared, realised, weight, exposed, 60)
    assert empirical["weighted_mean_signed_deviation_standard_error_basis"] == (
        "across draws"
    )
    assert analytic["weighted_mean_signed_deviation_standard_error_basis"] == (
        "independent Bernoulli"
    )
    # Same point estimate either way; only the uncertainty differs, and the
    # across-draws version carries the dependence the balancing induces.
    assert empirical["weighted_mean_signed_deviation"] == pytest.approx(
        analytic["weighted_mean_signed_deviation"]
    )
    assert empirical["weighted_mean_signed_deviation_standard_error"] > 0
    with pytest.raises(ValueError, match="draw count"):
        summarise(declared, realised, weight, exposed, 59, draws=draws)


def test_binomial_absolute_deviation_matches_simulation() -> None:
    """The analytic noise floor must be the real one, not an approximation."""
    rng = np.random.default_rng(11)
    for n_draws in (7, 30, 200):
        for p in (0.001, 0.0089, 0.05, 0.4):
            simulated = np.abs(rng.binomial(n_draws, p, 400_000) / n_draws - p).mean()
            exact = float(binomial_absolute_deviation(np.array([p]), n_draws)[0])
            assert exact == pytest.approx(simulated, abs=4e-4, rel=0.02)
    assert binomial_absolute_deviation(np.array([0.0]), 10)[0] == 0.0


def test_noise_floor_is_the_realised_bernoulli_deviation() -> None:
    """The floor reported for a design is what that design actually delivers."""
    declared, realised, weight, exposed, _ = _fixture("bernoulli", 120)
    floor = bernoulli_noise_floor(declared, weight, exposed, 120)
    observed = summarise(declared, realised, weight, exposed, 120)
    assert observed["weighted_mean_absolute_deviation"] == pytest.approx(
        floor["absolute"], rel=0.5
    )
    assert floor["signed_standard_error"] > 0


def test_common_seed_contrast_differences_the_designs_draw_by_draw() -> None:
    declared, _, weight, exposed, balanced = _fixture("balanced", 80)
    bernoulli = inclusion_draws(_persons(), _scenario("bernoulli"), 80)
    contrast = common_seed_contrast(
        declared, weight, exposed, {"balanced": balanced, "bernoulli": bernoulli}
    )
    signed = {
        design: summarise(
            declared, matrix.mean(axis=0), weight, exposed, 80, draws=matrix
        )["weighted_mean_signed_deviation"]
        for design, matrix in (("balanced", balanced), ("bernoulli", bernoulli))
    }
    # The paired mean IS the difference of the two designs' signed deviations;
    # what pairing buys is the standard error, not a different point estimate.
    assert contrast["paired_mean_signed_deviation_difference"] == pytest.approx(
        signed["balanced"] - signed["bernoulli"]
    )
    assert contrast["n_draws"] == 80
    assert contrast["paired_standard_error"] > 0
    assert contrast["unpaired_standard_error"] > 0


def test_summarise_refuses_a_zero_weight_population() -> None:
    with pytest.raises(RuntimeError, match="positive survey weight"):
        summarise(
            np.array([0.1]), np.array([0.2]), np.array([0.0]), np.array([True]), 10
        )


def test_default_paths_are_anchored_on_the_repository_root() -> None:
    """House convention: analysis scripts must not depend on the caller's cwd."""
    assert DATASET.is_absolute() and OUTPUT.is_absolute()
    assert DATASET.parent == ROOT / "data"
    assert OUTPUT.parent == ROOT / "results"
