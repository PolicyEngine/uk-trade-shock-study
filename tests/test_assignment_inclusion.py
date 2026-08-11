"""Synthetic tests for the balanced-versus-Bernoulli inclusion diagnostic.

The script itself needs licensed FRS data, but its two pure functions do not,
so the arithmetic that the manuscript will quote is testable here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.assignment_inclusion_diagnostic import (
    inclusion_frequencies,
    summarise,
)
from uk_trade_shock_study.shocks import TradeShockScenario


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


def test_frequencies_are_bounded_and_averaged_over_draws() -> None:
    freq = inclusion_frequencies(_persons(), _scenario("bernoulli"), 20)
    assert freq.shape == (60,)
    assert np.all((freq >= 0.0) & (freq <= 1.0))
    # Every value must be a multiple of 1/n_draws: it is a count over draws.
    assert np.allclose(freq * 20, np.round(freq * 20))


def test_bernoulli_frequencies_converge_on_the_declared_probability() -> None:
    """The comparator's whole purpose: it preserves first-order probabilities."""
    from uk_trade_shock_study.shocks import _person_shock

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
        inclusion_frequencies(_persons(), _scenario("bernoulli"), 0)


def test_summarise_reports_weighted_and_maximum_deviation() -> None:
    declared = np.array([0.10, 0.10, 0.10, 0.0])
    realised = np.array([0.10, 0.20, 0.05, 0.90])
    weight = np.array([1.0, 3.0, 1.0, 99.0])
    exposed = np.array([True, True, True, False])
    out = summarise(declared, realised, weight, exposed)
    # Unexposed record is excluded despite its huge weight and deviation.
    assert out["n_exposed_records"] == 3
    assert out["max_absolute_deviation"] == pytest.approx(0.10)
    assert out["weighted_mean_absolute_deviation"] == pytest.approx(
        (0.0 * 1 + 0.10 * 3 + 0.05 * 1) / 5
    )


def test_summarise_refuses_a_zero_weight_population() -> None:
    with pytest.raises(RuntimeError, match="positive survey weight"):
        summarise(
            np.array([0.1]), np.array([0.2]), np.array([0.0]), np.array([True])
        )
