import json

import numpy as np
import pytest

from uk_trade_shock_study.runner import (
    MonteCarloResult,
    _household_income_per_person,
    write_result,
)


def test_write_result_serializes_nonfinite_optional_values_as_null(tmp_path):
    result = MonteCarloResult(
        scenario="test",
        n_draws=1,
        exchequer_cost_mean=0.0,
        exchequer_cost_sd=0.0,
        poverty_rate_change_bhc_mean=0.0,
        poverty_rate_change_bhc_sd=0.0,
        gini_change_mean=0.0,
        gini_change_sd=0.0,
        displaced_weighted_mean=0.0,
        cushioning_rate_mean=float("nan"),
    )
    path = tmp_path / "result.json"

    write_result(result, path)

    assert json.loads(path.read_text())["cushioning_rate_mean"] is None
    assert "NaN" not in path.read_text()


class _PersonMappedHouseholdSim:
    def __init__(self, income, people):
        self.income = np.asarray(income)
        self.people = np.asarray(people)

    def calculate(self, variable, period=None, map_to=None):
        assert map_to == "person"
        values = {
            "hbai_household_net_income": self.income,
            "household_count_people": self.people,
        }[variable]
        return type("Result", (), {"values": values})()


def test_household_income_is_divided_by_household_size_for_person_metrics():
    # The first two people share one £30k household; the third lives alone.
    sim = _PersonMappedHouseholdSim([30_000, 30_000, 20_000], [2, 2, 1])
    np.testing.assert_array_equal(
        _household_income_per_person(sim, 2026), [15_000, 15_000, 20_000]
    )


def test_household_income_per_person_rejects_nonpositive_household_size():
    sim = _PersonMappedHouseholdSim([30_000], [0])
    with pytest.raises(ValueError, match="must be positive"):
        _household_income_per_person(sim, 2026)
