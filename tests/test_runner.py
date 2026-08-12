import json

import numpy as np
import pytest

from uk_trade_shock_study.runner import (
    MonteCarloResult,
    _finite_mean_sd,
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


def test_finite_draw_summary_ignores_undefined_cushioning():
    mean, standard_deviation, count = _finite_mean_sd(
        [0.3, np.nan, 0.5, np.inf]
    )
    assert mean == pytest.approx(0.4)
    assert standard_deviation == pytest.approx(np.sqrt(0.02))
    assert count == 2


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


def test_every_artifact_writer_shares_one_nonfinite_sanitiser():
    """Referee R1: results/lfs_selection_sensitivity.json shipped literal NaN.

    `run_lfs_selection_sensitivity.py` json.dumps'd `asdict(result)` raw while
    its siblings sanitised, so the fix is one shared helper rather than a
    fourth private copy. Assert the sharing, not just the behaviour: a local
    re-implementation is exactly how the three copies drifted apart.
    """
    import ast
    from pathlib import Path

    from uk_trade_shock_study.runner import json_value

    for script in (
        "analysis/run_lfs_selection_sensitivity.py",
        "analysis/run_leave_one_sector_out.py",
        "analysis/scenario_testing.py",
    ):
        source = Path(script).read_text()
        tree = ast.parse(source)
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and node.module == "uk_trade_shock_study.runner"
            for alias in node.names
        }
        assert "json_value" in imported, (
            f"{script} must import runner.json_value rather than redefine a "
            "non-finite sanitiser of its own."
        )
        assert "allow_nan=False" in source, (
            f"{script} must dump with allow_nan=False so a missed branch fails "
            "at write time instead of shipping a non-RFC 8259 artifact."
        )

    assert json_value({"a": [float("nan"), float("inf"), 1.0]}) == {
        "a": [None, None, 1.0]
    }
