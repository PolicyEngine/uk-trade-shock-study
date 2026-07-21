import json

from uk_trade_shock_study.runner import MonteCarloResult, write_result


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
