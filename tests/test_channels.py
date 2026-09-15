import numpy as np
import pandas as pd
import pytest

from uk_trade_shock_study.channels import calibrated_worker_probabilities, real_income_after_price_shock


def test_heterogeneity_preserves_weighted_sector_total():
    base = np.full(4, 0.25)
    x = pd.DataFrame({"age": [20, 30, 50, 60], "earnings": [1, 2, 5, 8]})
    out = calibrated_worker_probabilities(base, x, coefficients=pd.Series([1.0, -0.5]), weights=[1, 2, 1, 3])
    assert np.average(out, weights=[1, 2, 1, 3]) == pytest.approx(0.25)
    assert out[0] != out[-1]


def test_price_channel_reduces_real_income():
    shares = pd.DataFrame({"food": [0.4, 0.2], "other": [0.6, 0.8]})
    prices = pd.Series({"food": 0.10, "other": 0.0})
    out = real_income_after_price_shock([100, 100], shares, prices)
    assert out[0] == pytest.approx(100 / 1.04)
    assert out[1] == pytest.approx(100 / 1.02)


def test_invalid_probability_rejected():
    with pytest.raises(ValueError):
        calibrated_worker_probabilities([1.2], pd.DataFrame({"x": [1]}))
