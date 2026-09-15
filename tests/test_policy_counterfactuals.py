import numpy as np
import pytest

from uk_trade_shock_study.policy_counterfactuals import targeted_transfer, wage_insurance_payment


def test_wage_insurance_scales_by_rate_and_duration():
    np.testing.assert_allclose(wage_insurance_payment([1000, 2000], 0.5, 6), [250, 500])


def test_targeted_transfer_is_thresholded():
    np.testing.assert_allclose(targeted_transfer([10, 20, 30], 20, 100), [100, 0, 0])


def test_invalid_policy_parameters_rejected():
    with pytest.raises(ValueError):
        wage_insurance_payment([100], 1.1)
