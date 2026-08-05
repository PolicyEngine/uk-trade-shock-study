import numpy as np
import pytest

from uk_trade_shock_study.shocks import _baseline_flag_values_and_rate


class _WeightedFlags:
    values = np.array([True, False, False])

    def mean(self):
        return 0.75


class _BenefitUnitSimulation:
    def calculate(self, variable, period=None, map_to=None):
        assert variable == "would_claim_uc"
        assert period == 2026
        assert map_to == "benunit"
        return _WeightedFlags()


def test_baseline_flag_rate_preserves_benefit_unit_weights():
    values, rate = _baseline_flag_values_and_rate(_BenefitUnitSimulation(), period=2026)

    np.testing.assert_array_equal(values, [True, False, False])
    assert values.dtype == np.dtype(bool)
    assert rate == pytest.approx(0.75)
    assert rate != pytest.approx(values.mean())
