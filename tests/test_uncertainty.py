import numpy as np
import pytest

from uk_trade_shock_study.uncertainty import latin_hypercube


def test_latin_hypercube_respects_bounds_and_seed():
    a = latin_hypercube({"elasticity": (0.4, 2.0), "incidence": (0.5, 1.0)}, 100, 7)
    b = latin_hypercube({"elasticity": (0.4, 2.0), "incidence": (0.5, 1.0)}, 100, 7)
    assert a.equals(b)
    assert a["elasticity"].between(0.4, 2.0).all()
    assert a["incidence"].between(0.5, 1.0).all()
    assert np.isclose(a["elasticity"].mean(), 1.2, atol=0.03)


def test_invalid_bounds_rejected():
    with pytest.raises(ValueError):
        latin_hypercube({"x": (2.0, 1.0)}, 10)
