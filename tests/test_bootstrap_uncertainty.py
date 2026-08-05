import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

from bootstrap_uncertainty import bootstrap_contrast  # noqa: E402


def _contributions():
    # 2 draws x 4 households, constructed so cushioning differs by margin.
    return {
        "displacement": {
            "gross": np.array([[100.0, 0.0, 50.0, 50.0], [80.0, 20.0, 50.0, 50.0]]),
            "net": np.array([[70.0, 0.0, 30.0, 40.0], [60.0, 10.0, 30.0, 40.0]]),
            "exch": np.array([[30.0, 0.0, 20.0, 10.0], [20.0, 10.0, 20.0, 10.0]]),
        },
        "wage_cut": {
            "gross": np.array([[100.0, 0.0, 50.0, 50.0], [100.0, 0.0, 50.0, 50.0]]),
            "net": np.array([[50.0, 0.0, 25.0, 25.0], [50.0, 0.0, 25.0, 25.0]]),
            "exch": np.array([[50.0, 0.0, 25.0, 25.0], [50.0, 0.0, 25.0, 25.0]]),
        },
    }


def test_point_estimates_average_over_draws():
    summary = bootstrap_contrast(_contributions(), n_boot=10, seed=1)
    # displacement cushioning: draw 1 -> 1-140/200=0.30, draw 2 -> 1-140/200=0.30
    assert summary["displacement_cushioning"]["point"] == pytest.approx(0.30)
    assert summary["wage_cut_cushioning"]["point"] == pytest.approx(0.50)
    assert summary["cushioning_difference"]["point"] == pytest.approx(0.20)
    assert summary["wage_cut_exchequer"]["point"] == pytest.approx(100.0)


def test_bootstrap_is_seeded_and_disperses():
    a = bootstrap_contrast(_contributions(), n_boot=200, seed=7)
    b = bootstrap_contrast(_contributions(), n_boot=200, seed=7)
    assert a["cushioning_difference"]["bootstrap_se"] == pytest.approx(
        b["cushioning_difference"]["bootstrap_se"]
    )
    assert a["cushioning_difference"]["bootstrap_se"] > 0
    assert (
        a["cushioning_difference"]["ci_2_5"]
        <= a["cushioning_difference"]["point"]
        <= a["cushioning_difference"]["ci_97_5"]
    )


def test_household_count_mismatch_raises():
    contributions = _contributions()
    contributions["wage_cut"]["gross"] = contributions["wage_cut"]["gross"][:, :3]
    with pytest.raises(ValueError):
        bootstrap_contrast(contributions, n_boot=5, seed=1)
