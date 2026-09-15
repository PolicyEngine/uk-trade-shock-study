import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

from bootstrap_uncertainty import (  # noqa: E402
    CONTRIBUTION_KEYS,
    SCENARIOS,
    bootstrap_contrast,
    load_cached_contributions,
    support_sizes,
)


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


def _thin_denominator_contributions():
    """Second assignment draw carries an almost negligible gross loss.

    This is the configuration that makes a mean of per-draw ratios unstable:
    the thin draw's ratio counts as much as the well-populated draw's.
    """
    return {
        "displacement": {
            "gross": np.array([[100.0, 0.0, 50.0, 50.0], [1.0, 0.0, 0.0, 0.0]]),
            "net": np.array([[70.0, 0.0, 30.0, 40.0], [0.5, 0.0, 0.0, 0.0]]),
            "exch": np.array([[30.0, 0.0, 20.0, 10.0], [0.5, 0.0, 0.0, 0.0]]),
        },
        "wage_cut": {
            "gross": np.array([[100.0, 0.0, 50.0, 50.0], [100.0, 0.0, 50.0, 50.0]]),
            "net": np.array([[50.0, 0.0, 25.0, 25.0], [50.0, 0.0, 25.0, 25.0]]),
            "exch": np.array([[50.0, 0.0, 25.0, 25.0], [50.0, 0.0, 25.0, 25.0]]),
        },
    }


def test_pooled_ratio_equals_the_hand_computed_pooled_ratio():
    summary = bootstrap_contrast(_thin_denominator_contributions(), n_boot=5, seed=3)
    # Pooled over both draws: gross 200 + 1, net 140 + 0.5.
    assert summary["displacement_cushioning_pooled"]["point"] == pytest.approx(
        1.0 - 140.5 / 201.0
    )
    assert summary["wage_cut_cushioning_pooled"]["point"] == pytest.approx(
        1.0 - 200.0 / 400.0
    )
    assert summary["cushioning_difference_pooled"]["point"] == pytest.approx(
        0.5 - (1.0 - 140.5 / 201.0)
    )


def test_pooled_ratio_is_not_pulled_by_a_thin_draw():
    summary = bootstrap_contrast(_thin_denominator_contributions(), n_boot=5, seed=3)
    # Mean of per-draw ratios: (0.30 + 0.50) / 2 = 0.40, dominated by the draw
    # whose gross loss is a single pound.
    assert summary["displacement_cushioning"]["point"] == pytest.approx(0.40)
    assert summary["displacement_cushioning_pooled"]["point"] == pytest.approx(
        0.3009950, abs=1e-6
    )


def test_pooled_and_mean_of_ratios_agree_when_draws_are_balanced():
    summary = bootstrap_contrast(_contributions(), n_boot=5, seed=3)
    for key in ("displacement", "wage_cut"):
        assert summary[f"{key}_cushioning_pooled"]["point"] == pytest.approx(
            summary[f"{key}_cushioning"]["point"]
        )


def test_pooled_estimator_is_bootstrapped_too():
    summary = bootstrap_contrast(_contributions(), n_boot=200, seed=11)
    entry = summary["cushioning_difference_pooled"]
    assert entry["bootstrap_se"] > 0
    assert entry["ci_2_5"] <= entry["point"] <= entry["ci_97_5"]


def test_support_sizes_count_contributing_households():
    support = support_sizes(_contributions())
    # Household 1 never has a positive gross loss under wage_cut, and only in
    # the second displacement draw.
    assert support["displacement"]["households_positive_gross_any_draw"] == 4
    assert support["displacement"]["households_positive_gross_every_draw"] == 3
    assert support["wage_cut"]["households_positive_gross_any_draw"] == 3
    assert support["any_margin"]["households_positive_gross_any_draw"] == 4
    assert support["displacement"]["households_positive_gross_per_draw_min"] == 3
    assert support["displacement"][
        "households_positive_gross_per_draw_mean"
    ] == pytest.approx(3.5)


def test_support_sizes_flag_a_thin_draw():
    support = support_sizes(_thin_denominator_contributions())
    assert support["displacement"]["households_positive_gross_per_draw_min"] == 1
    assert support["displacement"][
        "smallest_draw_gross_share_of_mean"
    ] == pytest.approx(1.0 / 100.5)


def test_household_count_mismatch_raises():
    contributions = _contributions()
    contributions["wage_cut"]["gross"] = contributions["wage_cut"]["gross"][:, :3]
    with pytest.raises(ValueError):
        bootstrap_contrast(contributions, n_boot=5, seed=1)


# ---------------------------------------------------------------------------
# Partial-cache resume
#
# The margin loop is the expensive part: each margin runs N_DRAWS PolicyEngine
# simulations. A run interrupted between margins must be resumable from the
# cache it already wrote, so the cache loader has to load the margins the
# archive HOLDS rather than the margins the script wants.
# ---------------------------------------------------------------------------


def _write_cache(path, margins):
    arrays = {}
    for index, margin in enumerate(margins):
        for key in CONTRIBUTION_KEYS:
            arrays[f"{margin}_{key}"] = np.full((2, 4), float(index + 1))
    np.savez_compressed(path, **arrays)
    return path


def test_full_cache_loads_every_margin(tmp_path):
    cache = _write_cache(tmp_path / "contributions.npz", SCENARIOS)
    loaded = load_cached_contributions(np.load(cache))
    assert set(loaded) == set(SCENARIOS)
    for margin in SCENARIOS:
        assert set(loaded[margin]) == set(CONTRIBUTION_KEYS)
        assert loaded[margin]["gross"].shape == (2, 4)


def test_partial_cache_resumes_instead_of_raising(tmp_path):
    """Regression: a cache missing one margin used to raise a KeyError.

    Loading every margin unconditionally meant a one-margin cache blew up on
    ``wage_cut_gross is not a file in the archive`` before the resume guards
    downstream could see it, so an interrupted run lost the whole cache.
    """
    cache = _write_cache(tmp_path / "contributions.npz", ["displacement"])
    loaded = load_cached_contributions(np.load(cache))

    assert set(loaded) == {"displacement"}
    # ...and the guards the loader feeds now see a genuinely partial cache.
    assert set(loaded) != set(SCENARIOS)
    assert [m for m in SCENARIOS if m not in loaded] == ["wage_cut"]


def test_partial_cache_preserves_the_completed_margin(tmp_path):
    cache = _write_cache(tmp_path / "contributions.npz", ["wage_cut"])
    loaded = load_cached_contributions(np.load(cache))
    np.testing.assert_array_equal(loaded["wage_cut"]["net"], np.full((2, 4), 1.0))


def test_half_written_margin_is_recomputed_not_half_loaded(tmp_path):
    """A margin missing one of its three arrays is not a resumable margin."""
    cache = tmp_path / "contributions.npz"
    np.savez_compressed(
        cache,
        displacement_gross=np.ones((2, 4)),
        displacement_net=np.ones((2, 4)),
        **{f"wage_cut_{k}": np.ones((2, 4)) for k in CONTRIBUTION_KEYS},
    )
    loaded = load_cached_contributions(np.load(cache))
    assert set(loaded) == {"wage_cut"}


def test_empty_cache_loads_nothing(tmp_path):
    cache = tmp_path / "contributions.npz"
    np.savez_compressed(cache)
    assert load_cached_contributions(np.load(cache)) == {}
