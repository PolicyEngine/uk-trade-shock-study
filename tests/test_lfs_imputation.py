import numpy as np
import pandas as pd

from uk_trade_shock_study.lfs_imputation import (
    align_lfs_to_bres,
    banded_job_exit_probabilities,
    calibrate_probabilities,
    calibrate_receiver_transitions,
    impute_frs_transition_parameters,
    prepare_lfs_transitions,
    transition_cells,
)


def raw_lfs():
    return pd.DataFrame(
        {
            "LGWT22": [2, 1, 1, 1],
            "SEX": [1, 2, 1, 2],
            "AGE1": [30, 31, 46, 47],
            "AGE5": [31, 32, 47, 48],
            "ILODEFR1": [1, 1, 1, 2],
            "ILODEFR5": [1, 2, 1, 1],
            "INDD07M1": [29, 29, 30, 29],
            "INDD07M5": [29, -9, 30, 29],
            "GRSSWK1": [500, 400, 600, -9],
            "GRSSWK5": [550, -9, 600, 300],
        }
    )


def test_prepare_lfs_transitions_defines_population_and_outcomes():
    result = prepare_lfs_transitions(raw_lfs())
    assert len(result) == 3
    assert result["job_exit"].tolist() == [0, 1, 0]
    assert np.isclose(result.iloc[0]["log_wage_change"], np.log(1.1))
    assert np.isnan(result.iloc[1]["log_wage_change"])


def test_prepare_lfs_transitions_detects_weight_vintage():
    raw = raw_lfs().rename(columns={"LGWT22": "LGWT24"})
    result = prepare_lfs_transitions(raw)
    assert result["weight"].sum() == 4


def test_missing_wave5_status_is_not_counted_as_retained_employment():
    raw = raw_lfs()
    raw.loc[0, "ILODEFR5"] = -8
    result = prepare_lfs_transitions(raw)
    assert np.isnan(result.iloc[0]["job_exit"])


def test_bres_alignment_hits_supported_sector_targets():
    lfs = prepare_lfs_transitions(raw_lfs())
    result, diagnostics = align_lfs_to_bres(
        lfs, pd.Series({29: 300.0, 30: 100.0})
    )
    totals = result.groupby("sic_1")["calibrated_weight"].sum()
    assert totals.to_dict() == {29: 300.0, 30: 100.0}
    assert diagnostics["supported"].all()


def test_transition_cells_join_many_to_one_without_row_growth():
    lfs, _ = align_lfs_to_bres(
        prepare_lfs_transitions(raw_lfs()),
        pd.Series({29: 300.0, 30: 100.0}),
    )
    cells = transition_cells(lfs, shrinkage_weight=1)
    frs = pd.DataFrame(
        {
            "sic_division": [29, 30, 99],
            "gender": [1, 1, 2],
            "age": [30, 46, 40],
            "weight": [10, 20, 5],
        }
    )
    result, diagnostics = impute_frs_transition_parameters(frs, cells)
    assert len(result) == len(frs)
    assert result["job_exit_probability"].notna().sum() == 2
    assert (
        diagnostics.set_index("metric").loc[
            "person_match_rate_with_sector_fallback", "value"
        ]
        == 2 / 3
    )


def test_probability_calibration_hits_weighted_target():
    result = calibrate_probabilities(
        np.array([0.1, 0.2, 0.4, 0.9]),
        0.25,
        np.array([True, True, True, False]),
        np.array([3.0, 1.0, 2.0, 100.0]),
    )
    assert result[3] == 0
    assert np.isclose(np.average(result[:3], weights=[3, 1, 2]), 0.25)


def test_probability_calibration_handles_hard_sparse_predictions():
    result = calibrate_probabilities(
        np.array([0, 0, 0, 1], float),
        0.25,
        np.ones(4, dtype=bool),
        np.ones(4),
    )
    assert np.isclose(result.mean(), 0.25)
    assert np.all((result >= 0) & (result <= 1))


def test_banded_exit_estimator_preserves_gradient_and_target():
    result = banded_job_exit_probabilities(
        np.array([10, 11, 12, 20, 21, 22, 30, 31, 32], float),
        np.array([1, 1, 0, 1, 0, 0, 0, 0, 0], float),
        np.ones(9),
        0.2,
        np.array([11, 21, 31, 11], float),
        np.array([True, True, True, False]),
        np.ones(4),
    )
    assert result[0] > result[1] > result[2]
    assert result[3] == 0
    assert np.isclose(result[:3].mean(), 0.2)


def test_receiver_calibration_matches_direct_lfs_levels():
    lfs, _ = align_lfs_to_bres(
        prepare_lfs_transitions(raw_lfs()),
        pd.Series({29: 300.0, 30: 100.0}),
    )
    cells = transition_cells(lfs, shrinkage_weight=1)
    frs = pd.DataFrame(
        {
            "sic_division": [29, 30],
            "gender": [1, 1],
            "age": [30, 46],
            "employment_income": [20_000, 40_000],
            "weight": [10, 30],
        }
    )
    imputed, _ = impute_frs_transition_parameters(frs, cells)
    _, diagnostics = calibrate_receiver_transitions(imputed, lfs)
    assert np.isclose(
        diagnostics["frs_job_exit_mean"], diagnostics["lfs_job_exit_target"]
    )
    assert np.isclose(
        diagnostics["frs_banded_job_exit_mean"],
        diagnostics["lfs_job_exit_target"],
    )
