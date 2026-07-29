"""Benchmark calibrated QRF against the primary LFS-to-FRS cell imputation.

The QRF supplies person-level shape. Its aggregate level is calibrated to the
same direct, BRES-composition-adjusted LFS targets as the primary estimator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from microimpute.comparisons import autoimpute
from microimpute.models import QRF

from uk_trade_shock_study.lfs_imputation import (
    WEEKS_IN_YEAR,
    align_lfs_to_bres,
    bres_sector_targets,
    calibrate_probabilities,
    prepare_lfs_transitions,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BRES = ROOT / "data/public/bres_manufacturing_employment_2015_2024.csv"
DEFAULT_FRS = ROOT / "results/lfs_imputed_frs_transition_parameters.csv.gz"
DEFAULT_OUT = ROOT / "results/lfs_qrf_benchmark.csv.gz"


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    valid = values.notna() & weights.gt(0)
    return float(np.average(values[valid], weights=weights[valid]))


def qrf_impute(
    donors: pd.DataFrame,
    receivers: pd.DataFrame,
    outcome: str,
) -> np.ndarray:
    predictors = ["age", "gender", "employment_income"]
    result = autoimpute(
        donor_data=donors[predictors + [outcome, "weight"]].dropna(),
        receiver_data=receivers[predictors].copy(),
        predictors=predictors,
        imputed_variables=[outcome],
        weight_col="weight",
        models=[QRF],
        hyperparameters={
            "QRF": {
                "rfc": {"n_estimators": 500, "min_samples_leaf": 20},
                "qrf": {"n_estimators": 500, "min_samples_leaf": 20},
            }
        },
        random_state=0,
        k_folds=5,
    )
    # ``receiver_data`` contains one stochastic median-quantile draw. For a
    # benchmark score use the fitted QRF conditional mean instead; for a
    # binary numeric target this is a continuous estimated event probability.
    fitted = result.fitted_models["best_method"].models[outcome]
    feature_names = (
        fitted.classifier.feature_names_in_
        if hasattr(fitted, "classifier")
        else fitted.qrf.feature_names_in_
    )
    encoded = receivers[predictors].copy()
    if "gender_2" in feature_names:
        encoded["gender_2"] = (encoded["gender"] == 2).astype(float)
        encoded = encoded.drop(columns="gender")
    encoded = encoded[list(feature_names)]
    if hasattr(fitted, "classifier"):
        classes = fitted.classifier.classes_
        positive = int(np.flatnonzero(classes == 1)[0])
        return fitted.classifier.predict_proba(encoded)[:, positive]
    return np.asarray(
        fitted.qrf.predict(encoded, quantiles="mean"), dtype=float
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lfs-tab", type=Path, required=True)
    parser.add_argument("--bres", type=Path, default=DEFAULT_BRES)
    parser.add_argument("--frs-imputed", type=Path, default=DEFAULT_FRS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    lfs = prepare_lfs_transitions(
        pd.read_csv(args.lfs_tab, sep="\t", low_memory=False)
    )
    targets = bres_sector_targets(pd.read_csv(args.bres))
    manufacturing, _ = align_lfs_to_bres(lfs, targets)
    frs = pd.read_csv(args.frs_imputed)
    receivers = frs.rename(columns={"age": "age", "gender": "gender"})[
        ["age", "gender", "employment_income", "weight"]
    ].copy()

    # Use all employed adults for stable demographic/earnings shape, while
    # manufacturing and BRES determine the aggregate target level.
    exit_donors = lfs.rename(
        columns={
            "age_1": "age",
            "weekly_pay_1": "weekly_pay",
        }
    ).copy()
    exit_donors["employment_income"] = exit_donors["weekly_pay"] * WEEKS_IN_YEAR
    raw_exit = qrf_impute(exit_donors, receivers, "job_exit")
    exit_target = weighted_mean(
        manufacturing["job_exit"], manufacturing["calibrated_weight"]
    )
    qrf_exit = calibrate_probabilities(
        raw_exit,
        exit_target,
        np.ones(len(receivers), dtype=bool),
        receivers["weight"].to_numpy(),
    )

    wage_donors = exit_donors.dropna(subset=["log_wage_change"]).copy()
    wage_bounds = wage_donors["log_wage_change"].quantile([0.01, 0.99])
    wage_donors["log_wage_change"] = wage_donors["log_wage_change"].clip(
        wage_bounds.iloc[0], wage_bounds.iloc[1]
    )
    raw_wage = qrf_impute(wage_donors, receivers, "log_wage_change")
    wage_target = weighted_mean(
        manufacturing["log_wage_change"], manufacturing["calibrated_weight"]
    )
    raw_wage_mean = np.average(raw_wage, weights=receivers["weight"])
    qrf_wage = raw_wage + wage_target - raw_wage_mean

    output = frs[
        [
            "person_id",
            "weight",
            "sic_division",
            "employment_income",
            "job_exit_probability",
            "job_exit_probability_banded",
            "log_wage_change_mean",
        ]
    ].copy()
    output["qrf_job_exit_raw"] = raw_exit
    output["qrf_job_exit_calibrated"] = qrf_exit
    output["qrf_log_wage_change_raw"] = raw_wage
    output["qrf_log_wage_change_calibrated"] = qrf_wage
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)

    valid = output["job_exit_probability"].notna()
    w = output.loc[valid, "weight"]
    diagnostics = {
        "method": "QRF benchmark calibrated to the primary model's direct LFS targets",
        "lfs_all_employed_donors": len(lfs),
        "lfs_exit_training_donors": int(exit_donors["job_exit"].notna().sum()),
        "lfs_wage_training_donors": len(wage_donors),
        "manufacturing_target_donors": len(manufacturing),
        "frs_receivers": len(receivers),
        "job_exit_target": exit_target,
        "qrf_job_exit_weighted_mean": float(
            np.average(output["qrf_job_exit_calibrated"], weights=output["weight"])
        ),
        "wage_change_target": wage_target,
        "qrf_wage_change_weighted_mean": float(
            np.average(
                output["qrf_log_wage_change_calibrated"], weights=output["weight"]
            )
        ),
        "cell_qrf_exit_correlation": float(
            np.corrcoef(
                output.loc[valid, "job_exit_probability"],
                output.loc[valid, "qrf_job_exit_calibrated"],
            )[0, 1]
        ),
        "cell_qrf_wage_correlation": float(
            np.corrcoef(
                output.loc[valid, "log_wage_change_mean"],
                output.loc[valid, "qrf_log_wage_change_calibrated"],
            )[0, 1]
        ),
        "qrf_exit_range": [
            float(output["qrf_job_exit_calibrated"].min()),
            float(output["qrf_job_exit_calibrated"].max()),
        ],
        "qrf_wage_change_range": [
            float(output["qrf_log_wage_change_calibrated"].min()),
            float(output["qrf_log_wage_change_calibrated"].max()),
        ],
        "wage_training_winsorisation": {
            "lower_quantile": 0.01,
            "upper_quantile": 0.99,
            "lower_value": float(wage_bounds.iloc[0]),
            "upper_value": float(wage_bounds.iloc[1]),
        },
        "caveat": (
            "QRF estimates predictive heterogeneity, not a causal tariff effect; "
            "aggregate levels are imposed by LFS calibration."
        ),
    }
    diagnostics_path = args.output.with_suffix("").with_suffix(".diagnostics.json")
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2) + "\n")
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
