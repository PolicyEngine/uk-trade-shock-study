"""LFS-to-FRS labour-transition imputation with public aggregate constraints.

This module follows the pattern used elsewhere in PolicyEngine UK data:
estimate relationships from a donor microdataset, then align aggregate mass to
an authoritative public target. It never labels imputed LFS outcomes as ASHE
observations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


WEEKS_IN_YEAR = 365.25 / 7
LFS_COLUMNS = {
    "weight": "LGWT22",
    "gender": "SEX",
    "age_1": "AGE1",
    "age_5": "AGE5",
    "status_1": "ILODEFR1",
    "status_5": "ILODEFR5",
    "sic_1": "INDD07M1",
    "sic_5": "INDD07M5",
    "weekly_pay_1": "GRSSWK1",
    "weekly_pay_5": "GRSSWK5",
}


def _case_insensitive_columns(
    table: pd.DataFrame, requested: dict[str, str]
) -> dict[str, str]:
    available = {column.upper(): column for column in table.columns}
    missing = [source for source in requested.values() if source.upper() not in available]
    if missing:
        raise KeyError(f"LFS input is missing required columns: {missing}")
    return {name: available[source.upper()] for name, source in requested.items()}


def prepare_lfs_transitions(
    raw: pd.DataFrame,
    columns: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Create clean wave-1 to wave-5 outcomes from five-quarter LFS data.

    Job exit is defined only for respondents employed at wave 1. Wage change
    is defined only for respondents employed with valid positive main-job pay
    at both endpoints. Negative UKDS missing-value codes are converted to NaN.
    """
    requested = dict(columns or LFS_COLUMNS)
    if columns is None and requested["weight"].upper() not in {
        column.upper() for column in raw.columns
    }:
        weight_columns = [
            column for column in raw.columns if column.upper().startswith("LGWT")
        ]
        if len(weight_columns) != 1:
            raise KeyError(
                "Could not identify a unique longitudinal LFS weight column; "
                f"found {weight_columns}."
            )
        requested["weight"] = weight_columns[0]
    mapping = _case_insensitive_columns(raw, requested)
    data = pd.DataFrame(
        {name: pd.to_numeric(raw[source], errors="coerce") for name, source in mapping.items()}
    )
    for column in data:
        data[column] = data[column].where(data[column] >= 0)
    data = data[(data["status_1"] == 1) & data["weight"].gt(0)].copy()
    data["age_band"] = pd.cut(
        data["age_1"],
        [15, 24, 34, 44, 54, 64, np.inf],
        labels=["16-24", "25-34", "35-44", "45-54", "55-64", "65+"],
    )
    data["job_exit"] = data["status_5"].map({1.0: 0.0, 2.0: 1.0, 3.0: 1.0})
    valid_pay = (
        data["status_5"].eq(1)
        & data["weekly_pay_1"].gt(0)
        & data["weekly_pay_5"].gt(0)
    )
    data["log_wage_change"] = np.nan
    data.loc[valid_pay, "log_wage_change"] = np.log(
        data.loc[valid_pay, "weekly_pay_5"]
        / data.loc[valid_pay, "weekly_pay_1"]
    )
    return data


def bres_sector_targets(bres: pd.DataFrame, year: int = 2024) -> pd.Series:
    """Extract Great Britain manufacturing employee counts by SIC division."""
    table = bres.copy()
    table.columns = table.columns.str.lower()
    selected = table[
        table["date"].eq(year)
        & table["geography_name"].eq("Great Britain")
        & table["employment_status_name"].eq("Employees")
    ].copy()
    selected["sic_division"] = (
        selected["industry_name"].str.extract(r"^(\d{2})")[0].astype(int)
    )
    if selected["sic_division"].duplicated().any():
        raise ValueError("BRES target has duplicate SIC divisions.")
    return selected.set_index("sic_division")["obs_value"].astype(float)


def align_lfs_to_bres(
    lfs: pd.DataFrame,
    targets: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scale LFS weights within sector to public BRES employment totals.

    The returned diagnostics expose unsupported target sectors and scaling
    factors. Scaling cannot repair sparse within-sector demographic cells.
    """
    result = lfs[lfs["sic_1"].isin(targets.index)].copy()
    donor_totals = result.groupby("sic_1", observed=True)["weight"].sum()
    factors = targets.div(donor_totals.reindex(targets.index))
    result["aggregate_calibration_factor"] = result["sic_1"].map(factors)
    result["calibrated_weight"] = (
        result["weight"] * result["aggregate_calibration_factor"]
    )
    diagnostics = pd.DataFrame(
        {
            "bres_target": targets,
            "lfs_weighted_total": donor_totals.reindex(targets.index),
            "calibration_factor": factors,
        }
    )
    diagnostics["supported"] = diagnostics["lfs_weighted_total"].gt(0)
    return result, diagnostics


def _weighted_moments(values: pd.Series, weights: pd.Series) -> tuple[float, float]:
    valid = values.notna() & weights.gt(0)
    if not valid.any():
        return np.nan, np.nan
    x, w = values[valid].to_numpy(), weights[valid].to_numpy()
    mean = np.average(x, weights=w)
    variance = np.average((x - mean) ** 2, weights=w)
    return float(mean), float(np.sqrt(variance))


def calibrate_probabilities(
    probabilities: np.ndarray,
    target: float,
    treated: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Rescale receiver probabilities to a survey-weighted donor target."""
    result = np.zeros(len(probabilities), dtype=float)
    valid = (
        np.asarray(treated, dtype=bool)
        & np.isfinite(probabilities)
        & np.isfinite(weights)
        & (np.asarray(weights) > 0)
    )
    if not valid.any() or not np.isfinite(target) or not 0 <= target <= 1:
        return result
    raw = np.clip(np.asarray(probabilities, dtype=float), 0, 1)
    mean = np.average(raw[valid], weights=np.asarray(weights)[valid])
    if mean <= 0:
        result[valid] = target
        return result
    result[valid] = raw[valid]
    for _ in range(100):
        current = np.average(result[valid], weights=np.asarray(weights)[valid])
        if np.isclose(current, target, atol=1e-10, rtol=0):
            break
        if current <= 0:
            result[valid] = target
            break
        result[valid] = np.clip(result[valid] * target / current, 0, 1)
    current = np.average(result[valid], weights=np.asarray(weights)[valid])
    if not np.isclose(current, target, atol=1e-8):
        # Multiplicative scaling cannot raise structural zeros once all
        # positive scores have clipped at one. Fall back to an odds-intercept
        # calibration, retaining score order while allowing every observation
        # non-zero support.
        score = np.clip(raw[valid], 1e-9, 1 - 1e-9)
        logit = np.log(score / (1 - score))
        low, high = -30.0, 30.0
        valid_weights = np.asarray(weights)[valid]
        for _ in range(200):
            midpoint = (low + high) / 2
            shifted = 1 / (1 + np.exp(-(logit + midpoint)))
            if np.average(shifted, weights=valid_weights) < target:
                low = midpoint
            else:
                high = midpoint
        result[valid] = 1 / (1 + np.exp(-(logit + (low + high) / 2)))
    return result


def banded_job_exit_probabilities(
    donor_income: np.ndarray,
    donor_exit: np.ndarray,
    donor_weights: np.ndarray,
    target: float,
    receiver_income: np.ndarray,
    treated: np.ndarray,
    receiver_weights: np.ndarray,
) -> np.ndarray:
    """Income-tercile sensitivity imputation calibrated to the LFS exit rate."""
    donor_income = np.asarray(donor_income, dtype=float)
    donor_exit = np.asarray(donor_exit, dtype=float)
    donor_weights = np.asarray(donor_weights, dtype=float)
    usable = (
        np.isfinite(donor_income)
        & np.isfinite(donor_exit)
        & np.isfinite(donor_weights)
        & (donor_income > 0)
        & (donor_weights > 0)
    )
    if usable.sum() < 3:
        return calibrate_probabilities(
            np.ones(len(receiver_income)), target, treated, receiver_weights
        )
    cuts = np.quantile(donor_income[usable], [1 / 3, 2 / 3])
    donor_band = np.digitize(donor_income[usable], cuts, right=True)
    rates = np.empty(3)
    for band in range(3):
        in_band = donor_band == band
        rates[band] = (
            np.average(
                donor_exit[usable][in_band],
                weights=donor_weights[usable][in_band],
            )
            if in_band.any()
            else target
        )
    receiver_band = np.digitize(
        np.asarray(receiver_income, dtype=float), cuts, right=True
    )
    raw = rates[np.clip(receiver_band, 0, 2)]
    return calibrate_probabilities(raw, target, treated, receiver_weights)


def calibrate_receiver_transitions(
    imputed: pd.DataFrame,
    lfs: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Calibrate FRS transition levels and add a banded sensitivity estimate."""
    result = imputed.copy()
    target_exit, _ = _weighted_moments(lfs["job_exit"], lfs["calibrated_weight"])
    target_wage, _ = _weighted_moments(
        lfs["log_wage_change"], lfs["calibrated_weight"]
    )
    treated = result["job_exit_probability"].notna().to_numpy()
    receiver_weights = result.get(
        "weight", pd.Series(1.0, index=result.index)
    ).to_numpy(dtype=float)
    result["job_exit_probability"] = calibrate_probabilities(
        result["job_exit_probability"].fillna(0).to_numpy(dtype=float),
        target_exit,
        treated,
        receiver_weights,
    )
    result.loc[~treated, "job_exit_probability"] = np.nan
    result["job_exit_probability_banded"] = banded_job_exit_probabilities(
        lfs["weekly_pay_1"].to_numpy(dtype=float) * WEEKS_IN_YEAR,
        lfs["job_exit"].to_numpy(dtype=float),
        lfs["calibrated_weight"].to_numpy(dtype=float),
        target_exit,
        result["employment_income"].to_numpy(dtype=float),
        treated,
        receiver_weights,
    )
    result.loc[~treated, "job_exit_probability_banded"] = np.nan
    wage_valid = result["log_wage_change_mean"].notna().to_numpy()
    if wage_valid.any() and np.isfinite(target_wage):
        receiver_wage = np.average(
            result.loc[wage_valid, "log_wage_change_mean"],
            weights=receiver_weights[wage_valid],
        )
        result.loc[wage_valid, "log_wage_change_mean"] += target_wage - receiver_wage
    diagnostics = {
        "lfs_job_exit_target": target_exit,
        "lfs_log_wage_change_target": target_wage,
        "frs_job_exit_mean": float(
            np.average(
                result.loc[treated, "job_exit_probability"],
                weights=receiver_weights[treated],
            )
        ),
        "frs_banded_job_exit_mean": float(
            np.average(
                result.loc[treated, "job_exit_probability_banded"],
                weights=receiver_weights[treated],
            )
        ),
    }
    return result, diagnostics


def transition_cells(
    lfs: pd.DataFrame,
    shrinkage_weight: float = 50.0,
) -> pd.DataFrame:
    """Estimate hierarchical outcomes with manufacturing and sector priors."""
    rows = []
    global_exit, _ = _weighted_moments(
        lfs["job_exit"], lfs["calibrated_weight"]
    )
    global_wage, global_wage_sd = _weighted_moments(
        lfs["log_wage_change"], lfs["calibrated_weight"]
    )
    for sic, sector in lfs.groupby("sic_1", observed=True):
        sw = sector["calibrated_weight"]
        raw_sector_exit, _ = _weighted_moments(sector["job_exit"], sw)
        raw_sector_wage, raw_sector_wage_sd = _weighted_moments(
            sector["log_wage_change"], sw
        )
        sector_n = len(sector)
        sector_credibility = sector_n / (sector_n + shrinkage_weight)
        sector_exit = (
            sector_credibility * raw_sector_exit
            + (1 - sector_credibility) * global_exit
        )
        sector_wage = (
            sector_credibility * raw_sector_wage
            + (1 - sector_credibility) * global_wage
            if np.isfinite(raw_sector_wage)
            else global_wage
        )
        sector_wage_sd = (
            sector_credibility * raw_sector_wage_sd
            + (1 - sector_credibility) * global_wage_sd
            if np.isfinite(raw_sector_wage_sd)
            else global_wage_sd
        )
        rows.append(
            {
                "sic_division": int(sic),
                "gender": pd.NA,
                "age_band": "__sector__",
                "cell_level": "sector_fallback",
                "donor_count": sector_n,
                "credibility": sector_credibility,
                "job_exit_probability": sector_exit,
                "log_wage_change_mean": sector_wage,
                "log_wage_change_sd": sector_wage_sd,
            }
        )
        for (gender, age_band), cell in sector.groupby(
            ["gender", "age_band"], observed=True
        ):
            cw = cell["calibrated_weight"]
            exit_mean, _ = _weighted_moments(cell["job_exit"], cw)
            wage_mean, wage_sd = _weighted_moments(cell["log_wage_change"], cw)
            n = len(cell)
            credibility = n / (n + shrinkage_weight)
            rows.append(
                {
                    "sic_division": int(sic),
                    "gender": int(gender),
                    "age_band": str(age_band),
                    "cell_level": "sector_gender_age",
                    "donor_count": n,
                    "credibility": credibility,
                    "job_exit_probability": (
                        credibility * exit_mean + (1 - credibility) * sector_exit
                    ),
                    "log_wage_change_mean": (
                        credibility * wage_mean + (1 - credibility) * sector_wage
                        if np.isfinite(wage_mean)
                        else sector_wage
                    ),
                    "log_wage_change_sd": (
                        credibility * wage_sd
                        + (1 - credibility) * sector_wage_sd
                        if np.isfinite(wage_sd)
                        else sector_wage_sd
                    ),
                }
            )
    return pd.DataFrame(rows)


def impute_frs_transition_parameters(
    frs: pd.DataFrame,
    cells: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach LFS-estimated parameters to FRS people and report match coverage."""
    result = frs.copy()
    result["age_band"] = pd.cut(
        result["age"],
        [15, 24, 34, 44, 54, 64, np.inf],
        labels=["16-24", "25-34", "35-44", "45-54", "55-64", "65+"],
    ).astype("string")
    keys = ["sic_division", "gender", "age_band"]
    exact_cells = cells[cells["cell_level"].eq("sector_gender_age")].copy()
    sector_cells = cells[cells["cell_level"].eq("sector_fallback")].copy()
    if exact_cells.duplicated(keys).any():
        raise ValueError("LFS transition cells are not unique at matching grain.")
    before = len(result)
    result = result.merge(exact_cells, how="left", on=keys, validate="many_to_one")
    if len(result) != before:
        raise RuntimeError("LFS-to-FRS join changed the number of person rows.")
    exact_match = result["job_exit_probability"].notna()
    fallback = sector_cells.set_index("sic_division")
    parameter_columns = [
        "donor_count",
        "credibility",
        "job_exit_probability",
        "log_wage_change_mean",
        "log_wage_change_sd",
    ]
    for column in parameter_columns:
        result.loc[~exact_match, column] = result.loc[
            ~exact_match, "sic_division"
        ].map(fallback[column])
    result.loc[~exact_match & result["job_exit_probability"].notna(), "cell_level"] = (
        "sector_fallback"
    )
    matched = result["job_exit_probability"].notna()
    weights = result.get("weight", pd.Series(1.0, index=result.index)).astype(float)
    diagnostics = pd.DataFrame(
        {
            "metric": [
                "exact_person_match_rate",
                "exact_weighted_match_rate",
                "person_match_rate_with_sector_fallback",
                "weighted_match_rate_with_sector_fallback",
            ],
            "value": [
                float(exact_match.mean()),
                float(np.average(exact_match, weights=weights)),
                float(matched.mean()),
                float(np.average(matched, weights=weights)),
            ],
        }
    )
    return result, diagnostics
