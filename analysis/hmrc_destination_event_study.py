"""Product-destination benchmark for UK exports around the 2025 US tariffs.

The model compares each HS4 product's exports to the United States with the
same product's mean log exports to Canada, Japan and Australia:

    gap_pt = product FE + month-of-year FE + trend + beta * post_t + error_pt

It is a public-data reduced-form benchmark, not a structural tariff model.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/public/hmrc_exports_product_destination_201801_202606.csv.gz"
OUT_MONTHLY = ROOT / "results/hmrc_destination_event_study_monthly.csv"
OUT_DIAGNOSTICS = ROOT / "results/hmrc_destination_event_study.json"
OUT_FIGURE = ROOT / "results/figures/hmrc_destination_event_study.png"
DESTINATIONS = ["US", "CA", "JP", "AU"]
POLICY_MONTH = 202505
ANTICIPATION_MONTH = 202504


def build_gap_panel(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate detailed commodities to a balanced HS4-destination panel."""
    data = raw[raw["country_code"].isin(DESTINATIONS)].copy()
    numeric_code = data["commodity_id"].astype("int64").astype(str)
    # HMRC also returns a small set of 2-digit aggregate rows. Seven-digit
    # values are valid HS8 codes whose leading zero was lost on JSON parsing.
    data = data[numeric_code.str.len().ge(7)].copy()
    data["hs4"] = data["commodity_id"].astype("int64").astype(str).str.zfill(8).str[:4]
    aggregated = data.groupby(
        ["hs4", "country_code", "month"], as_index=False
    )["value_gbp"].sum()
    index = pd.MultiIndex.from_product(
        [
            sorted(aggregated["hs4"].unique()),
            DESTINATIONS,
            sorted(aggregated["month"].unique()),
        ],
        names=["hs4", "country_code", "month"],
    )
    balanced = (
        aggregated.set_index(["hs4", "country_code", "month"])
        .reindex(index, fill_value=0)
        .reset_index()
    )
    balanced["log_value"] = np.log1p(balanced["value_gbp"])
    logwide = balanced.pivot(
        index=["hs4", "month"], columns="country_code", values="log_value"
    ).reset_index()
    logwide["gap"] = logwide["US"] - logwide[["CA", "JP", "AU"]].mean(axis=1)
    for destination in ["CA", "JP", "AU"]:
        logwide[f"gap_{destination.lower()}"] = (
            logwide["US"] - logwide[destination]
        )
    return logwide[
        ["hs4", "month", "gap", "gap_ca", "gap_jp", "gap_au"]
    ], balanced


def product_weights(
    balanced: pd.DataFrame,
    start: int,
    end: int,
    cap_quantile: float | None = 0.99,
) -> pd.Series:
    selected = balanced[
        balanced["country_code"].eq("US")
        & balanced["month"].between(start, end)
    ]
    weights = selected.groupby("hs4")["value_gbp"].mean()
    weights = weights[weights.gt(0)]
    if cap_quantile is not None:
        weights = weights.clip(upper=weights.quantile(cap_quantile))
    return weights


def fit_post_model(
    panel: pd.DataFrame,
    weights: pd.Series,
    policy_month: int,
    anticipation_month: int,
    end_month: int | None = None,
    outcome: str = "gap",
    start_month: int | None = None,
) -> dict[str, float]:
    data = panel[panel["hs4"].isin(weights.index)].copy()
    data = data[data["month"].ne(anticipation_month)]
    if end_month is not None:
        data = data[data["month"].le(end_month)]
    if start_month is not None:
        data = data[data["month"].ge(start_month)]
    data["weight"] = data["hs4"].map(weights)
    date = pd.to_datetime(data["month"].astype(str), format="%Y%m")
    data["trend"] = (date.dt.year - date.dt.year.min()) * 12 + date.dt.month - 1
    data["post"] = data["month"].ge(policy_month).astype(float)
    data["month_of_year"] = date.dt.month

    for column in [outcome, "trend", "post"]:
        data[column] -= data.groupby("hs4")[column].transform("mean")
    seasonal = pd.get_dummies(
        data["month_of_year"], prefix="month", drop_first=True, dtype=float
    )
    seasonal -= seasonal.groupby(data["hs4"]).transform("mean")
    design = pd.concat([data[["trend", "post"]], seasonal], axis=1)
    result = sm.WLS(data[outcome], design, weights=data["weight"]).fit(
        cov_type="cluster", cov_kwds={"groups": data["hs4"]}
    )
    beta = float(result.params["post"])
    standard_error = float(result.bse["post"])
    return {
        "log_effect": beta,
        "standard_error": standard_error,
        "p_value": float(result.pvalues["post"]),
        "ci_lower": beta - 1.96 * standard_error,
        "ci_upper": beta + 1.96 * standard_error,
        "proportional_effect": float(np.expm1(beta)),
        "product_count": int(data["hs4"].nunique()),
        "row_count": len(data),
        "effective_product_count": float(
            weights.sum() ** 2 / weights.pow(2).sum()
        ),
        "differential_linear_trend": float(result.params["trend"]),
        "differential_linear_trend_standard_error": float(result.bse["trend"]),
        "differential_linear_trend_p_value": float(result.pvalues["trend"]),
    }


def monthly_gap_series(
    panel: pd.DataFrame,
    weights: pd.Series,
) -> pd.DataFrame:
    data = panel[panel["hs4"].isin(weights.index)].copy()
    data["weight"] = data["hs4"].map(weights)

    def summarise(group: pd.DataFrame) -> pd.Series:
        mean = np.average(group["gap"], weights=group["weight"])
        variance = np.average(
            (group["gap"] - mean) ** 2, weights=group["weight"]
        )
        effective_n = group["weight"].sum() ** 2 / group["weight"].pow(2).sum()
        return pd.Series(
            {"weighted_gap": mean, "standard_error": np.sqrt(variance / effective_n)}
        )

    result = data.groupby("month").apply(summarise, include_groups=False).reset_index()
    baseline = result.loc[
        result["month"].isin([202501, 202502, 202503]), "weighted_gap"
    ].mean()
    result["normalised_gap"] = result["weighted_gap"] - baseline
    return result


def previous_month(month: int) -> int:
    return int((pd.Period(str(month), freq="M") - 1).strftime("%Y%m"))


def main() -> None:
    raw = pd.read_csv(INPUT)
    panel, balanced = build_gap_panel(raw)
    weights = product_weights(balanced, 202201, 202412)
    equal_weights = pd.Series(1.0, index=weights.index)
    uncapped_weights = product_weights(balanced, 202201, 202412, None)

    primary = fit_post_model(
        panel, weights, POLICY_MONTH, ANTICIPATION_MONTH
    )
    unweighted = fit_post_model(
        panel, equal_weights, POLICY_MONTH, ANTICIPATION_MONTH
    )
    uncapped = fit_post_model(
        panel, uncapped_weights, POLICY_MONTH, ANTICIPATION_MONTH
    )
    individual_controls = {
        destination: fit_post_model(
            panel,
            weights,
            POLICY_MONTH,
            ANTICIPATION_MONTH,
            outcome=f"gap_{destination.lower()}",
        )
        for destination in ["CA", "JP", "AU"]
    }
    sample_start_sensitivity = {
        str(start): fit_post_model(
            panel,
            weights,
            POLICY_MONTH,
            ANTICIPATION_MONTH,
            start_month=start,
        )
        for start in [202001, 202201, 202301]
    }
    high_tariff = weights.index.str[:2].isin(["72", "73", "87"])
    tariff_intensity_groups = {
        "steel_and_auto_chapters": fit_post_model(
            panel,
            weights[high_tariff],
            POLICY_MONTH,
            ANTICIPATION_MONTH,
        ),
        "other_products": fit_post_model(
            panel,
            weights[~high_tariff],
            POLICY_MONTH,
            ANTICIPATION_MONTH,
        ),
    }

    placebos = {}
    for year in range(2019, 2024):
        cutoff = year * 100 + 5
        end = (year + 1) * 100 + 5
        start = max(201801, (year - 3) * 100 + 1)
        placebo_weights = product_weights(
            balanced, start, previous_month(cutoff)
        )
        placebos[str(year)] = fit_post_model(
            panel,
            placebo_weights,
            cutoff,
            previous_month(cutoff),
            end_month=end,
        )

    monthly = monthly_gap_series(panel, weights)
    OUT_MONTHLY.parent.mkdir(parents=True, exist_ok=True)
    OUT_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(OUT_MONTHLY, index=False)

    date = pd.to_datetime(monthly["month"].astype(str), format="%Y%m")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(date, monthly["normalised_gap"], color="#2C6496", linewidth=2)
    ax.axvline(pd.Timestamp("2025-04-01"), color="#B50D0D", linestyle="--")
    ax.axhline(0, color="#777777", linewidth=0.8)
    ax.set(
        title="UK exports to the US relative to three comparison destinations",
        ylabel="Weighted log-export gap relative to Jan–Mar 2025",
        xlabel="",
    )
    fig.tight_layout()
    fig.savefig(OUT_FIGURE, dpi=200)
    plt.close(fig)

    diagnostics = {
        "source": str(INPUT.relative_to(ROOT)),
        "observed_period": [
            int(balanced["month"].min()),
            int(balanced["month"].max()),
        ],
        "policy_month": POLICY_MONTH,
        "anticipation_month_omitted": ANTICIPATION_MONTH,
        "comparison_destinations": ["Canada", "Japan", "Australia"],
        "outcome": (
            "log(1 + monthly export value) to US minus mean for same HS4 "
            "product to comparison destinations"
        ),
        "specification": (
            "HS4 product fixed effects, month-of-year effects, linear trend; "
            "WLS with 2022-24 pre-policy US values capped at P99; "
            "standard errors clustered by HS4"
        ),
        "primary_capped_value_weighted": primary,
        "uncapped_value_weighted": uncapped,
        "equal_product_weighted": unweighted,
        "individual_control_destinations": individual_controls,
        "sample_start_sensitivity": sample_start_sensitivity,
        "tariff_intensity_groups": tariff_intensity_groups,
        "placebo_may_cutoffs": placebos,
        "interpretation": (
            "Descriptive destination-differential evidence. Identification "
            "still requires the parallel-trends assumption and may be "
            "confounded by destination-specific demand, exchange rates, "
            "shipping, composition and policy changes."
        ),
    }
    OUT_DIAGNOSTICS.write_text(json.dumps(diagnostics, indent=2) + "\n")
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
