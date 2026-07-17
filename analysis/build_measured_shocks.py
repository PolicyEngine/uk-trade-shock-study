"""Build the MEASURED scenario inputs from the realised 2025-26 trade outturn.

Pulls monthly UK goods exports to the United States from the HMRC uktradeinfo
OTS endpoint (the RTS endpoint's monthly coverage stops being complete after
April 2025, so unlike the 2024 intensity build we use OTS with an OData
$apply groupby throughout), January 2023 - February 2026, keyed by SITC and
mapped to SIC 2007 divisions with the same crosswalk as
analysis/build_trade_by_sic.py.

REALISED EXPORT FALL, per division j:

    fall_j = 1 - (exports in POST window) / (exports in BASELINE window)

POST window: May 2025 - February 2026 (10 months). April 2025 - the first
full tariff month - is EXCLUDED: it mixes the initial collapse with the
unwinding of the Q1-2025 front-running spike and is the month the
epsilon = 2.0 anchor was calibrated on, so excluding it also keeps the
validation out-of-sample.

BASELINE window (documented choice): for each post calendar month, the MEAN
of the same calendar month one and two years earlier (e.g. May 2025 vs the
mean of May 2024 and May 2023; February 2026 vs the mean of February 2025
and February 2024). Comparing same calendar months nets out seasonality.
Averaging two years matters because the one-year-earlier baseline is itself
contaminated by tariff anticipation: exports were pulled forward into
Q4-2024/Q1-2025 (front-running), so a pure year-on-year comparison for the
Jan-Feb 2026 post months would divide by an inflated Jan-Feb 2025 base and
overstate the fall. The one-year YoY variant is computed and stored alongside
as a sensitivity.

MEASURED EARNINGS SHOCK: s_j^measured = max(fall_j, 0) x us_export_share_j.
No elasticity and no tariff rate enter - the realised fall IS the demand
response, with the EPD (autos quota rate, steel relief, pharma exemption)
already embedded in the outturn. Divisions whose US exports ROSE over the
window take a zero shock (the displacement machinery cannot model gains;
the raw negative falls are preserved in the outputs). Pass-through of the
export fall to the sector wage bill remains 1.0, as in the calibrated
families.

Outputs:
- uk_trade_shock_study/data/measured_export_falls_by_sic.csv (packaged;
  consumed by exposure.sector_earnings_shocks("measured")).
- results/validation_sectors.json: predicted (epsilon-model, full and EPD)
  vs realised export falls for the top-10 exposed divisions + Spearman rank
  correlations.

Usage: .venv/bin/python analysis/build_measured_shocks.py
Raw API pulls cached as JSON in data/.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from build_trade_by_sic import DIVISION_NAMES, SITC2_TO_SIC, SITC3_TO_SIC, SPLIT_CHAPTERS

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT_CSV = ROOT / "uk_trade_shock_study" / "data" / "measured_export_falls_by_sic.csv"
OUT_VALIDATION = ROOT / "results" / "validation_sectors.json"

API = "https://api.uktradeinfo.com"
US, EXPORTS_NON_EU = 400, 4
FIRST_MONTH, LAST_MONTH = 202301, 202602
POST_MONTHS = [202505, 202506, 202507, 202508, 202509, 202510, 202511, 202512, 202601, 202602]

# epsilon-model primitives for the validation table (mirror exposure.py).
ELASTICITY = 2.0
TARIFF_FULL = {29: 0.25, 24: 0.25}
TARIFF_EPD = {29: 0.10, 24: 0.125, 21: 0.0}
BASELINE_TARIFF = 0.10


def _paged(url: str) -> list[dict]:
    rows = []
    while url:
        r = requests.get(url, timeout=180)
        r.raise_for_status()
        payload = r.json()
        rows += payload["value"]
        url = payload.get("@odata.nextLink")
    return rows


def _cached(name: str, url: str) -> list[dict]:
    path = DATA / name
    if path.exists():
        return json.loads(path.read_text())
    rows = _paged(url)
    path.write_text(json.dumps(rows))
    return rows


def fetch_monthly_by_division() -> pd.DataFrame:
    """Monthly UK goods exports to the US (pounds) by SIC division.

    One OTS pull, grouped server-side by (MonthId, CommoditySitcId); SITC3 =
    CommoditySitcId // 100, mapped through the SITC3 crosswalk for the split
    chapters and the SITC2 chapter crosswalk otherwise.
    """
    url = (
        f"{API}/OTS?$apply=filter(MonthId ge {FIRST_MONTH} and MonthId le {LAST_MONTH} "
        f"and CountryId eq {US} and FlowTypeId eq {EXPORTS_NON_EU})/"
        "groupby((MonthId,CommoditySitcId),aggregate(Value with sum as V))"
    )
    rows = _cached(f"ots_us_monthly_{FIRST_MONTH}_{LAST_MONTH}.json", url)
    records = []
    for row in rows:
        sitc3 = int(row["CommoditySitcId"]) // 100
        chapter = sitc3 // 10
        div = SITC3_TO_SIC.get(sitc3) if chapter in SPLIT_CHAPTERS else SITC2_TO_SIC.get(chapter)
        if div is None:
            continue
        records.append((int(row["MonthId"]), div, float(row["V"])))
    frame = pd.DataFrame(records, columns=["month", "division", "value"])
    return frame.groupby(["month", "division"], as_index=False)["value"].sum()


def _window_sum(monthly: pd.DataFrame, months: list[int]) -> pd.Series:
    sel = monthly[monthly["month"].isin(months)]
    return sel.groupby("division")["value"].sum()


def realised_falls(monthly: pd.DataFrame) -> pd.DataFrame:
    """Per-division realised export fall: post window vs the two baselines."""
    post = _window_sum(monthly, POST_MONTHS)

    def shift(m: int, years: int) -> int:
        return m - 100 * years

    base_1y = _window_sum(monthly, [shift(m, 1) for m in POST_MONTHS])
    base_2y = _window_sum(monthly, [shift(m, 2) for m in POST_MONTHS])
    base_avg = (base_1y + base_2y) / 2.0

    table = pd.DataFrame(
        {
            "post": post,
            "baseline_avg_2yr": base_avg,
            "baseline_yoy_1yr": base_1y,
        }
    ).dropna()
    table["export_fall"] = 1.0 - table["post"] / table["baseline_avg_2yr"]
    table["export_fall_yoy1"] = 1.0 - table["post"] / table["baseline_yoy_1yr"]
    table.index.name = "sic_division"
    return table


def main() -> None:
    DATA.mkdir(exist_ok=True)
    monthly = fetch_monthly_by_division()

    # sanity: 2024 calendar total must reproduce the intensity build's basis
    total_2024 = monthly[(monthly.month >= 202401) & (monthly.month <= 202412)]["value"].sum()
    print(f"2024 mapped US-export total: £{total_2024 / 1e9:.1f}bn (build_trade_by_sic: ~£52.5bn)")
    # sanity: the April 2025 aggregate fall that anchored epsilon
    apr = monthly[monthly.month == 202504]["value"].sum()
    apr_base = monthly[monthly.month.isin([202404, 202304])].groupby("month")["value"].sum().mean()
    print(f"April 2025 aggregate YoY(2yr-avg) fall: {1 - apr / apr_base:.1%} (ONS anchor: 24.7% YoY)")

    falls = realised_falls(monthly)
    agg_fall = 1.0 - falls["post"].sum() / falls["baseline_avg_2yr"].sum()
    agg_yoy = 1.0 - falls["post"].sum() / falls["baseline_yoy_1yr"].sum()
    print(f"Aggregate May25-Feb26 fall: {agg_fall:.1%} (2yr-avg baseline), {agg_yoy:.1%} (1yr YoY)")

    intensity = pd.read_csv(
        ROOT / "uk_trade_shock_study" / "data" / "us_export_intensity_by_sic.csv", comment="#"
    ).set_index("sic_division")

    lines = [
        "# Realised UK-to-US export falls by SIC 2007 division - MEASURED scenario input.",
        "# Post window: May 2025 - Feb 2026 (Apr 2025 excluded: front-running unwind +",
        "# the epsilon-calibration month). Baseline: mean of the same calendar months",
        "# one and two years earlier (2-yr average, diluting Q4-24/Q1-25 front-running",
        "# in the 1-yr base; the pure 1-yr YoY fall is kept as a sensitivity column).",
        "# Source: HMRC uktradeinfo OTS API, SITC->SIC crosswalk of",
        "# analysis/build_trade_by_sic.py; written by analysis/build_measured_shocks.py.",
        "# export_fall may be negative (US exports rose); exposure.py clips at zero.",
        "sic_division,description,export_fall,export_fall_yoy1",
    ]
    for div in sorted(set(intensity.index) & set(falls.index)):
        lines.append(
            f"{div},{DIVISION_NAMES[div]},{falls.loc[div, 'export_fall']:.4f},"
            f"{falls.loc[div, 'export_fall_yoy1']:.4f}"
        )
    OUT_CSV.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_CSV}")

    # ---- validation table: predicted (epsilon-model) vs realised falls ----
    rows = {}
    for div in intensity.index:
        tau_f = TARIFF_FULL.get(div, BASELINE_TARIFF)
        tau_e = TARIFF_EPD.get(div, BASELINE_TARIFF)
        x = float(intensity.loc[div, "us_export_share"])
        if div not in falls.index:
            continue
        rows[int(div)] = {
            "description": DIVISION_NAMES[div],
            "us_export_intensity": x,
            "predicted_fall_full": ELASTICITY * tau_f,
            "predicted_fall_epd": ELASTICITY * tau_e,
            "realised_fall": float(falls.loc[div, "export_fall"]),
            "realised_fall_yoy1": float(falls.loc[div, "export_fall_yoy1"]),
            "shock_full": ELASTICITY * tau_f * x,
            "shock_measured": max(float(falls.loc[div, "export_fall"]), 0.0) * x,
            "post_exports_gbp": float(falls.loc[div, "post"]),
            "baseline_exports_gbp": float(falls.loc[div, "baseline_avg_2yr"]),
        }
    frame = pd.DataFrame(rows).T
    top10 = frame.sort_values("shock_full", ascending=False).head(10)

    def spearman(a: pd.Series, b: pd.Series) -> float:
        return float(a.rank().corr(b.rank()))

    validation = {
        "post_window": POST_MONTHS,
        "baseline": "mean of same calendar months 1 and 2 years earlier",
        "aggregate_realised_fall": float(agg_fall),
        "aggregate_realised_fall_yoy1": float(agg_yoy),
        "aggregate_predicted_fall_full": float(
            (frame["predicted_fall_full"] * frame["baseline_exports_gbp"]).sum()
            / frame["baseline_exports_gbp"].sum()
        ),
        "aggregate_predicted_fall_epd": float(
            (frame["predicted_fall_epd"] * frame["baseline_exports_gbp"]).sum()
            / frame["baseline_exports_gbp"].sum()
        ),
        "rank_correlation_top10_epd_vs_realised": spearman(
            top10["predicted_fall_epd"], top10["realised_fall"]
        ),
        "rank_correlation_top10_full_vs_realised": spearman(
            top10["predicted_fall_full"], top10["realised_fall"]
        ),
        "rank_correlation_all_shock_full_vs_measured": spearman(
            frame["shock_full"], frame["shock_measured"]
        ),
        "top10": {int(d): top10.loc[d].to_dict() for d in top10.index},
        "all_divisions": {int(d): frame.loc[d].to_dict() for d in frame.index},
    }
    OUT_VALIDATION.parent.mkdir(exist_ok=True)
    OUT_VALIDATION.write_text(json.dumps(validation, indent=2))
    print(f"wrote {OUT_VALIDATION}")
    print(top10[["predicted_fall_full", "predicted_fall_epd", "realised_fall", "shock_measured"]])
    print(
        "Spearman top-10 (EPD pred vs realised):",
        validation["rank_correlation_top10_epd_vs_realised"],
    )


if __name__ == "__main__":
    main()
