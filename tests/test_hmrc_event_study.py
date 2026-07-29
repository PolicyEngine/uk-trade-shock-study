import numpy as np
import pandas as pd

from analysis.hmrc_destination_event_study import (
    build_gap_panel,
    fit_post_model,
    previous_month,
)


def test_build_gap_panel_excludes_hmrc_aggregate_rows_and_balances_zeros():
    raw = pd.DataFrame(
        {
            "month": [202501, 202501, 202502],
            "country_code": ["US", "CA", "US"],
            "commodity_id": [87032110, 87032110, 87],
            "value_gbp": [100.0, 50.0, 1_000.0],
        }
    )
    panel, balanced = build_gap_panel(raw)
    assert set(panel["hs4"]) == {"8703"}
    assert len(balanced) == 4
    assert balanced.loc[balanced["country_code"].eq("JP"), "value_gbp"].iloc[0] == 0


def test_post_model_recovers_known_destination_gap():
    products = [f"{value:04d}" for value in range(100, 150)]
    months = pd.period_range("2022-01", "2024-12", freq="M")
    rows = []
    rng = np.random.default_rng(0)
    for product_index, product in enumerate(products):
        product_effect = product_index / 100
        for period in months:
            month = int(period.strftime("%Y%m"))
            post = month >= 202401
            rows.append(
                {
                    "hs4": product,
                    "month": month,
                    "gap": product_effect + 0.01 * period.month - 0.2 * post
                    + rng.normal(0, 0.01),
                }
            )
    panel = pd.DataFrame(rows)
    weights = pd.Series(1.0, index=products)
    result = fit_post_model(panel, weights, 202401, 202312)
    assert np.isclose(result["log_effect"], -0.2, atol=0.02)


def test_previous_month_handles_year_boundary():
    assert previous_month(202501) == 202412
