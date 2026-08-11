import json

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

import analysis.hmrc_destination_event_study as event_study
from analysis.hmrc_destination_event_study import (
    ANTICIPATION_LEAD_MONTHS,
    ANTICIPATION_MONTHS,
    ANTICIPATION_WINDOW,
    CONTINUITY_WINDOW_END,
    ESTIMATE_FIELDS,
    LEGACY_ANTICIPATION_MONTH,
    POLICY_MONTH,
    _backtracking_line_search,
    _ppml_irls,
    build_gap_panel,
    continuity_months,
    continuous_trade_products,
    fit_post_model,
    fit_ppml_model,
    format_month_window,
    group_contrast,
    month_range,
    monthly_gap_series,
    precision_summary,
    previous_month,
    window_before,
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


def test_month_window_helpers():
    assert month_range(202411, 202502) == [202411, 202412, 202501, 202502]
    assert window_before(202505, 4) == (202501, 202504)
    assert format_month_window(202401, 202412) == "2024"
    assert format_month_window(202501, 202504) == "Jan-Apr 2025"
    assert format_month_window(202412, 202502) == "Dec 2024-Feb 2025"


def test_anticipation_window_is_derived_from_the_policy_month_and_the_lead():
    # The primary exclusion window and the placebo windows must be the same
    # object of a different vintage: the placebos are built with
    # window_before(cutoff, ANTICIPATION_LEAD_MONTHS), so widening the lead has
    # to widen the primary window too or the symmetry is silently broken.
    assert ANTICIPATION_WINDOW == window_before(POLICY_MONTH, ANTICIPATION_LEAD_MONTHS)
    assert ANTICIPATION_MONTHS == month_range(*ANTICIPATION_WINDOW)
    assert len(ANTICIPATION_MONTHS) == ANTICIPATION_LEAD_MONTHS
    assert ANTICIPATION_MONTHS[-1] == previous_month(POLICY_MONTH)
    # A placebo cutoff excludes a window of exactly the same length.
    placebo = month_range(*window_before(202105, ANTICIPATION_LEAD_MONTHS))
    assert len(placebo) == len(ANTICIPATION_MONTHS)
    # The legacy single-month specification must be nested in the window for
    # the reported sensitivity to be like-for-like.
    assert LEGACY_ANTICIPATION_MONTH in ANTICIPATION_MONTHS
    # Continuity is judged strictly before the anticipation window opens.
    assert CONTINUITY_WINDOW_END == previous_month(ANTICIPATION_WINDOW[0])
    assert CONTINUITY_WINDOW_END < ANTICIPATION_WINDOW[0] < POLICY_MONTH


def _anticipation_panel(surge: float, post_effect: float = -0.2, seed: int = 0):
    """Gap panel with a front-running surge in the anticipation window."""
    products = [f"{value:04d}" for value in range(30)]
    months = pd.period_range("2022-01", "2025-12", freq="M")
    rng = np.random.default_rng(seed)
    rows = []
    for product_index, product in enumerate(products):
        for period in months:
            month = int(period.strftime("%Y%m"))
            gap = product_index / 100 + 0.01 * period.month
            if month in ANTICIPATION_MONTHS:
                gap += surge
            if month >= 202505:
                gap += post_effect
            rows.append(
                {"hs4": product, "month": month, "gap": gap + rng.normal(0, 0.01)}
            )
    panel = pd.DataFrame(rows)
    return panel, pd.Series(1.0, index=products)


def test_anticipation_window_is_dropped_from_the_estimation_sample():
    panel, weights = _anticipation_panel(surge=0.0)
    months = panel["month"].nunique()
    fitted = fit_post_model(panel, weights, 202505, ANTICIPATION_MONTHS)
    assert fitted["months_excluded"] == ANTICIPATION_MONTHS
    assert fitted["row_count"] == weights.size * (months - len(ANTICIPATION_MONTHS))
    # Nothing survives from the excluded window: poisoning it cannot move the fit.
    poisoned = panel.copy()
    poisoned.loc[poisoned["month"].isin(ANTICIPATION_MONTHS), "gap"] += 1_000.0
    assert fit_post_model(poisoned, weights, 202505, ANTICIPATION_MONTHS)[
        "log_effect"
    ] == pytest.approx(fitted["log_effect"])


def test_front_running_inflates_the_single_month_specification():
    panel, weights = _anticipation_panel(surge=0.5, post_effect=-0.2)
    window = fit_post_model(panel, weights, 202505, ANTICIPATION_MONTHS)
    single_month = fit_post_model(panel, weights, 202505, LEGACY_ANTICIPATION_MONTH)
    none_excluded = fit_post_model(panel, weights, 202505, None)
    assert window["log_effect"] == pytest.approx(-0.2, abs=0.02)
    # Leaving the surge in the pre-period exaggerates the post-policy decline.
    assert single_month["log_effect"] < window["log_effect"] - 0.05
    assert none_excluded["log_effect"] < window["log_effect"] - 0.05
    assert single_month["months_excluded"] == [LEGACY_ANTICIPATION_MONTH]
    assert none_excluded["months_excluded"] == []


def test_monthly_series_is_normalised_on_the_configured_base():
    panel, weights = _anticipation_panel(surge=0.5)
    monthly = monthly_gap_series(panel, weights, (202401, 202412))
    base = monthly.loc[monthly["month"].between(202401, 202412), "normalised_gap"]
    assert base.mean() == pytest.approx(0.0, abs=1e-12)
    assert monthly["base_period"].unique().tolist() == ["2024"]
    # The anticipation months are above the clean base, so normalising on them
    # instead would shift the whole series down.
    anticipation_base = monthly_gap_series(panel, weights, (202501, 202504))
    shift = (
        anticipation_base["normalised_gap"] - monthly["normalised_gap"]
    ).to_numpy()
    assert np.allclose(shift, shift[0])
    assert shift[0] < -0.4


def test_monthly_series_rejects_a_base_outside_the_sample():
    panel, weights = _anticipation_panel(surge=0.0)
    with pytest.raises(ValueError):
        monthly_gap_series(panel, weights, (201901, 201912))


def _levels_panel(zeros: bool = False, post_effect: float = -0.3, seed: int = 1):
    """Balanced product-destination-month panel of export levels."""
    rng = np.random.default_rng(seed)
    products = [f"{value:04d}" for value in range(6)]
    # At least three calendar years: with a shorter span the post dummy is
    # collinear with the trend and the month-of-year effects.
    months = [
        int(period.strftime("%Y%m"))
        for period in pd.period_range("2023-01", "2025-12", freq="M")
    ]
    rows = []
    for product_index, product in enumerate(products):
        for destination in ["US", "CA", "JP", "AU"]:
            for position, month in enumerate(months):
                us_premium = 0.4 if destination == "US" else 0.0
                level = 5.0 + 0.1 * product_index + us_premium
                linear = level + 0.002 * position
                if destination == "US" and month >= 202505:
                    linear += post_effect
                value = float(rng.poisson(np.exp(linear)))
                if zeros and destination == "JP" and product_index == 1:
                    # A comparison destination with a genuinely absent flow.
                    value = 0.0 if position % 2 else value
                rows.append(
                    {
                        "hs4": product,
                        "country_code": destination,
                        "month": month,
                        "value_gbp": value,
                    }
                )
    return pd.DataFrame(rows), products


def test_ppml_matches_statsmodels_poisson_with_explicit_dummies():
    balanced, products = _levels_panel(zeros=True)
    excluded = ANTICIPATION_MONTHS
    fitted = fit_ppml_model(balanced, products, 202505, excluded)

    data = (
        balanced[~balanced["month"].isin(excluded)]
        .sort_values(["hs4", "country_code", "month"])
        .reset_index(drop=True)
    )
    date = pd.to_datetime(data["month"].astype(str), format="%Y%m")
    trend = ((date.dt.year - date.dt.year.min()) * 12 + date.dt.month - 1).to_numpy(
        dtype=float
    )
    post = data["month"].ge(202505).to_numpy(dtype=float)
    is_us = data["country_code"].eq("US").to_numpy(dtype=float)
    seasonal = pd.get_dummies(
        date.dt.month, prefix="month", drop_first=True, dtype=float
    ).to_numpy()
    common = np.column_stack([trend, post, seasonal])
    pair = pd.get_dummies(
        data["hs4"] + "|" + data["country_code"], dtype=float
    ).to_numpy()
    design = np.column_stack([common, common * is_us[:, None], pair])
    reference = sm.GLM(
        data["value_gbp"].to_numpy(dtype=float),
        design,
        family=sm.families.Poisson(),
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": pd.factorize(data["hs4"].to_numpy())[0]},
    )
    us_post = common.shape[1] + 1
    assert fitted["converged"]
    assert not fitted["rank_deficient"]
    assert fitted["log_effect"] == pytest.approx(reference.params[us_post], rel=1e-8)
    assert fitted["standard_error"] == pytest.approx(reference.bse[us_post], rel=1e-8)
    assert fitted["zero_observation_share"] > 0


def test_ppml_recovers_the_known_levels_effect():
    balanced, products = _levels_panel(zeros=True, post_effect=-0.3)
    fitted = fit_ppml_model(balanced, products, 202505, ANTICIPATION_MONTHS)
    assert fitted["log_effect"] == pytest.approx(-0.3, abs=0.05)
    assert fitted["proportional_effect"] == pytest.approx(np.expm1(-0.3), abs=0.05)
    assert fitted["months_excluded"] == ANTICIPATION_MONTHS


def test_ppml_drops_pairs_that_never_trade():
    balanced, products = _levels_panel()
    balanced.loc[
        balanced["hs4"].eq("0003") & balanced["country_code"].eq("AU"), "value_gbp"
    ] = 0.0
    fitted = fit_ppml_model(balanced, products, 202505, ANTICIPATION_MONTHS)
    assert fitted["separated_pairs_dropped"] == 1
    assert fitted["converged"]


def _poisson_irls_inputs(seed: int = 3, scale: float = 1.0):
    """Small Poisson problem with one absorbed group dimension."""
    rng = np.random.default_rng(seed)
    group_index = np.repeat(np.arange(5), 20)
    design = rng.normal(size=(100, 2))
    mean = np.exp(1.0 + 0.5 * design[:, 0] - 0.3 * design[:, 1] + 0.2 * group_index)
    return rng.poisson(mean).astype(float) * scale, design, group_index, 5


def test_line_search_never_adopts_a_rejected_candidate():
    """An exhausted line search is a failure, not a step and not convergence."""
    beta = np.array([0.25, -0.5])
    step = np.array([1.0, 1.0])

    def always_worse(candidate: np.ndarray) -> tuple[float, str]:
        return -1e6, "candidate"

    accepted, new_beta, new_loglik, extra, damping = _backtracking_line_search(
        always_worse, beta, step, loglik=-10.0
    )
    assert not accepted
    # The caller's iterate survives untouched; nothing from the rejected
    # candidate leaks into beta, the log-likelihood or the step size.
    assert np.array_equal(new_beta, beta)
    assert new_loglik == -10.0
    assert extra is None
    assert damping == 0.0

    # A non-finite objective is a rejection too, not an acceptance.
    accepted, _, _, _, _ = _backtracking_line_search(
        lambda candidate: (np.nan, None), beta, step, loglik=-10.0
    )
    assert not accepted


def test_line_search_acceptance_tolerance_is_relative_to_the_log_likelihood():
    """The same absolute wobble is noise at 1e9 and a real decrease at 1e2."""
    beta = np.zeros(2)
    step = np.ones(2)
    decrease = 5e-4

    def objective_for(base: float):
        return lambda candidate: (base - decrease, "candidate")

    # The concentrated log-likelihood is y @ linear, so with y in pounds it is
    # of order 1e9 and a 5e-4 wobble is far below one unit in the last place.
    accepted, _, _, _, damping = _backtracking_line_search(
        objective_for(-1e9), beta, step, loglik=-1e9
    )
    assert accepted
    assert damping == 1.0
    # At 1e2 the same wobble is a genuine decrease and is damped away.
    accepted, _, _, _, _ = _backtracking_line_search(
        objective_for(-1e2), beta, step, loglik=-1e2
    )
    assert not accepted


def test_irls_reports_a_failed_line_search_and_keeps_the_last_accepted_iterate(
    monkeypatch,
):
    y, design, group_index, n_groups = _poisson_irls_inputs()
    real_search = event_study._backtracking_line_search
    accepted_betas: list[np.ndarray] = []

    def sabotaged(objective, beta, step, loglik, **kwargs):
        if accepted_betas:
            # Run the real search over an objective that never improves, so the
            # 30 halvings are genuinely exhausted rather than short-circuited.
            return real_search(
                lambda candidate: (-np.inf, None), beta, step, loglik, **kwargs
            )
        result = real_search(objective, beta, step, loglik, **kwargs)
        assert result[0]
        accepted_betas.append(np.asarray(result[1]).copy())
        return result

    monkeypatch.setattr(event_study, "_backtracking_line_search", sabotaged)
    beta, _, _, _, converged, iterations, line_search_failed = _ppml_irls(
        y, design, group_index, n_groups
    )
    assert line_search_failed
    # The old code reported convergence here, because the damping left over
    # from the exhausted search made max(|damping * step|) vanishingly small.
    assert not converged
    assert iterations == 2
    # beta is the iterate accepted at iteration 1: neither the starting value
    # nor the candidate the search rejected.
    assert np.array_equal(beta, accepted_betas[0])
    assert not np.allclose(beta, np.zeros_like(beta))


def test_irls_converges_without_a_failed_line_search_on_a_clean_problem():
    y, design, group_index, n_groups = _poisson_irls_inputs()
    _, _, _, _, converged, iterations, line_search_failed = _ppml_irls(
        y, design, group_index, n_groups
    )
    assert converged
    assert not line_search_failed
    assert iterations < 100


def test_ppml_is_invariant_to_the_currency_scale():
    """Pounds or pence, the fit must follow the same path to the same answer.

    Scaling the outcome by a constant is absorbed exactly by the concentrated
    group effects, so beta, the Newton steps and the iteration count are all
    invariant. Only the acceptance test can break that, and only if it uses an
    absolute tolerance: at 1e9 a spurious rejection near the optimum is the
    normal case, not an edge case.
    """
    balanced, products = _levels_panel(zeros=True)
    small = balanced.copy()
    large = balanced.assign(value_gbp=balanced["value_gbp"] * 1e7)
    assert small["value_gbp"].max() < 1e3
    assert large["value_gbp"].max() > 1e9

    small_fit = fit_ppml_model(small, products, 202505, ANTICIPATION_MONTHS)
    large_fit = fit_ppml_model(large, products, 202505, ANTICIPATION_MONTHS)
    assert small_fit["estimate_available"] and large_fit["estimate_available"]
    assert small_fit["iterations"] == large_fit["iterations"]
    assert not small_fit["line_search_failed"]
    assert not large_fit["line_search_failed"]
    assert large_fit["log_effect"] == pytest.approx(small_fit["log_effect"], rel=1e-12)
    assert large_fit["standard_error"] == pytest.approx(
        small_fit["standard_error"], rel=1e-12
    )


def test_ppml_suppresses_a_rank_deficient_fit_instead_of_reporting_it():
    """A short span makes post collinear with the trend and season dummies.

    The pseudo-inverse will always return some split of the collinear
    coefficients; on this panel it returns wildly different numbers for
    different sample starts. None of them is an estimate.
    """
    balanced, products = _levels_panel(zeros=True, post_effect=-0.3)
    for start in [202401, 202407, 202410]:
        short = balanced[balanced["month"].ge(start)]
        fitted = fit_ppml_model(short, products, 202505, ANTICIPATION_MONTHS)
        assert fitted["rank_deficient"]
        assert not fitted["estimate_available"]
        for field in ESTIMATE_FIELDS:
            assert fitted[field] is None
        assert "rank deficient" in fitted["status"]
        # Sample-description fields are still reported, and the whole payload
        # stays strict JSON: no NaN, no numpy scalars.
        assert fitted["row_count"] > 0
        assert json.loads(json.dumps(fitted, allow_nan=False))["log_effect"] is None


def test_ppml_suppresses_an_estimate_separated_on_us_post():
    """All US post-policy flows zero: the MLE is minus infinity, not -100.5.

    ``separated_pairs_dropped`` only catches pairs that never trade, so this
    case reaches the estimator; the old code reported log_effect = -100.5 with
    a standard error of 1e-16 and p = 0.
    """
    balanced, products = _levels_panel(zeros=True)
    separated = balanced.copy()
    us_post = separated["country_code"].eq("US") & separated["month"].ge(202505)
    separated.loc[us_post, "value_gbp"] = 0.0
    fitted = fit_ppml_model(separated, products, 202505, ANTICIPATION_MONTHS)
    assert fitted["us_post_separated"]
    assert not fitted["estimate_available"]
    for field in ESTIMATE_FIELDS:
        assert fitted[field] is None
    assert "separated" in fitted["status"]
    # The all-zero-pair check does not fire here: the pairs do trade pre-policy.
    assert fitted["separated_pairs_dropped"] == 0
    json.dumps(fitted, allow_nan=False)


def test_ppml_reports_a_clean_fit_as_available():
    balanced, products = _levels_panel(zeros=True)
    fitted = fit_ppml_model(balanced, products, 202505, ANTICIPATION_MONTHS)
    assert fitted["estimate_available"]
    assert fitted["status"] == "ok"
    assert not fitted["us_post_separated"]
    assert not fitted["line_search_failed"]
    for field in ESTIMATE_FIELDS:
        assert isinstance(fitted[field], float)
    json.dumps(fitted, allow_nan=False)


def test_continuous_trade_restriction_keeps_only_products_without_zero_fills():
    months = [202401, 202402]
    rows = []
    for product in ["1111", "2222", "3333"]:
        for destination in ["US", "CA", "JP", "AU"]:
            for month in months:
                value = 10.0
                if product == "2222" and destination == "JP" and month == 202402:
                    value = 0.0  # extensive-margin gap in a comparison destination
                if product == "3333" and destination == "US" and month == 202401:
                    value = 0.0
                rows.append(
                    {
                        "hs4": product,
                        "country_code": destination,
                        "month": month,
                        "value_gbp": value,
                    }
                )
    balanced = pd.DataFrame(rows)
    assert list(continuous_trade_products(balanced)) == ["1111"]
    # Restricting the months considered can readmit a product.
    assert list(continuous_trade_products(balanced, months=[202401])) == [
        "1111",
        "2222",
    ]


def _continuity_panel() -> pd.DataFrame:
    """Balanced levels panel spanning the continuity window and the policy.

    ``2222`` trades continuously before the policy and stops shipping to the US
    afterwards -- the extensive-margin exit the zero-robustness check exists to
    measure. ``3333`` has a genuine pre-period gap.
    """
    months = [
        int(period.strftime("%Y%m"))
        for period in pd.period_range("2024-01", "2025-12", freq="M")
    ]
    rows = []
    for product in ["1111", "2222", "3333"]:
        for destination in ["US", "CA", "JP", "AU"]:
            for month in months:
                value = 10.0
                if product == "2222" and destination == "US" and month >= POLICY_MONTH:
                    value = 0.0
                if product == "3333" and destination == "JP" and month == 202403:
                    value = 0.0
                rows.append(
                    {
                        "hs4": product,
                        "country_code": destination,
                        "month": month,
                        "value_gbp": value,
                    }
                )
    return pd.DataFrame(rows)


def test_continuity_months_stop_before_the_anticipation_window():
    balanced = _continuity_panel()
    months = continuity_months(balanced)
    assert months[0] == 202401
    assert months[-1] == CONTINUITY_WINDOW_END
    assert max(months) < ANTICIPATION_WINDOW[0]
    assert not set(months) & set(ANTICIPATION_MONTHS)
    assert all(month < POLICY_MONTH for month in months)


def test_continuous_trade_sample_is_selected_on_pre_policy_months_only():
    balanced = _continuity_panel()
    # Judging continuity over the whole panel selects on the outcome: the
    # product whose US flow died after the tariff is exactly the observation
    # the check is about, and dropping it biases the restricted estimate
    # toward zero by construction.
    assert list(continuous_trade_products(balanced)) == ["1111"]
    # Restricted to pre-treatment months the dying product is RETAINED.
    selected = continuous_trade_products(balanced, months=continuity_months(balanced))
    assert list(selected) == ["1111", "2222"]
    # The pre-period gap is still disqualifying.
    assert "3333" not in selected


def test_precision_summary_reports_power_not_just_significance():
    fit = {
        "log_effect": -0.2235,
        "standard_error": 0.2110,
        "p_value": 0.29,
        "ci_lower": -0.6372,
        "ci_upper": 0.1901,
        "product_count": 70,
        "effective_product_count": 7.48,
    }
    summary = precision_summary(fit)
    assert summary["ci_contains_zero"]
    assert summary["confidence_interval_width_log"] == pytest.approx(0.8273, abs=1e-3)
    assert summary["minimum_detectable_log_effect_80_power"] == pytest.approx(
        2.8016 * 0.2110, rel=1e-6
    )
    # The smallest detectable decline is larger than the headline effect, so a
    # null in this group is uninformative rather than a failed falsification.
    assert summary["minimum_detectable_log_effect_80_power"] > abs(fit["log_effect"])


def test_group_contrast_combines_independent_standard_errors():
    high = {"log_effect": -0.2235, "standard_error": 0.2110}
    other = {"log_effect": -0.3984, "standard_error": 0.1149}
    contrast = group_contrast(high, other)
    assert contrast["log_effect_difference"] == pytest.approx(0.1749, abs=1e-4)
    assert contrast["standard_error"] == pytest.approx(
        np.sqrt(0.2110**2 + 0.1149**2), rel=1e-9
    )
    assert contrast["p_value"] > 0.05
