"""Product-destination benchmark for UK exports around the 2025 US tariffs.

The model compares each HS4 product's exports to the United States with the
same product's mean log exports to Canada, Japan and Australia:

    gap_pt = product FE + month-of-year FE + trend + beta * post_t + error_pt

It is a public-data reduced-form benchmark, not a structural tariff model.

Two design choices deserve emphasis because they materially change the
estimate.

Anticipation. UK goods exports to the United States surged through the first
quarter of 2025 ahead of the April 2025 tariff announcement. Dropping only the
announcement month leaves the front-running months inside the pre-period, which
inflates the estimated post-policy decline and, if the plotted series is
normalised on those months, measures the post-policy gap against the
front-running peak. The estimation sample therefore excludes the whole
anticipation window (``ANTICIPATION_WINDOW``, default 2025m1-2025m4) and the
figure is normalised on a clean pre-anticipation base (``NORMALISATION_BASE``,
default calendar 2024). The original single-month choice is retained as a named
specification check so the sensitivity is visible in the diagnostics rather
than silently replaced.

Zeros. The panel is balanced with ``fill_value=0`` and the outcome is
``log1p(value)``, so an absent flow enters the control term as log1p(0) = 0
rather than as missing. Extensive-margin movement in the small comparison
destinations therefore maps mechanically into the outcome gap. Two robustness
specifications bound that: a restriction to products traded continuously and
positively with all four destinations, and a PPML (Poisson pseudo-maximum
likelihood) estimator on export levels, which admits zeros without a log
transform.

The continuity restriction is judged on pre-treatment months only. Requiring
positive trade in the post-policy months as well would drop exactly the
products whose US flow went to zero after the tariff, which is selection on the
dependent variable and drives the restricted estimate toward zero by
construction. The PPML fit reports null estimates with a ``status`` string
rather than numbers whenever it is not identified: rank deficiency, a failed
line search, non-convergence, or complete separation on the ``us_post``
regressor.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/public/hmrc_exports_product_destination_201801_202606.csv.gz"
OUT_MONTHLY = ROOT / "results/hmrc_destination_event_study_monthly.csv"
OUT_DIAGNOSTICS = ROOT / "results/hmrc_destination_event_study.json"
OUT_FIGURE = ROOT / "results/figures/hmrc_destination_event_study.png"
DESTINATIONS = ["US", "CA", "JP", "AU"]
POLICY_MONTH = 202505
# Number of pre-policy months treated as anticipation-contaminated. This is the
# single knob: the primary exclusion window and the symmetric placebo windows
# are both derived from it, so widening one cannot silently desynchronise the
# other.
ANTICIPATION_LEAD_MONTHS = 4
# The pre-2026 vintage of this script dropped only the announcement month.
# Kept so the older specification can be reproduced and reported.
LEGACY_ANTICIPATION_MONTH = 202504
# Base period for the plotted series: a clean window before the front-running.
NORMALISATION_BASE = (202401, 202412)
# Window used to build the value weights: must end before the anticipation
# window so that front-running does not enter the weights either.
WEIGHT_WINDOW = (202201, 202412)


def month_range(start: int, end: int) -> list[int]:
    """Inclusive list of YYYYMM months from ``start`` to ``end``."""
    periods = pd.period_range(str(start), str(end), freq="M")
    return [int(period.strftime("%Y%m")) for period in periods]


def previous_month(month: int) -> int:
    return int((pd.Period(str(month), freq="M") - 1).strftime("%Y%m"))


def window_before(month: int, n_months: int) -> tuple[int, int]:
    """Inclusive (start, end) of the ``n_months`` months before ``month``."""
    start = int((pd.Period(str(month), freq="M") - n_months).strftime("%Y%m"))
    return start, previous_month(month)


def format_month_window(start: int, end: int) -> str:
    """Human-readable label for a YYYYMM window, e.g. ``2024`` or ``Jan-Apr 2025``."""
    if start == end:
        return pd.Period(str(start), freq="M").strftime("%b %Y")
    if start // 100 == end // 100:
        if start % 100 == 1 and end % 100 == 12:
            return str(start // 100)
        return (
            f"{pd.Period(str(start), freq='M').strftime('%b')}-"
            f"{pd.Period(str(end), freq='M').strftime('%b %Y')}"
        )
    return (
        f"{pd.Period(str(start), freq='M').strftime('%b %Y')}-"
        f"{pd.Period(str(end), freq='M').strftime('%b %Y')}"
    )


def _as_month_list(excluded_months: int | Iterable[int] | None) -> list[int]:
    """Normalise the excluded-month argument to a list of YYYYMM integers."""
    if excluded_months is None:
        return []
    if isinstance(excluded_months, (int, np.integer)):
        return [int(excluded_months)]
    return [int(month) for month in excluded_months]


# Months dropped from the estimation sample: the announcement month plus the
# front-running window that preceded it. Derived from POLICY_MONTH and
# ANTICIPATION_LEAD_MONTHS rather than written out, so that the primary
# exclusion window and the placebo windows built by ``window_before`` in
# ``main`` can never drift apart.
ANTICIPATION_WINDOW = window_before(POLICY_MONTH, ANTICIPATION_LEAD_MONTHS)
ANTICIPATION_MONTHS = month_range(*ANTICIPATION_WINDOW)
# The anticipation window must run right up to the policy month and must
# contain the legacy single-month exclusion, otherwise the "window" and
# "single month" specifications are not nested and the sensitivity reported in
# ``anticipation_sensitivity`` is not a like-for-like comparison.
assert len(ANTICIPATION_MONTHS) == ANTICIPATION_LEAD_MONTHS
assert ANTICIPATION_WINDOW[1] == previous_month(POLICY_MONTH)
assert ANTICIPATION_WINDOW[0] <= LEGACY_ANTICIPATION_MONTH <= ANTICIPATION_WINDOW[1]
# Continuity for the zero-robustness sample is judged on this window only: it
# ends before the anticipation window, so no post-treatment month can select
# the sample. See ``continuous_trade_products``.
CONTINUITY_WINDOW_END = previous_month(ANTICIPATION_WINDOW[0])


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


def continuous_trade_products(
    balanced: pd.DataFrame,
    months: Iterable[int] | None = None,
) -> pd.Index:
    """HS4 codes with strictly positive trade in every destination-month.

    The balanced panel fills unobserved product-destination-months with zero,
    which the log1p outcome cannot distinguish from a genuine small flow. This
    selects the products for which no such fill occurs, so the gap outcome is
    an intensive-margin object throughout.

    ``months`` restricts the window over which continuity is judged and must be
    supplied with a pre-treatment window whenever the selected sample is then
    used to estimate a treatment effect. Judging continuity over the whole
    panel would require positive trade in the post-policy months too, which
    drops precisely the products whose US flow went to zero after the tariff --
    selection on the outcome, biasing the estimate toward zero by construction.
    :func:`continuity_months` builds the pre-treatment window used by ``main``.
    """
    data = balanced
    if months is not None:
        data = data[data["month"].isin(list(months))]
    positive = data.assign(_positive=data["value_gbp"].gt(0))
    complete = positive.groupby("hs4")["_positive"].all()
    return pd.Index(complete.index[complete.to_numpy()], name="hs4")


def continuity_months(
    balanced: pd.DataFrame,
    end_month: int = CONTINUITY_WINDOW_END,
) -> list[int]:
    """Observed months at or before ``end_month``, sorted.

    The default ends the month before the anticipation window opens, so the
    continuous-trade sample is chosen entirely on pre-treatment information.
    """
    months = sorted(int(month) for month in pd.unique(balanced["month"]))
    return [month for month in months if month <= end_month]


def fit_post_model(
    panel: pd.DataFrame,
    weights: pd.Series,
    policy_month: int,
    excluded_months: int | Iterable[int] | None = None,
    end_month: int | None = None,
    outcome: str = "gap",
    start_month: int | None = None,
) -> dict[str, float]:
    """WLS of the destination gap on a post dummy, product FE, season and trend.

    ``excluded_months`` are dropped from the estimation sample: a single month
    reproduces the original specification, the anticipation window removes the
    front-running episode as well.
    """
    dropped = _as_month_list(excluded_months)
    data = panel[panel["hs4"].isin(weights.index)].copy()
    data = data[~data["month"].isin(dropped)]
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
        "months_excluded": dropped,
        "differential_linear_trend": float(result.params["trend"]),
        "differential_linear_trend_standard_error": float(result.bse["trend"]),
        "differential_linear_trend_p_value": float(result.pvalues["trend"]),
    }


def _weighted_group_demean(
    values: np.ndarray,
    group_index: np.ndarray,
    weights: np.ndarray,
    n_groups: int,
) -> np.ndarray:
    """Subtract weighted within-group means, column by column."""
    totals = np.bincount(group_index, weights=weights, minlength=n_groups)
    demeaned = np.empty_like(values)
    for column in range(values.shape[1]):
        sums = np.bincount(
            group_index, weights=weights * values[:, column], minlength=n_groups
        )
        demeaned[:, column] = values[:, column] - (sums / totals)[group_index]
    return demeaned


def _backtracking_line_search(
    objective,
    beta: np.ndarray,
    step: np.ndarray,
    loglik: float,
    max_halvings: int = 30,
    loglik_atol: float = 1e-10,
    loglik_rtol: float = 1e-12,
) -> tuple[bool, np.ndarray, float, object, float]:
    """Halve ``step`` until ``objective`` no longer falls below ``loglik``.

    The acceptance threshold is ``loglik - (atol + rtol * |loglik|)``, i.e.
    relative to the scale of the objective. The concentrated Poisson
    log-likelihood is dominated by ``y @ linear`` with ``y`` in pounds, so at
    HMRC magnitudes (1e8-1e9) one unit in the last place is many orders of
    magnitude larger than any fixed absolute tolerance; with an absolute
    tolerance a step that is numerically an improvement is routinely rejected
    near the optimum, which is where the search spends most of its time.

    Returns ``(accepted, beta, loglik, extra, damping)``. ``damping`` is the
    factor that was actually applied to the accepted candidate, so the caller
    can test convergence on the step it really took. If nothing is accepted the
    caller's own iterate is returned unchanged with ``damping = 0.0``: a
    rejected candidate has a worse log-likelihood and must never be adopted.
    """
    threshold = loglik - (loglik_atol + loglik_rtol * abs(loglik))
    damping = 1.0
    for _ in range(max_halvings):
        candidate = beta + damping * step
        candidate_loglik, extra = objective(candidate)
        if np.isfinite(candidate_loglik) and candidate_loglik >= threshold:
            return True, candidate, float(candidate_loglik), extra, damping
        damping /= 2.0
    return False, beta, loglik, None, 0.0


def _ppml_irls(
    y: np.ndarray,
    design: np.ndarray,
    group_index: np.ndarray,
    n_groups: int,
    max_iter: int = 100,
    tolerance: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool, int, bool]:
    """Poisson MLE with one absorbed fixed-effect dimension.

    Equivalent to ``statsmodels`` GLM with a Poisson family and a full set of
    group dummies, but the group effects are concentrated out analytically
    (``alpha_g = log(sum_g y / sum_g exp(x'b))``), which is what makes the
    product-destination fixed effects tractable at this panel size. The
    returned Hessian and residuals use the concentrated (within-group demeaned)
    design, so the sandwich formed from them is the usual PPML cluster-robust
    covariance.

    Returns ``(beta, hessian, demeaned, residual, converged, iterations,
    line_search_failed)``. ``beta`` is always the last iterate that the line
    search accepted. A failed line search is a failure, not a convergence:
    ``converged`` is False and ``line_search_failed`` is True, and callers must
    treat the returned coefficients as unusable.
    """
    n_params = design.shape[1]
    beta = np.zeros(n_params)
    group_y = np.bincount(group_index, weights=y, minlength=n_groups)

    def concentrated(beta_value: np.ndarray) -> tuple[float, np.ndarray]:
        linear = design @ beta_value
        linear = linear - linear.max()
        expo = np.exp(linear)
        group_expo = np.bincount(group_index, weights=expo, minlength=n_groups)
        safe = np.where(group_expo > 0, group_expo, 1.0)
        scale = np.where(group_expo > 0, group_y / safe, 0.0)
        mu = expo * scale[group_index]
        loglik = float(
            y @ linear - group_y @ np.log(np.where(group_expo > 0, group_expo, 1.0))
        )
        return loglik, mu

    loglik, mu = concentrated(beta)
    converged = False
    line_search_failed = False
    iterations = 0
    for iterations in range(1, max_iter + 1):
        demeaned = _weighted_group_demean(design, group_index, mu, n_groups)
        residual = y - mu
        score = demeaned.T @ residual
        hessian = demeaned.T @ (mu[:, None] * demeaned)
        try:
            step = np.linalg.solve(hessian, score)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, score, rcond=None)[0]
        accepted, candidate, candidate_loglik, candidate_mu, damping = (
            _backtracking_line_search(concentrated, beta, step, loglik)
        )
        if not accepted:
            # Every damped candidate lowered the log-likelihood. Keep the last
            # accepted iterate and report the failure; adopting the rejected
            # candidate would move the fit to a worse point, and testing
            # convergence on the damping left over from the exhausted search
            # would call that move a success.
            line_search_failed = True
            break
        beta, loglik, mu = candidate, candidate_loglik, candidate_mu
        if np.max(np.abs(damping * step)) < tolerance:
            converged = True
            break
    demeaned = _weighted_group_demean(design, group_index, mu, n_groups)
    hessian = demeaned.T @ (mu[:, None] * demeaned)
    return beta, hessian, demeaned, y - mu, converged, iterations, line_search_failed


def _cluster_covariance(
    hessian: np.ndarray,
    demeaned: np.ndarray,
    residual: np.ndarray,
    cluster_index: np.ndarray,
    n_clusters: int,
    n_absorbed: int = 0,
) -> np.ndarray:
    """Cluster-robust sandwich for the concentrated Poisson estimator.

    The absorbed fixed effects are counted in the degrees-of-freedom
    adjustment, so the result matches a Poisson GLM fitted with explicit group
    dummies and clustered standard errors.
    """
    scores = np.column_stack(
        [
            np.bincount(
                cluster_index,
                weights=demeaned[:, column] * residual,
                minlength=n_clusters,
            )
            for column in range(demeaned.shape[1])
        ]
    )
    meat = scores.T @ scores
    bread = np.linalg.pinv(hessian)
    n_obs, n_free = demeaned.shape
    n_params = n_free + n_absorbed
    correction = (n_clusters / max(n_clusters - 1, 1)) * (
        (n_obs - 1) / max(n_obs - n_params, 1)
    )
    return bread @ meat @ bread * correction


ESTIMATE_FIELDS = (
    "log_effect",
    "standard_error",
    "p_value",
    "ci_lower",
    "ci_upper",
    "proportional_effect",
)


def _separation_reason(y: np.ndarray, indicator: np.ndarray) -> str | None:
    """Describe complete separation of ``y`` on a 0/1 regressor, else ``None``.

    ``separated_pairs_dropped`` in :func:`fit_ppml_model` only catches product
    -destination pairs that never trade. It does not catch separation on the
    regressor of interest: if every US post-policy flow is zero the Poisson
    likelihood is maximised as the coefficient goes to minus infinity, and the
    IRLS iteration simply stops at whatever large negative number it reached,
    with a standard error near machine zero and a p-value of exactly zero. That
    is a non-existent maximum, not a precise estimate.
    """
    treated = indicator > 0
    if not treated.any():
        return "no observation has us_post = 1, so the coefficient is not identified"
    if treated.all():
        return "every observation has us_post = 1, so the coefficient is not identified"
    if not (y[treated] > 0).any():
        return (
            "every US post-policy flow is zero, so the us_post coefficient is "
            "completely separated and its maximum likelihood estimate is minus "
            "infinity"
        )
    if not (y[~treated] > 0).any():
        return (
            "every flow outside the US post-policy cell is zero, so the us_post "
            "coefficient is completely separated"
        )
    return None


def fit_ppml_model(
    balanced: pd.DataFrame,
    products: Iterable[str],
    policy_month: int,
    excluded_months: int | Iterable[int] | None = None,
    start_month: int | None = None,
    end_month: int | None = None,
) -> dict[str, object]:
    """PPML on export levels: the zero-robust analogue of the gap regression.

        E[value_pdt] = exp(a_pd + trend + season + post
                           + US_d * (trend + season + post))

    Product-destination fixed effects ``a_pd`` are absorbed; the coefficient on
    ``US x post`` is the differential log change in US-bound exports, directly
    comparable with ``beta`` from :func:`fit_post_model`. Because the estimator
    works on levels, months in which a product is not shipped to a destination
    enter as genuine zeros rather than as ``log1p(0)``.

    Unlike the gap model this conditions on common (rather than
    product-specific) month effects, since product-month effects would be a
    second absorbed dimension.

    When the fit is not usable -- the concentrated Hessian is rank deficient,
    the IRLS line search failed, the iteration did not converge, or ``us_post``
    is completely separated -- every estimate field is ``None`` rather than a
    number, ``estimate_available`` is False and ``status`` says why. A
    pseudo-inverse will always return *some* split of a collinear coefficient
    and a separated fit will always return *some* large coefficient with a
    negligible standard error; neither is an estimate, so neither is reported.
    """
    dropped = _as_month_list(excluded_months)
    product_list = pd.Index(list(products))
    data = balanced[balanced["hs4"].isin(product_list)].copy()
    data = data[~data["month"].isin(dropped)]
    if start_month is not None:
        data = data[data["month"].ge(start_month)]
    if end_month is not None:
        data = data[data["month"].le(end_month)]
    data = data.sort_values(["hs4", "country_code", "month"]).reset_index(drop=True)

    date = pd.to_datetime(data["month"].astype(str), format="%Y%m")
    trend = ((date.dt.year - date.dt.year.min()) * 12 + date.dt.month - 1).to_numpy(
        dtype=float
    )
    post = data["month"].ge(policy_month).to_numpy(dtype=float)
    is_us = data["country_code"].eq("US").to_numpy(dtype=float)
    seasonal = pd.get_dummies(
        date.dt.month, prefix="month", drop_first=True, dtype=float
    ).to_numpy()

    common = np.column_stack([trend, post, seasonal])
    differential = common * is_us[:, None]
    design = np.column_stack([common, differential])
    names = (
        ["trend", "post"]
        + [f"season_{index}" for index in range(seasonal.shape[1])]
        + ["us_trend", "us_post"]
        + [f"us_season_{index}" for index in range(seasonal.shape[1])]
    )
    post_position = names.index("us_post")

    pair = data["hs4"].astype(str) + "|" + data["country_code"].astype(str)
    group_index, groups = pd.factorize(pair)
    y = data["value_gbp"].to_numpy(dtype=float)
    group_totals = np.bincount(group_index, weights=y, minlength=len(groups))
    # Product-destination pairs with no trade at all are perfectly separated:
    # their fitted mean is zero everywhere and they carry no information.
    keep = group_totals[group_index] > 0
    separated = int((group_totals <= 0).sum())
    if not keep.all():
        data = data[keep]
        design = design[keep]
        y = y[keep]
        group_index = pd.factorize(pd.Series(group_index[keep]))[0]

    n_groups = int(group_index.max()) + 1 if len(group_index) else 0
    cluster_index, clusters = pd.factorize(data["hs4"].to_numpy())
    beta, hessian, demeaned, residual, converged, iterations, line_search_failed = (
        _ppml_irls(y, design, group_index, n_groups)
    )
    covariance = _cluster_covariance(
        hessian, demeaned, residual, cluster_index, len(clusters), n_groups
    )
    # A short sample makes the post dummy collinear with the trend and the
    # month-of-year effects; flag it rather than report a pseudo-inverse split.
    rank_deficient = bool(np.linalg.matrix_rank(hessian) < hessian.shape[0])
    separation = _separation_reason(y, design[:, post_position])
    effect = float(beta[post_position])
    variance = float(covariance[post_position, post_position])
    standard_error = float(np.sqrt(variance)) if variance > 0 else float("nan")

    problems: list[str] = []
    if separation is not None:
        problems.append(separation)
    if rank_deficient:
        problems.append(
            "the concentrated Hessian is rank deficient, so the post dummy is "
            "collinear with the trend and the month-of-year effects over this "
            "sample and any reported split between them is an artefact of the "
            "pseudo-inverse"
        )
    if line_search_failed:
        problems.append(
            f"the IRLS line search found no improving step at iteration "
            f"{iterations}"
        )
    elif not converged:
        problems.append(
            f"the IRLS iteration did not converge within {iterations} iterations"
        )
    if not np.isfinite(standard_error) or standard_error <= 0:
        problems.append(
            "the clustered variance of the us_post coefficient is not positive"
        )
    estimate_available = not problems

    if estimate_available:
        z_statistic = effect / standard_error
        estimate: dict[str, float | None] = {
            "log_effect": effect,
            "standard_error": standard_error,
            "p_value": float(2 * stats.norm.sf(abs(z_statistic))),
            "ci_lower": effect - 1.96 * standard_error,
            "ci_upper": effect + 1.96 * standard_error,
            "proportional_effect": float(np.expm1(effect)),
        }
        status = "ok"
    else:
        # JSON null, not a number: a suppressed estimate must not be readable
        # as a point estimate by anything downstream.
        estimate = dict.fromkeys(ESTIMATE_FIELDS, None)
        status = "estimate suppressed: " + "; ".join(problems)

    return {
        **estimate,
        "estimate_available": estimate_available,
        "status": status,
        "product_count": int(data["hs4"].nunique()),
        "row_count": int(len(data)),
        "zero_observation_share": float(np.mean(y <= 0)),
        "separated_pairs_dropped": separated,
        "us_post_separated": separation is not None,
        "months_excluded": dropped,
        "converged": bool(converged),
        "line_search_failed": bool(line_search_failed),
        "rank_deficient": rank_deficient,
        "iterations": int(iterations),
        "estimator": (
            "Poisson pseudo-maximum likelihood on export levels; "
            "product-destination fixed effects absorbed; common and "
            "US-differential trend, month-of-year effects and post dummy; "
            "standard errors clustered by HS4"
        ),
    }


def monthly_gap_series(
    panel: pd.DataFrame,
    weights: pd.Series,
    base_window: tuple[int, int] = NORMALISATION_BASE,
) -> pd.DataFrame:
    """Weighted monthly gap, normalised on a pre-anticipation base window.

    The base must sit before the front-running episode; normalising on the
    anticipation months themselves would measure the post-policy gap relative
    to the front-running peak.
    """
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
    base_months = month_range(*base_window)
    in_base = result["month"].isin(base_months)
    if not in_base.any():
        raise ValueError(
            f"normalisation base {base_window} is outside the observed months"
        )
    baseline = result.loc[in_base, "weighted_gap"].mean()
    result["normalised_gap"] = result["weighted_gap"] - baseline
    result["base_period"] = format_month_window(*base_window)
    return result


def precision_summary(fit: dict[str, float]) -> dict[str, float]:
    """Power and precision facts for a single estimate.

    Reported so that a null result can be read as uninformative rather than as
    evidence of no effect.
    """
    standard_error = float(fit["standard_error"])
    return {
        "product_count": fit["product_count"],
        "effective_product_count": fit["effective_product_count"],
        "log_effect": fit["log_effect"],
        "standard_error": standard_error,
        "p_value": fit["p_value"],
        "confidence_interval_width_log": float(fit["ci_upper"] - fit["ci_lower"]),
        "confidence_interval_width_percentage_points": float(
            100 * (np.expm1(fit["ci_upper"]) - np.expm1(fit["ci_lower"]))
        ),
        # Two-sided 5% test, 80% power: (1.96 + 0.8416) * SE.
        "minimum_detectable_log_effect_80_power": float(2.8016 * standard_error),
        "minimum_detectable_proportional_effect_80_power": float(
            np.expm1(-2.8016 * standard_error)
        ),
        "ci_contains_zero": bool(fit["ci_lower"] <= 0 <= fit["ci_upper"]),
    }


def group_contrast(first: dict[str, float], second: dict[str, float]) -> dict:
    """Difference between two independently estimated group effects."""
    difference = float(first["log_effect"] - second["log_effect"])
    standard_error = float(
        np.sqrt(first["standard_error"] ** 2 + second["standard_error"] ** 2)
    )
    z_statistic = difference / standard_error if standard_error > 0 else np.nan
    return {
        "log_effect_difference": difference,
        "standard_error": standard_error,
        "p_value": float(2 * stats.norm.sf(abs(z_statistic))),
        "ci_lower": difference - 1.96 * standard_error,
        "ci_upper": difference + 1.96 * standard_error,
        "note": (
            "Difference of two separately estimated coefficients on disjoint "
            "product sets; the standard error assumes independence across the "
            "two sets."
        ),
    }


def main() -> None:
    # Imported here so the estimators can be exercised without a plotting stack.
    import matplotlib.pyplot as plt

    raw = pd.read_csv(INPUT)
    panel, balanced = build_gap_panel(raw)
    weights = product_weights(balanced, *WEIGHT_WINDOW)
    equal_weights = pd.Series(1.0, index=weights.index)
    uncapped_weights = product_weights(balanced, *WEIGHT_WINDOW, cap_quantile=None)

    primary = fit_post_model(panel, weights, POLICY_MONTH, ANTICIPATION_MONTHS)
    single_month = fit_post_model(
        panel, weights, POLICY_MONTH, LEGACY_ANTICIPATION_MONTH
    )
    no_exclusion = fit_post_model(panel, weights, POLICY_MONTH, None)
    anticipation_sensitivity = {
        "window_excluded_primary": primary,
        "single_month_excluded_original": single_month,
        "no_month_excluded": no_exclusion,
        "log_effect_shift_window_minus_single_month": float(
            primary["log_effect"] - single_month["log_effect"]
        ),
        "proportional_effect_shift_window_minus_single_month": float(
            primary["proportional_effect"] - single_month["proportional_effect"]
        ),
        "note": (
            "The original specification dropped only the announcement month "
            f"({LEGACY_ANTICIPATION_MONTH}), leaving the first-quarter 2025 "
            "front-running surge inside the pre-period. The primary "
            "specification drops the whole anticipation window "
            f"{ANTICIPATION_MONTHS[0]}-{ANTICIPATION_MONTHS[-1]}. Both are "
            "reported so the contamination can be quantified."
        ),
    }

    unweighted = fit_post_model(panel, equal_weights, POLICY_MONTH, ANTICIPATION_MONTHS)
    uncapped = fit_post_model(
        panel, uncapped_weights, POLICY_MONTH, ANTICIPATION_MONTHS
    )
    individual_controls = {
        destination: fit_post_model(
            panel,
            weights,
            POLICY_MONTH,
            ANTICIPATION_MONTHS,
            outcome=f"gap_{destination.lower()}",
        )
        for destination in ["CA", "JP", "AU"]
    }
    sample_start_sensitivity = {
        str(start): fit_post_model(
            panel,
            weights,
            POLICY_MONTH,
            ANTICIPATION_MONTHS,
            start_month=start,
        )
        for start in [202001, 202201, 202301]
    }
    start_effects = {
        start: fit["proportional_effect"]
        for start, fit in sample_start_sensitivity.items()
    }

    # Zero-robustness. (i) products traded continuously and positively with all
    # four destinations, so no log1p(0) enters the outcome; (ii) PPML on levels.
    # Continuity is judged on pre-treatment months only. Requiring positive
    # trade in the post-policy months as well would drop exactly the products
    # whose US flow went to zero after the tariff -- the extensive-margin exits
    # this check is about -- which is selection on the dependent variable and
    # biases the restricted estimate toward zero by construction.
    continuity_window = continuity_months(balanced)
    continuous_products = continuous_trade_products(balanced, months=continuity_window)
    continuous_weights = weights[weights.index.isin(continuous_products)]
    continuity_note = {
        "continuity_window": [continuity_window[0], continuity_window[-1]]
        if continuity_window
        else [],
        "continuity_window_label": format_month_window(
            continuity_window[0], continuity_window[-1]
        )
        if continuity_window
        else "",
        "continuity_window_month_count": len(continuity_window),
        "continuity_selection_note": (
            "Products are selected on continuous positive trade with all four "
            "destinations over pre-treatment months only (ending the month "
            "before the anticipation window). Imposing continuity over the "
            "post-policy months as well would select on the outcome by "
            "dropping the products that stopped shipping to the US."
        ),
    }
    if len(continuous_weights) >= 2:
        continuous_fit = fit_post_model(
            panel, continuous_weights, POLICY_MONTH, ANTICIPATION_MONTHS
        )
        continuous_fit["weight_share_of_primary"] = float(
            continuous_weights.sum() / weights.sum()
        )
        continuous_fit["product_share_of_primary"] = float(
            len(continuous_weights) / len(weights)
        )
    else:
        continuous_fit = {
            "unavailable": (
                "fewer than two products are traded positively with all four "
                "destinations in every pre-treatment month"
            ),
            "product_count": int(len(continuous_weights)),
        }
    continuous_fit.update(continuity_note)
    ppml_fit = fit_ppml_model(
        balanced, weights.index, POLICY_MONTH, ANTICIPATION_MONTHS
    )
    zero_robustness = {
        "continuous_positive_trade_products": continuous_fit,
        "ppml_levels": ppml_fit,
        "note": (
            "The primary outcome is log1p of a zero-filled balanced panel, so "
            "an absent flow to a comparison destination enters the control "
            "term as log1p(0) = 0 and extensive-margin movement in the small "
            "comparison destinations passes into the gap. These two "
            "specifications bound that channel; neither replaces the primary "
            "estimate. The PPML estimate fields are null when the fit is not "
            "usable (rank deficiency, a failed line search, non-convergence or "
            "separation on us_post); read ppml_levels.status in that case."
        ),
    }

    high_tariff = weights.index.str[:2].isin(["72", "73", "87"])
    tariff_intensity_groups = {
        "steel_and_auto_chapters": fit_post_model(
            panel, weights[high_tariff], POLICY_MONTH, ANTICIPATION_MONTHS
        ),
        "other_products": fit_post_model(
            panel, weights[~high_tariff], POLICY_MONTH, ANTICIPATION_MONTHS
        ),
    }
    high = tariff_intensity_groups["steel_and_auto_chapters"]
    other = tariff_intensity_groups["other_products"]
    tariff_intensity_power = {
        "steel_and_auto_chapters": precision_summary(high),
        "other_products": precision_summary(other),
        "difference_high_minus_other": group_contrast(high, other),
        "high_tariff_ci_contains_other_products_estimate": bool(
            high["ci_lower"] <= other["log_effect"] <= high["ci_upper"]
        ),
        "note": (
            "The steel and auto chapters carry a small share of the weighted "
            "product mass, so their estimate is imprecise. Read the "
            "minimum detectable effect and the confidence-interval width "
            "before describing this split as a falsification: an estimate "
            "whose interval contains both zero and the other-products effect "
            "cannot discriminate between them."
        ),
    }

    placebos = {}
    for year in range(2019, 2024):
        cutoff = year * 100 + 5
        end = (year + 1) * 100 + 5
        placebo_window = window_before(cutoff, ANTICIPATION_LEAD_MONTHS)
        placebo_excluded = month_range(*placebo_window)
        start = max(201801, (year - 3) * 100 + 1)
        placebo_weights = product_weights(
            balanced, start, previous_month(placebo_window[0])
        )
        placebos[str(year)] = fit_post_model(
            panel,
            placebo_weights,
            cutoff,
            placebo_excluded,
            end_month=end,
        )

    monthly = monthly_gap_series(panel, weights, NORMALISATION_BASE)
    OUT_MONTHLY.parent.mkdir(parents=True, exist_ok=True)
    OUT_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(OUT_MONTHLY, index=False)

    base_label = format_month_window(*NORMALISATION_BASE)
    date = pd.to_datetime(monthly["month"].astype(str), format="%Y%m")
    anticipation_start = pd.Timestamp(
        f"{ANTICIPATION_MONTHS[0] // 100}-{ANTICIPATION_MONTHS[0] % 100:02d}-01"
    )
    policy_start = pd.Timestamp(f"{POLICY_MONTH // 100}-{POLICY_MONTH % 100:02d}-01")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(date, monthly["normalised_gap"], color="#2C6496", linewidth=2)
    ax.axvspan(
        anticipation_start,
        policy_start,
        color="#B50D0D",
        alpha=0.08,
        label="Anticipation window (excluded from estimation)",
    )
    ax.axvline(policy_start, color="#B50D0D", linestyle="--")
    ax.axhline(0, color="#777777", linewidth=0.8)
    ax.set(
        title="UK exports to the US relative to three comparison destinations",
        ylabel=f"Weighted log-export gap relative to {base_label}",
        xlabel="",
    )
    ax.legend(loc="lower left", frameon=False, fontsize=8)
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
        "anticipation_lead_months": ANTICIPATION_LEAD_MONTHS,
        "anticipation_window_omitted": list(ANTICIPATION_WINDOW),
        "anticipation_months_omitted": ANTICIPATION_MONTHS,
        "legacy_single_anticipation_month": LEGACY_ANTICIPATION_MONTH,
        "weight_window": list(WEIGHT_WINDOW),
        "monthly_figure_normalisation_base": list(NORMALISATION_BASE),
        "monthly_figure_normalisation_label": base_label,
        "comparison_destinations": ["Canada", "Japan", "Australia"],
        "outcome": (
            "log(1 + monthly export value) to US minus mean for same HS4 "
            "product to comparison destinations"
        ),
        "specification": (
            "HS4 product fixed effects, month-of-year effects, linear trend; "
            "WLS with 2022-24 pre-policy US values capped at P99; "
            "standard errors clustered by HS4; the "
            f"{ANTICIPATION_MONTHS[0]}-{ANTICIPATION_MONTHS[-1]} anticipation "
            "window is excluded from the estimation sample"
        ),
        "primary_capped_value_weighted": primary,
        "anticipation_sensitivity": anticipation_sensitivity,
        "uncapped_value_weighted": uncapped,
        "equal_product_weighted": unweighted,
        "individual_control_destinations": individual_controls,
        "zero_robustness": zero_robustness,
        "sample_start_sensitivity": sample_start_sensitivity,
        "sample_start_sensitivity_note": {
            "proportional_effect_by_start_month": start_effects,
            "range_percentage_points": float(
                100 * (max(start_effects.values()) - min(start_effects.values()))
            ),
            "note": (
                "The estimate attenuates as the pre-period shortens. Together "
                "with the differential linear trend this is consistent with "
                "the post dummy absorbing part of a pre-existing differential "
                "trend; the level of the headline effect should be read as "
                "bounded by this range rather than as a point identified "
                "independently of the sample window."
            ),
        },
        "pre_trend_diagnostics": {
            "primary_differential_linear_trend": primary[
                "differential_linear_trend"
            ],
            "primary_differential_linear_trend_standard_error": primary[
                "differential_linear_trend_standard_error"
            ],
            "primary_differential_linear_trend_p_value": primary[
                "differential_linear_trend_p_value"
            ],
            "note": (
                "Monthly differential trend in the US-versus-comparison gap "
                "estimated on the pre-policy sample alongside the post dummy. "
                "A trend that is not clearly zero means the post dummy and "
                "the trend are separated only by functional form."
            ),
        },
        "tariff_intensity_groups": tariff_intensity_groups,
        "tariff_intensity_power": tariff_intensity_power,
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
