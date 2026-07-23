"""Optional, transparent extensions beyond the baseline SIC-only stress test.

These functions are deliberately pure and data-agnostic. They provide a
calibrated interface for future worker-level first-stage estimates and a
household price channel without pretending that coefficients are identified
from the current cross-sectional FRS snapshot.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def calibrated_worker_probabilities(
    base_probability: np.ndarray | pd.Series,
    covariates: pd.DataFrame,
    coefficients: pd.Series | None = None,
    weights: np.ndarray | pd.Series | None = None,
) -> np.ndarray:
    """Apply a logit heterogeneity index while preserving the weighted mean.

    ``base_probability`` is the sector-level shock probability. Covariates are
    z-scored and multiplied by ``coefficients``; the resulting odds are
    multiplicatively tilted, then a scalar intercept is solved by bisection so
    that the weighted mean remains the original sector probability. This makes
    the extension accounting-consistent while coefficients can later be
    replaced by estimates from linked longitudinal data.
    """
    p = np.asarray(base_probability, dtype=float)
    if np.any((p < 0) | (p > 1)):
        raise ValueError("base_probability must lie in [0, 1]")
    x = covariates.astype(float).to_numpy()
    if x.ndim != 2 or len(x) != len(p):
        raise ValueError("covariates and base_probability must have equal length")
    beta = np.zeros(x.shape[1]) if coefficients is None else np.asarray(coefficients, dtype=float)
    if beta.shape != (x.shape[1],):
        raise ValueError("coefficients must have one value per covariate")
    scale = np.nanstd(x, axis=0)
    scale[scale == 0] = 1.0
    z = np.nan_to_num((x - np.nanmean(x, axis=0)) / scale)
    score = z @ beta
    w = np.ones(len(p)) if weights is None else np.asarray(weights, dtype=float)
    if np.any(w < 0) or not np.any(w > 0):
        raise ValueError("weights must be non-negative with positive total")
    target = float(np.average(p, weights=w))
    if target in (0.0, 1.0):
        return np.full(len(p), target)
    logit_target = np.log(target / (1 - target))

    def mean_for(intercept: float) -> float:
        logits = np.clip(logit_target + intercept + score, -35, 35)
        return float(np.average(1 / (1 + np.exp(-logits)), weights=w))

    lo, hi = -40.0, 40.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if mean_for(mid) < target:
            lo = mid
        else:
            hi = mid
    logits = np.clip(logit_target + (lo + hi) / 2 + score, -35, 35)
    return 1 / (1 + np.exp(-logits))


def real_income_after_price_shock(
    nominal_income: np.ndarray | pd.Series,
    expenditure_shares: pd.DataFrame,
    price_changes: pd.Series,
) -> np.ndarray:
    """Return nominal income divided by a household-specific price index.

    Shares are expenditure weights and should sum to at most one; unmodelled
    expenditure is assigned zero price change. This is a price-channel
    accounting interface, not a welfare model or a demand-estimation result.
    """
    income = np.asarray(nominal_income, dtype=float)
    shares = expenditure_shares.astype(float).reindex(columns=price_changes.index, fill_value=0.0)
    if len(shares) != len(income):
        raise ValueError("nominal_income and expenditure_shares must have equal length")
    if (shares < 0).any().any() or (price_changes < -1).any():
        raise ValueError("shares must be non-negative and price changes above -100%")
    index = 1.0 + shares.to_numpy() @ price_changes.to_numpy(dtype=float)
    return income / index
