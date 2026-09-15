"""Transparent policy-counterfactual accounting helpers.

These functions calculate gross programme cost and household income support
before tax-benefit interactions. They are intended as inputs to a full
PolicyEngine rerun, not as standalone welfare estimates.
"""

from __future__ import annotations

import numpy as np


def wage_insurance_payment(
    earnings_loss: np.ndarray,
    replacement_rate: float,
    duration_months: float = 12.0,
) -> np.ndarray:
    """Payment replacing a share of an annual earnings loss."""
    loss = np.asarray(earnings_loss, dtype=float)
    if np.any(loss < 0) or not 0 <= replacement_rate <= 1:
        raise ValueError("loss must be non-negative and replacement_rate in [0, 1]")
    if not 0 <= duration_months <= 12:
        raise ValueError("duration_months must lie in [0, 12]")
    return loss * replacement_rate * duration_months / 12.0


def targeted_transfer(
    household_income: np.ndarray,
    threshold: float,
    amount: float,
) -> np.ndarray:
    """Annual transfer for households below an explicit income threshold."""
    income = np.asarray(household_income, dtype=float)
    if threshold < 0 or amount < 0:
        raise ValueError("threshold and amount must be non-negative")
    return np.where(income < threshold, amount, 0.0)
