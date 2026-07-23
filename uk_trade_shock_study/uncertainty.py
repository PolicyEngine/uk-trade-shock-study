"""Structured parameter draws for scenario uncertainty analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd


def latin_hypercube(
    bounds: dict[str, tuple[float, float]], n_draws: int = 500, seed: int = 0
) -> pd.DataFrame:
    """Generate a reproducible Latin-hypercube parameter design.

    Bounds describe scenario uncertainty, not confidence intervals. Each
    column stratifies its interval once, while independent permutations avoid
    a Cartesian explosion of expensive microsimulation runs.
    """
    if n_draws <= 0 or not bounds:
        raise ValueError("n_draws and bounds must be non-empty")
    rng = np.random.default_rng(seed)
    out: dict[str, np.ndarray] = {}
    for name, (low, high) in bounds.items():
        if not np.isfinite(low) or not np.isfinite(high) or high < low:
            raise ValueError(f"invalid bounds for {name!r}")
        u = (np.arange(n_draws) + rng.random(n_draws)) / n_draws
        rng.shuffle(u)
        out[name] = low + u * (high - low)
    return pd.DataFrame(out)
