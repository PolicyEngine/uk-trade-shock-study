"""Leave-one-record-out sensitivity of the primary cushioning contrast.

Referee major point 4: at the unit 12-month displacement stress the result
rests on roughly ten affected FRS records per assignment (about six
effective loss-contributing records), so one influential record could in
principle drive the headline contrast. This script quantifies that directly:
it removes each loss-contributing FRS household in turn — from BOTH margins,
since the same survey record underlies both counterfactuals — and recomputes
the draw-averaged wage-cut minus displacement cushioning contrast exactly,
using the stored per-household weighted contributions from the bootstrap
cache (results/bootstrap_contributions.npz, 25 common assignment draws).
No re-simulation is needed: all aggregates are additive over households.

Writes results/leave_one_record_out.json.

Usage: .venv/bin/python analysis/leave_one_record_out.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CACHE = Path("results/bootstrap_contributions.npz")
OUT = Path("results/leave_one_record_out.json")
MARGINS = ("wage_cut", "displacement")


def load_contributions() -> dict[str, dict[str, np.ndarray]]:
    stored = np.load(CACHE)
    return {
        margin: {key: stored[f"{margin}_{key}"] for key in ("gross", "net")}
        for margin in MARGINS
    }


def contrast(contributions, exclude: int | None = None) -> float:
    """Draw-averaged cushioning contrast with one household removed."""
    out = {}
    for margin in MARGINS:
        gross = contributions[margin]["gross"].sum(axis=1)
        net = contributions[margin]["net"].sum(axis=1)
        if exclude is not None:
            gross = gross - contributions[margin]["gross"][:, exclude]
            net = net - contributions[margin]["net"][:, exclude]
        with np.errstate(divide="ignore", invalid="ignore"):
            cushioning = np.where(gross > 0, 1.0 - net / gross, np.nan)
        out[margin] = float(np.nanmean(cushioning))
    return out["wage_cut"] - out["displacement"]


def main() -> None:
    contributions = load_contributions()
    point = contrast(contributions)
    # Candidates: every household that contributes a positive weighted gross
    # earnings loss to ANY displacement draw — the sparse-support set the
    # referee is worried about.
    disp_gross = contributions["displacement"]["gross"]
    candidates = np.flatnonzero((disp_gross > 0).any(axis=0))
    loo = {int(h): contrast(contributions, exclude=int(h)) for h in candidates}
    values = np.asarray(list(loo.values()))
    most_influential = max(loo, key=lambda h: abs(loo[h] - point))
    out = {
        "design": {
            "source": str(CACHE),
            "n_draws": int(disp_gross.shape[0]),
            "n_households": int(disp_gross.shape[1]),
            "n_loss_contributing_households": int(candidates.size),
            "notes": (
                "Each loss-contributing FRS household is removed from both "
                "margins and the draw-averaged unit 12-month wage-cut minus "
                "displacement cushioning contrast is recomputed exactly from "
                "stored per-household contributions. This is a record-"
                "influence diagnostic, not a sampling-variance estimate."
            ),
        },
        "point_contrast_pp": point * 100,
        "loo_min_pp": float(values.min() * 100),
        "loo_max_pp": float(values.max() * 100),
        "loo_mean_abs_shift_pp": float(np.abs(values - point).mean() * 100),
        "most_influential_household": {
            "index": int(most_influential),
            "contrast_without_pp": loo[most_influential] * 100,
            "shift_pp": (loo[most_influential] - point) * 100,
        },
        "all_positive": bool((values > 0).all()),
        "per_household_contrast_pp": {
            str(h): v * 100 for h, v in sorted(loo.items())
        },
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(
        f"point {point * 100:.2f}pp; LOO range "
        f"[{values.min() * 100:.2f}, {values.max() * 100:.2f}]pp over "
        f"{candidates.size} households; all positive: {(values > 0).all()}"
    )


if __name__ == "__main__":
    main()
