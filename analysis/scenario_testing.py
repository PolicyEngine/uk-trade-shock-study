"""Factorial tariff stress testing, following the UK AI study architecture.

The grid separates two assumptions that should not be bundled:

1. aggregate sector-shock calibration (export-demand elasticity); and
2. adjustment composition (share delivered through displacement rather than
   survivor wage cuts).

Every cell uses the full-tariff schedule, common seeds, the common UC rule and
the same sector exposure profile. Five assignments per cell make this an
exploratory scenario surface, not an uncertainty interval.

Outputs:
  results/scenario_testing.csv
  results/scenario_testing.json
  results/figures/scenario_testing.png
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from uk_trade_shock_study.runner import _baseline_and_persons, _one_draw
from uk_trade_shock_study.shocks import TradeShockScenario

PERIOD = 2026
DATASET = Path("data/frs_2024_25.h5")
OUT = Path("results")
FIGURES = OUT / "figures"
ELASTICITIES = (0.4, 1.0, 2.0, 3.0)
DISPLACEMENT_SHARES = (0.0, 0.25, 0.5, 0.75, 1.0)
N_DRAWS = 5


def _mean_sd(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=1))


def _json_value(value):
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def main() -> None:
    dataset, baseline, persons = _baseline_and_persons(DATASET, None, PERIOD)
    rows = []
    draws_out = []
    for elasticity in ELASTICITIES:
        for share in DISPLACEMENT_SHARES:
            scenario = TradeShockScenario(
                name=f"scenario_e{elasticity:g}_d{share:g}",
                tariff_scenario="full_tariff",
                margin="mixed",
                elasticity=elasticity,
                displacement_share=share,
            )
            draws = [
                _one_draw(dataset, baseline, persons, scenario, PERIOD, seed)
                for seed in range(N_DRAWS)
            ]
            gross_m, gross_sd = _mean_sd([d.gross_earnings_loss / 1e9 for d in draws])
            ex_m, ex_sd = _mean_sd([d.exchequer_cost / 1e9 for d in draws])
            cush_m, cush_sd = _mean_sd([100 * d.cushioning_rate for d in draws])
            pov_m, pov_sd = _mean_sd([100 * d.poverty_rate_change_bhc for d in draws])
            disp_m, disp_sd = _mean_sd([d.displaced_weighted / 1e3 for d in draws])
            row = {
                "elasticity": elasticity,
                "displacement_share": share,
                "n_assignments": N_DRAWS,
                "gross_loss_bn_mean": gross_m,
                "gross_loss_bn_sd": gross_sd,
                "exchequer_cost_bn_mean": ex_m,
                "exchequer_cost_bn_sd": ex_sd,
                "cushioning_pct_mean": cush_m,
                "cushioning_pct_sd": cush_sd,
                "poverty_bhc_pp_mean": pov_m,
                "poverty_bhc_pp_sd": pov_sd,
                "displaced_thousands_mean": disp_m,
                "displaced_thousands_sd": disp_sd,
            }
            rows.append(row)
            draws_out.extend(
                {
                    "elasticity": elasticity,
                    "displacement_share": share,
                    "seed": seed,
                    **_json_value(asdict(draw)),
                }
                for seed, draw in enumerate(draws)
            )
            print(
                f"elasticity={elasticity:g}, displacement={share:.0%}: "
                f"gross £{gross_m:.2f}bn, fiscal £{ex_m:.2f}bn, "
                f"cushioning {cush_m:.1f}%",
                flush=True,
            )

    frame = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / "scenario_testing.csv", index=False)
    payload = {
        "design": {
            "tariff_schedule": "full_tariff",
            "elasticities": list(ELASTICITIES),
            "displacement_shares": list(DISPLACEMENT_SHARES),
            "n_assignments_per_cell": N_DRAWS,
            "interpretation": "exploratory factorial stress test; assignment SD, not inference",
        },
        "cells": rows,
        "draws": draws_out,
    }
    (OUT / "scenario_testing.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False)
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), constrained_layout=True)
    panels = (
        ("exchequer_cost_bn_mean", "Annual Exchequer cost (£bn)", "Blues"),
        ("cushioning_pct_mean", "Tax–benefit cushioning (%)", "YlGnBu"),
    )
    for ax, (metric, title, cmap) in zip(axes, panels):
        matrix = frame.pivot(
            index="elasticity", columns="displacement_share", values=metric
        ).loc[list(ELASTICITIES), list(DISPLACEMENT_SHARES)]
        image = ax.imshow(matrix, aspect="auto", origin="lower", cmap=cmap)
        ax.set_xticks(range(len(DISPLACEMENT_SHARES)), [f"{100*x:.0f}" for x in DISPLACEMENT_SHARES])
        ax.set_yticks(range(len(ELASTICITIES)), [f"{x:g}" for x in ELASTICITIES])
        ax.set_xlabel("Share of shock delivered through displacement (%)")
        ax.set_ylabel("Export-demand calibration")
        ax.set_title(title)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix.iloc[i, j]
                ax.text(j, i, f"{value:.2f}" if metric.endswith("bn_mean") else f"{value:.1f}",
                        ha="center", va="center", fontsize=8)
        fig.colorbar(image, ax=ax, shrink=0.82)
    fig.savefig(FIGURES / "scenario_testing.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
