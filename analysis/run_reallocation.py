"""Run the REALLOCATION margin (literal sectoral switch into services).

Writes:
  results/full_tariff_reallocation.json   instant reallocation, 20 draws
  results/epd_reallocation.json           instant reallocation, 20 draws
  results/full_tariff_reallocation_lag3.json   3-month lag variant
  results/epd_reallocation_lag3.json           3-month lag variant
  results/reallocation_cushioning.json    four-way margin comparison with
      paired-draw statistics against the displacement family

Seeds 0..19 are shared with the displacement runs, so the reallocation and
displacement draws are PAIRED (identical worker sets); the comparison block
reports paired mean differences and their standard errors.

Usage: .venv/bin/python analysis/run_reallocation.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from uk_trade_shock_study.runner import run_monte_carlo, write_result
from uk_trade_shock_study.shocks import (
    DEFAULT_REALLOCATION_PENALTY,
    HOURLY_REALLOCATION_PENALTY,
    TradeShockScenario,
)

PERIOD = 2026
N_DRAWS = 20  # reallocation family + paired comparison runs
DATASET = Path("data/frs_2024_25.h5")
RESULTS = Path("results")


def scenarios():
    for tariff in ("full_tariff", "epd"):
        yield f"{tariff}_reallocation", TradeShockScenario(
            f"{tariff}_reallocation", tariff, "reallocation"
        )
        yield f"{tariff}_reallocation_lag3", TradeShockScenario(
            f"{tariff}_reallocation_lag3", tariff, "reallocation",
            reallocation_lag_months=3.0,
        )
        yield f"{tariff}_reallocation_lowpenalty", TradeShockScenario(
            f"{tariff}_reallocation_lowpenalty", tariff, "reallocation",
            reallocation_penalty=HOURLY_REALLOCATION_PENALTY,
        )


def summarise(result) -> dict:
    draws = result.draws
    return {
        "n_draws": result.n_draws,
        "exchequer_cost_mean": result.exchequer_cost_mean,
        "exchequer_cost_sd": result.exchequer_cost_sd,
        "poverty_bhc_mean": result.poverty_rate_change_bhc_mean,
        "poverty_bhc_sd": result.poverty_rate_change_bhc_sd,
        "poverty_ahc_mean": float(np.mean([d["poverty_rate_change_ahc"] for d in draws])),
        "gini_change_mean": result.gini_change_mean,
        "gini_change_sd": result.gini_change_sd,
        "cushioning_rate_mean": result.cushioning_rate_mean,
        "cushioning_rate_sd": result.cushioning_rate_sd,
        "gross_earnings_loss_mean": float(np.mean([d["gross_earnings_loss"] for d in draws])),
        "net_disposable_loss_mean": float(np.mean([d["net_disposable_loss"] for d in draws])),
        "affected_weighted_mean": float(
            np.mean([max(d["displaced_weighted"], d["reallocated_weighted"]) for d in draws])
        ),
    }


def paired(a, b, key) -> dict:
    """Paired-draw difference a - b over the shared seeds (same draw index)."""
    xa = np.array([d[key] for d in a.draws], dtype=float)
    xb = np.array([d[key] for d in b.draws], dtype=float)
    n = min(len(xa), len(xb))
    if n < 2 or len(xa) != len(xb):
        return {"mean_difference": float(xa[:n].mean() - xb[:n].mean()), "paired": False}
    diff = xa - xb
    return {
        "mean_difference": float(diff.mean()),
        "sd_of_paired_difference": float(diff.std(ddof=1)),
        "se_of_paired_difference": float(diff.std(ddof=1) / np.sqrt(n)),
        "paired": True,
        "n_pairs": int(n),
    }


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    runs = {}
    for name, scenario in scenarios():
        result = run_monte_carlo(DATASET, scenario, period=PERIOD, n_draws=N_DRAWS)
        write_result(result, RESULTS / f"{name}.json")
        runs[name] = result
        print(f"[written] {name}")

    # comparison margins (re-run so every family is on the SAME shared seeds)
    for tariff in ("full_tariff", "epd"):
        for margin in ("displacement", "inactivity", "wage_cut"):
            name = f"{tariff}_{margin}"
            runs[name] = run_monte_carlo(
                DATASET,
                TradeShockScenario(name, tariff, margin),
                period=PERIOD,
                n_draws=N_DRAWS,
            )
            print(f"[compared] {name}")

    comparison = {
        "calibration": {
            "reallocation_penalty": DEFAULT_REALLOCATION_PENALTY,
            "reallocation_penalty_lower_bound": HOURLY_REALLOCATION_PENALTY,
            "source": "FRS 2024-25 weighted mean annual employee earnings, "
            "exposed goods divisions vs the four services destinations "
            "(analysis/reallocation_calibration.py); lower bound is the "
            "hours/age-controlled log gap.",
        },
        "n_draws": N_DRAWS,
        "seeds": list(range(N_DRAWS)),
    }
    for tariff in ("full_tariff", "epd"):
        block = {
            name.replace(f"{tariff}_", ""): summarise(runs[name])
            for name in runs
            if name.startswith(tariff)
        }
        base = runs[f"{tariff}_displacement"]
        block["paired_vs_displacement"] = {
            name.replace(f"{tariff}_", ""): {
                key: paired(runs[name], base, key)
                for key in ("exchequer_cost", "poverty_rate_change_bhc", "cushioning_rate")
            }
            for name in runs
            if name.startswith(tariff) and not name.endswith("displacement")
        }
        comparison[tariff] = block

    (RESULTS / "reallocation_cushioning.json").write_text(json.dumps(comparison, indent=2))
    print(json.dumps(comparison["full_tariff"], indent=2)[:3000])


if __name__ == "__main__":
    main()
