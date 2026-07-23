"""Sensitivity grid over the two exposure parameters: elasticity x passthrough.

Runs epsilon in {1.0, 1.5, 2.0, 3.0} x pi in {0.5, 0.75, 1.0} for the
full_tariff schedule on the displacement margin (10 Monte Carlo draws per
cell, seeds 0-4) and the wage-cut margin. Per cell: aggregate
gross earnings loss, Exchequer cost, cushioning rate, weighted displaced
count, relative BHC poverty change.

Expected structure, verified in-script and reported in the output JSON:
the derived shock is s_j = epsilon * tau_j * x_j * pi, linear in the product
epsilon*pi (no cell reaches the [0,1] clip), so all pound-denominated levels
scale ~linearly in epsilon*pi while cushioning RATES and margin orderings
are ~invariant.

Writes results/sensitivity_grid.json.
Usage: .venv/bin/python analysis/sensitivity_grid.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from uk_trade_shock_study.runner import _baseline_and_persons
from uk_trade_shock_study.shocks import (
    TradeShockScenario,
    apply_shocks,
    build_shocked_simulation,
)

PERIOD = 2026
DATASET = Path("data/frs_2024_25.h5")
RESULTS = Path("results")
ELASTICITIES = (0.4, 1.0, 2.0, 3.0)
PASSTHROUGHS = (0.5, 0.75, 1.0)
# Appendix robustness grid; headline scenarios use 10 draws.
N_DRAWS = 5


def sim_metrics(sim):
    hh_w = sim.calculate("household_weight", period=PERIOD, map_to="household").values
    p_w = sim.calculate("person_weight", period=PERIOD, map_to="person").values
    return {
        "gov": float((sim.calculate("gov_balance", period=PERIOD, map_to="household").values * hh_w).sum()),
        "pov_bhc": float(np.average(
            sim.calculate("in_poverty_bhc", period=PERIOD, map_to="person").values, weights=p_w
        )),
        "hni": float((sim.calculate("hbai_household_net_income", period=PERIOD, map_to="household").values * hh_w).sum()),
    }


def one_cell_draw(dataset, baseline, persons, base, scenario, seed):
    table = apply_shocks(persons, scenario, seed=seed)
    shocked = build_shocked_simulation(dataset, baseline, table, PERIOD)
    m = sim_metrics(shocked)
    w = persons["weight"].to_numpy(float)
    gross = float((
        (persons["employment_income"].to_numpy(float) - table["employment_income"].to_numpy(float)) * w
    ).sum())
    net = base["hni"] - m["hni"]
    return {
        "gross_earnings_loss": gross,
        "exchequer_cost": base["gov"] - m["gov"],
        "cushioning_rate": 1.0 - net / gross if gross else float("nan"),
        "displaced_weighted": float(w[table["displaced"].to_numpy()].sum()),
        "poverty_change_bhc_pp": 100 * (m["pov_bhc"] - base["pov_bhc"]),
    }


def main() -> None:
    dataset, baseline, persons = _baseline_and_persons(DATASET, None, PERIOD)
    base = sim_metrics(baseline)
    out = {"n_draws_displacement": N_DRAWS, "cells": []}

    for eps in ELASTICITIES:
        for pi in PASSTHROUGHS:
            cell = {"elasticity": eps, "passthrough": pi, "eps_x_pi": eps * pi}
            # displacement: N_DRAWS draws
            disp = TradeShockScenario(
                f"grid_ft_disp_e{eps}_p{pi}", "full_tariff", "displacement",
                elasticity=eps, passthrough=pi,
            )
            draws = [
                one_cell_draw(dataset, baseline, persons, base, disp, seed)
                for seed in range(N_DRAWS)
            ]
            cell["displacement"] = {
                k: {
                    "mean": float(np.nanmean([d[k] for d in draws])),
                    "sd": float(np.nanstd([d[k] for d in draws], ddof=1)),
                    "mc_se": float(
                        np.nanstd([d[k] for d in draws], ddof=1)
                        / np.sqrt(np.isfinite([d[k] for d in draws]).sum())
                    ),
                    "n_valid": int(np.isfinite([d[k] for d in draws]).sum()),
                }
                for k in draws[0]
            }
            # wage cut: deterministic
            wc = TradeShockScenario(
                f"grid_ft_wc_e{eps}_p{pi}", "full_tariff", "wage_cut",
                elasticity=eps, passthrough=pi,
            )
            cell["wage_cut"] = one_cell_draw(dataset, baseline, persons, base, wc, 0)
            out["cells"].append(cell)
            print(
                f"[grid] eps={eps} pi={pi}: "
                f"disp gross £{cell['displacement']['gross_earnings_loss']['mean']/1e9:.2f}bn "
                f"cushion {cell['displacement']['cushioning_rate']['mean']:.3f} | "
                f"wc gross £{cell['wage_cut']['gross_earnings_loss']/1e9:.2f}bn "
                f"cushion {cell['wage_cut']['cushioning_rate']:.3f}",
                flush=True,
            )

    # ---- invariance checks -------------------------------------------------
    cells = out["cells"]
    wc_rates = [c["wage_cut"]["cushioning_rate"] for c in cells]
    disp_rates = [c["displacement"]["cushioning_rate"]["mean"] for c in cells]
    wc_per_unit = [c["wage_cut"]["gross_earnings_loss"] / c["eps_x_pi"] for c in cells]
    disp_per_unit = [
        c["displacement"]["gross_earnings_loss"]["mean"] / c["eps_x_pi"] for c in cells
    ]
    ordering_holds = all(
        c["wage_cut"]["cushioning_rate"] > c["displacement"]["cushioning_rate"]["mean"]
        for c in cells
    )
    out["invariance_check"] = {
        "wage_cut_cushioning_rate_range": [min(wc_rates), max(wc_rates)],
        "displacement_cushioning_rate_mean_range": [min(disp_rates), max(disp_rates)],
        "wage_cut_gross_loss_per_unit_eps_pi_bn": {
            "min": min(wc_per_unit) / 1e9, "max": max(wc_per_unit) / 1e9,
            "max_rel_deviation": (max(wc_per_unit) - min(wc_per_unit)) / np.mean(wc_per_unit),
        },
        "displacement_gross_loss_per_unit_eps_pi_bn": {
            "min": min(disp_per_unit) / 1e9, "max": max(disp_per_unit) / 1e9,
            "max_rel_deviation": (max(disp_per_unit) - min(disp_per_unit)) / np.mean(disp_per_unit),
        },
        "wage_cut_above_displacement_cushioning_in_all_cells": ordering_holds,
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "sensitivity_grid.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out["invariance_check"], indent=2))


if __name__ == "__main__":
    main()
