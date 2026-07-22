"""Sensitivity of the headline results to POST-SHOCK Universal Credit take-up.

``would_claim_uc`` is a stored FRS input drawn at dataset build time to hit a
population-wide flag share of 0.55 over ALL benunits (not take-up among the
entitled) and conditioned on PRE-SHOCK circumstances. The central runs re-draw
it post-shock for affected benunits at ``uc_takeup`` = 0.80
(shocks.DEFAULT_UC_TAKEUP). This script traces every headline quantity over

    stale_baseline_flag   pre-fix behaviour: post-shock flags restored to the
                          baseline stored draw (a newly displaced family is
                          modelled as not claiming because it was not claiming
                          while in work; measured take-up among the displaced
                          0.469)
    0.55, 0.70, 0.80, 0.90, 1.00   take-up among NEWLY ENTITLED benunits

for both tariff schedules (full_tariff, epd) and both headline margins
(displacement, wage_cut). Both margins use the same new-entitlement rule and
are evaluated over the same seed count.

Reports per cell: cushioning rate, Exchequer cost, BHC/AHC poverty change,
Gini change, and the component decomposition of the cushion (income tax,
employee NI, UC, other benefits, residual).

Writes results/takeup_sensitivity.json.
Usage: .venv/bin/python analysis/takeup_sensitivity.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mechanism_decomposition import cushioning_components  # noqa: E402

from uk_trade_shock_study.runner import _baseline_and_persons, gini  # noqa: E402
from uk_trade_shock_study.shocks import (  # noqa: E402
    TradeShockScenario,
    apply_shocks,
    build_shocked_simulation,
)

PERIOD = 2026
DATASET = Path("data/frs_2024_25.h5")
RESULTS = Path("results")
# This is a robustness grid rather than the headline Monte Carlo.  Five
# common-seed assignment draws per cell keep the 24-cell exercise tractable;
# the resulting SDs describe assignment dispersion, not sampling precision.
N_DRAWS = 5
TAKEUPS = (0.55, 0.70, 0.80, 0.90, 1.00)
STALE = "stale_baseline_flag"


def sim_metrics(sim):
    hh_w = sim.calculate("household_weight", period=PERIOD, map_to="household").values
    p_w = sim.calculate("person_weight", period=PERIOD, map_to="person").values
    hh_n = sim.calculate("household_count_people", period=PERIOD, map_to="household").values
    equiv = sim.calculate(
        "equiv_hbai_household_net_income", period=PERIOD, map_to="household"
    ).values
    return {
        "gov": float(
            (sim.calculate("gov_balance", period=PERIOD, map_to="household").values * hh_w).sum()
        ),
        "pov_bhc": float(
            np.average(
                sim.calculate("in_poverty_bhc", period=PERIOD, map_to="person").values,
                weights=p_w,
            )
        ),
        "pov_ahc": float(
            np.average(
                sim.calculate("in_poverty_ahc", period=PERIOD, map_to="person").values,
                weights=p_w,
            )
        ),
        "gini": gini(equiv, hh_w * hh_n),
    }


def summarise(rows) -> dict:
    keys = rows[0].keys()
    return {
        k: {
            "mean": float(np.mean([r[k] for r in rows])),
            "sd": float(np.std([r[k] for r in rows], ddof=1)) if len(rows) > 1 else 0.0,
        }
        for k in keys
    }


def main() -> None:
    dataset, baseline, persons = _baseline_and_persons(DATASET, None, PERIOD)
    w = persons["weight"].to_numpy(float)
    base = sim_metrics(baseline)
    baseline_flag = np.asarray(
        baseline.calculate("would_claim_uc", period=PERIOD, map_to="benunit").values,
        dtype=bool,
    )

    out = {
        "n_draws_displacement": N_DRAWS,
        "takeups": list(TAKEUPS),
        "notes": {
            "stale_baseline_flag": "pre-fix behaviour: post-shock would_claim_uc "
            "restored to the baseline stored draw.",
            "wage_cut": "earnings cuts are deterministic, but newly entitled "
            "benefit units receive seeded claiming draws under the common rule.",
            "baseline_flag_rate_all_benunits": float(baseline_flag.mean()),
        },
    }

    for tariff in ("full_tariff", "epd"):
        for margin in ("displacement", "wage_cut"):
            n = N_DRAWS
            for label in (STALE, *TAKEUPS):
                takeup = 0.80 if label == STALE else float(label)
                scen = TradeShockScenario(
                    f"{tariff}_{margin}", tariff, margin, uc_takeup=takeup
                )
                rows, comps = [], []
                for seed in range(n):
                    table = apply_shocks(persons, scen, seed=seed)
                    sim = build_shocked_simulation(dataset, baseline, table, PERIOD)
                    if label == STALE:
                        sim.set_input("would_claim_uc", PERIOD, baseline_flag)
                    m = sim_metrics(sim)
                    comp = cushioning_components(baseline, sim, persons, table, PERIOD)
                    comps.append(comp)
                    rows.append(
                        {
                            "cushioning_rate": comp["cushioning_rate"],
                            "exchequer_cost_bn": (base["gov"] - m["gov"]) / 1e9,
                            "poverty_change_bhc_pp": 100 * (m["pov_bhc"] - base["pov_bhc"]),
                            "poverty_change_ahc_pp": 100 * (m["pov_ahc"] - base["pov_ahc"]),
                            "gini_change": m["gini"] - base["gini"],
                            "gross_earnings_loss_bn": comp["gross_earnings_loss"] / 1e9,
                            "displaced_weighted": float(
                                w[table["displaced"].to_numpy()].sum()
                            ),
                        }
                    )
                    del sim
                key = f"{tariff}_{margin}"
                block = out.setdefault(key, {})
                block[str(label)] = summarise(rows)
                block[str(label)]["components_share_of_gross_loss"] = {
                    c: float(
                        np.mean([x["components_share_of_gross_loss"][c] for x in comps])
                    )
                    for c in comps[0]["components_share_of_gross_loss"]
                }
                block[str(label)]["components_gbp"] = {
                    c: float(np.mean([x["components_gbp"][c] for x in comps]))
                    for c in comps[0]["components_gbp"]
                }
                print(
                    f"[{key}] {label}: cushioning "
                    f"{block[str(label)]['cushioning_rate']['mean']:.4f}",
                    flush=True,
                )

    # Record, rather than assume, the wage-cut sensitivity to take-up.
    for tariff in ("full_tariff", "epd"):
        vals = {
            k: v["cushioning_rate"]["mean"]
            for k, v in out[f"{tariff}_wage_cut"].items()
        }
        spread = max(vals.values()) - min(vals.values())
        out["notes"][f"{tariff}_wage_cut_takeup_max_spread"] = spread

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "takeup_sensitivity.json").write_text(json.dumps(out, indent=2))
    print("[written] results/takeup_sensitivity.json")


if __name__ == "__main__":
    main()
