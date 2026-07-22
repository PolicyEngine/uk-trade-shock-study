"""Supply-chain amplification scenario (appendix): run s^total = s^direct + s^upstream.

Leontief upstream extension (uk_trade_shock_study/supply_chain.py): the
direct export-demand falls are propagated through the ONS 2022 domestic
product-by-product IO tables, each supplying division's output fall is
converted to an earnings-shock rate, and the full-tariff displacement and
wage-cut margins are rerun on the TOTAL shock. Writes:

  results/supply_chain_shocks.json        amplification factor, top upstream
                                          divisions, per-division shock table
  results/supply_chain_displacement.json  Monte Carlo displacement results
  results/supply_chain_wage_cut.json      wage cut with seeded UC take-up draws
  results/supply_chain_cushioning_seed0.json  seed-0 cushioning accounting
                                          (household-weighted net loss)

Usage: .venv/bin/python analysis/supply_chain_scenario.py
       [--data-dir data] [--period 2026] [--n-draws 10]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from uk_trade_shock_study import supply_chain as sc
from uk_trade_shock_study.runner import (
    MonteCarloResult,
    ScenarioResult,
    _baseline_and_persons,
    _metrics,
    write_result,
)
from uk_trade_shock_study.shocks import build_shocked_simulation

TARIFF = "full_tariff"


def _one_draw(dataset, baseline, persons, shock, period, seed, margin) -> ScenarioResult:
    if margin == "wage_cut":
        shocked_table = sc.apply_wage_cut_with_shock(persons, shock)
    else:
        shocked_table = sc.apply_displacement_with_shock(persons, shock, seed=seed)
    shocked = build_shocked_simulation(dataset, baseline, shocked_table, period)
    base, shk = _metrics(baseline, period), _metrics(shocked, period)

    weight = persons["weight"].to_numpy()
    displaced = shocked_table["displaced"].to_numpy()
    income_delta = shk["hni"] - base["hni"]
    equiv = baseline.calculate(
        "equiv_hbai_household_net_income", period=period, map_to="person"
    ).values
    order = np.argsort(equiv)
    cum = np.cumsum(weight[order])
    ranks = np.empty(len(equiv), dtype=float)
    ranks[order] = cum / cum[-1]
    deciles = np.clip(np.ceil(ranks * 10).astype(int), 1, 10)
    decile_change = {
        int(d): float(np.average(income_delta[deciles == d], weights=weight[deciles == d]))
        for d in range(1, 11)
    }
    return ScenarioResult(
        scenario=f"supply_chain_{margin}",
        tariff_scenario=TARIFF,
        margin=margin,
        exchequer_cost=base["gov_balance"] - shk["gov_balance"],
        poverty_rate_change_bhc=shk["poverty_bhc"] - base["poverty_bhc"],
        poverty_rate_change_ahc=shk["poverty_ahc"] - base["poverty_ahc"],
        absolute_poverty_rate_change_bhc=shk["abs_poverty_bhc"] - base["abs_poverty_bhc"],
        gini_baseline=base["gini"],
        gini_shocked=shk["gini"],
        displaced_weighted=float(weight[displaced].sum()),
        inactive_weighted=0.0,
        decile_income_change=decile_change,
    )


def cushioning_seed0(dataset, baseline, persons, shock, period) -> dict:
    """1 - net/gross at seed 0, mirroring results/cushioning_seed0.json.

    Net loss is the HOUSEHOLD-weighted fall in HBAI disposable income
    (mapping the household variable to persons broadcasts it to every
    member, which would overcount by average household size).
    """
    hw = baseline.calculate("household_weight", period=period, map_to="household").values
    base_hh = np.asarray(
        baseline.calculate("hbai_household_net_income", period=period, map_to="household").values,
        float,
    )
    weight = persons["weight"].to_numpy()
    out = {}
    for margin in ("displacement", "wage_cut"):
        if margin == "wage_cut":
            tbl = sc.apply_wage_cut_with_shock(persons, shock)
        else:
            tbl = sc.apply_displacement_with_shock(persons, shock, seed=0)
        sim = build_shocked_simulation(dataset, baseline, tbl, period)
        shocked_hh = np.asarray(
            sim.calculate("hbai_household_net_income", period=period, map_to="household").values,
            float,
        )
        gross = float(
            (
                (persons["employment_income"].to_numpy() - tbl["employment_income"].to_numpy())
                * weight
            ).sum()
        )
        net = float(((base_hh - shocked_hh) * hw).sum())
        out[f"supply_chain_{margin}"] = {
            "gross_earnings_loss": gross,
            "net_disposable_loss": net,
            "cushioning_rate": 1.0 - net / gross,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--period", type=int, default=2026)
    parser.add_argument("--n-draws", type=int, default=10)
    args = parser.parse_args()

    results = Path("results")
    results.mkdir(exist_ok=True)

    summary = sc.amplification_summary(TARIFF)
    shocks_table = sc.total_sector_shocks(TARIFF)
    print(
        f"[supply chain] IO-accounts amplification factor "
        f"{summary['amplification_factor']:.2f} "
        f"(direct £{summary['direct_earnings_loss_gbp_m'] / 1e3:.2f}bn, "
        f"upstream £{summary['upstream_earnings_loss_gbp_m'] / 1e3:.2f}bn)"
    )
    (results / "supply_chain_shocks.json").write_text(
        json.dumps(
            {**summary, "sector_shocks": shocks_table.round(6).to_dict(orient="index")},
            indent=2,
        )
    )

    dataset_path = Path(args.data_dir) / "frs_2024_25.h5"
    dataset, baseline, persons = _baseline_and_persons(dataset_path, None, args.period)
    shock = sc.person_total_shock(persons["sic_division"], TARIFF, shocks=shocks_table)
    emp = persons["employment_income"].to_numpy()
    w = persons["weight"].to_numpy()
    print(
        f"[supply chain] FRS aggregate gross earnings loss with s^total: "
        f"£{float((shock * emp * w).sum()) / 1e9:.2f}bn/yr"
    )

    cushioning = cushioning_seed0(dataset, baseline, persons, shock, args.period)
    (results / "supply_chain_cushioning_seed0.json").write_text(json.dumps(cushioning, indent=2))
    for name, rec in cushioning.items():
        print(
            f"{name}: gross £{rec['gross_earnings_loss'] / 1e9:.2f}bn, "
            f"net £{rec['net_disposable_loss'] / 1e9:.2f}bn, "
            f"cushioning {rec['cushioning_rate'] * 100:.1f}%"
        )

    for margin in ("displacement", "wage_cut"):
        n = args.n_draws
        draws = [
            _one_draw(dataset, baseline, persons, shock, args.period, seed, margin)
            for seed in range(n)
        ]
        cost = np.array([d.exchequer_cost for d in draws])
        pov = np.array([d.poverty_rate_change_bhc for d in draws])
        gini_change = np.array([d.gini_shocked - d.gini_baseline for d in draws])
        result = MonteCarloResult(
            scenario=f"supply_chain_{margin}",
            n_draws=n,
            exchequer_cost_mean=float(cost.mean()),
            exchequer_cost_sd=float(cost.std(ddof=1)) if n > 1 else 0.0,
            poverty_rate_change_bhc_mean=float(pov.mean()),
            poverty_rate_change_bhc_sd=float(pov.std(ddof=1)) if n > 1 else 0.0,
            gini_change_mean=float(gini_change.mean()),
            gini_change_sd=float(gini_change.std(ddof=1)) if n > 1 else 0.0,
            displaced_weighted_mean=float(np.mean([d.displaced_weighted for d in draws])),
            draws=[asdict(d) for d in draws],
        )
        write_result(result, results / f"supply_chain_{margin}.json")
        print(
            f"supply_chain_{margin}: exchequer £{result.exchequer_cost_mean / 1e9:.2f}bn "
            f"(SD {result.exchequer_cost_sd / 1e9:.2f}), "
            f"poverty BHC {result.poverty_rate_change_bhc_mean * 100:+.3f}pp, "
            f"displaced {result.displaced_weighted_mean / 1e3:.1f}k"
        )


if __name__ == "__main__":
    main()
