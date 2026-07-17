"""Run all tariff/margin presets and write results to results/.

Usage: python analysis/run_scenarios.py [--data-dir DATA] [--period 2026]
       [--n-draws 20] [--scenarios full_tariff_displacement ...]
"""

import argparse
from pathlib import Path

from uk_trade_shock_study.runner import run_monte_carlo, write_result
from uk_trade_shock_study.shocks import PRESETS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--period", type=int, default=2026)
    parser.add_argument("--n-draws", type=int, default=20)
    parser.add_argument("--scenarios", nargs="*", default=list(PRESETS))
    args = parser.parse_args()

    data = Path(args.data_dir)
    dataset = data / "frs_2024_25.h5"
    # SIC comes from the h5's sic_industry_division variable; pass
    # adult_tab_path=... to run_monte_carlo to use the legacy adult.tab join.
    results = Path("results")
    results.mkdir(exist_ok=True)

    # print the calibrated aggregate gross earnings loss per tariff scenario
    from policyengine_uk import Microsimulation
    from policyengine_uk.data import UKSingleYearDataset

    import pandas as pd

    from uk_trade_shock_study.exposure import (
        sector_earnings_shocks,
        simulation_sic_division,
    )

    sim = Microsimulation(dataset=UKSingleYearDataset(file_path=str(dataset)))
    sic = simulation_sic_division(sim, args.period)
    emp = sim.calculate("employment_income", period=args.period, map_to="person").values
    w = sim.calculate("person_weight", period=args.period, map_to="person").values
    for tariff in ("full_tariff", "epd"):
        shock = pd.Series(sic).map(sector_earnings_shocks(tariff)).fillna(0.0).to_numpy()
        loss = float((shock * emp * w).sum())
        print(f"[calibration] {tariff}: aggregate gross earnings loss £{loss / 1e9:.2f}bn/yr")

    for name in args.scenarios:
        result = run_monte_carlo(
            dataset, name, period=args.period, n_draws=args.n_draws
        )
        write_result(result, results / f"{name}.json")
        print(
            f"{name}: exchequer £{result.exchequer_cost_mean/1e9:.2f}bn "
            f"(SD {result.exchequer_cost_sd/1e9:.2f}), "
            f"poverty BHC {result.poverty_rate_change_bhc_mean*100:+.2f}pp, "
            f"gini {result.gini_change_mean:+.4f}"
        )


if __name__ == "__main__":
    main()
