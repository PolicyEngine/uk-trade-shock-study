"""Run the compact submission scenario design.

The design crosses two declared loss anchors (OBR-style elasticity 0.4 and
unit stress), two adjustment margins, and three annual duration-equivalent
scales. Displacement uses repeated balanced systematic assignment within
industries. A Bernoulli 12-month comparator quantifies the precision gain and
any mean shift from the assignment design.

Three- and six-month rows are annual duration-equivalent stresses: the model
assigns a correspondingly smaller probability of a coherent full-year
displaced state. They are not simulated partial-year unemployment spells.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from uk_trade_shock_study.exposure import ELASTICITY_SCENARIOS
from uk_trade_shock_study.runner import run_monte_carlo, write_result
from uk_trade_shock_study.shocks import TradeShockScenario


ANCHORS = {
    "obr": ELASTICITY_SCENARIOS["obr_low"],
    "unit": 1.0,
}
DURATIONS = {3: 0.25, 6: 0.5, 12: 1.0}


def scenarios() -> dict[str, TradeShockScenario]:
    result = {}
    for anchor, elasticity in ANCHORS.items():
        for months, scale in DURATIONS.items():
            for margin in ("displacement", "wage_cut"):
                name = f"submission_{anchor}_{months}m_{margin}"
                result[name] = TradeShockScenario(
                    name,
                    "full_tariff",
                    margin,
                    elasticity=elasticity,
                    duration_equivalent=scale,
                    selection_method="balanced",
                )
        comparator = f"submission_{anchor}_12m_displacement_bernoulli"
        result[comparator] = TradeShockScenario(
            comparator,
            "full_tariff",
            "displacement",
            elasticity=elasticity,
            selection_method="bernoulli",
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/frs_2024_25.h5"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/submission"))
    parser.add_argument("--period", type=int, default=2026)
    parser.add_argument("--n-draws", type=int, default=100)
    parser.add_argument("--scenarios", nargs="*")
    args = parser.parse_args()

    declared = scenarios()
    selected = args.scenarios or list(declared)
    unknown = sorted(set(selected) - set(declared))
    if unknown:
        raise ValueError(f"unknown submission scenarios: {unknown}")
    args.results_dir.mkdir(parents=True, exist_ok=True)
    for name in selected:
        result = run_monte_carlo(
            args.dataset,
            declared[name],
            period=args.period,
            n_draws=args.n_draws,
        )
        write_result(result, args.results_dir / f"{name}.json")
        print(
            f"{name}: gross="
            f"{sum(d['gross_earnings_loss'] for d in result.draws) / args.n_draws / 1e6:.0f}m; "
            f"Exchequer={result.exchequer_cost_mean / 1e6:.0f}m; "
            f"cushion={result.cushioning_rate_mean * 100:.1f}%; "
            f"poverty={result.poverty_rate_change_bhc_mean * 100:+.3f}pp",
            flush=True,
        )


if __name__ == "__main__":
    main()
