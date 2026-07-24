"""Generate LaTeX macros for the paper's central numerical results.

The paper must never silently mix Monte Carlo runs with different draw counts.
This script reads the eight central artifacts, checks that they share the
declared production draw count, and writes a small generated LaTeX file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CENTRAL = (
    "full_tariff_displacement",
    "full_tariff_wage_cut",
    "full_tariff_inactivity",
    "epd_displacement",
    "epd_wage_cut",
    "epd_inactivity",
    "measured_displacement",
    "measured_wage_cut",
)


def _load(results_dir: Path, name: str) -> dict:
    path = results_dir / f"{name}.json"
    with path.open() as f:
        return json.load(f)


def _fmt(value: float, scale: float = 1.0, digits: int = 1) -> str:
    return f"{value * scale:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--output", type=Path, default=Path("paper/generated_results.tex")
    )
    parser.add_argument("--expected-draws", type=int, default=100)
    args = parser.parse_args()

    data = {name: _load(args.results_dir, name) for name in CENTRAL}
    draw_counts = {name: item["n_draws"] for name, item in data.items()}
    if set(draw_counts.values()) != {args.expected_draws}:
        details = ", ".join(f"{k}={v}" for k, v in draw_counts.items())
        raise ValueError(
            f"Central artifacts must all use {args.expected_draws} draws: {details}"
        )

    fd = data["full_tariff_displacement"]
    fw = data["full_tariff_wage_cut"]
    fi = data["full_tariff_inactivity"]
    ed = data["epd_displacement"]
    ew = data["epd_wage_cut"]
    ei = data["epd_inactivity"]
    md = data["measured_displacement"]
    mw = data["measured_wage_cut"]
    full_draws = fd["draws"]
    epd_draws = ed["draws"]
    if len(full_draws) != len(epd_draws):
        raise ValueError("Full-tariff and EPD displacement draws must be paired")
    paired_workers = [
        full["displaced_weighted"] - epd["displaced_weighted"]
        for full, epd in zip(full_draws, epd_draws, strict=True)
    ]
    paired_gross = [
        full["gross_earnings_loss"] - epd["gross_earnings_loss"]
        for full, epd in zip(full_draws, epd_draws, strict=True)
    ]
    paired_exchequer = [
        full["exchequer_cost"] - epd["exchequer_cost"]
        for full, epd in zip(full_draws, epd_draws, strict=True)
    ]
    paired_poverty = [
        full["poverty_rate_change_bhc"] - epd["poverty_rate_change_bhc"]
        for full, epd in zip(full_draws, epd_draws, strict=True)
    ]

    macros = {
        "ProductionDraws": str(args.expected_draws),
        "FullDisplacedWorkers": _fmt(fd["displaced_weighted_mean"], 1 / 1_000, 1),
        "FullDisplacedGross": _fmt(
            sum(d["gross_earnings_loss"] for d in fd["draws"]) / len(fd["draws"]),
            1 / 1e9,
            3,
        ),
        "FullDisplacedGrossSD": _fmt(
            _sample_sd([d["gross_earnings_loss"] for d in fd["draws"]]),
            1 / 1e9,
            3,
        ),
        "FullDisplacedExchequer": _fmt(fd["exchequer_cost_mean"], 1 / 1e9, 3),
        "FullDisplacedExchequerSD": _fmt(fd["exchequer_cost_sd"], 1 / 1e9, 3),
        "FullDisplacedExchequerMCSE": _fmt(
            fd["exchequer_cost_mc_se"], 1 / 1e9, 3
        ),
        "FullDisplacedCushion": _fmt(fd["cushioning_rate_mean"], 100, 1),
        "FullDisplacedCushionSD": _fmt(fd["cushioning_rate_sd"], 100, 1),
        "FullDisplacedCushionMCSE": _fmt(
            fd["cushioning_rate_mc_se"], 100, 1
        ),
        "FullDisplacedPoverty": _fmt(fd["poverty_rate_change_bhc_mean"], 100, 3),
        "FullDisplacedPovertySD": _fmt(fd["poverty_rate_change_bhc_sd"], 100, 3),
        "FullWageExchequer": _fmt(fw["exchequer_cost_mean"], 1 / 1e9, 3),
        "FullWageCushion": _fmt(fw["cushioning_rate_mean"], 100, 1),
        "FullInactiveExchequer": _fmt(fi["exchequer_cost_mean"], 1 / 1e9, 3),
        "FullInactiveExchequerSD": _fmt(fi["exchequer_cost_sd"], 1 / 1e9, 3),
        "FullInactiveCushion": _fmt(fi["cushioning_rate_mean"], 100, 1),
        "FullInactiveCushionSD": _fmt(fi["cushioning_rate_sd"], 100, 1),
        "EPDDisplacedWorkers": _fmt(ed["displaced_weighted_mean"], 1 / 1_000, 1),
        "EPDDisplacedGross": _fmt(
            _mean([d["gross_earnings_loss"] for d in ed["draws"]]), 1 / 1e9, 3
        ),
        "EPDDisplacedGrossSD": _fmt(
            _sample_sd([d["gross_earnings_loss"] for d in ed["draws"]]),
            1 / 1e9,
            3,
        ),
        "EPDDisplacedExchequer": _fmt(ed["exchequer_cost_mean"], 1 / 1e9, 3),
        "EPDDisplacedExchequerSD": _fmt(ed["exchequer_cost_sd"], 1 / 1e9, 3),
        "EPDDisplacedCushion": _fmt(ed["cushioning_rate_mean"], 100, 1),
        "EPDDisplacedCushionSD": _fmt(ed["cushioning_rate_sd"], 100, 1),
        "EPDDisplacedPoverty": _fmt(ed["poverty_rate_change_bhc_mean"], 100, 3),
        "EPDDisplacedPovertySD": _fmt(ed["poverty_rate_change_bhc_sd"], 100, 3),
        "EPDWageExchequer": _fmt(ew["exchequer_cost_mean"], 1 / 1e9, 3),
        "EPDWageCushion": _fmt(ew["cushioning_rate_mean"], 100, 1),
        "EPDInactiveExchequer": _fmt(ei["exchequer_cost_mean"], 1 / 1e9, 3),
        "EPDInactiveExchequerSD": _fmt(ei["exchequer_cost_sd"], 1 / 1e9, 3),
        "EPDInactiveCushion": _fmt(ei["cushioning_rate_mean"], 100, 1),
        "EPDInactiveCushionSD": _fmt(ei["cushioning_rate_sd"], 100, 1),
        "MeasuredDisplacedWorkers": _fmt(md["displaced_weighted_mean"], 1 / 1_000, 1),
        "MeasuredDisplacedExchequer": _fmt(md["exchequer_cost_mean"], 1 / 1e9, 3),
        "MeasuredDisplacedExchequerSD": _fmt(md["exchequer_cost_sd"], 1 / 1e9, 3),
        "MeasuredDisplacedCushion": _fmt(md["cushioning_rate_mean"], 100, 1),
        "MeasuredDisplacedCushionSD": _fmt(md["cushioning_rate_sd"], 100, 1),
        "MeasuredWageExchequer": _fmt(mw["exchequer_cost_mean"], 1 / 1e9, 3),
        "MeasuredWageCushion": _fmt(mw["cushioning_rate_mean"], 100, 1),
        "EPDWorkerDifference": _fmt(_mean(paired_workers), 1.0, 0),
        "EPDWorkerDifferenceSD": _fmt(_sample_sd(paired_workers), 1.0, 0),
        "EPDGrossDifference": _fmt(_mean(paired_gross), 1 / 1e9, 3),
        "EPDGrossDifferenceSD": _fmt(_sample_sd(paired_gross), 1 / 1e9, 3),
        "EPDExchequerDifference": _fmt(_mean(paired_exchequer), 1 / 1e9, 3),
        "EPDExchequerDifferenceSD": _fmt(_sample_sd(paired_exchequer), 1 / 1e9, 3),
        "EPDPovertyDifference": _fmt(_mean(paired_poverty), 100, 4),
        "EPDPovertyDifferenceSD": _fmt(_sample_sd(paired_poverty), 100, 4),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    lines = ["% Generated by analysis/write_paper_results.py; do not edit by hand."]
    lines.extend(
        f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in macros.items()
    )
    args.output.write_text("\n".join(lines) + "\n")


def _sample_sd(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((value - mean) ** 2 for value in values) / (len(values) - 1)) ** 0.5


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


if __name__ == "__main__":
    main()
