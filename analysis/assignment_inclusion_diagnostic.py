"""Per-record empirical inclusion probabilities: balanced versus Bernoulli.

Why this exists. The primary displacement estimator is a BALANCED repeated
assignment: it generates systematic candidates and retains the one that best
matches the expected industry wage-bill and weighted-headcount targets. That
conditioning is what makes the common aggregate loss comparable draw by draw,
but it necessarily distorts record-level marginal inclusion probabilities away
from the declared per-record risk ``s_j``. The manuscript says so, and the
Bernoulli comparator (which preserves the declared first-order probabilities
exactly) is reported alongside it -- but until now nothing measured HOW FAR the
balanced design's realised inclusion frequencies drift from ``s_j``.

That matters here more than it usually would. The unit displacement loss rests
on roughly six effective record contributions and its largest single record
supplies close to a third of the loss on average, so a reader is entitled to
ask whether that record is drawn at its nominal rate or systematically over- or
under-selected by the balancing score. This script answers that question
directly: it runs both designs over a common set of seeds, accumulates each
exposed record's realised selection frequency, and compares it with the
declared probability.

Outputs ``results/assignment_inclusion_diagnostic.json``:

- ``per_record``: for the highest-leverage exposed records (ranked by expected
  weighted loss), the declared probability, the realised frequency under each
  design, and the ratio between them.
- ``summary``: weighted mean absolute deviation from the declared probability
  under each design, and the maximum single-record deviation. Under Bernoulli
  these should be pure sampling noise, shrinking like 1/sqrt(n_draws); under
  the balanced design any systematic component persists as n_draws grows,
  which is exactly the quantity the manuscript cannot currently report.

Requires the licensed FRS input, so it cannot run in the public CI contract.
``make assignment-inclusion`` runs it when the data is present.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uk_trade_shock_study.runner import _baseline_and_persons  # noqa: E402
from uk_trade_shock_study.shocks import (  # noqa: E402
    TradeShockScenario,
    _person_shock,
    draw_displaced,
)

DATASET = Path("data/frs_2024_25.h5")
PERIOD = 2026
OUTPUT = Path("results/assignment_inclusion_diagnostic.json")
DESIGNS = ("balanced", "bernoulli")


def inclusion_frequencies(
    persons,
    scenario: TradeShockScenario,
    n_draws: int,
) -> np.ndarray:
    """Realised per-record selection frequency over ``n_draws`` seeds.

    Seeds are the plain integers 0..n_draws-1, matching every other family in
    this project, so the frequencies are directly comparable with the stored
    scenario runs rather than being a fresh randomisation.
    """
    if n_draws < 1:
        raise ValueError("n_draws must be at least 1")
    counts = np.zeros(len(persons), dtype=float)
    for seed in range(n_draws):
        counts += draw_displaced(persons, scenario, seed=seed).astype(float)
    return counts / n_draws


def summarise(
    declared: np.ndarray,
    realised: np.ndarray,
    weight: np.ndarray,
    exposed: np.ndarray,
) -> dict:
    """Deviation of realised frequencies from the declared probabilities.

    Weighted by survey weight, because a drift on a record carrying 19 thousand
    people matters and a drift on a record carrying 200 does not.
    """
    dev = np.abs(realised - declared)[exposed]
    w = weight[exposed]
    if not w.sum():
        raise RuntimeError("no exposed records carry positive survey weight")
    return {
        "weighted_mean_absolute_deviation": float(np.average(dev, weights=w)),
        "max_absolute_deviation": float(dev.max()),
        "max_absolute_deviation_declared_probability": float(
            declared[exposed][int(dev.argmax())]
        ),
        "n_exposed_records": int(exposed.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--period", type=int, default=PERIOD)
    parser.add_argument("--n-draws", type=int, default=200)
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    _, _, persons = _baseline_and_persons(args.dataset, None, args.period)
    base = TradeShockScenario(
        "inclusion_diagnostic_unit_12m",
        "full_tariff",
        "displacement",
        elasticity=1.0,
        duration_equivalent=1.0,
        selection_method="balanced",
    )

    declared = np.asarray(_person_shock(persons, base), dtype=float)
    weight = persons["weight"].to_numpy(dtype=float)
    earnings = persons["employment_income"].to_numpy(dtype=float)
    exposed = (declared > 0) & (earnings > 0)
    if not exposed.any():
        raise RuntimeError("no exposed employee records; check the SIC join")

    realised = {
        design: inclusion_frequencies(
            persons, replace(base, selection_method=design), args.n_draws
        )
        for design in DESIGNS
    }

    # Leverage = expected weighted loss, i.e. what actually drives the headline.
    leverage = declared * weight * earnings
    order = np.argsort(leverage)[::-1]
    top = [int(i) for i in order[: args.top] if exposed[i]]

    out = {
        "design": {
            "scenario": base.name,
            "n_draws": args.n_draws,
            "period": args.period,
            "designs": list(DESIGNS),
            "seeds": "0..n_draws-1, matching the stored scenario families",
            "notes": (
                "Balanced repeated assignment conditions on aggregate wage-bill "
                "and headcount targets, so it does not preserve declared "
                "record-level marginal inclusion probabilities; Bernoulli does. "
                "Deviations under Bernoulli are sampling noise and shrink like "
                "1/sqrt(n_draws); any systematic component of the balanced "
                "deviations persists as n_draws grows."
            ),
        },
        "summary": {
            design: summarise(declared, realised[design], weight, exposed)
            for design in DESIGNS
        },
        "per_record": [
            {
                "rank_by_expected_weighted_loss": rank,
                "survey_weight": float(weight[i]),
                "employment_income": float(earnings[i]),
                "expected_weighted_loss": float(leverage[i]),
                "declared_probability": float(declared[i]),
                **{
                    f"realised_frequency_{design}": float(realised[design][i])
                    for design in DESIGNS
                },
                **{
                    f"ratio_{design}": (
                        float(realised[design][i] / declared[i])
                        if declared[i] > 0
                        else None
                    )
                    for design in DESIGNS
                },
            }
            for rank, i in enumerate(top, start=1)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(f"[written] {args.output}")
    for design in DESIGNS:
        s = out["summary"][design]
        print(
            f"  {design:9s} weighted MAD "
            f"{s['weighted_mean_absolute_deviation']:.5f}, "
            f"max {s['max_absolute_deviation']:.5f}"
        )


if __name__ == "__main__":
    main()
