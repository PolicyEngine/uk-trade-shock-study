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

WHAT THE STATISTICS CAN AND CANNOT ESTABLISH. A realised frequency over
``n_draws`` seeds is a noisy estimate of a design's true marginal inclusion
probability, so every deviation from ``s_j`` mixes systematic drift with
sampling noise. The two summaries separate differently and must be read
differently:

- The SIGNED weighted mean deviation is unbiased for the systematic component:
  the noise enters with mean zero and averages away, so this statistic is
  centred on zero for a design without drift at ANY draw count, and its
  standard error (computed across draws, so it also carries the cross-record
  dependence the balancing induces) says how precisely it is measured. This is
  the statistic that can support a claim about drift.
- The ABSOLUTE weighted mean deviation cannot. ``E|noise|`` does not vanish,
  so the statistic is bounded below by a noise floor of order
  ``sqrt(2 s_j (1 - s_j) / (pi n_draws))`` per record -- which at the declared
  probabilities in this application is comparable with ``s_j`` itself at
  practical draw counts. It is reported alongside the analytic
  Bernoulli-equivalent floor for the same design and draw count precisely so
  that no reader mistakes the floor for drift: an absolute deviation at or
  below its floor is evidence of NOTHING.
- ``max_absolute_deviation`` is a maximum over records, so it grows with the
  record count and shrinks like ``1/sqrt(n_draws)`` whether or not any drift
  exists. It is retained as a scale check only and must not be compared across
  designs of different size.
- The balanced-minus-Bernoulli CONTRAST is taken draw by draw on common seeds,
  which differences out whatever component of the noise the two designs share,
  and is reported with both its paired and its unpaired standard error so the
  reader can see how much the pairing actually bought (on this pair, little:
  the two designs consume the same seed through different samplers).

A deviation is only interpretable against its own uncertainty, so the
per-record block carries a standard error for every realised frequency and
ratio rather than a bare ratio at full float precision.

Outputs ``results/assignment_inclusion_diagnostic.json``:

- ``per_record``: for the highest-leverage exposed records (ranked by expected
  weighted loss), the declared probability, the realised frequency under each
  design with its standard error, and the ratio between them with its own.
- ``summary``: for each design, the signed and absolute weighted mean
  deviations, their standard errors, the analytic Bernoulli noise floor for
  the same design and draw count, and the excess of the absolute deviation
  over that floor.
- ``contrast``: the paired balanced-minus-Bernoulli difference on common seeds.

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
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from uk_trade_shock_study.runner import _baseline_and_persons  # noqa: E402
from uk_trade_shock_study.shocks import (  # noqa: E402
    TradeShockScenario,
    _person_shock,
    draw_displaced,
)

DATASET = ROOT / "data/frs_2024_25.h5"
PERIOD = 2026
OUTPUT = ROOT / "results/assignment_inclusion_diagnostic.json"
DESIGNS = ("balanced", "bernoulli")


def inclusion_draws(
    persons,
    scenario: TradeShockScenario,
    n_draws: int,
) -> np.ndarray:
    """Per-draw selection indicators, shape ``(n_draws, len(persons))``.

    Seeds are the plain integers 0..n_draws-1, matching every other family in
    this project, so the frequencies are directly comparable with the stored
    scenario runs rather than being a fresh randomisation -- and so that the
    two designs can be contrasted draw by draw on COMMON seeds.

    The individual draws are kept (rather than only their running total)
    because every uncertainty statement below is computed across draws: an
    analytic per-record standard error would assume the records are
    independent, which is exactly what the balanced design is not.
    """
    if n_draws < 1:
        raise ValueError("n_draws must be at least 1")
    return np.vstack(
        [
            draw_displaced(persons, scenario, seed=seed).astype(bool)
            for seed in range(n_draws)
        ]
    )


def inclusion_frequencies(
    persons,
    scenario: TradeShockScenario,
    n_draws: int,
) -> np.ndarray:
    """Realised per-record selection frequency over ``n_draws`` seeds."""
    return inclusion_draws(persons, scenario, n_draws).mean(axis=0)


def binomial_absolute_deviation(probability: np.ndarray, n_draws: int) -> np.ndarray:
    """``E|X / n - p|`` for ``X ~ Binomial(n, p)``, exactly.

    De Moivre's mean absolute deviation, ``E|X - np| = 2 m (1 - p) P(X = m)``
    with ``m = floor(np) + 1``. Exact rather than the half-normal
    approximation because the declared probabilities here are small enough
    (``np`` below one for most records at the default draw count) for the
    normal approximation to misstate the floor.
    """
    if n_draws < 1:
        raise ValueError("n_draws must be at least 1")
    probability = np.asarray(probability, dtype=float)
    m = np.floor(n_draws * probability) + 1.0
    return (
        2.0 * m * (1.0 - probability) * stats.binom.pmf(m, n_draws, probability)
    ) / n_draws


def bernoulli_noise_floor(
    declared: np.ndarray,
    weight: np.ndarray,
    exposed: np.ndarray,
    n_draws: int,
) -> dict:
    """Sampling noise a DRIFT-FREE design of this shape cannot get below.

    ``absolute`` is the weighted mean of the per-record expected absolute
    deviation -- the value the absolute summary takes when the true marginal
    inclusion probabilities are exactly the declared ones. ``signed`` is the
    standard error of the weighted mean SIGNED deviation under independent
    Bernoulli sampling, i.e. the scale against which a signed deviation is
    read. The signed floor shrinks like ``1/sqrt(n_draws)`` and the statistic
    it bounds is centred on zero; the absolute floor also shrinks like
    ``1/sqrt(n_draws)`` but the statistic it bounds is not.
    """
    p = np.asarray(declared, dtype=float)[exposed]
    w = np.asarray(weight, dtype=float)[exposed]
    total = w.sum()
    if not total:
        raise RuntimeError("no exposed records carry positive survey weight")
    return {
        "absolute": float(np.average(binomial_absolute_deviation(p, n_draws), weights=w)),
        "signed_standard_error": float(
            np.sqrt((w**2 * p * (1.0 - p) / n_draws).sum()) / total
        ),
    }


def _weighted_draw_deviations(
    draws: np.ndarray,
    declared: np.ndarray,
    weight: np.ndarray,
    exposed: np.ndarray,
) -> np.ndarray:
    """Weighted mean signed deviation of each individual draw."""
    deviations = draws[:, exposed].astype(float) - np.asarray(declared, float)[exposed]
    w = np.asarray(weight, dtype=float)[exposed]
    return (deviations * w).sum(axis=1) / w.sum()


def summarise(
    declared: np.ndarray,
    realised: np.ndarray,
    weight: np.ndarray,
    exposed: np.ndarray,
    n_draws: int,
    draws: np.ndarray | None = None,
) -> dict:
    """Deviation of realised frequencies from the declared probabilities.

    Weighted by survey weight, because a drift on a record carrying 19 thousand
    people matters and a drift on a record carrying 200 does not.

    Reports the SIGNED deviation (unbiased for systematic drift, no noise
    floor) as the primary statistic and the ABSOLUTE deviation next to the
    analytic noise floor that bounds it from below, so the two can be told
    apart. Pass ``draws`` -- the per-draw indicator matrix that produced
    ``realised`` -- to get a standard error computed across draws, which holds
    whatever cross-record dependence the design induces; without it the
    reported standard error is the independent-Bernoulli one, which is right
    for the comparator and wrong for the balanced design in an unsignable
    direction (balancing on the weighted aggregate happens to REDUCE the
    across-draw variance of this particular statistic, but nothing guarantees
    that for another weighting).
    """
    declared = np.asarray(declared, dtype=float)
    realised = np.asarray(realised, dtype=float)
    signed = (realised - declared)[exposed]
    w = np.asarray(weight, dtype=float)[exposed]
    if not w.sum():
        raise RuntimeError("no exposed records carry positive survey weight")
    absolute = np.abs(signed)
    floor = bernoulli_noise_floor(declared, weight, exposed, n_draws)

    mean_signed = float(np.average(signed, weights=w))
    if draws is not None:
        per_draw = _weighted_draw_deviations(draws, declared, weight, exposed)
        if per_draw.shape != (n_draws,):
            raise ValueError("draws do not match the declared draw count")
        signed_standard_error = (
            float(per_draw.std(ddof=1) / np.sqrt(n_draws)) if n_draws > 1 else float("nan")
        )
        standard_error_basis = "across draws"
    else:
        signed_standard_error = floor["signed_standard_error"]
        standard_error_basis = "independent Bernoulli"

    mean_absolute = float(np.average(absolute, weights=w))
    per_record_se = np.sqrt(realised * (1.0 - realised) / max(n_draws - 1, 1))
    return {
        "n_draws": int(n_draws),
        "n_exposed_records": int(exposed.sum()),
        "weighted_mean_signed_deviation": mean_signed,
        "weighted_mean_signed_deviation_standard_error": signed_standard_error,
        "weighted_mean_signed_deviation_standard_error_basis": standard_error_basis,
        "weighted_mean_signed_deviation_t": (
            mean_signed / signed_standard_error if signed_standard_error else float("nan")
        ),
        "weighted_mean_absolute_deviation": mean_absolute,
        "bernoulli_absolute_deviation_noise_floor": floor["absolute"],
        "weighted_mean_absolute_deviation_excess_over_noise_floor": (
            mean_absolute - floor["absolute"]
        ),
        "bernoulli_signed_deviation_standard_error": floor["signed_standard_error"],
        # Retained as a scale check only: a maximum over records grows with the
        # record count and shrinks like 1/sqrt(n_draws) with or without drift.
        "max_absolute_deviation": float(absolute.max()),
        "max_absolute_deviation_declared_probability": float(
            declared[exposed][int(absolute.argmax())]
        ),
        "max_absolute_deviation_standard_error": float(
            per_record_se[exposed][int(absolute.argmax())]
        ),
        "mean_declared_probability": float(np.average(declared[exposed], weights=w)),
    }


def common_seed_contrast(
    declared: np.ndarray,
    weight: np.ndarray,
    exposed: np.ndarray,
    draws_by_design: dict,
    treatment: str = "balanced",
    comparator: str = "bernoulli",
) -> dict:
    """Paired ``treatment - comparator`` deviation, seed by seed.

    Both designs are run on the same seeds, so differencing them draw by draw
    removes whatever component of the sampling noise they share. The paired
    and unpaired standard errors are BOTH reported because how much that buys
    is an empirical matter and, on this pair, not much: a common seed does not
    mean common noise here, since the balanced design consumes the random
    stream through a different sampler than the Bernoulli comparator. The
    contrast is still the right object to quote -- it is the balanced design's
    drift measured against a comparator known to have none -- but its
    precision must be read off ``paired_standard_error``, not assumed.
    """
    per_draw = {
        design: _weighted_draw_deviations(draws, declared, weight, exposed)
        for design, draws in draws_by_design.items()
    }
    difference = per_draw[treatment] - per_draw[comparator]
    n_draws = difference.size
    standard_error = (
        float(difference.std(ddof=1) / np.sqrt(n_draws)) if n_draws > 1 else float("nan")
    )
    mean = float(difference.mean())
    return {
        "treatment": treatment,
        "comparator": comparator,
        "n_draws": int(n_draws),
        "paired_mean_signed_deviation_difference": mean,
        "paired_standard_error": standard_error,
        "paired_t": mean / standard_error if standard_error else float("nan"),
        "unpaired_standard_error": float(
            np.sqrt(
                per_draw[treatment].var(ddof=1) / n_draws
                + per_draw[comparator].var(ddof=1) / n_draws
            )
        )
        if n_draws > 1
        else float("nan"),
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

    draws = {
        design: inclusion_draws(
            persons, replace(base, selection_method=design), args.n_draws
        )
        for design in DESIGNS
    }
    realised = {design: matrix.mean(axis=0) for design, matrix in draws.items()}
    per_record_se = {
        design: np.sqrt(
            realised[design] * (1.0 - realised[design]) / max(args.n_draws - 1, 1)
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
                "Read the SIGNED deviation against its standard error: it is "
                "unbiased for the systematic component under both designs. The "
                "ABSOLUTE deviation cannot be read that way -- it has a "
                "sampling-noise floor that does not average away, reported "
                "here as bernoulli_absolute_deviation_noise_floor for the same "
                "design and draw count, and an absolute deviation at or below "
                "that floor establishes nothing. max_absolute_deviation is a "
                "maximum over records and scales with the record count and "
                "1/sqrt(n_draws) rather than with drift."
            ),
        },
        "summary": {
            design: summarise(
                declared,
                realised[design],
                weight,
                exposed,
                args.n_draws,
                draws=draws[design],
            )
            for design in DESIGNS
        },
        "contrast": common_seed_contrast(declared, weight, exposed, draws),
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
                    f"realised_frequency_standard_error_{design}": float(
                        per_record_se[design][i]
                    )
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
                **{
                    f"ratio_standard_error_{design}": (
                        float(per_record_se[design][i] / declared[i])
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
            f"  {design:9s} signed "
            f"{s['weighted_mean_signed_deviation']:+.5f} "
            f"(se {s['weighted_mean_signed_deviation_standard_error']:.5f}), "
            f"absolute {s['weighted_mean_absolute_deviation']:.5f} "
            f"(noise floor {s['bernoulli_absolute_deviation_noise_floor']:.5f})"
        )
    contrast = out["contrast"]
    print(
        "  contrast  balanced - bernoulli "
        f"{contrast['paired_mean_signed_deviation_difference']:+.5f} "
        f"(paired se {contrast['paired_standard_error']:.5f})"
    )


if __name__ == "__main__":
    main()
