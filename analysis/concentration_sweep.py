"""Continuous sweep of cushioning against the CONCENTRATION of a fixed loss.

WHY THIS EXISTS (referee objection to the factorial design). The manuscript
compares three points: a diffuse proportional wage cut spread over every
exposed FRS employee record, a "concentrated wage cut" that imposes a
displacement draw's exact worker-level losses on the roughly ten drawn
records, and displacement of those same records. Concentration, however, is
CONTINUOUS, and the two-point-plus-middle-cell design confounds it with WHICH
workers are selected: once the aggregate is fixed and concentrated on ten
drawn records, those records lose 100 per cent of earnings BY CONSTRUCTION.
There is no free variation left to trace the shape of the relationship.

THE DESIGN HERE. Hold each division's aggregate expected wage-bill loss fixed
at its declared earnings shock ``s_j`` (uk_trade_shock_study.exposure), and
deliver it as a proportional cut of fraction ``phi`` imposed on a randomly
selected share ``s_j / phi`` of that division's exposed employees. Then

    E[loss per exposed employee] = (s_j / phi) x phi x earnings = s_j x earnings

for EVERY phi: the expected wage-bill loss of every division is invariant
along the sweep, and only the concentration of that loss varies. Sweeping
phi from s_j (everyone takes a small cut) to 1.0 (a few workers lose
everything) therefore traces cushioning against concentration continuously
with the aggregate, the sector composition and the adjustment margin
(employment state: EMPLOYED throughout) all held fixed.

ENDPOINT EQUIVALENCE — the sweep nests the paper's existing cells.

- At any phi <= min_j s_j every division has selection probability one, so
  every exposed employee takes a cut of exactly s_j: this reproduces
  ``shocks.apply_wage_cut`` (the diffuse cell) record for record.
- At phi = 1.0 the selection probability is exactly s_j, which is the
  Bernoulli displacement probability, and selected workers lose all of their
  earnings while staying EMPLOYED: under ``selection_method="bernoulli"``
  this reproduces ``shocks.apply_concentrated_wage_cut`` at the same seed,
  bit for bit, because both consume ``np.random.default_rng(seed).random(n)``
  against the same probability vector.

Both equivalences are asserted in tests/test_concentration_sweep.py. The
sweep is therefore a continuous interpolation THROUGH the paper's two
wage-cut cells, not a separate exercise; the displacement cell sits off this
line by exactly the employment-state step of the factorial decomposition.

CLIPPING, AND DIVISIONS WHOSE s_j IS ALREADY LARGE. Two clips are needed and
both are one-sided:

- phi is clipped ABOVE at 1.0 — a worker cannot lose more than all earnings;
- the selection probability s_j / phi is clipped ABOVE at 1.0 — a division
  cannot select more than all of its employees.

The second clip binds exactly when phi < s_j. Clipping the probability alone
would silently under-deliver that division's aggregate loss (it would impose
phi < s_j on everyone), which would break the invariance the whole exercise
rests on. We therefore raise phi to s_j in those divisions instead:
``phi_effective_j = clip(phi, s_j, 1.0)`` and ``q_j = s_j / phi_effective_j``,
so ``q_j x phi_effective_j == s_j`` EXACTLY at every grid point and the
aggregate is preserved by construction. The practical consequence is that at
grid points below a division's own s_j that division sits at its diffuse
endpoint and has not yet begun to concentrate: with the packaged intensity
build the exposed s_j span roughly 0.0002 to 0.027, so the lowest grid points
concentrate the small-shock divisions while the large-shock divisions (autos,
machinery) are still fully diffuse. Each point's payload records
``n_divisions_clipped`` and the per-division schedule so this is visible
rather than implicit. An alternative parametrisation that avoids the clip
entirely is phi_j(t) = s_j^(1-t) for t in [0, 1], which moves every division
along its own range in lockstep; it is not used here because it makes the
x-axis a unitless index rather than a loss fraction a reader can interpret.

WHAT VARIES AND WHAT DOES NOT. Nobody's employment state changes at any
point on the sweep (``displaced``/``inactive``/``lcwra`` are all False), so
the sweep isolates concentration from the employment-state channel. The
post-shock UC take-up re-draw is applied by the shared
``build_shocked_simulation``, exactly as for every other family.

Seeds are plain integers 0..n_seeds-1, matching every other scenario family.

Writes results/concentration_sweep.json. Needs the licensed FRS microdata
and policyengine-uk.

Usage: .venv/bin/python analysis/concentration_sweep.py [--phi-points 13]
       [--n-seeds 5] [--selection-method balanced] [--elasticity 1.0]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from uk_trade_shock_study.exposure import DEFAULT_ELASTICITY
from uk_trade_shock_study.runner import _baseline_and_persons, _metrics
from uk_trade_shock_study.shocks import (
    DEFAULT_UC_TAKEUP,
    DEFAULT_UC_TAKEUP_SCOPE,
    TradeShockScenario,
    _person_shock,
    balanced_probability_sample,
    build_shocked_simulation,
    systematic_probability_sample,
)

PERIOD = 2026
DATASET = Path("data/frs_2024_25.h5")
OUTPUT = Path("results/concentration_sweep.json")

#: Default grid size. Geometric, because concentration is multiplicative: a
#: linear grid over [s_min, 1] would put almost every point in the fully
#: concentrated region and none in the range the paper's diffuse cell lives in.
DEFAULT_PHI_POINTS = 13
DEFAULT_N_SEEDS = 5


def sweep_scenario(
    tariff_scenario: str = "full_tariff",
    elasticity: float = DEFAULT_ELASTICITY,
    selection_method: str = "balanced",
    uc_takeup: float = DEFAULT_UC_TAKEUP,
    uc_takeup_scope: str = DEFAULT_UC_TAKEUP_SCOPE,
) -> TradeShockScenario:
    """Scenario carrying the shock calibration used along the sweep.

    Only the fields that feed ``_person_shock`` (tariff scenario, elasticity,
    wage-bill incidence, duration equivalent, excluded divisions) and the
    selection/take-up conventions are used: the sweep supplies its own
    application path, so the ``margin`` label is nominal. It is set to
    ``concentrated_wage_cut`` because that is the cell the sweep generalises.
    """
    return TradeShockScenario(
        name=f"concentration_sweep_{tariff_scenario}",
        tariff_scenario=tariff_scenario,
        margin="concentrated_wage_cut",
        elasticity=elasticity,
        selection_method=selection_method,
        uc_takeup=uc_takeup,
        uc_takeup_scope=uc_takeup_scope,
    )


def concentration_schedule(
    shock: np.ndarray, phi: float
) -> tuple[np.ndarray, np.ndarray]:
    """Per-person (loss fraction, selection probability) at concentration phi.

    ``phi`` is clipped above at 1.0 and below, per person, at that person's
    own sector shock; the selection probability is then s / phi_effective,
    which lies in (0, 1] by construction. Their product is exactly the sector
    shock, so the expected wage-bill loss is invariant in phi (see the module
    docstring on why the lower clip is applied to phi and not to the
    probability). Unexposed persons (s = 0) get probability 0 and never
    contribute a loss.
    """
    shock = np.asarray(shock, dtype=float)
    if not np.isfinite(shock).all() or (shock < 0).any() or (shock > 1).any():
        raise ValueError("sector shocks must be finite and lie in [0, 1]")
    if not 0.0 < phi <= 1.0 + 1e-12:
        raise ValueError("phi must lie in (0, 1]")
    phi_effective = np.clip(min(float(phi), 1.0), shock, 1.0)
    probability = np.divide(
        shock,
        phi_effective,
        out=np.zeros_like(shock),
        where=phi_effective > 0,
    )
    probability = np.minimum(probability, 1.0)
    return phi_effective, probability


def phi_grid(shock: np.ndarray, n_points: int = DEFAULT_PHI_POINTS) -> list[float]:
    """Geometric grid from the fully diffuse endpoint to full concentration.

    The first point is the smallest positive sector shock: at (and below) it
    EVERY division has selection probability one, so that point is the diffuse
    wage cut. The last point is 1.0, full concentration. Points in between
    concentrate the small-shock divisions first (see the module docstring).
    """
    shock = np.asarray(shock, dtype=float)
    positive = shock[shock > 0]
    if not positive.size:
        raise ValueError("no exposed records: every sector shock is zero")
    if n_points < 2:
        raise ValueError("phi grid needs at least the two endpoints")
    return [float(x) for x in np.geomspace(float(positive.min()), 1.0, n_points)]


def select_records(
    persons: pd.DataFrame,
    probabilities: np.ndarray,
    selection_method: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw the loss-bearing records, mirroring ``shocks.draw_displaced``.

    The same three assignment designs, the same shared samplers and the same
    "employees only" restriction as the job-loss margins; only the probability
    vector differs (s / phi_effective here, the sector shock there — the two
    coincide at phi = 1, which is why the fully concentrated endpoint
    reproduces the concentrated-wage-cut cell exactly under Bernoulli).
    """
    employed = persons["employment_income"].to_numpy(dtype=float) > 0
    probabilities = np.where(employed, np.asarray(probabilities, dtype=float), 0.0)
    if selection_method == "bernoulli":
        return employed & (rng.random(len(persons)) < probabilities)
    if selection_method == "systematic":
        sampled = systematic_probability_sample(
            probabilities,
            persons["sic_division"].to_numpy(),
            persons["age"].to_numpy(dtype=float),
            persons["employment_income"].to_numpy(dtype=float),
            rng,
        )
    elif selection_method == "balanced":
        # Balances the realised selected wage bill against the target
        # probability-weighted wage bill within division, which is exactly the
        # target aggregate loss divided by phi — so the balanced design
        # stabilises the imposed aggregate at every point of the sweep.
        sampled = balanced_probability_sample(probabilities, persons, rng)
    else:
        raise ValueError(
            "selection_method must be 'bernoulli', 'systematic' or 'balanced'"
        )
    return employed & sampled


def concentrated_cut_table(
    persons: pd.DataFrame,
    scenario: TradeShockScenario,
    phi: float,
    seed: int = 0,
) -> pd.DataFrame:
    """Shocked person table at concentration ``phi``: a partial wage cut.

    Selected employees lose the fraction ``phi_effective`` of annual earnings
    and NOBODY changes employment state, so the table follows the
    wage-cut contract that ``build_shocked_simulation`` expects
    (``displaced``/``inactive``/``lcwra`` all False, ``earnings_changed``
    marking the affected records for the UC take-up re-draw).

    This does not route through ``shocks.apply_wage_cut`` or
    ``supply_chain.apply_wage_cut_with_shock``: both reject a shock of 1.0
    ("derived sector shock >= 100% of earnings"), which is precisely the
    fully concentrated endpoint of this sweep, where a selected worker loses
    all earnings while remaining employed — the concentrated-wage-cut
    construction. The guards below are the same ones those functions apply.
    """
    shock = _person_shock(persons, scenario)
    phi_effective, probability = concentration_schedule(shock, phi)
    rng = np.random.default_rng(seed)
    selected = select_records(persons, probability, scenario.selection_method, rng)

    shocked = persons.copy()
    earnings = shocked["employment_income"].to_numpy(dtype=float)
    new = np.where(selected, earnings * (1.0 - phi_effective), earnings)
    if (new < 0).any() or (new > earnings + 1e-9).any():
        raise AssertionError("concentrated cut produced negative earnings or a raise")
    if not np.array_equal(new[~selected], earnings[~selected]):
        raise AssertionError("concentrated cut touched unselected records")
    shocked["employment_income"] = new
    shocked["displaced"] = np.zeros(len(persons), dtype=bool)
    shocked["inactive"] = np.zeros(len(persons), dtype=bool)
    shocked["lcwra"] = np.zeros(len(persons), dtype=bool)
    shocked["reallocated"] = np.zeros(len(persons), dtype=bool)
    shocked["destination_division"] = np.full(len(persons), np.nan)
    shocked["earnings_changed"] = selected
    # Same attrs contract as apply_shocks: build_shocked_simulation reads them
    # to re-draw would_claim_uc for affected benefit units.
    shocked.attrs["uc_takeup"] = float(scenario.uc_takeup)
    shocked.attrs["uc_takeup_scope"] = scenario.uc_takeup_scope
    shocked.attrs["seed"] = int(seed)
    return shocked


def expected_gross_loss(persons: pd.DataFrame, scenario: TradeShockScenario) -> float:
    """Weighted aggregate earnings loss the sweep holds fixed at every phi."""
    shock = _person_shock(persons, scenario)
    earnings = persons["employment_income"].to_numpy(dtype=float)
    weight = persons["weight"].to_numpy(dtype=float)
    return float((np.where(earnings > 0, shock, 0.0) * earnings * weight).sum())


def division_schedule(
    persons: pd.DataFrame, scenario: TradeShockScenario, phi: float
) -> dict:
    """Per-division realised phi and selection probability at this grid point."""
    shock = _person_shock(persons, scenario)
    phi_effective, probability = concentration_schedule(shock, phi)
    division = persons["sic_division"].to_numpy(dtype=float)
    employed = persons["employment_income"].to_numpy(dtype=float) > 0
    out = {}
    exposed = employed & (shock > 0) & np.isfinite(division)
    for code in np.unique(division[exposed]):
        rows = exposed & (division == code)
        out[str(int(code))] = {
            "sector_shock": float(shock[rows][0]),
            "phi_effective": float(phi_effective[rows][0]),
            "selection_probability": float(probability[rows][0]),
            "clipped": bool(phi_effective[rows][0] > phi + 1e-15),
            "exposed_records": int(rows.sum()),
        }
    return out


def loss_diagnostics(persons: pd.DataFrame, shocked_table: pd.DataFrame) -> dict:
    """Concentration diagnostics of the realised record-level losses.

    ``effective_loss_records`` is the inverse-Herfindahl count of weighted
    record losses used throughout the paper (see
    ``runner.ScenarioResult.effective_loss_records``): the number of equally
    sized records that would produce the same concentration. It is the
    sweep's dependent measure of concentration and falls from the full
    exposed record count at the diffuse endpoint to a handful at phi = 1.
    """
    weight = persons["weight"].to_numpy(dtype=float)
    record_loss = (
        persons["employment_income"].to_numpy(dtype=float)
        - shocked_table["employment_income"].to_numpy(dtype=float)
    ) * weight
    positive = record_loss > 0
    total = float(record_loss[positive].sum())
    squared = float(np.square(record_loss[positive]).sum())
    return {
        "gross_earnings_loss": total,
        "affected_records": int(positive.sum()),
        "affected_weighted": float(weight[positive].sum()),
        "effective_loss_records": (total**2 / squared) if squared > 0 else 0.0,
        "max_record_loss_share": (
            float(record_loss[positive].max() / total)
            if positive.any() and total > 0
            else 0.0
        ),
    }


def sweep_point_draw(
    dataset,
    baseline,
    persons: pd.DataFrame,
    scenario: TradeShockScenario,
    phi: float,
    seed: int,
    baseline_metrics: dict,
    period: int = PERIOD,
) -> dict:
    """One (phi, seed) draw: cushioning, Exchequer cost, concentration."""
    table = concentrated_cut_table(persons, scenario, phi, seed=seed)
    shocked = build_shocked_simulation(dataset, baseline, table, period)
    # Baseline-frozen relative poverty lines, as everywhere else (referee H1).
    shocked_metrics = _metrics(
        shocked,
        period,
        poverty_lines=(
            baseline_metrics["poverty_line_bhc"],
            baseline_metrics["poverty_line_ahc"],
        ),
    )
    diagnostics = loss_diagnostics(persons, table)
    gross = diagnostics["gross_earnings_loss"]
    net = baseline_metrics["hni_total"] - shocked_metrics["hni_total"]
    return {
        "seed": int(seed),
        "cushioning_rate": (1.0 - net / gross) if gross else float("nan"),
        "exchequer_cost": baseline_metrics["gov_balance"]
        - shocked_metrics["gov_balance"],
        "net_disposable_loss": net,
        "poverty_rate_change_bhc": (
            shocked_metrics["poverty_bhc"] - baseline_metrics["poverty_bhc"]
        ),
        **diagnostics,
    }


def _summarise(draws: list[dict], key: str) -> dict:
    values = np.asarray([d[key] for d in draws], dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return {"mean": None, "sd": None, "valid_draws": 0}
    return {
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "valid_draws": int(values.size),
    }


SUMMARY_KEYS = (
    "cushioning_rate",
    "exchequer_cost",
    "gross_earnings_loss",
    "net_disposable_loss",
    "effective_loss_records",
    "affected_records",
    "affected_weighted",
    "max_record_loss_share",
    "poverty_rate_change_bhc",
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--period", type=int, default=PERIOD)
    parser.add_argument("--tariff-scenario", default="full_tariff")
    parser.add_argument("--elasticity", type=float, default=DEFAULT_ELASTICITY)
    parser.add_argument(
        "--selection-method",
        default="balanced",
        choices=("balanced", "systematic", "bernoulli"),
        help=(
            "assignment design; 'balanced' matches the submission scenarios, "
            "'bernoulli' is the transparent comparator whose endpoints "
            "reproduce apply_wage_cut and apply_concentrated_wage_cut exactly"
        ),
    )
    parser.add_argument("--phi-points", type=int, default=DEFAULT_PHI_POINTS)
    parser.add_argument("--n-seeds", type=int, default=DEFAULT_N_SEEDS)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    scenario = sweep_scenario(
        tariff_scenario=args.tariff_scenario,
        elasticity=args.elasticity,
        selection_method=args.selection_method,
    )
    dataset, baseline, persons = _baseline_and_persons(
        args.dataset, None, args.period
    )
    baseline_metrics = _metrics(baseline, args.period)
    shock = _person_shock(persons, scenario)
    grid = phi_grid(shock, args.phi_points)
    target = expected_gross_loss(persons, scenario)

    points = []
    for phi in grid:
        schedule = division_schedule(persons, scenario, phi)
        draws = [
            sweep_point_draw(
                dataset,
                baseline,
                persons,
                scenario,
                phi,
                seed,
                baseline_metrics,
                args.period,
            )
            for seed in range(args.n_seeds)
        ]
        summary = {key: _summarise(draws, key) for key in SUMMARY_KEYS}
        points.append(
            {
                "phi": float(phi),
                "n_divisions": len(schedule),
                "n_divisions_clipped": sum(
                    1 for d in schedule.values() if d["clipped"]
                ),
                "expected_gross_earnings_loss": target,
                "realised_over_expected_loss": (
                    summary["gross_earnings_loss"]["mean"] / target
                    if target
                    else float("nan")
                ),
                "division_schedule": schedule,
                "summary": summary,
                "draws": draws,
            }
        )
        print(
            f"[concentration] phi={phi:.5f} "
            f"cushion={summary['cushioning_rate']['mean']:.4f} "
            f"eff_records={summary['effective_loss_records']['mean']:.1f} "
            f"cost={summary['exchequer_cost']['mean'] / 1e6:.1f}m "
            f"loss/target={points[-1]['realised_over_expected_loss']:.3f}",
            flush=True,
        )

    payload = {
        "design": {
            "period": args.period,
            "tariff_scenario": args.tariff_scenario,
            "elasticity": args.elasticity,
            "selection_method": args.selection_method,
            "uc_takeup": scenario.uc_takeup,
            "uc_takeup_scope": scenario.uc_takeup_scope,
            "n_seeds": args.n_seeds,
            "seeds": list(range(args.n_seeds)),
            "phi_grid": grid,
            "expected_gross_earnings_loss": target,
            "n_exposed_employee_records": int(
                ((persons["employment_income"].to_numpy(dtype=float) > 0)
                 & (shock > 0)).sum()
            ),
            "notes": (
                "Aggregate expected wage-bill loss is held fixed at each "
                "division's declared s_j at every phi: the loss fraction is "
                "clip(phi, s_j, 1) and the selection probability is "
                "s_j / clip(phi, s_j, 1), whose product is s_j exactly. "
                "Nobody changes employment state anywhere on the sweep, so "
                "the only thing varying is the concentration of a fixed "
                "loss. phi <= min_j s_j reproduces the diffuse wage cut; "
                "phi = 1 reproduces the concentrated wage cut (bit-identical "
                "under bernoulli assignment)."
            ),
        },
        "points": points,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[written] {args.output}")


if __name__ == "__main__":
    main()
