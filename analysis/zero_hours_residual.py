"""Isolate the employment-STATUS flag by removing the hours channel.

The factorial's employment-state step compares a concentrated wage cut against
displacement. Section 4.3 shows that step is not a status effect at all: the
two cells differ in ``hours_worked``, which displacement zeroes and the
concentrated cell leaves at baseline, and hours gate three childcare-linked
entitlements through ``in_work``. ``employment_status`` itself is read by no
tax or benefit formula.

``apply_concentrated_wage_cut(..., zero_hours=True)`` removes that difference
by zeroing hours in the concentrated cell as well, leaving the two cells
differing ONLY in the (inert) status flags. The residual should then collapse
to zero. This script runs that comparison and writes the result, so the claim
is measured rather than predicted.

Reads nothing but the FRS dataset; writes results/zero_hours_residual.json.
Needs the licensed FRS microdata and policyengine-uk.

Usage: PYTHONPATH=. .venv/bin/python analysis/zero_hours_residual.py [--n-seeds 25]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from uk_trade_shock_study.runner import _baseline_and_persons, json_value
from uk_trade_shock_study.shocks import (
    TradeShockScenario,
    apply_concentrated_wage_cut,
    apply_shocks,
    build_shocked_simulation,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mechanism_decomposition import cushioning_components  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PERIOD = 2026
DATASET = ROOT / "data/frs_2024_25.h5"
OUTPUT = ROOT / "results/zero_hours_residual.json"
#: Matches the submission design's unit anchor.
ELASTICITY = 1.0


def scenario(margin: str) -> TradeShockScenario:
    return TradeShockScenario(
        f"zero_hours_{margin}",
        "full_tariff",
        margin,
        elasticity=ELASTICITY,
        duration_equivalent=1.0,
        selection_method="balanced",
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-seeds", type=int, default=25)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)

    dataset, baseline, persons = _baseline_and_persons(args.dataset, None, PERIOD)
    # `hours_worked` is not in runner.PERSON_VARIABLES, so the persons frame the
    # rest of the pipeline builds does not carry it -- which is why the
    # zero_hours option, though present in shocks.py since the hours diagnosis,
    # could never actually be exercised: it raises KeyError on the frame every
    # other script constructs. Attach it here rather than widening
    # PERSON_VARIABLES, which would change the frame every stored result was
    # produced from.
    persons = persons.copy()
    persons["hours_worked"] = baseline.calculate(
        "hours_worked", period=PERIOD, map_to="person"
    ).values

    conc_scen = scenario("concentrated_wage_cut")
    disp_scen = scenario("displacement")
    rows: list[dict] = []
    for seed in range(args.n_seeds):
        # Standard concentrated cell: hours left at baseline.
        table_std = apply_concentrated_wage_cut(
            persons, conc_scen, seed=seed, zero_hours=False
        )
        std = cushioning_components(
            baseline,
            build_shocked_simulation(dataset, baseline, table_std, PERIOD),
            persons,
            table_std,
            PERIOD,
        )
        # Status-only cell: hours zeroed here too, so the ONLY remaining
        # difference against displacement is the inert status flag.
        table_zero = apply_concentrated_wage_cut(
            persons, conc_scen, seed=seed, zero_hours=True
        )
        zero = cushioning_components(
            baseline,
            build_shocked_simulation(dataset, baseline, table_zero, PERIOD),
            persons,
            table_zero,
            PERIOD,
        )
        table_disp = apply_shocks(persons, disp_scen, seed=seed)
        disp = cushioning_components(
            baseline,
            build_shocked_simulation(dataset, baseline, table_disp, PERIOD),
            persons,
            table_disp,
            PERIOD,
        )
        rows.append(
            {
                "seed": seed,
                "concentrated": std["cushioning_rate"],
                "concentrated_zero_hours": zero["cushioning_rate"],
                "displacement": disp["cushioning_rate"],
            }
        )
        print(
            f"[zero_hours] seed {seed}: conc {std['cushioning_rate']:.4f} "
            f"conc0h {zero['cushioning_rate']:.4f} disp {disp['cushioning_rate']:.4f}",
            flush=True,
        )

    def arr(key: str) -> np.ndarray:
        return np.array([r[key] for r in rows], dtype=float)

    conc, conc0, disp = arr("concentrated"), arr("concentrated_zero_hours"), arr("displacement")
    state_step = (conc - disp) * 100
    status_only = (conc0 - disp) * 100
    hours_channel = (conc - conc0) * 100

    payload = {
        "design": {
            "n_seeds": args.n_seeds,
            "seeds": list(range(args.n_seeds)),
            "elasticity": ELASTICITY,
            "selection_method": "balanced",
            "period": PERIOD,
            "note": (
                "employment_state_step splits into an hours channel and a "
                "status-only residual. zero_hours=True removes hours from the "
                "concentrated cell, so status_only_residual is the step "
                "attributable to the employment_status flags alone."
            ),
        },
        "employment_state_step_pp": {
            "mean": float(state_step.mean()),
            "sd": float(state_step.std(ddof=1)),
        },
        "status_only_residual_pp": {
            "mean": float(status_only.mean()),
            "sd": float(status_only.std(ddof=1)),
            "max_abs": float(np.abs(status_only).max()),
        },
        "hours_channel_pp": {
            "mean": float(hours_channel.mean()),
            "sd": float(hours_channel.std(ddof=1)),
        },
        "draws": rows,
    }
    args.output.write_text(json.dumps(json_value(payload), indent=2, allow_nan=False) + "\n")
    print(f"\n  employment-state step : {state_step.mean():+.3f} pp")
    print(f"  hours channel         : {hours_channel.mean():+.3f} pp")
    print(f"  status-only residual  : {status_only.mean():+.3f} pp "
          f"(max |.| {np.abs(status_only).max():.3f})")
    print(f"[written] {args.output}")


if __name__ == "__main__":
    main()
