"""Emit manuscript macros and the figure for the continuous concentration sweep.

The factorial's two wage-cut cells are the endpoints of a one-parameter
family: impose the same aggregate loss as a cut of fraction phi on a 1/phi
share of exposed workers, and let phi run from its diffuse limit to 1. The
sweep traces cushioning along it.

Two things the manuscript must state and cannot hand-type. First, whether the
endpoints reproduce the factorial cells --- if they do not, the sweep is a
different exercise rather than an interpolation through the paper's own
design, and the macros below assert the match. Second, WHERE along phi the
movement happens: the relationship turns out to be flat over most of the
range and to fall only at large individual loss fractions, which makes
concentration a threshold effect rather than a gradient and is not what an
earlier draft's prose implied.

Reads results/concentration_sweep.json and results/factorial_decomposition.json,
writes paper/generated_sweep.tex and paper/figures/concentration_sweep.png.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
#: The endpoints must reproduce the factorial cells to this many points.
#: Both are 50-draw balanced means of the same quantity, so anything above
#: rounding indicates the two designs have drifted apart.
ENDPOINT_TOLERANCE_PP = 0.05


def fmt(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", type=Path, default=ROOT / "results/concentration_sweep.json")
    parser.add_argument(
        "--factorial", type=Path, default=ROOT / "results/factorial_decomposition.json"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "paper/generated_sweep.tex")
    parser.add_argument("--figure", type=Path, default=ROOT / "paper/figures/concentration_sweep.png")
    parser.add_argument("--min-seeds", type=int, default=50)
    args = parser.parse_args()

    sweep = json.loads(args.sweep.read_text())
    design = sweep["design"]
    n_seeds = design["n_seeds"]
    if n_seeds < args.min_seeds:
        raise ValueError(
            f"the sweep artifact has {n_seeds} seeds, below the {args.min_seeds} "
            "of the submission design. At five seeds the assignment noise at "
            "the concentrated end exceeds the movement being plotted and the "
            "endpoint does not reproduce the factorial cell. Re-run "
            "`python analysis/concentration_sweep.py --n-seeds 50`."
        )

    points = sorted(sweep["points"], key=lambda p: p["phi"])
    phis = [p["phi"] for p in points]
    cushions = [100 * p["summary"]["cushioning_rate"]["mean"] for p in points]
    sds = [100 * p["summary"]["cushioning_rate"]["sd"] for p in points]
    eff = [p["summary"]["effective_loss_records"]["mean"] for p in points]

    unit = json.loads(args.factorial.read_text())["anchors"]["unit"]
    diffuse_cell = unit["cushioning_percent"]["wage_cut"]
    concentrated_cell = unit["cushioning_percent"]["concentrated_wage_cut"]
    for label, swept, cell in (
        ("diffuse", cushions[0], diffuse_cell),
        ("concentrated", cushions[-1], concentrated_cell),
    ):
        if abs(swept - cell) > ENDPOINT_TOLERANCE_PP:
            raise ValueError(
                f"the sweep's {label} endpoint is {swept:.2f} per cent against "
                f"the factorial's {cell:.2f}. The sweep is documented as an "
                "interpolation THROUGH the paper's two wage-cut cells; if the "
                "endpoints no longer reproduce them it is a different "
                "exercise and must not be presented as nesting them."
            )

    # Where the movement happens. The plateau is the largest phi whose
    # cushioning is still within two Monte Carlo standard errors of the
    # diffuse endpoint: below it the schedule's marginal rate has not yet
    # been crossed and concentration does nothing.
    plateau_phi, plateau_cushion = phis[0], cushions[0]
    for phi, cushion, sd in zip(phis, cushions, sds):
        se = sd / math.sqrt(n_seeds) if n_seeds else 0.0
        if abs(cushion - cushions[0]) <= max(2 * se, 0.2):
            plateau_phi, plateau_cushion = phi, cushion
    total_drop = cushions[0] - cushions[-1]
    drop_after_plateau = plateau_cushion - cushions[-1]

    macros = {
        "SweepSeeds": str(n_seeds),
        "SweepPoints": str(len(points)),
        "SweepDiffuseCushion": fmt(cushions[0], 2),
        "SweepConcentratedCushion": fmt(cushions[-1], 2),
        "SweepTotalDrop": fmt(total_drop),
        "SweepPlateauPhi": fmt(100 * plateau_phi, 0),
        "SweepPlateauCushion": fmt(plateau_cushion, 1),
        "SweepDropAfterPlateau": fmt(drop_after_plateau),
        "SweepDropAfterPlateauShare": fmt(100 * drop_after_plateau / total_drop, 0),
        "SweepDiffuseEffRecords": fmt(eff[0], 0),
        "SweepConcentratedEffRecords": fmt(eff[-1], 1),
        "SweepConcentratedSD": fmt(sds[-1]),
    }

    lines = [
        "% Generated by analysis/write_sweep_results.py; do not edit.",
        *(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()),
    ]
    args.output.write_text("\n".join(lines) + "\n")
    for k, v in macros.items():
        print(f"  \\{k} = {v}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; macros written, figure skipped")
        return

    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    lower = [c - 1.96 * s / math.sqrt(n_seeds) for c, s in zip(cushions, sds)]
    upper = [c + 1.96 * s / math.sqrt(n_seeds) for c, s in zip(cushions, sds)]
    ax.fill_between(phis, lower, upper, alpha=0.18, color="#2C6496", linewidth=0)
    ax.plot(phis, cushions, marker="o", markersize=4, color="#2C6496", linewidth=1.6)
    ax.axhline(diffuse_cell, color="#616161", linestyle=":", linewidth=1)
    ax.axhline(concentrated_cell, color="#616161", linestyle=":", linewidth=1)
    ax.annotate(
        "diffuse wage-cut cell",
        xy=(phis[1], diffuse_cell),
        xytext=(0, 5),
        textcoords="offset points",
        fontsize=8,
        color="#616161",
    )
    ax.annotate(
        "concentrated cell",
        xy=(phis[1], concentrated_cell),
        xytext=(0, -12),
        textcoords="offset points",
        fontsize=8,
        color="#616161",
    )
    ax.set_xscale("log")
    ax.set_xlabel(r"$\varphi$: fraction of annual earnings lost by each affected worker")
    ax.set_ylabel("Cushioning rate (per cent)")
    ax.grid(alpha=0.25, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=200)
    print(f"Wrote {args.output} and {args.figure}")


if __name__ == "__main__":
    main()
