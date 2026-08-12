"""Generate manuscript table rows and macros from LFS selection sensitivity results.

Besides the table body, this writer emits the scalars the prose used to
hardcode: the uniform-assignment cushioning level, the range across the three
LFS-shaped selection models, the signed shift each model implies relative to
uniform assignment, and the wage-cut-minus-displacement gap recomputed with an
LFS-shaped displacement side. The wage-cut side is read from the same 100-draw
production artifact family as the LFS runs, and the draw counts are asserted
equal so the comparison can never silently mix the 100-draw production design
with the 50-draw submission design.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
UNIFORM = ROOT / "results/full_tariff_displacement.json"
WAGE_CUT = ROOT / "results/full_tariff_wage_cut.json"
SENSITIVITY = ROOT / "results/lfs_selection_sensitivity.json"
OUTPUT = ROOT / "paper/generated_lfs_selection.tex"

LFS_MODELS = ("cells", "income_terciles", "qrf")

#: How a non-conforming or undefined artifact gets fixed. It cannot be
#: hand-edited: the numbers come out of simulations over the licensed FRS
#: microdata, so the only remedy is to re-run the producing script.
SENSITIVITY_RERUN_FIX = (
    "Re-run `python analysis/run_lfs_selection_sensitivity.py` (part of `make "
    "results`) against the licensed FRS microdata. The writer now routes "
    "its payload through `runner.json_value` with `allow_nan=False`, so a "
    "fresh artifact is RFC 8259-conforming. Do not hand-edit the JSON."
)


def load_sensitivity(path: Path = SENSITIVITY) -> dict:
    """Read the sensitivity artifact, warning if it predates the NaN fix.

    Python's ``json`` accepts the non-standard ``NaN``/``Infinity`` literals
    that ``run_lfs_selection_sensitivity.py`` used to emit; strict RFC 8259
    parsers (Go, Rust, R's jsonlite) reject them outright, so an artifact that
    still carries them is not readable by a conforming consumer of the
    replication package. Loading is deliberately not blocked — the fields this
    writer actually quotes are guarded in ``metric`` — but the artifact is
    named as needing a licensed re-run rather than passing unremarked.
    """
    text = path.read_text()
    nonstandard = sorted(
        {token for token in ("NaN", "-Infinity", "Infinity") if token in text}
    )
    if nonstandard:
        print("!" * 72)
        print(
            f"WARNING: {path.name} contains the non-standard JSON literal(s) "
            f"{nonstandard}, which strict RFC 8259 parsers reject. This "
            "artifact predates the sanitiser and is still defective as "
            f"shipped. {SENSITIVITY_RERUN_FIX}"
        )
        print("!" * 72)
    return json.loads(text)


def metric(item: dict, field: str) -> tuple[float, float]:
    """Draw mean and SD for ``field``, refusing to quote undefined values.

    A NaN draw (or a ``null``, which is how the sanitiser now records one)
    would otherwise flow into a macro as the string "nan" and into the
    manuscript as a silently broken number. Fail here instead.
    """
    raw = [draw.get(field) for draw in item["draws"]]
    values = np.array(
        [np.nan if value is None else value for value in raw], dtype=float
    )
    undefined = int((~np.isfinite(values)).sum())
    if undefined:
        raise ValueError(
            f"{item.get('scenario', '<unnamed>')}: {undefined} of "
            f"{len(values)} draws have no finite `{field}`, so its mean and SD "
            f"cannot be quoted. {SENSITIVITY_RERUN_FIX}"
        )
    return float(values.mean()), float(values.std(ddof=1))


def row(label: str, item: dict) -> str:
    gross, gross_sd = metric(item, "gross_earnings_loss")
    exchequer, exchequer_sd = metric(item, "exchequer_cost")
    cushion, cushion_sd = metric(item, "cushioning_rate")
    poverty, poverty_sd = metric(item, "poverty_rate_change_bhc")
    workers, workers_sd = metric(item, "displaced_weighted")
    return (
        f"{label} & {workers / 1_000:.1f}~$\\pm$~{workers_sd / 1_000:.1f} "
        f"& {gross / 1e6:.0f}~$\\pm$~{gross_sd / 1e6:.0f} "
        f"& {exchequer / 1e6:.0f}~$\\pm$~{exchequer_sd / 1e6:.0f} "
        f"& {100 * cushion:.1f}~$\\pm$~{100 * cushion_sd:.1f} "
        f"& {100 * poverty:.3f}~$\\pm$~{100 * poverty_sd:.3f} \\\\"
    )


def cushion_percent(item: dict) -> float:
    """Mean cushioning rate over the item's draws, in percentage points."""
    return 100 * metric(item, "cushioning_rate")[0]


def main() -> None:
    uniform = json.loads(UNIFORM.read_text())
    wage_cut = json.loads(WAGE_CUT.read_text())
    sensitivity = load_sensitivity()
    if uniform["n_draws"] != sensitivity["n_draws"]:
        raise ValueError("Uniform and LFS selection models must use equal draw counts.")
    if wage_cut["n_draws"] != sensitivity["n_draws"]:
        raise ValueError(
            "The wage-cut comparator must come from the same design as the LFS "
            f"selection runs: {WAGE_CUT.name} has {wage_cut['n_draws']} draws "
            f"against {sensitivity['n_draws']}. Do not mix the 100-draw "
            "production suite with the 50-draw submission design."
        )
    missing = [name for name in LFS_MODELS if name not in sensitivity["models"]]
    if missing:
        raise KeyError(
            f"results/lfs_selection_sensitivity.json is missing models: {missing}"
        )

    rows = [
        row("Uniform central assignment", uniform),
        row("LFS calibrated cells", sensitivity["models"]["cells"]),
        row("LFS income terciles", sensitivity["models"]["income_terciles"]),
        row("LFS regularised QRF", sensitivity["models"]["qrf"]),
    ]

    uniform_cushion = cushion_percent(uniform)
    wage_cushion = cushion_percent(wage_cut)
    lfs_cushions = [
        cushion_percent(sensitivity["models"][name]) for name in LFS_MODELS
    ]
    # Signed selection shift and the implied headline gap when the
    # displacement side is LFS-shaped rather than uniformly assigned.
    shifts = [cushion - uniform_cushion for cushion in lfs_cushions]
    gaps = [wage_cushion - cushion for cushion in lfs_cushions]
    macros = {
        "LFSSelectionDraws": f"{uniform['n_draws']}",
        "LFSUniformCushion": f"{uniform_cushion:.1f}",
        "LFSSelectionCushionMin": f"{min(lfs_cushions):.1f}",
        "LFSSelectionCushionMax": f"{max(lfs_cushions):.1f}",
        "LFSSelectionShiftMin": f"{min(shifts):.1f}",
        "LFSSelectionShiftMax": f"{max(shifts):.1f}",
        "LFSSelectionGapMin": f"{min(gaps):.1f}",
        "LFSSelectionGapMax": f"{max(gaps):.1f}",
        # The gap these runs narrow is the one implied by THEIR OWN design
        # (100-draw Bernoulli), not the 50-draw balanced PrimaryCushionDifference
        # of the submission table.  The manuscript must quote this baseline when
        # it narrates the narrowing, or it silently mixes two designs.
        "LFSSelectionBaselineGap": f"{wage_cushion - uniform_cushion:.1f}",
    }

    OUTPUT.write_text(
        "% Generated by analysis/write_lfs_selection_results.py\n"
        + "\n".join(
            f"\\newcommand{{\\{name}}}{{{value}}}"
            for name, value in macros.items()
        )
        + "\n\\newcommand{\\LFSSelectionRows}{%\n"
        + "\n".join(rows)
        + "\n}\n"
    )
    for name, value in macros.items():
        print(f"  \\{name} = {value}")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
