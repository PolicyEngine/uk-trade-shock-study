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
#: Comparators for the LFS models. These deliberately point at the PRE-FIX
#: pipeline copies, because results/lfs_selection_sensitivity.json could not be
#: regenerated in the 2026-08-13 re-run: it is built from the LFS five-quarter
#: file (UKDA SN 9490), licensed separately from the FRS and absent from the
#: Hugging Face download.
#:
#: The manuscript quotes a SHIFT (LFS-shaped minus uniform), and the Universal
#: Credit award-cache fix moves both sides by about +2.7 points, so the shift
#: survives it — but only within one vintage. Pairing the pre-fix LFS models
#: against the re-run comparator returns +0.07/+0.58/-0.78 in place of the
#: correct +2.82/+3.33/+1.97, reversing the finding that worker selection
#: matters. See results/prefix_pipeline/README.md.
UNIFORM = ROOT / "results/prefix_pipeline/full_tariff_displacement.json"
WAGE_CUT = ROOT / "results/prefix_pipeline/full_tariff_wage_cut.json"
CURRENT_UNIFORM = ROOT / "results/full_tariff_displacement.json"
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
    # VINTAGE GUARD. `selection_method` was added to MonteCarloResult after
    # these artifacts were first written, so its presence dates an artifact:
    # anything from the re-run carries it, anything older does not. The LFS
    # models and their comparators must sit on the same side of that line, or
    # the shift mixes a pre-fix minuend with a post-fix subtrahend.
    def vintage(item: dict) -> str:
        return "re-run" if item.get("selection_method") is not None else "pre-fix"

    lfs_vintages = {
        name: vintage(sensitivity["models"][name])
        for name in LFS_MODELS
        if name in sensitivity["models"]
    }
    comparator_vintages = {UNIFORM.name: vintage(uniform), WAGE_CUT.name: vintage(wage_cut)}
    all_vintages = set(lfs_vintages.values()) | set(comparator_vintages.values())
    if len(all_vintages) > 1:
        raise ValueError(
            "LFS selection models and their comparators come from different "
            "pipeline vintages, so their difference is not a selection shift.\n"
            f"  LFS models : {lfs_vintages}\n"
            f"  comparators: {comparator_vintages}\n"
            "The Universal Credit award-cache fix moved every displacement-"
            "family cushioning rate by about +2.7 points. Subtracting across "
            "that boundary turns shifts of roughly +2 to +3 points into "
            "roughly zero. Pair like with like: either re-run the LFS "
            "sensitivity (needs UKDA SN 9490) and point UNIFORM/WAGE_CUT at "
            "results/full_tariff_*.json, or leave both on the pre-fix copies "
            "in results/prefix_pipeline/."
        )
    if all_vintages == {"pre-fix"} and CURRENT_UNIFORM.exists():
        current = json.loads(CURRENT_UNIFORM.read_text())
        if current.get("selection_method") is not None:
            print(
                "NOTE: the LFS selection sensitivity is still on the pre-fix "
                "pipeline and is compared against pre-fix comparators, which "
                "is correct for a shift but means its LEVELS are stale. "
                "Re-run it against UKDA SN 9490 to retire "
                "results/prefix_pipeline/."
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

    # How far the pre-fix levels sit below their corrected counterparts,
    # measured rather than asserted: the same scenario, same design, same
    # draw count, run either side of the Universal Credit award-cache fix.
    # The manuscript quotes this when it warns that the LFS levels are on the
    # older pipeline, so the warning cannot drift from the artifacts.
    if CURRENT_UNIFORM.exists():
        current = json.loads(CURRENT_UNIFORM.read_text())
        if current.get("selection_method") is not None:
            shift = cushion_percent(current) - uniform_cushion
            if current["n_draws"] != uniform["n_draws"]:
                raise ValueError(
                    "the pre-fix and re-run comparators must share a draw "
                    f"count to measure the cache shift: {uniform['n_draws']} "
                    f"vs {current['n_draws']}"
                )
            macros["FactorialCacheShift"] = f"{shift:.1f}"

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
