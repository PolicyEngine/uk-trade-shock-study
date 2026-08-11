"""Generate manuscript macros from the HMRC destination-panel benchmark.

Several of the macros here are PROVENANCE macros rather than estimates:
``\\HMRCAnticipationSpec`` says which anticipation convention produced the
stored point estimates, ``\\HMRCDofConvention`` says which degrees-of-freedom
convention produced the stored standard errors, and ``\\HMRCFigureBase`` /
``\\HMRCFigureBaseIsAnticipation`` say which base period the STORED monthly
figure is actually normalised on. All exist because the artifacts and the
estimator can fall out of step: a stored CSV/PNG predating a change to
``NORMALISATION_BASE`` carries the OLD base under unchanged file names, and a
hand-written caption that names the new base then contradicts the figure it
sits under. Deriving the caption's base from the plotted data makes that
contradiction impossible: if the figure is stale the macro says so, and the
caption is stale with it rather than silently wrong.

The same artifact is stale along three independent dimensions, so each gets
its own macro rather than one "this is old" flag: the anticipation window
moves the POINT ESTIMATES, the absorbed-fixed-effect degrees-of-freedom
correction moves the STANDARD ERRORS (and with them every interval and
p-value), and the normalisation base moves only the FIGURE. A re-run fixes
all three at once, but until then the manuscript has to say which of them it
is quoting.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results/hmrc_destination_event_study.json"
MONTHLY = ROOT / "results/hmrc_destination_event_study_monthly.csv"
OUTPUT = ROOT / "paper/generated_trade_benchmarks.tex"

# Single source of truth for the anticipation window: imported from the
# estimator rather than re-spelled here, so the figure-base warning can never
# describe a window the estimator no longer uses. Works both as
# `python analysis/write_trade_benchmark_results.py` and as
# `python -m analysis.write_trade_benchmark_results` from the repo root.
try:  # pragma: no cover - import plumbing
    from analysis.hmrc_destination_event_study import ANTICIPATION_WINDOW
except ModuleNotFoundError:  # pragma: no cover - run as a bare script
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from hmrc_destination_event_study import ANTICIPATION_WINDOW

#: A base window is "detected" from the plotted series by finding the
#: contiguous month range whose mean ``normalised_gap`` is zero — normalising
#: subtracts the base-window mean, so that mean is zero to floating-point
#: noise and nothing else in the series comes near it. The tolerance is many
#: orders of magnitude above the accumulated rounding error (~1e-16) and many
#: orders below the dispersion of the series (~1e-1).
BASE_DETECTION_TOLERANCE = 1e-9

MONTH_NAMES = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)
#: Abbreviations used by hmrc_destination_event_study.format_month_window,
#: which is what a ``base_period`` column contains.
MONTH_ABBREVIATIONS = tuple(name[:3] for name in MONTH_NAMES)


def anticipation_spec_label(data: dict) -> str:
    """Describe WHICH anticipation convention produced the stored estimates.

    ``primary_capped_value_weighted`` is a stable key whose meaning changed:
    it used to hold the April-2025-only omission and now holds the
    January--April 2025 window omission.  A stored artifact predating that
    change therefore carries a legacy number under the current key name, and
    the manuscript must say which one it is quoting rather than silently
    relabel it.  Emitting the provenance as a macro is what keeps the prose
    honest between the code change and the next data-bearing run.
    """
    months = data.get("anticipation_months_omitted")
    if not months:
        return (
            "April 2025 only (legacy convention; re-run "
            "\\texttt{make trade-event-study} for the anticipation-window "
            "specification)"
        )
    return "January--April 2025 anticipation window"


# ---------------------------------------------------------------------------
# Which degrees-of-freedom convention produced the STORED standard errors?
# ---------------------------------------------------------------------------

#: Fields the corrected estimator writes into every fitted block. A block that
#: carries them was fitted with the absorbed HS4 fixed effects added back to
#: the residual degrees of freedom before the cluster-robust finite-sample
#: correction; a block that carries neither predates that fix.
DOF_CORRECTION_KEYS = ("absorbed_fixed_effects", "absorbed_dof_scale")

DOF_CONVENTION_CORRECTED = "absorbed_dof_corrected"
DOF_CONVENTION_LEGACY = "absorbed_dof_uncorrected"

#: How the convention reads in the manuscript. An unrecognised state raises
#: rather than being passed through, for the same reason as the entitled-scope
#: and JSA-rate tokens in ``analysis/write_referee_macros.py``: the point of a
#: provenance macro is that the prose states something the artifact vouches
#: for, and a half-corrected artifact vouches for nothing.
DOF_CONVENTION_PHRASES = {
    DOF_CONVENTION_CORRECTED: (
        "with the cluster-robust standard errors corrected for the HS4 fixed "
        "effects absorbed by within-demeaning"
    ),
    DOF_CONVENTION_LEGACY: (
        "on the superseded convention that counts residual degrees of freedom "
        "without the absorbed HS4 fixed effects, so every standard error, "
        "confidence interval and $p$-value quoted from this artifact is "
        "slightly too tight (re-run \\texttt{make trade-event-study} for the "
        "corrected inference)"
    ),
}


def _fitted_blocks(node, path: str = "") -> list[tuple[str, dict]]:
    """Every nested dict that is a fitted estimate, i.e. carries a standard error."""
    found: list[tuple[str, dict]] = []
    if isinstance(node, dict):
        if "standard_error" in node:
            found.append((path or "<root>", node))
        for key, value in node.items():
            found.extend(_fitted_blocks(value, f"{path}.{key}" if path else str(key)))
    return found


def dof_convention_state(data: dict) -> str:
    """Token for the degrees-of-freedom convention the stored estimates use.

    The corrected estimator stamps ``absorbed_fixed_effects`` and
    ``absorbed_dof_scale`` on every block it fits, so the state is read off
    the artifact rather than off a file date. Anything other than "all blocks
    corrected" or "no block corrected" raises: a partially re-run artifact
    would put corrected and uncorrected inference in the same table under one
    macro, which is precisely the silent mismatch these macros exist to
    prevent.
    """
    blocks = _fitted_blocks(data)
    if not blocks:
        raise ValueError(
            "no fitted blocks (nothing carrying a standard_error) found in the "
            "HMRC artifact, so its degrees-of-freedom convention cannot be "
            "established."
        )
    corrected, legacy, partial = [], [], []
    for path, block in blocks:
        present = [key for key in DOF_CORRECTION_KEYS if key in block]
        if len(present) == len(DOF_CORRECTION_KEYS):
            corrected.append(path)
        elif not present:
            legacy.append(path)
        else:
            partial.append(path)
    if partial:
        raise ValueError(
            f"HMRC artifact blocks {sorted(partial)} carry only part of "
            f"{list(DOF_CORRECTION_KEYS)}, so the degrees-of-freedom "
            "convention behind their standard errors is unrecognisable."
        )
    if corrected and legacy:
        raise ValueError(
            f"HMRC artifact mixes conventions: {len(corrected)} blocks carry "
            f"the absorbed-fixed-effect correction and {len(legacy)} do not "
            f"(uncorrected: {sorted(legacy)}). Re-run "
            "`make trade-event-study` so one macro can describe the whole "
            "artifact."
        )
    if legacy:
        return DOF_CONVENTION_LEGACY
    bad_scales = [
        path
        for path, block in blocks
        if not (float(block["absorbed_dof_scale"]) >= 1.0)
        or int(block["absorbed_fixed_effects"]) < 0
    ]
    if bad_scales:
        raise ValueError(
            f"HMRC artifact blocks {sorted(bad_scales)} record an absorbed "
            "degrees-of-freedom scale below 1 (or a negative absorbed count), "
            "which is not a state the correction can produce: the scale-up "
            "for absorbed fixed effects is never a scale-down."
        )
    return DOF_CONVENTION_CORRECTED


def dof_convention_label(data: dict) -> str:
    """Render the degrees-of-freedom provenance, or raise on an unknown state."""
    state = dof_convention_state(data)
    if state not in DOF_CONVENTION_PHRASES:  # pragma: no cover - guarded above
        raise KeyError(
            f"HMRC standard errors resolved to convention {state!r}, which "
            f"this writer has no manuscript phrasing for (known: "
            f"{sorted(DOF_CONVENTION_PHRASES)})."
        )
    return DOF_CONVENTION_PHRASES[state]


# ---------------------------------------------------------------------------
# Which base period does the STORED monthly figure actually use?
# ---------------------------------------------------------------------------


def _month_index(month: int) -> int:
    """Months since year 0, so windows can be compared as integer intervals."""
    return (month // 100) * 12 + (month % 100) - 1


def format_base_period(start: int, end: int) -> str:
    """Human-readable LaTeX label for an inclusive YYYYMM window.

    A full calendar year reads ``calendar 2024``; a within-year window reads
    ``January--March 2025``; anything else names both endpoints.
    """
    for month in (start, end):
        if not 1 <= month % 100 <= 12:
            raise ValueError(f"{month} is not a YYYYMM month")
    if _month_index(end) < _month_index(start):
        raise ValueError(f"base window {(start, end)} ends before it starts")
    start_name = MONTH_NAMES[start % 100 - 1]
    end_name = MONTH_NAMES[end % 100 - 1]
    if start == end:
        return f"{start_name} {start // 100}"
    if start // 100 == end // 100:
        if start % 100 == 1 and end % 100 == 12:
            return f"calendar {start // 100}"
        return f"{start_name}--{end_name} {start // 100}"
    return f"{start_name} {start // 100}--{end_name} {end // 100}"


def parse_base_period(label: str) -> tuple[int, int]:
    """Invert ``hmrc_destination_event_study.format_month_window``.

    Accepts the three shapes that function emits (``2024``, ``Jan-Apr 2025``,
    ``Jan 2025``, ``Dec 2024-Feb 2025``) and the long forms this module emits.
    Raises on anything else: guessing a base period would defeat the point of
    reading the column at all.
    """
    text = " ".join(str(label).split()).replace("--", "-").replace("–", "-")

    def month_number(name: str) -> int:
        cleaned = name.strip().strip(".").lower()
        for index, full in enumerate(MONTH_NAMES, start=1):
            if cleaned in (full.lower(), MONTH_ABBREVIATIONS[index - 1].lower()):
                return index
        raise ValueError(f"unrecognised month name {name!r} in base period {label!r}")

    body = text[len("calendar "):].strip() if text.lower().startswith("calendar ") else text
    if body.isdigit() and len(body) == 4:  # a bare calendar year
        return int(body) * 100 + 1, int(body) * 100 + 12

    parts = [part.strip() for part in body.split("-")]
    if len(parts) == 1:  # "Jan 2025"
        name, year = parts[0].rsplit(" ", 1)
        return int(year) * 100 + month_number(name), int(year) * 100 + month_number(name)
    if len(parts) != 2:
        raise ValueError(f"cannot parse base period {label!r}")
    left, right = parts
    end_name, end_year = right.rsplit(" ", 1)
    if " " in left:  # "Dec 2024-Feb 2025"
        start_name, start_year = left.rsplit(" ", 1)
    else:  # "Jan-Apr 2025": the year is carried by the right endpoint
        start_name, start_year = left, end_year
    return (
        int(start_year) * 100 + month_number(start_name),
        int(end_year) * 100 + month_number(end_name),
    )


def detect_base_period(frame: pd.DataFrame) -> tuple[tuple[int, int], str]:
    """((start, end), source) for the base period the stored series uses.

    Prefers an explicit ``base_period`` column (written by the revised
    estimator). Falls back to detecting the window from the data: the plotted
    series is the weighted gap minus its base-window mean, so that window is
    the contiguous month range whose mean ``normalised_gap`` is zero. This is
    what makes the caption drift-proof — the label describes the CSV that was
    actually plotted, not what the current estimator would produce.
    """
    if "base_period" in frame.columns:
        labels = {
            str(value).strip()
            for value in frame["base_period"].dropna().unique()
            if str(value).strip()
        }
        if len(labels) > 1:
            raise ValueError(
                f"results CSV carries {len(labels)} different base_period "
                f"labels {sorted(labels)}; it cannot describe one figure."
            )
        if labels:
            return parse_base_period(labels.pop()), "base_period column"

    months = frame["month"].to_numpy(dtype=np.int64)
    gaps = frame["normalised_gap"].to_numpy(dtype=float)
    if not np.isfinite(gaps).all():
        raise ValueError("normalised_gap contains non-finite values")
    # Contiguity in the row order must be contiguity in calendar months, or a
    # "contiguous range" would not be a window at all.
    positions = np.array([_month_index(int(m)) for m in months])
    if not (np.diff(positions) == 1).all():
        raise ValueError("monthly series is not a contiguous run of months")

    candidates: list[tuple[int, int, int]] = []
    for i in range(len(gaps)):
        total = 0.0
        for j in range(i, len(gaps)):
            total += gaps[j]
            if abs(total / (j - i + 1)) <= BASE_DETECTION_TOLERANCE:
                candidates.append((j - i + 1, int(months[i]), int(months[j])))
    if not candidates:
        raise ValueError(
            "no contiguous month window in "
            f"{MONTHLY.name} has a mean normalised_gap within "
            f"{BASE_DETECTION_TOLERANCE} of zero, so the base period the "
            "stored figure uses cannot be identified. Re-run "
            "`make trade-event-study` so the CSV carries a base_period column."
        )
    longest = max(length for length, _, _ in candidates)
    winners = [c for c in candidates if c[0] == longest]
    if len(winners) > 1:
        raise ValueError(
            f"base-period detection is ambiguous: {len(winners)} windows of "
            f"{longest} months all have a mean normalised_gap of zero "
            f"({[(s, e) for _, s, e in winners]})."
        )
    _, start, end = winners[0]
    return (start, end), "detected from normalised_gap"


def base_anticipation_phrase(
    start: int,
    end: int,
    window: tuple[int, int] = ANTICIPATION_WINDOW,
) -> str:
    """Short LaTeX clause saying whether the base sits in the front-running window.

    A base period drawn from the anticipation months measures the post-policy
    gap against the front-running PEAK rather than against a clean pre-shock
    level, so the caption has to carry the warning. Emitting it as a macro
    means the warning appears and disappears with the data, not with whoever
    last edited the caption.
    """
    window_label = format_base_period(*window)
    base_lo, base_hi = _month_index(start), _month_index(end)
    win_lo, win_hi = _month_index(window[0]), _month_index(window[1])
    if base_lo >= win_lo and base_hi <= win_hi:
        return (
            f"\\emph{{inside}} the {window_label} anticipation window, so the "
            "plotted gap is measured against the front-running peak"
        )
    if base_lo <= win_hi and base_hi >= win_lo:
        return (
            f"\\emph{{partly inside}} the {window_label} anticipation window, "
            "so the plotted gap is measured partly against the front-running peak"
        )
    return f"outside the {window_label} anticipation window"


def figure_base_macros(frame: pd.DataFrame) -> tuple[dict[str, str], str]:
    """(macro values, provenance source) for the stored figure's base period."""
    (start, end), source = detect_base_period(frame)
    return (
        {
            "HMRCFigureBase": format_base_period(start, end),
            "HMRCFigureBaseIsAnticipation": base_anticipation_phrase(start, end),
        },
        source,
    )


def main() -> None:
    data = json.loads(INPUT.read_text())
    monthly = pd.read_csv(MONTHLY)
    primary = data["primary_capped_value_weighted"]
    equal = data["equal_product_weighted"]
    high = data["tariff_intensity_groups"]["steel_and_auto_chapters"]
    other = data["tariff_intensity_groups"]["other_products"]
    recent = data["sample_start_sensitivity"]["202301"]
    figure_base, base_source = figure_base_macros(monthly)
    macros = {
        "HMRCPanelProducts": primary["product_count"],
        "HMRCPanelEffectiveProducts": f"{primary['effective_product_count']:.0f}",
        "HMRCWeightedEffect": f"{100 * primary['proportional_effect']:.1f}",
        "HMRCWeightedCILower": f"{100 * np.expm1(primary['ci_lower']):.1f}",
        "HMRCWeightedCIUpper": f"{100 * np.expm1(primary['ci_upper']):.1f}",
        "HMRCWeightedPValue": (
            "<0.001"
            if primary["p_value"] < 0.001
            else f"{primary['p_value']:.3f}"
        ),
        "HMRCEqualEffect": f"{100 * equal['proportional_effect']:.1f}",
        "HMRCEqualPValue": f"{equal['p_value']:.3f}",
        "HMRCTrendPValue": f"{primary['differential_linear_trend_p_value']:.3f}",
        "HMRCRecentEffect": f"{100 * recent['proportional_effect']:.1f}",
        "HMRCRecentPValue": f"{recent['p_value']:.3f}",
        "HMRCHighTariffEffect": f"{100 * high['proportional_effect']:.1f}",
        "HMRCHighTariffPValue": f"{high['p_value']:.3f}",
        "HMRCOtherEffect": f"{100 * other['proportional_effect']:.1f}",
        "HMRCOtherPValue": f"{other['p_value']:.3f}",
        "HMRCAnticipationSpec": anticipation_spec_label(data),
        "HMRCDofConvention": dof_convention_label(data),
        "HMRCHighTariffTrendPValue": (
            f"{high['differential_linear_trend_p_value']:.3f}"
        ),
        **figure_base,
    }
    OUTPUT.write_text(
        "% Generated by analysis/write_trade_benchmark_results.py\n"
        + "\n".join(
            f"\\newcommand{{\\{name}}}{{{value}}}"
            for name, value in macros.items()
        )
        + "\n"
    )
    print(f"Wrote {OUTPUT}")
    print(
        f"[figure base] {MONTHLY.name}: {figure_base['HMRCFigureBase']} "
        f"(source: {base_source})"
    )
    if dof_convention_state(data) == DOF_CONVENTION_LEGACY:
        print(
            "[dof] WARNING: the stored HMRC estimates predate the "
            "absorbed-fixed-effect degrees-of-freedom correction, so every "
            "standard error, confidence interval and p-value in "
            f"{INPUT.name} is too tight. Re-run `make trade-event-study`; "
            "until then the prose must carry \\HMRCDofConvention."
        )
    if "inside" in figure_base["HMRCFigureBaseIsAnticipation"]:
        print(
            "[figure base] WARNING: the STORED figure is normalised on the "
            "anticipation window. Re-run `make trade-event-study` to "
            "regenerate the CSV and PNG on the clean base; until then the "
            "caption must carry \\HMRCFigureBaseIsAnticipation."
        )


if __name__ == "__main__":
    main()
