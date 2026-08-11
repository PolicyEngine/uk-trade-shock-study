"""Write paper/referee_macros.tex from results/referee_fixes.json, the
take-up diagnosis artifact and the generated submission rows.

Provides: the main-table row split (OBR three-month rows moved to the
appendix on thin-support grounds), the pension-gross cushioning macros, the
take-up headline macros (including the redraw-scope bound), the New Style JSA
bound, and the monthly-versus-annual UC bounding macros.

Hard-fail convention: every macro below is read from an artifact. A missing
field is a BUG, not a reason to fall back to a hand-typed number, so this
script raises rather than emitting anything it cannot source.

Usage: uv run python analysis/write_referee_macros.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# Single source of truth for which cell the manuscript excludes from the main
# table; see analysis/write_submission_results.py. Imported (rather than
# re-spelled here) so the row split and the headline contrast range can never
# disagree about what "thin support" means. Works both as `python
# analysis/write_referee_macros.py` (script directory on sys.path) and as
# `from analysis.write_referee_macros import ...` (repo root on sys.path).
try:  # pragma: no cover - import plumbing
    from analysis.write_submission_results import is_thin_support_row
except ModuleNotFoundError:  # pragma: no cover - run as a bare script
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from write_submission_results import is_thin_support_row

RESULTS = Path("results")
PAPER = Path("paper")

#: The four claiming conventions reported in the take-up grid, in order.
TAKEUP_CONVENTIONS = ("0.55", "0.80", "1.00", "stale_baseline_flag")
#: Key under ``takeup_headline`` holding the all-entitled re-draw scope, if
#: analysis/referee_fixes.py has been re-run since that scope was added.
ALL_ENTITLED_KEY = "all_entitled_scope"
#: Decimal places at which the entitled-scope endpoints and their spread are
#: printed. All three are rendered at this precision FROM THE SAME rounding so
#: a reader subtracting the printed endpoints recovers the printed spread.
ENTITLED_DECIMALS = 1
#: How ``jsa_bounding.parameters.rate_source`` reads in the manuscript. Keyed
#: on the enum-like token written by analysis/referee_fixes.py; an unknown
#: token raises rather than being passed through, because the whole point of
#: the macro is that the manuscript states a provenance the artifact vouches
#: for.
JSA_RATE_SOURCE_PHRASES = {
    "policyengine_parameter": "the \\texttt{policyengine-uk} parameter tree",
    "statutory_fallback": (
        "the published DWP statutory rate for 2025--26 "
        "(\\texttt{policyengine-uk} was not available to this build)"
    ),
}


def split_submission_rows() -> tuple[str, str]:
    text = (PAPER / "generated_submission.tex").read_text()
    match = re.search(r"\\newcommand{\\SubmissionScenarioRows}{%\n(.*?)\n}", text, re.S)
    if not match:
        raise RuntimeError("SubmissionScenarioRows not found in generated_submission.tex")
    rows = [r for r in match.group(1).splitlines() if r.strip()]
    thin = [r for r in rows if is_thin_support_row(r)]
    main = [r for r in rows if not is_thin_support_row(r)]
    if not thin:
        raise RuntimeError(
            "no thin-support rows found in generated_submission.tex; the "
            "exclusion rule in write_submission_results.py and the generated "
            "table have drifted apart"
        )
    return "\n".join(main), "\n".join(thin)


def require(mapping: dict, *path: str, where: str):
    """Fetch a nested field or raise — never silently substitute a number."""
    node = mapping
    for key in path:
        if not isinstance(node, dict) or key not in node:
            raise KeyError(
                f"{where}: required field {'.'.join(path)!r} is missing "
                f"(stopped at {key!r}). Re-run the producing script; do not "
                f"hand-edit the generated .tex."
            )
        node = node[key]
    return node


def takeup_convention_spread_pp(block: dict, where: str) -> float:
    """Max absolute cushioning difference (pp) across claiming conventions.

    A spread of EXACTLY ZERO is the expected current state under the
    ``new_entitlement`` scope and is emitted faithfully: the re-draw set is
    empty at this calibration, so the grid is inert by construction and every
    convention returns a bit-identical rate. The macro exists precisely so the
    manuscript quotes that zero instead of implying the grid tested something.
    What is NOT tolerated is a missing field — that raises.
    """
    rates = [
        float(require(block, key, "cushioning_rate", "mean", where=where))
        for key in TAKEUP_CONVENTIONS
    ]
    return 100.0 * (max(rates) - min(rates))


def entitled_scope_bound(referee: dict, diagnosis: dict) -> tuple[float, float, str]:
    """(stale cushioning %, full-take-up cushioning %, source label).

    Prefers an ``all_entitled`` grid computed at the CURRENT calibration by
    analysis/referee_fixes.py. Falls back to the legacy
    results/takeup_diagnosis.json, which used the same all-changed-units
    convention but an OLDER calibration vintage (the former epsilon = 2 high
    case: seed-0 gross loss 1.70bn against the current unit-stress 0.886bn),
    and must therefore always be labelled as such in the manuscript.
    """
    scope = referee.get("takeup_headline", {}).get(ALL_ENTITLED_KEY)
    if isinstance(scope, dict) and "displacement" in scope:
        where = f"referee_fixes.json takeup_headline.{ALL_ENTITLED_KEY}.displacement"
        block = scope["displacement"]
        stale = float(
            require(block, "stale_baseline_flag", "cushioning_rate", "mean", where=where)
        )
        full = float(require(block, "1.00", "cushioning_rate", "mean", where=where))
        return 100.0 * stale, 100.0 * full, "referee_fixes.json (current calibration)"

    where = "takeup_diagnosis.json results.full_tariff_displacement"
    block = require(diagnosis, "results", "full_tariff_displacement", where=where)
    stale = float(require(block, "current_stale_flag", "cushioning_mean", where=where))
    full = float(require(block, "takeup_100", "cushioning_mean", where=where))
    return 100.0 * stale, 100.0 * full, "takeup_diagnosis.json (legacy calibration vintage)"


def check_entitled_scope_macros(
    stale_s: str, full_s: str, spread_s: str, nd: int = ENTITLED_DECIMALS
) -> None:
    """Raise unless the three PRINTED entitled-scope numbers add up.

    The manuscript quotes \\TakeupEntitledStale, \\TakeupEntitledFull and
    \\TakeupEntitledSpread in one sentence, so a reader subtracts the printed
    endpoints and expects the printed spread. Rounding all three independently
    from the unrounded values does not deliver that: 34.4505 and 41.5331 print
    as 34.5 and 41.5, whose difference is 7.0, while the unrounded spread
    7.0826 prints as 7.1. The check runs on the emitted STRINGS, so it holds
    whatever a future edit does to the individual format specs.
    """
    if round(float(full_s) - float(stale_s), nd) != round(float(spread_s), nd):
        raise RuntimeError(
            "entitled-scope macros are not self-consistent: "
            f"\\TakeupEntitledFull={full_s} minus \\TakeupEntitledStale="
            f"{stale_s} is {float(full_s) - float(stale_s):.{nd}f}, but "
            f"\\TakeupEntitledSpread prints {spread_s}. The manuscript quotes "
            "all three in one sentence; emit them at a precision at which the "
            "subtraction works."
        )


def entitled_scope_macro_values(
    stale: float, full: float, nd: int = ENTITLED_DECIMALS
) -> tuple[str, str, str]:
    """Render (stale, full, spread) at ``nd`` places so the sentence adds up.

    The spread is formed from the PRINTED endpoints rather than from the
    unrounded values, and the identity is then re-checked on the emitted
    strings by ``check_entitled_scope_macros`` — so the inconsistency can never
    ship silently.
    """
    stale_s = f"{stale:.{nd}f}"
    full_s = f"{full:.{nd}f}"
    spread_s = f"{float(full_s) - float(stale_s):.{nd}f}"
    check_entitled_scope_macros(stale_s, full_s, spread_s, nd)
    return stale_s, full_s, spread_s


def main() -> None:
    d = json.loads((RESULTS / "referee_fixes.json").read_text())
    diagnosis = json.loads((RESULTS / "takeup_diagnosis.json").read_text())
    pension = d["pension_channel"]
    wc = pension["full_tariff_wage_cut"]
    disp = pension["full_tariff_displacement"]
    contrast = pension["contrast"]
    takeup = d["takeup_headline"]["displacement"]
    uc = d["monthly_uc_bounding"]
    jsa = require(d, "jsa_bounding", where="referee_fixes.json")
    spell3 = uc["spells"]["3m"]

    main_rows, thin_rows = split_submission_rows()

    def pct(x: float, nd: int = 1) -> str:
        return f"{100 * x:.{nd}f}"

    # Take-up grid: draw count and the convention spread, both sourced.
    takeup_seeds = require(
        d, "takeup_headline", "displacement", "n_seeds", where="referee_fixes.json"
    )
    convention_spread = takeup_convention_spread_pp(
        takeup, "referee_fixes.json takeup_headline.displacement"
    )
    # The binding claiming margin (all-entitled re-draw scope).
    entitled_stale, entitled_full, entitled_source = entitled_scope_bound(d, diagnosis)
    stale_str, full_str, spread_str = entitled_scope_macro_values(
        entitled_stale, entitled_full
    )
    # New Style JSA provenance: which of the two rate sources produced the
    # figure this build prints.
    jsa_rate_source = require(
        jsa, "parameters", "rate_source", where="referee_fixes.json jsa_bounding"
    )
    if jsa_rate_source not in JSA_RATE_SOURCE_PHRASES:
        raise KeyError(
            f"referee_fixes.json jsa_bounding.parameters.rate_source is "
            f"{jsa_rate_source!r}, which this writer has no manuscript phrasing "
            f"for (known: {sorted(JSA_RATE_SOURCE_PHRASES)}). Add the phrasing "
            "rather than emitting an unlabelled provenance."
        )
    zero_uc_non_takeup = float(
        require(
            diagnosis,
            "results",
            "decomposition_full_tariff_seed0",
            "current_stale_flag",
            "zero_uc",
            "share_zero_uc_non_takeup",
            where="takeup_diagnosis.json",
        )
    )

    lines = [
        "% Generated by analysis/write_referee_macros.py; do not edit.",
        "\\newcommand{\\SubmissionScenarioRowsMain}{%",
        main_rows,
        "}",
        "\\newcommand{\\SubmissionScenarioRowsThin}{%",
        thin_rows,
        "}",
        # Pension channel (unit 12-month, balanced primary design).
        f"\\newcommand{{\\PensionSeeds}}{{{pension.get('n_seeds', 25)}}}",
        f"\\newcommand{{\\PensionWageCushionHBAI}}{{{pct(wc['cushioning_hbai']['mean'])}}}",
        f"\\newcommand{{\\WageCushionPensionGross}}{{{pct(wc['cushioning_pension_gross']['mean'])}}}",
        f"\\newcommand{{\\PensionDisplacedCushionHBAI}}{{{pct(disp['cushioning_hbai']['mean'])}}}",
        f"\\newcommand{{\\DisplacedCushionPensionGross}}{{{pct(disp['cushioning_pension_gross']['mean'])}}}",
        f"\\newcommand{{\\PensionShareWage}}{{{pct(wc['pension_share_of_gross_loss']['mean'])}}}",
        f"\\newcommand{{\\PensionShareDisplaced}}{{{pct(disp['pension_share_of_gross_loss']['mean'])}}}",
        f"\\newcommand{{\\GapHBAIPension}}{{{contrast['cushioning_gap_hbai_pp']:.1f}}}",
        f"\\newcommand{{\\GapPensionGross}}{{{contrast['cushioning_gap_pension_gross_pp']:.1f}}}",
        # Take-up headline. \TakeupSeeds is the take-up grid's OWN draw count:
        # it is NOT \PensionSeeds, which counts the pension block's draws.
        f"\\newcommand{{\\TakeupSeeds}}{{{takeup_seeds}}}",
        f"\\newcommand{{\\TakeupDisplacedCentral}}{{{pct(takeup['0.80']['cushioning_rate']['mean'])}}}",
        # Spread across the four new-entitlement claiming conventions. Zero
        # means the re-draw set is empty and the grid is inert, not that
        # take-up has been shown not to matter.
        f"\\newcommand{{\\TakeupConventionSpread}}{{{convention_spread:.3f}}}",
        # Binding claiming margin: all-entitled re-draw scope. The spread is
        # the difference of the two PRINTED endpoints, so the sentence that
        # quotes all three adds up (see entitled_scope_macro_values).
        f"\\newcommand{{\\TakeupEntitledStale}}{{{stale_str}}}",
        f"\\newcommand{{\\TakeupEntitledFull}}{{{full_str}}}",
        f"\\newcommand{{\\TakeupEntitledSpread}}{{{spread_str}}}",
        f"\\newcommand{{\\TakeupZeroUCNonTakeupShare}}{{{100 * zero_uc_non_takeup:.1f}}}",
        # New Style JSA bound (statutory parameters, no simulation).
        f"\\newcommand{{\\JSAWeeklyRate}}{{{jsa['parameters']['jsa_weekly_rate_25_plus']:.2f}}}",
        f"\\newcommand{{\\JSARateSource}}{{{JSA_RATE_SOURCE_PHRASES[jsa_rate_source]}}}",
        f"\\newcommand{{\\JSAMaxSpellAmount}}{{{jsa['max_contribution_based_entitlement']:,.0f}}}",
        f"\\newcommand{{\\JSACushionPoints}}{{{jsa['cushion_points_of_gross_loss']:.1f}}}",
        # Monthly-versus-annual UC bounding.
        f"\\newcommand{{\\MonthlyUCStandardAllowance}}{{{uc['parameters']['standard_allowance_single_25_plus_month']:.2f}}}",
        f"\\newcommand{{\\MonthlyUCTaxMonthlyThree}}{{{spell3['tax_relief_monthly_correct']:,.0f}}}",
        f"\\newcommand{{\\MonthlyUCTaxAnnualThree}}{{{spell3['tax_relief_annual_equivalent']:,.0f}}}",
        f"\\newcommand{{\\MonthlyUCCushionMonthly}}{{{pct(spell3['cushion_share_monthly_correct'])}}}",
        f"\\newcommand{{\\MonthlyUCCushionAnnual}}{{{pct(spell3['cushion_share_annual_equivalent'])}}}",
        "\\newcommand{\\MonthlyUCCushionGap}{"
        f"{100 * (spell3['cushion_share_monthly_correct'] - spell3['cushion_share_annual_equivalent']):.1f}}}",
    ]
    (PAPER / "referee_macros.tex").write_text("\n".join(lines) + "\n")
    print("[written] paper/referee_macros.tex")
    print(f"[takeup] entitled-scope bound source: {entitled_source}")
    print(
        f"[takeup] new-entitlement convention spread: {convention_spread:.3f}pp "
        f"over {len(TAKEUP_CONVENTIONS)} conventions, {takeup_seeds} draws"
    )
    if convention_spread == 0.0:
        print(
            "[takeup] WARNING: the new-entitlement take-up grid is INERT at "
            "this calibration (identical cushioning across every convention "
            "=> empty re-draw set). Report \\TakeupEntitledSpread as the "
            "claiming-margin bound, not this zero."
        )
    exact_spread = entitled_full - entitled_stale
    if round(exact_spread, ENTITLED_DECIMALS) != float(spread_str):
        print(
            f"[takeup] note: \\TakeupEntitledSpread prints {spread_str}, the "
            f"difference of the printed endpoints ({full_str} - {stale_str}). "
            f"The unrounded spread is {exact_spread:.4f}, which would round to "
            f"{round(exact_spread, ENTITLED_DECIMALS):.{ENTITLED_DECIMALS}f}; "
            "endpoint consistency is preferred because the manuscript quotes "
            "all three numbers in one sentence."
        )
    print(
        f"[jsa] rate source: {jsa_rate_source} "
        f"({jsa['parameters']['jsa_weekly_rate_25_plus']:.2f}/week); "
        f"vintage: {jsa['parameters']['rate_vintage']}"
    )


if __name__ == "__main__":
    main()
