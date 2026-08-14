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
        "the published DWP statutory rate for 2025--26, applied directly "
        "rather than read from the simulation's parameter tree; the 2026--27 "
        "uprating is not applied, which makes the bound conservative, since a "
        "higher uprated rate would only raise it"
    ),
}

#: Provenance tokens for the entitled-scope bound, mirroring
#: ``analysis/referee_fixes.ENTITLED_SCOPE_SOURCES`` (pinned equal by
#: tests/test_paper_results.py). Duplicated rather than imported for the same
#: reason as the JSA tokens: analysis/referee_fixes.py imports the simulation
#: stack, which this writer must never need.
ENTITLED_SCOPE_SOURCE_CURRENT = "referee_fixes_all_entitled_scope"
ENTITLED_SCOPE_SOURCE_LEGACY = "takeup_diagnosis_legacy_vintage"
#: How ``takeup_headline.entitled_scope_source`` reads in the manuscript.
#: \TakeupEntitledStale, \TakeupEntitledFull and \TakeupEntitledSpread come
#: from whichever artifact this writer could resolve, and the two artifacts
#: were produced at DIFFERENT calibrations, so the vintage has to be printed
#: beside the numbers rather than announced in a build log that no reader of
#: the paper ever sees. An unknown token raises: see JSA_RATE_SOURCE_PHRASES.
#:
#: File names are set with ``\texttt`` and an ESCAPED underscore, not with
#: ``\path``/``\url``: these phrases become ``\newcommand`` bodies, which TeX
#: tokenises at definition time, fixing a raw ``_`` as a subscript before
#: url.sty's verbatim catcode handling could ever run. It is also the
#: manuscript's own convention (e.g. \texttt{new\_entitlement}).
ENTITLED_SCOPE_SOURCE_PHRASES = {
    ENTITLED_SCOPE_SOURCE_CURRENT: (
        "the all-entitled re-draw grid in "
        "\\texttt{results/\\allowbreak referee\\_fixes.json}, computed at the current "
        "calibration"
    ),
    ENTITLED_SCOPE_SOURCE_LEGACY: (
        "the legacy \\texttt{results/\\allowbreak takeup\\_diagnosis.json} grid, which "
        "applies the same all-changed-units convention at a "
        "\\emph{superseded} calibration---the former $\\varepsilon=2$ high "
        "case, whose seed-0 "
        "gross earnings loss of \\pounds 1.70 billion is roughly twice the "
        "\\pounds 0.886 billion of the current unit stress"
    ),
}

#: Tokens for HOW the new-entitlement grid's inertness was established.
INERT_BASIS_MEASURED = "redraw_diagnostic"
INERT_BASIS_INFERRED = "convention_spread"
#: How the manuscript may state that basis. The two are NOT equivalent claims.
#: A measured ``n_redrawn`` of zero says no benefit unit was ever re-drawn. A
#: zero convention spread is only one-directional evidence for that: a
#: NON-empty re-draw set whose units all end up with a negligible award would
#: also return bit-identical cushioning, which is exactly the case the
#: diagnostic was built to separate. The macro therefore states which of the
#: two the printed grid actually rests on.
#: The grid is NOT inert: the diagnostic records benefit units being
#: re-drawn. Until the Universal Credit award-cache correction of 2026-08-13
#: this case could not arise -- a shocked simulation was served its BASELINE
#: award, so changing the claiming rate changed nothing and the grid returned
#: bit-identical cushioning at every convention. That artefact was read as the
#: new-entitlement scope having an empty domain. It does not: with the cache
#: correctly invalidated the same grid moves displacement cushioning by
#: several points. Emitting a phrase here rather than raising lets the
#: manuscript state what the grid now shows; the guard still raises if the
#: diagnostic and the spread contradict each other.
BASIS_LIVE = "live_grid"
TAKEUP_INERT_BASIS_PHRASES = {
    BASIS_LIVE: (
        "measured directly: the re-draw diagnostic stored with the grid "
        "records benefit units being re-drawn in most seeds, so the grid is "
        "live and the claiming rate is a binding parameter"
    ),
    INERT_BASIS_MEASURED: (
        "measured directly: the re-draw diagnostic stored with the grid "
        "records no benefit unit re-drawn in any seed"
    ),
    INERT_BASIS_INFERRED: (
        "inferred from bit-identical cushioning across all four claiming "
        "conventions; the direct re-draw count is not present in this artifact"
    ),
}
#: Field carrying the per-cell re-draw diagnostic written by
#: analysis/referee_fixes.py (``_summarise_diagnostics``).
REDRAW_DIAGNOSTIC_KEY = "redraw_diagnostic"


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
    ``new_entitlement`` scope and is emitted faithfully, so the manuscript
    quotes that zero instead of implying the grid tested something. Note what
    the zero does and does not establish: an empty re-draw set forces it, but
    it does not by itself PROVE the set was empty — a non-empty set whose
    units all end with a negligible award reads identically. Use
    ``takeup_inert_basis`` to say which of the two grounds the artifact
    actually supports. What is NOT tolerated is a missing field — that raises.
    """
    rates = [
        float(require(block, key, "cushioning_rate", "mean", where=where))
        for key in TAKEUP_CONVENTIONS
    ]
    return 100.0 * (max(rates) - min(rates))


def entitled_scope_bound(referee: dict, diagnosis: dict) -> tuple[float, float, str]:
    """(stale cushioning %, full-take-up cushioning %, source token).

    Prefers an ``all_entitled`` grid computed at the CURRENT calibration by
    analysis/referee_fixes.py. Falls back to the legacy
    results/takeup_diagnosis.json, which used the same all-changed-units
    convention but an OLDER calibration vintage (the former epsilon = 2 high
    case: seed-0 gross loss 1.70bn against the current unit-stress 0.886bn),
    and must therefore always be labelled as such in the manuscript.

    The third element is one of ``ENTITLED_SCOPE_SOURCE_PHRASES``' keys, not
    prose: the caller renders it through that table into \\TakeupEntitledSource
    and records it in the artifact. The fallback used to be announced only by a
    ``print()``, which meant three legacy-vintage numbers could ship in the
    macros with a clean exit code and nothing in the emitted file saying so.
    """
    scope = referee.get("takeup_headline", {}).get(ALL_ENTITLED_KEY)
    if isinstance(scope, dict) and "displacement" in scope:
        where = f"referee_fixes.json takeup_headline.{ALL_ENTITLED_KEY}.displacement"
        block = scope["displacement"]
        stale = float(
            require(block, "stale_baseline_flag", "cushioning_rate", "mean", where=where)
        )
        full = float(require(block, "1.00", "cushioning_rate", "mean", where=where))
        return 100.0 * stale, 100.0 * full, ENTITLED_SCOPE_SOURCE_CURRENT

    where = "takeup_diagnosis.json results.full_tariff_displacement"
    block = require(diagnosis, "results", "full_tariff_displacement", where=where)
    stale = float(require(block, "current_stale_flag", "cushioning_mean", where=where))
    full = float(require(block, "takeup_100", "cushioning_mean", where=where))
    return 100.0 * stale, 100.0 * full, ENTITLED_SCOPE_SOURCE_LEGACY


def entitled_scope_source_phrase(source: str) -> str:
    """Render the entitled-scope provenance, or raise on an unknown token."""
    if source not in ENTITLED_SCOPE_SOURCE_PHRASES:
        raise KeyError(
            f"entitled-scope bound resolved to source {source!r}, which this "
            f"writer has no manuscript phrasing for (known: "
            f"{sorted(ENTITLED_SCOPE_SOURCE_PHRASES)}). Add the phrasing "
            "rather than emitting three cushioning numbers with an unlabelled "
            "calibration vintage."
        )
    return ENTITLED_SCOPE_SOURCE_PHRASES[source]


def takeup_inert_basis(block: dict, convention_spread_pp: float, where: str) -> str:
    """Token naming HOW the new-entitlement grid was shown to be inert.

    The manuscript's central take-up claim is that no benefit unit is ever
    re-drawn under the published ``new_entitlement`` scope. There are two very
    different grounds for saying so, and the paper should not print one while
    holding the other:

    * MEASURED — every cell carries the ``redraw_diagnostic`` that
      shocks.uc_takeup_redraw_diagnostic produced, and its ``n_redrawn`` is
      zero in every seed. This is a direct observation of the re-draw set.
    * INFERRED — the artifact has no diagnostic and inertness is read off an
      exact-zero spread of simulated cushioning rates. That inference runs one
      way only: an empty re-draw set implies a zero spread, but a zero spread
      does NOT imply an empty re-draw set (a non-empty set whose units all end
      with a negligible award gives the same zero).

    The diagnostic wins whenever it is present. A diagnostic that CONTRADICTS
    inertness, or a missing diagnostic beside a non-zero spread, raises: in
    neither case is the grid inert, so there is no honest phrase to emit.
    """
    diagnostics = {
        key: block[key][REDRAW_DIAGNOSTIC_KEY]
        for key in TAKEUP_CONVENTIONS
        if isinstance(block.get(key), dict)
        and isinstance(block[key].get(REDRAW_DIAGNOSTIC_KEY), dict)
    }
    if diagnostics:
        if len(diagnostics) != len(TAKEUP_CONVENTIONS):
            raise KeyError(
                f"{where}: only {sorted(diagnostics)} of the "
                f"{len(TAKEUP_CONVENTIONS)} claiming conventions carry a "
                f"{REDRAW_DIAGNOSTIC_KEY!r}. 'No benefit unit is re-drawn in "
                "any seed' cannot be measured from a subset of the grid; "
                "re-run analysis/referee_fixes.py --only takeup."
            )
        counts = {
            key: int(require(diag, "n_redrawn_max", where=f"{where} {key}"))
            for key, diag in diagnostics.items()
        }
        live = {key: n for key, n in counts.items() if n > 0}
        if live:
            # A non-empty re-draw set with a ZERO spread is the pathological
            # case the diagnostic exists to catch: units were re-drawn and
            # nothing moved, which is what a stale award cache looks like.
            # Refuse that one; a non-empty set with a real spread is simply a
            # live grid and is reported as such.
            if convention_spread_pp == 0.0:
                raise RuntimeError(
                    f"{where}: the diagnostic reports a NON-EMPTY re-draw set "
                    f"({live}) yet the claiming conventions return "
                    "bit-identical cushioning. Units were re-drawn and their "
                    "awards did not move, which is the signature of an award "
                    "served from a stale cache rather than recomputed. Do not "
                    "report either number; re-run the take-up grid."
                )
            return BASIS_LIVE
        if convention_spread_pp != 0.0:
            raise RuntimeError(
                f"{where}: the diagnostic says nothing was ever re-drawn, yet "
                f"the conventions differ by {convention_spread_pp:.6f}pp. An "
                "empty re-draw set forces bit-identical cushioning, so the "
                "artifact is internally inconsistent — re-run the take-up "
                "grid rather than reporting either number."
            )
        return INERT_BASIS_MEASURED
    if convention_spread_pp != 0.0:
        raise RuntimeError(
            f"{where}: this artifact carries no {REDRAW_DIAGNOSTIC_KEY!r}, and "
            f"the conventions differ by {convention_spread_pp:.6f}pp, so the "
            "grid is NOT inert on the only evidence available. The manuscript "
            "cannot claim an empty re-draw set here; report the spread."
        )
    return INERT_BASIS_INFERRED


def takeup_inert_basis_phrase(basis: str) -> str:
    """Render the inertness basis, or raise on an unknown token."""
    if basis not in TAKEUP_INERT_BASIS_PHRASES:
        raise KeyError(
            f"unknown take-up inertness basis {basis!r} (known: "
            f"{sorted(TAKEUP_INERT_BASIS_PHRASES)})."
        )
    return TAKEUP_INERT_BASIS_PHRASES[basis]


def check_entitled_scope_macros(
    stale_s: str,
    full_s: str,
    spread_s: str,
    *,
    stale: float,
    full: float,
    nd: int = ENTITLED_DECIMALS,
) -> None:
    """Raise unless the three PRINTED entitled-scope numbers are BOTH
    self-consistent and faithful to the unrounded artifact values.

    The manuscript quotes \\TakeupEntitledStale, \\TakeupEntitledFull and
    \\TakeupEntitledSpread in one sentence, so a reader subtracts the printed
    endpoints and expects the printed spread. Rounding all three independently
    from the unrounded values does not deliver that: 34.4505 and 41.5331 print
    as 34.5 and 41.5, whose difference is 7.0, while the unrounded spread
    7.0826 prints as 7.1. So the emitter derives the printed spread from the
    printed endpoints.

    That makes the printed-subtraction identity TRUE BY CONSTRUCTION as this
    module calls it, and a guard that re-derives its own input can never fire.
    The unrounded ``stale`` and ``full`` are therefore required arguments and
    the real work here is checking the printed strings against THEM:

    1. each string carries exactly ``nd`` decimals — the precision the
       tolerances below are tied to;
    2. each printed endpoint is within half a unit in the last place of its
       unrounded source, i.e. it really is that value rounded to ``nd``;
    3. the printed spread is within ONE unit in the last place of the exact
       ``full - stale`` — the worst case when both endpoints move half a unit
       in opposite directions. A spread that drifted further has stopped
       describing the artifact;
    4. and only then, that the printed numbers still subtract correctly.

    Checks 2 and 3 are the ones with teeth: they fail if a future edit prints
    an endpoint from the wrong field, at the wrong scale (a rate rather than a
    percentage), or from a stale variable.
    """
    ulp = 10.0 ** (-nd)
    # 1e-9 absorbs binary representation error only; it is far below the ulp.
    half_ulp = ulp / 2.0 + 1e-9
    for name, text in (
        ("\\TakeupEntitledStale", stale_s),
        ("\\TakeupEntitledFull", full_s),
        ("\\TakeupEntitledSpread", spread_s),
    ):
        decimals = text.split(".")[1] if "." in text else ""
        if len(decimals) != nd:
            raise RuntimeError(
                f"entitled-scope macro {name} prints {text!r} with "
                f"{len(decimals)} decimal places, not {nd}. The tolerances "
                "that check these numbers against the artifact are tied to "
                "the print precision, so the precision cannot drift silently."
            )
    for name, text, source in (
        ("\\TakeupEntitledStale", stale_s, stale),
        ("\\TakeupEntitledFull", full_s, full),
    ):
        if abs(float(text) - source) > half_ulp:
            raise RuntimeError(
                f"entitled-scope macro {name} prints {text}, which is not "
                f"{source!r} rounded to {nd} decimal places (off by "
                f"{abs(float(text) - source):.6f}, tolerance {ulp / 2.0:g}). "
                "The printed endpoint has stopped describing the artifact "
                "value it claims to report."
            )
    exact_spread = full - stale
    if abs(float(spread_s) - exact_spread) > ulp + 1e-9:
        raise RuntimeError(
            f"\\TakeupEntitledSpread prints {spread_s}, but the unrounded "
            f"endpoints differ by {exact_spread:.6f} — further than the "
            f"{ulp:g} that rounding both endpoints to {nd} decimal places can "
            "explain. The printed spread is not this artifact's spread."
        )
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
    unrounded values, and the emitted strings are then checked back against
    the UNROUNDED inputs by ``check_entitled_scope_macros`` — so neither the
    inconsistency nor a printed number that has drifted off the artifact can
    ship silently.
    """
    stale_s = f"{stale:.{nd}f}"
    full_s = f"{full:.{nd}f}"
    spread_s = f"{float(full_s) - float(stale_s):.{nd}f}"
    check_entitled_scope_macros(stale_s, full_s, spread_s, stale=stale, full=full, nd=nd)
    return stale_s, full_s, spread_s


#: Fields ``record_takeup_provenance`` stamps under ``takeup_headline``.
ENTITLED_SCOPE_SOURCE_FIELD = "entitled_scope_source"
INERT_BASIS_FIELD = "inert_basis"


def takeup_provenance_fields(source: str, basis: str) -> dict:
    """The provenance block recorded under ``takeup_headline``.

    Rendering the provenance into the macros puts it in front of a reader of
    the PAPER; this puts it in front of a reader of the ARTIFACT. Both matter,
    because the artifact outlives any one build and the fallback that chooses
    ``source`` leaves no other trace in it.
    """
    return {
        ENTITLED_SCOPE_SOURCE_FIELD: source,
        f"{ENTITLED_SCOPE_SOURCE_FIELD}_options": sorted(ENTITLED_SCOPE_SOURCE_PHRASES),
        f"{ENTITLED_SCOPE_SOURCE_FIELD}_detail": entitled_scope_source_phrase(source),
        INERT_BASIS_FIELD: basis,
        f"{INERT_BASIS_FIELD}_options": sorted(TAKEUP_INERT_BASIS_PHRASES),
        f"{INERT_BASIS_FIELD}_detail": takeup_inert_basis_phrase(basis),
        "provenance_recorded_by": "analysis/write_referee_macros.py",
    }


def record_takeup_provenance(path: Path, source: str, basis: str) -> bool:
    """Stamp the resolved provenance into results/referee_fixes.json in place.

    Returns True when the file changed. Everything else in the artifact is
    round-tripped untouched, and re-running with the same inputs is a no-op,
    so this never churns the file.
    """
    artifact = json.loads(path.read_text())
    headline = artifact.setdefault("takeup_headline", {})
    fields = takeup_provenance_fields(source, basis)
    if all(headline.get(k) == v for k, v in fields.items()):
        return False
    headline.update(fields)
    path.write_text(json.dumps(artifact, indent=2))
    return True


def main() -> None:
    d = json.loads((RESULTS / "referee_fixes.json").read_text())
    diagnosis = json.loads((RESULTS / "takeup_diagnosis.json").read_text())
    pension = d["pension_channel"]
    wc = pension["full_tariff_wage_cut"]
    disp = pension["full_tariff_displacement"]
    contrast = pension["contrast"]
    takeup = d["takeup_headline"]["displacement"]
    uc = d["monthly_uc_bounding"]
    sched = d["schedule_benchmark"]
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
    # How the "the grid is inert" claim is established for THIS artifact:
    # a measured re-draw count when one is stored, otherwise the weaker
    # inference from an exactly-zero convention spread.
    inert_basis = takeup_inert_basis(
        takeup, convention_spread, "referee_fixes.json takeup_headline.displacement"
    )
    # The binding claiming margin (all-entitled re-draw scope).
    entitled_stale, entitled_full, entitled_source = entitled_scope_bound(d, diagnosis)
    entitled_source_phrase = entitled_scope_source_phrase(entitled_source)
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

    # PROVENANCE OF THE ZERO-UC SHARE, SEPARATELY FROM THE ENTITLED SPREAD.
    # These two quantities come from DIFFERENT artifacts and an earlier
    # revision attributed both to \TakeupEntitledSource, which now names the
    # current all-entitled grid. The zero-UC share does not come from there:
    # it is read from results/takeup_diagnosis.json, a single seed at the
    # superseded eps=2 calibration, computed BEFORE the Universal Credit
    # award-cache correction. The deterministic wage-cut cell dates it beyond
    # argument -- it reproduces exactly across the cache fix, so a value that
    # differs from the current one is a different calibration outright.
    legacy_wage_cut = float(
        require(
            diagnosis, "headline", "full_tariff_wage_cut_cushioning",
            where="takeup_diagnosis.json",
        )
    )
    zero_uc_source = (
        "a single seed of the legacy \\texttt{results/\\allowbreak takeup\\_diagnosis.json} "
        "grid, at the superseded $\\varepsilon=2$ calibration (its "
        "deterministic wage-cut cell returns "
        f"{100 * legacy_wage_cut:.1f} per cent against the "
        "\\SubmissionUnitWageCushion\\ per cent of the current design) and "
        "on the pipeline as it stood before the Universal Credit award-cache "
        "correction; no diagnostic of this kind has been computed at the unit "
        "stress"
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
        # Spread across the four new-entitlement claiming conventions. ONE
        # decimal: this is a max-minus-min over a 25-seed Bernoulli grid whose
        # per-cell assignment SDs are around four points, so a third decimal
        # asserts a precision the design cannot support -- and the discussion
        # tells the reader in as many words that the contrast is "not a figure
        # known to a decimal place". \TakeupInertBasis states whether the grid
        # is live and how that was established.
        f"\\newcommand{{\\TakeupConventionSpread}}{{{convention_spread:.1f}}}",
        # HOW that zero was established: measured re-draw count, or inferred
        # from the spread. The two are different claims (see
        # takeup_inert_basis) and the manuscript must state which it holds.
        f"\\newcommand{{\\TakeupInertBasis}}{{{takeup_inert_basis_phrase(inert_basis)}}}",
        # Size of the re-draw set, so the manuscript can say how large the
        # live domain is rather than only that it is non-empty. Both are read
        # off the same diagnostic the basis phrase above is derived from.
        f"\\newcommand{{\\TakeupRedrawnBenunits}}{{{takeup['0.80'][REDRAW_DIAGNOSTIC_KEY]['n_redrawn_mean']:.1f}}}",
        f"\\newcommand{{\\TakeupRedrawnMax}}{{{int(takeup['0.80'][REDRAW_DIAGNOSTIC_KEY]['n_redrawn_max'])}}}",
        f"\\newcommand{{\\TakeupRedrawnWeighted}}{{{takeup['0.80'][REDRAW_DIAGNOSTIC_KEY]['weighted_redrawn_mean']/1000:.1f}}}",
        f"\\newcommand{{\\TakeupLowConvention}}{{{pct(takeup['0.55']['cushioning_rate']['mean'])}}}",
        f"\\newcommand{{\\TakeupFullConvention}}{{{pct(takeup['1.00']['cushioning_rate']['mean'])}}}",
        # Binding claiming margin: all-entitled re-draw scope. The spread is
        # the difference of the two PRINTED endpoints, so the sentence that
        # quotes all three adds up (see entitled_scope_macro_values).
        f"\\newcommand{{\\TakeupEntitledStale}}{{{stale_str}}}",
        f"\\newcommand{{\\TakeupEntitledFull}}{{{full_str}}}",
        f"\\newcommand{{\\TakeupEntitledSpread}}{{{spread_str}}}",
        # ...and WHICH artifact, at which calibration, those three came from.
        f"\\newcommand{{\\TakeupEntitledSource}}{{{entitled_source_phrase}}}",
        f"\\newcommand{{\\TakeupZeroUCNonTakeupShare}}{{{100 * zero_uc_non_takeup:.1f}}}",
        f"\\newcommand{{\\TakeupZeroUCSource}}{{{zero_uc_source}}}",
        # New Style JSA bound (statutory parameters, no simulation).
        f"\\newcommand{{\\JSAWeeklyRate}}{{{jsa['parameters']['jsa_weekly_rate_25_plus']:.2f}}}",
        f"\\newcommand{{\\JSARateSource}}{{{JSA_RATE_SOURCE_PHRASES[jsa_rate_source]}}}",
        f"\\newcommand{{\\JSAMaxSpellAmount}}{{{jsa['max_contribution_based_entitlement']:,.0f}}}",
        f"\\newcommand{{\\JSACushionPoints}}{{{jsa['cushion_points_of_gross_loss']:.1f}}}",
        # Statutory schedule benchmark: the one-worker arithmetic that
        # predicts the sign and rough size of the headline contrast before any
        # simulation. Emitted so the manuscript can be checked in three lines.
        f"\\newcommand{{\\ScheduleMarginalRate}}{{{100 * sched['marginal_deduction_rate']:.1f}}}",
        f"\\newcommand{{\\ScheduleAverageRate}}{{{100 * sched['average_deduction_rate']:.1f}}}",
        f"\\newcommand{{\\ScheduleImpliedGap}}{{{sched['implied_gap_percentage_points']:.1f}}}",
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
    # The provenance now travels with the artifact as well as the macros, so a
    # legacy-vintage fallback is visible to a reader of either.
    changed = record_takeup_provenance(
        RESULTS / "referee_fixes.json", entitled_source, inert_basis
    )
    print(
        f"[written] results/referee_fixes.json takeup_headline provenance "
        f"({'updated' if changed else 'already current'})"
    )
    print(f"[takeup] entitled-scope bound source: {entitled_source}")
    print(f"[takeup]   \\TakeupEntitledSource: {entitled_source_phrase}")
    if entitled_source == ENTITLED_SCOPE_SOURCE_LEGACY:
        print(
            "[takeup] WARNING: \\TakeupEntitledStale/Full/Spread are "
            "LEGACY-VINTAGE numbers from results/takeup_diagnosis.json (the "
            "superseded epsilon=2 high case, seed-0 gross loss 1.70bn against "
            "the current unit stress 0.886bn). results/referee_fixes.json "
            f"carries no takeup_headline.{ALL_ENTITLED_KEY} block; re-run "
            "analysis/referee_fixes.py --only takeup to replace them. The "
            "manuscript must quote \\TakeupEntitledSource beside them."
        )
    print(
        f"[takeup] new-entitlement convention spread: {convention_spread:.3f}pp "
        f"over {len(TAKEUP_CONVENTIONS)} conventions, {takeup_seeds} draws"
    )
    print(f"[takeup] inertness basis: {inert_basis}")
    print(f"[takeup]   \\TakeupInertBasis: {takeup_inert_basis_phrase(inert_basis)}")
    if convention_spread == 0.0:
        print(
            "[takeup] WARNING: the new-entitlement take-up grid is INERT at "
            "this calibration (identical cushioning across every convention "
            "=> empty re-draw set). Report \\TakeupEntitledSpread as the "
            "claiming-margin bound, not this zero."
        )
    if inert_basis == INERT_BASIS_INFERRED:
        print(
            "[takeup] WARNING: that zero is INFERRED, not measured. This "
            "artifact stores no redraw_diagnostic, and a zero spread is only "
            "one-directional evidence for an empty re-draw set (a non-empty "
            "set whose units all end with a negligible award reads the same). "
            "Re-run analysis/referee_fixes.py --only takeup to record "
            "n_redrawn directly."
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
