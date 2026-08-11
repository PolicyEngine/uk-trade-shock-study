import re
from pathlib import Path

from analysis.write_paper_results import ANCHORS, CENTRAL
from analysis.run_submission_scenarios import scenarios


def test_central_result_artifacts_are_declared() -> None:
    results = Path("results")
    assert all((results / f"{name}.json").exists() for name in CENTRAL)


def test_anchor_result_artifacts_are_declared() -> None:
    results = Path("results")
    assert all((results / f"{name}.json").exists() for name in ANCHORS)


def test_submission_result_artifacts_are_declared() -> None:
    results = Path("results/submission")
    assert all((results / f"{name}.json").exists() for name in scenarios())


def test_manuscript_loads_generated_results() -> None:
    main = Path("paper/main.tex").read_text()
    assert r"\input{generated_results}" in main
    assert r"\input{generated_lfs_benchmarks}" in main
    assert r"\input{generated_trade_benchmarks}" in main
    assert r"\input{generated_lfs_selection}" in main
    assert r"\input{generated_submission}" in main


def test_core_appendix_stays_in_main_and_rest_is_supplemented() -> None:
    """Referee major point 8: a short main paper plus an online supplement.

    The main manuscript keeps only the core appendix (provenance, shock
    mechanics, Monte Carlo conventions, duration scope, monthly-UC bounding);
    exploratory and benchmark material lives in the standalone supplement.
    """
    main = Path("paper/main.tex").read_text()
    assert r"\input{sections/appendix}" in main
    assert r"\input{generated_factorial}" in main
    supplement = Path("paper/supplement.tex").read_text()
    assert r"\input{sections/supplement_body}" in supplement
    appendix = Path("paper/sections/appendix.tex").read_text()
    body = Path("paper/sections/supplement_body.tex").read_text()
    for moved in (
        "HMRC destination-panel",
        "Constituency machinery",
        "Supply-chain sensitivity",
        "Demographic diagnostic",
        "Reallocation sensitivity",
    ):
        assert moved not in appendix
        assert moved in body


def test_abstract_is_at_most_150_words() -> None:
    main = Path("paper/main.tex").read_text()
    abstract = re.search(
        r"\\begin\{abstract\}(.*?)\\end\{abstract\}", main, re.DOTALL
    )
    assert abstract is not None
    text = abstract.group(1)
    text = re.sub(r"\\(?:Full\w+|ProductionDraws)", " value ", text)
    text = text.replace(r"\pounds", " pounds ")
    text = re.sub(r"\\[A-Za-z]+(?:\{[^}]*\})?", " ", text)
    text = re.sub(r"[$~{}\\]", " ", text)
    words = re.findall(r"\b[\w£–—'-]+\b", text)
    assert len(words) <= 150


def test_manuscript_uses_british_spelling() -> None:
    files = [
        Path("paper/main.tex"),
        *sorted(Path("paper/sections").glob("*.tex")),
    ]
    prose = "\n".join(
        path.read_text() for path in files if path.name != "references.tex"
    )
    american_forms = re.compile(
        r"\b(?:behavior|behavioral|centered|favor|modeled|modeling|"
        r"organization|organize|organized|realize|realized|summarize|toward)\b",
        re.IGNORECASE,
    )
    assert not american_forms.search(prose)


def test_manuscript_does_not_overstate_mixed_margin_calibration() -> None:
    files = [
        Path("paper/main.tex"),
        *sorted(Path("paper/sections").glob("*.tex")),
    ]
    prose = "\n".join(
        path.read_text() for path in files if path.name != "references.tex"
    ).lower()
    assert "rent-sharing-calibrated mixed scenario" not in prose
    assert "empirically calibrated mixed scenario" not in prose


def test_thin_support_exclusion_has_one_definition() -> None:
    """The main-table split and the headline contrast range must agree.

    \\SubmissionCushionDifferenceMin/Max are computed over MAIN-TABLE cells
    only; the OBR three-month cell the manuscript drops on thin-support
    grounds is quoted separately. Both consumers read the rule from
    analysis/write_submission_results.py, so this pins that they are in fact
    the same rule rather than two copies that can drift.
    """
    from analysis.write_referee_macros import split_submission_rows
    from analysis.write_submission_results import (
        THIN_CELL_CONTRAST_KEY,
        is_thin_support_contrast,
        is_thin_support_row,
    )

    assert THIN_CELL_CONTRAST_KEY == "obr_3m_wage_minus_displacement"
    assert is_thin_support_contrast(THIN_CELL_CONTRAST_KEY)
    assert is_thin_support_row(r"OBR & 3 & Displacement & 89 \\")
    assert not is_thin_support_row(r"OBR & 12 & Displacement & 354 \\")
    assert not is_thin_support_row(r"UNIT & 3 & Displacement & 222 \\")

    main_rows, thin_rows = split_submission_rows()
    assert thin_rows and not any(is_thin_support_row(r) for r in main_rows.splitlines())
    assert all(is_thin_support_row(r) for r in thin_rows.splitlines())


def test_headline_contrast_range_excludes_the_thin_cell() -> None:
    """Regression: the max used to come from the excluded OBR 3-month cell."""
    import json

    macros = Path("paper/generated_submission.tex").read_text()

    def macro(name: str) -> float:
        match = re.search(rf"\\newcommand\{{\\{name}\}}\{{([-\d.]+)\}}", macros)
        assert match is not None, f"{name} not emitted"
        return float(match.group(1))

    summary = json.loads(Path("results/submission_summary.json").read_text())
    contrasts = summary["contrasts"]
    thin = contrasts["obr_3m_wage_minus_displacement"]["cushioning_difference_pp"]
    main_values = [
        item["cushioning_difference_pp"]
        for key, item in contrasts.items()
        if key.endswith("wage_minus_displacement")
        and key != "obr_3m_wage_minus_displacement"
    ]
    assert macro("SubmissionCushionDifferenceMin") == round(min(main_values), 1)
    assert macro("SubmissionCushionDifferenceMax") == round(max(main_values), 1)
    assert macro("SubmissionThinCellCushionDifference") == round(thin, 1)
    # the excluded cell is genuinely outside the reported main-table range
    assert thin > max(main_values)


def test_takeup_and_pension_draw_counts_are_separate_macros() -> None:
    """\\PensionSeeds must not be reused to state the take-up grid's draws."""
    import json

    macros = Path("paper/referee_macros.tex").read_text()
    referee = json.loads(Path("results/referee_fixes.json").read_text())
    pension_seeds = referee["pension_channel"]["n_seeds"]
    takeup_seeds = referee["takeup_headline"]["displacement"]["n_seeds"]
    assert pension_seeds != takeup_seeds  # otherwise the test proves nothing
    assert rf"\newcommand{{\PensionSeeds}}{{{pension_seeds}}}" in macros
    assert rf"\newcommand{{\TakeupSeeds}}{{{takeup_seeds}}}" in macros


def test_takeup_macros_are_emitted_from_artifacts() -> None:
    for name in (
        "TakeupSeeds",
        "TakeupConventionSpread",
        "TakeupEntitledStale",
        "TakeupEntitledFull",
        "TakeupEntitledSpread",
        "TakeupZeroUCNonTakeupShare",
        "JSAWeeklyRate",
        "JSARateSource",
        "JSAMaxSpellAmount",
        "JSACushionPoints",
    ):
        assert rf"\newcommand{{\{name}}}" in Path("paper/referee_macros.tex").read_text()


def test_hmrc_figure_base_macros_are_emitted() -> None:
    text = Path("paper/generated_trade_benchmarks.tex").read_text()
    for name in ("HMRCFigureBase", "HMRCFigureBaseIsAnticipation"):
        assert rf"\newcommand{{\{name}}}" in text


def test_takeup_macro_writer_fails_loudly_on_a_missing_field() -> None:
    """Repo hard-fail convention: never substitute a hand-typed number."""
    import pytest

    from analysis.write_referee_macros import takeup_convention_spread_pp

    block = {"0.55": {"cushioning_rate": {"mean": 0.3}}}
    with pytest.raises(KeyError, match="0.80"):
        takeup_convention_spread_pp(block, "unit test")


def _macro(text: str, name: str) -> str:
    """Body of ``\\newcommand{\\name}{...}``, allowing nested braces."""
    start = text.index(rf"\newcommand{{\{name}}}{{") + len(
        rf"\newcommand{{\{name}}}{{"
    )
    depth, index = 1, start
    while depth:
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                break
        index += 1
    return text[start:index]


# ---------------------------------------------------------------------------
# The entitled-scope macros must be quotable in one sentence
# ---------------------------------------------------------------------------


def test_entitled_scope_macros_are_self_consistent_in_the_generated_file() -> None:
    """A reader subtracting the printed endpoints must get the printed spread.

    The manuscript quotes \\TakeupEntitledStale, \\TakeupEntitledFull and
    \\TakeupEntitledSpread in a single sentence. Rounding all three
    independently from the unrounded values gave 34.5, 41.5 and 7.1, so the
    subtraction visibly failed.
    """
    macros = Path("paper/referee_macros.tex").read_text()
    stale = float(_macro(macros, "TakeupEntitledStale"))
    full = float(_macro(macros, "TakeupEntitledFull"))
    spread_text = _macro(macros, "TakeupEntitledSpread")
    assert round(full - stale, 1) == round(float(spread_text), 1)
    # printed at the same precision as the endpoints it sits beside
    assert len(spread_text.split(".")[-1]) == len(
        _macro(macros, "TakeupEntitledFull").split(".")[-1]
    )


def test_entitled_scope_macro_values_derive_the_spread_from_the_endpoints() -> None:
    from analysis.write_referee_macros import entitled_scope_macro_values

    # The live values: 41.5331 - 34.4505 = 7.0826, which rounds to 7.1 while
    # the printed endpoints differ by 7.0. Endpoint consistency wins.
    assert entitled_scope_macro_values(34.4505, 41.5331) == ("34.5", "41.5", "7.0")
    # a case where endpoint rounding and spread rounding already agree
    assert entitled_scope_macro_values(30.02, 41.44) == ("30.0", "41.4", "11.4")


def test_entitled_scope_check_rejects_the_pre_fix_emission() -> None:
    """Exactly what used to ship: 34.5 and 41.5 quoted beside a spread of 7.1."""
    import pytest

    from analysis.write_referee_macros import check_entitled_scope_macros

    with pytest.raises(RuntimeError, match="not self-consistent"):
        check_entitled_scope_macros("34.5", "41.5", "7.1")
    # the emitted triple passes
    check_entitled_scope_macros("34.5", "41.5", "7.0")


# ---------------------------------------------------------------------------
# New Style JSA rate provenance
# ---------------------------------------------------------------------------


def test_jsa_rate_source_is_an_enum_and_reaches_the_manuscript() -> None:
    """\\JSAWeeklyRate can come from either source; the paper must say which.

    `make paper-values` re-runs `referee_fixes.py --only jsa` on every build
    and policyengine-uk is an optional dependency, so the printed rate flips
    between the model parameter and the statutory constant depending on the
    build machine.
    """
    import json

    from analysis.referee_fixes import JSA_RATE_SOURCES
    from analysis.write_referee_macros import JSA_RATE_SOURCE_PHRASES

    assert set(JSA_RATE_SOURCE_PHRASES) == set(JSA_RATE_SOURCES)

    parameters = json.loads(Path("results/referee_fixes.json").read_text())[
        "jsa_bounding"
    ]["parameters"]
    assert parameters["rate_source"] in JSA_RATE_SOURCES
    assert parameters["rate_vintage"]

    macros = Path("paper/referee_macros.tex").read_text()
    assert (
        _macro(macros, "JSARateSource")
        == JSA_RATE_SOURCE_PHRASES[parameters["rate_source"]]
    )


def test_jsa_macro_writer_rejects_an_unknown_rate_source() -> None:
    import copy
    import json

    import pytest

    from analysis.write_referee_macros import JSA_RATE_SOURCE_PHRASES, require

    d = copy.deepcopy(json.loads(Path("results/referee_fixes.json").read_text()))
    d["jsa_bounding"]["parameters"]["rate_source"] = "vibes"
    source = require(d, "jsa_bounding", "parameters", "rate_source", where="test")
    assert source not in JSA_RATE_SOURCE_PHRASES
    with pytest.raises(KeyError):
        require(
            {"jsa_bounding": {"parameters": {}}},
            "jsa_bounding",
            "parameters",
            "rate_source",
            where="test",
        )


def test_jsa_weekly_rate_is_validated_as_a_weekly_figure() -> None:
    """A parameter node with the wrong periodicity is wrong by 4.33x or 52x."""
    import pytest

    from analysis.referee_fixes import (
        JSA_WEEKLY_RATE_2025_26,
        JSA_WEEKLY_RATE_PLAUSIBLE_RANGE,
        _check_weekly_rate,
    )

    low, high = JSA_WEEKLY_RATE_PLAUSIBLE_RANGE
    assert low < JSA_WEEKLY_RATE_2025_26 < high
    assert _check_weekly_rate(JSA_WEEKLY_RATE_2025_26, "test") == JSA_WEEKLY_RATE_2025_26
    for wrong in (
        JSA_WEEKLY_RATE_2025_26 * 52,  # annual
        JSA_WEEKLY_RATE_2025_26 * 52 / 12,  # monthly
        JSA_WEEKLY_RATE_2025_26 / 7,  # daily
        0.0,
        float("nan"),
    ):
        with pytest.raises(RuntimeError, match="plausible weekly NS-JSA range"):
            _check_weekly_rate(wrong, "test")


def test_jsa_block_records_a_stable_provenance() -> None:
    from analysis.referee_fixes import JSA_RATE_SOURCES, jsa_block

    parameters = jsa_block()["parameters"]
    assert parameters["rate_source"] in JSA_RATE_SOURCES
    assert parameters["rate_source_options"] == list(JSA_RATE_SOURCES)
    # both branches' figures are recorded, so a build-machine flip is visible
    # in the artifact rather than only in the printed number
    assert parameters["statutory_fallback_rate"] == 92.05
    assert (
        parameters["weekly_rate_plausible_range"][0]
        < parameters["jsa_weekly_rate_25_plus"]
        < parameters["weekly_rate_plausible_range"][1]
    )


# ---------------------------------------------------------------------------
# Which base period does the STORED HMRC figure use?
# ---------------------------------------------------------------------------


def test_hmrc_figure_base_macro_describes_the_stored_csv() -> None:
    """The caption's base period is derived from the plotted data.

    results/figures/hmrc_destination_event_study.png and its monthly CSV were
    not regenerated when the estimator's NORMALISATION_BASE changed, so the
    stored figure is still normalised on the January--March 2025 front-running
    window while the estimator's default is calendar 2024. Deriving the macro
    from the CSV means a caption written against it can never contradict it.
    """
    import pandas as pd

    from analysis.write_trade_benchmark_results import (
        MONTHLY,
        detect_base_period,
        format_base_period,
    )

    frame = pd.read_csv(MONTHLY)
    (start, end), source = detect_base_period(frame)
    assert source in ("base_period column", "detected from normalised_gap")
    base = frame[(frame["month"] >= start) & (frame["month"] <= end)]
    assert len(base) == (end // 100 - start // 100) * 12 + (end % 100 - start % 100) + 1
    # normalising subtracts the base-window mean, so it is zero on that window
    assert abs(base["normalised_gap"].mean()) < 1e-9

    macros = Path("paper/generated_trade_benchmarks.tex").read_text()
    assert _macro(macros, "HMRCFigureBase") == format_base_period(start, end)


def test_hmrc_figure_base_warns_when_it_sits_in_the_anticipation_window() -> None:
    import pandas as pd

    from analysis.hmrc_destination_event_study import ANTICIPATION_WINDOW
    from analysis.write_trade_benchmark_results import (
        MONTHLY,
        base_anticipation_phrase,
        detect_base_period,
    )

    lo, hi = ANTICIPATION_WINDOW
    inside = base_anticipation_phrase(lo, hi)
    assert "\\emph{inside}" in inside and "front-running peak" in inside
    # calendar 2024, the estimator's clean default, must NOT carry the warning
    clean = base_anticipation_phrase(202401, 202412)
    assert clean.startswith("outside")
    assert "inside" not in clean
    # a base straddling the window edge is flagged as partly contaminated
    assert "partly inside" in base_anticipation_phrase(202410, 202502)

    # the emitted macro is exactly the phrase for the stored figure's base
    (start, end), _ = detect_base_period(pd.read_csv(MONTHLY))
    macros = Path("paper/generated_trade_benchmarks.tex").read_text()
    assert _macro(macros, "HMRCFigureBaseIsAnticipation") == base_anticipation_phrase(
        start, end
    )


def test_hmrc_figure_base_prefers_an_explicit_base_period_column() -> None:
    """When the revised estimator writes the column, the column wins."""
    import pandas as pd

    from analysis.write_trade_benchmark_results import detect_base_period

    frame = pd.read_csv(
        Path("results/hmrc_destination_event_study_monthly.csv")
    )
    # detection alone finds the stale front-running base...
    assert detect_base_period(frame)[0] == (202501, 202503)
    # ...but a base_period column is authoritative and overrides it.
    labelled = frame.assign(base_period="2024")
    assert detect_base_period(labelled) == ((202401, 202412), "base_period column")


def test_base_period_labels_round_trip() -> None:
    import pytest

    from analysis.write_trade_benchmark_results import (
        format_base_period,
        parse_base_period,
    )

    assert format_base_period(202401, 202412) == "calendar 2024"
    assert format_base_period(202501, 202503) == "January--March 2025"
    assert format_base_period(202501, 202501) == "January 2025"
    assert format_base_period(202411, 202502) == "November 2024--February 2025"

    for window in ((202401, 202412), (202501, 202503), (202501, 202501), (202411, 202502)):
        assert parse_base_period(format_base_period(*window)) == window
    # and the abbreviated labels hmrc_destination_event_study.format_month_window
    # writes into a base_period column
    assert parse_base_period("2024") == (202401, 202412)
    assert parse_base_period("Jan-Apr 2025") == (202501, 202504)
    assert parse_base_period("Jan 2025") == (202501, 202501)
    assert parse_base_period("Dec 2024-Feb 2025") == (202412, 202502)
    with pytest.raises(ValueError):
        parse_base_period("sometime in 2024")


def test_base_period_detection_refuses_to_guess() -> None:
    import pandas as pd
    import pytest

    from analysis.write_trade_benchmark_results import detect_base_period

    # no window averages to zero: the series was not normalised at all
    unnormalised = pd.DataFrame(
        {"month": [202401, 202402, 202403], "normalised_gap": [1.0, 2.0, 3.0]}
    )
    with pytest.raises(ValueError, match="cannot be identified"):
        detect_base_period(unnormalised)
    # two different labels cannot describe one figure
    conflicting = pd.DataFrame(
        {
            "month": [202401, 202402],
            "normalised_gap": [0.5, -0.5],
            "base_period": ["2024", "Jan-Apr 2025"],
        }
    )
    with pytest.raises(ValueError, match="different base_period"):
        detect_base_period(conflicting)


def test_takeup_convention_spread_emits_zero_faithfully() -> None:
    """An inert grid must report 0, not be silently patched over."""
    from analysis.write_referee_macros import (
        TAKEUP_CONVENTIONS,
        takeup_convention_spread_pp,
    )

    flat = {k: {"cushioning_rate": {"mean": 0.32987738933694183}} for k in TAKEUP_CONVENTIONS}
    assert takeup_convention_spread_pp(flat, "unit test") == 0.0
    moved = dict(flat, **{"1.00": {"cushioning_rate": {"mean": 0.4}}})
    assert takeup_convention_spread_pp(moved, "unit test") > 0.0
