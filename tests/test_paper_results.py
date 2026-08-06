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
