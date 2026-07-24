import re
from pathlib import Path

from analysis.write_paper_results import CENTRAL


def test_central_result_artifacts_are_declared() -> None:
    results = Path("results")
    assert all((results / f"{name}.json").exists() for name in CENTRAL)


def test_manuscript_loads_generated_results() -> None:
    main = Path("paper/main.tex").read_text()
    assert r"\input{generated_results}" in main


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
