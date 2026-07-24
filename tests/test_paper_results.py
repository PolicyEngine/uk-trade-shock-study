from pathlib import Path

from analysis.write_paper_results import CENTRAL


def test_central_result_artifacts_are_declared() -> None:
    results = Path("results")
    assert all((results / f"{name}.json").exists() for name in CENTRAL)


def test_manuscript_loads_generated_results() -> None:
    main = Path("paper/main.tex").read_text()
    assert r"\input{generated_results}" in main
