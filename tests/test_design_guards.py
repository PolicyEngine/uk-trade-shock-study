"""Design guards that are not draw counts.

F5: ``analysis/write_paper_results.py`` enforced a shared draw count across the
central artifacts but ignored the record-selection design. The 100-draw legacy
artifacts come from ``shocks.PRESETS``, which takes ``TradeShockScenario``'s
default ``selection_method="bernoulli"``; the 50-draw submission design passes
``"balanced"``. Those are different estimators.

F4: ``analysis/factorial_decomposition.py`` hardcoded the channel split's seed
count, while the manuscript twice promises re-running it at the full draw count
is cheap.
"""

import ast
import importlib.util
from pathlib import Path

import pytest

from analysis.write_paper_results import (
    SCENARIO_DEFAULT_SELECTION_METHOD,
    SELECTION_METHOD_PROVENANCE_FIX,
    _selection_method,
    check_selection_methods,
)


# ---------------------------------------------------------------------------
# F5: the assignment-design guard
# ---------------------------------------------------------------------------


def test_a_mixed_design_is_detected() -> None:
    with pytest.raises(ValueError, match="same record-selection design"):
        check_selection_methods(
            {
                "full_tariff_displacement": {"selection_method": "bernoulli"},
                "epd_displacement": {"selection_method": "balanced"},
            }
        )


def test_a_uniform_design_passes() -> None:
    recorded = check_selection_methods(
        {
            "a": {"selection_method": "bernoulli"},
            "b": {"selection_method": "bernoulli"},
        },
        expected="bernoulli",
    )
    assert set(recorded.values()) == {"bernoulli"}


def test_an_expected_design_that_does_not_match_is_detected() -> None:
    with pytest.raises(ValueError, match="--expected-selection-method"):
        check_selection_methods(
            {"a": {"selection_method": "bernoulli"}}, expected="balanced"
        )


def test_a_nested_design_block_is_read() -> None:
    """The shape results/factorial_decomposition.json already uses."""
    assert _selection_method({"design": {"selection_method": "balanced"}}) == "balanced"
    assert _selection_method({"selection_method": "balanced"}) == "balanced"
    assert _selection_method({"n_draws": 50}) is None
    assert _selection_method({"design": {"n_draws": 50}}) is None


def test_silence_is_not_read_as_the_scenario_default() -> None:
    """Assuming 'bernoulli' would let a balanced artifact pass unnoticed.

    An artifact that does not record its design has not been checked. This
    used to print a warning, because no shipped artifact carried the field
    and failing would have blocked every build. Since the re-run they all
    carry it, so silence now means the artifact is stale and the guard
    raises rather than narrating that it could not run.
    """
    with pytest.raises(ValueError) as excinfo:
        check_selection_methods({"a": {"n_draws": 100}, "b": {"n_draws": 100}})
    message = str(excinfo.value)
    assert "No central artifact records" in message
    assert SCENARIO_DEFAULT_SELECTION_METHOD in message
    assert SELECTION_METHOD_PROVENANCE_FIX in message


def test_a_partially_recorded_set_is_reported(capsys) -> None:
    check_selection_methods(
        {"a": {"selection_method": "bernoulli"}, "b": {"n_draws": 100}}
    )
    warning = capsys.readouterr().out
    assert "1 of 2 central artifacts record" in warning
    assert "['b']" in warning


def test_the_shipped_artifacts_record_the_assignment_design() -> None:
    """Every central artifact carries the design it was produced under.

    This replaces an earlier guard that asserted the opposite. The code side
    had long been done -- `MonteCarloResult` declared `selection_method`,
    `run_monte_carlo_prepared` set it, `write_result` serialised it -- but
    every shipped artifact predated the field, so the real guard in
    `check_selection_methods` could only warn that it was unable to run. The
    2026-08-13 re-run against the licensed FRS microdata landed the field
    everywhere, so the guard now runs for real and its absence is an error.
    """
    import json
    from pathlib import Path

    from analysis.write_paper_results import ANCHORS, CENTRAL

    recorded = {}
    for name in (*CENTRAL, *ANCHORS):
        item = json.loads(Path(f"results/{name}.json").read_text())
        recorded[name] = _selection_method(item)

    missing = sorted(n for n, v in recorded.items() if v is None)
    assert not missing, (
        f"central artifacts {missing} do not record selection_method; they "
        "predate the re-run and must not be quoted beside current ones. "
        "Re-run `make results`."
    )
    distinct = set(recorded.values())
    assert distinct == {"bernoulli"}, (
        "central artifacts must all use one assignment design; found "
        f"{sorted(distinct)}. These are built from shocks.PRESETS, which takes "
        "the TradeShockScenario default; the 50-draw submission family passes "
        "'balanced' explicitly and is compared against these in the manuscript."
    )


# ---------------------------------------------------------------------------
# F4: the channel split's seed count is a flag, not a source constant
# ---------------------------------------------------------------------------
#
# analysis/factorial_decomposition.py imports mechanism_decomposition, which
# imports policyengine-uk, so these read the module as source rather than
# importing it: the flag's existence is a property of the file, not of a
# machine that happens to have the optional dependency installed.

FACTORIAL = Path(__file__).resolve().parents[1] / "analysis/factorial_decomposition.py"


def _factorial_source() -> str:
    return FACTORIAL.read_text()


def _factorial_ast() -> ast.Module:
    return ast.parse(_factorial_source())


def _function(name: str) -> ast.FunctionDef:
    for node in _factorial_ast().body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"analysis/factorial_decomposition.py has no {name}()")


def _module_constant(name: str):
    for node in _factorial_ast().body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"analysis/factorial_decomposition.py has no {name}")


def test_channel_seeds_is_a_cli_flag_with_the_previous_default() -> None:
    """The hardcoded ``CHANNEL_SEEDS = range(5)`` is now the flag's default."""
    source = _factorial_source()
    assert "--channel-seeds" in source
    assert _module_constant("DEFAULT_CHANNEL_SEEDS") == 5
    # the flag's default reproduces the previous behaviour, so no existing run
    # changes and the manuscript's current channel numbers stand
    main = _function("main")
    defaults = [
        kw.value
        for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and getattr(node.func, "attr", None) == "add_argument"
        and node.args
        and getattr(node.args[0], "value", None) == "--channel-seeds"
        for kw in node.keywords
        if kw.arg == "default"
    ]
    assert defaults, "--channel-seeds must declare an explicit default"
    assert getattr(defaults[0], "id", None) == "DEFAULT_CHANNEL_SEEDS"


def test_channel_split_takes_the_seed_count_as_an_argument() -> None:
    split = _function("channel_split")
    names = [a.arg for a in split.args.args]
    assert names[-1] == "n_seeds"
    assert getattr(split.args.defaults[-1], "id", None) == "DEFAULT_CHANNEL_SEEDS"
    # ...and the flag is threaded through to it, not silently dropped
    assert "args.channel_seeds" in ast.unparse(_function("main"))


def test_a_non_positive_seed_count_is_rejected_at_both_ends() -> None:
    """argparse and the function both refuse; neither may run zero seeds."""
    assert "at least 1" in ast.unparse(_function("channel_split"))
    assert "at least 1" in ast.unparse(_function("main"))


def test_channel_seed_count_is_recorded_in_the_artifact_schema() -> None:
    """A reader must be able to tell which run produced the channel numbers."""
    assert "'channel_seeds': args.channel_seeds" in ast.unparse(_function("main"))
    assert "'n_seeds': n_seeds" in ast.unparse(_function("channel_split"))


@pytest.mark.skipif(
    importlib.util.find_spec("policyengine_uk") is None,
    reason="analysis.factorial_decomposition imports policyengine-uk",
)
def test_channel_split_raises_on_zero_seeds_at_runtime() -> None:
    import analysis.factorial_decomposition as fd

    with pytest.raises(ValueError, match="at least 1"):
        fd.channel_split(None, None, None, "unit", 0)
