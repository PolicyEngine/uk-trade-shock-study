"""Check the manuscript's hours-channel enumeration against policyengine-uk.

Section~\\ref{sec:results-factorial} claims that the employment-state step is
an *hours* effect, and it names the channels: `in_work` has exactly three
consumers (the Universal Credit childcare work condition plus two childcare
subsidies outside UC), and a separate set of gates reads `weekly_hours`.

An earlier draft asserted the wrong pair -- it named the Housing Benefit
disregard and the council-tax-reduction deduction as the two further
`in_work` gates, when those read `weekly_hours` and the actual second and
third `in_work` consumers are the DfE extended childcare entitlement and
HMRC Tax-Free Childcare. Prose about someone else's package drifts silently,
so it is checked here rather than trusted.

Skipped when policyengine-uk is not installed: the rest of the suite runs
without it, and the replication package must stay buildable without the
licensed FRS. When it *is* installed, this fails on drift.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest


pytest.importorskip(
    "policyengine_uk",
    reason="policyengine-uk not installed; hours-channel claims unverified",
)

RESULTS_TEX = Path(__file__).resolve().parents[1] / "paper/sections/results.tex"

#: The `in_work` consumers the manuscript names, by variable name.
CLAIMED_IN_WORK_CONSUMERS = {
    "uc_childcare_work_condition",
    "extended_childcare_entitlement_work_condition",
    "tax_free_childcare_treated_as_in_work",
}


def _variable_sources() -> dict[str, str]:
    """Map every policyengine-uk variable name to its formula source."""
    import policyengine_uk

    package_root = Path(inspect.getfile(policyengine_uk)).parent
    sources = {}
    for path in (package_root / "variables").rglob("*.py"):
        if path.name == "__init__.py":
            continue
        sources[path.stem] = path.read_text(errors="ignore")
    return sources


def _consumers_of(dependency: str) -> set[str]:
    """Variables whose formula reads ``dependency`` by name.

    Matches the quoted-string form PolicyEngine formulas use --
    ``person("in_work", period)`` -- so a variable's own class definition
    does not count itself.
    """
    pattern = re.compile(rf"""["']{re.escape(dependency)}["']""")
    return {
        name
        for name, source in _variable_sources().items()
        if name != dependency and pattern.search(source)
    }


def test_employment_status_is_read_by_no_tax_or_benefit_formula():
    """The manuscript's central claim about the near-zero residual.

    If any formula under `variables/` ever reads `employment_status`, the
    step stops being a pure hours effect and the prose is wrong.
    """
    consumers = _consumers_of("employment_status")
    assert not consumers, (
        "employment_status is now read by tax/benefit formulas "
        f"{sorted(consumers)}; the manuscript states its only consumer is the "
        "behavioural labour-supply module, which this design switches off."
    )


def test_in_work_consumers_are_the_three_the_manuscript_names():
    consumers = _consumers_of("in_work")
    assert consumers == CLAIMED_IN_WORK_CONSUMERS, (
        "the set of `in_work` consumers has drifted from what "
        "paper/sections/results.tex enumerates.\n"
        f"  package: {sorted(consumers)}\n"
        f"  claimed: {sorted(CLAIMED_IN_WORK_CONSUMERS)}\n"
        "Update the prose -- it names each of these individually."
    )


def test_manuscript_does_not_put_the_weekly_hours_gates_under_in_work():
    """Guard the specific error an earlier draft made."""
    prose = RESULTS_TEX.read_text()
    passage = prose[prose.index("Hours enter through") : prose.index("The measured step")]
    weekly_hours_only = _consumers_of("weekly_hours") - _consumers_of("in_work")
    assert weekly_hours_only, "expected weekly_hours to have its own consumers"
    assert "weekly\\_hours" in passage, (
        "the passage names in_work gates but never distinguishes the "
        f"weekly_hours ones ({sorted(weekly_hours_only)})"
    )
    # Housing Benefit is a weekly_hours gate, so it must be introduced only
    # AFTER the passage switches to weekly_hours -- not among the in_work
    # consumers, which is where an earlier draft put it.
    switch = passage.index("weekly\\_hours")
    assert "Housing Benefit" not in passage[:switch], (
        "the Housing Benefit disregard reads weekly_hours, not in_work, but "
        "the passage names it before it switches to weekly_hours gates"
    )
    assert "Housing Benefit" in passage[switch:], (
        "the passage no longer names the Housing Benefit worker disregard "
        "among the weekly_hours gates"
    )
