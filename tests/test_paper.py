"""Checks that run on public data only (no licensed microdata):

* every pinned raw input hashes to its recorded SHA256 and the statutory
  Universal Credit rates appear in the pinned source documents;
* every ``\\Ms...`` / ``\\Ss...`` macro the manuscript uses is defined by a
  generated macro file;
* the generated macro and table files under ``paper/`` are byte-identical to
  their ``results/`` originals;
* the first pass regenerates ``results/results.json`` and its macros
  byte-identically from the pinned inputs;
* the six-month Energy Price Guarantee aggregate equals the financial-year
  aggregate (the FY-mean caps already average the Guarantee's half-year).
"""

import hashlib
import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
PAPER = REPO / "paper"
RESULTS = REPO / "results"
SECTIONS = sorted((PAPER / "sections").glob("*.tex")) + [PAPER / "main.tex"]
GENERATED = sorted(RESULTS.glob("generated_*.tex")) + sorted(RESULTS.glob("table_*.tex"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_pinned_inputs_verify():
    from uk_trade_shock_study import incidence

    incidence.verify_inputs()
    incidence.verify_statutory_rates()


def test_every_manuscript_macro_is_defined():
    defined = set()
    for f in GENERATED:
        defined |= set(re.findall(r"\\newcommand\{\\((?:Ms|Ss)[A-Za-z]+)\}", f.read_text()))
    used = set()
    for f in SECTIONS + [PAPER / "figures" / "generated_table_firststages.tex",
                         PAPER / "figures" / "generated_table_decile.tex"]:
        if f.exists():
            used |= set(re.findall(r"\\((?:Ms|Ss)[A-Za-z]+)", f.read_text()))
    missing = sorted(used - defined)
    assert not missing, f"macros used but not defined: {missing}"


def test_paper_and_results_generated_files_match():
    for src in GENERATED:
        dst = PAPER / src.name
        assert dst.exists(), f"{dst} missing"
        assert dst.read_bytes() == src.read_bytes(), f"{src.name} differs between results/ and paper/"


def test_first_pass_reproduces_committed_results():
    from uk_trade_shock_study import incidence

    before = {p: _sha(p) for p in (RESULTS / "results.json", RESULTS / "generated_multishock.tex")}
    incidence.main()
    after = {p: _sha(p) for p in before}
    assert before == after, "first pass changed results.json or its macros"


def test_six_month_epg_equals_fy_aggregate():
    res = json.loads((RESULTS / "results.json").read_text())
    cushion = res["episodes"]["E2_energy"]["epg_cushion"]
    assert cushion["aggregate_gbp_bn_six_months"] == pytest.approx(
        cushion["aggregate_gbp_bn_per_year"], abs=0.05
    )
