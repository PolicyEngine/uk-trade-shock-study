"""The frozen-input manifest, and what it actually pins.

This test used to be ``assert validate() == []`` and nothing else, which passed
trivially: all eight raw inputs are absent from a checkout (``validate()``
skips missing raw inputs by design) and ``derived_outputs`` held two packaged
CSVs, so NOTHING the manuscript quotes was hash-pinned. The manifest could pass
while every headline number had moved.

``quoted_results`` closes that: it pins the ``results/`` artifacts the paper's
generated macros are computed from. The tests below check that the pinned set
is the set the writers actually read, that it excludes the artifacts the paper
build itself rewrites, and that ``validate()`` really fails on drift.
"""

import json
from pathlib import Path

import pytest

from analysis.validate_manifest import MANIFEST, ROOT, refresh, sha256, validate

MANIFEST_DATA = json.loads(MANIFEST.read_text())


def test_frozen_manifest_matches_available_inputs():
    assert validate() == []


# ---------------------------------------------------------------------------
# The manifest pins what the manuscript quotes
# ---------------------------------------------------------------------------


def test_quoted_results_is_not_empty_and_every_entry_exists() -> None:
    entries = MANIFEST_DATA["quoted_results"]
    assert len(entries) >= 5
    for item in entries:
        assert (ROOT / item["path"]).exists(), item["path"]
        assert len(item["sha256"]) == 64
        # provenance: who quotes it, and which target regenerates it
        assert item["quoted_by"]
        assert item["produced_by"]


def test_every_quoted_result_is_pinned() -> None:
    """The result files behind this paper's generated macros are pinned."""
    pinned = {item["path"] for item in MANIFEST_DATA["quoted_results"]}
    for name in (
        "results.json",
        "second_stage_energy.json",
        "grid_energy.json",
        "e3_constants.json",
        "full_tariff_displacement.json",
        "full_tariff_wage_cut.json",
        "full_tariff_obr_low_displacement.json",
        "full_tariff_obr_low_wage_cut.json",
    ):
        assert f"results/{name}" in pinned, name


def test_the_paper_builds_own_outputs_are_not_pinned() -> None:
    """A file the next build step rewrites must not be hash-pinned.

    `make paper-values` runs `referee_fixes.py --only jsa` and
    `write_referee_macros.py` (both write results/referee_fixes.json) and
    `write_submission_results.py` (which writes results/submission_summary.json).
    Pinning either would make `make check` fail against itself.
    """
    build_products = set(MANIFEST_DATA["build_products"])
    assert build_products == {
        "results/referee_fixes.json",
        "results/submission_summary.json",
    }
    for section in ("derived_outputs", "quoted_results"):
        pinned = {item["path"] for item in MANIFEST_DATA[section]}
        assert not (pinned & build_products)


def test_pinned_results_are_tracked_by_git() -> None:
    """A pinned artifact that is gitignored would pin nothing for a reader."""
    import subprocess

    tracked = set(
        subprocess.run(
            ["git", "ls-files", "results"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
    )
    for item in MANIFEST_DATA["quoted_results"]:
        assert item["path"] in tracked, item["path"]


# ---------------------------------------------------------------------------
# validate() actually validates
# ---------------------------------------------------------------------------


def test_validate_detects_a_changed_quoted_result(tmp_path, monkeypatch) -> None:
    """The check the old test could never have exercised."""
    import analysis.validate_manifest as vm

    target = MANIFEST_DATA["quoted_results"][0]["path"]
    real = (ROOT / target).read_bytes()
    scratch = tmp_path / "repo"
    for item in (
        *MANIFEST_DATA["derived_outputs"],
        *MANIFEST_DATA["quoted_results"],
    ):
        dest = scratch / item["path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((ROOT / item["path"]).read_bytes())
    monkeypatch.setattr(vm, "ROOT", scratch)

    assert vm.validate() == []
    (scratch / target).write_bytes(real + b"\n")
    errors = vm.validate()
    assert any("hash mismatch" in e and target in e for e in errors)
    assert any("--refresh" in e for e in errors)


def test_validate_detects_a_missing_quoted_result(tmp_path, monkeypatch) -> None:
    import analysis.validate_manifest as vm

    monkeypatch.setattr(vm, "ROOT", tmp_path)
    errors = vm.validate()
    assert any("missing quoted results" in e for e in errors)
    assert any("missing derived outputs" in e for e in errors)
    # ...but a missing RAW input is still tolerated: they are not distributed
    assert not any("raw input" in e for e in errors)


def test_refresh_is_a_no_op_on_a_current_manifest() -> None:
    """`--refresh` must not churn the manifest when nothing has moved."""
    before = MANIFEST.read_text()
    try:
        assert refresh() == []
    finally:
        MANIFEST.write_text(before)


@pytest.mark.parametrize("section", ["inputs", "derived_outputs", "quoted_results"])
def test_no_duplicate_paths_within_a_section(section: str) -> None:
    key = "local_path" if section == "inputs" else "path"
    paths = [item[key] for item in MANIFEST_DATA[section]]
    assert len(paths) == len(set(paths))


def test_the_two_packaged_csvs_are_still_pinned() -> None:
    paths = {item["path"] for item in MANIFEST_DATA["derived_outputs"]}
    assert paths == {
        "uk_trade_shock_study/data/us_export_intensity_by_sic.csv",
        "uk_trade_shock_study/data/measured_export_falls_by_sic.csv",
    }


def test_manifest_paths_are_repository_relative() -> None:
    for section, key in (
        ("inputs", "local_path"),
        ("derived_outputs", "path"),
        ("quoted_results", "path"),
    ):
        for item in MANIFEST_DATA[section]:
            assert not Path(item[key]).is_absolute()
            assert ".." not in Path(item[key]).parts
