"""Validate frozen input, derived-output and quoted-result hashes.

Three sections, three contracts:

``inputs``
    Raw upstream files. Most are licensed or cached and are NOT distributed,
    so a MISSING file is reported but does not fail validation; a file that is
    present must match its hash.

``derived_outputs``
    Packaged intermediates committed to the repository. Must always exist and
    match.

``quoted_results``
    The ``results/`` artifacts the manuscript's generated macros are computed
    from. Must always exist and match.

Why ``quoted_results`` exists. Every number the paper prints is produced by a
writer in ``analysis/write_*.py`` from a JSON or CSV artifact in ``results/``.
Those artifacts are the ONLY reproducible object a reader gets: the FRS
microdata behind them is licensed and cannot be redistributed, so nobody
outside the project can regenerate them. Before this section the manifest
pinned two packaged CSVs and nothing the manuscript quotes, which meant the
manifest could pass while every headline number had silently moved.

What is deliberately NOT pinned, and why: artifacts that the paper build
itself rewrites. ``make paper-values`` re-runs
``analysis/referee_fixes.py --only jsa`` (which rewrites
``results/referee_fixes.json``), ``analysis/write_referee_macros.py`` (which
stamps provenance back into the same file) and
``analysis/write_submission_results.py`` (which writes
``results/submission_summary.json``). Pinning a file that the very next build
step rewrites would make ``make check`` fail against itself, and the hash
would record the previous build rather than the frozen evidence. Those two
are listed in ``build_products`` for the record, and the test suite asserts
that no pinned path is also a build product.

Hash-pinning is the right mechanism here rather than "git already tracks
them": git records that a file CHANGED, this records which vintage the
manuscript's printed numbers were computed from, and it fails a build that
regenerates an artifact without an explicit decision to re-freeze. Re-freeze
with ``python analysis/validate_manifest.py --refresh``, which prints every
hash it moves so the change is reviewable rather than silent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "uk_trade_shock_study" / "data" / "input_manifest.json"

#: Manifest sections whose entries must exist and whose hashes must match.
#: ``inputs`` is handled separately because missing raw inputs are expected.
PINNED_SECTIONS = ("derived_outputs", "quoted_results")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate() -> list[str]:
    manifest = json.loads(MANIFEST.read_text())
    errors: list[str] = []
    for item in manifest["inputs"]:
        path = ROOT / item["local_path"]
        if path.exists() and sha256(path) != item["sha256"]:
            errors.append(f"raw input hash mismatch: {item['local_path']}")
    build_products = set(manifest.get("build_products", []))
    for section in PINNED_SECTIONS:
        for item in manifest.get(section, []):
            rel = item["path"]
            path = ROOT / rel
            if rel in build_products:
                # A pinned path that the paper build rewrites would fail
                # against itself on the next `make check`.
                errors.append(
                    f"{section} entry {rel} is also listed in build_products; a "
                    "file the paper build regenerates must not be hash-pinned"
                )
                continue
            if not path.exists():
                errors.append(f"missing {section.replace('_', ' ')}: {rel}")
            elif sha256(path) != item["sha256"]:
                errors.append(
                    f"{section.replace('_', ' ')} hash mismatch: {rel}. If this "
                    "is an intended re-freeze, re-run "
                    "`python analysis/validate_manifest.py --refresh` and review "
                    "the printed hash changes."
                )
    return errors


def refresh() -> list[str]:
    """Re-pin every existing hash.

    Returns ONLY the hashes that actually moved, so an unchanged manifest
    returns an empty list and re-running is a no-op. Files that are absent
    (the non-distributed raw inputs, normally) keep their recorded hash and are
    reported separately: a missing file is not evidence that its hash changed.
    """
    manifest = json.loads(MANIFEST.read_text())
    changes: list[str] = []
    absent: list[str] = []
    for section, key in (
        ("inputs", "local_path"),
        ("derived_outputs", "path"),
        ("quoted_results", "path"),
    ):
        for item in manifest.get(section, []):
            path = ROOT / item[key]
            if not path.exists():
                absent.append(f"{section}: {item[key]} absent, hash left as-is")
                continue
            digest = sha256(path)
            if digest != item["sha256"]:
                changes.append(
                    f"{section}: {item[key]} {item['sha256'][:12]} -> {digest[:12]}"
                )
                item["sha256"] = digest
    for line in absent:
        print(f"note: {line}")
    if changes:
        MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    return changes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help=(
            "re-pin hashes for every listed file that exists, printing each "
            "change. Use after a deliberate regeneration; review the printed "
            "list before committing."
        ),
    )
    args = parser.parse_args()
    if args.refresh:
        changes = refresh()
        for line in changes:
            print(line)
        print(f"input manifest refreshed ({len(changes)} change(s))")
        return
    errors = validate()
    if errors:
        raise SystemExit("\n".join(errors))
    print("input manifest valid")


if __name__ == "__main__":
    main()
