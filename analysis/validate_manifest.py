"""Validate frozen input and derived-output hashes.

Missing non-distributed raw inputs are reported but do not fail validation.
Committed derived outputs must always exist and match.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "uk_trade_shock_study" / "data" / "input_manifest.json"


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
    for item in manifest["derived_outputs"]:
        path = ROOT / item["path"]
        if not path.exists():
            errors.append(f"missing derived output: {item['path']}")
        elif sha256(path) != item["sha256"]:
            errors.append(f"derived output hash mismatch: {item['path']}")
    return errors


def main() -> None:
    errors = validate()
    if errors:
        raise SystemExit("\n".join(errors))
    print("input manifest valid")


if __name__ == "__main__":
    main()
