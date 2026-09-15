"""Download the FRS microdata from PolicyEngine's Hugging Face repo.

Requires HUGGING_FACE_TOKEN in the environment (token with access to
policyengine/policyengine-uk-data). Files land in data/ (gitignored).

PINNED BY REVISION AND VERIFIED BY HASH. The upstream repo republishes
frs_2024_25.h5 under the same filename on every data release: the file that
produced these results (release 1.56.6, 2026-06-19) was replaced by 1.56.12 on
2026-07-21 and again by 1.56.14 on 2026-07-26. A bare download of the current
file therefore does NOT reproduce the paper -- it silently shifts every
simulated cell. This script pins the revision recorded in the input manifest
and refuses to proceed if the bytes do not hash to the manifest's sha256, so
the failure is loud instead of showing up as numbers that nearly match.

Pass --latest to deliberately fetch the current upstream file instead. That is
useful for checking whether results survive a data refresh; it is not a
reproduction of the published results, and the script says so.
"""

import argparse
import hashlib
import json
import os
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "uk_trade_shock_study" / "data" / "input_manifest.json"
REPO = "policyengine/policyengine-uk-data"
FILES = ("frs_2024_25.h5", "frs_2024_25.zip")

#: Manifest entry carrying the pinned revision and expected hash.
PINNED_ID = "frs_2024_25"


def pinned_entry() -> dict:
    manifest = json.loads(MANIFEST.read_text())
    for entry in manifest["inputs"]:
        if entry["id"] == PINNED_ID:
            return entry
    raise KeyError(f"{MANIFEST} has no input with id {PINNED_ID!r}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--latest",
        action="store_true",
        help=(
            "fetch the current upstream file instead of the pinned revision; "
            "does NOT reproduce the published results"
        ),
    )
    args = parser.parse_args(argv)

    token = os.environ["HUGGING_FACE_TOKEN"]
    entry = pinned_entry()
    revision = None if args.latest else entry.get("upstream_revision")
    if not args.latest and not revision:
        raise KeyError(
            f"{MANIFEST} entry {PINNED_ID!r} has no `upstream_revision`; it is "
            "required to reproduce the published results. Re-pin it, or run "
            "with --latest to accept whatever upstream currently serves."
        )

    data = Path("data")
    data.mkdir(exist_ok=True)
    for name in FILES:
        path = hf_hub_download(
            REPO, name, revision=revision, token=token, local_dir=data
        )
        print(path)

    h5 = data / "frs_2024_25.h5"
    digest = sha256(h5)
    if args.latest:
        print(f"\n--latest: {h5.name} sha256 {digest}")
        if digest != entry["sha256"]:
            print(
                "  This does NOT match the manifest hash, so results built "
                "from it are not the published ones. Expected "
                f"{entry['sha256']}."
            )
    elif digest != entry["sha256"]:
        raise SystemExit(
            f"\nFATAL: {h5} hashes to\n  {digest}\nbut the manifest pins\n  "
            f"{entry['sha256']}\nat revision {revision}. Upstream may have "
            "rewritten history, or the download is corrupt. Do not run the "
            "pipeline against this file: it will produce numbers that differ "
            "from the published results without any other visible symptom."
        )
    else:
        print(f"\nsha256 verified against the manifest: {digest}")

    with zipfile.ZipFile(data / "frs_2024_25.zip") as zf:
        zf.extractall(data / "frs_2024_25")
    print(
        "extracted adult.tab:",
        (data / "frs_2024_25" / "UKDA-9563-tab" / "tab" / "adult.tab").exists(),
    )


if __name__ == "__main__":
    main()
