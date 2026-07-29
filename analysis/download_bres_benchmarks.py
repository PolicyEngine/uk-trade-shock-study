"""Download public BRES manufacturing employment benchmarks from Nomis.

No API key or account is required. The open estimates are disclosure-rounded,
so they are suitable for aggregate validation but are not business microdata.
"""

from __future__ import annotations

import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "public" / "bres_manufacturing_employment_2015_2024.csv"
META = OUT.with_suffix(".metadata.json")
DATASET = "NM_189_1"
API = f"https://www.nomisweb.co.uk/api/v01/dataset/{DATASET}"
GEOGRAPHIES = {
    "2092957698": "Great Britain",
    "2092957699": "England",
    "2092957700": "Wales",
    "2092957701": "Scotland",
}


def _codes(concept: str) -> list[dict]:
    response = requests.get(f"{API}/{concept}.def.sdmx.json", timeout=120)
    response.raise_for_status()
    return response.json()["structure"]["codelists"]["codelist"][0]["code"]


def manufacturing_divisions() -> dict[str, str]:
    """Return Nomis codes and labels for SIC 2007 divisions 10--33."""
    result = {}
    for item in _codes("industry"):
        annotations = item.get("annotations", {}).get("annotation", [])
        metadata = {
            entry["annotationtitle"]: entry["annotationtext"]
            for entry in annotations
        }
        label = str(item["description"]["value"])
        if metadata.get("TypeName") != "SIC 2007 division (2 digit)":
            continue
        division = int(label.split(" :", 1)[0])
        if 10 <= division <= 33:
            result[str(item["value"])] = label
    return result


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    industries = manufacturing_divisions()
    params = {
        "geography": ",".join(GEOGRAPHIES),
        "time": ",".join(str(year) for year in range(2015, 2025)),
        "industry": ",".join(industries),
        "employment_status": "1",
        "measure": "1",
        "measures": "20100",
        "select": (
            "date,date_name,geography_name,geography_code,industry_name,"
            "industry_code,employment_status_name,measure_name,obs_value,"
            "obs_status,obs_round"
        ),
    }
    response = requests.get(f"{API}.data.csv", params=params, timeout=180)
    response.raise_for_status()
    table = pd.read_csv(io.BytesIO(response.content))
    table.columns = table.columns.str.lower()
    table = table.sort_values(
        ["date", "geography_name", "industry_code"]
    ).reset_index(drop=True)
    table.to_csv(OUT, index=False)

    metadata = {
        "dataset": "Business Register and Employment Survey: open access",
        "nomis_dataset_id": DATASET,
        "source": "https://www.nomisweb.co.uk/datasets/newbres6pub",
        "api_url": response.url,
        "downloaded_utc": datetime.now(timezone.utc).isoformat(),
        "coverage": "2015-2024, SIC 2007 manufacturing divisions 10-33",
        "geography": list(GEOGRAPHIES.values()),
        "employment_measure": "Employees (workplace jobs)",
        "access": "Open access; rounded for disclosure control",
        "row_count": len(table),
        "sha256": hashlib.sha256(OUT.read_bytes()).hexdigest(),
    }
    META.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Wrote {len(table):,} rows to {OUT}")


if __name__ == "__main__":
    main()
