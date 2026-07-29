"""Download a public HMRC monthly product-destination export panel.

The UK Trade Info OData API is open access and requires no credentials.
Downloads are cached one destination-month at a time so interrupted builds
resume safely and source revisions remain inspectable.

Default destinations are the United States and three non-EU advanced-economy
comparators. EU destinations are deliberately omitted from the default
because the trade collection regime changes around EU exit.

Output (gitignored):
    data/public/hmrc_exports_product_destination_201801_202606.csv.gz
    data/public/hmrc_exports_product_destination_201801_202606.metadata.json

Usage:
    .venv/bin/python analysis/download_hmrc_panel.py
    .venv/bin/python analysis/download_hmrc_panel.py --start 202301 --end 202312
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import requests

API = "https://api.uktradeinfo.com"
EXPORTS_NON_EU = 4
DEFAULT_COUNTRIES = {
    400: ("US", "United States"),
    404: ("CA", "Canada"),
    732: ("JP", "Japan"),
    800: ("AU", "Australia"),
}
ROOT = Path(__file__).resolve().parent.parent
PUBLIC_DATA = ROOT / "data" / "public"
CACHE = PUBLIC_DATA / "hmrc_ots_cache"

FIELDS = (
    "month",
    "country_id",
    "country_code",
    "country_name",
    "commodity_id",
    "value_gbp",
    "net_mass_kg",
)


def iter_months(start: int, end: int):
    """Yield inclusive YYYYMM integers."""
    year, month = divmod(start, 100)
    while year * 100 + month <= end:
        yield year * 100 + month
        month += 1
        if month == 13:
            year, month = year + 1, 1


def query_url(month: int, country_id: int) -> str:
    """Server-aggregated OTS query at month-country-commodity grain."""
    return (
        f"{API}/OTS?$apply=filter(MonthId eq {month} "
        f"and CountryId eq {country_id} and FlowTypeId eq {EXPORTS_NON_EU})/"
        "groupby((MonthId,CountryId,CommodityId),"
        "aggregate(Value with sum as Value,NetMass with sum as NetMass))"
    )


def paged_get(url: str, session: requests.Session) -> list[dict]:
    rows: list[dict] = []
    while url:
        response = None
        for attempt in range(6):
            response = session.get(
                url,
                timeout=180,
                headers={"User-Agent": "PolicyEngine-UK-trade-shock-study/0.1"},
            )
            if response.status_code not in {403, 429, 500, 502, 503, 504}:
                break
            if attempt == 5:
                break
            retry_after = response.headers.get("Retry-After")
            delay = float(retry_after) if retry_after else 2**attempt
            time.sleep(min(delay, 30.0))
        assert response is not None
        response.raise_for_status()
        payload = response.json()
        rows.extend(payload["value"])
        url = payload.get("@odata.nextLink")
    return rows


def cached_month(
    month: int, country_id: int, session: requests.Session
) -> list[dict]:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"ots_exports_{country_id}_{month}.json"
    if path.exists():
        return json.loads(path.read_text())
    rows = paged_get(query_url(month, country_id), session)
    path.write_text(json.dumps(rows, separators=(",", ":")))
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_panel(start: int, end: int, countries: dict[int, tuple[str, str]]) -> Path:
    PUBLIC_DATA.mkdir(parents=True, exist_ok=True)
    stem = f"hmrc_exports_product_destination_{start}_{end}"
    output = PUBLIC_DATA / f"{stem}.csv.gz"
    session = requests.Session()
    row_count = 0
    totals = {code: 0.0 for code, _ in countries.values()}
    observed_months: set[int] = set()
    empty_queries: list[dict[str, int | str]] = []

    with gzip.open(output, "wt", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for country_id, (code, name) in countries.items():
            for month in iter_months(start, end):
                rows = cached_month(month, country_id, session)
                if not rows:
                    empty_queries.append({"country_code": code, "month": month})
                for row in rows:
                    observed_months.add(int(row["MonthId"]))
                    value = float(row.get("Value") or 0.0)
                    writer.writerow(
                        {
                            "month": int(row["MonthId"]),
                            "country_id": int(row["CountryId"]),
                            "country_code": code,
                            "country_name": name,
                            "commodity_id": int(row["CommodityId"]),
                            "value_gbp": value,
                            "net_mass_kg": float(row.get("NetMass") or 0.0),
                        }
                    )
                    row_count += 1
                    totals[code] += value
                print(f"{code} {month}: {len(rows):,} rows", flush=True)

    metadata = {
        "source": "HMRC UK Trade Info Overseas Trade Statistics API",
        "source_url": API,
        "api_authentication": "none",
        "retrieved_at": datetime.now(UTC).isoformat(),
        "start_month": start,
        "end_month": end,
        "observed_start_month": min(observed_months) if observed_months else None,
        "observed_end_month": max(observed_months) if observed_months else None,
        "empty_country_month_queries": empty_queries,
        "flow_type_id": EXPORTS_NON_EU,
        "countries": {
            str(key): {"code": value[0], "name": value[1]}
            for key, value in countries.items()
        },
        "grain": ["month", "country_id", "commodity_id"],
        "measures": ["value_gbp", "net_mass_kg"],
        "row_count": row_count,
        "country_totals_gbp": totals,
        "output": str(output.relative_to(ROOT)),
        "sha256": sha256(output),
        "notes": (
            "CommodityId can represent CN8 or a more aggregated/suppressed "
            "commodity. Merge to the API Commodity endpoint before tariff matching."
        ),
    }
    (PUBLIC_DATA / f"{stem}.metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=201801)
    parser.add_argument("--end", type=int, default=202606)
    parser.add_argument(
        "--country-ids",
        type=int,
        nargs="+",
        default=list(DEFAULT_COUNTRIES),
        choices=list(DEFAULT_COUNTRIES),
    )
    args = parser.parse_args()
    if args.start > args.end:
        raise SystemExit("--start must not be after --end")
    countries = {key: DEFAULT_COUNTRIES[key] for key in args.country_ids}
    print(build_panel(args.start, args.end, countries))


if __name__ == "__main__":
    main()
