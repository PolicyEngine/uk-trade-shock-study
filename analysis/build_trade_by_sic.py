"""Build uk_trade_shock_study/data/us_export_intensity_by_sic.csv from real data.

PROVENANCE (all public, fetched programmatically; raw pulls cached in data/):

NUMERATOR — UK goods exports to the US, 2024 calendar year, £:
  HMRC uktradeinfo OData API (https://api.uktradeinfo.com), the machine-readable
  publication of HMRC Overseas Trade Statistics / Regional Trade Statistics:
  - RTS endpoint: MonthId 202401-202412, CountryId 400 (United States),
    FlowTypeId 4 (non-EU exports), summed over all Government Office Regions
    and months, keyed by 2-digit SITC (rev 4) chapter. Values are £.
  - OTS endpoint: same filter, pulled only for SITC chapters that straddle
    SIC divisions (33, 66, 68, 71, 77, 78, 79, 87, 89), aggregated to 3-digit
    SITC to split them (e.g. 714 aero engines -> SIC 30, not 28).
  Cross-checked totals (2024): all-commodity US exports £55.6bn on the
  RTS/OTS basis vs ONS BoP-basis ~£59.3bn (BoP adds coverage/valuation
  adjustments); SITC 78 road vehicles £9.0bn (SMMT ~£8-9bn); SITC 54
  medicinal & pharmaceutical £6.0bn (ABPI £6.6bn, BoP basis). All pass.

CROSSWALK — SITC (rev 4) -> SIC 2007 division, coded in SITC2_TO_SIC /
  SITC3_TO_SIC below. Standard concordance for manufacturing; judgement calls:
  - 681 (silver/platinum) and 896 (works of art) EXCLUDED: overwhelmingly
    London bullion/art-market re-exports, not UK production employment.
  - 714 aero engines, 792 aircraft, 793 ships, 791 rail, 785 motorcycles
    -> SIC 30 (other transport, incl aerospace).
  - 891 arms & ammunition -> SIC 25 (fabricated metal, 25.4).
  - 872 medical instruments -> SIC 32; 871/873/874 -> SIC 26.
  - 776 semiconductors -> SIC 26; rest of 77 -> SIC 27.
  - 333 crude oil -> SIC 06 (dropped: extraction, negligible FRS employment);
    334/335 refined -> SIC 19.
  Excluded chapters (91, 93, 96, 97, non-goods SITC 0 divisions of ch. 9
  special transactions) total < £0.3bn.

DENOMINATOR — division total turnover, ONS Annual Business Survey
  ("Non-financial business economy, UK: Sections A to S", abssectionsas.xlsx,
  Section C sheet), latest year available (2023), £m. Turnover proxies gross
  output at basic prices; intensity = US exports / turnover.

Usage: .venv/bin/python analysis/build_trade_by_sic.py
Writes the packaged CSV in place. Requires network on first run; raw API
pulls are cached as JSON in data/.
"""

from __future__ import annotations

import json
from pathlib import Path

import openpyxl
import requests

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "uk_trade_shock_study" / "data" / "us_export_intensity_by_sic.csv"
ABS_XLSX = DATA / "abssectionsas.xlsx"
ABS_URL = (
    "https://www.ons.gov.uk/file?uri=/businessindustryandtrade/business/"
    "businessservices/datasets/uknonfinancialbusinesseconomyannualbusiness"
    "surveysectionsas/current/abssectionsas.xlsx"
)

API = "https://api.uktradeinfo.com"
US, EXPORTS_NON_EU = 400, 4
YEAR = 2024

# SITC2 chapter -> SIC 2007 division (None = excluded / non-production).
SITC2_TO_SIC = {
    0: 10, 1: 10, 2: 10, 3: 10, 4: 10, 5: 10, 6: 10, 7: 10, 8: 10, 9: 10,
    11: 11, 12: 12,
    21: None, 22: 1, 23: 20, 24: 16, 25: 17, 26: 13, 27: 8, 28: 7, 29: 1,
    32: 5, 34: 6,
    41: 10, 42: 10, 43: 10,
    51: 20, 52: 20, 53: 20, 55: 20, 56: 20, 57: 20, 58: 20, 59: 20,
    54: 21,
    61: 15, 62: 22, 63: 16, 64: 17, 65: 13, 67: 24, 69: 25,
    72: 28, 73: 28, 74: 28,
    75: 26, 76: 26,
    81: 23, 82: 31, 83: 15, 84: 14, 85: 15, 88: 26,
    91: None, 93: None, 96: None, 97: None,
}
# Chapters split at SITC3 (fetched from OTS): SITC3 -> SIC division.
SPLIT_CHAPTERS = (33, 66, 68, 71, 77, 78, 79, 87, 89)
SITC3_TO_SIC = {
    333: None, 334: 19, 335: 19, 342: 6, 344: 6,  # crude dropped (SIC 06)
    661: 23, 662: 23, 663: 23, 664: 23, 665: 23, 666: 23, 667: 32,
    681: None, 682: 24, 683: 24, 684: 24, 685: 24, 686: 24, 687: 24, 689: 24,
    711: 28, 712: 28, 713: 28, 714: 30, 716: 28, 718: 28,
    771: 27, 772: 27, 773: 27, 774: 27, 775: 27, 776: 26, 778: 27,
    781: 29, 782: 29, 783: 29, 784: 29, 785: 30, 786: 29,
    791: 30, 792: 30, 793: 30,
    871: 26, 872: 32, 873: 26, 874: 26,
    891: 25, 892: 18, 893: 22, 894: 32, 895: 32, 896: None, 897: 32,
    898: 32, 899: 32,
}

DIVISION_NAMES = {
    10: "Food products", 11: "Beverages", 12: "Tobacco",
    13: "Textiles", 14: "Wearing apparel", 15: "Leather",
    16: "Wood products", 17: "Paper", 18: "Printing",
    19: "Coke and refined petroleum", 20: "Chemicals",
    21: "Pharmaceuticals", 22: "Rubber and plastic products",
    23: "Other non-metallic mineral products",
    24: "Basic metals (incl. steel)", 25: "Fabricated metal products",
    26: "Computer electronic and optical products",
    27: "Electrical equipment", 28: "Machinery and equipment n.e.c.",
    29: "Motor vehicles trailers and semi-trailers",
    30: "Other transport equipment (incl. aerospace)",
    31: "Furniture", 32: "Other manufacturing",
}


def _paged(url: str) -> list[dict]:
    rows = []
    while url:
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        payload = r.json()
        rows += payload["value"]
        url = payload.get("@odata.nextLink")
    return rows


def _cached(name: str, url: str) -> list[dict]:
    path = DATA / name
    if path.exists():
        return json.loads(path.read_text())
    rows = _paged(url)
    path.write_text(json.dumps(rows))
    return rows


def fetch_us_exports_by_division() -> dict[int, float]:
    """US goods exports 2024 (£) keyed by SIC 2007 division."""
    flt = (
        f"MonthId ge {YEAR}01 and MonthId le {YEAR}12 "
        f"and CountryId eq {US} and FlowTypeId eq {EXPORTS_NON_EU}"
    )
    rts = _cached("rts_us_2024.json", f"{API}/RTS?$filter={flt}")
    sitc2 = {}
    for row in rts:
        sitc2[row["CommoditySitc2Id"]] = sitc2.get(row["CommoditySitc2Id"], 0.0) + row["Value"]

    by_div: dict[int, float] = {}

    def add(div, value):
        if div is not None:
            by_div[div] = by_div.get(div, 0.0) + value

    for chapter, value in sitc2.items():
        if chapter in SPLIT_CHAPTERS:
            continue
        add(SITC2_TO_SIC.get(chapter), value)

    for chapter in SPLIT_CHAPTERS:
        ots = _cached(
            f"ots_us_2024_sitc{chapter}.json",
            f"{API}/OTS?$filter={flt} and CommoditySitcId ge {chapter * 1000} "
            f"and CommoditySitcId lt {(chapter + 1) * 1000}",
        )
        total3 = 0.0
        for row in ots:
            add(SITC3_TO_SIC.get(row["CommoditySitcId"] // 100), row["Value"])
            total3 += row["Value"]
        # OTS split must reproduce the RTS chapter total (within rounding)
        if abs(total3 - sitc2.get(chapter, 0.0)) > 0.01 * max(total3, 1.0):
            raise AssertionError(
                f"SITC {chapter}: OTS split {total3:.0f} != RTS {sitc2.get(chapter):.0f}"
            )
    return by_div


def abs_turnover_by_division() -> tuple[dict[int, float], int]:
    """ABS total turnover (£) by manufacturing division, latest year."""
    if not ABS_XLSX.exists():
        ABS_XLSX.write_bytes(requests.get(ABS_URL, timeout=120).content)
    ws = openpyxl.load_workbook(ABS_XLSX, read_only=True)["Section C"]
    records = {}
    for row in ws.iter_rows(values_only=True):
        sic, year, turnover = str(row[0] or ""), row[2], row[4]
        if sic.isdigit() and len(sic) == 2 and str(year).isdigit():
            try:
                records[(int(sic), int(year))] = float(turnover) * 1e6
            except (TypeError, ValueError):
                pass
    latest = max(y for _, y in records)
    # per-division latest non-suppressed year (e.g. beverages are "[c]"
    # confidentiality-suppressed in 2023 -> fall back to 2022)
    out = {}
    for (d, y), v in records.items():
        if d not in out or y > out[d][0]:
            out[d] = (y, v)
    return {d: v for d, (y, v) in out.items()}, latest


def main() -> None:
    DATA.mkdir(exist_ok=True)
    exports = fetch_us_exports_by_division()
    turnover, abs_year = abs_turnover_by_division()

    total = sum(exports.values())
    print(f"US goods exports 2024 mapped to SIC divisions: £{total / 1e9:.1f}bn")
    print(f"  (sanity: autos div 29 £{exports.get(29, 0) / 1e9:.2f}bn, "
          f"pharma div 21 £{exports.get(21, 0) / 1e9:.2f}bn, "
          f"aerospace div 30 £{exports.get(30, 0) / 1e9:.2f}bn)")

    lines = [
        "# US export intensity by SIC 2007 division — REAL DATA build.",
        "# Numerator: HMRC uktradeinfo RTS/OTS API, UK goods exports to the",
        f"# United States, calendar {YEAR}, £, SITC->SIC crosswalk documented in",
        "# analysis/build_trade_by_sic.py (which wrote this file; rerun to rebuild).",
        f"# Denominator: ONS Annual Business Survey division total turnover, {abs_year}",
        "# (per-division latest non-suppressed year; beverages fall back to 2022).",
        "# us_export_share = US exports of the division / division turnover.",
        "sic_division,description,us_export_share",
    ]
    for div in sorted(DIVISION_NAMES):
        exp = exports.get(div, 0.0)
        turn = turnover.get(div)
        if not turn or exp <= 0:
            continue
        share = exp / turn
        lines.append(f"{div},{DIVISION_NAMES[div]},{share:.4f}")
        print(f"  {div:>2} {DIVISION_NAMES[div][:42]:<42} "
              f"exports £{exp / 1e9:6.2f}bn / turnover £{turn / 1e9:6.1f}bn "
              f"= {share:.3f}")
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
