"""Validate the earnings of the model's tariff-exposed employees against ASHE.

WHY. The E3 gross earnings loss divided by the expected number of displaced
workers implies mean annual earnings of about GBP 48,000 for the workers the
tariff scenario displaces. This script checks that figure against the ONS
Annual Survey of Hours and Earnings (ASHE), the PAYE-based benchmark for
employee pay by industry.

MODEL SIDE. Employees (employment_income > 0) in the exposed SIC 2007
divisions (sector shock s_j > 0; divisions 10-32), Microcosm UK projected to
the simulation year (PERIOD, tax year 2026-27): weighted mean, weighted
median and the share above the higher-rate threshold, by division and
overall. "Overall" is reported two ways: weighted by employees, and
weighted by s_j x employees -- the EXPECTED DISPLACED workers, which is the
weighting behind the GBP 48,000 (gross loss / displaced workers).

ASHE SIDE. Table 4.7a, "Annual pay - Gross (GBP) - For all employee jobs:
United Kingdom", by two-digit SIC 2007, sheet "All": mean, median and the
published percentiles. Annual estimates are for the tax year ending 5 April
of the reference year and cover employees on adult rates who have been in
the same job for more than a year. The 2025 revised edition (tax year
2024-25, the model's base year) is the benchmark; the 2026 provisional
edition (tax year 2025-26) is a cross-check. Each is uprated to 2026-27 by
the model's own growth in aggregate employment income between the two tax
years (the OBR average-earnings index the model uprates earnings with), so
the comparison is of levels in the base year and does not test the uprating.

AGGREGATION. Division means are combined with the SAME weights on both
sides -- the model's s_j x employees -- so the comparison holds the
industry composition fixed and isolates within-division pay. A second ASHE
aggregate uses ASHE's own (indicative) job counts in place of the model's
head-counts, which tests the composition instead. Medians do not aggregate,
so the like-for-like median is the weighted mean of division medians on
both sides; the model's true pooled median is reported separately. The
"loss-weighted ratio" is the mean of division-level model/ASHE ratios
weighted by each division's share of the gross earnings loss.

Divisions ASHE suppresses (tobacco; the leather median) are dropped from
BOTH sides of the affected aggregate, and their weight is reported.

Raw files: data/raw/ashe_table4_7a_annual_pay_gross_{edition}.xlsx (Open
Government Licence v3.0; extracted unmodified from the ONS zip named in
EDITIONS; SHA256 verified below).

Writes results/e3_ashe.json. Macros: analysis/emit_e3_macros.py.
Usage: PYTHONPATH=. .venv/bin/python analysis/ashe_validation.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from tariff_pipeline.exposure import load_us_export_intensity, sector_earnings_shocks
from tariff_pipeline.runner import _baseline_and_persons, _weighted_median, json_value

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
RAW = ROOT / "data" / "raw"
PERIOD = 2026
ONS_PAGE = (
    "https://www.ons.gov.uk/employmentandlabourmarket/peopleinwork/"
    "earningsandworkinghours/datasets/industry2digitsicashetable4"
)
_ZIP = (
    "https://www.ons.gov.uk/file?uri=/employmentandlabourmarket/peopleinwork/"
    "earningsandworkinghours/datasets/industry2digitsicashetable4/{e}/ashetable4{e}.zip"
)
#: edition -> (local file, sha256, zip, file inside the zip, model period of
#: the tax year the annual estimates refer to)
EDITIONS = {
    "2025 revised": {
        "file": "ashe_table4_7a_annual_pay_gross_2025_revised.xlsx",
        "sha256": "5ec5fe60b9aaae2555996cbb66f8d4c29868105e59384b95d85db36462728baf",
        "zip_url": _ZIP.format(e="2025revised"),
        "zip_sha256": "df87c313d1cb40bf499f1ec001076bcb9c9884c6443d76b186f09e84ff614bc3",
        "zip_member": "sic07 Industry (2) Table 4.7a   Annual pay - Gross 2025.xlsx",
        "tax_year": "2024-25",
        "model_period": 2024,
    },
    "2026 provisional": {
        "file": "ashe_table4_7a_annual_pay_gross_2026_provisional.xlsx",
        "sha256": "d9d2ad3b89e303b33e754811383ed668746bb725d886bb1d6c66aba15a329bd0",
        "zip_url": _ZIP.format(e="2026provisional"),
        "zip_sha256": "afa6848f20c5554ef16fbc7fc099aee05d12714a36b239b1892cb90c2a7c687c",
        "zip_member": "PROV - sic07 Industry (2) Table 4.7a   Annual pay - Gross 2026.xlsx",
        "tax_year": "2025-26",
        "model_period": 2025,
    },
}
BENCHMARK = "2025 revised"
TABLE_NAME = "ASHE Table 4.7a: Annual pay - Gross (GBP) - For all employee jobs, UK, by two-digit SIC 2007"
PERCENTILES = (10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90)
#: Records dropped in the "without the largest records" robustness mean.
TOP_RECORDS_DROPPED = 10


def _number(cell) -> float | None:
    """A published estimate, or None for a suppressed / not-applicable cell."""
    return float(cell) if isinstance(cell, (int, float)) else None


def read_ashe(path: Path, expected_sha: str) -> dict[int, dict]:
    """Division -> {jobs_thousand, median, mean, percentiles{p: value}}."""
    import openpyxl

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected_sha:
        raise RuntimeError(f"{path.name}: SHA256 {digest} is not the recorded {expected_sha}")
    sheet = openpyxl.load_workbook(path, read_only=True, data_only=True)["All"]
    rows = list(sheet.iter_rows(values_only=True))
    title = str(rows[0][0])
    if "Annual pay - Gross" not in title or "all employee jobs" not in title:
        raise RuntimeError(f"{path.name}: unexpected table title {title!r}")
    header = next(r for r in rows if r[0] == "Description")
    if list(header[1:7]) != ["Code", "(thousand)", "Median", "change", "Mean", "change"] or list(
        header[7:17]
    ) != [p for p in PERCENTILES if p != 50]:
        raise RuntimeError(f"{path.name}: unexpected column layout {header[:17]}")
    out = {}
    for row in rows:
        code = str(row[1]).strip().lstrip("'").strip() if row[1] is not None else ""
        if not code.isdigit():
            continue
        percentiles = dict(zip([p for p in PERCENTILES if p != 50], map(_number, row[7:17])))
        percentiles[50] = _number(row[3])
        out[int(code)] = {
            "description": str(row[0]).strip(),
            "jobs_thousand": _number(row[2]),
            "median": _number(row[3]),
            "mean": _number(row[5]),
            "percentiles": percentiles,
        }
    return out


def share_above(percentiles: dict[int, float | None], threshold: float) -> dict | None:
    """Share of jobs paid above ``threshold``, bracketed by the two published
    percentiles either side and linearly interpolated between them. None when
    the threshold lies outside the published (unsuppressed) percentiles."""
    known = sorted((p, v) for p, v in percentiles.items() if v is not None)
    for (p_lo, v_lo), (p_hi, v_hi) in zip(known, known[1:]):
        if v_lo <= threshold <= v_hi:
            at = p_lo + (p_hi - p_lo) * (threshold - v_lo) / (v_hi - v_lo) if v_hi > v_lo else p_lo
            return {
                "interpolated": 1.0 - at / 100.0,
                "lower_bound": 1.0 - p_hi / 100.0,
                "upper_bound": 1.0 - p_lo / 100.0,
            }
    return None


def _wmean(values, weights) -> float:
    return float(np.average(values, weights=weights))


def compare(model: dict[int, dict], ashe: dict[int, dict], factor: float, threshold: float) -> dict:
    """Like-for-like aggregates of the model and one uprated ASHE edition."""
    divisions = sorted(model)
    exp_w = {j: model[j]["shock"] * model[j]["employees"] for j in divisions}
    loss_w = {j: model[j]["shock"] * model[j]["wage_bill"] for j in divisions}
    rows = {}
    for j in divisions:
        a = ashe.get(j, {})
        mean = a.get("mean")
        median = a.get("median")
        above = share_above(a.get("percentiles", {}), threshold / factor)
        rows[j] = {
            "ashe_mean": None if mean is None else factor * mean,
            "ashe_median": None if median is None else factor * median,
            "ashe_jobs_thousand": a.get("jobs_thousand"),
            "ashe_share_above_higher_rate": above,
            "model_over_ashe_mean": None if mean is None else model[j]["mean"] / (factor * mean),
            "model_over_ashe_median": None
            if median is None
            else model[j]["median"] / (factor * median),
        }

    def aggregate(stat_model, stat_ashe, weights):
        """Weighted means over the divisions where both sides have the stat."""
        both = [j for j in divisions if rows[j][stat_ashe] is not None]
        w = np.array([weights[j] for j in both])
        return {
            "model": _wmean([model[j][stat_model] for j in both], w),
            "ashe": _wmean([rows[j][stat_ashe] for j in both], w),
            "divisions": len(both),
            "weight_covered": float(w.sum() / sum(weights.values())),
        }

    mean = aggregate("mean", "ashe_mean", exp_w)
    median = aggregate("median", "ashe_median", exp_w)
    jobs = [j for j in divisions if rows[j]["ashe_mean"] is not None and rows[j]["ashe_jobs_thousand"]]
    jobs_w = np.array([model[j]["shock"] * rows[j]["ashe_jobs_thousand"] for j in jobs])
    hr = [j for j in divisions if rows[j]["ashe_share_above_higher_rate"] is not None]
    hr_w = np.array([exp_w[j] for j in hr])
    ratio = [j for j in divisions if rows[j]["model_over_ashe_mean"] is not None]
    ratio_w = np.array([loss_w[j] for j in ratio])
    ratio_med = [j for j in divisions if rows[j]["model_over_ashe_median"] is not None]
    return {
        "uprating_factor": factor,
        "expected_displaced_weighted_mean": mean,
        "expected_displaced_weighted_mean_of_division_medians": median,
        "model_over_ashe_mean": mean["model"] / mean["ashe"],
        "model_over_ashe_median": median["model"] / median["ashe"],
        "ashe_mean_with_ashe_job_counts": {
            "ashe": _wmean([rows[j]["ashe_mean"] for j in jobs], jobs_w),
            "divisions": len(jobs),
        },
        "loss_weighted_ratio_of_means": _wmean(
            [rows[j]["model_over_ashe_mean"] for j in ratio], ratio_w
        ),
        "loss_weighted_ratio_of_medians": _wmean(
            [rows[j]["model_over_ashe_median"] for j in ratio_med],
            np.array([loss_w[j] for j in ratio_med]),
        ),
        "share_above_higher_rate": {
            "model": _wmean([model[j]["share_above_higher_rate"] for j in hr], hr_w),
            **{
                f"ashe_{k}": _wmean([rows[j]["ashe_share_above_higher_rate"][k] for j in hr], hr_w)
                for k in ("interpolated", "lower_bound", "upper_bound")
            },
            "divisions": len(hr),
            "weight_covered": float(hr_w.sum() / sum(exp_w.values())),
        },
        "by_division": rows,
    }


def main() -> None:
    _, baseline, persons = _baseline_and_persons(None, None, PERIOD)
    weight = persons["weight"].to_numpy(dtype=float)
    earnings = persons["employment_income"].to_numpy(dtype=float)
    division = persons["sic_division"].to_numpy(dtype=float)
    shocks = sector_earnings_shocks("full_tariff")
    shocks = shocks[shocks > 0]
    descriptions = load_us_export_intensity()["description"]
    shock = persons["sic_division"].map(shocks).fillna(0.0).to_numpy(dtype=float)
    exposed = (earnings > 0) & (shock > 0)

    parameters = baseline.tax_benefit_system.parameters(f"{PERIOD}-01-01")
    income_tax = parameters.gov.hmrc.income_tax
    # thresholds of the UK scale are measured above the personal allowance:
    # [0] basic, [1] higher, [2] additional
    threshold = float(income_tax.allowances.personal_allowance.amount) + float(
        income_tax.rates.uk.thresholds[1]
    )
    if not 45_000 < threshold < 60_000:
        raise RuntimeError(f"implausible higher-rate threshold {threshold}")

    def aggregate_earnings(year: int) -> float:
        values = baseline.calculate("employment_income", period=year, map_to="person").values
        return float((np.asarray(values, dtype=float) * weight).sum())

    growth = {
        name: aggregate_earnings(PERIOD) / aggregate_earnings(edition["model_period"])
        for name, edition in EDITIONS.items()
    }

    model = {}
    for j in sorted(int(d) for d in shocks.index):
        mask = exposed & (division == j)
        if not mask.any():
            continue
        e, w = earnings[mask], weight[mask]
        model[j] = {
            "description": str(descriptions.get(j, "")),
            "shock": float(shocks[j]),
            "records": int(mask.sum()),
            "employees": float(w.sum()),
            "wage_bill": float((e * w).sum()),
            "mean": _wmean(e, w),
            "median": _weighted_median(e, w),
            "share_above_higher_rate": float(w[e > threshold].sum() / w.sum()),
        }
    e, w, s = earnings[exposed], weight[exposed], shock[exposed]
    loss = s * w * e
    order = np.argsort(loss)[::-1]
    keep = np.ones(loss.size, dtype=bool)
    keep[order[:TOP_RECORDS_DROPPED]] = False
    overall = {
        "records": int(exposed.sum()),
        "employees": float(w.sum()),
        "expected_displaced": float((s * w).sum()),
        "gross_earnings_loss": float(loss.sum()),
        "employee_weighted": {
            "mean": _wmean(e, w),
            "median": _weighted_median(e, w),
            "share_above_higher_rate": float(w[e > threshold].sum() / w.sum()),
        },
        "expected_displaced_weighted": {
            "mean": _wmean(e, s * w),
            "median": _weighted_median(e, s * w),
            "share_above_higher_rate": float((s * w)[e > threshold].sum() / (s * w).sum()),
            "share_of_gross_loss_above_higher_rate": float(loss[e > threshold].sum() / loss.sum()),
            f"mean_without_{TOP_RECORDS_DROPPED}_largest_loss_records": _wmean(
                e[keep], (s * w)[keep]
            ),
            f"loss_share_of_{TOP_RECORDS_DROPPED}_largest_loss_records": float(
                loss[~keep].sum() / loss.sum()
            ),
            "max_earnings": float(e.max()),
        },
    }

    editions = {}
    for name, edition in EDITIONS.items():
        path = RAW / edition["file"]
        if not path.exists():
            if name == BENCHMARK:
                raise FileNotFoundError(
                    f"{path} missing: download {edition['zip_url']} and extract "
                    f"{edition['zip_member']!r} to that path"
                )
            continue
        ashe = read_ashe(path, edition["sha256"])
        editions[name] = {
            "table": TABLE_NAME,
            "edition": name,
            "tax_year": edition["tax_year"],
            "dataset_page": ONS_PAGE,
            "zip_url": edition["zip_url"],
            "zip_sha256": edition["zip_sha256"],
            "zip_member": edition["zip_member"],
            "local_file": f"data/raw/{edition['file']}",
            "sha256": edition["sha256"],
            "licence": "Open Government Licence v3.0",
            "manufacturing_section_c": {
                "mean_uprated": growth[name] * ashe_c["mean"],
                "median_uprated": growth[name] * ashe_c["median"],
            }
            if (ashe_c := _section_c(path)) is not None
            else None,
            **compare(model, ashe, growth[name], threshold),
        }

    out = {
        "period": PERIOD,
        "tax_year": f"{PERIOD}-{(PERIOD + 1) % 100:02d}",
        "scenario": "full_tariff",
        "benchmark_edition": BENCHMARK,
        "higher_rate_threshold": threshold,
        "higher_rate_threshold_note": (
            "Personal allowance plus the UK basic-rate limit from the model's "
            "parameters; employment income above it. Scottish taxpayers meet "
            "their higher rate at a lower threshold, which is not applied here."
        ),
        "uprating_note": (
            "ASHE annual pay is uprated from its tax year to the simulation "
            "year by the model's growth in aggregate employment income."
        ),
        "coverage_note": (
            "ASHE annual estimates: employee JOBS on adult rates, in the same "
            "job for more than a year, UK. Model: PERSONS with positive "
            "employment income in the survey-based Microcosm UK (FRS) data, "
            "classified by the SIC division of their main job; part-year and "
            "new-in-job workers are included."
        ),
        "model": {"overall": overall, "by_division": model},
        "ashe": editions,
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "e3_ashe.json").write_text(
        json.dumps(json_value(out), indent=2, allow_nan=False) + "\n"
    )

    bench = editions[BENCHMARK]
    print(f"{'div':>3} {'n':>5} {'s%':>5} {'model mean':>10} {'ASHE mean':>10} "
          f"{'model med':>10} {'ASHE med':>10} {'HR% model':>9}")
    for j, m in model.items():
        a = bench["by_division"][j]
        fmt = lambda x: f"{x:10,.0f}" if x is not None else f"{'--':>10}"
        print(f"{j:>3} {m['records']:>5} {100 * m['shock']:5.2f} {fmt(m['mean'])} "
              f"{fmt(a['ashe_mean'])} {fmt(m['median'])} {fmt(a['ashe_median'])} "
              f"{100 * m['share_above_higher_rate']:9.1f}")
    print(json.dumps(json_value(overall), indent=1))
    for name, ed in editions.items():
        print(name, json.dumps({k: v for k, v in ed.items() if k not in ("by_division",) and not isinstance(v, str)}, indent=1))


def _section_c(path: Path) -> dict | None:
    """ASHE section C (all manufacturing, divisions 10-33) mean and median."""
    import openpyxl

    sheet = openpyxl.load_workbook(path, read_only=True, data_only=True)["All"]
    for row in sheet.iter_rows(values_only=True):
        if row[1] is not None and str(row[1]).strip().lstrip("'").strip() == "C":
            return {"mean": _number(row[5]), "median": _number(row[3])}
    return None


if __name__ == "__main__":
    main()
