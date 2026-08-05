"""Validate submission artifacts and emit the primary LaTeX table/macros."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ANCHORS = ("obr", "unit")
MONTHS = (3, 6, 12)
MARGINS = ("displacement", "wage_cut")


def load(path: Path) -> dict:
    with path.open() as stream:
        return json.load(stream)


def mean(values) -> float:
    return float(np.mean(np.asarray(list(values), dtype=float)))


def sd(values) -> float:
    values = np.asarray(list(values), dtype=float)
    return float(values.std(ddof=1)) if values.size > 1 else 0.0


def fmt(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results/submission"))
    parser.add_argument("--output", type=Path, default=Path("paper/generated_submission.tex"))
    parser.add_argument("--diagnostics", type=Path, default=Path("results/submission_summary.json"))
    parser.add_argument(
        "--leave-one-sector-out",
        type=Path,
        default=Path("results/leave_one_sector_out.json"),
    )
    parser.add_argument(
        "--bootstrap",
        type=Path,
        default=Path("results/bootstrap_uncertainty.json"),
    )
    parser.add_argument("--expected-draws", type=int, default=100)
    args = parser.parse_args()

    data = {}
    for anchor in ANCHORS:
        for months in MONTHS:
            for margin in MARGINS:
                name = f"submission_{anchor}_{months}m_{margin}"
                data[name] = load(args.results_dir / f"{name}.json")
        comparator = f"submission_{anchor}_12m_displacement_bernoulli"
        data[comparator] = load(args.results_dir / f"{comparator}.json")
    bad_draws = {
        name: item["n_draws"]
        for name, item in data.items()
        if item["n_draws"] != args.expected_draws
    }
    if bad_draws:
        raise ValueError(f"submission artifacts have incorrect draw counts: {bad_draws}")

    rows = []
    diagnostics = {"n_draws": args.expected_draws, "scenarios": {}, "contrasts": {}}
    macros = {"SubmissionDraws": str(args.expected_draws)}
    for anchor in ANCHORS:
        for months in MONTHS:
            items = {
                margin: data[f"submission_{anchor}_{months}m_{margin}"]
                for margin in MARGINS
            }
            for margin, item in items.items():
                draws = item["draws"]
                gross = mean(d["gross_earnings_loss"] for d in draws) / 1e6
                gross_sd = sd(d["gross_earnings_loss"] for d in draws) / 1e6
                effective = mean(d["effective_loss_records"] for d in draws)
                max_share = mean(d["max_record_loss_share"] for d in draws) * 100
                affected = mean(d["affected_records"] for d in draws)
                rows.append(
                    f"{anchor.upper()} & {months} & "
                    f"{'Displacement' if margin == 'displacement' else 'Wage cut'} & "
                    f"{gross:.0f}"
                    + (f" $\\pm$ {gross_sd:.0f}" if gross_sd >= 0.5 else "")
                    + f" & {item['exchequer_cost_mean'] / 1e6:.0f}"
                    + (
                        f" $\\pm$ {item['exchequer_cost_sd'] / 1e6:.0f}"
                        if item["exchequer_cost_sd"] / 1e6 >= 0.5
                        else ""
                    )
                    + f" & {item['cushioning_rate_mean'] * 100:.1f}"
                    + (
                        f" $\\pm$ {item['cushioning_rate_sd'] * 100:.1f}"
                        if item["cushioning_rate_sd"] * 100 >= 0.05
                        else ""
                    )
                    + f" & {item['poverty_rate_change_bhc_mean'] * 100:.3f} \\\\"
                )
                diagnostics["scenarios"][f"{anchor}_{months}m_{margin}"] = {
                    "gross_loss_mean_million": gross,
                    "gross_loss_sd_million": gross_sd,
                    "exchequer_cost_mean_million": item["exchequer_cost_mean"] / 1e6,
                    "exchequer_cost_sd_million": item["exchequer_cost_sd"] / 1e6,
                    "cushioning_mean_percent": item["cushioning_rate_mean"] * 100,
                    "cushioning_sd_pp": item["cushioning_rate_sd"] * 100,
                    "poverty_change_mean_pp": item["poverty_rate_change_bhc_mean"] * 100,
                    "affected_records_mean": affected,
                    "effective_loss_records_mean": effective,
                    "max_record_loss_share_mean_percent": max_share,
                }

            displacement = items["displacement"]["draws"]
            wage_cut = items["wage_cut"]["draws"]
            cushion_diff = [
                w["cushioning_rate"] - d["cushioning_rate"]
                for d, w in zip(displacement, wage_cut, strict=True)
                if d["cushioning_rate"] is not None and w["cushioning_rate"] is not None
            ]
            exchequer_diff = [
                w["exchequer_cost"] - d["exchequer_cost"]
                for d, w in zip(displacement, wage_cut, strict=True)
            ]
            diagnostics["contrasts"][f"{anchor}_{months}m_wage_minus_displacement"] = {
                "cushioning_difference_pp": mean(cushion_diff) * 100,
                "cushioning_difference_assignment_sd_pp": sd(cushion_diff) * 100,
                "exchequer_difference_million": mean(exchequer_diff) / 1e6,
                "exchequer_difference_assignment_sd_million": sd(exchequer_diff) / 1e6,
            }

        balanced = data[f"submission_{anchor}_12m_displacement"]
        bernoulli = data[f"submission_{anchor}_12m_displacement_bernoulli"]
        diagnostics["contrasts"][f"{anchor}_balanced_vs_bernoulli"] = {
            "gross_loss_sd_ratio": (
                sd(d["gross_earnings_loss"] for d in balanced["draws"])
                / sd(d["gross_earnings_loss"] for d in bernoulli["draws"])
            ),
            "exchequer_sd_ratio": balanced["exchequer_cost_sd"]
            / bernoulli["exchequer_cost_sd"],
            "balanced_exchequer_mean_million": balanced["exchequer_cost_mean"] / 1e6,
            "bernoulli_exchequer_mean_million": bernoulli["exchequer_cost_mean"] / 1e6,
            "balanced_cushioning_percent": balanced["cushioning_rate_mean"] * 100,
            "bernoulli_cushioning_percent": bernoulli["cushioning_rate_mean"] * 100,
            "balanced_gross_loss_mean_million": mean(
                d["gross_earnings_loss"] for d in balanced["draws"]
            )
            / 1e6,
            "bernoulli_gross_loss_mean_million": mean(
                d["gross_earnings_loss"] for d in bernoulli["draws"]
            )
            / 1e6,
        }

    unit_12 = diagnostics["contrasts"]["unit_12m_wage_minus_displacement"]
    contrast_values = [
        item["cushioning_difference_pp"]
        for key, item in diagnostics["contrasts"].items()
        if key.endswith("wage_minus_displacement")
    ]
    macros.update(
        {
            "PrimaryCushionDifference": fmt(
                unit_12["cushioning_difference_pp"], 1
            ),
            "PrimaryCushionDifferenceSD": fmt(
                unit_12["cushioning_difference_assignment_sd_pp"], 1
            ),
            "PrimaryExchequerDifference": fmt(
                unit_12["exchequer_difference_million"], 0
            ),
            "BalancedGrossSDRatio": fmt(
                diagnostics["contrasts"]["unit_balanced_vs_bernoulli"][
                    "gross_loss_sd_ratio"
                ],
                2,
            ),
            "BalancedExchequerSDRatio": fmt(
                diagnostics["contrasts"]["unit_balanced_vs_bernoulli"][
                    "exchequer_sd_ratio"
                ],
                2,
            ),
            "BalancedUnitCushion": fmt(
                diagnostics["contrasts"]["unit_balanced_vs_bernoulli"][
                    "balanced_cushioning_percent"
                ],
                1,
            ),
            "BernoulliUnitCushion": fmt(
                diagnostics["contrasts"]["unit_balanced_vs_bernoulli"][
                    "bernoulli_cushioning_percent"
                ],
                1,
            ),
            "BalancedUnitExchequer": fmt(
                diagnostics["contrasts"]["unit_balanced_vs_bernoulli"][
                    "balanced_exchequer_mean_million"
                ],
                0,
            ),
            "BernoulliUnitExchequer": fmt(
                diagnostics["contrasts"]["unit_balanced_vs_bernoulli"][
                    "bernoulli_exchequer_mean_million"
                ],
                0,
            ),
            "UnitEffectiveLossRecords": fmt(
                diagnostics["scenarios"]["unit_12m_displacement"][
                    "effective_loss_records_mean"
                ],
                1,
            ),
            "UnitMaxRecordLossShare": fmt(
                diagnostics["scenarios"]["unit_12m_displacement"][
                    "max_record_loss_share_mean_percent"
                ],
                1,
            ),
            "OBRThreeMonthEffectiveLossRecords": fmt(
                diagnostics["scenarios"]["obr_3m_displacement"][
                    "effective_loss_records_mean"
                ],
                1,
            ),
            "OBRThreeMonthMaxRecordLossShare": fmt(
                diagnostics["scenarios"]["obr_3m_displacement"][
                    "max_record_loss_share_mean_percent"
                ],
                1,
            ),
            "SubmissionOBRTwelveLoss": fmt(
                diagnostics["scenarios"]["obr_12m_wage_cut"][
                    "gross_loss_mean_million"
                ],
                0,
            ),
            "SubmissionUnitTwelveLoss": fmt(
                diagnostics["scenarios"]["unit_12m_wage_cut"][
                    "gross_loss_mean_million"
                ],
                0,
            ),
            "SubmissionUnitWageCushion": fmt(
                diagnostics["scenarios"]["unit_12m_wage_cut"][
                    "cushioning_mean_percent"
                ],
                1,
            ),
            "SubmissionUnitWageExchequer": fmt(
                diagnostics["scenarios"]["unit_12m_wage_cut"][
                    "exchequer_cost_mean_million"
                ],
                0,
            ),
            "SubmissionCushionDifferenceMin": fmt(min(contrast_values), 1),
            "SubmissionCushionDifferenceMax": fmt(max(contrast_values), 1),
        }
    )
    leave_out = load(args.leave_one_sector_out)
    leave_differences = [
        item["wage_cut_cushioning_percent"]
        - item["displacement_cushioning_percent"]
        for item in leave_out["excluded_divisions"].values()
    ]
    macros["LeaveSectorCushionDifferenceMin"] = fmt(min(leave_differences), 1)
    macros["LeaveSectorCushionDifferenceMax"] = fmt(max(leave_differences), 1)
    macros["LeaveSectorCount"] = str(len(leave_differences))
    diagnostics["leave_one_sector_out"] = {
        "division_count": len(leave_differences),
        "cushioning_difference_min_pp": min(leave_differences),
        "cushioning_difference_max_pp": max(leave_differences),
    }
    bootstrap = load(args.bootstrap)
    estimates = bootstrap["estimates"]
    design = bootstrap["design"]
    diff = estimates["cushioning_difference"]
    macros.update(
        {
            "BootstrapDraws": str(design["n_draws"]),
            "BootstrapReplicates": str(design["n_boot"]),
            "BootstrapHouseholds": f"{design['n_households']:,}",
            "BootstrapCushionDifference": fmt(diff["point"] * 100, 1),
            "BootstrapCushionDifferenceSE": fmt(diff["bootstrap_se"] * 100, 1),
            "BootstrapCushionDifferenceCILower": fmt(diff["ci_2_5"] * 100, 1),
            "BootstrapCushionDifferenceCIUpper": fmt(diff["ci_97_5"] * 100, 1),
            "BootstrapDisplacementCushionSE": fmt(
                estimates["displacement_cushioning"]["bootstrap_se"] * 100, 1
            ),
            "BootstrapWageCushionSE": fmt(
                estimates["wage_cut_cushioning"]["bootstrap_se"] * 100, 1
            ),
            "BootstrapDisplacementExchequerSE": fmt(
                estimates["displacement_exchequer"]["bootstrap_se"] / 1e6, 0
            ),
            "BootstrapWageExchequerSE": fmt(
                estimates["wage_cut_exchequer"]["bootstrap_se"] / 1e6, 0
            ),
        }
    )
    diagnostics["bootstrap"] = {
        "cushioning_difference_pp": diff["point"] * 100,
        "cushioning_difference_se_pp": diff["bootstrap_se"] * 100,
        "cushioning_difference_ci_pp": [diff["ci_2_5"] * 100, diff["ci_97_5"] * 100],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "% Generated by analysis/write_submission_results.py; do not edit.",
        *(f"\\newcommand{{\\{key}}}{{{value}}}" for key, value in macros.items()),
        "\\newcommand{\\SubmissionScenarioRows}{%",
        *rows,
        "}",
    ]
    args.output.write_text("\n".join(lines) + "\n")
    args.diagnostics.write_text(json.dumps(diagnostics, indent=2) + "\n")
    print(f"Wrote {args.output} and {args.diagnostics}")


if __name__ == "__main__":
    main()
