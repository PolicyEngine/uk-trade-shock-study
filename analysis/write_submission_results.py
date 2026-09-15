"""Validate submission artifacts and emit the primary LaTeX table/macros."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ANCHORS = ("obr", "unit")
MONTHS = (3, 6, 12)
MARGINS = ("displacement", "wage_cut")

#: THE ONE DEFINITION of the thin-support cell that the manuscript excludes
#: from its main table and reports in the appendix instead: the OBR anchor at
#: the three-month duration-equivalent stress, whose displacement draws rest
#: on ~1.2 effective record contributions with the largest single record
#: carrying ~92 per cent of the loss.
#:
#: Both consumers import it from here so they can never drift apart:
#: - this script, for the headline cushioning-contrast range, which must be
#:   computed over MAIN-TABLE cells only (quoting a range whose endpoint comes
#:   from the cell the paper excludes on thin-support grounds is exactly the
#:   inconsistency this constant exists to prevent);
#: - analysis/write_referee_macros.py, for the main/appendix row split.
THIN_CELL_ANCHOR = "obr"
THIN_CELL_MONTHS = 3
#: Row prefix as emitted into \SubmissionScenarioRows below.
THIN_CELL_ROW_PREFIX = f"{THIN_CELL_ANCHOR.upper()} & {THIN_CELL_MONTHS} "
#: Key used in ``diagnostics["contrasts"]``.
THIN_CELL_CONTRAST_KEY = (
    f"{THIN_CELL_ANCHOR}_{THIN_CELL_MONTHS}m_wage_minus_displacement"
)


def is_thin_support_row(row: str) -> bool:
    """True for a generated table row belonging to the excluded thin cell."""
    return row.startswith(THIN_CELL_ROW_PREFIX)


def is_thin_support_contrast(key: str) -> bool:
    """True for the ``contrasts`` key of the excluded thin cell."""
    return key == THIN_CELL_CONTRAST_KEY


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
    # MAIN-TABLE CELLS ONLY. The excluded thin-support cell is quoted
    # separately as \SubmissionThinCellCushionDifference for the appendix; it
    # must not set the endpoint of the headline range, because the manuscript
    # drops that cell from the main table on thin-support grounds.
    contrast_values = [
        item["cushioning_difference_pp"]
        for key, item in diagnostics["contrasts"].items()
        if key.endswith("wage_minus_displacement") and not is_thin_support_contrast(key)
    ]
    if not contrast_values:
        raise ValueError("no main-table cushioning contrasts survived the thin-cell filter")
    thin_contrast = diagnostics["contrasts"].get(THIN_CELL_CONTRAST_KEY)
    if thin_contrast is None:
        raise ValueError(
            f"thin-support contrast {THIN_CELL_CONTRAST_KEY!r} not found; the "
            "main-table exclusion rule and the scenario grid have drifted apart"
        )
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
            # Signed balanced-minus-Bernoulli gaps at BOTH anchors. The
            # manuscript used to hand-type "within 0.6 of a percentage point"
            # for the unit anchor (it is 0.75) and did not report the OBR
            # anchor at all, where the two designs disagree by 1.8 points.
            # Emitting both stops the agreement being quoted more favourably
            # than the artifact supports.
            "AssignmentGapUnit": fmt(
                diagnostics["contrasts"]["unit_balanced_vs_bernoulli"][
                    "balanced_cushioning_percent"
                ]
                - diagnostics["contrasts"]["unit_balanced_vs_bernoulli"][
                    "bernoulli_cushioning_percent"
                ],
                2,
            ),
            "AssignmentGapOBR": fmt(
                diagnostics["contrasts"]["obr_balanced_vs_bernoulli"][
                    "balanced_cushioning_percent"
                ]
                - diagnostics["contrasts"]["obr_balanced_vs_bernoulli"][
                    "bernoulli_cushioning_percent"
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
            "SubmissionUnitDisplacedCushion": fmt(
                diagnostics["scenarios"]["unit_12m_displacement"][
                    "cushioning_mean_percent"
                ],
                1,
            ),
            "SubmissionUnitDisplacedCushionSD": fmt(
                diagnostics["scenarios"]["unit_12m_displacement"][
                    "cushioning_sd_pp"
                ],
                1,
            ),
            "SubmissionCushionDifferenceMin": fmt(min(contrast_values), 1),
            "SubmissionCushionDifferenceMax": fmt(max(contrast_values), 1),
            "SubmissionThinCellCushionDifference": fmt(
                thin_contrast["cushioning_difference_pp"], 1
            ),
        }
    )
    diagnostics["main_table_cells"] = {
        "excluded_thin_cell": THIN_CELL_CONTRAST_KEY,
        "n_main_table_contrasts": len(contrast_values),
        "cushioning_difference_min_pp": min(contrast_values),
        "cushioning_difference_max_pp": max(contrast_values),
        "thin_cell_cushioning_difference_pp": thin_contrast["cushioning_difference_pp"],
    }
    leave_out = load(args.leave_one_sector_out)
    leave_differences = [
        item["wage_cut_cushioning_percent"]
        - item["displacement_cushioning_percent"]
        for item in leave_out["excluded_divisions"].values()
    ]
    macros["LeaveSectorCushionDifferenceMin"] = fmt(min(leave_differences), 1)
    macros["LeaveSectorCushionDifferenceMax"] = fmt(max(leave_differences), 1)
    macros["LeaveSectorCount"] = str(len(leave_differences))
    # This audit runs at its own draw count, which is neither the submission
    # design's nor the legacy production one.  The manuscript cites it as
    # robustness for the headline, so the count must be stated rather than
    # left for a reader to find in the artifact.
    macros["LeaveSectorDraws"] = str(leave_out["n_draws"])
    diagnostics["leave_one_sector_out"] = {
        "division_count": len(leave_differences),
        "cushioning_difference_min_pp": min(leave_differences),
        "cushioning_difference_max_pp": max(leave_differences),
    }
    bootstrap = load(args.bootstrap)
    # RATIO-OF-POOLED-SUMS. Immune to the near-zero per-draw denominators that
    # skew the mean-of-ratios interval, so it is the more defensible primary
    # statistic -- and it is WIDER, not narrower. Two manuscript sections used
    # to state this estimator was absent from the stored artifact; it is
    # present, and quoting its absence while shipping it is exactly the kind
    # of stale prose the generated-macro pipeline exists to prevent.
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
            "BootstrapPooledDifference": fmt(
                bootstrap["estimates"]["cushioning_difference_pooled"]["point"] * 100, 1
            ),
            "BootstrapPooledDifferenceSE": fmt(
                bootstrap["estimates"]["cushioning_difference_pooled"]["bootstrap_se"] * 100, 1
            ),
            "BootstrapPooledCILower": fmt(
                bootstrap["estimates"]["cushioning_difference_pooled"]["ci_2_5"] * 100, 1
            ),
            "BootstrapPooledCIUpper": fmt(
                bootstrap["estimates"]["cushioning_difference_pooled"]["ci_97_5"] * 100, 1
            ),
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
