"""Emit LaTeX macros for the factorial decomposition and leave-one-record-out
sensitivity (referee major points 1 and 4).

Reads results/factorial_decomposition.json and results/leave_one_record_out.json,
writes paper/generated_factorial.tex.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def fmt(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--factorial", type=Path, default=Path("results/factorial_decomposition.json")
    )
    parser.add_argument(
        "--loo", type=Path, default=Path("results/leave_one_record_out.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("paper/generated_factorial.tex")
    )
    parser.add_argument("--expected-draws", type=int, default=50)
    args = parser.parse_args()

    factorial = json.loads(args.factorial.read_text())
    if factorial["design"]["n_draws"] != args.expected_draws:
        raise ValueError(
            f"factorial artifact has {factorial['design']['n_draws']} draws, "
            f"expected {args.expected_draws}"
        )
    unit = factorial["anchors"]["unit"]
    macros = {
        "FactorialDraws": str(factorial["design"]["n_draws"]),
        "FactorialValidDraws": str(unit["valid_draws"]),
        "FactorialWageCushion": fmt(unit["cushioning_percent"]["wage_cut"]),
        "FactorialConcCushion": fmt(
            unit["cushioning_percent"]["concentrated_wage_cut"]
        ),
        "FactorialDispCushion": fmt(unit["cushioning_percent"]["displacement"]),
        "FactorialGapTotal": fmt(unit["total_gap"]["mean_pp"]),
        "FactorialGapTotalSD": fmt(unit["total_gap"]["assignment_sd_pp"]),
        "FactorialConcStep": fmt(
            unit["concentration_and_selection_step"]["mean_pp"]
        ),
        "FactorialConcStepSD": fmt(
            unit["concentration_and_selection_step"]["assignment_sd_pp"]
        ),
        "FactorialStateStep": fmt(unit["employment_state_step"]["mean_pp"]),
        "FactorialStateStepSD": fmt(
            unit["employment_state_step"]["assignment_sd_pp"]
        ),
        "FactorialStateShare": fmt(
            unit["employment_state_share_of_gap"] * 100, 0
        ),
    }
    if "obr" in factorial["anchors"]:
        obr = factorial["anchors"]["obr"]
        macros.update(
            {
                "FactorialOBRConcCushion": fmt(
                    obr["cushioning_percent"]["concentrated_wage_cut"]
                ),
                "FactorialOBRConcStep": fmt(
                    obr["concentration_and_selection_step"]["mean_pp"]
                ),
                "FactorialOBRStateStep": fmt(
                    obr["employment_state_step"]["mean_pp"]
                ),
            }
        )
    channels = unit.get("channel_split")
    if channels:
        state = channels["employment_state_step_by_channel_pp"]
        conc = channels["concentration_and_selection_step_by_channel_pp"]
        for key, label in (
            ("income_tax", "IncomeTax"),
            ("employee_national_insurance", "NI"),
            ("universal_credit", "UC"),
            ("other_benefits", "OtherBenefits"),
            ("other_residual", "Residual"),
        ):
            macros[f"FactorialStateStep{label}"] = fmt(state[key])
            macros[f"FactorialConcStep{label}"] = fmt(conc[key])
        # Channel LEVELS, not just steps.  The manuscript reconciles the
        # displacement margin against the Dolls, Fuest and Peichl (2012) UK
        # unemployment-shock decomposition, which reports the tax, social
        # insurance and out-of-work benefit contributions separately; that
        # comparison needs levels and must never be hand-typed.
        levels = channels["components_share_of_gross_loss"]
        macros["FactorialChannelSeeds"] = str(len(channels["seeds"]))
        for margin, prefix in (
            ("displacement", "DispChannel"),
            ("wage_cut", "WageChannel"),
        ):
            share = levels[margin]
            for key, label in (
                ("income_tax", "IncomeTax"),
                ("employee_national_insurance", "NI"),
                ("universal_credit", "UC"),
                ("other_benefits", "OtherBenefits"),
                ("other_residual", "Residual"),
            ):
                macros[f"{prefix}{label}"] = fmt(100 * share[key])
            # Benefits as a single block, the object Dolls et al. report.
            macros[f"{prefix}Benefits"] = fmt(
                100 * (share["universal_credit"] + share["other_benefits"])
            )
            macros[f"{prefix}TaxAndNI"] = fmt(
                100 * (share["income_tax"] + share["employee_national_insurance"])
            )
            # The channel split runs on a handful of common seeds while the
            # cushioning rates are estimated over the full paired design, so
            # the components do NOT sum to the headline on the stochastic
            # margins.  Emit the sum and the shortfall rather than letting the
            # manuscript compare a decomposition that does not add up against
            # Dolls, Fuest and Peichl's, which does.
            channel_sum = 100 * sum(share.values())
            macros[f"{prefix}Sum"] = fmt(channel_sum)
            macros[f"{prefix}Shortfall"] = fmt(
                unit["cushioning_percent"][margin] - channel_sum
            )
        # The factorial STEPS on the channel seeds, for the same reason: the
        # manuscript must not present the 50-draw step and the 5-seed channel
        # arithmetic as though they were the same decomposition.
        sums = {
            key: 100 * sum(levels[key].values())
            for key in ("wage_cut", "concentrated_wage_cut", "displacement")
        }
        macros["FactorialConcStepChannelSeeds"] = fmt(
            sums["wage_cut"] - sums["concentrated_wage_cut"]
        )
        macros["FactorialStateStepChannelSeeds"] = fmt(
            sums["concentrated_wage_cut"] - sums["displacement"]
        )

    loo = json.loads(args.loo.read_text())
    macros.update(
        {
            "LOOHouseholds": str(loo["design"]["n_loss_contributing_households"]),
            "LOODraws": str(loo["design"]["n_draws"]),
            "LOOPoint": fmt(loo["point_contrast_pp"]),
            "LOOMin": fmt(loo["loo_min_pp"]),
            "LOOMax": fmt(loo["loo_max_pp"]),
            "LOOMaxShift": fmt(
                abs(loo["most_influential_household"]["shift_pp"]), 1
            ),
        }
    )
    if not loo["all_positive"]:
        raise ValueError(
            "leave-one-record-out contrast changes sign; manuscript text "
            "asserting sign robustness must be revised before building."
        )

    lines = [
        "% Generated by analysis/write_factorial_results.py; do not edit.",
        *(f"\\newcommand{{\\{key}}}{{{value}}}" for key, value in macros.items()),
    ]
    args.output.write_text("\n".join(lines) + "\n")
    print(f"Wrote {args.output} ({len(macros)} macros)")


if __name__ == "__main__":
    main()
