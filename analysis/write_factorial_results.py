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
            # Magnitudes, for prose that carries the sign in words. Universal
            # Credit's contribution to the concentration step is negative --
            # it offsets the gap -- and "worth -7.3 points" reads as a typo.
            macros[f"FactorialConcStep{label}Abs"] = fmt(abs(conc[key]))
            macros[f"FactorialStateStep{label}Abs"] = fmt(abs(state[key]))
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
            ("concentrated_wage_cut", "ConcChannel"),
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
        sums = {
            key: 100 * sum(levels[key].values())
            for key in ("wage_cut", "concentrated_wage_cut", "displacement")
        }
        # The WAGE-BILL-WEIGHTED schedule wedge. The representative-worker
        # benchmark elsewhere in the paper is computed at the exposed-sector
        # mean, but the object actually being compared is a wage-bill-weighted
        # aggregate, and the exposed population is far more concentrated above
        # the higher-rate and upper-earnings-limit thresholds than one worker
        # on the mean. Taking income tax and employee NI together gives the
        # schedule's own prediction for the concentration step; the residual
        # against the realised step is what the household system contributes,
        # and both must be read on the SAME seed basis.
        wedge = (
            100 * (levels["wage_cut"]["income_tax"]
                   + levels["wage_cut"]["employee_national_insurance"])
            - 100 * (levels["concentrated_wage_cut"]["income_tax"]
                     + levels["concentrated_wage_cut"]["employee_national_insurance"])
        )
        conc_step_seeds = sums["wage_cut"] - sums["concentrated_wage_cut"]
        # The wedge is a TAX-AND-NI object, not a participation tax rate: it
        # nets no out-of-work entitlement, and the means-tested awards that a
        # participation rate would include run the other way (see
        # ScheduleWedgeOffsets). The manuscript must not name it as a PTR.
        #
        # It is defined against the CONCENTRATED cell, so quote
        # \ConcChannelTaxAndNI beside it. Displacement happens to carry the
        # same tax-and-NI level -- the worker-level earnings losses are the
        # same draw and only benefits react to employment status -- and
        # earlier drafts silently displayed the displacement macro instead.
        # That coincidence is asserted rather than trusted: if a future run
        # breaks it, the swap becomes an error and this stops the build.
        conc_tax_ni = 100 * (
            levels["concentrated_wage_cut"]["income_tax"]
            + levels["concentrated_wage_cut"]["employee_national_insurance"]
        )
        disp_tax_ni = 100 * (
            levels["displacement"]["income_tax"]
            + levels["displacement"]["employee_national_insurance"]
        )
        if abs(conc_tax_ni - disp_tax_ni) > 0.05:
            raise ValueError(
                "concentrated and displacement tax-and-NI relief have "
                f"diverged ({conc_tax_ni:.2f} vs {disp_tax_ni:.2f} per cent); "
                "the manuscript's schedule-wedge passage assumes they "
                "coincide and must be rewritten before building."
            )
        # RECONCILIATION AGAINST DOLLS, FUEST AND PEICHL (2012), UK
        # unemployment cell (NBER WP 16275, App. Tables 2-3). Their two
        # components are external literature constants; everything DERIVED
        # from them is emitted here so the manuscript never hand-types a
        # difference or a ratio that a re-run can silently invalidate. An
        # earlier draft hardcoded "a factor of roughly six" and an
        # "11.2-point difference", both of which were artifacts of a 5-seed
        # channel split and a UC award served from a stale cache.
        dolls_tax, dolls_benefits = 25.2, 16.3
        dolls_total = dolls_tax + dolls_benefits
        disp_sum = sums["displacement"]
        disp_benefits = 100 * (
            levels["displacement"]["universal_credit"]
            + levels["displacement"]["other_benefits"]
        )
        disp_residual = 100 * levels["displacement"]["other_residual"]
        dolls_gap = dolls_total - disp_sum
        macros["DollsUKTax"] = fmt(dolls_tax)
        macros["DollsUKBenefits"] = fmt(dolls_benefits)
        macros["DollsUKTotal"] = fmt(dolls_total)
        macros["DollsGapPoints"] = fmt(dolls_gap)
        macros["DollsBenefitRatio"] = fmt(
            dolls_benefits / disp_benefits if disp_benefits else float("nan")
        )
        # The pension/salary-sacrifice residual is a component of THIS paper's
        # sum with no counterpart in Dolls et al.'s, and it is signed UPWARD.
        # Setting it aside to put the two on a comparable basis therefore
        # WIDENS the discrepancy rather than explaining it away. An earlier
        # revision emitted only its share of the raw gap and read that share
        # as "most of the difference is accounted for", which inverts the
        # direction: a term present on one side only, pushing that side up,
        # cannot explain why that side is low. Emit the comparable-basis sum
        # and gap so the manuscript quotes the widening rather than a ratio
        # that invites the wrong inference.
        macros["ResidualShareOfDollsGap"] = fmt(
            100 * disp_residual / dolls_gap if dolls_gap else float("nan"), 0
        )
        disp_sum_comparable = disp_sum - disp_residual
        macros["DispSumExPension"] = fmt(disp_sum_comparable)
        macros["DollsGapExPension"] = fmt(dolls_total - disp_sum_comparable)
        macros["ResidualShareOfDispSum"] = fmt(
            100 * disp_residual / disp_sum if disp_sum else float("nan"), 0
        )
        # HSV progressivity implied by the weighted tax-and-NI schedule.
        # T(y)=y-lambda*y^(1-tau) gives 1-m=(1-tau)(1-a), so tau=(m-a)/(1-a).
        # Both this and the naive m-a it is contrasted against were hand-typed
        # in an earlier draft and went stale when the wedge moved; generate
        # them so the contrast cannot drift from the levels it is derived
        # from.
        m_rate = 100 * (
            levels["wage_cut"]["income_tax"]
            + levels["wage_cut"]["employee_national_insurance"]
        )
        a_rate = conc_tax_ni
        macros["ScheduleTau"] = f"{(m_rate - a_rate) / (100 - a_rate):.2f}"
        macros["ScheduleTauNaive"] = f"{wedge / 100:.2f}"
        macros["ScheduleWedgeWeighted"] = fmt(wedge)
        macros["ScheduleWedgeOffsets"] = fmt(wedge - conc_step_seeds)

        # The factorial STEPS on the channel seeds, for the same reason: the
        # manuscript must not present the 50-draw step and the 5-seed channel
        # arithmetic as though they were the same decomposition.
        macros["FactorialConcStepChannelSeeds"] = fmt(
            sums["wage_cut"] - sums["concentrated_wage_cut"]
        )
        macros["FactorialStateStepChannelSeeds"] = fmt(
            sums["concentrated_wage_cut"] - sums["displacement"]
        )

    # ZERO-HOURS DECOMPOSITION of the employment-state step, when it has been
    # run. Splits the step into the hours channel (which gates three
    # childcare entitlements through in_work) and a status-only residual that
    # ought to be exactly zero, since employment_status is read by no tax or
    # benefit formula. The manuscript previously asserted the step WAS the
    # hours effect; measured, the hours channel is a small fraction of it and
    # the remainder is noise around zero.
    zero_hours_path = args.factorial.parent / "zero_hours_residual.json"
    if zero_hours_path.exists():
        zh = json.loads(zero_hours_path.read_text())
        n_zh = zh["design"]["n_seeds"]
        macros["ZeroHoursSeeds"] = str(n_zh)
        for key, label in (
            ("hours_channel_pp", "ZeroHoursChannel"),
            ("status_only_residual_pp", "ZeroHoursStatusResidual"),
            ("employment_state_step_pp", "ZeroHoursStateStep"),
        ):
            block = zh[key]
            macros[label] = fmt(block["mean"], 2)
            macros[f"{label}SE"] = fmt(block["sd"] / (n_zh ** 0.5), 2)

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
