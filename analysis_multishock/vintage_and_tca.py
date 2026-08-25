"""Executes the two pieces of the design the referee rounds identified as
promised but not delivered.

1. THE RULEBOOK-VINTAGE RUN (energy episode).  The enhanced FRS is a
   single-period (2023) dataset, so a 2022 calculation on the raw dataset
   has no inputs and is meaningless.  Here every input column is copied
   to the 2022 period, so the SAME households are scored under the 2022
   and 2023 parameter systems -- the framework's "previous year's rules"
   estimand, computed rather than approximated.  The two-vintage
   difference measures what a legislated uprating delivers through the
   full statutory structure (tapers included); dividing by the April
   2023 uprating rate (10.1 per cent) calibrates pounds-per-uprating-
   point, from which the crisis-year lag cost (the withheld 7.0 points)
   follows as a rulebook-calibrated figure rather than a flat
   base-times-gap multiplication.  The linearity of that final step is
   declared.

2. THE TCA FOOD EPISODE ON HOUSEHOLD RECORDS.  The common cushioning
   estimand, computed for a second episode -- a trade-POLICY shock --
   through the same second stage: burden by decile, the (empty)
   discretionary layer, and the within/between variance decomposition,
   on the enhanced FRS food-consumption imputation.

Also recomputes the cost-of-living entitlement flags under both
parameter vintages (the eligibility-composition sensitivity).

Run:  python vintage_and_tca.py --dataset <enhanced_frs.h5>
Emits out/generated_vintage.tex, copied to the paper.
"""

import argparse

import h5py  # noqa: F401  (h5 read via pandas)
import numpy as np
import pandas as pd

from second_stage_energy import P, weighted, wmean, wquantile, variance_split

UPRATED_CASH = ["universal_credit", "pension_credit", "tax_credits",
                "income_support", "housing_benefit", "jsa_income",
                "esa_income", "state_pension", "child_benefit", "pip",
                "dla", "attendance_allowance", "carers_allowance",
                "winter_fuel_allowance"]
MEANS_TESTED = ["universal_credit", "pension_credit", "tax_credits",
                "income_support", "housing_benefit", "jsa_income",
                "esa_income"]

APRIL_2023_UPRATING = 0.101   # DATA: CPI Sep 2022, applied April 2023
# TCA first stage (imported, Section 4): 8% end-state on food prices,
# window-average path = published-cumulative rescale (0.30 of end-state).
TCA_END_STATE = 0.08
TCA_WINDOW_SHARE = 0.30
# Rebase (ASSUMPTION, mirrors the energy convention): the enhanced-FRS
# food imputation sits at its own calibrated level, so the weighted mean
# is rebased to the first pass's ONS FYE2022 base -- GBP 258.8/yr at the
# 8% end-state (results.json, E1 central) implies GBP 3,235/yr mean food
# spend.  The share denominators stay current-vintage, the same hybrid
# declared for energy.
TCA_FOOD_BASE_YR = 258.8 / TCA_END_STATE


def deciles_of(equiv_income, w):
    order = np.argsort(equiv_income)
    cw = np.cumsum(w[order])
    dec = np.zeros(len(equiv_income), dtype=int)
    dec[order] = np.minimum((cw / cw[-1] * 10).astype(int), 9)
    return dec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()

    from policyengine_uk import Microsimulation

    sim = Microsimulation(dataset=args.dataset)
    vars_ = sim.tax_benefit_system.variables

    # --- Copy every dataset input to the 2022 period ------------------------
    n, failed = 0, []
    for ent in ("person", "benunit", "household"):
        df = pd.read_hdf(args.dataset, ent)
        for col in df.columns:
            if col in vars_:
                try:
                    sim.set_input(col, 2022, df[col].values)
                    n += 1
                except Exception as e:
                    failed.append((col, str(e)[:60]))
    if failed:
        print(f"WARNING: {len(failed)} input copies failed:", failed)

    w = np.asarray(sim.calculate("household_weight", 2023))

    def agg(v, yr):
        return weighted(np.asarray(sim.calculate(v, yr, map_to="household")),
                        w) / 1e9

    b22 = sum(agg(v, 2022) for v in UPRATED_CASH)
    b23 = sum(agg(v, 2023) for v in UPRATED_CASH)
    uprating_effect = b23 - b22                      # GBP bn from +10.1%
    per_point = uprating_effect / (100 * APRIL_2023_UPRATING)
    gap_pp = 100 * (P["cpi_fy2022_23_mean"] - P["uprating_applied_apr_2022"])
    lag_cost = per_point * gap_pp                    # linear-in-rate step

    # CoL eligibility composition under both vintages.
    def mt_flag(yr):
        amt = sum(np.asarray(sim.calculate(v, yr, map_to="household"))
                  for v in MEANS_TESTED)
        return amt > 0

    f22, f23 = mt_flag(2022), mt_flag(2023)
    col_hh_22 = weighted(f22.astype(float), w) / 1e6
    col_hh_23 = weighted(f23.astype(float), w) / 1e6

    # --- TCA food episode on household records ------------------------------
    food = np.asarray(sim.calculate(
        "food_and_non_alcoholic_beverages_consumption", 2023,
        map_to="household"))
    total_spend = np.asarray(sim.calculate("consumption", 2023)) * (
        P["ons_total_spend_yr"] / wmean(np.asarray(sim.calculate("consumption", 2023)), w))
    equiv_income = np.asarray(
        sim.calculate("equiv_hbai_household_net_income", 2023))
    dec = deciles_of(equiv_income, w)

    food_rebase = TCA_FOOD_BASE_YR / wmean(food, w)
    burden = food * food_rebase * TCA_END_STATE * TCA_WINDOW_SHARE  # window-avg
    G = weighted(burden, w) / 1e9
    dshare = [100 * weighted(burden[dec == d], w[dec == d])
              / weighted(total_spend[dec == d], w[dec == d]) for d in range(10)]
    dgbp = [wmean(burden[dec == d], w[dec == d]) for d in range(10)]

    ss = burden / np.maximum(total_spend, 1)
    lo, hi = wquantile(ss, w, 0.01), wquantile(ss, w, 0.99)
    t = (ss >= lo) & (ss <= hi)
    between, within = variance_split(ss[t], w[t], dec[t])
    p10 = 100 * wquantile(ss[dec == 0], w[dec == 0], 0.10)
    p90 = 100 * wquantile(ss[dec == 0], w[dec == 0], 0.90)

    # --- Macros -------------------------------------------------------------
    m = {
        "VinInputsCopied": f"{n}",
        "VinBenefitBaseRuleTwentyTwo": f"{b22:,.1f}",
        "VinBenefitBaseRuleTwentyThree": f"{b23:,.1f}",
        "VinUpratingEffectBn": f"{uprating_effect:,.1f}",
        "VinPerPointBn": f"{per_point:,.2f}",
        "VinLagCostBn": f"{lag_cost:,.1f}",
        "VinLagPctOfBase": f"{100*lag_cost/b22:.1f}",
        "VinColHouseholdsRuleTwentyTwoM": f"{col_hh_22:,.1f}",
        "VinColHouseholdsRuleTwentyThreeM": f"{col_hh_23:,.1f}",
        "TcaSsAggBn": f"{G:,.1f}",
        "TcaSsFoodRebase": f"{food_rebase:.3f}",
        "TcaSsPctSpendDecOne": f"{dshare[0]:.2f}",
        "TcaSsPctSpendDecTen": f"{dshare[9]:.2f}",
        "TcaSsGbpDecOne": f"{dgbp[0]:,.0f}",
        "TcaSsGbpDecTen": f"{dgbp[9]:,.0f}",
        "TcaSsBetweenPct": f"{between:.1f}",
        "TcaSsWithinPct": f"{within:.1f}",
        "TcaSsPTenDecOne": f"{p10:.2f}",
        "TcaSsPNinetyDecOne": f"{p90:.2f}",
    }
    with open("out/generated_vintage.tex", "w") as f:
        f.write("% generated by vintage_and_tca.py -- do not edit by hand.\n")
        for k, v in m.items():
            f.write(f"\\newcommand{{\\Ss{k}}}{{{v}}}\n")
    import shutil
    shutil.copy("out/generated_vintage.tex", "../paper_multishock/generated_vintage.tex")

    for k, v in m.items():
        print(k, v)


if __name__ == "__main__":
    main()
