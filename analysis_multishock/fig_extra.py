"""Three referee-round additions to the energy second stage.

1. fig_decomp.png  -- additive decile decomposition of the counterfactual
   shock into the part households bore (realised burden) and what each
   discretionary instrument returned (EPG, EBSS, cost-of-living payments),
   in pounds a year, with bootstrap intervals on the borne component.
2. fig_paths.png   -- three shock paths by decile as a share of spending:
   gross counterfactual, realised, and cash-outlay (the REALISED burden
   scaled by the observed demand response; note the first pass applies
   the same scale to the gross counterfactual instead).
3. Household-bootstrap (sampling) intervals on the headline second-stage
   quantities, emitted as macros.  These are SAMPLING intervals over the
   53,508-record base only: imputation (donor-model) uncertainty is not
   captured and is flagged in the manuscript.

Run:  python fig_extra.py --dataset <enhanced_frs.h5>
"""

import argparse
import json

import numpy as np

from second_stage_energy import P, weighted, wmean, wquantile, variance_split

CASH_OUTLAY_SCALE = 0.77   # observed demand response (MsEnergyCashOutlayScalePct);
                           # applied here to the realised path (the first pass
                           # applies it to the gross counterfactual)
N_BOOT = 300
SEED = 0


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

    yr = P["sim_year"]
    sim = Microsimulation(dataset=args.dataset)

    def hh(name):
        return np.asarray(sim.calculate(name, yr, map_to="household"))

    w = np.asarray(sim.calculate("household_weight", yr))
    energy_raw = np.asarray(sim.calculate("domestic_energy_consumption", yr))
    equiv_income = np.asarray(sim.calculate("equiv_hbai_household_net_income", yr))
    total_spend = np.asarray(sim.calculate("consumption", yr))
    dec = deciles_of(equiv_income, w)

    base_cap_fy = (P["cap_apr_2021"] + P["cap_oct_2021"]) / 2.0
    rebase = base_cap_fy / wmean(energy_raw, w)
    energy = energy_raw * rebase
    counter_rise = (P["cap_apr_2022"] + P["cap_oct_2022"]) / 2.0 / base_cap_fy - 1.0
    realised_rise = (P["cap_apr_2022"] + P["epg_level"]) / 2.0 / base_cap_fy - 1.0
    dE_counter = energy * counter_rise
    dE_realised = energy * realised_rise
    epg = dE_counter - dE_realised

    ebss = np.full_like(energy, P["ebss_per_household"])
    means_tested_amt = sum(hh(v) for v in
                           ["universal_credit", "pension_credit", "tax_credits",
                            "income_support", "housing_benefit", "jsa_income",
                            "esa_income"])
    col = ((means_tested_amt > 0) * P["col_means_tested"]
           + (hh("state_pension") > 0) * P["col_pensioner"]
           + ((hh("pip") + hh("dla") + hh("attendance_allowance")) > 0)
           * P["col_disability"])

    # --- Per-decile means (point estimates) --------------------------------
    def dmeans(x):
        return np.array([wmean(x[dec == d], w[dec == d]) for d in range(10)])

    m_counter, m_real, m_epg = dmeans(dE_counter), dmeans(dE_realised), dmeans(epg)
    m_ebss, m_col = dmeans(ebss), dmeans(col)
    net_borne = m_counter - m_epg - m_ebss - m_col   # can be negative (D1)

    spend = np.array([weighted(total_spend[dec == d], w[dec == d]) /
                      weighted(w[dec == d] * 0 + 1, w[dec == d]) for d in range(10)])
    pct = lambda x: 100 * np.array(
        [weighted(x[dec == d], w[dec == d]) / weighted(total_spend[dec == d],
                                                       w[dec == d])
         for d in range(10)])
    p_gross, p_real = pct(dE_counter), pct(dE_realised)
    p_cash = p_real * CASH_OUTLAY_SCALE

    # --- Household bootstrap (sampling only) -------------------------------
    rng = np.random.default_rng(SEED)
    n = len(w)
    boot_net, boot_within, boot_d1cushion, boot_rate = [], [], [], []
    disc = epg + ebss + col
    spend_share = dE_realised / np.maximum(total_spend, 1)
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        wb = w[idx]
        db = deciles_of(equiv_income[idx], wb)
        boot_rate.append(100 * weighted(disc[idx], wb) / weighted(dE_counter[idx], wb))
        m1 = db == 0
        boot_d1cushion.append(100 * weighted(disc[idx][m1], wb[m1]) /
                              weighted(dE_counter[idx][m1], wb[m1]))
        boot_net.append([wmean((dE_counter - disc)[idx][db == d], wb[db == d])
                         for d in range(10)])
        ss = spend_share[idx]
        lo, hi = wquantile(ss, wb, 0.01), wquantile(ss, wb, 0.99)
        t = (ss >= lo) & (ss <= hi)
        b, wi = variance_split(ss[t], wb[t], db[t])
        boot_within.append(wi)
    boot_net = np.array(boot_net)
    ci = lambda a: (np.percentile(a, 2.5, axis=0), np.percentile(a, 97.5, axis=0))
    net_lo, net_hi = ci(boot_net)
    within_lo, within_hi = np.percentile(boot_within, [2.5, 97.5])
    d1_lo, d1_hi = np.percentile(boot_d1cushion, [2.5, 97.5])
    rate_lo, rate_hi = np.percentile(boot_rate, [2.5, 97.5])

    # --- Macros ------------------------------------------------------------
    with open("../paper_multishock/generated_extra.tex", "w") as f:
        f.write("% generated by fig_extra.py -- do not edit by hand.\n")
        f.write(f"\\newcommand{{\\SsBootReps}}{{{N_BOOT}}}\n")
        for k, v in [("SsWithinLo", within_lo), ("SsWithinHi", within_hi),
                     ("SsRateLo", rate_lo), ("SsRateHi", rate_hi),
                     ("SsCushionDecOneLo", d1_lo), ("SsCushionDecOneHi", d1_hi)]:
            f.write(f"\\newcommand{{\\{k}}}{{{v:.1f}}}\n")

    # --- Figures -----------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from make_figures import apply_style, BLUE, TEAL, BLUE_LIGHT, INK

    GREY = "#808080"
    x = np.arange(1, 11)
    apply_style()

    # 1. Decomposition.
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    fig.subplots_adjust(left=0.11, right=0.97, top=0.91, bottom=0.30)
    borne = np.maximum(net_borne, 0)
    over = np.minimum(net_borne, 0)
    ax.bar(x, borne, color=BLUE, label="Borne by the household", zorder=3)
    ax.bar(x, m_epg, bottom=borne, color=BLUE_LIGHT,
           label="Returned: Energy Price Guarantee", zorder=3)
    ax.bar(x, m_ebss, bottom=borne + m_epg, color=TEAL,
           label="Returned: Bills Support Scheme", zorder=3)
    ax.bar(x, m_col, bottom=borne + m_epg + m_ebss, color=TEAL, hatch="///",
           edgecolor="white", linewidth=0, label="Returned: cost-of-living payments",
           zorder=3)
    ax.bar(x, over, color=TEAL, hatch="xxx", edgecolor="white", linewidth=0,
           label="Over-compensation (net gain)", zorder=3)
    ax.errorbar(x, net_borne, yerr=[net_borne - net_lo, net_hi - net_borne],
                fmt="none", ecolor=INK, elinewidth=1.0, capsize=2.5, zorder=4,
                label="95% sampling interval, net position")
    ax.plot(x, m_counter, color=INK, lw=1.2, marker="o", ms=4, ls="--",
            label="Counterfactual shock", zorder=5)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(x)
    ax.set_xlabel("Equivalised disposable income decile")
    ax.set_ylabel("£ per household per year")
    ax.set_title("Who bore the energy shock, and what each instrument returned",
                 color=INK)
    ax.legend(fontsize=8, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, -0.14), frameon=False)
    fig.savefig("../paper_multishock/figures/fig_decomp.png", dpi=300)

    # 2. Paths.
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    fig.subplots_adjust(left=0.10, right=0.97, top=0.90, bottom=0.24)
    ax.plot(x, p_gross, color=BLUE, ls="--", marker="o", mfc="white",
            label="Gross counterfactual (no policy)")
    ax.plot(x, p_real, color=BLUE, ls="-", marker="o",
            label="Realised path (EPG in force)")
    ax.plot(x, p_cash, color=GREY, ls=":", marker="s", mfc="white",
            label="Cash outlay (realised × observed demand response)")
    ax.set_xticks(x)
    ax.set_xlabel("Equivalised disposable income decile")
    ax.set_ylabel("% of household expenditure")
    ax.set_title("Three shock paths, one distribution", color=INK)
    ax.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.18),
              ncol=2, frameon=False)
    fig.savefig("../paper_multishock/figures/fig_paths.png", dpi=300)

    print(json.dumps({"within_ci": [round(within_lo, 1), round(within_hi, 1)],
                      "rate_ci": [round(rate_lo, 1), round(rate_hi, 1)],
                      "d1_cushion_ci": [round(d1_lo, 1), round(d1_hi, 1)]}))


if __name__ == "__main__":
    main()
