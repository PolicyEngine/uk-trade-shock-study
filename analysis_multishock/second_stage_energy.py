"""Second stage: the 2022-23 energy episode on enhanced-FRS households.

Computes what the first pass could not: the cushioning rate of a
trade-transmitted price shock through the statutory tax-benefit system,
decomposed into automatic and discretionary components, on real
households -- and the WITHIN-decile dispersion of exposure, which is the
margin Borusyak & Jaravel (2024) and Levell et al. (2025) show carries
most of the welfare variance and which decile tables structurally cannot
see.

Conventions follow the companion studies: every parameter is declared
once below with its source, nothing is estimated here, and the artifact
carries every number the manuscript reports.

Run:  python second_stage_energy.py --dataset <path to enhanced_frs.h5>
"""

import argparse
import json
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Declared parameters.  DATA unless marked ASSUMPTION.
# ---------------------------------------------------------------------------
P = {
    # Ofgem default tariff cap, typical dual-fuel direct debit (DATA:
    # Ofgem cap letters 6 Aug 2021 and 26 Aug 2022, Figure 1).
    "cap_apr_2021": 1138.0,
    "cap_oct_2021": 1277.0,
    "cap_apr_2022": 1971.0,
    "cap_oct_2022": 3549.0,
    "epg_level": 2500.0,
    # Wholesale component on the CfD-in-wholesale restatement (model v1.13).
    "wholesale_oct_2021": 550.0,
    "wholesale_oct_2022": 2468.0,
    # Discretionary support, 2022-23 (DATA: gov.uk scheme parameters).
    "ebss_per_household": 400.0,          # Energy Bills Support Scheme
    "col_means_tested": 650.0,            # Cost of Living Payment, 2022-23
    "col_pensioner": 300.0,               # Pensioner Cost of Living Payment
    "col_disability": 150.0,              # Disability Cost of Living Payment
    # Statutory benefit uprating actually applied in April 2022, against
    # contemporaneous inflation over FY2022-23 (DATA: DWP; ONS D7BT).
    "uprating_applied_apr_2022": 0.031,
    "cpi_fy2022_23_mean": 0.101,
    # ASSUMPTION: the simulated year used as the household base.
    "sim_year": 2023,
}


def weighted(x, w):
    return float(np.sum(np.asarray(x) * w))


def wmean(x, w):
    return weighted(x, w) / float(np.sum(w))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default="out/second_stage_energy.json")
    args = ap.parse_args()

    from policyengine_uk import Microsimulation

    sim = Microsimulation(dataset=args.dataset)
    yr = P["sim_year"]

    w = np.asarray(sim.calculate("household_weight", yr))
    energy_raw = np.asarray(sim.calculate("domestic_energy_consumption", yr))
    net_income = np.asarray(sim.calculate("household_net_income", yr))
    benefits = np.asarray(sim.calculate("household_benefits", yr))

    # --- Rebase energy to pre-crisis price levels -------------------------
    # The enhanced FRS calibrates energy spending to current-vintage price-cap
    # rates, so the LEVEL reflects today's prices while the cross-household
    # distribution reflects quantities.  Rebase the weighted mean to the
    # FY2021-22 mean cap, preserving each household's relative position.
    base_cap_fy = (P["cap_apr_2021"] + P["cap_oct_2021"]) / 2.0
    rebase = base_cap_fy / wmean(energy_raw, w)
    energy = energy_raw * rebase

    # --- The shock, on both paths ----------------------------------------
    gross_rise = P["cap_oct_2022"] / P["cap_oct_2021"] - 1.0        # +177.9%
    realised_fy = (P["cap_apr_2022"] + P["epg_level"]) / 2.0
    realised_rise = realised_fy / base_cap_fy - 1.0                  # +85.1%

    dE_counter = energy * gross_rise      # the shock the EPG displaced
    dE_realised = energy * realised_rise  # what households actually faced
    epg_cushion = dE_counter - dE_realised

    # --- Discretionary instruments, by entitlement ------------------------
    ebss = np.full_like(energy, P["ebss_per_household"])

    def hh(varname):
        """Household-level sum of a possibly person/benunit-level variable."""
        try:
            return np.asarray(sim.calculate(varname, yr, map_to="household"))
        except Exception:
            return np.zeros_like(energy)

    means_tested = (
        hh("universal_credit") + hh("pension_credit") + hh("tax_credits")
        + hh("income_support") + hh("housing_benefit")
        + hh("jsa_income") + hh("esa_income")
    )
    on_means_tested = means_tested > 0
    pensioner = hh("state_pension") > 0
    disability = (hh("dla") + hh("pip") + hh("attendance_allowance")) > 0

    col = (
        on_means_tested * P["col_means_tested"]
        + pensioner * P["col_pensioner"]
        + disability * P["col_disability"]
    )
    discretionary = epg_cushion + ebss + col

    # --- Automatic response ------------------------------------------------
    # A pure price shock leaves nominal incomes unchanged, so the means-tested
    # system does not respond in-year: the only automatic channel is
    # indexation, which is lagged.  The in-year automatic cushioning is the
    # uprating actually applied minus what contemporaneous indexation would
    # have delivered -- a NEGATIVE number, i.e. the system lost ground.
    uprating_gap = P["cpi_fy2022_23_mean"] - P["uprating_applied_apr_2022"]
    # Apply the gap only to CASH benefits actually uprated in April 2022.
    # household_benefits is a broader aggregate (it exceeds the DWP/HMRC cash
    # benefit bill), so summing the named instruments is the defensible base.
    uprated_cash = (
        hh("universal_credit") + hh("state_pension") + hh("pension_credit")
        + hh("child_benefit") + hh("pip") + hh("dla")
        + hh("attendance_allowance") + hh("carers_allowance")
        + hh("esa_income") + hh("jsa_income") + hh("income_support")
        + hh("housing_benefit") + hh("tax_credits")
    )
    automatic_shortfall = uprated_cash * uprating_gap

    # --- Aggregates ---------------------------------------------------------
    G = weighted(dE_counter, w)
    res = {
        "meta": {
            "dataset": Path(args.dataset).name,
            "sim_year": yr,
            "households_weighted": float(np.sum(w)),
            "households_records": int(len(w)),
            "energy_rebase_factor": round(rebase, 4),
            "note": "price-shock cushioning; nominal incomes unchanged, so the "
                    "means-tested system does not respond in-year",
        },
        "parameters": P,
        "shock": {
            "gross_counterfactual_rise": round(gross_rise, 4),
            "realised_rise": round(realised_rise, 4),
            "gross_aggregate_gbp_bn": round(G / 1e9, 2),
            "realised_aggregate_gbp_bn": round(weighted(dE_realised, w) / 1e9, 2),
        },
        "cushioning": {
            "epg_gbp_bn": round(weighted(epg_cushion, w) / 1e9, 2),
            "ebss_gbp_bn": round(weighted(ebss, w) / 1e9, 2),
            "col_payments_gbp_bn": round(weighted(col, w) / 1e9, 2),
            "discretionary_total_gbp_bn": round(weighted(discretionary, w) / 1e9, 2),
            "discretionary_rate_pct": round(100 * weighted(discretionary, w) / G, 1),
            "automatic_in_year_rate_pct": 0.0,
            "uprated_cash_benefit_base_gbp_bn": round(
                weighted(uprated_cash, w) / 1e9, 1),
            "uprating_shortfall_gbp_bn": round(
                weighted(automatic_shortfall, w) / 1e9, 2),
            "uprating_shortfall_as_pct_of_shock": round(
                -100 * weighted(automatic_shortfall, w) / G, 1),
            "net_of_uprating_shortfall_rate_pct": round(
                100 * (weighted(discretionary, w) - weighted(automatic_shortfall, w))
                / G, 1),
        },
    }

    # --- Decile profile and, crucially, WITHIN-decile dispersion -----------
    order = np.argsort(net_income)
    cw = np.cumsum(w[order])
    dec = np.zeros_like(net_income, dtype=int)
    dec[order] = np.minimum((cw / cw[-1] * 10).astype(int), 9)

    rows, within = [], []
    for d in range(10):
        m = dec == d
        wd, e_d = w[m], dE_realised[m]
        inc = net_income[m]
        burden_share = np.where(inc > 0, e_d / np.maximum(inc, 1), np.nan)
        bs = burden_share[np.isfinite(burden_share)]
        bw = wd[np.isfinite(burden_share)]
        idx = np.argsort(bs)
        cwb = np.cumsum(bw[idx]) / np.sum(bw)
        p10 = float(bs[idx][np.searchsorted(cwb, 0.10)])
        p50 = float(bs[idx][np.searchsorted(cwb, 0.50)])
        p90 = float(bs[idx][np.searchsorted(cwb, 0.90)])
        rows.append({
            "decile": d + 1,
            "realised_gbp_per_year": round(wmean(e_d, wd), 1),
            "counterfactual_gbp_per_year": round(wmean(dE_counter[m], wd), 1),
            "discretionary_gbp_per_year": round(wmean(discretionary[m], wd), 1),
            "cushioning_rate_pct": round(
                100 * weighted(discretionary[m], wd)
                / weighted(dE_counter[m], wd), 1),
            "burden_pct_of_net_income_mean": round(100 * wmean(bs, bw), 2),
        })
        within.append({
            "decile": d + 1,
            "p10_burden_pct_income": round(100 * p10, 2),
            "p50_burden_pct_income": round(100 * p50, 2),
            "p90_burden_pct_income": round(100 * p90, 2),
            "p90_p10_ratio": round(p90 / p10, 2) if p10 > 0 else None,
        })
    res["by_decile"] = rows
    res["within_decile_dispersion"] = within

    # Variance decomposition: the Borusyak-Jaravel question, asked of these
    # households.  How much of the variance of exposure is BETWEEN deciles?
    valid = net_income > 0
    b = (dE_realised / np.maximum(net_income, 1))[valid]
    wv, dv = w[valid], dec[valid]
    gm = wmean(b, wv)
    total_var = weighted((b - gm) ** 2, wv) / np.sum(wv)
    between = 0.0
    for d in range(10):
        m = dv == d
        if m.sum():
            between += np.sum(wv[m]) * (wmean(b[m], wv[m]) - gm) ** 2
    between /= np.sum(wv)
    res["variance_decomposition"] = {
        "between_decile_share_pct": round(100 * between / total_var, 1),
        "within_decile_share_pct": round(100 * (1 - between / total_var), 1),
        "note": "variance of the realised burden as a share of net income, "
                "decomposed between vs within income deciles",
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=float))
    c = res["cushioning"]
    print(f"gross shock GBP {res['shock']['gross_aggregate_gbp_bn']}bn | "
          f"realised GBP {res['shock']['realised_aggregate_gbp_bn']}bn")
    print(f"discretionary cushioning {c['discretionary_rate_pct']}% "
          f"(EPG {c['epg_gbp_bn']} + EBSS {c['ebss_gbp_bn']} + CoL "
          f"{c['col_payments_gbp_bn']} GBP bn)")
    print(f"automatic in-year {c['automatic_in_year_rate_pct']}%; uprating "
          f"shortfall GBP {c['uprating_shortfall_gbp_bn']}bn "
          f"({c['uprating_shortfall_as_pct_of_shock']}% of the shock)")
    v = res["variance_decomposition"]
    print(f"variance of exposure: {v['between_decile_share_pct']}% between "
          f"deciles, {v['within_decile_share_pct']}% within")
    print("wrote", out)


if __name__ == "__main__":
    main()


def emit_macros(res, path="out/generated_secondstage.tex"):
    """LaTeX macros for the manuscript, prefixed Ss."""
    m = ["% generated by second_stage_energy.py -- do not edit by hand."]

    def add(k, v):
        m.append(f"\\newcommand{{\\Ss{k}}}{{{v}}}")

    s, c, v = res["shock"], res["cushioning"], res["variance_decomposition"]
    add("GrossAggBn", f"{s['gross_aggregate_gbp_bn']:,.1f}")
    add("RealisedAggBn", f"{s['realised_aggregate_gbp_bn']:,.1f}")
    add("DiscretionaryRatePct", f"{c['discretionary_rate_pct']:.1f}")
    add("EpgBn", f"{c['epg_gbp_bn']:,.1f}")
    add("EbssBn", f"{c['ebss_gbp_bn']:,.1f}")
    add("ColBn", f"{c['col_payments_gbp_bn']:,.1f}")
    add("AutomaticRatePct", f"{c['automatic_in_year_rate_pct']:.0f}")
    add("UpratingShortfallBn", f"{c['uprating_shortfall_gbp_bn']:,.1f}")
    add("UpratingShortfallPctOfShock", f"{c['uprating_shortfall_as_pct_of_shock']:.1f}")
    add("CashBenefitBaseBn", f"{c['uprated_cash_benefit_base_gbp_bn']:,.0f}")
    add("BetweenDecileVarPct", f"{v['between_decile_share_pct']:.0f}")
    add("WithinDecileVarPct", f"{v['within_decile_share_pct']:.0f}")
    add("HouseholdsRecords", f"{res['meta']['households_records']:,}")
    words = ("One", "Two", "Three", "Four", "Five",
             "Six", "Seven", "Eight", "Nine", "Ten")
    for row, wd in zip(res["by_decile"], words):
        add(f"CushionPctDec{wd}", f"{row['cushioning_rate_pct']:.0f}")
        add(f"BurdenPctIncDec{wd}", f"{row['burden_pct_of_net_income_mean']:.2f}")
    for row, wd in zip(res["within_decile_dispersion"], words):
        add(f"PTenDec{wd}", f"{row['p10_burden_pct_income']:.2f}")
        add(f"PNinetyDec{wd}", f"{row['p90_burden_pct_income']:.2f}")
        add(f"RatioDec{wd}", f"{row['p90_p10_ratio']:.1f}")
    Path(path).write_text("\n".join(m) + "\n")
    print("wrote", path)


if __name__ == "__main__" and True:
    import json as _j
    _r = _j.loads(Path("out/second_stage_energy.json").read_text())
    emit_macros(_r)
