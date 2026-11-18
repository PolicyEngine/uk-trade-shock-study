"""Consumption-side aggregation of the measured energy-episode incidence:
an MPC-weighted first-round demand calculation and an Atkinson-weighted
welfare aggregation.  Policy simulation, not causal estimation: every
behavioural parameter is imported from the literature and declared, and
the outputs are conditional on the paper's declared first stage.

Method and sources
------------------
Demand.  First-round consumption effect of each instrument (and of the
net position) is sum_h w_h * MPC_h * transfer_h, with MPC assigned by
equivalised-income decile.  Three declared MPC scenarios:
  - "gradient" (central): MPC falls linearly 0.8 (D1) -> 0.3 (D10) --
    a deliberately steep STYLISATION of the cash-on-hand/liquidity
    gradients of Jappelli-Pistaferri (2014) and Fagereng et al. (2021);
    mapped into income deciles the empirical gradient is flatter, which
    the flat-0.5 scenario brackets.
  - "flat": MPC = 0.5 for all households (Fagereng et al.'s first-year
    average), which switches off the heterogeneity channel and isolates
    what the gradient contributes.
  - "asymmetric": UK survey evidence (Bunn, Le Roux, Reinold and
    Surico, 2018) finds MPCs out of negative income shocks (~0.5) far
    larger than out of positive ones (~0.14); burdens are weighted with
    the former and transfers with the latter, both flat.  NOTE: the low
    gain-MPC concerns windfalls, while the EPG truncates a loss, so
    valuing every instrument at 0.14 makes this an OUTER BOUND on the
    demand gap, not a co-equal scenario.
All parameters are IMPORTED, bracketed by scenario, never estimated.

Welfare.  Atkinson (1970) social-welfare weights on equivalised income,
lambda_h proportional to y_h^(-gamma), normalised to a weighted mean of
one, for gamma in {0, 1, 2}.  The reported object is the "regressivity
premium": the welfare-weighted aggregate burden divided by the
unweighted (gamma = 0) aggregate -- the factor by which distributional
concentration inflates the welfare cost of a pound-identical shock.

All weighted arithmetic is delegated to microdf via the shared helpers.

Run:  python mpc_welfare.py   (certified Microcosm UK via policyengine.py)
Emits out/generated_mpc.tex and out/table_mpc.tex, copied to the paper.
"""

import argparse

import numpy as np

from second_stage_energy import P, load_sim, col_components, ctr_rebate, energy_convention, weighted, wmean
from second_stage_energy import RESULTS_DIR, PAPER_DIR

MPC_GRADIENT = np.linspace(0.8, 0.3, 10)   # Jappelli-Pistaferri / FHN
MPC_FLAT = np.full(10, 0.5)                # FHN first-year average
MPC_NEG, MPC_POS = 0.5, 0.14               # Bunn et al. (2018), UK


def deciles_of(equiv_income, w):
    order = np.argsort(equiv_income)
    cw = np.cumsum(w[order])
    dec = np.zeros(len(equiv_income), dtype=int)
    dec[order] = np.minimum((cw / cw[-1] * 10).astype(int), 9)
    return dec


def _tex(x: float) -> str:
    """A signed number with a proper minus sign in LaTeX."""
    return f"$-${abs(x):,.1f}" if x < 0 else f"{x:,.1f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None,
                    help="ignored; data come from the certified Microcosm UK build")
    args = ap.parse_args()

    yr = P["sim_year"]
    sim = load_sim()

    def hh(name):
        return np.asarray(sim.calculate(name, yr, map_to="household"))

    w = np.asarray(sim.calculate("household_weight", yr))
    energy_raw = np.asarray(sim.calculate("domestic_energy_consumption", yr))
    equiv_income = np.asarray(sim.calculate("equiv_hbai_household_net_income", yr))
    dec = deciles_of(equiv_income, w)

    # Shock and instruments, identical conventions to the second stage.
    # Central price-path convention (second_stage_energy.energy_convention).
    conv = energy_convention(sim, w, energy_raw, yr)
    rebase = conv["rebase"]
    energy = energy_raw * rebase
    counter_rise = conv["counter_rise"]
    realised_rise = conv["realised_rise"]
    dE_counter = energy * counter_rise
    epg = energy * (counter_rise - realised_rise)
    ebss = np.full_like(energy, P["ebss_per_household"])
    mt = sum(hh(v) for v in ["universal_credit", "pension_credit", "tax_credits",
                             "income_support", "housing_benefit", "jsa_income",
                             "esa_income"])
    col = sum(col_components(sim))

    ctr = ctr_rebate(sim)
    flows = {"burden": -dE_counter, "epg": epg, "ebss": ebss, "ctr": ctr,
             "col": col}
    flows["net"] = sum(flows.values())

    # --- MPC-weighted first-round demand ------------------------------------
    def demand(flow, mpc_by_dec=None, mpc_scalar=None):
        mpc = mpc_by_dec[dec] if mpc_by_dec is not None else mpc_scalar
        return weighted(flow * mpc, w) / 1e9

    out = {}
    for name, flow in flows.items():
        out[f"{name}_gradient"] = demand(flow, mpc_by_dec=MPC_GRADIENT)
        out[f"{name}_flat"] = demand(flow, mpc_by_dec=MPC_FLAT)
    # Asymmetric: burden at MPC_NEG, transfers at MPC_POS.
    out["net_asym"] = (demand(flows["burden"], mpc_scalar=MPC_NEG)
                       + sum(demand(flows[k], mpc_scalar=MPC_POS)
                             for k in ("epg", "ebss", "ctr", "col")))
    # Demand-efficiency: consumption protected per pound of fiscal cost.
    for k in ("epg", "ebss", "ctr", "col"):
        out[f"{k}_per_pound"] = out[f"{k}_gradient"] / (weighted(flows[k], w) / 1e9)

    # --- Atkinson-weighted welfare aggregation ------------------------------
    y = np.maximum(equiv_income, 1000.0)     # DECLARED dial: GBP 1,000 floor
                                             # against near-zero survey incomes;
                                             # the gamma=2 premium is sensitive to it
    prem = {}
    for gamma in (0.0, 1.0, 2.0):
        lam = y ** (-gamma)
        lam = lam / wmean(lam, w)            # weighted mean one
        prem[gamma] = weighted(dE_counter * lam, w) / weighted(dE_counter, w)

    # --- Macros and table ---------------------------------------------------
    G = weighted(dE_counter, w) / 1e9
    m = {
        "MpcNetGradientBn": f"{out['net_gradient']:,.1f}",
        "MpcNetFlatBn": f"{out['net_flat']:,.1f}",
        "MpcNetAsymBn": f"{out['net_asym']:,.1f}",
        "MpcBurdenGradientBn": f"{-out['burden_gradient']:,.1f}",
        "MpcBurdenFlatBn": f"{-out['burden_flat']:,.1f}",
        "MpcEpgPerPound": f"{out['epg_per_pound']:.2f}",
        "MpcEbssPerPound": f"{out['ebss_per_pound']:.2f}",
        "MpcCtrPerPound": f"{out['ctr_per_pound']:.2f}",
        "MpcColPerPound": f"{out['col_per_pound']:.2f}",
        "WelfPremiumGammaOne": f"{prem[1.0]:.2f}",
        "WelfPremiumGammaTwo": f"{prem[2.0]:.2f}",
    }
    with open(RESULTS_DIR / "generated_mpc.tex", "w") as f:
        f.write("% generated by mpc_welfare.py -- do not edit by hand.\n")
        for k, v in m.items():
            f.write(f"\\newcommand{{\\Ss{k}}}{{{v}}}\n")

    rows = [("Counterfactual burden (no policy)", "burden"),
            ("Energy Price Guarantee", "epg"),
            ("Energy Bills Support Scheme", "ebss"),
            ("Council Tax energy rebate", "ctr"),
            ("Cost of Living Payments", "col"),
            ("Net position (full stack)", "net")]
    with open(RESULTS_DIR / "table_mpc.tex", "w") as f:
        f.write("% generated by mpc_welfare.py -- do not edit by hand.\n")
        f.write("\\begin{tabular}{lrrr}\n\\toprule\n")
        f.write(" & \\multicolumn{3}{c}{First-round demand effect, \\pounds bn}\\\\\n")
        f.write("\\cmidrule(lr){2-4}\n")
        f.write("Flow & MPC gradient & MPC flat & Fiscal flow \\\\\n\\midrule\n")
        for label, k in rows:
            fiscal = weighted(flows[k], w) / 1e9
            f.write(f"{label} & {_tex(out[f'{k}_gradient'])} & "
                    f"{_tex(out[f'{k}_flat'])} & {_tex(fiscal)} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    import shutil
    shutil.copy(str(RESULTS_DIR / "generated_mpc.tex"), str(PAPER_DIR / "generated_mpc.tex"))
    shutil.copy(str(RESULTS_DIR / "table_mpc.tex"), str(PAPER_DIR / "table_mpc.tex"))

    print({k: round(v, 2) if isinstance(v, float) else v for k, v in out.items()})
    print("welfare premium:", {g: round(p, 2) for g, p in prem.items()})
    print("shock:", round(G, 1))


if __name__ == "__main__":
    main()
