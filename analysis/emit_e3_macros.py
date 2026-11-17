"""Emit the tariff-episode (E3) support and robustness macros.

Reads the E3 result files written by ``analysis/run_scenarios.py --e3`` and
writes results/generated_e3.tex, copied to paper/. Every E3 number the
appendix quotes comes from here.

Three families of macro:

* ``\\EThreeDisp...`` / ``\\EThreeWage...`` -- the CENTRAL estimates. The
  displacement margin is the exact-expectation design with the New Style
  JSA approximation (results/full_tariff_displacement.json); ``...NoJsa...``
  is the same without it.
* ``\\EThreeBern...`` -- the superseded 20-draw Bernoulli run (no New Style
  JSA), kept as a comparator for the per-draw dispersion.
* LEGACY names the paper already uses for per-draw quantities
  (``\\EThreeDispDraws``, ``\\EThreeDispRateMin/Max/Pooled/McSe``,
  ``\\EThreeDispGrossMeanM/MinM/MaxM``, ``\\EThreeDispRecords``,
  ``\\EThreeDispEffRecords``, ``\\EThreeDispMaxShare``,
  ``\\EThreeDispBelowWage``) stay defined and still describe the BERNOULLI
  draws -- the only thing a per-draw statistic can describe.

Split macros: ``\\EThree{Disp,DispNoJsa,Wage}<Line>SharePct`` is the line's
share of the total cushion (the lines sum to 100);
``...<Line>OfGrossPct`` is its share of the gross earnings loss (the lines
sum to the cushioning rate).

Run after run_scenarios:  python analysis/emit_e3_macros.py
"""

import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

#: result-file group -> macro stem
SPLIT_LINES = {
    "income_tax": "Tax",
    "national_insurance": "Ni",
    "universal_credit": "Uc",
    "other_means_tested": "OtherBen",
    "jsa_contrib": "JsaContrib",
    "esa_contrib": "EsaContrib",
    "child_benefit": "ChildBen",
    "pension_contributions": "Pension",
    "student_loan_repayments": "StudentLoan",
    "statutory_pay": "StatPay",
    "other": "Other",
    "negative_income_floor": "Floor",
}


def _load(name):
    path = RESULTS / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


def _split(result: dict, key: str) -> dict:
    shares = result["cushion_component_shares"]
    missing = set(SPLIT_LINES) ^ set(shares["share_of_cushion"])
    if missing:
        raise KeyError(f"split lines out of step with the result file: {missing}")
    total = sum(shares["share_of_cushion"].values())
    if abs(total - 1.0) > 1e-6:
        raise AssertionError(f"{key}: split shares sum to {total}, not 1")
    m = {}
    for line, stem in SPLIT_LINES.items():
        m[f"EThree{key}{stem}SharePct"] = f"{100 * shares['share_of_cushion'][line]:.1f}"
        m[f"EThree{key}{stem}OfGrossPct"] = f"{100 * shares['share_of_gross_loss'][line]:.1f}"
    m[f"EThree{key}CushionM"] = f"{result['cushion_components_mean']['cushion'] / 1e6:,.0f}"
    return m


def main() -> None:
    disp = _load("full_tariff_displacement")
    no_jsa = _load("full_tariff_displacement_no_jsa")
    wage = _load("full_tariff_wage_cut")
    bern = _load("full_tariff_displacement_bernoulli")
    if None in (disp, no_jsa, wage, bern):
        raise FileNotFoundError("run analysis/run_scenarios.py --e3 first")
    if disp.get("design") != "exact_expectation":
        raise ValueError("full_tariff_displacement.json is not the exact-expectation run")
    d = bern["draws"]
    rates = np.array([x["cushioning_rate"] for x in d])
    gross = np.array([x["gross_earnings_loss"] for x in d])
    net = np.array([x["net_disposable_loss"] for x in d])
    wage_rate = wage["cushioning_rate_pooled"]
    bern_pooled = 1 - net.mean() / gross.mean()
    # linearised SE of the pooled (ratio-of-means) Bernoulli rate over draws
    z = (net - (net.mean() / gross.mean()) * gross) / gross.mean()
    bern_pooled_se = float(np.sqrt(z.var(ddof=1) / len(z)))
    comp, comp_nj = disp["cushion_components_mean"], no_jsa["cushion_components_mean"]
    m = {
        # ---- central: exact-expectation displacement, New Style JSA on
        "EThreeDispRatePct": f"{100 * disp['cushioning_rate']:.1f}",
        "EThreeDispSamplingSe": f"{100 * disp['cushioning_rate_sampling_se']:.1f}",
        "EThreeDispFirstOrderRatePct": f"{100 * disp['cushioning_rate_first_order']:.1f}",
        "EThreeDispGrossM": f"{disp['target_gross_earnings_loss'] / 1e6:,.0f}",
        "EThreeDispNetLossM": f"{disp['net_disposable_loss'] / 1e6:,.0f}",
        "EThreeDispExposedRecords": f"{disp['exposed_records']:,}",
        "EThreeDispExposedHouseholds": f"{disp['exposed_households']:,}",
        "EThreeDispMultiExposedHouseholds": f"{disp['households_with_two_or_more_exposed']:,}",
        "EThreeDispExactEffRecords": f"{disp['effective_loss_records']:.0f}",
        "EThreeDispExactMaxShare": f"{100 * disp['max_record_loss_share']:.1f}",
        "EThreeDispPasses": f"{disp['n_simulation_passes']}",
        "EThreeDispOmittedGrossPct": f"{100 * disp['omitted_gross_share']:.4f}",
        "EThreeDispWorkersK": f"{disp['displaced_weighted_mean'] / 1e3:.0f}",
        "EThreeDispExchequerM": f"{disp['exchequer_cost_mean'] / 1e6:,.0f}",
        "EThreeDispPovertyPp": f"{100 * disp['poverty_rate_change_bhc_mean']:.2f}",
        # ---- New Style JSA variant
        "EThreeDispNoJsaRatePct": f"{100 * no_jsa['cushioning_rate']:.1f}",
        "EThreeDispNoJsaSamplingSe": f"{100 * no_jsa['cushioning_rate_sampling_se']:.1f}",
        "EThreeDispJsaEffectPts": f"{100 * (disp['cushioning_rate'] - no_jsa['cushioning_rate']):.1f}",
        "EThreeDispJsaPaidM": f"{comp['jsa_contrib'] / 1e6:,.0f}",
        "EThreeDispJsaUcOffsetM": f"{(comp_nj['universal_credit'] - comp['universal_credit']) / 1e6:,.0f}",
        "EThreeDispJsaTaxOffsetM": f"{(comp_nj['income_tax'] - comp['income_tax']) / 1e6:,.0f}",
        "EThreeDispJsaWeeks": f"{disp['new_style_jsa_weeks']:.0f}",
        "EThreeDispNoJsaExchequerM": f"{no_jsa['exchequer_cost_mean'] / 1e6:,.0f}",
        # ---- wage cut
        "EThreeWageRatePct": f"{100 * wage_rate:.1f}",
        "EThreeWageDraws": f"{wage['n_draws']}",
        "EThreeWageExchequerM": f"{wage['exchequer_cost_mean'] / 1e6:,.0f}",
        "EThreeDispMinusWagePts": f"{100 * (disp['cushioning_rate'] - wage_rate):.1f}",
        # ---- Bernoulli comparator (superseded 20-draw run, no New Style JSA)
        "EThreeBernDraws": f"{bern['n_draws']}",
        "EThreeBernRateMean": f"{100 * rates.mean():.1f}",
        "EThreeBernRatePooled": f"{100 * bern_pooled:.1f}",
        "EThreeBernRateSd": f"{100 * rates.std(ddof=1):.1f}",
        "EThreeBernRatePooledSe": f"{100 * bern_pooled_se:.1f}",
        "EThreeBernMinusExactPts": f"{100 * (bern_pooled - no_jsa['cushioning_rate']):.1f}",
        # ---- legacy per-draw names: still the Bernoulli draws
        "EThreeDispDraws": f"{bern['n_draws']}",
        "EThreeDispRateMin": f"{100 * rates.min():.1f}",
        "EThreeDispRateMax": f"{100 * rates.max():.1f}",
        "EThreeDispRatePooled": f"{100 * bern_pooled:.1f}",
        "EThreeDispRateMcSe": f"{100 * bern['cushioning_rate_mc_se']:.1f}",
        "EThreeDispGrossMeanM": f"{gross.mean() / 1e6:,.0f}",
        "EThreeDispGrossMinM": f"{gross.min() / 1e6:,.0f}",
        "EThreeDispGrossMaxM": f"{gross.max() / 1e6:,.0f}",
        "EThreeDispRecords": f"{np.mean([x['affected_records'] for x in d]):.0f}",
        "EThreeDispEffRecords": f"{np.mean([x['effective_loss_records'] for x in d]):.1f}",
        "EThreeDispMaxShare": f"{100 * np.mean([x['max_record_loss_share'] for x in d]):.0f}",
        "EThreeDispBelowWage": "every" if (rates < wage_rate).all() else "not every",
    }
    m.update(_split(disp, "Disp"))
    m.update(_split(no_jsa, "DispNoJsa"))
    m.update(_split(wage, "Wage"))
    for margin, key in (("displacement", "Disp"), ("wage_cut", "Wage")):
        low = _load(f"full_tariff_obr_low_{margin}")
        if low is None:
            continue
        rate = low.get("cushioning_rate_pooled", low["cushioning_rate_mean"])
        m[f"EThreeObrLow{key}RatePct"] = f"{100 * rate:.1f}"
        g = low.get("target_gross_earnings_loss")
        if g is None:
            g = np.mean([x["gross_earnings_loss"] for x in low["draws"]])
        m[f"EThreeObrLow{key}GrossM"] = f"{g / 1e6:,.0f}"
    low_nj = _load("full_tariff_obr_low_displacement_no_jsa")
    if low_nj is not None:
        m["EThreeObrLowDispNoJsaRatePct"] = f"{100 * low_nj['cushioning_rate']:.1f}"
    lines = ["% Generated by analysis/emit_e3_macros.py -- do not edit."]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in m.items()]
    out = RESULTS / "generated_e3.tex"
    out.write_text("\n".join(lines) + "\n")
    shutil.copy(out, ROOT / "paper" / "generated_e3.tex")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
