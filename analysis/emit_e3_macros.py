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

* ``\\EThreeAshe...`` / ``\\EThreeHigherRate...`` -- the ASHE validation of
  the exposed employees' earnings (results/e3_ashe.json, written by
  analysis/ashe_validation.py). Emitted only when that file exists.
* ``\\EThreeSens...`` -- duration, take-up and New Style JSA sensitivity
  (results/e3_sensitivity.json, analysis/sensitivity_duration_takeup.py).
  Emitted only when that file exists.

* ``\\EThreeUnif...`` -- the uniform benchmark: the same gross loss spread
  over every employee (results/e3_uniform.json,
  analysis/e3_uniform_and_income_shares.py). ``...GapPts`` is tariff pattern
  minus uniform.
* ``\\EThree{Disp,Wage}SharePctDec<N>`` -- gross earnings loss as a
  percentage of the decile's baseline HBAI household net income;
  ``\\EThree{Disp,Wage}NetSharePctDec<N>`` -- the post-cushion net income
  loss on the same base (``\\EThreeNetSharePctDec<N>`` is the displacement
  one, the central margin); ``...SpendSharePctDec<N>`` -- the net loss as a
  percentage of consumption; ``\\EThreeLossSharePctDec<N>`` -- the decile's
  share of the total gross loss. ``<N>`` runs One..Ten, plus ``All``.
  DECILES HERE ARE HOUSEHOLD-WEIGHTED deciles of equivalised HBAI net income
  (as in the price episodes); the older ``...DecGbp<N>`` macros use
  person-weighted deciles. From results/e3_income_shares.json.
* ``\\EThreeInd...`` -- summary of the industry table
  (results/e3_industry.json, analysis/e3_industry_table.py).

The wage-cut margin mixes the Universal Credit claiming re-draw exactly
(all-claim and none-claim simulations, weights uc_takeup and 1 - uc_takeup),
like the displacement margin. ``\\EThreeWageDraws`` is kept because the
appendix uses it in "averages \\EThreeWageDraws\\ take-up draws"; it now
expands to words that keep that sentence true ("the all-claim and
none-claim") rather than to a draw count, which would be zero.

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


def _k(value: float) -> str:
    """GBP thousand, one decimal."""
    return f"{value / 1e3:.1f}"


def ashe_macros(ashe: dict) -> dict:
    """Macros for the ASHE validation of exposed employees' earnings."""
    bench = ashe["ashe"][ashe["benchmark_edition"]]
    overall = ashe["model"]["overall"]
    displaced = overall["expected_displaced_weighted"]
    mean = bench["expected_displaced_weighted_mean"]
    median = bench["expected_displaced_weighted_mean_of_division_medians"]
    above = bench["share_above_higher_rate"]
    top = next(k for k in displaced if k.startswith("loss_share_of_"))
    n_top = top.split("_")[3]
    m = {
        # mean earnings of the expected displaced workers (gross loss / workers)
        "EThreeAsheModelMeanK": _k(displaced["mean"]),
        "EThreeAsheMeanK": _k(mean["ashe"]),
        "EThreeAsheModelOverAsheMeanPct": f"{100 * bench['model_over_ashe_mean']:.0f}",
        # like-for-like median: weighted mean of division medians, both sides
        "EThreeAsheModelMedianK": _k(median["model"]),
        "EThreeAsheMedianK": _k(median["ashe"]),
        "EThreeAsheModelOverAsheMedianPct": f"{100 * bench['model_over_ashe_median']:.0f}",
        # the model's pooled median over expected displaced workers
        "EThreeAsheModelPooledMedianK": _k(displaced["median"]),
        "EThreeAsheModelEmployeeMeanK": _k(overall["employee_weighted"]["mean"]),
        "EThreeAsheModelEmployeeMedianK": _k(overall["employee_weighted"]["median"]),
        "EThreeAsheLossWeightedRatioPct": f"{100 * bench['loss_weighted_ratio_of_means']:.0f}",
        "EThreeAsheJobCountMeanK": _k(bench["ashe_mean_with_ashe_job_counts"]["ashe"]),
        "EThreeAsheManufacturingMeanK": _k(bench["manufacturing_section_c"]["mean_uprated"]),
        "EThreeAsheManufacturingMedianK": _k(bench["manufacturing_section_c"]["median_uprated"]),
        "EThreeAsheDivisions": f"{mean['divisions']}",
        "EThreeAsheEdition": ashe["benchmark_edition"],
        "EThreeAsheTaxYear": bench["tax_year"].replace("-", "--"),
        "EThreeAsheUpratingPct": f"{100 * (bench['uprating_factor'] - 1):.1f}",
        # higher-rate exposure
        "EThreeHigherRateThreshold": f"{ashe['higher_rate_threshold']:,.0f}",
        "EThreeHigherRateSharePct": f"{100 * displaced['share_above_higher_rate']:.0f}",
        "EThreeHigherRateLossSharePct": (
            f"{100 * displaced['share_of_gross_loss_above_higher_rate']:.0f}"
        ),
        "EThreeAsheHigherRateSharePct": f"{100 * above['ashe_interpolated']:.0f}",
        "EThreeAsheHigherRateLowPct": f"{100 * above['ashe_lower_bound']:.0f}",
        "EThreeAsheHigherRateHighPct": f"{100 * above['ashe_upper_bound']:.0f}",
        "EThreeAsheHigherRateModelSharePct": f"{100 * above['model']:.0f}",
        # concentration: the mean without the largest loss records
        "EThreeAsheTopRecords": n_top,
        "EThreeAsheTopLossSharePct": f"{100 * displaced[top]:.1f}",
        "EThreeAsheModelMeanExTopK": _k(
            displaced[f"mean_without_{n_top}_largest_loss_records"]
        ),
    }
    check = next((v for k, v in ashe["ashe"].items() if k != ashe["benchmark_edition"]), None)
    if check is not None:
        m["EThreeAsheCheckEdition"] = check["edition"]
        m["EThreeAsheCheckMeanK"] = _k(check["expected_displaced_weighted_mean"]["ashe"])
        m["EThreeAsheCheckMedianK"] = _k(
            check["expected_displaced_weighted_mean_of_division_medians"]["ashe"]
        )
    return m


def sensitivity_macros(sens: dict) -> dict:
    """Macros for the duration, take-up and New Style JSA sensitivity."""
    pct = lambda row, key="cushioning_rate": f"{100 * row[key]:.1f}"
    takeup = sens["uc_takeup"]
    grid = sorted(takeup, key=float)
    low, high = grid[0], grid[-1]
    m = {
        "EThreeSensTakeupLow": low,
        "EThreeSensTakeupHigh": high,
        "EThreeSensTakeupLowRatePct": pct(takeup[low]),
        "EThreeSensTakeupHighRatePct": pct(takeup[high]),
        "EThreeSensTakeupLowTaxBenRatePct": pct(takeup[low], "tax_benefit_cushioning_rate"),
        "EThreeSensTakeupHighTaxBenRatePct": pct(takeup[high], "tax_benefit_cushioning_rate"),
        "EThreeSensTakeupRangePts": (
            f"{100 * (takeup[high]['cushioning_rate'] - takeup[low]['cushioning_rate']):.1f}"
        ),
    }
    for weeks, stem in (("0", "Zero"), ("13", "Thirteen")):
        row = sens["new_style_jsa_weeks"][weeks]
        m[f"EThreeSensJsa{stem}RatePct"] = pct(row)
        m[f"EThreeSensJsa{stem}TaxBenRatePct"] = pct(row, "tax_benefit_cushioning_rate")
    lines = (("income_tax", "Tax"), ("national_insurance", "Ni"),
             ("universal_credit", "Uc"), ("jsa_contrib", "JsaContrib"))
    for label, stem in (("6m", "SixMonth"), ("3m", "ThreeMonth")):
        spell = sens["duration"].get(label)
        if spell is None:
            continue
        m[f"EThreeSens{stem}JsaWeeks"] = f"{spell['new_style_jsa_weeks']:.0f}"
        for method, tag in (("annual_averaging", "Annual"), ("monthly_state_mix", "Mix")):
            row = spell[method]
            m[f"EThreeSens{stem}{tag}RatePct"] = pct(row)
            m[f"EThreeSens{stem}{tag}TaxBenRatePct"] = pct(row, "tax_benefit_cushioning_rate")
            for line, line_stem in lines:
                m[f"EThreeSens{stem}{tag}{line_stem}OfGrossPct"] = (
                    f"{100 * row['share_of_gross_loss'][line]:.1f}"
                )
    return m


DECILE_WORDS = ["One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten"]
_LINES = (("income_tax", "Tax"), ("national_insurance", "Ni"), ("universal_credit", "Uc"),
          ("jsa_contrib", "JsaContrib"), ("pension_contributions", "Pension"))


def uniform_macros(unif: dict) -> dict:
    """Macros for the uniform (economy-wide) benchmark."""
    rows, gaps = unif["rows"], unif["tariff_minus_uniform"]
    m = {
        "EThreeUnifShockPct": f"{100 * unif['uniform_shock']:.3f}",
        "EThreeUnifEmployeesM": f"{unif['employees'] / 1e6:.1f}",
        "EThreeUnifEmployeeRecords": f"{unif['employee_records']:,}",
        "EThreeUnifGrossM": f"{unif['gross_earnings_loss'] / 1e6:,.0f}",
        "EThreeUnifDispWorkersK": (
            f"{rows['uniform_displacement']['extra']['displaced_weighted_mean'] / 1e3:.0f}"
        ),
    }
    for margin, key in (("wage_cut", "Wage"), ("displacement", "Disp")):
        row, gap = rows[f"uniform_{margin}"], gaps[margin]
        m[f"EThreeUnif{key}RatePct"] = f"{100 * row['cushioning_rate']:.1f}"
        m[f"EThreeUnif{key}TaxBenRatePct"] = f"{100 * row['tax_benefit_cushioning_rate']:.1f}"
        m[f"EThreeUnif{key}SamplingSe"] = f"{100 * row['cushioning_rate_sampling_se']:.1f}"
        m[f"EThreeUnif{key}ExchequerM"] = f"{row['exchequer_cost'] / 1e6:,.0f}"
        m[f"EThreeUnif{key}GapPts"] = f"{100 * gap['cushioning_rate']:+.1f}"
        m[f"EThreeUnif{key}TaxBenGapPts"] = f"{100 * gap['tax_benefit_cushioning_rate']:+.1f}"
        for line, stem in _LINES:
            m[f"EThreeUnif{key}{stem}OfGrossPct"] = f"{100 * row['share_of_gross_loss'][line]:.1f}"
            m[f"EThreeUnif{key}{stem}GapPts"] = f"{100 * gap['share_of_gross_loss'][line]:+.1f}"
    return m


def income_share_macros(income: dict) -> dict:
    """Macros for losses as shares of decile income and consumption."""
    shares = income["shares"]
    keys = [*map(str, range(1, 11)), "all"]
    words = [*DECILE_WORDS, "All"]
    m = {}

    def put(stem, table, field, digits=3):
        for key, word in zip(keys, words):
            m[f"EThree{stem}PctDec{word}"] = f"{100 * table[key][field]:.{digits}f}"

    for pattern, prefix in (("tariff", ""), ("uniform", "Unif")):
        for margin, key in (("displacement", "Disp"), ("wage_cut", "Wage")):
            table = shares[f"{pattern}_{margin}"]
            put(f"{prefix}{key}Share", table, "gross_loss_share_of_income")
            put(f"{prefix}{key}NetShare", table, "net_loss_share_of_income")
            put(f"{prefix}{key}NetSpendShare", table, "net_loss_share_of_consumption")
            put(f"{prefix}{key}CushionRate", table, "cushioning_rate", 1)
        put(f"{prefix}LossShare", shares[f"{pattern}_wage_cut"], "share_of_total_gross_loss", 1)
    # the central margin, under the short name
    put("NetShare", shares["tariff_displacement"], "net_loss_share_of_income")
    return m


def industry_macros(industry: dict) -> dict:
    rows = industry["rows"]
    return {
        "EThreeIndDivisions": f"{len(rows)}",
        "EThreeIndRowsShown": f"{industry['top_rows_shown']}",
        "EThreeIndTopLossSharePct": f"{100 * rows[0]['loss_share']:.0f}",
        "EThreeIndTopThreeLossSharePct": f"{100 * industry['top3_loss_share']:.0f}",
        "EThreeIndTopFiveLossSharePct": f"{100 * industry['top5_loss_share']:.0f}",
        "EThreeIndEmployeesM": f"{industry['total']['employees'] / 1e6:.2f}",
        "EThreeIndMeanShockPct": f"{100 * industry['total']['shock']:.2f}",
        "EThreeIndMaxShockPct": f"{100 * max(r['shock'] for r in rows):.2f}",
    }


def main() -> None:
    disp = _load("full_tariff_displacement")
    no_jsa = _load("full_tariff_displacement_no_jsa")
    wage = _load("full_tariff_wage_cut")
    bern = _load("full_tariff_displacement_bernoulli")
    if None in (disp, no_jsa, wage, bern):
        raise FileNotFoundError("run analysis/run_scenarios.py --e3 first")
    if disp.get("design") != "exact_expectation":
        raise ValueError("full_tariff_displacement.json is not the exact-expectation run")
    if wage.get("design") != "exact_takeup_expectation":
        raise ValueError(
            "full_tariff_wage_cut.json is not the exact-take-up run; run "
            "analysis/run_scenarios.py --e3 --e3-part wage_cut"
        )
    if wage["uc_takeup"] != disp["uc_takeup"]:
        raise ValueError("wage-cut and displacement runs use different UC take-up")
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
        # no draws: the claiming re-draw is mixed exactly (see module docstring)
        "EThreeWageDraws": "the all-claim and none-claim",
        "EThreeWageTakeupPasses": f"{wage['n_simulation_passes']}",
        "EThreeWageRedrawnUnits": f"{wage['redrawn_benefit_units']}",
        "EThreeWageAllClaimRatePct": f"{100 * wage['takeup_nodes']['1']['cushioning_rate']:.2f}",
        "EThreeWageNoneClaimRatePct": f"{100 * wage['takeup_nodes']['0']['cushioning_rate']:.2f}",
        "EThreeWageSamplingSe": f"{100 * wage['cushioning_rate_sampling_se']:.1f}",
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
    # Tax-and-benefit-only rates: the cushion net of employee pension
    # contributions ceasing (HBAI deducts them; they are neither tax nor benefit).
    for res, key in ((disp, "Disp"), (wage, "Wage")):
        g = res["cushion_component_shares"]["share_of_gross_loss"]
        rate = res["cushioning_rate"] if key == "Disp" else wage_rate
        m[f"EThree{key}TaxBenRatePct"] = f"{100 * (rate - g['pension_contributions']):.1f}"
    # Decile incidence of the tariff scenario (change in household income per
    # head, GBP a year, person-weighted deciles of equivalised income).
    def _dec(res):
        if "decile_income_change" in res:
            return res["decile_income_change"]
        draws = res["draws"]
        return {k: float(np.mean([dr["decile_income_change"][k] for dr in draws]))
                for k in draws[0]["decile_income_change"]}
    names = ["One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten"]
    for res, key in ((disp, "Disp"), (wage, "Wage")):
        dc = _dec(res)
        for i, n in enumerate(names, start=1):
            m[f"EThree{key}DecGbp{n}"] = f"{abs(dc[str(i)]):.1f}"
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
    ashe = _load("e3_ashe")
    if ashe is not None:
        m.update(ashe_macros(ashe))
    sens = _load("e3_sensitivity")
    if sens is not None:
        if abs(sens["central"]["cushioning_rate"] - disp["cushioning_rate"]) > 1e-6:
            raise ValueError("e3_sensitivity.json was built on a different central run")
        m.update(sensitivity_macros(sens))
    for name, build in (("e3_uniform", uniform_macros),
                        ("e3_income_shares", income_share_macros),
                        ("e3_industry", industry_macros)):
        payload = _load(name)
        if payload is not None:
            m.update(build(payload))
    bad = [k for k in m if not k.isalpha()]
    if bad:
        raise ValueError(f"macro names must be letters only: {bad}")
    lines = ["% Generated by analysis/emit_e3_macros.py -- do not edit."]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in m.items()]
    out = RESULTS / "generated_e3.tex"
    out.write_text("\n".join(lines) + "\n")
    shutil.copy(out, ROOT / "paper" / "generated_e3.tex")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
