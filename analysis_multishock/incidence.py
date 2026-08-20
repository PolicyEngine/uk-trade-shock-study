#!/usr/bin/env python3
"""Public-data household-incidence pipeline for the UK multi-shock trade study.

Runs five post-Covid UK trade-shock episodes through a common household lens:
  E1  TCA / Brexit NTBs, food prices 2021-23   (declared first stage: Bakker
      et al., CEP Brexit Analysis 18: +8% food, central; 6% sensitivity)
  E2  Energy price shock 2022-23               (Ofgem cap 1,277 -> 3,549 gross;
      2,500 under the Energy Price Guarantee)
  E3  US tariffs 2025                          (UK consumer side literally zero;
      earnings side imported as declared constants from the companion study)
  E4  India CETA 2026+                         (DBT IA Annex 9 Table 18 duty
      savings on final goods, allocated by category spending)
  E5  CPTPP near-zero benchmark                            (DBT central +GBP 2.0bn GDP)

plus a rulebook-arithmetic module (UC uprating lag 2022-23) and a
comparability table.

Conventions mirror the sibling pipelines (taa-pipeline/costing.py,
retaliation-prototype/analyze.py): pinned raw inputs verified by SHA256,
all reported numbers generated (never typed) into LaTeX macros prefixed Ms
(no digits in macro names), every assumption labelled in NOTES.md.

Inputs (raw/, provenance in NOTES.md):
  family_spending_wb1_fye2025.xlsx  ONS Family Spending workbook 1, FYE2025,
                                    sheet 3.1E (equivalised disposable income
                                    decile groups, GBP/week)
  family_spending_wb1_fye2022.xlsx  Same, FYE2022 edition (pre-energy-shock
                                    spending vintage)
  cpi_d7bt.csv                      ONS CPI all-items index D7BT (MM23)
  dwp_benefit_rates_2022_23.pdf     DWP benefit rates 2022/23 (UC standard
                                    allowance 324.84 -> 334.91)
  govuk_benefit_rates_2023_24.html  gov.uk rates 2023/24 (334.91 -> 368.74)

Outputs: out/results.json, out/generated_multishock.tex

Usage: <uk-trade-shock-study>/.venv/bin/python incidence.py   (dep: openpyxl)
"""

import hashlib
import json
import re
import zlib
from pathlib import Path

import openpyxl

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
OUT = ROOT / "out"
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# 0. Pinned raw inputs (SHA256 recorded at download time, 19 Aug 2026)
# ---------------------------------------------------------------------------
PINNED = {
    "family_spending_wb1_fye2025.xlsx":
        "af8d4a4464c0cc0a49195495a6e665667234379ac08293037a69e065d8931076",
    "family_spending_wb1_fye2022.xlsx":
        "549336c3488c9f49638cfc56ed52e94bdc2cb567444a9542b78810e6cb9a2e17",
    "cpi_d7bt.csv":
        "164f5f8987915fb4ed20619b4fae2d5619322719f8eb979a767cd66e018e223e",
    "dwp_benefit_rates_2022_23.pdf":
        "41559e15cbf01f0228f46289082e4ffe7da70bc2ca217561f4f8d51f6564f722",
    "govuk_benefit_rates_2023_24.html":
        "57afb1903302e9883571c813e968f58fe0d9d516c01944ee49c485df9446d733",
}


def verify_inputs():
    for name, expected in PINNED.items():
        p = RAW / name
        if not p.exists():
            raise FileNotFoundError(f"Missing pinned input {p}; see NOTES.md")
        got = hashlib.sha256(p.read_bytes()).hexdigest()
        if got != expected:
            raise ValueError(f"Hash mismatch for {name}: {got} != {expected}")
    print(f"Verified {len(PINNED)} pinned raw inputs.")


# ---------------------------------------------------------------------------
# Declared first stages and statutory constants (cited, not re-estimated)
# ---------------------------------------------------------------------------
D = {
    # E1 -- Bakker, Datta, Davies & De Lyon, CEP Brexit Analysis 18 (2023)
    "tca_food_rise_central": 0.08,     # cumulative food-price rise from NTBs
    "tca_food_rise_low": 0.06,         # sensitivity
    "tca_window_years": 3.25,          # Dec 2019 - Mar 2023
    "tca_published_cumul_gbp": 250.0,  # published cumulative per household
    # E2 -- Ofgem price cap / EPG, typical-consumption annual bill (GBP)
    "cap_winter_2021_22": 1277.0,
    "cap_oct_2022": 3549.0,
    "epg_level": 2500.0,
    # Half-year caps needed for the REALISED FY2022-23 path (Ofgem letters).
    "cap_apr_2021": 1138.0,
    "cap_apr_2022": 1971.0,
    # Wholesale component of the cap, dual fuel / direct debit / TDCV, on the
    # CfD-in-wholesale restatement that makes the two vintages comparable
    # (Ofgem model v1.13 restates Oct-21 wholesale as 550 with policy 137).
    "cap_wholesale_oct_2021": 550.0,
    "cap_wholesale_oct_2022": 2468.0,
    "obr_epg_cost_bn": 27.0,           # context anchor only
    "obr_energy_subsidy_bn": 47.0,     # context anchor only
    # E3 -- companion study declared constants (earnings side)
    "ust_gross_shock_m": 886.0,        # GBP m / yr gross earnings shock
    "ust_displacement_cushion_pct": 36.6,
    "ust_wage_cut_cushion_pct": 43.9,
    # E4 -- DBT India CETA impact assessment, Annex 9 Table 18 (GBP m / yr,
    # consumer duty reductions on final goods)
    "ceta_total_m": 180.0,
    "ceta_clothing_m": 132.3,
    "ceta_footwear_m": 13.2,
    "ceta_foodbev_m": 11.7,
    "ceta_passthrough": {"fifty": 0.50, "seventyfive": 0.75, "hundred": 1.00},
    # E5 -- DBT CPTPP central scenario
    "cptpp_gdp_bn": 2.0,
    "cptpp_households_m": 28.4,
    # Module 2 -- UC standard allowance, single adult 25+, GBP / month
    "uc_apr_2021": 324.84,
    "uc_apr_2022": 334.91,   # +3.1% (Sep 2021 CPI)
    "uc_apr_2023": 368.74,   # +10.1% (Sep 2022 CPI)
    "uc_uplift_monthly": 86.67,       # GBP 20/wk Covid uplift, removed Oct 2021
    "uc_uplift_annual": 1040.0,       # 20 x 52
}


def verify_statutory_rates():
    """Check the hard-coded UC values against the pinned source documents."""
    # gov.uk 2023/24 page carries both the 2022/23 and 2023/24 columns.
    html = (RAW / "govuk_benefit_rates_2023_24.html").read_text(
        encoding="utf-8", errors="replace")
    for v in ("334.91", "368.74"):
        if v not in html:
            raise ValueError(f"UC rate {v} not found in pinned gov.uk page")
    # DWP 2022/23 PDF carries the 2021/22 and 2022/23 columns; text lives in
    # flate-compressed content streams.
    data = (RAW / "dwp_benefit_rates_2022_23.pdf").read_bytes()
    txt = []
    for m in re.finditer(rb"stream\r?\n", data):
        chunk = data[m.end():data.find(b"endstream", m.end())]
        try:
            txt.append(zlib.decompress(chunk).decode("latin1"))
        except zlib.error:
            pass
    blob = " ".join(re.sub(r"[\\()]", "", t) for t in txt)
    for v in ("324.84", "334.91"):
        if v not in blob:
            raise ValueError(f"UC rate {v} not found in pinned DWP PDF")
    print("Verified UC standard-allowance values against pinned sources.")


# ---------------------------------------------------------------------------
# 1. ONS Family Spending, sheet 3.1E (equivalised disposable income deciles)
# ---------------------------------------------------------------------------
def num(x):
    """Parse a Family Spending cell: '..'/':'/'-' (suppressed) -> None,
    '[x]' (low reliability) -> x, '1,234' -> 1234."""
    if x is None:
        return None
    s = str(x).strip()
    if s in ("..", "", ":", "-"):
        return None
    s = s.strip("[]").replace(",", "").replace("~", "")
    try:
        return float(s)
    except ValueError:
        return None


class FamilySpending:
    """Label-based reader for sheet 3.1E of a Family Spending workbook 1.
    Row layout differs across vintages, so rows are located by their COICOP
    code / label, never by fixed index. Columns 4..13 = deciles 1..10
    (lowest..highest equivalised disposable income), column 14 = all."""

    def __init__(self, path, vintage):
        self.vintage = vintage
        wb = openpyxl.load_workbook(path, read_only=True)
        self.rows = list(wb["3.1E"].iter_rows(values_only=True))
        title = str(self.rows[1][0])
        assert "equivalised disposable income decile" in title, title

    def _find(self, pattern):
        for r in self.rows:
            for j in range(4):
                if r[j] is not None and re.fullmatch(
                        pattern, str(r[j]).strip()):
                    return r
        raise KeyError(f"{self.vintage} 3.1E: no row matching {pattern!r}")

    def deciles(self, pattern):
        """(list of ten decile values, all-households value), GBP/week."""
        r = self._find(pattern)
        return [num(r[j]) for j in range(4, 14)], num(r[14])

    @property
    def households_k(self):
        vals, allv = self.deciles(r"Weighted number of households.*")
        return vals, allv

    @property
    def total_expenditure(self):
        return self.deciles(r"Total expenditure\s*")


def build_fye2022_income(fs22, fs25):
    """FYE2022 3.1E publishes decile lower boundaries but not decile means.
    ASSUMPTION (NOTES.md #5): interior decile means ~ boundary midpoints
    (validated on FYE2025 where both are published: midpoints match means to
    <1% for deciles two-nine); open-ended deciles one and ten use the FYE2025
    shape ratios mean(one)/boundary(two) and mean(ten)/boundary(ten), which
    are invariant to the OECD-scale rebasing between vintages."""
    b22 = [num(fs22._find(r"Lower boundary.*")[j]) for j in range(4, 14)]
    b25 = [num(fs25._find(r"Lower boundary.*")[j]) for j in range(4, 14)]
    m25 = [num(fs25._find(r"Mean of group.*")[j]) for j in range(4, 14)]
    # b[0] is blank (open lower tail); b[1..9] are the deciles' lower bounds.
    mid_dev = max(abs((b25[d] + b25[d + 1]) / 2 - m25[d]) / m25[d]
                  for d in range(1, 9))
    inc22 = [None] * 10
    for d in range(1, 9):
        inc22[d] = (b22[d] + b22[d + 1]) / 2
    inc22[0] = b22[1] * (m25[0] / b25[1])
    inc22[9] = b22[9] * (m25[9] / b25[9])
    return inc22, m25, mid_dev


# ---------------------------------------------------------------------------
# 2. CPI (ONS D7BT, all-items index 2015=100)
# ---------------------------------------------------------------------------
def read_cpi():
    cpi = {}
    for line in (RAW / "cpi_d7bt.csv").read_text().splitlines():
        m = re.match(r'"(\d{4}) ([A-Z]{3})","([\d.]+)"', line)
        if m:
            cpi[(int(m.group(1)), m.group(2))] = float(m.group(3))
    return cpi


MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
          "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]


def fy_2022_23_months():
    return ([(2022, MONTHS[i]) for i in range(3, 12)]
            + [(2023, MONTHS[i]) for i in range(0, 3)])


# ---------------------------------------------------------------------------
# 3. Episode engines
# ---------------------------------------------------------------------------
def pct(x, base):
    return None if base in (None, 0) else 100.0 * x / base


def decile_block(cost_wk, texp, inc):
    """Per-decile GBP/yr, % of total spend, % of equivalised disp. income."""
    return [{
        "decile": d + 1,
        "gbp_per_year": round(cost_wk[d] * 52, 1),
        "pct_of_total_spend": round(pct(cost_wk[d], texp[d]), 3),
        "pct_of_equiv_disp_income": round(pct(cost_wk[d], inc[d]), 3),
    } for d in range(10)]


def episode_e1(fs22, inc22):
    """TCA food NTBs. Price vector: +8% (central) on COICOP 1 food &
    non-alcoholic drinks; pass-through embedded in the estimate (no dial).
    Base: FYE2022 spending (pre-Mar 2023 vintage, NOTES.md #6)."""
    food, food_all = fs22.deciles(r"1")
    texp, texp_all = fs22.total_expenditure
    hh_k, hh_all_k = fs22.households_k
    out = {}
    for tag, rise in [("central", D["tca_food_rise_central"]),
                      ("low", D["tca_food_rise_low"])]:
        cost_wk = [f * rise for f in food]
        annual_all = food_all * rise * 52
        # Cumulative over Dec 2019 - Mar 2023 under a LINEAR phase-in from 0
        # to the full price rise (ASSUMPTION, NOTES.md #6): the average effect
        # over the window is half the end-state effect.
        cumul = [c * 52 * D["tca_window_years"] * 0.5 for c in cost_wk]
        cumul_all = annual_all * D["tca_window_years"] * 0.5
        out[tag] = {
            "price_rise": rise,
            "per_decile_end_state_annual": decile_block(cost_wk, texp, inc22),
            "per_decile_cumulative_gbp_linear_phasein":
                [round(c, 1) for c in cumul],
            "per_decile_per_year_over_window":
                [round(c / D["tca_window_years"], 1) for c in cumul],
            "all_households_end_state_gbp_per_year": round(annual_all, 1),
            "all_households_cumulative_gbp_linear_phasein":
                round(cumul_all, 1),
            "aggregate_end_state_gbp_bn_per_year":
                round(annual_all * hh_all_k * 1000 / 1e9, 2),
        }
    c = out["central"]
    # Cross-check vs the published cumulative-per-household figure. The
    # implied average-effect share diagnoses how back-loaded the published
    # phase-in path is relative to our linear assumption (0.5).
    endstate_annual = c["all_households_end_state_gbp_per_year"]
    out["cross_check"] = {
        "published_cumulative_per_household_gbp": D["tca_published_cumul_gbp"],
        "our_cumulative_linear_phasein_gbp":
            c["all_households_cumulative_gbp_linear_phasein"],
        "ratio_ours_to_published": round(
            c["all_households_cumulative_gbp_linear_phasein"]
            / D["tca_published_cumul_gbp"], 2),
        "implied_average_effect_share_of_end_state": round(
            D["tca_published_cumul_gbp"]
            / (endstate_annual * D["tca_window_years"]), 3),
        "note": ("Published GBP 250 implies an average effect ~30% of the "
                 "end-state 8% (heavily back-loaded phase-in, and a food-"
                 "spend base measured before most of the 2022-23 food "
                 "inflation); our linear phase-in (50%) overshoots it, and "
                 "the FYE2022 nominal spend base adds the rest of the gap."),
    }
    return out


def episode_e2(fs22, inc22):
    """Energy 2022-23. Price vector: cap ratios applied to FYE2022 (pre-shock
    vintage) electricity + gas spending; fixed quantities (NOTES.md #7)."""
    elec, elec_all = fs22.deciles(r"4\.4\.1")
    gas, gas_all = fs22.deciles(r"4\.4\.2")
    texp, texp_all = fs22.total_expenditure
    hh_k, hh_all_k = fs22.households_k
    en = [e + g for e, g in zip(elec, gas)]
    en_all = elec_all + gas_all
    # Both paths on ONE basis: financial-year mean caps.  Mixing a
    # point-to-point cap ratio with a financial-year mean ratio changes the
    # denominator of the cushioning rate without saying so.
    _base_fy = (D["cap_apr_2021"] + D["cap_winter_2021_22"]) / 2.0
    _counter_fy = (D["cap_apr_2022"] + D["cap_oct_2022"]) / 2.0
    _realised_fy = (D["cap_apr_2022"] + D["epg_level"]) / 2.0
    gross_f = _counter_fy / _base_fy - 1.0
    net_f = _realised_fy / _base_fy - 1.0
    cushion_share = ((D["cap_oct_2022"] - D["epg_level"])
                     / (D["cap_oct_2022"] - D["cap_winter_2021_22"]))
    gross_wk = [e * gross_f for e in en]
    net_wk = [e * net_f for e in en]
    cushion_yr = [(g - n) * 52 for g, n in zip(gross_wk, net_wk)]
    agg = lambda wk: sum(w * 52 for w in wk) / 10 * hh_all_k * 1000 / 1e9
    gross_bn, net_bn = agg(gross_wk), agg(net_wk)
    return {
        "price_factors": {"gross": round(gross_f, 4), "net_epg": round(net_f, 4)},
        "base_spend_elec_gas_gbp_per_week": {
            "per_decile": en, "all_households": en_all},
        "gross": {"per_decile": decile_block(gross_wk, texp, inc22),
                  "aggregate_gbp_bn_per_year": round(gross_bn, 1)},
        "net_of_epg": {"per_decile": decile_block(net_wk, texp, inc22),
                       "aggregate_gbp_bn_per_year": round(net_bn, 1)},
        "epg_cushion": {
            "per_decile_gbp_per_year": [round(c, 1) for c in cushion_yr],
            "share_of_gross_shock": round(cushion_share, 4),
            "aggregate_gbp_bn_per_year": round(gross_bn - net_bn, 1),
            "aggregate_gbp_bn_six_months": round((gross_bn - net_bn) / 2, 1),
            "note": ("Cushioning share is constant across deciles by "
                     "construction (both shocks proportional to the same "
                     "spend base); the GBP cushion differs by decile."),
        },
        "cross_check": {
            "obr_epg_cost_bn": D["obr_epg_cost_bn"],
            "obr_energy_price_subsidies_bn": D["obr_energy_subsidy_bn"],
            "note": ("Order-of-magnitude context anchor only. Our six-month "
                     "cushion (GBP ~15bn) sits below OBR's GBP 27bn EPG "
                     "costing because OBR priced the EPG against the higher "
                     "caps then expected for Jan-Jun 2023 (GBP ~4,300+), over "
                     "a longer window, on wholesale-price-dependent volumes; "
                     "our counterfactual holds the Oct 2022 cap (GBP 3,549) "
                     "fixed. Same order of magnitude, as required."),
        },
    }


def episode_e3():
    """US tariffs 2025: no UK retaliation, so the UK consumer-price channel
    is literally zero at every decile. Earnings side imported from the
    companion study as declared constants."""
    return {
        "consumer_side": {
            "per_decile_gbp_per_year": [0.0] * 10,
            "note": ("Zero row: the UK imposed no retaliatory tariffs, so "
                     "there is no consumer-price first stage. The episode's "
                     "household incidence runs through earnings only."),
        },
        "earnings_side_declared": {
            "gross_shock_gbp_m_per_year": D["ust_gross_shock_m"],
            "displacement_cushioning_pct": D["ust_displacement_cushion_pct"],
            "wage_cut_cushioning_pct": D["ust_wage_cut_cushion_pct"],
            "source": "companion uk-trade-shock-study (declared constants)",
        },
    }


def episode_e4(fs25):
    """India CETA consumer duty reductions (DBT IA Annex 9 Table 18),
    allocated across deciles in proportion to each category's spending;
    pass-through dial 50/75/100%."""
    cats = {
        "clothing": (r"3\.1", D["ceta_clothing_m"]),
        "footwear": (r"3\.2", D["ceta_footwear_m"]),
        "food_beverages": (r"1", D["ceta_foodbev_m"]),
    }
    texp, texp_all = fs25.total_expenditure
    inc = [num(fs25._find(r"Mean of group.*")[j]) for j in range(4, 14)]
    hh_k, hh_all_k = fs25.households_k
    n_per_decile = hh_all_k * 1000 / 10
    detail = {}
    gain_hh_full = [0.0] * 10   # GBP/yr per household at 100% pass-through
    for cat, (pat, total_m) in cats.items():
        spend, spend_all = fs25.deciles(pat)
        shares = [s / sum(spend) for s in spend]
        per_hh = [total_m * 1e6 * sh / n_per_decile for sh in shares]
        gain_hh_full = [g + p for g, p in zip(gain_hh_full, per_hh)]
        detail[cat] = {
            "annex_table_gbp_m_per_year": total_m,
            "decile_spend_gbp_per_week": spend,
            "decile_share_of_category_spend": [round(s, 4) for s in shares],
            "gain_gbp_per_household_per_year_full_passthrough":
                [round(p, 2) for p in per_hh],
        }
    scen = {}
    for tag, pt in D["ceta_passthrough"].items():
        g = [x * pt for x in gain_hh_full]
        scen[tag] = {
            "passthrough": pt,
            "per_decile_gain_gbp_per_year": [round(x, 2) for x in g],
            "pct_of_total_spend": [round(pct(x / 52, texp[d]), 4)
                                   for d, x in enumerate(g)],
            "pct_of_equiv_disp_income": [round(pct(x / 52, inc[d]), 4)
                                         for d, x in enumerate(g)],
        }
    full = scen["hundred"]
    bot, top = full["pct_of_total_spend"][0], full["pct_of_total_spend"][9]
    return {"categories": detail, "scenarios": scen,
            "progressivity": {
                "bottom_decile_gain_pct_of_spend": bot,
                "top_decile_gain_pct_of_spend": top,
                "bottom_to_top_ratio": round(bot / top, 3),
                "verdict": verdict(bot, top, "gain")}}


def episode_e5():
    per_hh = D["cptpp_gdp_bn"] * 1e9 / (D["cptpp_households_m"] * 1e6)
    return {
        "dbt_central_gdp_gbp_bn_long_run": D["cptpp_gdp_bn"],
        "gdp_pct_long_run": 0.08,
        "naive_mean_gbp_per_household_per_year": round(per_hh, 1),
        "note": ("Near-zero benchmark row: a long-run GDP central estimate divided by "
                 "the household count. No distributional structure is "
                 "claimed or computable from this first stage."),
    }


# ---------------------------------------------------------------------------
# 4. Module 2 -- UC uprating-lag rulebook arithmetic
# ---------------------------------------------------------------------------
def module_two(cpi):
    base = cpi[(2021, "APR")]
    monthly = []
    for (y, mn) in fy_2022_23_months():
        cf = D["uc_apr_2021"] * cpi[(y, mn)] / base
        monthly.append({
            "month": f"{y}-{mn}",
            "cpi_index": cpi[(y, mn)],
            "actual_allowance": D["uc_apr_2022"],
            "counterfactual_contemporaneous_cpi": round(cf, 2),
            "shortfall": round(cf - D["uc_apr_2022"], 2),
        })
    annual_short = sum(m["shortfall"] for m in monthly)
    annual_allow = 12 * D["uc_apr_2022"]
    # Alternative (flat) counterfactual: uprate once in Apr 2022 by the
    # April-2022 year-on-year CPI instead of the lagged Sep-2021 rate.
    apr22_yoy = cpi[(2022, "APR")] / base
    flat_short = (D["uc_apr_2021"] * apr22_yoy - D["uc_apr_2022"]) * 12
    return {
        "parameters": {
            "uc_single_25plus_monthly": {
                "apr_2021": D["uc_apr_2021"], "apr_2022": D["uc_apr_2022"],
                "apr_2023": D["uc_apr_2023"],
                "uprating_apr_2022_pct": round(
                    100 * (D["uc_apr_2022"] / D["uc_apr_2021"] - 1), 2),
                "uprating_apr_2023_pct": round(
                    100 * (D["uc_apr_2023"] / D["uc_apr_2022"] - 1), 2),
            },
            "cpi_anchor": "D7BT April 2021 (allowance effective date)",
        },
        "monthly": monthly,
        "uprating_lag_cost_gbp_fy_2022_23": round(annual_short, 2),
        "uprating_lag_cost_pct_of_allowance": round(
            100 * annual_short / annual_allow, 2),
        "flat_alternative_gbp": round(flat_short, 2),
        "flat_alternative_note": ("Uprating by April-2022 y/y CPI "
                                  f"({100 * (apr22_yoy - 1):.1f}%) instead of "
                                  "the lagged 3.1%, held flat for the year."),
        "uplift_removal": {
            "gbp_per_month": D["uc_uplift_monthly"],
            "gbp_per_year": D["uc_uplift_annual"],
            "note": ("GBP 20/wk Covid uplift removed October 2021: a "
                     "discrete pre-shock cut of GBP 1,040/yr, separate from "
                     "and additional to the uprating lag."),
        },
    }


# ---------------------------------------------------------------------------
# 5. Module 3 -- comparability table
# ---------------------------------------------------------------------------
def verdict(bot, top, sign):
    """Regressivity verdict from bottom/top decile burden as % of spending."""
    if bot is None or top is None:
        return "n/a"
    if top == 0 and bot == 0:
        return "n/a (zero row)"
    r = bot / top if top else float("inf")
    if sign == "cost":
        if r > 1.2:
            return "regressive"
        if r < 0.8:
            return "progressive"
        return "roughly proportional"
    # gains: pro-poor gain = progressive
    if r > 1.2:
        return "progressive (pro-poor gain)"
    if r < 0.8:
        return "regressive (pro-rich gain)"
    return "roughly proportional"


def module_three(e1, e2, e3, e4, e5):
    rows = []

    def add(episode, size, size_label, sign, channel, bot, top, sgn_kind,
            cushioning, size_bn_for_norm):
        rows.append({
            "episode": episode,
            "gross_shock": size, "gross_shock_label": size_label,
            "sign": sign, "channel": channel,
            "bottom_decile_pct_of_spend": bot,
            "top_decile_pct_of_spend": top,
            "regressivity_verdict": verdict(bot, top, sgn_kind),
            "cushioning": cushioning,
            "bottom_decile_pct_of_spend_per_gbp_bn":
                (round(bot / size_bn_for_norm, 3)
                 if bot is not None and size_bn_for_norm else None),
        })

    b1 = e1["central"]["per_decile_end_state_annual"]
    s1 = e1["central"]["aggregate_end_state_gbp_bn_per_year"]
    add("E1 TCA food NTBs", s1, "GBP bn/yr, end-state 8% on food",
        "cost", "consumer prices (food)",
        b1[0]["pct_of_total_spend"], b1[9]["pct_of_total_spend"],
        "cost", "n/a (pass-through embedded)", s1)

    b2g = e2["gross"]["per_decile"]
    s2 = e2["gross"]["aggregate_gbp_bn_per_year"]
    add("E2 Energy 2022-23 (gross)", s2, "GBP bn/yr, gross cap shock",
        "cost", "consumer prices (energy)",
        b2g[0]["pct_of_total_spend"], b2g[9]["pct_of_total_spend"],
        "cost",
        f"EPG cushions {100 * e2['epg_cushion']['share_of_gross_shock']:.1f}%"
        " of the gross shock (all deciles)", s2)

    add("E3 US tariffs 2025", D["ust_gross_shock_m"] / 1000,
        "GBP bn/yr, gross earnings shock (companion study)",
        "cost", "earnings only (no UK retaliation)", 0.0, 0.0, "cost",
        (f"automatic stabilisers: {D['ust_displacement_cushion_pct']:.1f}% "
         f"(displacement) / {D['ust_wage_cut_cushion_pct']:.1f}% (wage cut)"),
        None)

    f4 = e4["scenarios"]["hundred"]
    add("E4 India CETA (2026+)", D["ceta_total_m"] / 1000,
        "GBP bn/yr duty reduction, final goods, 100% pass-through",
        "gain", "consumer prices (clothing, footwear, food)",
        f4["pct_of_total_spend"][0], f4["pct_of_total_spend"][9],
        "gain", "n/a", D["ceta_total_m"] / 1000)

    add("E5 CPTPP benchmark", D["cptpp_gdp_bn"],
        "GBP bn GDP, long-run central", "gain",
        "aggregate GDP (no household mapping)", None, None, "gain",
        "n/a", None)
    return rows


# ---------------------------------------------------------------------------
# 6. LaTeX macros (\newcommand, prefix Ms; no digits in macro names)
# ---------------------------------------------------------------------------
DEC = ["One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight",
       "Nine", "Ten"]


def fmt(x, dp=0):
    return f"{x:,.{dp}f}"


# Observed weather-adjusted fall in domestic GAS consumption, gas year
# mid-May 2022 to mid-May 2023 against the preceding twelve months: DESNZ
# Subnational Electricity and Gas Consumption Statistics, GB, 2022
# (published 25 Jan 2024) -- the one figure DESNZ states in prose.  Used for
# the cash-outlay variant.  DATA, not assumption.
ENERGY_OBSERVED_DEMAND_FALL = 0.127
# Dhingra & Page: MORE than this share of the CPI rise was imported.
IMPORTED_SHARE_LOWER_BOUND = 0.70


def derived_variants(res):
    """Variants and gradients the manuscript reports.

    Kept in one place and written to results.json so the artifact and the
    paper cannot drift apart.  ASSUMPTION tags mark values that are not
    read from a pinned input.
    """
    tca = res["episodes"]["E1_tca_food"]
    en = res["episodes"]["E2_energy"]
    share = tca["cross_check"]["implied_average_effect_share_of_end_state"]
    d = {"tca_window_average": {"share_of_end_state": share, "per_decile": []}}
    for row in tca["central"]["per_decile_end_state_annual"]:
        d["tca_window_average"]["per_decile"].append({
            "decile": row["decile"],
            "gbp_per_year": round(row["gbp_per_year"] * share, 1),
            "pct_of_total_spend": round(row["pct_of_total_spend"] * share, 3),
        })
    d["tca_window_average"]["aggregate_gbp_bn_per_year"] = round(
        tca["central"]["aggregate_end_state_gbp_bn_per_year"] * share, 2)

    # Energy: cash-outlay variant on an OBSERVED quantity response.  The
    # fixed-basket number is the first-order WELFARE measure (envelope
    # theorem); households that cut consumption spent less cash but bore the
    # foregone consumption as welfare loss, so the two answer different
    # questions and both are reported.  ASSUMPTION: observed weather-adjusted
    # domestic demand reduction, set in PARAMS and cited in NOTES.
    p_ratio = 1.0 + en["price_factors"]["gross"]
    q = 1.0 - ENERGY_OBSERVED_DEMAND_FALL
    d["energy_cash_outlay"] = {
        "observed_demand_fall": ENERGY_OBSERVED_DEMAND_FALL,
        "price_ratio": round(p_ratio, 4),
        "expenditure_ratio": round(p_ratio * q, 4),
        "burden_scale_vs_fixed_basket": round(
            (p_ratio * q - 1.0) / (p_ratio - 1.0), 4),
        "note": "cash outlay, not welfare; fixed-basket remains the "
                "first-order welfare measure",
    }
    d["energy_cash_outlay"]["aggregate_gbp_bn_per_year"] = round(
        en["gross"]["aggregate_gbp_bn_per_year"]
        * d["energy_cash_outlay"]["burden_scale_vs_fixed_basket"], 1)
    d["energy_cash_outlay"]["decile_one_gbp_per_year"] = round(
        en["gross"]["per_decile"][0]["gbp_per_year"]
        * d["energy_cash_outlay"]["burden_scale_vs_fixed_basket"])

    # REALISED FY2022-23 path.  The gross counterfactual holds the October
    # 2022 cap for a full year, which no household experienced: the first
    # half of FY2022-23 was capped at 1,971 and the second at the EPG's
    # 2,500.  The realised comparison is against the preceding financial
    # year's two caps (1,138 and 1,277).
    realised = (D["cap_apr_2022"] + D["epg_level"]) / 2.0
    base_fy = (D["cap_apr_2021"] + D["cap_winter_2021_22"]) / 2.0
    r_rise = realised / base_fy - 1.0
    d["energy_realised_path"] = {
        "fy2022_23_mean_cap": round(realised, 2),
        "fy2021_22_mean_cap": round(base_fy, 2),
        "realised_price_rise": round(r_rise, 4),
        "gross_counterfactual_price_rise": en["price_factors"]["gross"],
        "note": "half-year caps 1,971 (Apr-Sep 22) and 2,500 (EPG, Oct 22-Mar 23) "
                "against 1,138 and 1,277; what households actually faced",
    }
    scale = r_rise / en["price_factors"]["gross"]
    d["energy_realised_path"]["per_decile"] = [
        {"decile": row["decile"],
         "gbp_per_year": round(row["gbp_per_year"] * scale, 1),
         "pct_of_total_spend": round(row["pct_of_total_spend"] * scale, 3)}
        for row in en["gross"]["per_decile"]]
    d["energy_realised_path"]["aggregate_gbp_bn_per_year"] = round(
        en["gross"]["aggregate_gbp_bn_per_year"] * scale, 1)

    # MEASURED trade-transmitted share, superseding the CPI-based bound: the
    # cap's wholesale component is the traded block; network, policy,
    # operating costs, margin and VAT are domestic.
    w0, w1 = D["cap_wholesale_oct_2021"], D["cap_wholesale_oct_2022"]
    c0, c1 = D["cap_winter_2021_22"], D["cap_oct_2022"]
    d["energy_wholesale_decomposition"] = {
        "wholesale_oct_2021": w0, "wholesale_oct_2022": w1,
        "wholesale_share_of_cap_oct_2021": round(w0 / c0, 4),
        "wholesale_share_of_cap_oct_2022": round(w1 / c1, 4),
        "wholesale_share_of_cap_rise": round((w1 - w0) / (c1 - c0), 4),
        "non_wholesale_rise": round((c1 - c0) - (w1 - w0), 2),
        "source": "Ofgem default tariff cap letters (6 Aug 2021; 26 Aug 2022), "
                  "Figure 1; CfD-in-wholesale restatement (model v1.13)",
    }
    d["energy_wholesale_decomposition"]["trade_transmitted_aggregate_gbp_bn"] = round(
        en["gross"]["aggregate_gbp_bn_per_year"]
        * d["energy_wholesale_decomposition"]["wholesale_share_of_cap_rise"], 1)

    # Trade-transmitted share: a LOWER bound.  Dhingra-Page report that MORE
    # than this share of the CPI rise was imported, so applying it yields a
    # floor, not a cap.  Where an Ofgem wholesale-component decomposition is
    # available it supersedes this; see NOTES.
    d["energy_trade_transmitted"] = {
        "imported_share_lower_bound": IMPORTED_SHARE_LOWER_BOUND,
        "aggregate_gbp_bn_lower_bound": round(
            en["gross"]["aggregate_gbp_bn_per_year"]
            * IMPORTED_SHARE_LOWER_BOUND, 1),
        "direction": "lower bound (source reports 'more than' this share)",
    }

    # Category budget shares by decile: what the per-GBPbn ratio actually
    # measured.  share = burden %-of-spend divided by the price rise.
    d["budget_shares_pct_of_spend"] = {"food": [], "domestic_energy": []}
    for row in tca["central"]["per_decile_end_state_annual"]:
        d["budget_shares_pct_of_spend"]["food"].append({
            "decile": row["decile"],
            "pct": round(row["pct_of_total_spend"] / tca["central"]["price_rise"], 2)})
    for row in en["gross"]["per_decile"]:
        d["budget_shares_pct_of_spend"]["domestic_energy"].append({
            "decile": row["decile"],
            "pct": round(row["pct_of_total_spend"] / en["price_factors"]["gross"], 2)})
    return d


def latex_macros(res):
    m = []

    def add(name, value):
        assert not re.search(r"\d", name), name
        m.append(f"\\newcommand{{\\Ms{name}}}{{{value}}}")

    e1, e2 = res["episodes"]["E1_tca_food"], res["episodes"]["E2_energy"]
    e3, e4 = res["episodes"]["E3_us_tariffs"], res["episodes"]["E4_india_ceta"]
    e5, m2 = res["episodes"]["E5_cptpp_benchmark"], res["module2_uc_uprating"]

    # E1
    add("TcaFoodRisePct", fmt(100 * D["tca_food_rise_central"]))
    add("TcaFoodRiseLowPct", fmt(100 * D["tca_food_rise_low"]))
    add("TcaWindowYears", f"{D['tca_window_years']:.2f}")
    for d in range(10):
        b = e1["central"]["per_decile_end_state_annual"][d]
        add(f"TcaAnnGbpDec{DEC[d]}", fmt(b["gbp_per_year"]))
        add(f"TcaPctSpendDec{DEC[d]}", fmt(b["pct_of_total_spend"], 2))
        add(f"TcaPctIncDec{DEC[d]}", fmt(b["pct_of_equiv_disp_income"], 2))
        add(f"TcaCumulGbpDec{DEC[d]}", fmt(
            e1["central"]["per_decile_cumulative_gbp_linear_phasein"][d]))
    add("TcaAnnGbpAll", fmt(
        e1["central"]["all_households_end_state_gbp_per_year"]))
    add("TcaCumulGbpAll", fmt(
        e1["central"]["all_households_cumulative_gbp_linear_phasein"]))
    add("TcaPerYearGbpAll", fmt(
        e1["central"]["all_households_cumulative_gbp_linear_phasein"]
        / D["tca_window_years"]))
    add("TcaAggBnPerYear", fmt(
        e1["central"]["aggregate_end_state_gbp_bn_per_year"], 2))
    add("TcaPublishedCumulGbp", fmt(D["tca_published_cumul_gbp"]))
    add("TcaCrossCheckRatio", fmt(
        e1["cross_check"]["ratio_ours_to_published"], 2))
    add("TcaImpliedAvgEffectShare", fmt(
        e1["cross_check"]["implied_average_effect_share_of_end_state"], 2))

    # E2
    add("EnergyCapBase", fmt(D["cap_winter_2021_22"]))
    add("EnergyCapPeak", fmt(D["cap_oct_2022"]))
    add("EnergyEpgLevel", fmt(D["epg_level"]))
    add("EnergyGrossFactorPct", fmt(100 * e2["price_factors"]["gross"], 1))
    add("EnergyNetFactorPct", fmt(100 * e2["price_factors"]["net_epg"], 1))
    for d in range(10):
        g, n = e2["gross"]["per_decile"][d], e2["net_of_epg"]["per_decile"][d]
        add(f"EnergyGrossGbpDec{DEC[d]}", fmt(g["gbp_per_year"]))
        add(f"EnergyGrossPctSpendDec{DEC[d]}", fmt(g["pct_of_total_spend"], 2))
        add(f"EnergyGrossPctIncDec{DEC[d]}",
            fmt(g["pct_of_equiv_disp_income"], 2))
        add(f"EnergyNetGbpDec{DEC[d]}", fmt(n["gbp_per_year"]))
        add(f"EnergyCushionGbpDec{DEC[d]}",
            fmt(e2["epg_cushion"]["per_decile_gbp_per_year"][d]))
    add("EnergyGrossAggBn", fmt(e2["gross"]["aggregate_gbp_bn_per_year"], 1))
    add("EnergyNetAggBn", fmt(e2["net_of_epg"]["aggregate_gbp_bn_per_year"], 1))
    add("EnergyCushionAggBn", fmt(
        e2["epg_cushion"]["aggregate_gbp_bn_per_year"], 1))
    add("EnergyCushionAggSixMonthsBn", fmt(
        e2["epg_cushion"]["aggregate_gbp_bn_six_months"], 1))
    add("EnergyCushionSharePct", fmt(
        100 * e2["epg_cushion"]["share_of_gross_shock"], 1))
    add("EnergyObrEpgBn", fmt(D["obr_epg_cost_bn"]))
    add("EnergyObrSubsidyBn", fmt(D["obr_energy_subsidy_bn"]))

    # E3 (requested MsUSTariff... naming)
    add("USTariffConsumerGbpAllDeciles", fmt(0))
    add("USTariffGrossShockMillions", fmt(D["ust_gross_shock_m"]))
    add("USTariffDisplacementCushionPct",
        fmt(D["ust_displacement_cushion_pct"], 1))
    add("USTariffWageCutCushionPct", fmt(D["ust_wage_cut_cushion_pct"], 1))

    # E4
    add("CetaTotalMillions", fmt(D["ceta_total_m"]))
    add("CetaAllocatedMillions",
        fmt(D["ceta_clothing_m"] + D["ceta_footwear_m"] + D["ceta_foodbev_m"], 1))
    add("CetaClothingMillions", fmt(D["ceta_clothing_m"], 1))
    add("CetaFootwearMillions", fmt(D["ceta_footwear_m"], 1))
    add("CetaFoodBevMillions", fmt(D["ceta_foodbev_m"], 1))
    for tag, name in [("fifty", "PtFifty"), ("seventyfive", "PtSeventyFive"),
                      ("hundred", "PtHundred")]:
        s = e4["scenarios"][tag]
        for d in range(10):
            add(f"CetaGainGbp{name}Dec{DEC[d]}",
                fmt(s["per_decile_gain_gbp_per_year"][d], 2))
    full = e4["scenarios"]["hundred"]
    add("CetaBottomPctSpend", fmt(full["pct_of_total_spend"][0], 3))
    add("CetaTopPctSpend", fmt(full["pct_of_total_spend"][9], 3))
    add("CetaVerdict", e4["progressivity"]["verdict"])

    # E5
    add("CptppGdpBn", fmt(D["cptpp_gdp_bn"], 1))
    add("CptppHouseholdsMillions", fmt(D["cptpp_households_m"], 1))
    add("CptppNaiveGbpPerHousehold", fmt(
        e5["naive_mean_gbp_per_household_per_year"]))

    # Module 2
    add("UcAllowanceAprTwentyOne", fmt(D["uc_apr_2021"], 2))
    add("UcAllowanceAprTwentyTwo", fmt(D["uc_apr_2022"], 2))
    add("UcAllowanceAprTwentyThree", fmt(D["uc_apr_2023"], 2))
    add("UcLagCostGbp", fmt(m2["uprating_lag_cost_gbp_fy_2022_23"]))
    add("UcLagCostPctAllowance", fmt(m2["uprating_lag_cost_pct_of_allowance"], 1))
    add("UcLagCostFlatAltGbp", fmt(m2["flat_alternative_gbp"]))
    add("UcUpliftRemovalGbp", fmt(D["uc_uplift_annual"]))

    # Module 3 normalised column
    for row in res["module3_comparability"]:
        tag = {"E1 TCA food NTBs": "TcaNorm",
               "E2 Energy 2022-23 (gross)": "EnergyNorm"}.get(row["episode"])
        if tag and row["bottom_decile_pct_of_spend_per_gbp_bn"] is not None:
            add(f"{tag}BottomPctPerBn",
                fmt(row["bottom_decile_pct_of_spend_per_gbp_bn"], 3))
    ceta_row = res["module3_comparability"][3]
    if ceta_row["bottom_decile_pct_of_spend_per_gbp_bn"] is not None:
        add("CetaNormBottomPctPerBn",
            fmt(ceta_row["bottom_decile_pct_of_spend_per_gbp_bn"], 3))

    # --- Derived variants, read from res["derived"] so the macros and the
    #     artifact are guaranteed to agree (they are computed once, upstream).
    dv = res["derived"]
    add("TcaImpliedAvgShare", fmt(dv["tca_window_average"]["share_of_end_state"], 2))
    _wa = dv["tca_window_average"]["per_decile"]
    for idx, dn in ((0, "DecOne"), (9, "DecTen")):
        add(f"TcaWindowAvgGbp{dn}", fmt(_wa[idx]["gbp_per_year"]))
        add(f"TcaWindowAvgPctSpend{dn}", fmt(_wa[idx]["pct_of_total_spend"], 2))
    add("TcaWindowAvgAggBn", fmt(dv["tca_window_average"]["aggregate_gbp_bn_per_year"], 1))

    co = dv["energy_cash_outlay"]
    add("EnergyObservedDemandFallPct", fmt(100 * co["observed_demand_fall"]))
    add("EnergyCashOutlayScalePct", fmt(100 * co["burden_scale_vs_fixed_basket"]))
    add("EnergyCashOutlayGbpDecOne", fmt(co["decile_one_gbp_per_year"]))
    add("EnergyCashOutlayAggBn", fmt(co["aggregate_gbp_bn_per_year"], 1))

    rp = dv["energy_realised_path"]
    add("EnergyRealisedRisePct", fmt(100 * rp["realised_price_rise"], 1))
    add("EnergyRealisedMeanCapFyTwentyTwo", fmt(rp["fy2022_23_mean_cap"]))
    add("EnergyRealisedMeanCapFyTwentyOne", fmt(rp["fy2021_22_mean_cap"]))
    add("EnergyRealisedAggBn", fmt(rp["aggregate_gbp_bn_per_year"], 1))
    for idx, dn in ((0, "DecOne"), (9, "DecTen")):
        add(f"EnergyRealisedGbp{dn}", fmt(rp["per_decile"][idx]["gbp_per_year"]))
        add(f"EnergyRealisedPctSpend{dn}",
            fmt(rp["per_decile"][idx]["pct_of_total_spend"], 2))

    wd = dv["energy_wholesale_decomposition"]
    add("EnergyWholesaleBase", fmt(wd["wholesale_oct_2021"]))
    add("EnergyWholesalePeak", fmt(wd["wholesale_oct_2022"]))
    add("EnergyWholesaleShareCapBasePct", fmt(100 * wd["wholesale_share_of_cap_oct_2021"], 1))
    add("EnergyWholesaleShareCapPeakPct", fmt(100 * wd["wholesale_share_of_cap_oct_2022"], 1))
    add("EnergyWholesaleShareOfRisePct", fmt(100 * wd["wholesale_share_of_cap_rise"], 1))
    add("EnergyTradeTransmittedAggBn", fmt(wd["trade_transmitted_aggregate_gbp_bn"], 1))
    add("EnergyRealisedTradeTransmittedAggBn",
        fmt(rp["aggregate_gbp_bn_per_year"] * wd["wholesale_share_of_cap_rise"], 1))

    tt = dv["energy_trade_transmitted"]
    add("EnergyImportedShareLowerBound", fmt(100 * tt["imported_share_lower_bound"]))
    add("EnergyTradeTransmittedAggBnFloor", fmt(tt["aggregate_gbp_bn_lower_bound"], 1))

    bs = dv["budget_shares_pct_of_spend"]
    for key, tag in (("food", "Tca"), ("domestic_energy", "Energy")):
        for idx, dn in ((0, "DecOne"), (1, "DecTwo"), (8, "DecNine"), (9, "DecTen")):
            add(f"{tag}BudgetSharePct{dn}", fmt(bs[key][idx]["pct"], 1))
    return m


# ---------------------------------------------------------------------------
def main():
    verify_inputs()
    verify_statutory_rates()
    fs25 = FamilySpending(RAW / "family_spending_wb1_fye2025.xlsx", "FYE2025")
    fs22 = FamilySpending(RAW / "family_spending_wb1_fye2022.xlsx", "FYE2022")
    inc22, m25, mid_dev = build_fye2022_income(fs22, fs25)
    cpi = read_cpi()

    e1 = episode_e1(fs22, inc22)
    e2 = episode_e2(fs22, inc22)
    e3 = episode_e3()
    e4 = episode_e4(fs25)
    e5 = episode_e5()
    m2 = module_two(cpi)

    res = {
        "meta": {
            "run_date": "2026-08-19",
            "spend_source": "ONS Family Spending workbook 1, sheet 3.1E "
                            "(equivalised disposable income deciles)",
            "fye2022_income_construction_max_midpoint_dev_pct":
                round(100 * mid_dev, 2),
            "fye2022_constructed_decile_income_gbp_wk":
                [round(x, 1) for x in inc22],
        },
        "episodes": {
            "E1_tca_food": e1, "E2_energy": e2, "E3_us_tariffs": e3,
            "E4_india_ceta": e4, "E5_cptpp_benchmark": e5,
        },
        "module2_uc_uprating": m2,
    }
    res["module3_comparability"] = module_three(e1, e2, e3, e4, e5)

    # --- Derived variants, computed INTO the artifact (not only into the
    # LaTeX macros), so results.json contains every number the paper
    # reports and the data-availability statement is honest.
    res["derived"] = derived_variants(res)
    # The comparability table headlines the TCA window-average path, so the
    # stored row must be the window-average one, not the superseded
    # end-state row it replaced.
    share = res["episodes"]["E1_tca_food"]["cross_check"][
        "implied_average_effect_share_of_end_state"]
    for row in res["module3_comparability"]:
        if row.get("episode", "").startswith("E1"):
            for k in ("bottom_decile_pct_of_spend", "top_decile_pct_of_spend"):
                if isinstance(row.get(k), (int, float)):
                    row[k + "_end_state"] = row[k]
                    row[k] = round(row[k] * share, 3)
            row["gross_shock_end_state"] = row["gross_shock"]
            row["gross_shock"] = res["derived"]["tca_window_average"][
                "aggregate_gbp_bn_per_year"]
            row["gross_shock_label"] = "GBP bn/yr, window-average path"
            row["basis"] = "window-average path (end-state retained as *_end_state)"

    (OUT / "results.json").write_text(json.dumps(res, indent=1))
    macros = latex_macros(res)
    (OUT / "generated_multishock.tex").write_text(
        "% Auto-generated by incidence.py -- do not edit by hand.\n"
        + "\n".join(macros) + "\n")
    print(f"Wrote out/results.json and out/generated_multishock.tex "
          f"({len(macros)} macros).")

    # Console headline echo
    c = e1["central"]
    print(f"\nE1 TCA food: end-state GBP {c['all_households_end_state_gbp_per_year']:.0f}/hh/yr "
          f"(agg GBP {c['aggregate_end_state_gbp_bn_per_year']}bn/yr); cumulative "
          f"linear GBP {c['all_households_cumulative_gbp_linear_phasein']:.0f} vs published GBP 250")
    print(f"E2 energy: gross GBP {e2['gross']['aggregate_gbp_bn_per_year']}bn/yr, "
          f"net GBP {e2['net_of_epg']['aggregate_gbp_bn_per_year']}bn/yr, EPG cushion "
          f"{100 * e2['epg_cushion']['share_of_gross_shock']:.1f}% "
          f"(GBP {e2['epg_cushion']['aggregate_gbp_bn_per_year']}bn/yr; "
          f"6-mo GBP {e2['epg_cushion']['aggregate_gbp_bn_six_months']}bn vs OBR GBP 27bn)")
    print(f"E4 CETA: bottom {e4['progressivity']['bottom_decile_gain_pct_of_spend']}% "
          f"vs top {e4['progressivity']['top_decile_gain_pct_of_spend']}% of spend -> "
          f"{e4['progressivity']['verdict']}")
    print(f"E5 CPTPP benchmark: GBP {e5['naive_mean_gbp_per_household_per_year']}/hh/yr")
    print(f"M2 UC lag: GBP {m2['uprating_lag_cost_gbp_fy_2022_23']:.0f} "
          f"({m2['uprating_lag_cost_pct_of_allowance']}% of allowance); "
          f"flat alt GBP {m2['flat_alternative_gbp']:.0f}; uplift GBP 1,040")
    print("\nComparability table:")
    for r in res["module3_comparability"]:
        print(f"  {r['episode']:<28} {r['sign']:<5} bot {r['bottom_decile_pct_of_spend']} "
              f"top {r['top_decile_pct_of_spend']} -> {r['regressivity_verdict']}")


if __name__ == "__main__":
    main()
