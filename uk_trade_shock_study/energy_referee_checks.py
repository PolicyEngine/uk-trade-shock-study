"""Three referee checks on the 2022-23 energy episode (referee round,
November 2026).  Called from grid_energy.py on the simulation that script
already holds, so the checks cost no extra run; everything here is
arithmetic on the same household arrays.

1. The grid below the paper's rebase, with the MAXIMUM decile cushioning
   rate in every cell (the table used to print only D1 and D10, and the
   highest rate is at neither), and the rebase at which a decile's rate
   first reaches 100 per cent, in closed form.
2. The GBP 150 Council Tax Rebate (Energy Bills Rebate package, April
   2022), taken from PolicyEngine UK's own ``ebr_council_tax_rebate`` at
   period 2022.  It is IN the central stack, classed energy-specific
   (author decision, November 2026); this module checks its amount against
   official figures and reports the same numbers WITHOUT it (NoCtr), so
   the paper can say what the rebate adds.
3. The model's Cost of Living Payments against DWP management
   information.

Results go to results/energy_referee_checks.json (its own file:
grid_energy.json, seasonal_energy.json and second_stage_energy.json are
hash-pinned quoted results and are left byte-identical).  Official
figures are DATA with the source URL recorded beside each; nothing is
quoted that was not read off the cited page.
"""

import numpy as np

from second_stage_energy import COL_YEAR, P, weighted

WORDS = ("One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight",
         "Nine", "Ten")

#: Table rebases: one BELOW the paper's computed rebase, the paper's (None),
#: an intermediate value and no rebase.
TABLE_REBASES = [0.6, None, 0.8, 1.0]
LOW_REBASE = 0.6

# ---------------------------------------------------------------------------
# Official figures (DATA).  Each was read off the cited page; the notes say
# what the figure is and is not.
# ---------------------------------------------------------------------------
COL_OUTTURN = {
    "means_tested": {
        # Two instalments (GBP 326 + GBP 324); each spend figure is rounded
        # to the nearest GBP 100m on the page, so the sum is GBP 5.4bn
        # +/- 0.1bn.
        "spend_gbp_bn": 5.4,
        "first_payment": {"payments_m": 8.4, "spend_gbp_bn": 2.7,
                          "dwp_payments_m": 7.2, "hmrc_payments_m": 1.2,
                          "as_of": "2022-09-08"},
        "second_payment": {"payments_m": 8.3, "spend_gbp_bn": 2.7,
                           "dwp_payments_m": 7.3, "hmrc_payments_m": 1.0,
                           "as_of": "2022-12-12"},
        "payments_m": 8.4,
        "coverage": "not stated on the page (GB or UK)",
        "source": "DWP, Cost of Living Payment 2022 to 2023 management "
                  "information (last updated 12 December 2022)",
        "url": "https://www.gov.uk/government/publications/cost-of-living-payment-management-information/cost-of-living-payment-management-information",
    },
    "disability": {
        "spend_gbp_bn": 0.9, "payments_m": 6.0, "as_of": "2022-09-30",
        "coverage": "not stated on the page (GB or UK)",
        "source": "DWP, Cost of Living Payment 2022 to 2023 management "
                  "information (last updated 12 December 2022)",
        "url": "https://www.gov.uk/government/publications/cost-of-living-payment-management-information/cost-of-living-payment-management-information",
    },
    "pensioner": {
        # The GBP 300 top-up only; the page's GBP 4.6bn also includes the
        # Winter Fuel Payment itself.  8.7m is the page's ESTIMATE of
        # households, derived from spend.
        "spend_gbp_bn": 2.6, "households_m_estimate": 8.7,
        "as_of": "2022-12-20", "coverage": "UK",
        "source": "DWP, Pensioner Cost of Living Payment management "
                  "information (20 December 2022)",
        "url": "https://www.gov.uk/government/publications/pensioner-cost-of-living-payment-management-information/pensioner-cost-of-living-payment-management-information",
    },
    "caveat": "management information: snapshots at the dates shown, gross "
              "payments, rounded to the nearest GBP 100m and 100,000 "
              "payments; no later final outturn by component was found",
}

CTR_OUTTURN = {
    "england_core": {
        # Core scheme only (bands A-D); the discretionary fund is separate
        # and not in the model.  A PROGRESS figure to the end of September
        # 2022 (over 99 per cent of eligible households paid; claims stayed
        # open to 30 November 2022): no final total was found.
        "status": "progress figure to end-September 2022 (over 99 per cent "
                  "of eligible households paid), not a final outturn; no "
                  "final total was found",
        "spend_gbp_bn": 2.87, "households_m": 19.0,
        "households_note": "'more than 19 million households in England'",
        "as_of": "2022-09-30", "coverage": "England",
        "source": "DLUHC press release, Over 19 million households have "
                  "received their council tax rebate (31 October 2022)",
        "url": "https://www.gov.uk/government/news/over-19-million-households-have-received-their-council-tax-rebate",
    },
    "scotland": {
        "status": "announced reach, NOT an outturn; no spend total found",
        "households_m_announced": 1.85,
        "note": "bands A-D plus all Council Tax Reduction recipients; the "
                "model pays bands A-D only",
        "source": "Holyrood magazine, 10 February 2022, reporting the "
                  "Scottish Government announcement (around 1.85 million "
                  "households, 73 per cent)",
        "url": "https://www.holyrood.com/news/view,cost-of-living-crisis-kate-forbes-hands-out-150-to-three-quarters-of-scots-households",
        "rules_url": "https://www.gov.scot/publications/local-government-finance-circular-2-2022-guidance-to-local-authorities-april-2022-cost-of-living-award/",
    },
    "wales": {
        "status": "households paid only ('over one million' by the 30 "
                  "September 2022 close); no spend total found on the page",
        "households_m_paid_at_least": 1.0,
        "note": "bands A-D plus all Council Tax Reduction Scheme "
                "recipients; the model pays bands A-D only",
        "source": "Welsh Government, Cabinet Sub-Committee on Cost of "
                  "Living meeting, 10 October 2022, para 3.13",
        "url": "https://gov.wales/node/51547",
        "rules_url": "https://media.service.gov.wales/news/gbp-380m-to-help-tackle-the-cost-of-living-crisis-in-wales",
    },
    "northern_ireland": "no Council Tax Rebate (domestic rates, not "
                        "council tax); the model pays none",
}


# ---------------------------------------------------------------------------
# Pure arithmetic (no simulation), so the tests can exercise it.
# ---------------------------------------------------------------------------
def decile_rates(epg_rate, cash_by_decile, shock_unit_by_decile, rebase):
    """Cushioning rate (per cent) of each decile at ``rebase``.

    The Guarantee returns the same share ``epg_rate`` of every household's
    counterfactual loss; the flat-cash instruments do not scale with the
    rebase, while the loss does (``shock_unit`` is the decile's weighted
    counterfactual loss at rebase 1).  Pass ``epg_rate=0`` for a stack
    without the Guarantee.
    """
    cash = np.asarray(cash_by_decile, dtype=float)
    shock = np.asarray(shock_unit_by_decile, dtype=float) * rebase
    return 100.0 * (epg_rate + cash / shock)


def rebase_at_hundred(epg_rate, cash_by_decile, shock_unit_by_decile):
    """Rebase at which each decile's rate equals 100 per cent, closed form.

    rate_d(r) = epg_rate + cash_d / (r * shock_unit_d) = 1 gives
    r_d = cash_d / ((1 - epg_rate) * shock_unit_d).  A decile's rate is at
    or above 100 per cent for every rebase at or below its r_d, so the
    largest r_d is where the first decile gets there as the rebase falls.
    Returns (r_by_decile, decile_number_of_the_largest, largest_r).
    """
    cash = np.asarray(cash_by_decile, dtype=float)
    shock = np.asarray(shock_unit_by_decile, dtype=float)
    r = cash / ((1.0 - epg_rate) * shock)
    d = int(np.argmax(r))
    return r, d + 1, float(r[d])


def max_decile(rates, tol=1e-9):
    """(maximum rate, decile number) or (rate, None) on a ten-way tie."""
    rates = np.asarray(rates, dtype=float)
    if np.ptp(rates) <= tol:
        return float(rates[0]), None
    d = int(np.argmax(rates))
    return float(rates[d]), d + 1


# ---------------------------------------------------------------------------
# The checks, on the simulation grid_energy.py already holds.
# ---------------------------------------------------------------------------
def referee_checks(sim, w, dec, energy_raw, paper_rebase, counter_rise,
                   realised_rise, ebss, ctr, col_parts):
    """Returns (results dict, macros dict, table cells).

    Stacks: "epg"; "epg_ebss" (+ GBP 400); "energy" (+ Council Tax Rebate:
    the energy-specific stack); "full" (+ Cost of Living Payments).  The
    "_noctr" stacks leave the rebate out.
    """
    col_mt, col_pen, col_dis = col_parts
    col = col_mt + col_pen + col_dis
    epg_rate = (counter_rise - realised_rise) / counter_rise
    shock_unit = energy_raw * counter_rise          # loss at rebase 1

    def by_dec(x):
        return np.array([weighted(x[dec == d], w[dec == d])
                         for d in range(10)])

    country = np.asarray(sim.calculate("country", COL_YEAR)).astype(str)

    def country_is(name):
        return np.array([name in c.upper() for c in country])

    nations = {"england": country_is("ENGLAND"),
               "scotland": country_is("SCOTLAND"),
               "wales": country_is("WALES"),
               "northern_ireland": country_is("NORTHERN")}

    stacks_cash = {
        "none": None,
        "epg": np.zeros_like(energy_raw),
        "epg_ebss": ebss,
        "energy": ebss + ctr,
        "full": ebss + ctr + col,
        "full_noctr": ebss + col,
    }
    S_dec = by_dec(shock_unit)
    S_all = float(S_dec.sum())
    cash_dec = {k: (np.zeros(10) if v is None else by_dec(v))
                for k, v in stacks_cash.items()}

    def cell(rebase, label, stack):
        e = 0.0 if stack == "none" else epg_rate
        rates = decile_rates(e, cash_dec[stack], S_dec, rebase)
        mx, mx_d = max_decile(rates)
        cash_all = float(cash_dec[stack].sum())
        return {
            "rebase": round(rebase, 4), "rebase_label": label, "stack": stack,
            "counter_agg_gbp_bn": round(S_all * rebase / 1e9, 1),
            "discretionary_gbp_bn": round(
                (e * S_all * rebase + cash_all) / 1e9, 1),
            "rate_pct": round(100 * (e + cash_all / (S_all * rebase)), 1),
            "cushion_pct_by_decile": [round(float(x), 1) for x in rates],
            "cushion_pct_d1": round(float(rates[0]), 1),
            "cushion_pct_d10": round(float(rates[9]), 1),
            "max_decile_pct": round(mx, 1),
            "max_decile": mx_d,           # None = identical in every decile
            "deciles_above_100": [i + 1 for i, x in enumerate(rates)
                                  if x > 100.0],
        }

    def label(rb):
        return "paper" if rb is None else f"{rb:.1f}"

    table_stacks = ["none", "epg", "epg_ebss", "energy", "full"]
    table_cells = [cell(paper_rebase if rb is None else rb, label(rb), s)
                   for rb in TABLE_REBASES for s in table_stacks]
    # The same grid without the Council Tax Rebate ("epg_ebss" is the
    # energy-specific stack without it).
    noctr_cells = [cell(paper_rebase if rb is None else rb, label(rb),
                        "full_noctr") for rb in TABLE_REBASES]

    def find(cells, lab, stack):
        return next(c for c in cells
                    if c["rebase_label"] == lab and c["stack"] == stack)

    # --- 1. Rebase at which a decile first reaches 100 per cent ------------
    thresholds = {}
    for stack in ("epg_ebss", "energy", "full", "full_noctr"):
        r, d, rmax = rebase_at_hundred(epg_rate, cash_dec[stack], S_dec)
        thresholds[stack] = {
            "rebase_at_100_by_decile": [round(float(x), 4) for x in r],
            "first_decile": d,
            "rebase_at_100": round(rmax, 4),
            "ratio_to_paper_rebase": round(rmax / paper_rebase, 3),
            "aggregate_rebase_at_100": round(
                float(cash_dec[stack].sum()) / ((1 - epg_rate) * S_all), 4),
            "deciles_at_or_above_100_at_paper_rebase":
                [i + 1 for i, x in enumerate(r) if x >= paper_rebase],
        }

    grid = {
        "note": "rate_d(rebase) = epg_rate + cash_d / (rebase * shock_unit_d); "
                "a decile is at or above 100 per cent for every rebase at or "
                "below its rebase_at_100.  Stacks: epg_ebss = Guarantee + "
                "GBP 400; energy = + Council Tax Rebate (energy-specific); "
                "full = + Cost of Living Payments; full_noctr = full "
                "without the rebate",
        "paper_rebase": round(paper_rebase, 4),
        "epg_rate": round(epg_rate, 6),
        "shock_unit_by_decile_gbp_bn": [round(float(x) / 1e9, 6)
                                        for x in S_dec],
        "cash_by_decile_gbp_bn": {
            k: [round(float(x) / 1e9, 6) for x in v]
            for k, v in cash_dec.items() if k != "none"},
        "table_cells": table_cells,
        "without_ctr_cells": noctr_cells,
        "thresholds": thresholds,
    }

    # --- 2. Council Tax Rebate: amounts against official figures -----------
    def bn(x, m=None):
        m = np.ones_like(w, dtype=bool) if m is None else m
        return weighted(x[m], w[m]) / 1e9

    def hh_m(x, m=None):
        m = np.ones_like(w, dtype=bool) if m is None else m
        return weighted((x[m] > 0).astype(float), w[m]) / 1e6

    eng_bn = bn(ctr, nations["england"])
    eng_out = CTR_OUTTURN["england_core"]["spend_gbp_bn"]
    c_energy = find(table_cells, "paper", "energy")
    c_full = find(table_cells, "paper", "full")
    n_energy = find(table_cells, "paper", "epg_ebss")
    n_full = find(noctr_cells, "paper", "full_noctr")
    ctr_res = {
        "source_variable": "ebr_council_tax_rebate at period 2022 "
                           "(PolicyEngine UK's own rule: GBP 150 per "
                           "household in council tax bands A-D in England, "
                           "Scotland and Wales; none in Northern Ireland)",
        "classification": "energy-specific (part of the February 2022 "
                          "Energy Bills Rebate package); in the central stack",
        "model_gbp_bn": round(bn(ctr), 3),
        "model_households_m": round(hh_m(ctr), 3),
        "model_by_nation": {
            k: {"gbp_bn": round(bn(ctr, m), 3),
                "households_m": round(hh_m(ctr, m), 3)}
            for k, m in nations.items()},
        "outturn": CTR_OUTTURN,
        "england_model_over_outturn_pct": round(100 * (eng_bn / eng_out - 1), 1),
        "energy_specific_rate_pct": {"without": n_energy["rate_pct"],
                                     "with": c_energy["rate_pct"]},
        "full_stack_rate_pct": {"without": n_full["rate_pct"],
                                "with": c_full["rate_pct"]},
        "full_stack_by_decile_pct": {
            "without": n_full["cushion_pct_by_decile"],
            "with": c_full["cushion_pct_by_decile"]},
        "energy_specific_by_decile_pct": {
            "without": n_energy["cushion_pct_by_decile"],
            "with": c_energy["cushion_pct_by_decile"]},
        "caveats": [
            "the model omits the Scottish and Welsh awards to Council Tax "
            "Reduction recipients in bands E and above, and the English "
            "discretionary fund",
            "paid from April 2022 on council tax band, not on energy use: "
            "like the GBP 400 it is flat cash, so its rate rises as the "
            "rebase falls",
            "household weights exceed the ONS household count "
            "(second_stage_energy.json, weight_excess_pct), which overstates "
            "every per-household flat payment by about that share",
        ],
    }

    # --- 3. Cost of Living Payments against management information ---------
    gb = ~nations["northern_ireland"]
    amounts = {"means_tested": P["col_means_tested"],
               "pensioner": P["col_pensioner"],
               "disability": P["col_disability"]}
    parts = {"means_tested": col_mt, "pensioner": col_pen,
             "disability": col_dis}
    units = {"means_tested": "benefit units", "pensioner": "households",
             "disability": "persons"}
    col_res = {"components": {}, "outturn": COL_OUTTURN}
    out_total = 0.0
    col_cal = np.zeros_like(col)
    for k, x in parts.items():
        o = COL_OUTTURN[k]["spend_gbp_bn"]
        out_total += o
        uk = bn(x)
        col_cal = col_cal + x * (o / uk)
        col_res["components"][k] = {
            "model_uk_gbp_bn": round(uk, 3),
            "model_gb_gbp_bn": round(bn(x, gb), 3),
            # every recipient unit gets the same statutory amount, so the
            # recipient count is the spend over the amount
            "model_uk_recipients_m": round(uk * 1e3 / amounts[k], 3),
            "model_gb_recipients_m": round(bn(x, gb) * 1e3 / amounts[k], 3),
            "recipient_unit": units[k],
            "amount_gbp": amounts[k],
            "outturn_gbp_bn": o,
            "model_uk_over_outturn_pct": round(100 * (uk / o - 1), 1),
            "model_gb_over_outturn_pct": round(100 * (bn(x, gb) / o - 1), 1),
        }
    model_total = bn(col)
    col_res["total"] = {
        "model_uk_gbp_bn": round(model_total, 3),
        "model_gb_gbp_bn": round(bn(col, gb), 3),
        "outturn_gbp_bn": round(out_total, 1),
        "model_uk_over_outturn_pct": round(100 * (model_total / out_total - 1), 1),
    }
    # Outturn-scaled variants of the central full stack (same construction
    # as the outturn-scaled Bills Support variant, SsCalFullRatePct): each
    # Cost of Living component scaled to its management-information spend.
    G = S_all * paper_rebase
    epg_bn = epg_rate * G
    ebss_cal = ebss * (P["ebss_outturn_gbp_bn"] * 1e9 / weighted(ebss, w))
    cal_dec = decile_rates(epg_rate, by_dec(ebss + ctr + col_cal), S_dec,
                           paper_rebase)
    col_res["outturn_scaled"] = {
        "full_rate_pct_col_scaled": round(
            100 * (epg_bn + weighted(ebss + ctr + col_cal, w)) / G, 1),
        "full_rate_pct_col_and_ebss_scaled": round(
            100 * (epg_bn + weighted(ebss_cal + ctr + col_cal, w)) / G, 1),
        "full_rate_pct_central": c_full["rate_pct"],
        "cushion_pct_by_decile_col_scaled": [round(float(x), 1)
                                             for x in cal_dec],
        "deciles_above_100_col_scaled": [i + 1 for i, x in enumerate(cal_dec)
                                         if x > 100.0],
        "note": "proportional scaling of each component to its "
                "management-information spend; the geographic coverage of "
                "the means-tested and disability figures is not stated on "
                "the page, so the UK model total is the one scaled.  The "
                "Council Tax Rebate is left at the model's amount",
    }

    res = {"meta": {"col_year": COL_YEAR, "sim_year": P["sim_year"],
                    "paper_rebase": round(paper_rebase, 4),
                    "low_rebase": LOW_REBASE,
                    "central_stack": "Guarantee + Bills Support (GBP 400) + "
                                     "Council Tax Rebate (GBP 150) + Cost of "
                                     "Living Payments"},
           "grid": grid, "council_tax_rebate": ctr_res,
           "cost_of_living_outturn": col_res}

    # --- Macros (letters only; all new names) -------------------------------
    # ...Ebss... = the energy-specific stack (WITH the rebate), matching
    # grid_energy.py's paper-facing names; ...NoCtr... = without the rebate.
    def dec_label(d):
        return "all" if d is None else str(d)

    def dec_list(ds):
        if not ds:
            return "none"
        ds = [str(d) for d in ds]
        return ds[0] if len(ds) == 1 else ", ".join(ds[:-1]) + " and " + ds[-1]

    def signed(x):
        """Whole per cent with a typeset minus sign."""
        return f"{x:.0f}".replace("-", "$-$")

    m = {}
    low_full = find(table_cells, "0.6", "full")
    low_energy = find(table_cells, "0.6", "energy")
    unity_full = find(table_cells, "1.0", "full")
    m["GridLowRebase"] = f"{LOW_REBASE:.3f}"
    m["GridLowCounterAggBn"] = f"{low_full['counter_agg_gbp_bn']:.1f}"
    m["GridLowFullRatePct"] = f"{low_full['rate_pct']:.1f}"
    m["GridLowEbssRatePct"] = f"{low_energy['rate_pct']:.1f}"
    m["GridLowFullDOne"] = f"{low_full['cushion_pct_d1']:.1f}"
    m["GridLowFullDTen"] = f"{low_full['cushion_pct_d10']:.1f}"
    m["GridLowEbssDOne"] = f"{low_energy['cushion_pct_d1']:.1f}"
    for tag, c in (("LowFull", low_full), ("LowEbss", low_energy),
                   ("PaperFull", c_full), ("PaperEbss", c_energy),
                   ("UnityFull", unity_full)):
        m[f"Grid{tag}MaxDecilePct"] = f"{c['max_decile_pct']:.1f}"
        m[f"Grid{tag}MaxDecile"] = dec_label(c["max_decile"])
    t = thresholds["full"]
    m["GridMaxDecileRebaseAtHundred"] = f"{t['rebase_at_100']:.3f}"
    m["GridMaxDecileAtHundredDecile"] = str(t["first_decile"])
    m["GridMaxDecileRebaseAtHundredRatio"] = f"{t['ratio_to_paper_rebase']:.2f}"
    m["GridAggRebaseAtHundred"] = f"{t['aggregate_rebase_at_100']:.3f}"
    te = thresholds["energy"]
    m["GridEbssMaxDecileRebaseAtHundred"] = f"{te['rebase_at_100']:.3f}"
    m["GridEbssMaxDecileAtHundredDecile"] = str(te["first_decile"])
    # Deciles whose full-stack rate exceeds 100 per cent at the paper's rebase.
    above = c_full["deciles_above_100"]
    m["DecilesAboveHundred"] = str(len(above))
    m["DecilesAboveHundredList"] = dec_list(above)
    m["GridUnityDecilesAboveHundred"] = str(len(unity_full["deciles_above_100"]))
    m["GridLowDecilesAboveHundred"] = str(len(low_full["deciles_above_100"]))
    m["GridLowDecilesAboveHundredList"] = dec_list(low_full["deciles_above_100"])

    # Without the Council Tax Rebate.
    m["GridPaperNoCtrEbssRatePct"] = f"{n_energy['rate_pct']:.1f}"
    m["GridPaperNoCtrEbssDOne"] = f"{n_energy['cushion_pct_d1']:.1f}"
    m["GridPaperNoCtrEbssDTen"] = f"{n_energy['cushion_pct_d10']:.1f}"
    m["GridPaperNoCtrFullRatePct"] = f"{n_full['rate_pct']:.1f}"
    m["GridPaperNoCtrFullDOne"] = f"{n_full['cushion_pct_d1']:.1f}"
    m["GridPaperNoCtrFullDTen"] = f"{n_full['cushion_pct_d10']:.1f}"
    m["GridPaperNoCtrFullMaxDecilePct"] = f"{n_full['max_decile_pct']:.1f}"
    m["GridPaperNoCtrFullMaxDecile"] = dec_label(n_full["max_decile"])
    m["GridUnityNoCtrEbssRatePct"] = (
        f"{find(table_cells, '1.0', 'epg_ebss')['rate_pct']:.1f}")
    m["GridUnityNoCtrFullRatePct"] = (
        f"{find(noctr_cells, '1.0', 'full_noctr')['rate_pct']:.1f}")
    tn = thresholds["full_noctr"]
    m["GridNoCtrMaxDecileRebaseAtHundred"] = f"{tn['rebase_at_100']:.3f}"
    m["GridNoCtrMaxDecileAtHundredDecile"] = str(tn["first_decile"])
    m["NoCtrDecilesAboveHundred"] = str(len(n_full["deciles_above_100"]))
    m["CtrEbssRateAddPp"] = f"{c_energy['rate_pct'] - n_energy['rate_pct']:.1f}"

    # Council Tax Rebate against official figures (SsCtrBn and
    # SsCtrHouseholdsM come from second_stage_energy.py).
    m["CtrEnglandBn"] = f"{eng_bn:.2f}"
    m["CtrEnglandHouseholdsM"] = (
        f"{ctr_res['model_by_nation']['england']['households_m']:.1f}")
    m["CtrOutturnEnglandBn"] = f"{eng_out:.2f}"
    m["CtrEnglandExcessPct"] = signed(ctr_res["england_model_over_outturn_pct"])
    m["CtrScotlandBn"] = f"{ctr_res['model_by_nation']['scotland']['gbp_bn']:.2f}"
    m["CtrWalesBn"] = f"{ctr_res['model_by_nation']['wales']['gbp_bn']:.2f}"

    for tag, k in (("Mt", "means_tested"), ("Pen", "pensioner"),
                   ("Dis", "disability")):
        c = col_res["components"][k]
        m[f"ColModel{tag}Bn"] = f"{c['model_uk_gbp_bn']:.1f}"
        m[f"ColModel{tag}RecipientsM"] = f"{c['model_uk_recipients_m']:.1f}"
        m[f"ColOutturn{tag}Bn"] = f"{c['outturn_gbp_bn']:.1f}"
        m[f"ColOutturn{tag}ExcessPct"] = signed(c["model_uk_over_outturn_pct"])
    m["ColOutturnMtPaymentsM"] = f"{COL_OUTTURN['means_tested']['payments_m']:.1f}"
    m["ColOutturnDisPaymentsM"] = f"{COL_OUTTURN['disability']['payments_m']:.1f}"
    m["ColOutturnPenHouseholdsM"] = (
        f"{COL_OUTTURN['pensioner']['households_m_estimate']:.1f}")
    m["ColOutturnTotalBn"] = f"{col_res['total']['outturn_gbp_bn']:.1f}"
    m["ColModelTotalBn"] = f"{col_res['total']['model_uk_gbp_bn']:.1f}"
    m["ColOutturnTotalExcessPct"] = signed(
        col_res["total"]["model_uk_over_outturn_pct"])
    m["ColOutturnCalFullRatePct"] = (
        f"{col_res['outturn_scaled']['full_rate_pct_col_scaled']:.1f}")
    m["ColOutturnCalBothFullRatePct"] = (
        f"{col_res['outturn_scaled']['full_rate_pct_col_and_ebss_scaled']:.1f}")
    cal = col_res["outturn_scaled"]["cushion_pct_by_decile_col_scaled"]
    m["ColOutturnCalFullDOne"] = f"{cal[0]:.1f}"
    m["ColOutturnCalFullMaxDecilePct"] = f"{max(cal):.1f}"
    m["ColOutturnCalFullMaxDecile"] = str(int(np.argmax(cal)) + 1)
    m["ColOutturnCalDecilesAboveHundred"] = str(
        len(col_res["outturn_scaled"]["deciles_above_100_col_scaled"]))
    return res, m, table_cells
