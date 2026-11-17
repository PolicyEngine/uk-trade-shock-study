"""LCFS validation of the within-decile spread of energy exposure.

The household stage reports that about 92.5 per cent of the variance in
households' energy burden (as a share of total spending) lies within
equivalised-income deciles.  That figure is computed on Microcosm UK's
IMPUTED consumption.  This script recomputes the same statistics directly
on the Living Costs and Food Survey household files, where energy and total
spending are observed for the same household.

Licensed data: the LCFS is UK Data Service end-user-licence microdata.  The
files are read from local paths outside the repository; ONLY aggregates
(variance shares, decile means, quantiles, counts) are written.  No
record-level value is printed or saved.

Definitions, matched to the household stage as closely as the LCFS allows:

* energy = electricity + gas, built from the labelled LCFS parts exactly as
  the Microcosm UK build derives its donors' electricity and gas (LCF User
  Guide Volume F; microcosm.build.uk_runtime.lcfs_consumption): each fuel's
  account payment less rebate (b489, b490), prepayment-meter payments
  (b1751, b1701), the fuel components of combined bills and key/meter
  payments (b227/b226, b2332/b2331), the diary second-dwelling and
  slot-meter lines (c45112/c45114, c45212/c45214), and whatever a combined
  payment (b231 less its rebate b174; b233) leaves unsplit.  Negative
  amounts (refunds) are read as zero, as the build does.  Oil, bottled gas,
  solid fuel and heat are excluded.
* total spending = the twelve COICOP divisions p601..p612, the sum
  PolicyEngine UK's ``consumption`` variable adds.
* deciles: household-weighted deciles of the LCFS's own equivalised
  disposable income on the modified-OECD scale (eqincdop: normal weekly
  disposable income p389p over a scale of 1.0 for a single adult and 1.5
  for a couple, i.e. the HBAI before-housing-costs scale up to its
  normalisation), the closest LCFS counterpart of the stage's equivalised
  HBAI before-housing-costs net income.  (eqincdmp is the McClements
  version and is not used.)
* burden: the stage's construction -- energy rebased to the FY2021-22 cap
  mean, total spending rebased to the ONS FYE2022 mean, times the realised
  rise -- so decile means and p10/p90 are level-comparable.  The variance
  split is invariant to those scalings.
* weights: the annual household weight (weighta; cal_weight_annual from
  2024-25).

Run:  python lcfs_validation.py --lcfs-2024 PATH/dvhh_uk_anon_v3_2024.tab \
                                [--lcfs-dir DIR_WITH_EARLIER_DVHH_DTA]
Emits results/lcfs_validation.json and results/generated_lcfs.tex (copied
to paper/).  If no LCFS file is found the script exits without touching
the committed outputs.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from second_stage_energy import (P, PAPER_DIR, RESULTS_DIR, variance_split,
                                 weighted, wmean, wquantile)

OLDER = {"2021-22": "dvhh_lcfs__ukanon_202122.dta",
         "2022-23": "dvhh_ukanon_202223.dta"}

ELECTRICITY = ("b489", "b1751", "b227", "b2332", "c45112", "c45114")
GAS = ("b490", "b1701", "b226", "b2331", "c45212", "c45214")
COMBINED = (("b231", "b174", ("b226", "b227")), ("b233", None, ("b2331", "b2332")))
OTHER_FUELS = ("b017", "b018", "c45312", "c45222", "c45411", "c45412", "c45511")
COICOP = tuple(f"p6{i:02d}" for i in range(1, 13))
WORDS = ("One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight",
         "Nine", "Ten")
MIN_CELL = 30   # no statistic is reported for a cell with fewer households


def read(path: Path) -> pd.DataFrame:
    if path.suffix == ".dta":
        df = pd.read_stata(path, convert_categoricals=False)
    else:
        df = pd.read_csv(path, sep="\t", low_memory=False)
    df.columns = [c.lower() for c in df.columns]
    return df


def num(df, col):
    return pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(float)


def analyse(df: pd.DataFrame) -> dict:
    wcol = "cal_weight_annual" if "cal_weight_annual" in df.columns else "weighta"
    missing = [c for c in (*ELECTRICITY, *GAS, *OTHER_FUELS, *COICOP, "b231",
                           "b174", "b233", "p537", "eqincdop", wcol)
               if c not in df.columns]
    if missing:
        raise KeyError(f"LCFS file lacks {missing}")
    w = num(df, wcol)
    part = lambda c: np.clip(num(df, c), 0.0, None)          # noqa: E731
    energy = sum(part(c) for c in (*ELECTRICITY, *GAS))
    for total, rebate, parts in COMBINED:
        paid = num(df, total) - (num(df, rebate) if rebate else 0.0)
        split = sum(part(c) for c in parts)
        energy = energy + np.clip(np.clip(paid, 0.0, None) - split, 0.0, None)
    other = sum(part(c) for c in OTHER_FUELS)
    p537 = num(df, "p537")
    spend = sum(num(df, c) for c in COICOP)
    income = num(df, "eqincdop")

    keep = (w > 0) & (spend > 0)
    dropped = int((~keep).sum())
    w, energy, other, p537, spend, income = (
        x[keep] for x in (w, energy, other, p537, spend, income))

    # Deciles: identical construction to second_stage_energy.py.
    order = np.argsort(income, kind="stable")
    cw = np.cumsum(w[order])
    dec = np.zeros(len(income), dtype=int)
    dec[order] = np.minimum((cw / cw[-1] * 10).astype(int), 9)

    # The stage's burden construction, so levels are comparable.
    base_cap = (P["cap_apr_2021"] + P["cap_oct_2021"]) / 2.0
    realised_rise = (P["cap_apr_2022"] + P["epg_level"]) / 2.0 / base_cap - 1.0
    e = energy * (base_cap / wmean(energy, w))
    s = spend * (P["ons_total_spend_yr"] / wmean(spend, w))
    dE = e * realised_rise
    share = dE / s

    b_un, w_un = variance_split(share, w, dec)
    lo, hi = wquantile(share, w, 0.01), wquantile(share, w, 0.99)
    t = (share >= lo) & (share <= hi)
    b_tr, w_tr = variance_split(share[t], w[t], dec[t])
    pos = share > 0
    b_lg, w_lg = variance_split(np.log(share[pos]), w[pos], dec[pos])
    # Excluding zero-energy reports (billing-cycle and rent-inclusive cases).
    b_pz, w_pz = variance_split(share[pos], w[pos], dec[pos])

    rows = []
    for d in range(10):
        m = dec == d
        if int(m.sum()) < MIN_CELL:
            raise ValueError("decile cell below the disclosure threshold")
        p10, p50, p90 = (wquantile(share[m], w[m], q) for q in (0.1, 0.5, 0.9))
        rows.append({
            "decile": d + 1,
            "burden_pct_of_spending": round(
                100 * weighted(dE[m], w[m]) / weighted(s[m], w[m]), 2),
            "raw_energy_share_of_spending_pct": round(
                100 * weighted(energy[m], w[m]) / weighted(spend[m], w[m]), 2),
            "p10_burden_pct_spending": round(100 * p10, 2),
            "p50_burden_pct_spending": round(100 * p50, 2),
            "p90_burden_pct_spending": round(100 * p90, 2),
        })
    return {
        "households": int(len(w)),
        "dropped_zero_weight_or_spend": dropped,
        "weighted_households_m": round(float(w.sum()) / 1e3, 2),
        "weight_column": wcol,
        "energy_mean_gbp_week": round(wmean(energy, w), 2),
        "total_spend_mean_gbp_week": round(wmean(spend, w), 2),
        "raw_energy_share_of_spending_pct": round(
            100 * weighted(energy, w) / weighted(spend, w), 2),
        "zero_energy_share_pct": round(100 * weighted((~pos).astype(float), w)
                                       / float(w.sum()), 1),
        # How much of the published fuel, light and power total (p537) the
        # labelled electricity, gas and other-fuel parts reach.
        "labelled_parts_over_p537": round(
            weighted(energy + other, w) / weighted(p537, w), 3),
        "electricity_gas_over_p537": round(
            weighted(energy, w) / weighted(p537, w), 3),
        "variance": {
            "untrimmed": {"between_pct": b_un, "within_pct": w_un},
            "trimmed_1_99": {"between_pct": b_tr, "within_pct": w_tr},
            "log_positive": {"between_pct": b_lg, "within_pct": w_lg},
            "positive_energy_only": {"between_pct": b_pz, "within_pct": w_pz},
        },
        "by_decile": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lcfs-2024", default=None,
                    help="LCFS 2024-25 household file (dvhh, tab-delimited)")
    ap.add_argument("--lcfs-dir", default=None,
                    help="folder holding the 2021-22 and 2022-23 dvhh .dta files")
    args = ap.parse_args()
    if not args.lcfs_2024:
        print("no --lcfs-2024 given; committed outputs left unchanged")
        return

    files = {"2024-25": Path(args.lcfs_2024)}
    if args.lcfs_dir:
        files.update({k: Path(args.lcfs_dir) / v for k, v in OLDER.items()})
    years, problems = {}, {}
    for label, path in files.items():
        if not path.exists():
            problems[label] = "file not found"
            continue
        try:
            years[label] = analyse(read(path))
        except (KeyError, ValueError) as exc:
            problems[label] = str(exc)
    if "2024-25" not in years:
        print("LCFS 2024-25 not available; committed outputs left unchanged:",
              problems)
        return

    ss = json.loads((RESULTS_DIR / "second_stage_energy.json").read_text())
    mc_var = ss["variance_decomposition"]
    res = {
        "note": "aggregates only; computed on licensed LCFS household files "
                "read locally, none of which is stored in this repository",
        "definitions": __doc__.split("Definitions,")[1].split("Run:")[0].strip(),
        "lcfs": years,
        "not_computed": problems,
        "microcosm": {
            "variance": {
                "untrimmed": mc_var["expenditure_share"],
                "trimmed_1_99": mc_var["expenditure_share_trimmed_1_99"],
                "log": mc_var["expenditure_share_log"]},
            "by_decile": [
                {"decile": a["decile"],
                 "burden_pct_of_spending": a["burden_pct_of_spending"],
                 "p10_burden_pct_spending": b["p10_burden_pct_spending"],
                 "p90_burden_pct_spending": b["p90_burden_pct_spending"]}
                for a, b in zip(ss["by_decile"], ss["within_decile_dispersion"])],
        },
    }
    (RESULTS_DIR / "lcfs_validation.json").write_text(json.dumps(res, indent=1))

    m = ["% generated by lcfs_validation.py -- do not edit by hand."]

    def add(k, v):
        m.append(f"\\newcommand{{\\SsLcfs{k}}}{{{v}}}")

    tags = {"2024-25": "", "2021-22": "TwentyOne", "2022-23": "TwentyTwo"}
    for label, r in years.items():
        tg = tags[label]
        v = r["variance"]
        add(f"{tg}Households", f"{r['households']:,}")
        add(f"{tg}WithinPct", f"{v['trimmed_1_99']['within_pct']:.1f}")
        add(f"{tg}BetweenPct", f"{v['trimmed_1_99']['between_pct']:.1f}")
        add(f"{tg}WithinPctUntrimmed", f"{v['untrimmed']['within_pct']:.1f}")
        add(f"{tg}BetweenPctUntrimmed", f"{v['untrimmed']['between_pct']:.1f}")
        add(f"{tg}WithinPctLog", f"{v['log_positive']['within_pct']:.1f}")
        add(f"{tg}WithinPctPositive", f"{v['positive_energy_only']['within_pct']:.1f}")
        add(f"{tg}ZeroEnergyPct", f"{r['zero_energy_share_pct']:.1f}")
        add(f"{tg}EnergySharePct", f"{r['raw_energy_share_of_spending_pct']:.1f}")
        d = r["by_decile"]
        add(f"{tg}PTenDecOne", f"{d[0]['p10_burden_pct_spending']:.2f}")
        add(f"{tg}PNinetyDecOne", f"{d[0]['p90_burden_pct_spending']:.2f}")
        add(f"{tg}BurdenDecOne", f"{d[0]['burden_pct_of_spending']:.2f}")
        add(f"{tg}BurdenDecTen", f"{d[9]['burden_pct_of_spending']:.2f}")
        if label == "2024-25":
            for row, wd in zip(d, WORDS):
                add(f"BurdenPctSpendDec{wd}", f"{row['burden_pct_of_spending']:.2f}")
    path = RESULTS_DIR / "generated_lcfs.tex"
    path.write_text("\n".join(m) + "\n")
    shutil.copy(path, PAPER_DIR / "generated_lcfs.tex")

    for label, r in years.items():
        print(label, json.dumps({k: v for k, v in r.items() if k != "by_decile"}))
        print("  decile burden %:", [x["burden_pct_of_spending"] for x in r["by_decile"]])
        print("  D1 p10/p90:", r["by_decile"][0]["p10_burden_pct_spending"],
              r["by_decile"][0]["p90_burden_pct_spending"])
    print("microcosm:", json.dumps(res["microcosm"]["variance"]))
    print("  decile burden %:", [x["burden_pct_of_spending"]
                                 for x in res["microcosm"]["by_decile"]])
    print("not computed:", problems)


if __name__ == "__main__":
    main()
