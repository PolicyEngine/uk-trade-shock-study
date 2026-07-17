"""Constituency-level geographic impact of the tariff shock (650 Westminster seats).

Replicates uk-ai-study's "route B" constituency pipeline. The PolicyEngine
constituency weight matrix (data/parliamentary_constituency_weights.h5, key
"2025", 650 x 53,508 grossing weights; copied from the policyengine_uk_data
storage the sister study used, originally distributed via PolicyEngine's
Hugging Face data repo) indexes the households of the ENHANCED FRS 2023-24
dataset (53,508 households), not the plain FRS 2024-25 the headline runs use.
The enhanced dataset carries no sic_industry_division column, so SIC 2007
division is IMPUTED for enhanced-FRS employees by drawing from the plain-FRS
2024-25 weighted SIC distribution within (age band x gender x region x
employee earnings decile) cells, with the same documented fallback ladder as
the sister study's SOC imputation. Match rates land in
results/geo/imputation_notes.json.

Scenarios: full_tariff_displacement and epd_displacement, seed 0, period 2025
(the weights' calibration year; the headline runs use 2026 on a different
dataset, so levels are indicative of relative, not absolute, local impacts).

Outputs (results/geo/):
  constituency_impacts.csv   code, name, region, per-scenario metrics
  imputation_notes.json      match rates + method + top-10 seats
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

from uk_trade_shock_study.exposure import simulation_sic_division  # noqa: E402
from uk_trade_shock_study.shocks import (  # noqa: E402
    PRESETS,
    apply_shocks,
    build_shocked_simulation,
)

PLAIN_FRS = ROOT / "data" / "frs_2024_25.h5"
ENHANCED_FRS = ROOT / "data" / "enhanced_frs_2023_24.h5"
WEIGHTS_H5 = ROOT / "data" / "parliamentary_constituency_weights.h5"
CONSTITUENCIES = ROOT / "data" / "constituencies_2024.csv"
OUT = ROOT / "results" / "geo"
OUT.mkdir(parents=True, exist_ok=True)

PERIOD = 2025  # the weights' calibration year (h5 key "2025")
SEED = 0
SCENARIOS = ("full_tariff_displacement", "epd_displacement")
AGE_BANDS = ((16, 24), (25, 34), (35, 44), (45, 54), (55, 64), (65, 200))


def age_band(age: np.ndarray) -> np.ndarray:
    out = np.full(len(age), -1)
    for i, (lo, hi) in enumerate(AGE_BANDS):
        out[(age >= lo) & (age <= hi)] = i
    return out


def person_frame(sim, period):
    df = pd.DataFrame(
        {
            v: sim.calculate(v, period=period, map_to="person").values
            for v in ("person_id", "age", "gender", "employment_income")
        }
    )
    df["weight"] = sim.calculate("person_weight", period=period, map_to="person").values
    df["region"] = sim.calculate("region", period=period, map_to="person").values
    return df


def weighted_decile_edges(values, weights, n=10):
    order = np.argsort(values)
    cw = np.cumsum(weights[order])
    cw = cw / cw[-1]
    return np.interp(np.arange(1, n) / n, cw, values[order])


def main():
    from policyengine_uk import Microsimulation
    from policyengine_uk.data import UKSingleYearDataset

    notes = {
        "method": "route B: SIC-division imputation on enhanced FRS 2023-24",
        "period": PERIOD,
        "seed": SEED,
        "scenarios": list(SCENARIOS),
    }

    # ---- Step 1: plain-FRS donor SIC distribution -------------------------
    plain = Microsimulation(dataset=UKSingleYearDataset(file_path=str(PLAIN_FRS)))
    donor = person_frame(plain, PERIOD)
    donor["sic"] = simulation_sic_division(plain, PERIOD)
    demp = donor[donor["employment_income"] > 0].copy()
    notes["plain_frs_employees"] = int(len(demp))
    notes["plain_frs_employee_sic_observed_share"] = float(
        np.average(np.isfinite(demp["sic"]), weights=demp["weight"])
    )
    demp = demp[np.isfinite(demp["sic"])]
    edges = weighted_decile_edges(
        demp["employment_income"].to_numpy(float), demp["weight"].to_numpy(float)
    )
    demp["band"] = age_band(demp["age"].to_numpy())
    demp["dec"] = np.digitize(demp["employment_income"], edges)

    def dist(frame):
        g = frame.groupby("sic")["weight"].sum()
        return g.index.to_numpy(float), np.cumsum((g / g.sum()).to_numpy(float))

    full = {k: dist(g) for k, g in demp.groupby(["band", "gender", "region", "dec"])}
    fb1 = {k: dist(g) for k, g in demp.groupby(["band", "gender", "dec"])}
    fb2 = {k: dist(g) for k, g in demp.groupby(["band", "gender"])}
    marginal = dist(demp)
    del plain, donor

    # ---- Step 2: enhanced FRS + imputation ---------------------------------
    dataset = UKSingleYearDataset(file_path=str(ENHANCED_FRS))
    baseline = Microsimulation(dataset=dataset)
    persons = person_frame(baseline, PERIOD)
    emp = persons["employment_income"].to_numpy(float) > 0
    band = age_band(persons["age"].to_numpy())
    dec = np.digitize(persons["employment_income"], edges)
    gender = persons["gender"].to_numpy()
    region = persons["region"].to_numpy()

    rng = np.random.default_rng(SEED)
    sic = np.full(len(persons), np.nan)
    tier_counts = {"full_cell": 0.0, "age_gender_decile": 0.0, "age_gender": 0.0, "marginal": 0.0}
    w = persons["weight"].to_numpy(float)
    for i in np.flatnonzero(emp):
        key4 = (band[i], gender[i], region[i], dec[i])
        key3 = (band[i], gender[i], dec[i])
        key2 = (band[i], gender[i])
        if key4 in full:
            codes, p = full[key4]; tier_counts["full_cell"] += w[i]
        elif key3 in fb1:
            codes, p = fb1[key3]; tier_counts["age_gender_decile"] += w[i]
        elif key2 in fb2:
            codes, p = fb2[key2]; tier_counts["age_gender"] += w[i]
        else:
            codes, p = marginal; tier_counts["marginal"] += w[i]
        sic[i] = codes[min(np.searchsorted(p, rng.random()), len(codes) - 1)]
    tot = sum(tier_counts.values())
    notes["imputation_tier_shares_weighted"] = {k: v / tot for k, v in tier_counts.items()}
    notes["enhanced_frs_employees"] = int(emp.sum())
    persons["sic_division"] = sic

    # ---- Step 3: household-level baseline vectors --------------------------
    hh_id = baseline.calculate("household_id", period=PERIOD, map_to="household").values
    hh_people = baseline.calculate(
        "household_count_people", period=PERIOD, map_to="household"
    ).values
    hh_income_base = baseline.calculate(
        "hbai_household_net_income", period=PERIOD, map_to="household"
    ).values
    person_hh = baseline.calculate("household_id", period=PERIOD, map_to="person").values
    pos = pd.Series(np.arange(len(hh_id)), index=hh_id)
    pidx = pos.loc[person_hh].to_numpy()
    hh_workers = np.zeros(len(hh_id))
    np.add.at(hh_workers, pidx, emp.astype(float))

    # ---- Step 4: constituency weights --------------------------------------
    with h5py.File(WEIGHTS_H5) as f:
        W = f["2025"][:]  # 650 x n_households, grossing weights
    assert W.shape[1] == len(hh_id), (W.shape, len(hh_id))
    const = pd.read_csv(CONSTITUENCIES)
    assert len(const) == W.shape[0]

    people = W @ hh_people
    workers = W @ hh_workers
    df = const[["code", "name", "region"]].copy()

    # ---- Step 5: scenarios -------------------------------------------------
    for name in SCENARIOS:
        scenario = PRESETS[name]
        shocked_table = apply_shocks(persons, scenario, seed=SEED)
        shocked = build_shocked_simulation(dataset, baseline, shocked_table, PERIOD)
        displaced = shocked_table["displaced"].to_numpy()

        hh_income_shock = shocked.calculate(
            "hbai_household_net_income", period=PERIOD, map_to="household"
        ).values
        hh_delta = hh_income_shock - hh_income_base
        hh_displaced = np.zeros(len(hh_id))
        np.add.at(hh_displaced, pidx, displaced.astype(float))
        del shocked

        # mean disposable-income change per person: person-weighted mean of the
        # household income change (household value broadcast to members, as in
        # the runner's regional profile) = W @ (n_h * delta_h) / W @ n_h.
        col = "full" if name.startswith("full") else "epd"
        df[f"income_change_pp_{col}"] = (W @ (hh_people * hh_delta)) / people
        df[f"displaced_per_1000_workers_{col}"] = 1000 * (W @ hh_displaced) / workers
        notes[f"{name}_national_income_change_pp"] = float(
            (W @ (hh_people * hh_delta)).sum() / people.sum()
        )
        notes[f"{name}_displaced_weighted"] = float(
            (persons["weight"].to_numpy() * displaced).sum()
        )

    df["people"] = people
    df["workers"] = workers
    df.to_csv(OUT / "constituency_impacts.csv", index=False)

    top_full = df.nsmallest(10, "income_change_pp_full")[
        ["code", "name", "region", "income_change_pp_full", "income_change_pp_epd"]
    ]
    top_epd = df.nsmallest(10, "income_change_pp_epd")[
        ["code", "name", "region", "income_change_pp_full", "income_change_pp_epd"]
    ]
    notes["top10_hardest_hit_full"] = top_full.to_dict("records")
    notes["top10_hardest_hit_epd"] = top_epd.to_dict("records")
    (OUT / "imputation_notes.json").write_text(json.dumps(notes, indent=2))
    print(json.dumps(notes, indent=2))


if __name__ == "__main__":
    main()
