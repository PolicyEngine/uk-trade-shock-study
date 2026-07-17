"""Demographic incidence: who bears the shock by gender, age and household type.

For full_tariff displacement (20 Monte Carlo draws, seeds 0-19) and the
deterministic full_tariff wage cut, computes for each group:

- share of the displaced/affected (weighted);
- mean disposable-income change per affected worker (household HBAI net-income
  change attributed to affected workers within the household, proportional to
  each worker's gross earnings loss);
- cushioning rate = 1 - attributed net loss / gross earnings loss.

Groups: gender; age band; and benunit household type (single/couple x
with/without children, and couples split single- vs dual-earner at baseline)
— the household-type split visualises the partner-income taper gate from the
mechanism section: dual-earner couples should show less UC cushioning.

Writes results/demographics.json.
Usage: .venv/bin/python analysis/demographics.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from uk_trade_shock_study.runner import AGE_BANDS, _baseline_and_persons
from uk_trade_shock_study.shocks import PRESETS, apply_shocks, build_shocked_simulation

PERIOD = 2026
DATASET = Path("data/frs_2024_25.h5")
RESULTS = Path("results")
N_DRAWS = 20


def hh_arrays(sim):
    return {
        "id": sim.calculate("household_id", period=PERIOD, map_to="household").values.astype(int),
        "weight": sim.calculate("household_weight", period=PERIOD, map_to="household").values.astype(float),
        "hni": sim.calculate("hbai_household_net_income", period=PERIOD, map_to="household").values.astype(float),
    }


def build_person_context(baseline, persons):
    """Static person-level grouping columns from the baseline simulation."""
    p = persons.copy()
    p["gender"] = baseline.calculate("gender", period=PERIOD, map_to="person").values
    p["family_type"] = baseline.calculate("family_type", period=PERIOD, map_to="person").values
    p["benunit_id"] = baseline.calculate("benunit_id", period=PERIOD, map_to="person").values.astype(int)
    p["household_id"] = baseline.calculate("household_id", period=PERIOD, map_to="person").values.astype(int)
    earner = (p["employment_income"].to_numpy(float) > 0).astype(int)
    p["benunit_earners"] = pd.Series(earner).groupby(p["benunit_id"]).transform("sum")
    age = p["age"].to_numpy(float)
    band = np.full(len(p), "", dtype=object)
    for lo, hi in AGE_BANDS:
        band[(age >= lo) & (age <= hi)] = f"{lo}-{hi if hi < 200 else '+'}"
    p["age_band"] = band
    ft = p["family_type"].astype(str)
    couple = ft.str.startswith("COUPLE")
    hh_type = np.select(
        [
            ft == "SINGLE",
            ft == "LONE_PARENT",
            couple & (ft == "COUPLE_NO_CHILDREN") & (p["benunit_earners"] < 2),
            couple & (ft == "COUPLE_NO_CHILDREN") & (p["benunit_earners"] >= 2),
            couple & (ft == "COUPLE_WITH_CHILDREN") & (p["benunit_earners"] < 2),
            couple & (ft == "COUPLE_WITH_CHILDREN") & (p["benunit_earners"] >= 2),
        ],
        [
            "single_no_children",
            "single_with_children",
            "couple_no_children_single_earner",
            "couple_no_children_dual_earner",
            "couple_with_children_single_earner",
            "couple_with_children_dual_earner",
        ],
        default="other",
    )
    p["hh_type"] = hh_type
    p["couple_earners"] = np.where(
        couple, np.where(p["benunit_earners"] >= 2, "couple_dual_earner", "couple_single_earner"), "not_couple"
    )
    return p


def draw_incidence(ctx, base_hh, shock_hh, base_emp, shocked_emp, affected):
    """Attribute household net-income change to affected workers; group stats."""
    w = ctx["weight"].to_numpy(float)
    g = (base_emp - shocked_emp) * w  # weighted gross loss per person
    hh_row = pd.Series(np.arange(len(base_hh["id"])), index=base_hh["id"])
    p_row = ctx["household_id"].map(hh_row).to_numpy()
    hh_net = (base_hh["hni"] - shock_hh["hni"]) * base_hh["weight"]  # weighted £

    df = pd.DataFrame({"row": p_row, "g": g, "w": w, "affected": affected})
    gh = df[df.affected].groupby("row")["g"].sum()
    denom = gh.reindex(df["row"]).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        attributed = np.where(
            affected & (denom > 0), hh_net[p_row] * (g / denom), 0.0
        )
    total_net = float(hh_net.sum())
    coverage = float(attributed.sum() / total_net) if total_net else float("nan")

    def group_stats(labels):
        out = {}
        aw_total = float(w[affected].sum())
        for lab in sorted(set(labels[affected])):
            m = affected & (labels == lab)
            gross = float(g[m].sum())
            net = float(attributed[m].sum())
            out[str(lab)] = {
                "share_of_affected": float(w[m].sum() / aw_total) if aw_total else 0.0,
                "mean_income_change_per_affected": -net / float(w[m].sum()) if w[m].sum() else 0.0,
                "cushioning_rate": 1.0 - net / gross if gross else float("nan"),
            }
        return out

    return {
        "attribution_coverage_of_net_loss": coverage,
        "by_gender": group_stats(ctx["gender"].to_numpy(str)),
        "by_age_band": group_stats(ctx["age_band"].to_numpy(str)),
        "by_household_type": group_stats(ctx["hh_type"].to_numpy(str)),
        "by_couple_earners": group_stats(ctx["couple_earners"].to_numpy(str)),
    }


def summarise(draws):
    """MC mean +/- SD over the nested per-draw dicts."""
    out = {}
    for dim in ("by_gender", "by_age_band", "by_household_type", "by_couple_earners"):
        labels = sorted({lab for d in draws for lab in d[dim]})
        out[dim] = {}
        for lab in labels:
            out[dim][lab] = {}
            for k in ("share_of_affected", "mean_income_change_per_affected", "cushioning_rate"):
                vals = [d[dim][lab][k] for d in draws if lab in d[dim]]
                out[dim][lab][k] = {
                    "mean": float(np.mean(vals)),
                    "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                    "n_draws_present": len(vals),
                }
    out["attribution_coverage_of_net_loss_mean"] = float(
        np.mean([d["attribution_coverage_of_net_loss"] for d in draws])
    )
    return out


def main() -> None:
    dataset, baseline, persons = _baseline_and_persons(DATASET, None, PERIOD)
    ctx = build_person_context(baseline, persons)
    base_hh = hh_arrays(baseline)
    base_emp = persons["employment_income"].to_numpy(float)

    disp_draws = []
    for seed in range(N_DRAWS):
        table = apply_shocks(persons, PRESETS["full_tariff_displacement"], seed=seed)
        shocked = build_shocked_simulation(dataset, baseline, table, PERIOD)
        d = draw_incidence(
            ctx, base_hh, hh_arrays(shocked), base_emp,
            table["employment_income"].to_numpy(float),
            table["displaced"].to_numpy(),
        )
        del shocked
        disp_draws.append(d)
        print(f"[demographics] displacement seed {seed} done", flush=True)

    wc_table = apply_shocks(persons, PRESETS["full_tariff_wage_cut"], seed=0)
    wc_shocked = build_shocked_simulation(dataset, baseline, wc_table, PERIOD)
    wc_emp = wc_table["employment_income"].to_numpy(float)
    wc = draw_incidence(
        ctx, base_hh, hh_arrays(wc_shocked), base_emp, wc_emp, base_emp > wc_emp
    )

    out = {
        "n_draws_displacement": N_DRAWS,
        "full_tariff_displacement": summarise(disp_draws),
        "full_tariff_displacement_per_draw": disp_draws,
        "full_tariff_wage_cut": wc,
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "demographics.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out["full_tariff_displacement"], indent=2))


if __name__ == "__main__":
    main()
