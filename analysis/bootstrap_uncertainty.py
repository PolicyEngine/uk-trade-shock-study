"""Household-bootstrap sampling uncertainty for the primary contrast.

The primary table reports assignment SDs, which describe Monte Carlo
allocation dispersion, not survey sampling variance. This script adds the
missing piece: a household bootstrap of the FRS sample for the unit 12-month
wage-cut versus displacement cushioning contrast.

Method. For each margin, run the declared scenario for a set of common
assignment draws and store per-household weighted contributions to the gross
earnings loss, the net disposable-income loss and the Exchequer cost. All
sampling variation enters through which households are observed, so
resampling households with replacement (multiplying each household's weighted
contribution by its multinomial resample count) reproduces the bootstrap
distribution of every downstream ratio without re-running PolicyEngine.
Replicate statistics average over the common assignment draws first, so the
reported interval is sampling uncertainty of the draw-averaged contrast;
assignment dispersion remains reported separately in the primary table.

Caveats, stated in the output: the FRS design variables (PSUs and strata) are
not present in the public research file used here, so households are treated
as the independent resampling unit. This understates design effects from
geographic clustering and overstates them from stratification; the result is
an approximation to the design-based sampling variance, clearly labelled.

Writes results/bootstrap_uncertainty.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uk_trade_shock_study.runner import _baseline_and_persons  # noqa: E402
from uk_trade_shock_study.shocks import (  # noqa: E402
    TradeShockScenario,
    apply_shocks,
    build_shocked_simulation,
)

DATASET = Path("data/frs_2024_25.h5")
PERIOD = 2026
N_DRAWS = 25
N_BOOT = 999
BOOT_SEED = 20260805

SCENARIOS = {
    "displacement": TradeShockScenario(
        "submission_unit_12m_displacement",
        "full_tariff",
        "displacement",
        elasticity=1.0,
        duration_equivalent=1.0,
        selection_method="balanced",
    ),
    "wage_cut": TradeShockScenario(
        "submission_unit_12m_wage_cut",
        "full_tariff",
        "wage_cut",
        elasticity=1.0,
        duration_equivalent=1.0,
        selection_method="balanced",
    ),
}


def household_contributions(sim, baseline_hh, hh_index, persons, shocked_table, period):
    """Weighted per-household contributions to the three aggregates."""
    hh_w = np.asarray(
        sim.calculate("household_weight", period=period, map_to="household").values,
        dtype=float,
    )
    gov = np.asarray(
        sim.calculate("gov_balance", period=period, map_to="household").values,
        dtype=float,
    )
    hni = np.asarray(
        sim.calculate("hbai_household_net_income", period=period, map_to="household").values,
        dtype=float,
    )
    person_loss = (
        persons["employment_income"].to_numpy(dtype=float)
        - shocked_table["employment_income"].to_numpy(dtype=float)
    ) * persons["weight"].to_numpy(dtype=float)
    gross = np.bincount(hh_index, weights=person_loss, minlength=len(hh_w))
    exch = (baseline_hh["gov"] - gov) * hh_w
    net = (baseline_hh["hni"] - hni) * hh_w
    return gross, net, exch


def bootstrap_contrast(
    contributions: dict[str, dict[str, np.ndarray]],
    n_boot: int,
    seed: int,
) -> dict:
    """Bootstrap households; average ratio statistics over common draws.

    ``contributions[margin][key]`` is a (draws, households) array of weighted
    per-household contributions, key in {"gross", "net", "exch"}.
    """
    margins = sorted(contributions)
    n_households = {m: contributions[m]["gross"].shape[1] for m in margins}
    if len(set(n_households.values())) != 1:
        raise ValueError("margins disagree on household count")
    h = next(iter(n_households.values()))
    rng = np.random.default_rng(seed)

    def stats(multiplier: np.ndarray) -> dict[str, float]:
        # A resample can zero a draw's gross loss (displacement selects few
        # records), leaving that draw's cushioning undefined; average over the
        # defined draws, mirroring the runner's _finite_mean_sd convention.
        out = {}
        for m in margins:
            gross = contributions[m]["gross"] @ multiplier
            net = contributions[m]["net"] @ multiplier
            exch = contributions[m]["exch"] @ multiplier
            with np.errstate(divide="ignore", invalid="ignore"):
                cushioning = np.where(gross > 0, 1.0 - net / gross, np.nan)
            out[f"{m}_cushioning"] = float(np.nanmean(cushioning))
            out[f"{m}_valid_draws"] = float(np.isfinite(cushioning).sum())
            out[f"{m}_exchequer"] = float(exch.mean())
            out[f"{m}_gross"] = float(gross.mean())
        out["cushioning_difference"] = (
            out["wage_cut_cushioning"] - out["displacement_cushioning"]
        )
        return out

    point = stats(np.ones(h))
    replicates: dict[str, list[float]] = {k: [] for k in point}
    for _ in range(n_boot):
        counts = rng.multinomial(h, np.full(h, 1.0 / h)).astype(float)
        rep = stats(counts)
        for k, v in rep.items():
            replicates[k].append(v)

    summary = {}
    for k, values in replicates.items():
        arr = np.asarray(values)
        summary[k] = {
            "point": point[k],
            "bootstrap_se": float(arr.std(ddof=1)),
            "ci_2_5": float(np.percentile(arr, 2.5)),
            "ci_97_5": float(np.percentile(arr, 97.5)),
        }
    return summary


def main() -> None:
    dataset, baseline, persons = _baseline_and_persons(DATASET, None, PERIOD)
    hh_id = np.asarray(
        baseline.calculate("household_id", period=PERIOD, map_to="person").values
    )
    hh_ids, hh_index = np.unique(hh_id, return_inverse=True)
    baseline_hh = {
        "gov": np.asarray(
            baseline.calculate("gov_balance", period=PERIOD, map_to="household").values,
            dtype=float,
        ),
        "hni": np.asarray(
            baseline.calculate(
                "hbai_household_net_income", period=PERIOD, map_to="household"
            ).values,
            dtype=float,
        ),
    }

    cache = Path("results/bootstrap_contributions.npz")
    contributions: dict[str, dict[str, np.ndarray]] = {}
    if cache.exists():
        stored = np.load(cache)
        contributions = {
            margin: {key: stored[f"{margin}_{key}"] for key in ("gross", "net", "exch")}
            for margin in SCENARIOS
        }
        print(f"[cache] loaded per-household contributions from {cache}")
    for margin, scenario in SCENARIOS.items():
        if margin in contributions:
            continue
        gross_rows, net_rows, exch_rows = [], [], []
        for seed in range(N_DRAWS):
            shocked_table = apply_shocks(persons, scenario, seed=seed)
            shocked = build_shocked_simulation(dataset, baseline, shocked_table, PERIOD)
            gross, net, exch = household_contributions(
                shocked, baseline_hh, hh_index, persons, shocked_table, PERIOD
            )
            gross_rows.append(gross)
            net_rows.append(net)
            exch_rows.append(exch)
            print(
                f"[{margin}] draw {seed}: gross={gross.sum() / 1e6:.0f}m "
                f"exchequer={exch.sum() / 1e6:.0f}m",
                flush=True,
            )
        contributions[margin] = {
            "gross": np.vstack(gross_rows),
            "net": np.vstack(net_rows),
            "exch": np.vstack(exch_rows),
        }
    if not cache.exists():
        np.savez_compressed(
            cache,
            **{
                f"{margin}_{key}": array
                for margin, arrays in contributions.items()
                for key, array in arrays.items()
            },
        )

    summary = bootstrap_contrast(contributions, N_BOOT, BOOT_SEED)
    out = {
        "design": {
            "scenarios": {m: s.name for m, s in SCENARIOS.items()},
            "n_draws": N_DRAWS,
            "n_boot": N_BOOT,
            "bootstrap_seed": BOOT_SEED,
            "n_households": int(len(hh_ids)),
            "resampling_unit": "household",
            "notes": (
                "Household bootstrap of the FRS sample; PSU/strata design "
                "variables are unavailable in the research file, so the "
                "interval approximates design-based sampling variance. "
                "Assignment dispersion is reported separately in the primary "
                "table; replicate statistics average over common assignment "
                "draws before resampling variation is measured."
            ),
        },
        "estimates": summary,
    }
    Path("results/bootstrap_uncertainty.json").write_text(json.dumps(out, indent=2))
    print("[written] results/bootstrap_uncertainty.json")


if __name__ == "__main__":
    main()
