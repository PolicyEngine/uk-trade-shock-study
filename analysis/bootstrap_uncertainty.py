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

Two estimators of each cushioning ratio are reported side by side.

* ``*_cushioning`` is the original mean of per-draw ratios,
  ``mean_d (1 - net_d / gross_d)``, computed over the draws with a positive
  gross loss. Because the displacement margin selects few records, a resample
  can leave a draw with a very small gross loss; that draw is retained with a
  near-zero denominator, and the ratio it contributes is unbounded. This is
  what makes the replicate distribution right-skewed, so the percentile
  interval sits asymmetrically around the point estimate.
* ``*_cushioning_pooled`` is the ratio of pooled sums,
  ``1 - sum_d net_d / sum_d gross_d``. Numerator and denominator are pooled
  across the common assignment draws before the single ratio is formed, so no
  small denominator is inverted on its own. It is the better-behaved of the
  two and should be preferred when they disagree; both are reported so the
  difference is visible.

Caveats, stated in the output.

1. FRS design variables (PSUs and strata) are not present in the public
   research file used here, so households are treated as the independent
   resampling unit. This understates design effects from geographic clustering
   and overstates them from stratification; the result is an approximation to
   the design-based sampling variance, clearly labelled.
2. The resample reweights households but holds the *original* displacement
   assignment fixed. The per-household contributions are those of the
   assignment draws computed on the realised sample: the small set of records
   that were selected for displacement is the same in every replicate, and a
   replicate only changes how often each of them is counted. The interval is
   therefore conditional on that selection. It omits the variance component
   that arises because a different FRS sample would have offered a different
   pool of records to select from, and because the resampled sample would have
   drawn a different selection from it. The omission is one-sided: the
   reported interval is too narrow, that is, anti-conservative, and its
   nominal coverage is below 95%. The understatement grows as the number of
   distinct households carrying a positive gross loss falls, which is why that
   support size is reported alongside the interval. Removing the conditioning
   would require re-running PolicyEngine inside every bootstrap replicate,
   which is not computationally feasible here.

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

#: Per-margin arrays stored in the contributions cache. A margin counts as
#: cached only when ALL of them are present.
CONTRIBUTION_KEYS = ("gross", "net", "exch")

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


def support_sizes(
    contributions: dict[str, dict[str, np.ndarray]],
) -> dict[str, dict[str, float]]:
    """How many distinct households actually carry the estimate.

    A household bootstrap can only reflect variation in units that contribute.
    With displacement selecting few records, the whole interval rests on a
    handful of households, so the support size is reported next to it.
    """
    summary: dict[str, dict[str, float]] = {}
    ever_positive = None
    for margin in sorted(contributions):
        gross = contributions[margin]["gross"]
        positive = gross > 0
        any_draw = positive.any(axis=0)
        ever_positive = (
            any_draw if ever_positive is None else (ever_positive | any_draw)
        )
        per_draw = positive.sum(axis=1)
        draw_totals = gross.sum(axis=1)
        mean_total = float(draw_totals.mean())
        summary[margin] = {
            "households_positive_gross_any_draw": int(any_draw.sum()),
            "households_positive_gross_every_draw": int(positive.all(axis=0).sum()),
            "households_positive_gross_per_draw_mean": float(per_draw.mean()),
            "households_positive_gross_per_draw_min": int(per_draw.min()),
            "smallest_draw_gross_share_of_mean": (
                float(draw_totals.min() / mean_total)
                if mean_total > 0
                else float("nan")
            ),
        }
    summary["any_margin"] = {
        "households_positive_gross_any_draw": int(ever_positive.sum())
        if ever_positive is not None
        else 0
    }
    summary["note"] = (
        "Support of the household bootstrap: households whose weighted gross "
        "earnings loss is positive in at least one common assignment draw. "
        "The resample can only vary these households' multiplicities, so a "
        "small support means the reported interval understates sampling "
        "variability."
    )
    return summary


def bootstrap_contrast(
    contributions: dict[str, dict[str, np.ndarray]],
    n_boot: int,
    seed: int,
) -> dict:
    """Bootstrap households; average ratio statistics over common draws.

    ``contributions[margin][key]`` is a (draws, households) array of weighted
    per-household contributions, key in {"gross", "net", "exch"}.

    Each cushioning ratio is reported twice: ``*_cushioning`` is the original
    mean of per-draw ratios, ``*_cushioning_pooled`` is the ratio of the sums
    pooled across draws. The pooled form never inverts a single draw's small
    gross loss, so it is not subject to the right skew the mean-of-ratios
    estimator shows here.
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
            valid = np.isfinite(cushioning)
            out[f"{m}_cushioning"] = (
                float(cushioning[valid].mean()) if valid.any() else float("nan")
            )
            out[f"{m}_valid_draws"] = float(valid.sum())
            # Ratio of pooled sums: one denominator, formed after pooling the
            # common assignment draws, so no individual draw's near-zero gross
            # loss is inverted.
            pooled_gross = float(gross.sum())
            pooled_net = float(net.sum())
            out[f"{m}_cushioning_pooled"] = (
                1.0 - pooled_net / pooled_gross if pooled_gross > 0 else float("nan")
            )
            out[f"{m}_exchequer"] = float(exch.mean())
            out[f"{m}_gross"] = float(gross.mean())
        out["cushioning_difference"] = (
            out["wage_cut_cushioning"] - out["displacement_cushioning"]
        )
        out["cushioning_difference_pooled"] = (
            out["wage_cut_cushioning_pooled"] - out["displacement_cushioning_pooled"]
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
        finite = arr[np.isfinite(arr)]
        nan = float("nan")
        summary[k] = {
            "point": point[k],
            "bootstrap_se": float(finite.std(ddof=1)) if finite.size > 1 else nan,
            "ci_2_5": float(np.percentile(finite, 2.5)) if finite.size else nan,
            "ci_97_5": float(np.percentile(finite, 97.5)) if finite.size else nan,
            # A replicate is undefined when it leaves no gross loss to divide
            # by. Reported so a summary computed on a subset of replicates is
            # never mistaken for one computed on all of them.
            "finite_replicates": int(finite.size),
        }
    return summary


def load_cached_contributions(
    stored, margins=tuple(SCENARIOS)
) -> dict[str, dict[str, np.ndarray]]:
    """Load ONLY the margins the archive actually holds.

    A run interrupted part-way through the margin loop leaves a cache with one
    margin in it. Loading every margin unconditionally turns that partial cache
    into a ``KeyError`` from the archive, so the resume guards downstream are
    never reached and the completed margin has to be recomputed. Loading margin
    by margin is what makes the resume real: an absent (or half-written) margin
    is simply recomputed and the present one is reused.
    """
    available = set(getattr(stored, "files", ()) or stored.keys())
    contributions: dict[str, dict[str, np.ndarray]] = {}
    for margin in margins:
        names = {key: f"{margin}_{key}" for key in CONTRIBUTION_KEYS}
        present = [key for key, name in names.items() if name in available]
        if len(present) == len(CONTRIBUTION_KEYS):
            contributions[margin] = {
                key: np.asarray(stored[name]) for key, name in names.items()
            }
        elif present:
            # Half a margin is not a resumable margin: recompute it whole.
            print(
                f"[cache] margin {margin!r} is incomplete in the archive "
                f"(has {sorted(present)}); it will be recomputed."
            )
    return contributions


def main() -> None:
    cache = Path("results/bootstrap_contributions.npz")
    contributions: dict[str, dict[str, np.ndarray]] = {}
    if cache.exists():
        contributions = load_cached_contributions(np.load(cache))
        missing = [m for m in SCENARIOS if m not in contributions]
        print(
            f"[cache] loaded per-household contributions from {cache} for "
            f"{sorted(contributions) or 'no margins'}"
            + (f"; recomputing {missing}" if missing else "")
        )

    hh_ids = None
    # The cached contributions are the only PolicyEngine output the summary
    # needs, so a complete cache lets the JSON be rebuilt without the
    # microsimulation (and without the licensed FRS file).
    if set(contributions) != set(SCENARIOS):
        dataset, baseline, persons = _baseline_and_persons(DATASET, None, PERIOD)
        hh_id = np.asarray(
            baseline.calculate("household_id", period=PERIOD, map_to="person").values
        )
        hh_ids, hh_index = np.unique(hh_id, return_inverse=True)
        baseline_hh = {
            "gov": np.asarray(
                baseline.calculate(
                    "gov_balance", period=PERIOD, map_to="household"
                ).values,
                dtype=float,
            ),
            "hni": np.asarray(
                baseline.calculate(
                    "hbai_household_net_income", period=PERIOD, map_to="household"
                ).values,
                dtype=float,
            ),
        }
        for margin, scenario in SCENARIOS.items():
            if margin in contributions:
                continue
            gross_rows, net_rows, exch_rows = [], [], []
            for seed in range(N_DRAWS):
                shocked_table = apply_shocks(persons, scenario, seed=seed)
                shocked = build_shocked_simulation(
                    dataset, baseline, shocked_table, PERIOD
                )
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
        np.savez_compressed(
            cache,
            **{
                f"{margin}_{key}": array
                for margin, arrays in contributions.items()
                for key, array in arrays.items()
            },
        )
    n_households = (
        int(len(hh_ids))
        if hh_ids is not None
        else int(contributions[next(iter(SCENARIOS))]["gross"].shape[1])
    )

    summary = bootstrap_contrast(contributions, N_BOOT, BOOT_SEED)
    support = support_sizes(contributions)
    out = {
        "design": {
            "scenarios": {m: s.name for m, s in SCENARIOS.items()},
            "n_draws": N_DRAWS,
            "n_boot": N_BOOT,
            "bootstrap_seed": BOOT_SEED,
            "n_households": n_households,
            "resampling_unit": "household",
            "estimators": {
                "mean_of_per_draw_ratios": (
                    "keys ending _cushioning: mean over assignment draws of "
                    "1 - net/gross, computed on draws with a positive gross "
                    "loss. Retains near-zero denominators, which is the source "
                    "of the right skew in the replicate distribution."
                ),
                "ratio_of_pooled_sums": (
                    "keys ending _cushioning_pooled: 1 - (sum of net over "
                    "draws) / (sum of gross over draws). One denominator, "
                    "formed after pooling; preferred when the two disagree."
                ),
            },
            "support": support,
            "notes": (
                "Household bootstrap of the FRS sample; PSU/strata design "
                "variables are unavailable in the research file, so the "
                "interval approximates design-based sampling variance. "
                "Assignment dispersion is reported separately in the primary "
                "table; replicate statistics average over common assignment "
                "draws before resampling variation is measured. "
                "CONDITIONING LIMITATION: the resample reweights households "
                "but holds the original displacement assignment fixed. The "
                "per-household contributions come from assignment draws taken "
                "on the realised sample, so the same small set of selected "
                "records appears in every replicate and only their "
                "multiplicities change (see design.support for how many "
                "households that is). The interval is therefore conditional "
                "on that selection: it omits the variation that would come "
                "from a different FRS sample offering a different pool of "
                "records, and from re-drawing the assignment within each "
                "resampled sample. The bias is one-sided - the reported "
                "interval is too narrow and its coverage is below nominal - "
                "and it is larger the smaller the support. Removing the "
                "conditioning would require re-running PolicyEngine inside "
                "each of the "
                f"{N_BOOT} replicates, which is not computationally feasible; "
                "the interval should be read as a lower bound on sampling "
                "uncertainty."
            ),
        },
        "estimates": summary,
    }
    Path("results/bootstrap_uncertainty.json").write_text(json.dumps(out, indent=2))
    print("[written] results/bootstrap_uncertainty.json")


if __name__ == "__main__":
    main()
