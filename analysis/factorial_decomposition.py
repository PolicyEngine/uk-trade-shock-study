"""Factorial decomposition of the wage-cut vs displacement cushioning gap.

Referee major point 1: the headline contrast changes the adjustment margin
AND the concentration/identity of losses at once. This script adds the
missing factorial middle cell — a CONCENTRATED WAGE CUT that imposes the
displacement draw's exact worker-level losses (same drawn workers, same
seeds, same balanced assignment, 100% of annual earnings) while leaving
everyone EMPLOYED — and decomposes the headline gap sequentially:

    broad wage cut  ->  concentrated wage cut  ->  displacement
      [concentration + worker selection]   [employment state]

Because employment is binary, "diffuse displacement" is not a definable
cell, so this is the unique monotone path through the feasible factorial
lattice.  No Shapley interpretation is available or claimed: a Shapley
value would require the undefined diffuse-displacement coalition to be
valued, and the two factors are not symmetric (concentration is continuous,
employment state is binary).

Outputs
  results/submission/submission_{anchor}_12m_concentrated_wage_cut.json
  results/factorial_decomposition.json  (paired per-seed decomposition plus
  a channel split of each step into income tax, employee NI, Universal
  Credit, other benefits and the pension/other residual on seeds 0-4).

Usage: .venv/bin/python analysis/factorial_decomposition.py [--n-draws 50]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from uk_trade_shock_study.exposure import ELASTICITY_SCENARIOS  # noqa: E402
from uk_trade_shock_study.runner import (  # noqa: E402
    _baseline_and_persons,
    run_monte_carlo_prepared,
    write_result,
)
from uk_trade_shock_study.shocks import (  # noqa: E402
    TradeShockScenario,
    apply_shocks,
    build_shocked_simulation,
)
from mechanism_decomposition import cushioning_components  # noqa: E402

PERIOD = 2026
DATASET = Path("data/frs_2024_25.h5")
SUBMISSION = Path("results/submission")
ANCHORS = {"obr": ELASTICITY_SCENARIOS["obr_low"], "unit": 1.0}
CHANNEL_SEEDS = range(5)


def scenario(anchor: str, margin: str) -> TradeShockScenario:
    name = f"submission_{anchor}_12m_{margin}"
    return TradeShockScenario(
        name,
        "full_tariff",
        margin,
        elasticity=ANCHORS[anchor],
        duration_equivalent=1.0,
        selection_method="balanced",
    )


def paired_decomposition(anchor: str, n_draws: int) -> dict:
    """Per-seed sequential decomposition from the three 50-draw artifacts."""
    rates = {}
    for margin in ("wage_cut", "concentrated_wage_cut", "displacement"):
        path = SUBMISSION / f"submission_{anchor}_12m_{margin}.json"
        data = json.loads(path.read_text())
        if data["n_draws"] != n_draws:
            raise ValueError(f"{path} has {data['n_draws']} draws, expected {n_draws}")
        rates[margin] = np.asarray(
            [d["cushioning_rate"] for d in data["draws"]], dtype=float
        )
    wc, conc, disp = (
        rates["wage_cut"],
        rates["concentrated_wage_cut"],
        rates["displacement"],
    )
    valid = np.isfinite(wc) & np.isfinite(conc) & np.isfinite(disp)
    wc, conc, disp = wc[valid], conc[valid], disp[valid]

    def summary(values: np.ndarray) -> dict:
        return {
            "mean_pp": float(values.mean() * 100),
            "assignment_sd_pp": float(values.std(ddof=1) * 100),
        }

    total = wc - disp
    concentration = wc - conc
    state = conc - disp
    return {
        "valid_draws": int(valid.sum()),
        "cushioning_percent": {
            "wage_cut": float(wc.mean() * 100),
            "concentrated_wage_cut": float(conc.mean() * 100),
            "displacement": float(disp.mean() * 100),
        },
        "total_gap": summary(total),
        "concentration_and_selection_step": summary(concentration),
        "employment_state_step": summary(state),
        "employment_state_share_of_gap": float(state.mean() / total.mean())
        if total.mean()
        else float("nan"),
    }


def channel_split(dataset, baseline, persons, anchor: str) -> dict:
    """Cushioning components by margin, averaged over CHANNEL_SEEDS."""
    per_margin: dict[str, list[dict]] = {}
    for margin in ("wage_cut", "concentrated_wage_cut", "displacement"):
        scen = scenario(anchor, margin)
        rows = []
        for seed in CHANNEL_SEEDS:
            table = apply_shocks(persons, scen, seed=seed)
            shocked = build_shocked_simulation(dataset, baseline, table, PERIOD)
            comp = cushioning_components(baseline, shocked, persons, table, PERIOD)
            rows.append(comp["components_share_of_gross_loss"])
            print(
                f"[channels {anchor}] {margin} seed {seed}: "
                f"cushion {comp['cushioning_rate']:.4f}",
                flush=True,
            )
        per_margin[margin] = rows
    keys = per_margin["wage_cut"][0].keys()
    means = {
        margin: {k: float(np.mean([r[k] for r in rows])) for k in keys}
        for margin, rows in per_margin.items()
    }
    return {
        "seeds": list(CHANNEL_SEEDS),
        "components_share_of_gross_loss": means,
        "concentration_and_selection_step_by_channel_pp": {
            k: (means["wage_cut"][k] - means["concentrated_wage_cut"][k]) * 100
            for k in keys
        },
        "employment_state_step_by_channel_pp": {
            k: (means["concentrated_wage_cut"][k] - means["displacement"][k]) * 100
            for k in keys
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-draws", type=int, default=50)
    parser.add_argument("--anchors", nargs="*", default=list(ANCHORS))
    parser.add_argument("--skip-channels", action="store_true")
    args = parser.parse_args()

    dataset, baseline, persons = _baseline_and_persons(DATASET, None, PERIOD)
    out: dict = {"design": {
        "n_draws": args.n_draws,
        "selection_method": "balanced",
        "notes": (
            "Concentrated wage cut imposes the displacement draw's exact "
            "worker-level losses (paired seeds, balanced assignment) with "
            "no employment-state change. Sequential decomposition "
            "broad->concentrated->displaced is the unique monotone path "
            "because diffuse displacement is undefined.  It carries no Shapley "
            "interpretation: that would require the undefined "
            "diffuse-displacement coalition to be valued."
        ),
    }, "anchors": {}}
    for anchor in args.anchors:
        scen = scenario(anchor, "concentrated_wage_cut")
        result = run_monte_carlo_prepared(
            dataset, baseline, persons, scen, PERIOD, args.n_draws
        )
        write_result(result, SUBMISSION / f"{scen.name}.json")
        print(
            f"{scen.name}: cushion={result.cushioning_rate_mean * 100:.1f}%"
            f" +- {result.cushioning_rate_sd * 100:.1f}",
            flush=True,
        )
        out["anchors"][anchor] = paired_decomposition(anchor, args.n_draws)
        if not args.skip_channels and anchor == "unit":
            out["anchors"][anchor]["channel_split"] = channel_split(
                dataset, baseline, persons, anchor
            )
    Path("results/factorial_decomposition.json").write_text(
        json.dumps(out, indent=2) + "\n"
    )
    print("[written] results/factorial_decomposition.json")


if __name__ == "__main__":
    main()
