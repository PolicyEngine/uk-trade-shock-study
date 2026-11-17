"""Emit the episode-3 constants for the multishock paper from the
in-repo tariff pipeline.

Reads the E3 result files written by ``analysis/run_scenarios.py --e3``:

* results/full_tariff_displacement.json -- the CENTRAL displacement
  estimate: exact-expectation design (tariff_pipeline/expectation.py), New
  Style JSA approximation on;
* results/full_tariff_displacement_no_jsa.json -- the same without it;
* results/full_tariff_wage_cut.json -- the wage-cut margin;
* results/full_tariff_displacement_bernoulli.json -- the superseded 20-draw
  Bernoulli run (no New Style JSA), kept as a comparator: it is the only
  source of a per-draw dispersion.

Writes results/e3_constants.json, which incidence.py reads.

``displacement_cushion_sd_pct`` keeps its original meaning -- the standard
deviation of the cushioning rate ACROSS SINGLE BERNOULLI DRAWS -- and so
comes from the comparator file. It is not the precision of the central
estimate, which has no assignment noise; that is
``displacement_cushion_sampling_se_pct``.

Run after run_scenarios:  python analysis/emit_e3_constants.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def _load(name: str) -> dict:
    return json.loads((RESULTS / f"{name}.json").read_text())


def main() -> None:
    disp = _load("full_tariff_displacement")
    no_jsa = _load("full_tariff_displacement_no_jsa")
    wage = _load("full_tariff_wage_cut")
    bern = _load("full_tariff_displacement_bernoulli")
    if disp.get("design") != "exact_expectation" or not disp.get("new_style_jsa"):
        raise ValueError(
            "results/full_tariff_displacement.json is not the exact-expectation "
            "run with New Style JSA; run analysis/run_scenarios.py --e3"
        )

    out = {
        "source": "in-repo tariff pipeline (analysis/run_scenarios.py --e3)",
        "scenario": "full_tariff",
        "displacement_design": "exact_expectation",
        "displacement_new_style_jsa": True,
        "n_draws": disp["n_draws"],
        # the deterministic imposed shock: sum_i s_i x earnings_i x weight_i
        "gross_shock_gbp_m_per_year": round(disp["target_gross_earnings_loss"] / 1e6, 1),
        "displacement_cushion_pct": round(100 * disp["cushioning_rate"], 1),
        "wagecut_cushion_pct": round(100 * wage["cushioning_rate_pooled"], 1),
        "displacement_cushion_sd_pct": round(100 * bern["cushioning_rate_sd"], 1),
        "displacement_cushion_sd_source": (
            "SD across the 20 single Bernoulli draws of the superseded run "
            "(no New Style JSA); not the precision of the central estimate"
        ),
        "wagecut_cushion_sd_pct": round(100 * wage["cushioning_rate_sd"], 1),
        "displacement_cushion_mc_se_pct": 0.0,
        "displacement_cushion_sampling_se_pct": round(
            100 * disp["cushioning_rate_sampling_se"], 1
        ),
        "displacement_cushion_no_jsa_pct": round(100 * no_jsa["cushioning_rate"], 1),
        "bernoulli_draws": bern["n_draws"],
    }
    (RESULTS / "e3_constants.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
