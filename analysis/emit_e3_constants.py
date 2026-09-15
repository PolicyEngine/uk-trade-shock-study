"""Emit the episode-3 constants for the multishock paper from the
in-repo tariff pipeline, replacing the three values previously imported
from the unpublished companion manuscript.

Reads results/full_tariff_{displacement,wage_cut}.json (produced by
analysis/run_scenarios.py) and computes the deterministic gross
earnings shock with the same exposure arithmetic the runs used.
Writes results/e3_constants.json, which incidence.py reads when
present.

Run after run_scenarios:  python analysis/emit_e3_constants.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def main() -> None:
    import pandas as pd
    from policyengine_uk import Microsimulation
    from policyengine_uk.data import UKSingleYearDataset

    from tariff_pipeline.exposure import (
        sector_earnings_shocks,
        simulation_sic_division,
    )

    disp = json.loads((RESULTS / "full_tariff_displacement.json").read_text())
    wage = json.loads((RESULTS / "full_tariff_wage_cut.json").read_text())

    dataset = ROOT / "data" / "frs_2024_25.h5"
    sim = Microsimulation(dataset=UKSingleYearDataset(file_path=str(dataset)))
    period = 2026
    sic = simulation_sic_division(sim, period)
    emp = sim.calculate("employment_income", period=period, map_to="person").values
    w = sim.calculate("person_weight", period=period, map_to="person").values
    shock = pd.Series(sic).map(sector_earnings_shocks("full_tariff")).fillna(0.0).to_numpy()
    gross_m = float((shock * emp * w).sum()) / 1e6

    out = {
        "source": "in-repo tariff pipeline (analysis/run_scenarios.py)",
        "scenario": "full_tariff",
        "n_draws": disp["n_draws"],
        "gross_shock_gbp_m_per_year": round(gross_m, 1),
        "displacement_cushion_pct": round(100 * disp["cushioning_rate_mean"], 1),
        "wagecut_cushion_pct": round(100 * wage["cushioning_rate_mean"], 1),
        "displacement_cushion_sd_pct": round(100 * disp["cushioning_rate_sd"], 1),
        "wagecut_cushion_sd_pct": round(100 * wage["cushioning_rate_sd"], 1),
    }
    (RESULTS / "e3_constants.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
