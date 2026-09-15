"""Calibrate the reallocation margin's services wage penalty from the FRS.

Source: FRS 2024-25 (packaged frs_2024_25.h5), employees with positive
employment income, at the study period (2026). SOURCE = the exposed
goods-producing divisions in the US-export-intensity table (SIC 10-32);
DESTINATION = shocks.REALLOCATION_DESTINATIONS (47 retail, 86 health,
49 land transport/delivery, 56 food service).

Reports (all grossing-weighted):
  - mean annual earnings and annual hours in source and destination;
  - the headline ANNUAL EARNINGS penalty 1 - mean_dst/mean_src (the default,
    shocks.DEFAULT_REALLOCATION_PENALTY): it embeds the hours fall, which is
    the economically relevant quantity for household income;
  - the hours/age-controlled penalty from a weighted OLS of log annual
    earnings on a destination dummy, log hours, age and age^2 (the pure
    hourly-wage gap, reported as the lower bound);
  - the destination employment-weight mix (shocks.DESTINATION_SHARES).

Cross-check: ONS ASHE 2024 median gross weekly earnings for full-time
employees by SIC section put manufacturing (C) around £700/week against
wholesale & retail (G) ~£590, transport & storage (H) ~£650, human health
(Q) ~£670 and accommodation & food (I) ~£490 — a full-time-only gap of
roughly 10-20% that brackets the hours-controlled FRS figure, with the FRS
all-employee figure larger because it also picks up the part-time shift.

Usage: .venv/bin/python analysis/reallocation_calibration.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from uk_trade_shock_study.exposure import load_us_export_intensity, simulation_sic_division
from uk_trade_shock_study.shocks import REALLOCATION_DESTINATIONS

PERIOD = 2026
DATASET = Path("data/frs_2024_25.h5")
OUT = Path("results/reallocation_calibration.json")


def main() -> None:
    from policyengine_uk import Microsimulation
    from policyengine_uk.data import UKSingleYearDataset

    sim = Microsimulation(dataset=UKSingleYearDataset(file_path=str(DATASET)))
    frame = pd.DataFrame(
        {
            "sic": simulation_sic_division(sim, PERIOD),
            "earnings": sim.calculate("employment_income", period=PERIOD, map_to="person").values.astype(float),
            "hours": sim.calculate("hours_worked", period=PERIOD, map_to="person").values.astype(float),
            "age": sim.calculate("age", period=PERIOD, map_to="person").values.astype(float),
            "w": sim.calculate("person_weight", period=PERIOD, map_to="person").values.astype(float),
        }
    )
    frame = frame[frame.earnings > 0]
    source = list(load_us_export_intensity().index)
    dest = list(REALLOCATION_DESTINATIONS)

    def block(codes):
        s = frame[frame.sic.isin(codes)]
        return {
            "mean_annual_earnings": float(np.average(s.earnings, weights=s.w)),
            "mean_annual_hours": float(np.average(s.hours, weights=s.w)),
            "mean_hourly": float((s.earnings * s.w).sum() / (s.hours * s.w).sum()),
            "employees": float(s.w.sum()),
        }

    src, dst = block(source), block(dest)
    per_destination = {int(c): block([c]) for c in dest}
    total = sum(v["employees"] for v in per_destination.values())
    shares = {c: v["employees"] / total for c, v in per_destination.items()}

    reg = frame[frame.sic.isin(source + dest) & (frame.hours > 0)]
    D = reg.sic.isin(dest).astype(float).to_numpy()
    X = np.column_stack([np.ones(len(reg)), D, np.log(reg.hours), reg.age, reg.age ** 2])
    y = np.log(reg.earnings.to_numpy())
    rw = np.sqrt(reg.w.to_numpy())
    beta = np.linalg.lstsq(X * rw[:, None], y * rw, rcond=None)[0]

    out = {
        "source_divisions": [int(c) for c in source],
        "destination_divisions": dest,
        "source": src,
        "destination": dst,
        "per_destination": per_destination,
        "destination_shares": shares,
        "annual_earnings_penalty": 1.0 - dst["mean_annual_earnings"] / src["mean_annual_earnings"],
        "hourly_penalty_unconditional": 1.0 - dst["mean_hourly"] / src["mean_hourly"],
        "hours_age_controlled_penalty": float(1.0 - np.exp(beta[1])),
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
