"""Sector-level tariff exposure: tariff schedule x US-export intensity by SIC.

The shock enters through PRIMITIVES. A US tariff rate per SIC 2007 division
(the 2025 schedule, with and without the Economic Prosperity Deal) is combined
with the division's US-export intensity (US goods exports as a share of
division gross output — ONS/HMRC trade-by-SIC; the packaged CSV is a
PLACEHOLDER, see data/us_export_intensity_by_sic.csv and
analysis/build_trade_by_sic.py) and an export-demand elasticity to produce a
per-division earnings-shock size:

    earnings_shock_j = elasticity * tariff_j * us_export_share_j * passthrough

- ``elasticity`` is the proportional fall in US demand per unit tariff
  (trade elasticity; default 3.0, to be disciplined by De Lyon & Pessoa (2021)
  UK worker-level elasticities and the OBR Mar-2025 / Ignatenko et al. (2025)
  UK aggregate anchors — TODO before any paper-facing run).
- ``passthrough`` is the share of the sector output loss that lands on the
  sector wage bill (default 1.0 = full pass-through; a labour-share or
  margin-absorption parameter belongs here).

Employment is an OUTCOME of the derived shock (via the adjustment-margin
families in shocks.py), not a free input.

Tariff schedule (institutional detail, Commons Library CBP-10240; BEIS
Committee EPD report):
- 10% baseline on UK goods from April 2025;
- autos (SIC 29): 25%, reduced to 10% under the EPD on a 100,000-vehicle
  quota (8 May 2025; 25% above quota) — the EPD rate here is a quota-blended
  effective rate;
- steel (SIC 24): 25%, with conditional EPD relief (implementation was
  partial; parameterised). CAVEAT: ~80% of UK steel exports go to the EU
  (UK Steel), so the steel cell is more exposed to EU/UK safeguards than to
  US tariffs — flag in the paper, not in the code.
- pharmaceuticals (SIC 21): exempted (0%) for three years under the Dec 2025
  deal, paid for through VPAG/NHS pricing — a fiscal/pricing channel, not an
  employment channel; under ``full_tariff`` pharma faces the 10% baseline.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd

BASELINE_TARIFF = 0.10
AUTO_SIC = 29
STEEL_SIC = 24
PHARMA_SIC = 21

#: EPD auto quota: 100,000 vehicles at 10%, 25% above. SMMT: 2024 US exports
#: ~102k units, so the quota bites only marginally; effective rate modelled
#: at the in-quota 10% (TODO: blend with realised over-quota volumes).
EPD_AUTO_EFFECTIVE_RATE = 0.10
#: EPD steel relief was conditional and delayed; 0.5 = half of the 25% rate
#: relieved. TODO: update to the implemented terms.
EPD_STEEL_RELIEF_SHARE = 0.5

#: CALIBRATED (Jul 2026): elasticity = 2.0, anchored on the ONS realised
#: outturn — UK goods exports to the US fell 24.7% in April 2025 against a
#: trade-weighted average full-schedule tariff of 12.8% (computed from the
#: real intensity build: 0.247 / 0.128 = 1.93, rounded to 2.0; also mid-range
#: of short-run trade elasticities). With passthrough = 1.0 (the full sector
#: output loss lands on the sector wage bill, the right short-run displacement
#: assumption), the implied aggregate gross earnings loss under the full
#: tariff schedule on FRS 2024-25 is ~£1.8bn/yr, i.e. ~0.06% of GDP — the
#: DIRECT exposed-sector earnings channel only, sensibly below the OBR
#: Mar-2025 no-retaliation scenario peak of ~0.3-0.6% of GDP, which includes
#: uncertainty/investment and general-equilibrium channels a static
#: microsimulation deliberately excludes. run_scenarios.py prints the implied
#: aggregate at run time.
DEFAULT_ELASTICITY = 2.0
DEFAULT_PASSTHROUGH = 1.0

TARIFF_SCENARIOS = ("full_tariff", "epd")

US_EXPORT_INTENSITY_CSV = "us_export_intensity_by_sic.csv"


def tariff_rates(scenario: str) -> pd.Series:
    """US tariff rate per SIC division for a tariff scenario.

    ``full_tariff``: the pre-EPD schedule (10% baseline, 25% autos and steel,
    pharma at the 10% baseline). ``epd``: deal-mitigated (autos at the quota
    rate, steel partially relieved, pharma exempt).
    Returns a Series over the divisions in the packaged intensity table.
    """
    if scenario not in TARIFF_SCENARIOS:
        raise ValueError(f"unknown tariff scenario {scenario!r}; use one of {TARIFF_SCENARIOS}")
    divisions = load_us_export_intensity().index
    rates = pd.Series(BASELINE_TARIFF, index=divisions, dtype=float)
    if scenario == "full_tariff":
        rates.loc[AUTO_SIC] = 0.25
        rates.loc[STEEL_SIC] = 0.25
    else:  # epd
        rates.loc[AUTO_SIC] = EPD_AUTO_EFFECTIVE_RATE
        rates.loc[STEEL_SIC] = 0.25 * (1.0 - EPD_STEEL_RELIEF_SHARE)
        rates.loc[PHARMA_SIC] = 0.0
    return rates


def load_us_export_intensity() -> pd.DataFrame:
    """US-export intensity by SIC division (PLACEHOLDER numbers — see TODO)."""
    path = resources.files("uk_trade_shock_study") / "data" / US_EXPORT_INTENSITY_CSV
    table = pd.read_csv(str(path), comment="#")
    table = table.set_index("sic_division")
    if (table["us_export_share"] < 0).any() or (table["us_export_share"] > 1).any():
        raise ValueError("us_export_share must lie in [0, 1]")
    return table


def sector_earnings_shocks(
    scenario: str,
    elasticity: float = DEFAULT_ELASTICITY,
    passthrough: float = DEFAULT_PASSTHROUGH,
    intensity: pd.DataFrame | None = None,
) -> pd.Series:
    """Per-SIC-division earnings-shock size (fraction of the sector wage bill).

    shock_j = elasticity * tariff_j * us_export_share_j * passthrough,
    clipped to [0, 1].
    """
    if intensity is None:
        intensity = load_us_export_intensity()
    rates = tariff_rates(scenario).reindex(intensity.index).fillna(BASELINE_TARIFF)
    shock = elasticity * rates * intensity["us_export_share"] * passthrough
    return shock.clip(0.0, 1.0).rename("earnings_shock")


def simulation_sic_division(sim, period: int) -> np.ndarray:
    """Person-level SIC 2007 division straight from the simulation/h5.

    The packaged frs_2024_25.h5 carries ``sic_industry_division`` (a
    registered policyengine-uk variable; 2-digit SIC 2007 division, 0 =
    unknown/not applicable). This is the PRIMARY SIC source; the adult.tab
    join below (load_frs_adult_sic/attach_sic_division) is retained only as a
    documented fallback for h5 builds without the column. Division 0 is
    returned as NaN, so those persons take a zero shock (not exposed).
    """
    codes = sim.calculate("sic_industry_division", period=period, map_to="person").values
    codes = np.asarray(codes, dtype=float)
    return np.where(codes > 0, codes, np.nan)


def load_frs_adult_sic(adult_tab_path: str | Path) -> pd.Series:
    """SIC 2007 division keyed by ``SERNUM*1000 + PERSON`` from FRS adult.tab.

    Mirrors uk-ai-study's SOC join (person_id = SERNUM*1000 + PERSON, the
    current policyengine-uk-data convention). TODO: verify the FRS industry
    column name/coding in the 2024-25 adult.tab (candidates: ``SIC``,
    ``INDINC``); values outside known divisions become NaN.
    """
    adult = pd.read_csv(adult_tab_path, sep="\t")
    column = next((c for c in ("SIC", "INDINC") if c in adult.columns), None)
    if column is None:
        raise ValueError(
            "adult.tab has no recognised industry column (looked for SIC, "
            "INDINC); inspect the file and update load_frs_adult_sic."
        )
    keys = adult["SERNUM"].astype("int64") * 1000 + adult["PERSON"].astype("int64")
    if keys.duplicated().any():
        raise ValueError("adult.tab has duplicated SERNUM/PERSON keys.")
    sic = pd.to_numeric(adult[column], errors="coerce")
    sic = sic.where((sic >= 1) & (sic <= 99))
    return pd.Series(sic.to_numpy(), index=keys.to_numpy(), name="sic_division")


def attach_sic_division(
    person_ids: np.ndarray | pd.Series,
    adult_tab_path: str | Path,
) -> np.ndarray:
    """FRS SIC 2007 division (NaN for children/no-industry) per person."""
    lookup = load_frs_adult_sic(adult_tab_path)
    ids = pd.to_numeric(pd.Series(np.asarray(person_ids)), errors="raise").astype("int64")
    return ids.map(lookup).to_numpy(dtype=float)


def person_earnings_shock(
    sic_division: np.ndarray | pd.Series,
    scenario: str,
    elasticity: float = DEFAULT_ELASTICITY,
    passthrough: float = DEFAULT_PASSTHROUGH,
) -> np.ndarray:
    """Per-person earnings-shock size from SIC division codes.

    Divisions absent from the intensity table (services, unmatched, NaN)
    take a ZERO shock: only US-goods-exporting sectors are exposed.
    """
    shocks = sector_earnings_shocks(scenario, elasticity, passthrough)
    codes = pd.to_numeric(pd.Series(np.asarray(sic_division)), errors="coerce")
    return codes.map(shocks).fillna(0.0).to_numpy(dtype=float)
