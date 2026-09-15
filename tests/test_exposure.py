"""Unit tests for the tariff-exposure math (no FRS data required)."""

import numpy as np
import pandas as pd
import pytest

from uk_trade_shock_study.exposure import (
    AUTO_SIC,
    BASELINE_TARIFF,
    EPD_AUTO_EFFECTIVE_RATE,
    EPD_STEEL_RELIEF_SHARE,
    ELASTICITY_SCENARIOS,
    PHARMA_SIC,
    STEEL_SIC,
    load_us_export_intensity,
    person_earnings_shock,
    sector_earnings_shocks,
    tariff_rates,
)


def test_default_calibration_scenarios_are_explicit():
    assert ELASTICITY_SCENARIOS == {
        "obr_low": 0.4,
        "central": 1.0,
        "former_april_high": 2.0,
        "severe": 3.0,
    }


def test_full_tariff_schedule():
    rates = tariff_rates("full_tariff")
    assert rates.loc[AUTO_SIC] == 0.25
    assert rates.loc[STEEL_SIC] == 0.25
    assert rates.loc[PHARMA_SIC] == BASELINE_TARIFF
    other = rates.drop([AUTO_SIC, STEEL_SIC])
    assert (other == BASELINE_TARIFF).all()


def test_epd_schedule_mitigates_every_carveout():
    full, epd = tariff_rates("full_tariff"), tariff_rates("epd")
    assert epd.loc[AUTO_SIC] == EPD_AUTO_EFFECTIVE_RATE < full.loc[AUTO_SIC]
    assert epd.loc[STEEL_SIC] == pytest.approx(0.25 * (1 - EPD_STEEL_RELIEF_SHARE))
    assert epd.loc[PHARMA_SIC] == 0.0  # pharma exempt (Dec 2025 deal)
    assert (epd <= full).all()


def test_unknown_scenario_errors():
    with pytest.raises(ValueError):
        tariff_rates("no_such_scenario")


def test_intensity_table_shares_in_unit_interval():
    table = load_us_export_intensity()
    assert table["us_export_share"].between(0, 1).all()
    assert table.index.is_unique


def test_shock_formula():
    """shock_j = elasticity * tariff_j * share_j * passthrough, clipped to [0,1]."""
    intensity = pd.DataFrame(
        {"us_export_share": [0.1, 0.5]}, index=pd.Index([29, 21], name="sic_division")
    )
    shock = sector_earnings_shocks("full_tariff", elasticity=2.0, passthrough=0.5, intensity=intensity)
    assert shock.loc[29] == pytest.approx(2.0 * 0.25 * 0.1 * 0.5)
    assert shock.loc[21] == pytest.approx(2.0 * BASELINE_TARIFF * 0.5 * 0.5)
    huge = sector_earnings_shocks("full_tariff", elasticity=100.0, intensity=intensity)
    assert (huge <= 1.0).all()


def test_wage_bill_incidence_name_and_legacy_alias_match():
    canonical = sector_earnings_shocks("full_tariff", wage_bill_incidence=0.5)
    legacy = sector_earnings_shocks("full_tariff", passthrough=0.5)
    pd.testing.assert_series_equal(canonical, legacy)
    with pytest.raises(ValueError):
        sector_earnings_shocks(
            "full_tariff", passthrough=0.5, wage_bill_incidence=0.5
        )


def test_epd_pharma_shock_is_zero():
    shock = sector_earnings_shocks("epd")
    assert shock.loc[PHARMA_SIC] == 0.0


def test_person_shock_unexposed_sectors_zero():
    """Services / unmatched / NaN divisions take a zero shock."""
    codes = np.array([29.0, 62.0, np.nan])  # autos, IT services, missing
    shock = person_earnings_shock(codes, "full_tariff")
    assert shock[0] > 0
    assert shock[1] == 0.0
    assert shock[2] == 0.0
