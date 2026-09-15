"""Leontief upstream machinery on a toy 3-sector economy."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from uk_trade_shock_study import supply_chain as sc

# Toy economy: sector 0 buys from 1, sector 1 buys from 2, sector 2 buys
# nothing. Gross outputs 100 each; CoE 50/40/30.
Z = np.array(
    [
        [0.0, 0.0, 0.0],
        [20.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
    ]
)
TOY = sc.IOTables(
    products=("CPA_C29", "CPA_C24", "CPA_B05"),
    intermediate=Z,
    gross_output=np.array([100.0, 100.0, 100.0]),
    compensation=np.array([50.0, 40.0, 30.0]),
)

IOT_DATA = Path(__file__).resolve().parents[1] / "data" / "iot2022revisedproduct.xlsx"
requires_iot_data = pytest.mark.skipif(
    not IOT_DATA.exists(), reason="requires the non-distributed ONS IO workbook"
)


def test_technical_coefficients_and_leontief_inverse():
    A = TOY.technical_coefficients
    assert A[1, 0] == pytest.approx(0.2)
    assert A[2, 1] == pytest.approx(0.1)
    L = TOY.leontief_inverse
    # Triangular network: L = I + A + A^2 exactly (A^3 = 0).
    expected = np.eye(3) + A + A @ A
    assert np.allclose(L, expected)


def test_upstream_output_falls_exclude_direct_round():
    f = np.array([10.0, 0.0, 0.0])
    dg = sc.upstream_output_falls(TOY, f)
    # Sector 0's own fall excluded; sector 1 supplies 0.2 per unit -> 2.0;
    # sector 2 supplies 0.1 per unit of sector 1 -> 0.2.
    assert dg == pytest.approx([0.0, 2.0, 0.2])


def test_upstream_earnings_loss_uses_coe_ratio():
    f = np.array([10.0, 0.0, 0.0])
    dg = sc.upstream_output_falls(TOY, f)
    earnings = dg * TOY.coe_ratio
    assert earnings == pytest.approx([0.0, 2.0 * 0.4, 0.2 * 0.3])


def test_divisions_of_product_parsing():
    assert sc._divisions_of_product("CPA_C102_3") == (10,)
    assert sc._divisions_of_product("CPA_B06 & B07") == (6, 7)
    assert sc._divisions_of_product("CPA_F41, F42 & F43") == (41, 42, 43)
    assert sc._divisions_of_product("CPA_L68BXL683") == (68,)
    with pytest.raises(ValueError):
        sc._divisions_of_product("CPA_???")


def test_wage_cut_with_shock_conserves_wage_bill():
    persons = pd.DataFrame(
        {
            "employment_income": [30000.0, 20000.0, 0.0],
            "weight": [2.0, 1.0, 5.0],
            "sic_division": [24.0, 46.0, np.nan],
            "age": [40, 40, 40],
        }
    )
    shock = np.array([0.05, 0.01, 0.0])
    shocked = sc.apply_wage_cut_with_shock(persons, shock)
    removed = float(
        (
            (persons["employment_income"] - shocked["employment_income"])
            * persons["weight"]
        ).sum()
    )
    assert removed == pytest.approx(0.05 * 30000 * 2 + 0.01 * 20000 * 1)
    assert not shocked["displaced"].any()


def test_displacement_with_shock_expected_quota():
    rng_persons = pd.DataFrame(
        {
            "employment_income": np.full(200, 25000.0),
            "weight": np.full(200, 1.0),
            "sic_division": np.full(200, 46.0),
            "age": np.full(200, 40),
        }
    )
    shock = np.full(200, 0.10)
    draws = [
        sc.draw_displaced_with_shock(rng_persons, shock, seed=s).sum() for s in range(50)
    ]
    # Bernoulli expectation = 20; finite draws fluctuate around it.
    assert np.mean(draws) == pytest.approx(20.0, abs=1.0)


@requires_iot_data
def test_per_division_upstream_levels_sum_to_aggregate():
    """Spanning CPA groups must be SPLIT, not replicated, across divisions."""
    table = sc.upstream_sector_shocks("full_tariff")
    iot = sc.load_iot()
    f = sc.direct_final_demand_falls(iot, "full_tariff")
    dg = sc.upstream_output_falls(iot, f)
    aggregate = float((dg * iot.coe_ratio).sum())
    assert table["upstream_earnings_loss"].sum() == pytest.approx(aggregate, rel=1e-9)
    assert table["upstream_output_fall"].sum() == pytest.approx(float(dg.sum()), rel=1e-9)


@requires_iot_data
def test_iot_is_domestic_use_matrix():
    iot = sc.load_iot()
    # Domestic intermediate use of a product never exceeds its gross output.
    assert (iot.intermediate.sum(axis=1) <= iot.gross_output * (1 + 1e-6)).all()
