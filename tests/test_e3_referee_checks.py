"""Referee checks on E3: the part-year spell table, the annualised New
Style JSA state, the monthly-state mix, the ASHE table reader and the macro
emitters. Pure functions only -- no microdata."""

import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis import ashe_validation, emit_e3_macros
from analysis.sensitivity_duration_takeup import NET, NET_BEFORE_FLOOR, monthly_state_mix, summarise
from tariff_pipeline.runner import _hbai_lists
from tariff_pipeline.shocks import (
    PRESETS,
    affected_mask,
    apply_partial_year_spell,
    new_style_jsa_amounts,
)

ROOT = Path(__file__).resolve().parent.parent
RATES = dict(weekly_rate_18_24=75.65, weekly_rate_25_plus=95.55, weekly_lower_earnings_limit=125.0)


def _persons():
    return pd.DataFrame(
        {
            "person_id": [1, 2, 3, 4],
            "age": [30, 45, 52, 61],
            "employment_income": [30_000.0, 0.0, 18_000.0, 40_000.0],
            "sic_division": [29, 29, 24, 47],
            "weight": [100.0, 100.0, 50.0, 80.0],
        }
    )


# --------------------------------------------------------------------------
# part-year spell (annual averaging)
# --------------------------------------------------------------------------


def test_partial_year_spell_scales_earnings_and_hours_and_keeps_status():
    scenario = dataclasses.replace(PRESETS["full_tariff_displacement"], new_style_jsa=True)
    affected = np.array([True, False, False, False])
    hours = np.array([1900.0, 0.0, 1000.0, 1800.0])
    table = apply_partial_year_spell(_persons(), scenario, affected, 0.5, hours, seed=2)
    assert table["employment_income"].tolist() == [15_000.0, 0.0, 18_000.0, 40_000.0]
    assert table["hours_worked"].tolist() == [950.0, 0.0, 1000.0, 1800.0]
    # nobody is flagged displaced: the worker is in work for part of the year
    assert not table["displaced"].any() and not table["inactive"].any()
    assert table["new_style_jsa_recipient"].tolist() == affected.tolist()
    assert affected_mask(table).tolist() == affected.tolist()
    assert table.attrs["seed"] == 2 and table.attrs["new_style_jsa"] is True


@pytest.mark.parametrize("fraction, weeks", [(0.5, 26.0), (0.25, 13.0), (0.75, 26.0)])
def test_partial_year_spell_pays_jsa_for_the_weeks_inside_the_spell(fraction, weeks):
    scenario = dataclasses.replace(PRESETS["full_tariff_displacement"], new_style_jsa=True)
    table = apply_partial_year_spell(
        _persons(), scenario, np.array([True, False, False, False]), fraction, np.zeros(4)
    )
    assert table.attrs["new_style_jsa_weeks"] == pytest.approx(weeks)


def test_partial_year_spell_rejects_bad_inputs():
    scenario = PRESETS["full_tariff_displacement"]
    mask = np.array([True, False, False, False])
    for fraction in (0.0, 1.0, 1.5):
        with pytest.raises(ValueError, match="spell_fraction"):
            apply_partial_year_spell(_persons(), scenario, mask, fraction, np.zeros(4))
    with pytest.raises(ValueError, match="positive earnings"):
        apply_partial_year_spell(
            _persons(), scenario, np.array([False, True, False, False]), 0.5, np.zeros(4)
        )
    with pytest.raises(ValueError, match="displacement margin"):
        apply_partial_year_spell(
            _persons(), PRESETS["full_tariff_wage_cut"], mask, 0.5, np.zeros(4)
        )


def test_annualised_jsa_state_needs_an_explicit_ceiling():
    args = ([True], [40], [30_000], [False])
    with pytest.raises(ValueError, match="26 weeks"):
        new_style_jsa_amounts(*args, **RATES, weeks=52)
    assert new_style_jsa_amounts(*args, **RATES, weeks=52, max_weeks=52) == pytest.approx(
        [52 * 95.55]
    )
    with pytest.raises(ValueError):
        new_style_jsa_amounts(*args, **RATES, weeks=60, max_weeks=60)


# --------------------------------------------------------------------------
# monthly-state mix
# --------------------------------------------------------------------------


def _changes(rng, n=3):
    """Household changes obeying the HBAI identity (no zero floor)."""
    adds, subtracts = _hbai_lists()
    out = {name: rng.normal(0, 100, n) for name in (*adds, *subtracts)}
    out["employment_income"] = -rng.uniform(10_000, 40_000, n)
    before = sum(out[a] for a in adds) - sum(out[s] for s in subtracts)
    out[NET_BEFORE_FLOOR] = before
    out[NET] = before.copy()
    out["gov_balance"] = rng.normal(0, 100, n)
    return out


def test_monthly_state_mix_scales_every_line_but_income_tax():
    rng = np.random.default_rng(0)
    out_of_work, annual = _changes(rng), _changes(rng)
    mixed = monthly_state_mix(out_of_work, annual, 0.5)
    assert mixed["income_tax"] == pytest.approx(annual["income_tax"])
    for name in ("employment_income", "universal_credit", "national_insurance"):
        assert mixed[name] == pytest.approx(0.5 * out_of_work[name])
    # net income moves one for one (negatively) with the income-tax swap
    swap = annual["income_tax"] - 0.5 * out_of_work["income_tax"]
    assert mixed[NET] == pytest.approx(0.5 * out_of_work[NET] - swap)
    # and the HBAI identity still holds, so the cushion split still sums
    row = summarise(mixed, np.ones(3), -0.5 * out_of_work["employment_income"].sum())
    assert row["gross_vs_target"] == pytest.approx(1.0)
    assert sum(row["share_of_gross_loss_all_lines"].values()) == pytest.approx(
        row["cushioning_rate"]
    )


def test_monthly_state_mix_with_no_tax_difference_is_the_scaled_state():
    out_of_work = _changes(np.random.default_rng(1))
    annual = {"income_tax": 0.25 * out_of_work["income_tax"]}
    mixed = monthly_state_mix(out_of_work, annual, 0.25)
    for name, values in out_of_work.items():
        assert mixed[name] == pytest.approx(0.25 * values)


# --------------------------------------------------------------------------
# ASHE
# --------------------------------------------------------------------------


def test_share_above_brackets_and_interpolates():
    percentiles = {10: 10_000.0, 50: 30_000.0, 75: 50_000.0, 80: None, 90: 90_000.0}
    out = ashe_validation.share_above(percentiles, 40_000.0)
    assert out["lower_bound"] == pytest.approx(0.25)
    assert out["upper_bound"] == pytest.approx(0.50)
    assert out["interpolated"] == pytest.approx(1 - 0.625)
    # a suppressed percentile is skipped, not read as zero
    assert ashe_validation.share_above(percentiles, 70_000.0)["upper_bound"] == pytest.approx(0.25)
    assert ashe_validation.share_above(percentiles, 95_000.0) is None
    assert ashe_validation.share_above(percentiles, 5_000.0) is None


@pytest.mark.parametrize("edition", list(ashe_validation.EDITIONS))
def test_ashe_table_reads_and_hashes(edition):
    meta = ashe_validation.EDITIONS[edition]
    path = ashe_validation.RAW / meta["file"]
    if not path.exists():
        pytest.skip(f"{path.name} not present (data/ is not distributed)")
    table = ashe_validation.read_ashe(path, meta["sha256"])
    assert set(range(10, 33)) <= set(table)
    assert table[12]["mean"] is None  # tobacco is suppressed
    assert 40_000 < table[29]["mean"] < 70_000
    assert table[29]["percentiles"][50] == table[29]["median"]
    with pytest.raises(RuntimeError, match="SHA256"):
        ashe_validation.read_ashe(path, "0" * 64)


# --------------------------------------------------------------------------
# stored results and macros
# --------------------------------------------------------------------------


def _result(name):
    path = ROOT / "results" / f"{name}.json"
    if not path.exists():
        pytest.skip(f"{path.name} not built")
    return json.loads(path.read_text())


@pytest.mark.parametrize("name", ["full_tariff_wage_cut", "full_tariff_obr_low_wage_cut"])
def test_wage_cut_uses_the_displacement_take_up_treatment(name):
    wage = _result(name)
    disp = _result("full_tariff_displacement")
    assert wage["design"] == "exact_takeup_expectation"
    assert wage["n_draws"] == 0 and wage["draws"] == []
    assert wage["uc_takeup"] == disp["uc_takeup"] == 0.8
    nodes = wage["takeup_nodes"]
    mixed = 0.8 * nodes["1"]["net_disposable_loss"] + 0.2 * nodes["0"]["net_disposable_loss"]
    assert wage["net_disposable_loss"] == pytest.approx(mixed, rel=1e-9)
    assert wage["households_with_two_redrawn_units"] == 0
    shares = wage["cushion_component_shares"]["share_of_gross_loss"]
    # single-precision model arithmetic: the split closes to cushion_components' tolerance
    assert sum(shares.values()) == pytest.approx(wage["cushioning_rate"], abs=1e-5)


def test_sensitivity_file_is_internally_consistent():
    sens = _result("e3_sensitivity")
    disp = _result("full_tariff_displacement")
    takeup = sens["uc_takeup"]
    assert takeup["0.8"]["cushioning_rate"] == pytest.approx(disp["cushioning_rate"], abs=1e-6)
    # exact linear mix: the central net loss is the midpoint of 0.6 and 1.0
    assert takeup["0.8"]["net_disposable_loss"] == pytest.approx(
        0.5 * (takeup["0.6"]["net_disposable_loss"] + takeup["1.0"]["net_disposable_loss"]),
        rel=1e-9,
    )
    weeks = sens["new_style_jsa_weeks"]
    assert weeks["0"]["cushioning_rate"] < weeks["13"]["cushioning_rate"] < weeks["26"]["cushioning_rate"]
    for label, fraction in (("6m", 0.5), ("3m", 0.25)):
        spell = sens["duration"][label]
        for method in ("annual_averaging", "monthly_state_mix"):
            row = spell[method]
            assert row["gross_vs_target"] == pytest.approx(1.0, abs=1e-3)
            assert row["gross_earnings_loss"] == pytest.approx(
                fraction * disp["target_gross_earnings_loss"], rel=1e-3
            )
        # the two approximations share the income-tax line by construction
        assert spell["monthly_state_mix"]["share_of_gross_loss"]["income_tax"] == pytest.approx(
            spell["annual_averaging"]["share_of_gross_loss"]["income_tax"], abs=1e-6
        )


def test_new_macro_families_are_emitted_with_letter_only_names():
    ashe, sens = _result("e3_ashe"), _result("e3_sensitivity")
    macros = {**emit_e3_macros.ashe_macros(ashe), **emit_e3_macros.sensitivity_macros(sens)}
    assert all(name.isalpha() for name in macros)
    for name in (
        "EThreeAsheModelMeanK", "EThreeAsheMeanK", "EThreeAsheModelMedianK",
        "EThreeAsheMedianK", "EThreeHigherRateSharePct",
        "EThreeSensTakeupLowRatePct", "EThreeSensTakeupHighRatePct",
        "EThreeSensJsaThirteenRatePct", "EThreeSensSixMonthAnnualRatePct",
        "EThreeSensSixMonthMixRatePct",
    ):
        assert name in macros
    generated = (ROOT / "results" / "generated_e3.tex").read_text()
    for name, value in macros.items():
        assert f"\\newcommand{{\\{name}}}{{{value}}}" in generated
    # the appendix sentence that uses it must never read "averages 0 draws"
    assert "\\newcommand{\\EThreeWageDraws}{the all-claim and none-claim}" in generated


# --------------------------------------------------------------------------
# uniform benchmark, decile income shares, industry table
# --------------------------------------------------------------------------


def test_uniform_shock_overrides_the_sector_pattern():
    from tariff_pipeline.shocks import TradeShockScenario, _person_shock, apply_shocks

    scenario = TradeShockScenario("u", "full_tariff", "wage_cut", uniform_shock=0.01)
    persons = _persons()
    assert _person_shock(persons, scenario).tolist() == [0.01] * 4
    table = apply_shocks(persons, scenario)
    # every employee, including the one outside the exposed divisions (SIC 47)
    assert table["employment_income"].tolist() == pytest.approx([29_700.0, 0.0, 17_820.0, 39_600.0])
    for bad in (0.0, 1.0, -0.1):
        with pytest.raises(ValueError, match="uniform_shock"):
            TradeShockScenario("u", "full_tariff", "wage_cut", uniform_shock=bad)
    assert PRESETS["full_tariff_wage_cut"].uniform_shock is None


def test_household_deciles_are_household_weighted():
    from tariff_pipeline.expectation import household_deciles

    income = np.arange(20, dtype=float)[::-1].copy()
    decile = household_deciles(income, np.ones(20))
    assert decile.min() == 1 and decile.max() == 10
    # the price episodes' convention: a household exactly on a boundary of
    # cumulative weight goes to the decile above
    assert np.bincount(decile)[1:].tolist() == [1, 2, 2, 2, 2, 2, 2, 2, 2, 3]
    assert (np.diff(decile[np.argsort(income)]) >= 0).all()
    assert decile[np.argmin(income)] == 1 and decile[np.argmax(income)] == 10
    # a heavy household at the bottom fills the lowest deciles on its own
    heavy = household_deciles(np.array([1.0, 2.0, 3.0]), np.array([8.0, 1.0, 1.0]))
    assert heavy.tolist() == [9, 10, 10]


def test_loss_shares_by_decile_are_ratios_of_weighted_sums():
    from tariff_pipeline.expectation import loss_shares_by_decile

    expected = {
        "employment_income": np.array([-100.0, -300.0]),
        "hbai_household_net_income": np.array([-60.0, -150.0]),
    }
    w = np.array([2.0, 1.0])
    out = loss_shares_by_decile(
        expected, w, np.array([1, 10]), np.array([1_000.0, 6_000.0]), np.array([900.0, 3_000.0])
    )
    assert out["1"]["gross_loss_share_of_income"] == pytest.approx(200 / 2_000)
    assert out["1"]["net_loss_share_of_income"] == pytest.approx(120 / 2_000)
    assert out["10"]["net_loss_share_of_consumption"] == pytest.approx(150 / 3_000)
    assert out["10"]["cushioning_rate"] == pytest.approx(0.5)
    assert out["all"]["net_loss_share_of_income"] == pytest.approx(270 / 8_000)
    assert out["1"]["share_of_total_gross_loss"] + out["10"]["share_of_total_gross_loss"] == pytest.approx(1)
    assert out["5"]["gross_loss"] == 0 and out["5"]["cushioning_rate"] is None


def test_uniform_benchmark_file_imposes_the_tariff_loss():
    unif = _result("e3_uniform")
    disp = _result("full_tariff_displacement")
    wage = _result("full_tariff_wage_cut")
    rows = unif["rows"]
    assert rows["tariff_displacement"]["cushioning_rate"] == pytest.approx(disp["cushioning_rate"], abs=1e-6)
    assert rows["tariff_wage_cut"]["cushioning_rate"] == pytest.approx(wage["cushioning_rate"], abs=1e-6)
    for key in ("uniform_wage_cut", "uniform_displacement"):
        assert rows[key]["target_gross_earnings_loss"] == pytest.approx(
            disp["target_gross_earnings_loss"], rel=1e-9
        )
        assert rows[key]["gross_earnings_loss"] == pytest.approx(
            disp["target_gross_earnings_loss"], rel=1e-3
        )
        assert rows[key]["exposed_records"] == unif["employee_records"]
    assert unif["uniform_shock"] == pytest.approx(
        unif["gross_earnings_loss"] / unif["employee_earnings"]
    )


def test_income_share_file_adds_up():
    income = _result("e3_income_shares")
    disp = _result("full_tariff_displacement")
    for key, table in income["shares"].items():
        deciles = [table[str(d)] for d in range(1, 11)]
        assert sum(r["gross_loss"] for r in deciles) == pytest.approx(table["all"]["gross_loss"])
        assert sum(r["net_loss"] for r in deciles) == pytest.approx(table["all"]["net_loss"])
        assert sum(r["share_of_total_gross_loss"] for r in deciles) == pytest.approx(1.0)
    assert income["shares"]["tariff_displacement"]["all"]["net_loss"] == pytest.approx(
        disp["net_disposable_loss"], rel=1e-9
    )
    households = [income["decile_baseline"][str(d)]["households"] for d in range(1, 11)]
    assert max(households) / min(households) < 1.02  # household-weighted tenths


def test_industry_table_rows_sum_to_the_scenario():
    from analysis import e3_industry_table

    rows = e3_industry_table.build_rows(_result("e3_ashe"))
    assert len(rows) == 23 and set(r["division"] for r in rows) == set(range(10, 33))
    assert sum(r["loss_share"] for r in rows) == pytest.approx(1.0)
    assert [r["gross_loss"] for r in rows] == sorted((r["gross_loss"] for r in rows), reverse=True)
    for r in rows:
        assert r["shock"] == pytest.approx(r["tariff_rate"] * r["us_export_share"])
    total = e3_industry_table.pool(rows, "all")
    assert total["gross_loss"] == pytest.approx(
        _result("full_tariff_displacement")["target_gross_earnings_loss"], rel=1e-9
    )
    tex = (ROOT / "results" / "table_e3_industry.tex").read_text()
    assert tex.count("\\\\") == 2 + e3_industry_table.TOP_ROWS + 2
