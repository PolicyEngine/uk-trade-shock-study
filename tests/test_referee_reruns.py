"""Checks on the referee-round energy reruns (public inputs and committed
aggregates only; no licensed microdata is read)."""

import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
sys.path.insert(0, str(REPO / "uk_trade_shock_study"))

NEW_MACRO_FILES = ("generated_defl.tex", "generated_lagreform.tex",
                   "generated_lcfs.tex")


def _macros(name):
    text = (RESULTS / name).read_text()
    return dict(re.findall(r"\\newcommand\{\\(Ss[A-Za-z]+)\}\{([^}]*)\}", text))


@pytest.mark.parametrize("name", NEW_MACRO_FILES + ("generated_grid.tex",))
def test_macro_names_are_letters_only_and_unique(name):
    text = (RESULTS / name).read_text()
    lines = [ln for ln in text.splitlines() if ln.startswith("\\newcommand")]
    names = re.findall(r"\\newcommand\{\\(Ss[A-Za-z]+)\}", text)
    assert len(names) == len(lines), "a macro name has a non-letter character"
    assert len(names) == len(set(names)), "duplicate macro"


def test_new_macros_do_not_collide_with_existing_files():
    new = set()
    for name in NEW_MACRO_FILES:
        new |= set(_macros(name))
    for f in RESULTS.glob("generated_*.tex"):
        if f.name in NEW_MACRO_FILES:
            continue
        assert not new & set(_macros(f.name)), f"collision with {f.name}"


def test_seasonal_variants_reproduce_existing_paths():
    grid = _macros("generated_grid.tex")
    # The central convention is consumption weights on the caps Ofgem set.
    assert grid["SsSeasCwActualFullRatePct"] == grid["SsGridPaperFullRatePct"]
    assert grid["SsSeasCwActualEbssRatePct"] == grid["SsGridPaperEbssRatePct"]
    assert grid["SsSeasCwActualEpgRatePct"] == grid["SsGridPaperEpgRatePct"]
    # Equal weights on the caps Ofgem set is the announced-path variant.
    assert grid["SsSeasEqActualCounterCap"] == grid["SsAnnCounterFyCap"]
    assert grid["SsSeasEqActualFullRatePct"] == grid["SsAnnFullRatePct"]
    assert grid["SsSeasEqActualEbssRatePct"] == grid["SsAnnEbssRatePct"]
    # Equal weights with the October cap held (the earlier headline) stays a
    # labelled variant on its own caps.
    assert grid["SsSeasEqHeldBaseCap"] == "1,208"
    assert grid["SsSeasEqHeldCounterCap"] == "2,760"
    assert grid["SsSeasEqHeldRealisedCap"] == "2,236"
    # The ranges span the four variants and contain the central value.
    for tag, central in (("FullRate", "SsGridPaperFullRatePct"),
                         ("EnergySpecificRate", "SsGridPaperEbssRatePct"),
                         ("EpgRate", "SsGridPaperEpgRatePct")):
        four = [float(grid[f"SsSeas{v}{k}"]) for v in
                ("EqHeld", "EqActual", "CwHeld", "CwActual")
                for k in ({"FullRate": "FullRatePct",
                           "EnergySpecificRate": "EbssRatePct",
                           "EpgRate": "EpgRatePct"}[tag],)]
        assert float(grid[f"Ss{tag}MinPct"]) == min(four)
        assert float(grid[f"Ss{tag}MaxPct"]) == max(four)
        assert min(four) <= float(grid[central]) <= max(four)


def test_seasonal_fy_means_follow_from_the_caps_and_month_shares():
    seas = json.loads((RESULTS / "seasonal_energy.json").read_text())
    share = seas["month_share_jan_to_dec"]
    assert sum(share) == pytest.approx(1.0, abs=1e-3)
    apr_sep, oct_dec, jan_mar = sum(share[3:9]), sum(share[9:12]), sum(share[0:3])
    v = seas["consumption_actual"]
    assert v["base_fy_mean_cap"] == pytest.approx(
        apr_sep * 1138 + (oct_dec + jan_mar) * 1277, abs=0.5)
    assert v["counterfactual_fy_mean_cap"] == pytest.approx(
        apr_sep * 1971 + oct_dec * 3549 + jan_mar * 4279, abs=1.0)
    assert v["realised_fy_mean_cap"] == pytest.approx(
        apr_sep * 1971 + (oct_dec + jan_mar) * 2500, abs=1.0)


def test_lag_reform_parameter_list_covers_the_main_benefits():
    from uprating_lag_reform import FACTOR, lag_parameters

    selected, excluded, _ = lag_parameters()
    assert FACTOR == pytest.approx(1.101 / 1.031)
    for must in (
        "gov.dwp.universal_credit.standard_allowance.amount.SINGLE_OLD",
        "gov.dwp.universal_credit.elements.child.amount",
        "gov.hmrc.child_benefit.amount.eldest",
        "gov.dwp.state_pension.basic_state_pension.amount",
        "gov.dwp.state_pension.new_state_pension.amount",
        "gov.dwp.pension_credit.guarantee_credit.minimum_guarantee.SINGLE",
        "gov.dwp.pip.daily_living.standard",
        "gov.dwp.dla.self_care.middle",
        "gov.dwp.attendance_allowance.higher",
        "gov.dwp.carers_allowance.rate",
        "gov.dwp.ESA.income.amount_over_25",
        "gov.dwp.JSA.income.amount_over_25",
        "gov.dwp.housing_benefit.allowances.single.older",
    ):
        assert must in selected, must
    assert "gov.dwp.carers_allowance.earnings_limit" in excluded
    for name, v in selected.items():
        assert v["counterfactual_2022"] == pytest.approx(v["value_2022"] * FACTOR)


def test_lcfs_output_holds_aggregates_only():
    res = json.loads((RESULTS / "lcfs_validation.json").read_text())
    for year in res["lcfs"].values():
        assert len(year["by_decile"]) == 10
        assert year["households"] > 1000
        # No per-record arrays: every list in the output is a decile table.
        for value in year.values():
            assert not isinstance(value, list) or len(value) == 10
