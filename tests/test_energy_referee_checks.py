"""Checks on the referee-round energy checks: the rebase below the paper's
with the maximum decile rate, the Council Tax Rebate (in the central
stack, with a without-rebate variant) and the Cost of Living Payments
outturn comparison (committed aggregates only; no
microdata is read)."""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
sys.path.insert(0, str(REPO / "uk_trade_shock_study"))

REF = json.loads((RESULTS / "energy_referee_checks.json").read_text())
GRID = json.loads((RESULTS / "grid_energy.json").read_text())
SS = json.loads((RESULTS / "second_stage_energy.json").read_text())
SS_MACROS = dict(re.findall(
    r"\\newcommand\{\\(Ss[A-Za-z]+)\}\{([^}]*)\}",
    (RESULTS / "generated_secondstage.tex").read_text()))
MACROS = dict(re.findall(r"\\newcommand\{\\(Ss[A-Za-z]+)\}\{([^}]*)\}",
                         (RESULTS / "generated_grid.tex").read_text()))


def _cell(cells, label, stack):
    return next(c for c in cells
                if c["rebase_label"] == label and c["stack"] == stack)


def test_closed_form_helpers():
    from energy_referee_checks import decile_rates, max_decile, rebase_at_hundred

    cash, shock, epg = [30.0, 10.0], [100.0, 100.0], 0.4
    r, d, rmax = rebase_at_hundred(epg, cash, shock)
    assert d == 1 and rmax == pytest.approx(0.5)
    assert decile_rates(epg, cash, shock, rmax)[0] == pytest.approx(100.0)
    assert decile_rates(epg, cash, shock, 1.0) == pytest.approx([70.0, 50.0])
    assert max_decile([33.8] * 10) == (33.8, None)       # ten-way tie
    assert max_decile([1.0, 3.0, 2.0]) == (3.0, 2)


def test_table_has_a_rebase_below_the_papers_and_agrees_with_the_pinned_grid():
    cells = REF["grid"]["table_cells"]
    paper = REF["grid"]["paper_rebase"]
    assert min(c["rebase"] for c in cells) < paper
    for c in GRID["cells"]:
        t = _cell(cells, c["rebase_label"], c["stack"])
        for k in ("counter_agg_gbp_bn", "rate_pct", "cushion_pct_d1",
                  "cushion_pct_d10"):
            assert t[k] == pytest.approx(c[k], abs=0.051), (c, k)
    table = (RESULTS / "table_grid.tex").read_text()
    assert table.index("0.600 &") < table.index("(paper)")
    assert "Max decile" in table


def test_max_decile_is_the_maximum_over_all_ten():
    for c in REF["grid"]["table_cells"] + REF["grid"]["without_ctr_cells"]:
        rates = c["cushion_pct_by_decile"]
        assert len(rates) == 10
        assert c["max_decile_pct"] == max(rates)
        assert c["max_decile_pct"] >= max(c["cushion_pct_d1"], c["cushion_pct_d10"])
        assert c["deciles_above_100"] == [i + 1 for i, x in enumerate(rates) if x > 100]
        if c["max_decile"] is None:
            assert len(set(rates)) == 1
        else:
            assert rates[c["max_decile"] - 1] == max(rates)


@pytest.mark.parametrize("stack", ["epg_ebss", "energy", "full", "full_noctr"])
def test_rebase_at_hundred_reproduces_from_the_stored_decile_sums(stack):
    from energy_referee_checks import decile_rates, rebase_at_hundred

    g = REF["grid"]
    cash, shock = g["cash_by_decile_gbp_bn"][stack], g["shock_unit_by_decile_gbp_bn"]
    r, d, rmax = rebase_at_hundred(g["epg_rate"], cash, shock)
    t = g["thresholds"][stack]
    assert d == t["first_decile"]
    assert rmax == pytest.approx(t["rebase_at_100"], abs=2e-4)
    rates = decile_rates(g["epg_rate"], cash, shock, rmax)
    assert rates.max() == pytest.approx(100.0, abs=1e-6)
    # and the stored cells follow from the same sums
    cells = g["without_ctr_cells"] if stack == "full_noctr" else g["table_cells"]
    for c in (c for c in cells if c["stack"] == stack):
        assert decile_rates(g["epg_rate"], cash, shock, c["rebase"]) == pytest.approx(
            c["cushion_pct_by_decile"], abs=0.06)
    above = [i + 1 for i, x in enumerate(r) if x >= g["paper_rebase"]]
    assert above == t["deciles_at_or_above_100_at_paper_rebase"]


def test_council_tax_rebate_is_in_the_central_stack():
    c = REF["council_tax_rebate"]
    nat = c["model_by_nation"]
    assert nat["northern_ireland"]["gbp_bn"] == 0
    assert sum(v["gbp_bn"] for v in nat.values()) == pytest.approx(
        c["model_gbp_bn"], abs=0.002)
    for v in nat.values():      # GBP 150 per recipient household
        assert v["gbp_bn"] * 1e3 == pytest.approx(150 * v["households_m"], abs=1.0)
    out = c["outturn"]["england_core"]
    assert out["url"].startswith("https://www.gov.uk/")
    assert c["england_model_over_outturn_pct"] == pytest.approx(
        100 * (nat["england"]["gbp_bn"] / out["spend_gbp_bn"] - 1), abs=0.1)

    # second stage: the instruments add up, and the rebate is one of them
    d = SS["discretionary"]
    assert d["ctr_gbp_bn"] == pytest.approx(c["model_gbp_bn"], abs=0.006)
    assert d["total_gbp_bn"] == pytest.approx(
        d["epg_gbp_bn"] + d["ebss_gbp_bn"] + d["ctr_gbp_bn"]
        + d["col_payments_gbp_bn"], abs=0.03)
    assert d["energy_specific_gbp_bn"] == pytest.approx(
        d["epg_gbp_bn"] + d["ebss_gbp_bn"] + d["ctr_gbp_bn"], abs=0.03)
    nc = d["without_ctr"]
    assert d["total_gbp_bn"] - nc["total_gbp_bn"] == pytest.approx(
        d["ctr_gbp_bn"], abs=0.02)
    # a flat payment only raises the rates, in every decile
    assert d["rate_pct_of_counterfactual"] > nc["rate_pct_of_counterfactual"]
    for row in SS["by_decile"]:
        assert row["cushioning_rate_pct"] > row["cushioning_rate_pct_without_ctr"]
        assert row["energy_specific_rate_pct"] > row["energy_specific_rate_pct_without_ctr"]
        assert row["cushioning_rate_pct"] > row["energy_specific_rate_pct"]


def test_grid_second_stage_and_macros_agree_on_the_central_stack():
    d = SS["discretionary"]
    energy = _cell(GRID["cells"], "paper", "energy")
    full = _cell(GRID["cells"], "paper", "full")
    assert energy["rate_pct"] == d["energy_specific_rate_pct"]
    assert full["rate_pct"] == d["rate_pct_of_counterfactual"]
    # paper-facing names: ...Ebss... is the energy-specific stack WITH the rebate
    assert MACROS["SsGridPaperEbssRatePct"] == f"{energy['rate_pct']:.1f}"
    assert MACROS["SsGridPaperEbssRatePct"] == SS_MACROS["SsEnergySpecificRatePct"]
    assert MACROS["SsGridPaperFullRatePct"] == SS_MACROS["SsDiscretionaryRatePct"]
    # ...NoCtr... is the same stack without it
    nc = d["without_ctr"]
    assert MACROS["SsGridPaperNoCtrEbssRatePct"] == f"{nc['energy_specific_rate_pct']:.1f}"
    assert MACROS["SsGridPaperNoCtrFullRatePct"] == f"{nc['rate_pct_of_counterfactual']:.1f}"
    assert SS_MACROS["SsNoCtrDiscretionaryRatePct"] == MACROS["SsGridPaperNoCtrFullRatePct"]
    assert SS_MACROS["SsNoCtrEnergySpecificRatePct"] == MACROS["SsGridPaperNoCtrEbssRatePct"]
    assert _cell(GRID["cells"], "paper", "epg_ebss")["rate_pct"] == nc["energy_specific_rate_pct"]
    # deciles above 100 per cent at the paper's rebase
    above = [r["decile"] for r in SS["by_decile"] if r["cushioning_rate_pct"] > 100]
    ref_full = _cell(REF["grid"]["table_cells"], "paper", "full")
    assert ref_full["deciles_above_100"] == above
    assert MACROS["SsDecilesAboveHundred"] == str(len(above))
    for d_ in above:
        assert str(d_) in MACROS["SsDecilesAboveHundredList"]
    # fine sweep order: EPG -> EBSS -> CTR -> CoL components
    order = [c["stack"] for c in GRID["cells_fine"][:6]]
    assert order == ["epg", "epg_ebss", "ctr", "col_mt", "col_pen", "col_dis"]


def test_no_macro_is_defined_twice_across_generated_files():
    names = []
    for f in sorted(RESULTS.glob("generated_*.tex")):
        names += re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", f.read_text())
    dup = sorted({n for n in names if names.count(n) > 1})
    assert not dup, dup


def test_cost_of_living_outturn_comparison():
    o = REF["cost_of_living_outturn"]
    comps = o["components"]
    assert set(comps) == {"means_tested", "pensioner", "disability"}
    for k, v in comps.items():
        src = o["outturn"][k]
        assert src["url"].startswith("https://www.gov.uk/government/publications/")
        assert v["outturn_gbp_bn"] == src["spend_gbp_bn"]
        assert v["model_gb_gbp_bn"] <= v["model_uk_gbp_bn"]
        assert v["model_uk_over_outturn_pct"] == pytest.approx(
            100 * (v["model_uk_gbp_bn"] / v["outturn_gbp_bn"] - 1), abs=0.1)
    mt = o["outturn"]["means_tested"]
    assert mt["spend_gbp_bn"] == pytest.approx(
        mt["first_payment"]["spend_gbp_bn"] + mt["second_payment"]["spend_gbp_bn"])
    assert o["total"]["outturn_gbp_bn"] == pytest.approx(
        sum(v["outturn_gbp_bn"] for v in comps.values()))
    assert o["total"]["model_uk_gbp_bn"] == pytest.approx(
        sum(v["model_uk_gbp_bn"] for v in comps.values()), abs=0.002)
    # scaling the model to a lower outturn total lowers the full-stack rate
    s = o["outturn_scaled"]
    assert s["full_rate_pct_col_scaled"] < s["full_rate_pct_central"]
    assert MACROS["SsColOutturnCalFullRatePct"] == f"{s['full_rate_pct_col_scaled']:.1f}"


def test_new_macros_are_emitted():
    for name in ("SsGridMaxDecileRebaseAtHundred", "SsGridLowFullRatePct",
                 "SsGridPaperFullMaxDecilePct", "SsCtrOutturnEnglandBn",
                 "SsDecilesAboveHundredList", "SsGridNoCtrMaxDecileRebaseAtHundred",
                 "SsColOutturnTotalBn", "SsColOutturnDisExcessPct"):
        assert name in MACROS, name
    assert float(MACROS["SsGridMaxDecileRebaseAtHundred"]) == pytest.approx(
        REF["grid"]["thresholds"]["full"]["rebase_at_100"], abs=6e-4)
    assert np.isfinite(float(MACROS["SsGridLowFullRatePct"]))
    for name in ("SsCtrBn", "SsNoCtrDiscretionaryRatePct",
                 "SsNoCtrEnergySpecificRatePct", "SsNoCtrCushionPctDecOne",
                 "SsEnergySpecificPctDecTen"):
        assert name in SS_MACROS, name
