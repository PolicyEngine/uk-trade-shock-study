"""Post-shock UC take-up: the benefit-unit flag rate, the re-draw SCOPE, and
the redraw-set diagnostic.

The scope tests exist because a take-up grid run over an EMPTY re-draw set is
silently inert: every take-up value returns a bit-identical cushioning rate,
which reads as "take-up does not matter" when it actually means "no benefit
unit was re-drawn". The diagnostic is what makes that distinguishable, so it
is pinned here.
"""

import numpy as np
import pandas as pd
import pytest

from uk_trade_shock_study.shocks import (
    DEFAULT_UC_TAKEUP_SCOPE,
    UC_TAKEUP_SCOPES,
    TradeShockScenario,
    _baseline_flag_values_and_rate,
    apply_shocks,
    redraw_uc_takeup,
    uc_takeup_redraw_diagnostic,
)


class _WeightedFlags:
    values = np.array([True, False, False])

    def mean(self):
        return 0.75


class _BenefitUnitSimulation:
    def calculate(self, variable, period=None, map_to=None):
        assert variable == "would_claim_uc"
        assert period == 2026
        assert map_to == "benunit"
        return _WeightedFlags()


def test_baseline_flag_rate_preserves_benefit_unit_weights():
    values, rate = _baseline_flag_values_and_rate(_BenefitUnitSimulation(), period=2026)

    np.testing.assert_array_equal(values, [True, False, False])
    assert values.dtype == np.dtype(bool)
    assert rate == pytest.approx(0.75)
    assert rate != pytest.approx(values.mean())


# ---------------------------------------------------------------------------
# Synthetic fixtures for the re-draw scope (no licensed data)
# ---------------------------------------------------------------------------


class _ScopeSim:
    """Sim stub: persons map to benunits in fixed-size blocks.

    ``benunit_weight`` is served only when ``weights`` is supplied, so the
    diagnostic's weighted-count path and its degrade-to-None path are both
    reachable.
    """

    def __init__(self, n_persons, per_benunit=2, baseline_flag=None,
                 potential_uc=None, weights=None):
        self.n = n_persons
        self.k = per_benunit
        self.n_bu = (n_persons + per_benunit - 1) // per_benunit
        self.flag = (
            np.zeros(self.n_bu, dtype=bool) if baseline_flag is None else baseline_flag
        )
        self.stored = {}
        self.potential_uc = (
            np.ones(self.n_bu) if potential_uc is None else np.asarray(potential_uc, float)
        )
        self.weights = None if weights is None else np.asarray(weights, float)

    def calculate(self, var, period=None, map_to=None):
        import types

        if var == "would_claim_uc":
            values = self.stored.get(var, self.flag)
        elif var == "employment_income":
            values = np.ones(self.n)
        elif var == "universal_credit":
            claim = self.stored.get("would_claim_uc", self.flag)
            values = self.potential_uc * np.asarray(claim)
        elif var == "benunit_weight":
            if self.weights is None:
                raise ValueError("benunit_weight is not available in this simulation")
            values = self.weights
        else:
            values = np.zeros(self.n)
        return types.SimpleNamespace(values=np.asarray(values))

    def set_input(self, var, period, values):
        self.stored[var] = np.asarray(values)

    def _invalidate_all_caches(self):
        return None

    def map_result(self, values, source, target):
        assert (source, target) == ("person", "benunit")
        out = np.zeros(self.n_bu, dtype=float)
        for i, v in enumerate(np.asarray(values, dtype=float)):
            out[i // self.k] += v
        return out


def _scope_table(n, affected_idx, uc_takeup=0.8, seed=0, scope=None):
    table = pd.DataFrame(
        {
            "employment_income": np.ones(n),
            "displaced": np.zeros(n, dtype=bool),
            "inactive": np.zeros(n, dtype=bool),
            "reallocated": np.zeros(n, dtype=bool),
        }
    )
    table.loc[list(affected_idx), "displaced"] = True
    table.attrs["uc_takeup"] = uc_takeup
    table.attrs["seed"] = seed
    if scope is not None:
        table.attrs["uc_takeup_scope"] = scope
    return table


def _pair(n, baseline, potential_post, potential_base, **kwargs):
    """(shocked sim, baseline sim) sharing a baseline claiming flag."""
    return (
        _ScopeSim(n, 2, np.asarray(baseline).copy(), potential_uc=potential_post, **kwargs),
        _ScopeSim(n, 2, np.asarray(baseline).copy(), potential_uc=potential_base),
    )


# ---------------------------------------------------------------------------
# Scope option and its validation
# ---------------------------------------------------------------------------


def test_uc_takeup_scope_defaults_to_new_entitlement_and_is_carried_on_attrs():
    """The default MUST stay new_entitlement: every stored result uses it."""
    assert DEFAULT_UC_TAKEUP_SCOPE == "new_entitlement"
    assert set(UC_TAKEUP_SCOPES) == {"new_entitlement", "all_entitled"}

    scenario = TradeShockScenario("t", "full_tariff", "displacement")
    assert scenario.uc_takeup_scope == "new_entitlement"

    persons = pd.DataFrame(
        {
            "employment_income": np.linspace(10_000, 60_000, 40),
            "sic_division": np.full(40, 10),
            "age": np.full(40, 40),
            "weight": np.ones(40),
        }
    )
    table = apply_shocks(persons, scenario, seed=2)
    assert table.attrs["uc_takeup_scope"] == "new_entitlement"

    scoped = apply_shocks(
        persons,
        TradeShockScenario("t", "full_tariff", "displacement", uc_takeup_scope="all_entitled"),
        seed=2,
    )
    assert scoped.attrs["uc_takeup_scope"] == "all_entitled"


@pytest.mark.parametrize("bad", ["", "all", "newly_entitled", "ALL_ENTITLED", None, 1])
def test_uc_takeup_scope_rejects_unknown_values_on_the_scenario(bad):
    with pytest.raises(ValueError, match="uc_takeup_scope"):
        TradeShockScenario("t", "full_tariff", "displacement", uc_takeup_scope=bad)


def test_uc_takeup_scope_rejects_unknown_values_in_the_redraw():
    """A typo in an explicit argument must fail loudly, not fall back."""
    n = 4
    sim, base_sim = _pair(n, [False, False], [100.0, 100.0], [0.0, 0.0])
    table = _scope_table(n, [0, 2], uc_takeup=1.0)
    with pytest.raises(ValueError, match="uc_takeup_scope"):
        redraw_uc_takeup(sim, base_sim, table, 2026, uc_takeup_scope="everyone")


def test_uc_takeup_scope_rejects_unknown_value_carried_on_attrs():
    n = 4
    sim, base_sim = _pair(n, [False, False], [100.0, 100.0], [0.0, 0.0])
    table = _scope_table(n, [0, 2], uc_takeup=1.0, scope="everyone")
    with pytest.raises(ValueError, match="uc_takeup_scope"):
        redraw_uc_takeup(sim, base_sim, table, 2026)


# ---------------------------------------------------------------------------
# new_entitlement reproduces the published behaviour; all_entitled binds
# ---------------------------------------------------------------------------


def _three_unit_fixture():
    """Benunit 0 newly entitled, 1 already entitled, 2 never entitled."""
    n = 6
    baseline = np.array([False, False, False])
    potential_post = np.array([100.0, 100.0, 0.0])
    potential_base = np.array([0.0, 100.0, 0.0])
    return n, baseline, potential_post, potential_base


def test_new_entitlement_scope_reproduces_prior_behaviour():
    """Only the NEWLY entitled unit is re-drawn — the published convention."""
    n, baseline, post, base = _three_unit_fixture()
    for scope in (None, "new_entitlement"):
        sim, base_sim = _pair(n, baseline, post, base)
        table = _scope_table(n, [0, 2, 4], uc_takeup=1.0, scope=scope)
        out = redraw_uc_takeup(
            sim, base_sim, table, 2026, baseline_potential_award=base
        )
        # benunit 1 was already entitled: it keeps its baseline False flag
        # even at take-up 1.0, exactly as before the scope option existed.
        np.testing.assert_array_equal(out, np.array([True, False, False]))


def test_all_entitled_scope_redraws_already_entitled_changed_units():
    """The binding margin: baseline-entitled changed units ARE re-drawn."""
    n, baseline, post, base = _three_unit_fixture()
    sim, base_sim = _pair(n, baseline, post, base)
    table = _scope_table(n, [0, 2, 4], uc_takeup=1.0, scope="all_entitled")
    out = redraw_uc_takeup(sim, base_sim, table, 2026, baseline_potential_award=base)
    # benunit 1 (already entitled at baseline) now claims; benunit 2 has no
    # entitlement post-shock so is still never touched.
    np.testing.assert_array_equal(out, np.array([True, True, False]))
    np.testing.assert_array_equal(sim.stored["would_claim_uc"], out)


def test_all_entitled_scope_still_spares_unchanged_and_unentitled_units():
    n = 8
    baseline = np.array([True, False, True, False])
    post = np.array([100.0, 100.0, 100.0, 0.0])
    sim, base_sim = _pair(n, baseline, post, post)
    # only benunits 0 and 1 contain a changed person
    table = _scope_table(n, [0, 2], uc_takeup=0.0, scope="all_entitled")
    out = redraw_uc_takeup(sim, base_sim, table, 2026, baseline_potential_award=post)
    # 0 and 1 re-drawn at take-up 0 -> False; 2 and 3 keep their baseline draw
    np.testing.assert_array_equal(out, np.array([False, False, True, False]))


def test_scope_does_not_perturb_the_random_stream():
    """Switching scope changes WHICH units are re-drawn, never the draws.

    The two scopes must therefore agree exactly on every unit that both of
    them re-draw — otherwise the scope comparison would confound a claiming
    change with a different set of random numbers.
    """
    n = 400
    n_bu = n // 2
    rng = np.random.default_rng(11)
    baseline = rng.random(n_bu) < 0.55
    post = np.full(n_bu, 100.0)
    # half the units were already entitled at baseline
    base_award = np.where(np.arange(n_bu) % 2 == 0, 0.0, 100.0)
    affected = list(range(0, n, 2))

    outs = {}
    for scope in ("new_entitlement", "all_entitled"):
        sim, base_sim = _pair(n, baseline, post, base_award)
        outs[scope] = redraw_uc_takeup(
            sim,
            base_sim,
            _scope_table(n, affected, uc_takeup=0.8, seed=7, scope=scope),
            2026,
            baseline_potential_award=base_award,
        )
    newly = base_award <= 1e-8
    # units re-drawn under BOTH scopes get identical flags
    np.testing.assert_array_equal(outs["new_entitlement"][newly], outs["all_entitled"][newly])
    # and the extra units are exactly the baseline-entitled ones
    np.testing.assert_array_equal(outs["new_entitlement"][~newly], baseline[~newly])
    assert not np.array_equal(outs["all_entitled"][~newly], baseline[~newly])


# ---------------------------------------------------------------------------
# The redraw-set diagnostic
# ---------------------------------------------------------------------------


def test_redraw_diagnostic_counts_the_units_actually_redrawn():
    n, baseline, post, base = _three_unit_fixture()
    sim, base_sim = _pair(n, baseline, post, base, weights=[10.0, 20.0, 30.0])
    table = _scope_table(n, [0, 2, 4], uc_takeup=1.0, scope="all_entitled")
    redraw_uc_takeup(sim, base_sim, table, 2026, baseline_potential_award=base)

    diag = uc_takeup_redraw_diagnostic(table)
    assert diag is not None
    assert diag["uc_takeup_scope"] == "all_entitled"
    assert diag["uc_takeup"] == 1.0
    assert diag["n_benunits"] == 3
    assert diag["n_changed_benunits"] == 3
    assert diag["n_entitled_changed_benunits"] == 2
    assert diag["n_redrawn"] == 2
    assert diag["weights_available"] is True
    assert diag["weighted_redrawn"] == pytest.approx(30.0)  # units 0 and 1
    assert diag["weighted_changed_benunits"] == pytest.approx(60.0)
    # also reachable from the simulation object
    assert getattr(sim, "_uc_takeup_redraw") == diag


def test_redraw_diagnostic_reports_an_empty_redraw_set():
    """THE defect this diagnostic exists to expose.

    Every changed-and-entitled unit was already entitled at baseline, so the
    new_entitlement re-draw set is empty and the take-up grid is inert: the
    returned flags are bit-identical across take-up values. Without
    ``n_redrawn`` that is indistinguishable from a robustness result.
    """
    n = 6
    baseline = np.array([True, False, False])
    post = np.array([100.0, 100.0, 0.0])
    base = np.array([100.0, 100.0, 0.0])

    flags = {}
    for takeup in (0.55, 0.80, 1.00):
        sim, base_sim = _pair(n, baseline, post, base)
        table = _scope_table(n, [0, 2, 4], uc_takeup=takeup, scope="new_entitlement")
        flags[takeup] = redraw_uc_takeup(
            sim, base_sim, table, 2026, baseline_potential_award=base
        )
        diag = uc_takeup_redraw_diagnostic(table)
        assert diag["n_redrawn"] == 0
        assert diag["n_entitled_changed_benunits"] == 2

    # inert: identical for every take-up value
    for takeup in (0.80, 1.00):
        np.testing.assert_array_equal(flags[takeup], flags[0.55])
    np.testing.assert_array_equal(flags[1.00], baseline)

    # the same fixture under all_entitled is NOT inert
    sim, base_sim = _pair(n, baseline, post, base)
    table = _scope_table(n, [0, 2, 4], uc_takeup=1.0, scope="all_entitled")
    out = redraw_uc_takeup(sim, base_sim, table, 2026, baseline_potential_award=base)
    assert uc_takeup_redraw_diagnostic(table)["n_redrawn"] == 2
    assert not np.array_equal(out, baseline)


def test_redraw_diagnostic_recorded_when_nothing_changed():
    n = 6
    sim, base_sim = _pair(n, [False, False, False], [100.0] * 3, [0.0] * 3)
    table = _scope_table(n, [], uc_takeup=0.8)
    redraw_uc_takeup(sim, base_sim, table, 2026)

    diag = uc_takeup_redraw_diagnostic(table)
    assert diag["n_changed_benunits"] == 0
    assert diag["n_redrawn"] == 0
    assert "would_claim_uc" not in sim.stored


def test_redraw_diagnostic_degrades_when_weights_are_unavailable():
    """A sim without benunit_weight must still produce counts, not raise."""
    n, baseline, post, base = _three_unit_fixture()
    sim, base_sim = _pair(n, baseline, post, base)  # no weights
    table = _scope_table(n, [0, 2, 4], uc_takeup=1.0, scope="all_entitled")
    redraw_uc_takeup(sim, base_sim, table, 2026, baseline_potential_award=base)

    diag = uc_takeup_redraw_diagnostic(table)
    assert diag["weights_available"] is False
    assert diag["weighted_redrawn"] is None
    assert diag["n_redrawn"] == 2


# ---------------------------------------------------------------------------
# How analysis/referee_fixes.py reports the diagnostic
#
# These need no licensed data: they exercise the pure summarising helpers that
# turn per-seed diagnostics into the JSON the macros are read out of.
# ---------------------------------------------------------------------------


def _diagnostic(**overrides):
    diag = {
        "uc_takeup_scope": "new_entitlement",
        "uc_takeup": 0.8,
        "n_benunits": 3,
        "n_changed_benunits": 3,
        "n_entitled_changed_benunits": 2,
        "n_redrawn": 2,
        "weights_available": True,
        "weighted_redrawn": 30.0,
        "weighted_changed_benunits": 60.0,
    }
    diag.update(overrides)
    return diag


def test_summarised_diagnostic_reports_a_genuinely_empty_redraw_set():
    from analysis.referee_fixes import _summarise_diagnostics

    out = _summarise_diagnostics([_diagnostic(n_redrawn=0) for _ in range(3)])
    assert out["n_redrawn_max"] == 0
    assert out["redraw_set_empty_in_every_seed"] is True
    assert out["uc_takeup_scope"] == "new_entitlement"

    live = _summarise_diagnostics([_diagnostic(n_redrawn=n) for n in (0, 2, 5)])
    assert live["redraw_set_empty_in_every_seed"] is False
    assert live["n_redrawn_min"] == 0
    assert live["n_redrawn_max"] == 5


@pytest.mark.parametrize(
    "missing",
    ["n_redrawn", "n_entitled_changed_benunits", "n_changed_benunits", "uc_takeup_scope"],
)
def test_summarised_diagnostic_hard_fails_on_a_missing_field(missing):
    """A missing count must RAISE, never default to the alarm value.

    ``n_redrawn == 0`` is precisely the signal the diagnostic exists to raise
    ("the take-up grid is inert"). Defaulting a missing key to 0 would make
    every seed report it and set ``redraw_set_empty_in_every_seed`` to True,
    manufacturing the paper's finding out of a renamed key. referee_fixes
    already hard-fails when the diagnostic is absent entirely; a diagnostic
    that is present but incomplete must fail the same way.
    """
    from analysis.referee_fixes import _summarise_diagnostics

    diagnostics = [_diagnostic() for _ in range(3)]
    del diagnostics[1][missing]
    with pytest.raises(RuntimeError, match=missing):
        _summarise_diagnostics(diagnostics)


def test_summarised_diagnostic_tolerates_absent_weights():
    """weighted_redrawn is legitimately None; only the counts are required."""
    from analysis.referee_fixes import _summarise_diagnostics

    out = _summarise_diagnostics(
        [_diagnostic(weights_available=False, weighted_redrawn=None) for _ in range(2)]
    )
    assert out["weighted_redrawn_mean"] is None
    assert out["n_redrawn_mean"] == 2.0


def test_shared_stale_cell_is_relabelled_for_the_scope_it_is_reported_under():
    """The stale-flag cell is reused across scopes; its LABEL must not be.

    The values are scope-invariant by construction (the convention overwrites
    would_claim_uc wholesale), so reuse is legitimate. But
    write_referee_macros.entitled_scope_bound reads \\TakeupEntitledStale out
    of exactly this cell, so a reader auditing its provenance must not find it
    claiming the scope it was computed under rather than the one it is filed
    under.
    """
    from analysis.referee_fixes import (
        STALE_CELL_PATH,
        _share_stale_cell_across_scopes,
    )

    source = {
        "cushioning_rate": {"mean": 0.3445049495634098, "sd": 0.01},
        "exchequer_cost_m": {"mean": 100.0, "sd": 1.0},
        "redraw_diagnostic": {
            "uc_takeup_scope": "new_entitlement",
            "n_seeds": 25,
            "n_redrawn_max": 0,
            "redraw_set_empty_in_every_seed": True,
        },
    }
    shared = _share_stale_cell_across_scopes(
        source, computed_under="new_entitlement", reused_under="all_entitled"
    )

    # relabelled for where it is filed...
    assert shared["redraw_diagnostic"]["uc_takeup_scope"] == "all_entitled"
    # ...without losing where it came from
    assert shared["redraw_diagnostic"]["uc_takeup_scope_computed_under"] == "new_entitlement"
    assert shared["shared_across_scopes"] is True
    assert shared["shared_from"] == STALE_CELL_PATH
    assert "identical by construction" in shared["shared_note"]
    # the numbers are reused verbatim: that is the point of sharing
    assert shared["cushioning_rate"] == source["cushioning_rate"]
    # and the source cell is not mutated by being shared
    assert source["redraw_diagnostic"]["uc_takeup_scope"] == "new_entitlement"
    assert "shared_from" not in source
