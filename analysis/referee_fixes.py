"""Referee-revision computations: pension channel, take-up headline grid,
the monthly-versus-annual UC bounding exercise, and the New Style JSA bound.

Four artefacts requested by the referee report:

A. Pension-contribution channel. The HBAI disposable-income concept treats
   employee pension contributions as an outflow, so part of the wage-cut
   margin's measured cushioning is reduced retirement saving rather than
   tax--benefit insurance. This block recomputes the unit 12-month wage-cut
   and displacement cushioning rates under a pension-gross income concept
   (disposable income plus employee pension and salary-sacrifice
   contributions), so the pure statutory component of the wage-minus-
   displacement contrast can be reported. Runs on the primary submission
   estimator (balanced assignment, unit 12-month stress, 50 paired draws).

B. Take-up as a headline sensitivity. Full_tariff displacement at post-shock
   UC take-up 0.55/0.80/1.00 over 25 balanced-comparator (Bernoulli) seeds,
   and wage_cut at the same rates over 5 seeds (its dispersion is
   claiming-draw only). Run under BOTH re-draw scopes
   (shocks.UC_TAKEUP_SCOPES):

   - ``new_entitlement`` (the published convention, stored under
     ``takeup_headline``): re-draws only benefit units whose potential UC
     award moves from zero at baseline to positive post-shock;
   - ``all_entitled`` (stored under ``takeup_headline["all_entitled_scope"]``):
     re-draws every changed benefit unit with a positive post-shock potential
     award.

   Every cell also stores a REDRAW-SET DIAGNOSTIC (``redraw_diagnostic``).
   This is not decoration: if the re-draw set is empty the grid is inert and
   every take-up value returns a bit-identical cushioning rate, which reads
   as "take-up does not matter" when it actually means "no benefit unit was
   re-drawn". Always report ``n_redrawn`` alongside the grid.

C. Monthly-versus-annual UC bounding. Parameter-based accounting for a
   representative single displaced worker at the exposed-sector FRS mean wage:
   compares the paper's duration-equivalent annual stress with a correct
   monthly UC assessment and partial-year PAYE for 3-, 6- and 12-month spells.

D. New Style JSA bounding. The factorial decomposition attributes almost
   nothing to the employment state itself, partly because every institution
   that conditions on employment status is outside the annual model. This
   block sizes the largest such omission from statutory parameters alone.

Writes results/referee_fixes.json.
Usage: uv run python analysis/referee_fixes.py [--only pension takeup monthly jsa]

Note on runnability: the ``pension`` and ``takeup`` blocks need the licensed
FRS microdata, but ``--only jsa``, ``--only monthly`` and ``--only schedule``
are pure statutory arithmetic and run anywhere. The ``monthly`` block prefers
policyengine-uk's parameter tree and falls back to documented statutory
constants, recording which branch ran as
``monthly_uc_bounding.parameters.parameter_source``. That matters: the block's
prose NOTE is a module constant, and before the fallback existed the note could
only reach results/referee_fixes.json on a machine with policyengine-uk
installed, so a corrected note sat unshipped in the source while the artifact
carried the superseded (and inverted) one.

`make paper-values` currently re-runs only ``--only jsa``. Adding
``--only monthly schedule`` alongside it keeps every parameter-only block in
the artifact current on each build at no extra dependency cost.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np

from uk_trade_shock_study.runner import _baseline_and_persons
from uk_trade_shock_study.shocks import (
    DEFAULT_UC_TAKEUP_SCOPE,
    TradeShockScenario,
    _baseline_flag_values_and_rate,
    apply_shocks,
    build_shocked_simulation,
    uc_takeup_redraw_diagnostic,
)

PERIOD = 2026
DATASET = Path("data/frs_2024_25.h5")
RESULTS = Path("results")
PENSION_VARS = ("employee_pension_contributions", "pension_contributions_via_salary_sacrifice")
#: Pension channel now runs on the primary submission estimator: balanced
#: assignment at the unit 12-month stress, 50 paired draws — the same design
#: behind every headline number (referee presentation point on estimator
#: consistency).
PENSION_SEEDS = 50
TAKEUPS = (0.55, 0.80, 1.00)
TAKEUP_SEEDS_DISPLACEMENT = 25
TAKEUP_SEEDS_WAGE_CUT = 5
#: Key under ``takeup_headline`` holding the ``all_entitled`` re-draw scope —
#: the claiming margin that actually binds (see shocks.UC_TAKEUP_SCOPES). The
#: new-entitlement grid stays at the top level of ``takeup_headline`` so every
#: previously published path is unchanged.
ALL_ENTITLED_KEY = "all_entitled_scope"
#: Enum-like provenance labels for the entitled-scope bound the manuscript
#: quotes as \TakeupEntitledStale/Full/Spread. Recorded under
#: ``takeup_headline.entitled_scope_source`` and rendered into
#: \TakeupEntitledSource by analysis/write_referee_macros.py, exactly as
#: ``jsa_bounding.parameters.rate_source`` is rendered into \JSARateSource.
#: Which one applies DEPENDS ON THE ARTIFACT: the writer falls back to the
#: legacy results/takeup_diagnosis.json whenever this file carries no
#: ``all_entitled_scope`` block, and that fallback silently changes the
#: calibration behind three printed numbers, so it has to be recorded rather
#: than inferred from a build log.
ENTITLED_SCOPE_SOURCE_CURRENT = "referee_fixes_all_entitled_scope"
ENTITLED_SCOPE_SOURCE_LEGACY = "takeup_diagnosis_legacy_vintage"
ENTITLED_SCOPE_SOURCES = (ENTITLED_SCOPE_SOURCE_CURRENT, ENTITLED_SCOPE_SOURCE_LEGACY)
EXPOSED_MEAN_EARNINGS = 48_272.0  # FRS exposed-goods-division weighted mean


def _hni_total(sim) -> float:
    hh_w = sim.calculate("household_weight", period=PERIOD, map_to="household").values
    hni = sim.calculate("hbai_household_net_income", period=PERIOD, map_to="household").values
    return float((hni * hh_w).sum())


def _pension_total(sim, weights) -> float:
    total = 0.0
    for v in PENSION_VARS:
        total += float(
            (sim.calculate(v, period=PERIOD, map_to="person").values * weights).sum()
        )
    return total


def pension_block(dataset, baseline, persons) -> dict:
    w = persons["weight"].to_numpy(float)
    base_hni = _hni_total(baseline)
    base_pen = _pension_total(baseline, w)
    scenarios = {
        "full_tariff_wage_cut": (
            TradeShockScenario(
                "pension_unit_12m_wage_cut",
                "full_tariff",
                "wage_cut",
                elasticity=1.0,
                selection_method="balanced",
            ),
            1,
        ),
        "full_tariff_displacement": (
            TradeShockScenario(
                "pension_unit_12m_displacement",
                "full_tariff",
                "displacement",
                elasticity=1.0,
                selection_method="balanced",
            ),
            PENSION_SEEDS,
        ),
    }
    out = {"n_seeds": PENSION_SEEDS, "selection_method": "balanced"}
    for name, (scenario, seeds) in scenarios.items():
        rows = []
        for seed in range(seeds):
            table = apply_shocks(persons, scenario, seed=seed)
            sim = build_shocked_simulation(dataset, baseline, table, PERIOD)
            gross = float(
                (
                    (
                        persons["employment_income"].to_numpy(float)
                        - table["employment_income"].to_numpy(float)
                    )
                    * w
                ).sum()
            )
            d_hni = base_hni - _hni_total(sim)
            d_pen = base_pen - _pension_total(sim, w)
            rows.append(
                {
                    "gross_earnings_loss": gross,
                    "disposable_loss_hbai": d_hni,
                    "pension_outflow_fall": d_pen,
                    "cushioning_hbai": 1.0 - d_hni / gross,
                    # pension-gross income = disposable + pension outflows;
                    # its loss is d_hni + d_pen (contributions fall -> the
                    # pension-gross loss exceeds the HBAI loss).
                    "cushioning_pension_gross": 1.0 - (d_hni + d_pen) / gross,
                    "pension_share_of_gross_loss": d_pen / gross,
                }
            )
            print(f"[pension] {name} seed {seed}: {rows[-1]}", flush=True)
            del sim
        out[name] = {
            k: {
                "mean": float(np.mean([r[k] for r in rows])),
                "sd": float(np.std([r[k] for r in rows], ddof=1)) if len(rows) > 1 else 0.0,
            }
            for k in rows[0]
        }
    hbai_gap = (
        out["full_tariff_wage_cut"]["cushioning_hbai"]["mean"]
        - out["full_tariff_displacement"]["cushioning_hbai"]["mean"]
    )
    pg_gap = (
        out["full_tariff_wage_cut"]["cushioning_pension_gross"]["mean"]
        - out["full_tariff_displacement"]["cushioning_pension_gross"]["mean"]
    )
    out["contrast"] = {
        "cushioning_gap_hbai_pp": 100 * hbai_gap,
        "cushioning_gap_pension_gross_pp": 100 * pg_gap,
        "pension_channel_share_of_gap": 1.0 - pg_gap / hbai_gap if hbai_gap else float("nan"),
    }
    return out


def _summarise(rows: list[dict]) -> dict:
    return {
        k: {
            "mean": float(np.mean([r[k] for r in rows])),
            "sd": float(np.std([r[k] for r in rows], ddof=1)) if len(rows) > 1 else 0.0,
        }
        for k in rows[0]
    }


#: Fields every per-seed redraw diagnostic must carry. A MISSING field is a
#: schema drift, not a zero: defaulting ``n_redrawn`` to 0 would make every
#: seed report the diagnostic's own alarm value and set
#: ``redraw_set_empty_in_every_seed`` to True, fabricating the paper's finding
#: out of a renamed key. Consistent with the hard-fail in ``_takeup_grid``
#: when the diagnostic is absent altogether.
REQUIRED_DIAGNOSTIC_FIELDS = (
    "uc_takeup_scope",
    "n_redrawn",
    "n_entitled_changed_benunits",
    "n_changed_benunits",
)


def _require_diagnostic_field(diagnostic: dict, field: str, seed: int):
    if field not in diagnostic:
        raise RuntimeError(
            f"redraw-set diagnostic for seed {seed} is missing required field "
            f"{field!r} (has {sorted(diagnostic)}). The take-up grid cannot be "
            "reported from an incomplete diagnostic: substituting a default "
            "would silently report an inert re-draw set. Re-run the producing "
            "code (shocks.uc_takeup_redraw_diagnostic) rather than relaxing "
            "this check."
        )
    return diagnostic[field]


def _summarise_diagnostics(diagnostics: list[dict]) -> dict:
    """Collapse per-seed redraw-set diagnostics into a reportable summary.

    ``n_redrawn_max == 0`` is the signature of an INERT take-up grid: no
    benefit unit was re-drawn, so every take-up value must return a
    bit-identical cushioning rate. Storing it is the whole point of the
    diagnostic — which is exactly why a missing count raises here instead of
    defaulting to that same zero.
    """
    if not diagnostics:
        return {}
    for seed, diagnostic in enumerate(diagnostics):
        for field in REQUIRED_DIAGNOSTIC_FIELDS:
            _require_diagnostic_field(diagnostic, field, seed)
    counts = [int(d["n_redrawn"]) for d in diagnostics]
    entitled = [int(d["n_entitled_changed_benunits"]) for d in diagnostics]
    changed = [int(d["n_changed_benunits"]) for d in diagnostics]
    weighted = [
        d.get("weighted_redrawn") for d in diagnostics if d.get("weighted_redrawn") is not None
    ]
    out = {
        "uc_takeup_scope": diagnostics[0]["uc_takeup_scope"],
        "n_seeds": len(diagnostics),
        "n_redrawn_mean": float(np.mean(counts)),
        "n_redrawn_min": int(min(counts)),
        "n_redrawn_max": int(max(counts)),
        "n_entitled_changed_benunits_mean": float(np.mean(entitled)),
        "n_changed_benunits_mean": float(np.mean(changed)),
        "redraw_set_empty_in_every_seed": max(counts) == 0,
    }
    out["weighted_redrawn_mean"] = float(np.mean(weighted)) if weighted else None
    return out


def _apply_stale_baseline_flag(sim, stale_flag, period: int = PERIOD) -> None:
    """Force the pre-fix claiming convention onto ``sim``, and VERIFY it took.

    The stale-baseline-flag cell overwrites the post-shock ``would_claim_uc``
    wholesale with the pre-shock draw. ``shocks.redraw_uc_takeup`` reads its
    own ``set_input`` back and hard-errors on a mismatch; this sibling call
    did not, and the failure mode is worse here. A silently-rejected
    ``set_input`` would leave the simulation carrying the RE-DRAWN post-shock
    flag, so the stale-flag cell would reproduce the re-drawn cell's number
    and the grid would report all four claiming conventions agreeing — i.e. it
    would manufacture precisely the "take-up convention does not matter"
    result the manuscript quotes, out of an input that was never applied.

    Raising is the only safe behaviour: there is no correct number to fall
    back to.
    """
    expected = np.asarray(stale_flag, dtype=bool)
    sim.set_input("would_claim_uc", period, stale_flag)
    # The award was evaluated under the post-shock flag; drop formula outputs
    # so every downstream metric recomputes under the stale draw.
    sim._invalidate_all_caches()
    applied = np.asarray(
        sim.calculate("would_claim_uc", period=period, map_to="benunit").values,
        dtype=bool,
    )
    if applied.shape != expected.shape or not np.array_equal(applied, expected):
        n_wrong = (
            int((applied != expected).sum())
            if applied.shape == expected.shape
            else applied.size
        )
        raise RuntimeError(
            "stale-baseline-flag convention not applied: would_claim_uc read "
            f"back from the shocked simulation differs from the pre-shock draw "
            f"in {n_wrong} of {expected.size} benefit units (read-back shape "
            f"{applied.shape}, expected {expected.shape}). The cell would "
            "otherwise report the RE-DRAWN post-shock flag's cushioning rate "
            "under the stale-flag label, making all four claiming conventions "
            "agree by accident. Same hard-error contract as "
            "shocks.redraw_uc_takeup."
        )


def _takeup_grid(
    dataset,
    baseline,
    persons,
    *,
    scope: str,
    margins: tuple[tuple[str, int], ...],
    base_hni: float,
    hh_w,
    base_gov: float,
    with_stale_convention: bool,
) -> dict:
    """Run the take-up grid for one re-draw scope.

    ``stale_baseline_flag`` is the pre-fix convention (carry the pre-shock
    claiming draw through the shock) and is scope-independent by
    construction; it is recomputed per scope only when requested so each
    scope's block is self-contained.
    """
    w = persons["weight"].to_numpy(float)
    baseline_earnings = persons["employment_income"].to_numpy(float)
    stale_flag, _ = _baseline_flag_values_and_rate(baseline, PERIOD)

    def _one_draw(table, sim) -> dict:
        gross = float(
            ((baseline_earnings - table["employment_income"].to_numpy(float)) * w).sum()
        )
        d_hni = base_hni - _hni_total(sim)
        gov = float(
            (
                sim.calculate("gov_balance", period=PERIOD, map_to="household").values
                * hh_w
            ).sum()
        )
        return {
            "cushioning_rate": 1.0 - d_hni / gross,
            "exchequer_cost_m": (base_gov - gov) / 1e6,
        }

    out = {}
    for margin, n_seeds in margins:
        block = {}
        for takeup in TAKEUPS:
            scen = TradeShockScenario(
                f"full_tariff_{margin}",
                "full_tariff",
                margin,
                uc_takeup=takeup,
                uc_takeup_scope=scope,
            )
            rows, diagnostics = [], []
            for seed in range(n_seeds):
                table = apply_shocks(persons, scen, seed=seed)
                sim = build_shocked_simulation(dataset, baseline, table, PERIOD)
                rows.append(_one_draw(table, sim))
                diag = uc_takeup_redraw_diagnostic(table)
                if diag is None:
                    raise RuntimeError(
                        "redraw_uc_takeup did not record a redraw-set diagnostic; "
                        "the take-up grid cannot be reported without it."
                    )
                diagnostics.append(diag)
                del sim
            cell = _summarise(rows)
            cell["redraw_diagnostic"] = _summarise_diagnostics(diagnostics)
            block[f"{takeup:.2f}"] = cell
            print(
                f"[takeup:{scope}] {margin} {takeup}: {cell['cushioning_rate']} "
                f"redrawn={cell['redraw_diagnostic']['n_redrawn_mean']}",
                flush=True,
            )
        out[margin] = {"n_seeds": n_seeds, "uc_takeup_scope": scope, **block}

        if with_stale_convention and margin == "displacement":
            # Fourth convention: the stale pre-shock claiming flag (pre-fix
            # behaviour), displacement only.
            scen = TradeShockScenario(
                "full_tariff_displacement",
                "full_tariff",
                "displacement",
                uc_takeup_scope=scope,
            )
            rows = []
            for seed in range(n_seeds):
                table = apply_shocks(persons, scen, seed=seed)
                sim = build_shocked_simulation(dataset, baseline, table, PERIOD)
                _apply_stale_baseline_flag(sim, stale_flag)
                rows.append(_one_draw(table, sim))
                del sim
            cell = _summarise(rows)
            cell["redraw_diagnostic"] = {
                "uc_takeup_scope": scope,
                "n_seeds": n_seeds,
                "n_redrawn_mean": 0.0,
                "n_redrawn_min": 0,
                "n_redrawn_max": 0,
                "redraw_set_empty_in_every_seed": True,
                "note": (
                    "pre-fix convention: the post-shock claiming flag is "
                    "overwritten with the stale baseline draw, so nothing is "
                    "re-drawn by construction."
                ),
            }
            out[margin]["stale_baseline_flag"] = cell
    return out


#: JSON path of the stale-flag cell that both re-draw scopes share.
STALE_CELL_PATH = "takeup_headline.displacement.stale_baseline_flag"


def _share_stale_cell_across_scopes(cell: dict, *, computed_under: str, reused_under: str) -> dict:
    """Relabel the shared stale-flag cell for the scope it is reused under.

    The stale-baseline-flag convention overwrites ``would_claim_uc`` wholesale
    with the pre-shock draw, so its cushioning rate is BIT-IDENTICAL across
    re-draw scopes: reusing the cell is legitimate and saves 25 simulations.
    What is not legitimate is carrying the producing scope's label into the
    other scope's block — ``write_referee_macros.entitled_scope_bound`` reads
    ``\\TakeupEntitledStale`` out of exactly this cell, so a reader checking
    its provenance would find ``uc_takeup_scope`` claiming the wrong scope.
    The copy therefore states the scope it is REPORTED under, the scope it was
    COMPUTED under, and that the two are deliberately the same numbers.
    """
    shared = deepcopy(cell)
    diagnostic = shared.setdefault("redraw_diagnostic", {})
    diagnostic["uc_takeup_scope"] = reused_under
    diagnostic["uc_takeup_scope_computed_under"] = computed_under
    shared["shared_across_scopes"] = True
    shared["shared_from"] = STALE_CELL_PATH
    shared["shared_note"] = (
        "Deliberately shared between the "
        f"{computed_under!r} and {reused_under!r} re-draw scopes, NOT recomputed. "
        "The stale-baseline-flag convention overwrites the post-shock claiming "
        "flag wholesale with the pre-shock draw, so nothing is re-drawn under "
        "either scope and the two scopes' values are identical by construction. "
        f"Computed once under {computed_under!r} at "
        f"{STALE_CELL_PATH}; uc_takeup_scope here names the scope this copy is "
        "reported under, and uc_takeup_scope_computed_under names where it came "
        "from."
    )
    return shared


def takeup_block(dataset, baseline, persons) -> dict:
    base_hni = _hni_total(baseline)
    hh_w = baseline.calculate("household_weight", period=PERIOD, map_to="household").values
    base_gov = float(
        (baseline.calculate("gov_balance", period=PERIOD, map_to="household").values * hh_w).sum()
    )
    margins = (
        ("displacement", TAKEUP_SEEDS_DISPLACEMENT),
        ("wage_cut", TAKEUP_SEEDS_WAGE_CUT),
    )
    common = {
        "base_hni": base_hni,
        "hh_w": hh_w,
        "base_gov": base_gov,
        "margins": margins,
    }
    out = _takeup_grid(
        dataset,
        baseline,
        persons,
        scope=DEFAULT_UC_TAKEUP_SCOPE,
        with_stale_convention=True,
        **common,
    )
    # The binding claiming margin. Run displacement only: the wage-cut margin's
    # dispersion is claiming-draw only and its grid adds no information here.
    entitled = _takeup_grid(
        dataset,
        baseline,
        persons,
        scope="all_entitled",
        with_stale_convention=False,
        **{**common, "margins": (("displacement", TAKEUP_SEEDS_DISPLACEMENT),)},
    )
    # The stale-flag convention overwrites the post-shock claiming flag
    # wholesale, so it is BIT-IDENTICAL across re-draw scopes by construction.
    # Reuse it rather than burning another 25 simulations to reproduce it, and
    # keep it in this block so the scope's four conventions stay comparable —
    # but relabel the copy, because a cell that says new_entitlement inside the
    # all_entitled block is a provenance bug even when the numbers are right.
    source_cell = out["displacement"]["stale_baseline_flag"]
    entitled["displacement"]["stale_baseline_flag"] = _share_stale_cell_across_scopes(
        source_cell,
        computed_under=DEFAULT_UC_TAKEUP_SCOPE,
        reused_under="all_entitled",
    )
    # Say so on the producing side too, so the sharing is visible from either
    # end of the JSON rather than only from the copy.
    source_cell["shared_across_scopes"] = True
    source_cell["shared_into"] = (
        f"takeup_headline.{ALL_ENTITLED_KEY}.displacement.stale_baseline_flag"
    )
    out[ALL_ENTITLED_KEY] = entitled
    # Stamp the provenance of the entitled-scope bound INTO the artifact. A
    # run that reaches this line has just computed that grid at the current
    # calibration, so \TakeupEntitledStale/Full/Spread are current-vintage
    # numbers; write_referee_macros re-resolves and re-records this field when
    # it reads an artifact that predates the block, so the manuscript never has
    # to infer the vintage from a build log.
    out["entitled_scope_source"] = ENTITLED_SCOPE_SOURCE_CURRENT
    out["entitled_scope_source_options"] = list(ENTITLED_SCOPE_SOURCES)
    out[ALL_ENTITLED_KEY]["notes"] = (
        "Re-draws the claiming flag for EVERY changed benefit unit with a "
        "positive post-shock potential UC award, including units already "
        "entitled at baseline. Upper bound on the claiming margin, not a "
        "behavioural estimate. Compare against the new-entitlement grid "
        "stored at the top level of takeup_headline."
    )
    return out


# ---------------------------------------------------------------------------
# Statutory band structure shared by the two representative-worker blocks
# ---------------------------------------------------------------------------
#
# ``monthly_uc_block`` and ``schedule_benchmark_block`` both need income tax and
# employee National Insurance for a single representative worker. They used to
# implement the schedule TWICE, differently: the monthly block read the NI
# primary threshold from its own parameter node, while the schedule block
# aliased the INCOME TAX personal allowance as the NI threshold and ignored the
# higher-rate band and the upper earnings limit altogether. Those two
# constructions agree only for earnings between the personal allowance and the
# higher-rate threshold, in a year where the personal allowance and the NI
# primary threshold happen to coincide. Both blocks now call the same band
# functions over the same parameter dict, so they cannot disagree.

#: Machine-readable provenance for the statutory parameters, mirroring the
#: contract ``jsa_bounding.parameters.rate_source`` already carries: which of
#: the two branches produced the numbers DEPENDS ON THE MACHINE, because
#: policyengine-uk is an optional dependency, so the artifact has to say.
PARAMETER_SOURCE_POLICYENGINE = "policyengine_parameter"
PARAMETER_SOURCE_FALLBACK = "statutory_fallback"
PARAMETER_SOURCES = (PARAMETER_SOURCE_POLICYENGINE, PARAMETER_SOURCE_FALLBACK)

#: Documented statutory values used when policyengine-uk's parameter tree is
#: not importable here. Sources, field by field:
#:
#: - ``personal_allowance``, ``basic_rate``, ``higher_rate``,
#:   ``basic_rate_limit``: HMRC income tax rates and allowances, frozen at the
#:   2025-26 values (GBP 12,570 allowance, 20/40 per cent, GBP 37,700
#:   basic-rate limit on TAXABLE income).
#: - ``ni_employee_main``, ``ni_employee_above_uel``: class 1 employee rates,
#:   8 per cent between the primary threshold and the upper earnings limit and
#:   2 per cent above it.
#: - ``ni_primary_threshold_annual``, ``ni_upper_earnings_limit_annual``:
#:   HMRC's own PUBLISHED ANNUAL figures (GBP 12,570 and GBP 50,270), not
#:   weekly figures multiplied by 52. HMRC quotes the primary threshold as
#:   GBP 242 per week AND GBP 12,570 per year; 242 x 52 = 12,584 is an
#:   artefact of the weekly quote, and policyengine-uk resolves the same node
#:   to 241.73 per week = 12,569.96 per year. The three conventions differ by
#:   at most GBP 15 of earnings, i.e. GBP 1.20 of NI on the representative
#:   worker, which is below the precision of every macro this file feeds.
#: - ``standard_allowance_single_25_plus_month``: the UC standard allowance for
#:   a single claimant aged 25 or over that policyengine-uk resolves for
#:   PERIOD, recorded from the frozen build.
#:
#: A fallback is legitimate here for the same reason it is for the JSA rate:
#: these are published statutory constants, the block is pure arithmetic over
#: them, and the branch that ran is recorded in the artifact. What is NOT
#: legitimate is mixing the branches, so resolution is all-or-nothing.
FALLBACK_SCHEDULE_PARAMETERS = {
    "standard_allowance_single_25_plus_month": 424.90,
    "personal_allowance": 12_570.0,
    "basic_rate": 0.20,
    "higher_rate": 0.40,
    "basic_rate_limit": 37_700.0,
    "ni_employee_main": 0.08,
    "ni_employee_above_uel": 0.02,
    "ni_primary_threshold_annual": 12_570.0,
    "ni_upper_earnings_limit_annual": 50_270.0,
}
FALLBACK_SCHEDULE_PARAMETERS_VINTAGE = (
    "HMRC 2025-26 income tax and class 1 NI parameters (annual thresholds as "
    "published) with the PERIOD Universal Credit standard allowance recorded "
    "from the frozen policyengine-uk build; no 2026-27 uprating applied to the "
    "tax schedule"
)

#: Parameters BOTH representative-worker blocks require. A missing entry is a
#: hard failure, not a default: the schedule block used to carry an unread
#: ``higher_rate_threshold`` in its parameters dict, which looked like a guard
#: and acted as nothing.
SCHEDULE_REQUIRED_PARAMETERS = (
    "representative_earnings",
    "personal_allowance",
    "basic_rate",
    "higher_rate",
    "basic_rate_limit",
    "higher_rate_threshold",
    "ni_employee_main",
    "ni_employee_above_uel",
    "ni_primary_threshold_annual",
    "ni_upper_earnings_limit_annual",
)

#: Earnings at which the personal allowance starts to taper (GBP 1 withdrawn
#: per GBP 2 of income), producing a 60 per cent marginal band that these
#: blocks do NOT model. Together with the additional rate above it, this is
#: the upper end of the range the arithmetic here is valid over.
PERSONAL_ALLOWANCE_TAPER_THRESHOLD = 100_000.0

#: Plausible ranges for the periodicity each node is READ AS. See
#: ``_check_periodic_amount``: the JSA block already refused to accept a rate
#: whose periodicity could not be weekly, and every other periodic node this
#: file converts now gets the same treatment.
NI_PRIMARY_THRESHOLD_WEEKLY_RANGE = (150.0, 400.0)
NI_UPPER_EARNINGS_LIMIT_WEEKLY_RANGE = (700.0, 1_400.0)
UC_STANDARD_ALLOWANCE_MONTHLY_RANGE = (300.0, 700.0)
#: Annual figures are not converted, but a range still catches a node that
#: silently changed periodicity in the other direction.
PERSONAL_ALLOWANCE_ANNUAL_RANGE = (5_000.0, 30_000.0)
BASIC_RATE_LIMIT_ANNUAL_RANGE = (20_000.0, 80_000.0)
WEEKS_PER_YEAR = 52.0
MONTHS_PER_YEAR = 12.0


def _check_periodic_amount(
    value: float,
    *,
    where: str,
    what: str,
    periodicity: str,
    plausible_range: tuple[float, float],
    used_as: str,
) -> float:
    """Hard-fail an amount whose magnitude contradicts its assumed periodicity.

    Generalises the guard the JSA rate already had. A monthly or annual node
    read as weekly is wrong by a factor of 4.33 or 52, a weekly node read as
    monthly by 4.33, and nothing downstream would notice: the arithmetic is
    linear and the result is a plausible-looking number. The ranges are chosen
    so that every OTHER periodicity the node could plausibly carry falls
    outside them.
    """
    low, high = plausible_range
    if not np.isfinite(value) or not low <= value <= high:
        raise RuntimeError(
            f"{where} returned {value!r} for {what}, outside the plausible "
            f"{periodicity} range {plausible_range}. This is what a periodicity "
            "change looks like (a monthly or annual node read as weekly is "
            "wrong by a factor of 4.33 or 52, and vice versa), and this block "
            f"{used_as}. Fix the parameter path or the range; do not accept "
            "the value."
        )
    return float(value)


def income_tax(earnings: float, p: dict) -> float:
    """Annual income tax on ``earnings`` from the actual band structure.

    Personal allowance, then the basic rate up to ``basic_rate_limit`` of
    TAXABLE income, then the higher rate. The additional rate and the personal
    allowance taper are outside the validated range (see
    ``_check_schedule_validity``).
    """
    taxable = max(0.0, earnings - p["personal_allowance"])
    basic_slice = min(taxable, p["basic_rate_limit"])
    higher_slice = max(0.0, taxable - p["basic_rate_limit"])
    return p["basic_rate"] * basic_slice + p["higher_rate"] * higher_slice


def employee_ni(earnings: float, p: dict) -> float:
    """Annual class 1 employee NI from the actual band structure.

    Main rate between the primary threshold and the upper earnings limit, the
    reduced rate above the UEL. Aliasing the income tax personal allowance as
    the primary threshold, and dropping the UEL, is what this replaces.
    """
    pt = p["ni_primary_threshold_annual"]
    uel = p["ni_upper_earnings_limit_annual"]
    main_slice = max(0.0, min(earnings, uel) - pt)
    upper_slice = max(0.0, earnings - uel)
    return p["ni_employee_main"] * main_slice + p["ni_employee_above_uel"] * upper_slice


def marginal_income_tax_rate(earnings: float, p: dict) -> float:
    if earnings <= p["personal_allowance"]:
        return 0.0
    if earnings <= p["personal_allowance"] + p["basic_rate_limit"]:
        return p["basic_rate"]
    return p["higher_rate"]


def marginal_employee_ni_rate(earnings: float, p: dict) -> float:
    if earnings <= p["ni_primary_threshold_annual"]:
        return 0.0
    if earnings <= p["ni_upper_earnings_limit_annual"]:
        return p["ni_employee_main"]
    return p["ni_employee_above_uel"]


def _require_schedule_parameters(params: dict, where: str) -> dict:
    """Return the validated parameter dict, or raise naming what is missing."""
    missing = [k for k in SCHEDULE_REQUIRED_PARAMETERS if params.get(k) is None]
    if missing:
        raise RuntimeError(
            f"{where} is missing statutory parameters {missing} (has "
            f"{sorted(params)}). The representative-worker arithmetic reads the "
            "full band structure — personal allowance, basic-rate limit, higher "
            "rate, NI primary threshold, NI upper earnings limit and the "
            "above-UEL rate — and every one of them changes the emitted "
            "marginal and average deduction rates. Substituting a default, or "
            "aliasing one parameter for another, is how the block silently "
            "emitted wrong rates before. Re-run "
            "`analysis/referee_fixes.py --only monthly schedule`."
        )
    resolved = {k: float(params[k]) for k in SCHEDULE_REQUIRED_PARAMETERS}
    implied = resolved["personal_allowance"] + resolved["basic_rate_limit"]
    if abs(resolved["higher_rate_threshold"] - implied) > 1.0:
        raise RuntimeError(
            f"{where} is internally inconsistent: higher_rate_threshold "
            f"{resolved['higher_rate_threshold']} is not the personal allowance "
            f"plus the basic-rate limit ({implied}). These are two conventions "
            "for the same boundary — one on GROSS earnings, one on TAXABLE "
            "income — and mixing them shifts the higher-rate band by the "
            "allowance."
        )
    return resolved


def _check_schedule_validity(earnings: float, p: dict) -> dict:
    """Hard-fail representative earnings outside the block's valid range.

    At or below the LOWER of the personal allowance and the NI primary
    threshold neither instrument bites, both the marginal and the average rate
    are zero, and the marginal-versus-average contrast the block exists to
    state is degenerate. Above the personal allowance taper the true marginal
    rate is 60 per cent (and the additional rate follows above that), neither
    of which this arithmetic models.
    """
    floor = min(p["personal_allowance"], p["ni_primary_threshold_annual"])
    ceiling = PERSONAL_ALLOWANCE_TAPER_THRESHOLD
    if not floor < earnings < ceiling:
        raise RuntimeError(
            f"representative earnings {earnings!r} fall outside the range this "
            f"schedule benchmark is valid over, ({floor}, {ceiling}). At or "
            "below the lower of the personal allowance and the NI primary "
            "threshold both the marginal and the average deduction rate are "
            "zero, so the marginal-versus-average contrast the block reports is "
            "degenerate. At or above the personal allowance taper the true "
            "marginal rate is 60 per cent and the additional rate follows, "
            "neither of which this block models. Extend the band structure "
            "rather than widening the range."
        )
    return {"valid_earnings_range": [floor, ceiling]}


def _resolve_schedule_parameters() -> tuple[dict, str, str, str]:
    """(parameters, source, detail, vintage) for the statutory band structure.

    Prefers policyengine-uk so the figures are the ones the model itself would
    apply. Resolution is ALL-OR-NOTHING: if any node is missing or moved, the
    whole documented fallback set is used rather than a half-policyengine,
    half-constant hybrid that no reader could reconstruct. Every periodic node
    is periodicity-checked before it is annualised.
    """
    try:
        from policyengine_uk.system import system

        inst = f"{PERIOD}-01-01"
        gov = system.parameters.gov
        sa_month = _check_periodic_amount(
            float(gov.dwp.universal_credit.standard_allowance.amount.SINGLE_OLD(inst)),
            where="policyengine-uk gov.dwp.universal_credit.standard_allowance"
            ".amount.SINGLE_OLD",
            what="the UC standard allowance for a single claimant aged 25 or over",
            periodicity="monthly",
            plausible_range=UC_STANDARD_ALLOWANCE_MONTHLY_RANGE,
            used_as="multiplies the figure by the number of months out of work",
        )
        brackets = gov.hmrc.income_tax.rates.uk.brackets
        pa = _check_periodic_amount(
            float(gov.hmrc.income_tax.allowances.personal_allowance.amount(inst)),
            where="policyengine-uk gov.hmrc.income_tax.allowances.personal_allowance",
            what="the income tax personal allowance",
            periodicity="annual",
            plausible_range=PERSONAL_ALLOWANCE_ANNUAL_RANGE,
            used_as="subtracts the figure from annual earnings",
        )
        basic = float(brackets[0].rate(inst))
        higher = float(brackets[1].rate(inst))
        basic_rate_limit = _check_periodic_amount(
            float(brackets[1].threshold(inst)),
            where="policyengine-uk gov.hmrc.income_tax.rates.uk.brackets[1].threshold",
            what="the basic-rate limit on taxable income",
            periodicity="annual",
            plausible_range=BASIC_RATE_LIMIT_ANNUAL_RANGE,
            used_as="splits annual taxable income between the basic and higher rates",
        )
        ni_rates = gov.hmrc.national_insurance.class_1.rates.employee
        ni_thresholds = gov.hmrc.national_insurance.class_1.thresholds
        ni_main = float(ni_rates.main(inst))
        ni_above_uel = float(ni_rates.additional(inst))
        ni_pt_week = _check_periodic_amount(
            float(ni_thresholds.primary_threshold(inst)),
            where="policyengine-uk gov.hmrc.national_insurance.class_1.thresholds"
            ".primary_threshold",
            what="the class 1 primary threshold",
            periodicity="weekly",
            plausible_range=NI_PRIMARY_THRESHOLD_WEEKLY_RANGE,
            used_as=f"multiplies the figure by {WEEKS_PER_YEAR:.0f} weeks",
        )
        ni_uel_week = _check_periodic_amount(
            float(ni_thresholds.upper_earnings_limit(inst)),
            where="policyengine-uk gov.hmrc.national_insurance.class_1.thresholds"
            ".upper_earnings_limit",
            what="the class 1 upper earnings limit",
            periodicity="weekly",
            plausible_range=NI_UPPER_EARNINGS_LIMIT_WEEKLY_RANGE,
            used_as=f"multiplies the figure by {WEEKS_PER_YEAR:.0f} weeks",
        )
    except _PARAMETER_LOOKUP_ERRORS as exc:
        params = dict(FALLBACK_SCHEDULE_PARAMETERS)
        params["higher_rate_threshold"] = (
            params["personal_allowance"] + params["basic_rate_limit"]
        )
        return (
            params,
            PARAMETER_SOURCE_FALLBACK,
            "documented statutory constants (policyengine-uk parameter tree "
            f"unavailable here: {type(exc).__name__}: {exc})",
            FALLBACK_SCHEDULE_PARAMETERS_VINTAGE,
        )
    params = {
        "standard_allowance_single_25_plus_month": sa_month,
        "personal_allowance": pa,
        "basic_rate": basic,
        "higher_rate": higher,
        "basic_rate_limit": basic_rate_limit,
        "higher_rate_threshold": pa + basic_rate_limit,
        "ni_employee_main": ni_main,
        "ni_employee_above_uel": ni_above_uel,
        "ni_primary_threshold_annual": ni_pt_week * WEEKS_PER_YEAR,
        "ni_upper_earnings_limit_annual": ni_uel_week * WEEKS_PER_YEAR,
        "ni_primary_threshold_weekly": ni_pt_week,
        "ni_upper_earnings_limit_weekly": ni_uel_week,
    }
    return (
        params,
        PARAMETER_SOURCE_POLICYENGINE,
        "policyengine-uk parameter tree: gov.hmrc.income_tax, "
        "gov.hmrc.national_insurance.class_1, gov.dwp.universal_credit",
        f"policyengine-uk parameter values for {PERIOD}",
    )


#: How the monthly-versus-annual comparison actually works, stated so that it
#: agrees with the block's own ``cushion_share_*`` fields.
#:
#: The earlier text had this exactly backwards. At the representative earnings
#: of GBP 48,272 the whole taxable slice sits in the basic-rate band (taxable
#: income 35,702 against a basic-rate limit of 37,700), so there is NO
#: higher-rate income to forgo on either side of the comparison. What differs
#: is the RATE AT WHICH the lost earnings are relieved: a three-month loss of
#: 12,068 comes off the top of the year's earnings and is relieved at the 20
#: per cent marginal rate (2,413.60), while zeroing a whole year and scaling by
#: the probability m/12 relieves the same loss at the year's AVERAGE rate of
#: 14.79 per cent (7,140.40 x 0.25 = 1,785.10), because a full-year zeroing
#: also gives up the untaxed personal allowance. Less relief means less
#: measured cushioning, so the annual model UNDERSTATES cushioning at
#: sub-annual durations (31.3 against 36.5 per cent at three and six months)
#: and is exact at twelve, where the two constructions coincide.
MONTHLY_UC_NOTES = (
    "UC entitlement per worker-month is identical by construction: m months "
    "at the full standard allowance equals probability m/12 of 12 months, and "
    "employee National Insurance is likewise proportional above the primary "
    "threshold. The difference is income tax, and it runs in ONE direction: "
    "the annual duration-equivalent stress zeroes a FULL year of earnings and "
    "scales by m/12, which spreads the loss across the untaxed personal "
    "allowance as well as the taxed slices and therefore relieves it at the "
    "year's AVERAGE tax rate, whereas a genuine partial-year loss comes off "
    "the top of the year's earnings and is relieved at the MARGINAL rate. At "
    "the representative earnings used here the entire taxable slice is "
    "basic-rate, so no higher-rate relief arises on either construction. "
    "Average-rate relief is the smaller of the two, so the annual model "
    "UNDERSTATES tax relief and hence understates cushioning at sub-annual "
    "durations (compare cushion_share_annual_equivalent with "
    "cushion_share_monthly_correct for the 3- and 6-month spells); the two "
    "coincide exactly at twelve months, where the constructions are the same. "
    "The 5-week payment wait shifts timing, not annual entitlement."
)


def monthly_uc_block(parameters: dict | None = None) -> dict:
    """Parameter-based bounding of the annual model against a monthly UC
    assessment for a representative single displaced worker (aged 25+,
    no children, no housing element, capital below the lower limit).

    ``parameters`` lets a caller re-derive the block from a parameter set that
    is already stored (see ``--only monthly`` and the ``schedule`` block);
    passing None resolves them, preferring policyengine-uk and falling back to
    the documented statutory constants so this block — and therefore the
    corrected note it carries — can be regenerated on a machine without
    policyengine-uk installed.
    """
    if parameters is None:
        params, source, detail, vintage = _resolve_schedule_parameters()
    else:
        params, source, detail, vintage = (
            dict(parameters),
            str(parameters.get("parameter_source", PARAMETER_SOURCE_FALLBACK)),
            str(parameters.get("parameter_source_detail", "supplied by caller")),
            str(parameters.get("parameter_vintage", "supplied by caller")),
        )
    e = EXPOSED_MEAN_EARNINGS
    params.setdefault("representative_earnings", e)
    p = _require_schedule_parameters(params, "resolved statutory parameters")
    sa_month = _check_periodic_amount(
        float(params["standard_allowance_single_25_plus_month"]),
        where="resolved statutory parameters",
        what="the UC standard allowance for a single claimant aged 25 or over",
        periodicity="monthly",
        plausible_range=UC_STANDARD_ALLOWANCE_MONTHLY_RANGE,
        used_as="multiplies the figure by the number of months out of work",
    )
    _check_schedule_validity(e, p)

    rows = {}
    for m in (3, 6, 12):
        f = m / MONTHS_PER_YEAR
        # Monthly-correct: m months out of work at full standard allowance
        # (zero earned income), (12-m) months at earnings that taper UC to
        # zero; PAYE reconciles to annual tax on partial-year earnings.
        uc_monthly = m * sa_month
        tax_relief_monthly = income_tax(e, p) - income_tax((1 - f) * e, p)
        # NI is assessed per pay period, so m months of relief is m twelfths of
        # the annual liability: every band boundary scales by the same 1/12.
        ni_relief_monthly = m * (employee_ni(e, p) / MONTHS_PER_YEAR)
        # Annual duration-equivalent stress: probability f of a coherent
        # full-year displaced state; expectations per exposed worker.
        uc_annual = f * MONTHS_PER_YEAR * sa_month
        tax_relief_annual = f * income_tax(e, p)
        ni_relief_annual = f * employee_ni(e, p)
        rows[f"{m}m"] = {
            "uc_monthly_correct": uc_monthly,
            "uc_annual_equivalent": uc_annual,
            "tax_relief_monthly_correct": tax_relief_monthly,
            "tax_relief_annual_equivalent": tax_relief_annual,
            "ni_relief_monthly_correct": ni_relief_monthly,
            "ni_relief_annual_equivalent": ni_relief_annual,
            "gross_loss": f * e,
            "cushion_share_monthly_correct": (
                uc_monthly + tax_relief_monthly + ni_relief_monthly
            )
            / (f * e),
            "cushion_share_annual_equivalent": (
                uc_annual + tax_relief_annual + ni_relief_annual
            )
            / (f * e),
        }
    stored = {k: params[k] for k in sorted(params) if not k.startswith("parameter_")}
    stored.update(p)
    stored["standard_allowance_single_25_plus_month"] = sa_month
    stored["representative_earnings"] = e
    # Provenance travels with the numbers, exactly as it does for the JSA rate:
    # which branch resolved these parameters depends on the build machine.
    stored["parameter_source"] = source
    stored["parameter_source_options"] = list(PARAMETER_SOURCES)
    stored["parameter_source_detail"] = detail
    stored["parameter_vintage"] = vintage
    return {
        "parameters": stored,
        "spells": rows,
        "notes": MONTHLY_UC_NOTES,
    }


#: NEW STYLE JOBSEEKER'S ALLOWANCE, contribution-based, standard personal
#: allowance for claimants aged 25 or over, WEEKLY, 2025-26 statutory rate
#: (£90.50 in 2024-25 uprated by the September 2024 CPI figure of 1.7%).
#: Used only when policyengine-uk's parameter tree is unavailable; see
#: jsa_block for the source/vintage labelling contract.
JSA_WEEKLY_RATE_2025_26 = 92.05
JSA_RATE_SOURCE_FALLBACK = (
    "DWP benefit and pension rates 2025 to 2026: New Style (contribution-based) "
    "Jobseeker's Allowance, personal allowance, aged 25 or over"
)
#: Statutory maximum duration of contribution-based JSA.
JSA_MAX_DAYS = 182
#: Enum-like provenance labels for the rate. Recorded verbatim in
#: results/referee_fixes.json as ``jsa_bounding.parameters.rate_source`` and
#: rendered into ``\JSARateSource`` by analysis/write_referee_macros.py. The
#: value itself DEPENDS ON THE MACHINE (policyengine-uk is an optional
#: dependency and `make paper-values` re-runs `--only jsa` on every build), so
#: the artifact has to say which of the two produced the printed number
#: instead of leaving it to be inferred from a free-text sentence.
JSA_RATE_SOURCE_PARAMETER = "policyengine_parameter"
JSA_RATE_SOURCE_FALLBACK_ID = "statutory_fallback"
JSA_RATE_SOURCES = (JSA_RATE_SOURCE_PARAMETER, JSA_RATE_SOURCE_FALLBACK_ID)
#: Plausible range for a WEEKLY NS-JSA personal allowance for a claimant aged
#: 25 or over, in pounds. The 2024-25 rate was 90.50 and the 2025-26 rate
#: 92.05, so this brackets any realistic uprating while excluding every other
#: periodicity the parameter node could carry: a daily figure (~13), a monthly
#: one (~399) and an annual one (~4,787) all fall outside it. A parameter with
#: a different periodicity would otherwise be wrong by a factor of 52 in
#: silence, because the block multiplies the rate by 26 weeks.
JSA_WEEKLY_RATE_PLAUSIBLE_RANGE = (60.0, 200.0)
#: Exceptions that mean "policyengine-uk's parameter tree does not expose this
#: node here": the package is absent, or it exists under a different path.
#: Deliberately NOT bare ``Exception`` — a parameter that is present but
#: implausible must raise, not be papered over with the statutory constant.
_PARAMETER_LOOKUP_ERRORS = (ImportError, AttributeError, KeyError, TypeError)


def _check_weekly_rate(rate: float, where: str) -> float:
    """Hard-fail a rate that cannot be a weekly NS-JSA personal allowance.

    Thin wrapper over the general periodicity guard, kept because this is the
    one the JSA block's error message is written for. Every other periodic node
    this file converts goes through ``_check_periodic_amount`` directly.
    """
    return _check_periodic_amount(
        rate,
        where=where,
        what="the NS-JSA personal allowance for a claimant aged 25 or over",
        periodicity="weekly NS-JSA",
        plausible_range=JSA_WEEKLY_RATE_PLAUSIBLE_RANGE,
        used_as=f"multiplies this figure by {JSA_MAX_DAYS / 7:.0f} weeks",
    )


def _jsa_weekly_rate() -> tuple[float, str, str, str]:
    """(rate, rate_source, source_detail, vintage) for the NS-JSA 25+ weekly rate.

    Prefers policyengine-uk's parameter tree for the simulation year so the
    figure is the one the model itself would apply. Falls back to the
    documented 2025-26 statutory rate when policyengine-uk is not installed —
    and says so in the vintage string rather than presenting an unverified
    2026-27 figure as current.

    ``rate_source`` is one of ``JSA_RATE_SOURCES``: a stable, machine-readable
    token, because which branch runs depends on the build machine and the
    manuscript has to be able to state which one produced the number it prints.
    The retrieved rate is validated as a WEEKLY figure before it is accepted.
    """
    try:
        from policyengine_uk.system import system

        inst = f"{PERIOD}-01-01"
        rate = float(
            system.parameters.gov.dwp.JSA.contrib.amount_25_plus(inst)
        )
    except _PARAMETER_LOOKUP_ERRORS as exc:
        return (
            _check_weekly_rate(JSA_WEEKLY_RATE_2025_26, "statutory fallback constant"),
            JSA_RATE_SOURCE_FALLBACK_ID,
            f"{JSA_RATE_SOURCE_FALLBACK} (policyengine-uk parameter tree "
            f"unavailable here: {type(exc).__name__})",
            "2025-26 statutory rate; 2026-27 uprating not applied",
        )
    return (
        _check_weekly_rate(
            rate, "policyengine-uk gov.dwp.JSA.contrib.amount_25_plus"
        ),
        JSA_RATE_SOURCE_PARAMETER,
        "policyengine-uk parameter tree: gov.dwp.JSA.contrib.amount_25_plus",
        f"policyengine-uk parameter value for {PERIOD}",
    )


def schedule_benchmark_block(monthly: dict) -> dict:
    """Marginal versus average deduction rate for a representative worker.

    The paper's headline is that a diffuse earnings cut is cushioned more than
    a complete loss of the same aggregate. That ordering is implied by the
    statutory schedule alone, before any benefit, pension or household effect:
    a marginal cut is relieved at the MARGINAL rate, a total loss at the
    AVERAGE rate, and a personal allowance guarantees the former exceeds the
    latter at every earnings level.

    Reporting the one-worker arithmetic turns the simulated gap from something
    a reader must take on trust into something checkable in three lines, and
    it pre-empts the obvious objection that 43.9 per cent is impossible on a
    20 per cent basic rate. The residual between this benchmark and the
    simulated gap is what the microsimulation actually contributes: means-
    tested support, pension contributions, household composition, and the
    loss-weighted mix of marginal rates across the exposed population.
    """
    # Reuse the statutory parameters the monthly-UC block already resolved, so
    # the two blocks can never disagree about the schedule and this one runs
    # without policyengine when that block is already stored.
    #
    # It computes the full band structure. The previous implementation aliased
    # the income tax personal allowance as the NI primary threshold, ignored the
    # higher-rate band and the upper earnings limit, and stored a
    # ``higher_rate_threshold`` it never read — a guard that acted on nothing.
    # Those three faults cancel only for earnings between the allowance and the
    # higher-rate threshold in a year where the allowance and the NI primary
    # threshold coincide, and nothing raised outside that window.
    stored = monthly["parameters"]
    p = _require_schedule_parameters(
        stored, "referee_fixes.json monthly_uc_bounding.parameters"
    )
    earnings = p["representative_earnings"]
    validity = _check_schedule_validity(earnings, p)

    tax = income_tax(earnings, p)
    ni = employee_ni(earnings, p)
    marginal = marginal_income_tax_rate(earnings, p) + marginal_employee_ni_rate(
        earnings, p
    )
    average = (tax + ni) / earnings
    if not marginal > average:
        raise RuntimeError(
            f"the schedule benchmark computed a marginal deduction rate "
            f"{marginal!r} that does not exceed the average rate {average!r}. "
            "The block exists to state that a progressive schedule relieves a "
            "marginal cut at more than a total loss; a non-positive gap means "
            "the parameters no longer describe a progressive schedule and the "
            "emitted \\ScheduleImpliedGap would contradict the manuscript's "
            "mechanism argument."
        )
    band = (
        "basic-rate"
        if earnings <= p["higher_rate_threshold"]
        else "higher-rate"
    )
    ni_band = (
        "main-rate"
        if earnings <= p["ni_upper_earnings_limit_annual"]
        else "above the upper earnings limit"
    )
    return {
        "parameters": {
            **p,
            "parameter_source": stored.get("parameter_source"),
            "parameter_vintage": stored.get("parameter_vintage"),
            **validity,
        },
        "taxable_income": max(0.0, earnings - p["personal_allowance"]),
        "income_tax": tax,
        "employee_national_insurance": ni,
        "national_insurance_main_rate_band": max(
            0.0,
            min(earnings, p["ni_upper_earnings_limit_annual"])
            - p["ni_primary_threshold_annual"],
        ),
        "national_insurance_above_uel_band": max(
            0.0, earnings - p["ni_upper_earnings_limit_annual"]
        ),
        "income_tax_band": band,
        "national_insurance_band": ni_band,
        "marginal_deduction_rate": marginal,
        "average_deduction_rate": average,
        "implied_gap_percentage_points": 100.0 * (marginal - average),
        "notes": (
            f"Single worker on {earnings:,.0f} of employment income: {band} for "
            f"income tax, {ni_band} for employee National Insurance. Tax and "
            "National Insurance only, computed from the full band structure "
            "(personal allowance, basic rate to the higher-rate threshold, "
            "higher rate above it; NI at the main rate between the primary "
            "threshold and the upper earnings limit and at the reduced rate "
            "above it). The NI primary threshold is read as its own parameter: "
            "it coincides with the personal allowance only by arithmetic "
            "coincidence in some years, and aliasing the two silently "
            "misstates both rates when they diverge. The marginal rate is what "
            "a small diffuse cut is relieved at; the average rate is what a "
            "complete loss is relieved at, because zeroing the year also "
            "removes the untaxed personal allowance. The difference is the "
            "schedule's own prediction for the sign and rough size of the "
            "paper's headline contrast, with no benefit, pension or household "
            "effect in it. Valid for earnings strictly between "
            f"{validity['valid_earnings_range'][0]:,.0f} and "
            f"{validity['valid_earnings_range'][1]:,.0f}; outside that the "
            "personal allowance taper and the additional rate apply and this "
            "block raises rather than emitting a rate."
        ),
    }


def jsa_block() -> dict:
    """Bound the omitted New Style JSA channel (parameter arithmetic only).

    The factorial decomposition attributes only about 0.1pp of the
    wage-cut-versus-displacement cushioning gap to the employment state
    itself. That near-zero is partly MECHANICAL: every UK institution that
    conditions on employment status sits outside the annual model — New Style
    JSA is not activated by the imposed transition, and work-search
    conditionality carries no fiscal machinery at all. This block states how
    large the largest such omission is, in the same transparent
    parameter-based style as ``monthly_uc_block``: no simulation, explicit
    statutory parameters, arithmetic in the open.
    """
    rate, rate_source, source, vintage = _jsa_weekly_rate()
    weeks = JSA_MAX_DAYS / 7.0
    max_spell = rate * weeks
    e = EXPOSED_MEAN_EARNINGS
    return {
        "parameters": {
            "jsa_weekly_rate_25_plus": rate,
            "jsa_max_days": JSA_MAX_DAYS,
            "jsa_max_weeks": weeks,
            "representative_earnings": e,
            # Machine-readable provenance. `make paper-values` re-runs this
            # block on every build and policyengine-uk is optional, so the
            # printed rate can differ between machines; rate_source records
            # WHICH branch produced it, statutory_fallback_rate records what
            # the other branch would have given, and both are surfaced in the
            # manuscript through \JSARateSource.
            "rate_source": rate_source,
            "rate_source_options": list(JSA_RATE_SOURCES),
            "statutory_fallback_rate": JSA_WEEKLY_RATE_2025_26,
            "weekly_rate_plausible_range": list(JSA_WEEKLY_RATE_PLAUSIBLE_RANGE),
            "source": source,
            "rate_vintage": vintage,
        },
        "max_contribution_based_entitlement": max_spell,
        "gross_loss": e,
        "cushion_points_of_gross_loss": 100.0 * max_spell / e,
        "notes": (
            "New Style JSA is contribution-based and paid for a maximum of "
            f"{JSA_MAX_DAYS} days ({weeks:.0f} weeks) at the standard "
            "personal allowance for claimants aged 25 or over. The figure is "
            "the MAXIMUM contribution-based entitlement over a full "
            "displacement spell for a representative displaced worker on the "
            "exposed-sector mean wage, expressed as percentage points of that "
            "worker's gross earnings loss. It is an UPPER BOUND on the net "
            "addition to cushioning for two reasons: New Style JSA is "
            "taxable, and it is deducted pound for pound from any Universal "
            "Credit award, so a displaced family already receiving UC gains "
            "nothing from it. It also assumes the claimant satisfies the "
            "class 1 contribution conditions and claims for the full 182 "
            "days. Reported to show the scale of an omitted institution, not "
            "as an estimate of what NS-JSA would add."
        ),
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        nargs="*",
        choices=("monthly", "pension", "takeup", "jsa", "schedule"),
        help=(
            "recompute only the named blocks, merging into the existing "
            "results/referee_fixes.json (all blocks when omitted)"
        ),
    )
    args = parser.parse_args()
    blocks = set(args.only or ("monthly", "pension", "takeup", "jsa", "schedule"))

    artifact = RESULTS / "referee_fixes.json"
    out = json.loads(artifact.read_text()) if args.only and artifact.exists() else {}
    dataset = baseline = persons = None
    if blocks & {"pension", "takeup"}:
        dataset, baseline, persons = _baseline_and_persons(DATASET, None, PERIOD)
    if "monthly" in blocks:
        # Runs anywhere: statutory arithmetic over parameters that come from
        # policyengine-uk when it is installed and from the documented
        # constants otherwise. This is what lets the block's NOTE — which was
        # shipped stale and INVERTED in the artifact for want of a rerun — be
        # regenerated from code rather than hand-edited into the JSON.
        out["monthly_uc_bounding"] = monthly_uc_block()
        source = out["monthly_uc_bounding"]["parameters"]["parameter_source"]
        print(f"[monthly] statutory parameter source: {source}")
        if source == PARAMETER_SOURCE_FALLBACK:
            print(
                "[monthly] WARNING: policyengine-uk was not importable here, so "
                "monthly_uc_bounding.parameters are the documented statutory "
                "constants rather than the model's own parameter tree. The "
                "difference is at most a few pounds of National Insurance on "
                "the representative worker (see FALLBACK_SCHEDULE_PARAMETERS), "
                "below the precision of every macro this block feeds, and the "
                "branch that ran is recorded as parameters.parameter_source."
            )
    if "jsa" in blocks:
        out["jsa_bounding"] = jsa_block()
    if "schedule" in blocks:
        monthly = out.get("monthly_uc_bounding")
        if monthly is None:
            raise SystemExit(
                "schedule_benchmark reuses the monthly_uc_bounding parameters; "
                "run `--only monthly schedule` or a full run first."
            )
        out["schedule_benchmark"] = schedule_benchmark_block(monthly)
    if "pension" in blocks:
        out["pension_channel"] = pension_block(dataset, baseline, persons)
    if "takeup" in blocks:
        out["takeup_headline"] = takeup_block(dataset, baseline, persons)
    RESULTS.mkdir(exist_ok=True)
    artifact.write_text(json.dumps(out, indent=2))
    print("[written] results/referee_fixes.json")


if __name__ == "__main__":
    main()
