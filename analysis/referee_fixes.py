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

Note on runnability: the ``monthly`` block needs policyengine-uk's parameter
tree and the ``pension``/``takeup`` blocks need the licensed FRS microdata,
but ``--only jsa`` is pure arithmetic and runs anywhere.
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


def monthly_uc_block() -> dict:
    """Parameter-based bounding of the annual model against a monthly UC
    assessment for a representative single displaced worker (aged 25+,
    no children, no housing element, capital below the lower limit).
    """
    from policyengine_uk.system import system

    inst = f"{PERIOD}-01-01"
    gov = system.parameters.gov
    sa_month = float(gov.dwp.universal_credit.standard_allowance.amount.SINGLE_OLD(inst))
    pa = float(gov.hmrc.income_tax.allowances.personal_allowance.amount(inst))
    brackets = gov.hmrc.income_tax.rates.uk.brackets
    basic = float(brackets[0].rate(inst))
    higher = float(brackets[1].rate(inst))
    higher_threshold = float(brackets[1].threshold(inst))
    ni_main = float(gov.hmrc.national_insurance.class_1.rates.employee.main(inst))
    ni_pt_week = float(gov.hmrc.national_insurance.class_1.thresholds.primary_threshold(inst))
    e = EXPOSED_MEAN_EARNINGS

    def income_tax(y: float) -> float:
        taxable = max(0.0, y - pa)
        return basic * min(taxable, higher_threshold) + higher * max(
            0.0, taxable - higher_threshold
        )

    def ni_annualised(y: float) -> float:
        return ni_main * max(0.0, y - ni_pt_week * 52)

    rows = {}
    for m in (3, 6, 12):
        f = m / 12
        # Monthly-correct: m months out of work at full standard allowance
        # (zero earned income), (12-m) months at earnings that taper UC to
        # zero; PAYE reconciles to annual tax on partial-year earnings.
        uc_monthly = m * sa_month
        tax_relief_monthly = income_tax(e) - income_tax((1 - f) * e)
        ni_relief_monthly = ni_main * max(0.0, e / 12 - ni_pt_week * 52 / 12) * m
        # Annual duration-equivalent stress: probability f of a coherent
        # full-year displaced state; expectations per exposed worker.
        uc_annual = f * 12 * sa_month
        tax_relief_annual = f * income_tax(e)
        ni_relief_annual = f * ni_annualised(e)
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
    return {
        "parameters": {
            "standard_allowance_single_25_plus_month": sa_month,
            "personal_allowance": pa,
            "basic_rate": basic,
            "higher_rate": higher,
            "ni_employee_main": ni_main,
            "representative_earnings": e,
        },
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
    """Hard-fail a rate that cannot be a weekly NS-JSA personal allowance."""
    low, high = JSA_WEEKLY_RATE_PLAUSIBLE_RANGE
    if not np.isfinite(rate) or not low <= rate <= high:
        raise RuntimeError(
            f"{where} returned {rate!r}, outside the plausible weekly NS-JSA "
            f"range {JSA_WEEKLY_RATE_PLAUSIBLE_RANGE} for a claimant aged 25 "
            "or over. This is what a periodicity change looks like (a monthly "
            "or annual node read as weekly is wrong by a factor of 4.33 or "
            "52), and the JSA bound multiplies this figure by "
            f"{JSA_MAX_DAYS / 7:.0f} weeks. Fix the parameter path or the "
            "range; do not accept the value."
        )
    return rate


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
    params = monthly["parameters"]
    earnings = float(params["representative_earnings"])
    allowance = float(params["personal_allowance"])
    basic = float(params["basic_rate"])
    ni = float(params["ni_employee_main"])
    taxable = earnings - allowance
    income_tax = basic * taxable
    employee_ni = ni * taxable
    marginal = basic + ni
    average = (income_tax + employee_ni) / earnings
    return {
        "parameters": {
            "representative_earnings": earnings,
            "personal_allowance": allowance,
            "basic_rate": basic,
            "ni_employee_main": ni,
            "higher_rate_threshold": 50_270.0,
        },
        "taxable_income": taxable,
        "income_tax": income_tax,
        "employee_national_insurance": employee_ni,
        "marginal_deduction_rate": marginal,
        "average_deduction_rate": average,
        "implied_gap_percentage_points": 100.0 * (marginal - average),
        "notes": (
            "Single worker below the higher-rate threshold, tax and National "
            "Insurance only. The marginal rate is what a small diffuse cut is "
            "relieved at; the average rate is what a complete loss is relieved "
            "at, because zeroing the year also removes the untaxed personal "
            "allowance. The difference is the schedule's own prediction for "
            "the sign and rough size of the paper's headline contrast, with no "
            "benefit, pension or household effect in it."
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
        out["monthly_uc_bounding"] = monthly_uc_block()
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
