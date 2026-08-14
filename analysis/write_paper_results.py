"""Generate LaTeX macros for the paper's central numerical results.

The paper must never silently mix Monte Carlo runs with different draw counts
OR different assignment designs. This script reads the eight central
artifacts, checks that they share the declared production draw count AND the
declared record-selection design, and writes a small generated LaTeX file.

It also emits the appendix's exploratory reallocation, supply-chain and
factorial figures (``EXPLORATORY_DRAWS``, ``SCENARIO_TESTING``). Those runs
are small-assignment stress exercises and are checked against the design the
appendix claims for them, NOT against the production draw count: they are
not comparable with the central scenarios and the appendix does not claim
they are.

Why the second check exists. ``shocks.PRESETS`` builds every scenario in
``CENTRAL``, ``ANCHORS`` and ``REQUIRED_TRANSITION`` from
``TradeShockScenario``'s DEFAULT ``selection_method="bernoulli"``, while the
50-draw submission design (``analysis/run_submission_scenarios.py``,
``analysis/factorial_decomposition.py``, ``analysis/bootstrap_uncertainty.py``,
the pension block of ``analysis/referee_fixes.py``) passes
``selection_method="balanced"`` explicitly. The two designs have different
assignment variance by construction, so a draw-count check alone cannot tell
that two artifacts are comparable: 100 Bernoulli draws and 50 balanced draws
are different estimators, not the same estimator at different precision.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CENTRAL = (
    "full_tariff_displacement",
    "full_tariff_wage_cut",
    "full_tariff_inactivity",
    "epd_displacement",
    "epd_wage_cut",
    "epd_inactivity",
    "measured_displacement",
    "measured_wage_cut",
)

#: Optional artifacts: loaded (and draw-count-checked) only if present, so
#: `make paper-values` keeps working for the eight central scenarios before
#: the rent-sharing runs have been produced. Generate them with
#: `make results` (or `python analysis/run_scenarios.py --n-draws 100
#: --scenarios full_tariff_rentsharing epd_rentsharing`).
OPTIONAL = (
    "full_tariff_rentsharing",
    "epd_rentsharing",
)
REQUIRED_TRANSITION = ("full_tariff_transition_central", "epd_transition_central")
ANCHORS = (
    "full_tariff_obr_low_displacement",
    "full_tariff_obr_low_wage_cut",
)

#: Exploratory small-assignment artifacts the appendix quotes, mapped to the
#: assignment count the prose claims for each ("five paired assignments", "the
#: exploratory ten-assignment microsimulation").
#:
#: They are deliberately kept OUT of the comparability check below: each is a
#: 5- or 10-assignment stress exercise, not a production run, so folding them
#: into ``compared`` would trip the 100-draw guard and would also assert a
#: comparability nobody claims. What they get instead is the check that they
#: still carry the design the appendix describes.
#:
#: Why they are read here at all: these numbers used to be typed into the
#: appendix by hand. When the prose was reverted to an earlier draft the main
#: text was re-macro-ified and the appendix was not, so it printed a vintage
#: the paper's own figures contradicted.
EXPLORATORY_DRAWS = {
    "full_tariff_reallocation": 5,
    "full_tariff_reallocation_lag3": 5,
    "full_tariff_reallocation_lowpenalty": 5,
    "supply_chain_displacement": 10,
    "supply_chain_wage_cut": 10,
}

#: Appendix macro prefix per reallocation-penalty variant.
REALLOCATION_PREFIXES = {
    "full_tariff_reallocation": "ReallocCentral",
    "full_tariff_reallocation_lag3": "ReallocLag",
    "full_tariff_reallocation_lowpenalty": "ReallocLowPenalty",
}

#: The factorial adjustment surface behind Figure 1
#: (``analysis/scenario_testing.py``). It stores cells, one per
#: (export-demand calibration, displacement share), rather than draws.
SCENARIO_TESTING = "scenario_testing"
SCENARIO_ASSIGNMENTS_PER_CELL = 5

#: The surface's pure wage-cut endpoint: none of the sector loss is taken on
#: the extensive margin, so one cell per export-demand calibration.
WAGE_CUT_DISPLACEMENT_SHARE = 0.0


#: Where a scenario artifact may record the record-selection design. The first
#: is the shape ``uk_trade_shock_study.runner.write_result`` produces —
#: ``MonteCarloResult`` carries ``selection_method`` and
#: ``run_monte_carlo_prepared`` sets it from ``scenario.selection_method`` — and
#: the second is the shape the hand-written design blocks use
#: (``results/factorial_decomposition.json``'s ``design.selection_method``).
SELECTION_METHOD_KEYS = ("selection_method", ("design", "selection_method"))

#: What ``TradeShockScenario`` uses when a caller does not say. Recorded here
#: only to be named in the warning below — it is deliberately NOT used as a
#: default for an artifact that is silent, because assuming the default is
#: exactly the inference that would paper over a mixed-design comparison.
SCENARIO_DEFAULT_SELECTION_METHOD = "bernoulli"

#: What actually fixes a silent artifact, quoted verbatim in the warnings so
#: the fix is actionable rather than a complaint.
#:
#: The code side is DONE: ``MonteCarloResult`` declares ``selection_method``,
#: ``run_monte_carlo_prepared`` sets it from ``scenario.selection_method``, and
#: ``runner.write_result`` serialises it alongside ``n_draws``. What is stale
#: is the STORED ARTIFACTS: every results/*.json this script reads was written
#: before that field existed, so it carries no design to check. Only a re-run
#: puts the field on disk — the file must not be hand-edited, because a
#: hand-written design string would assert a provenance nobody verified.
SELECTION_METHOD_PROVENANCE_FIX = (
    "This is not a code gap: `uk_trade_shock_study.runner.MonteCarloResult` "
    "already declares `selection_method`, `run_monte_carlo_prepared` sets it "
    "from `scenario.selection_method`, and `runner.write_result` serialises it "
    "into every artifact it writes. The artifacts named above simply predate "
    "that change. Re-run the affected families against the licensed FRS "
    "microdata so they record the field: `make results` for the 100-draw "
    "central, transition, rent-sharing and OBR-low artifacts "
    "(analysis/run_scenarios.py --n-draws 100), and `make submission-results` "
    "for the 50-draw balanced design. Do not hand-edit the JSON: a design "
    "string typed into an artifact asserts a provenance nobody verified. Until "
    "those runs land this guard can only verify artifacts that already carry "
    "the field, and the manuscript's cross-design comparisons are unchecked."
)


def _load(results_dir: Path, name: str) -> dict:
    path = results_dir / f"{name}.json"
    with path.open() as f:
        return json.load(f)


def _selection_method(item: dict) -> str | None:
    """Recorded record-selection design, or None when the artifact is silent.

    Returns None rather than the scenario default: an artifact that does not
    say which design produced it has not been checked, and treating silence as
    "bernoulli" would let a balanced-design artifact pass the guard it exists
    to trip.
    """
    for key in SELECTION_METHOD_KEYS:
        if isinstance(key, tuple):
            value = item
            for part in key:
                value = value.get(part) if isinstance(value, dict) else None
            if isinstance(value, str):
                return value
            continue
        value = item.get(key)
        if isinstance(value, str):
            return value
    return None


def check_selection_methods(
    artifacts: dict[str, dict], expected: str | None = None
) -> dict[str, str | None]:
    """Fail when compared artifacts disagree about the assignment design.

    Three outcomes:

    - two or more artifacts record DIFFERENT designs: raise. They are different
      estimators and the macros this script emits are quoted side by side.
    - some record a design and it is uniform (and matches ``expected`` when one
      is given): pass.
    - none record a design: warn, loudly and specifically. The check cannot be
      performed, and saying so is the honest outcome; inventing the field, or
      inferring it from the draw count, would manufacture a provenance the
      artifacts do not carry.
    """
    recorded = {name: _selection_method(item) for name, item in artifacts.items()}
    present = {name: value for name, value in recorded.items() if value is not None}
    distinct = set(present.values())
    if len(distinct) > 1:
        details = ", ".join(f"{k}={v}" for k, v in sorted(present.items()))
        raise ValueError(
            "Central artifacts must all use the same record-selection design; "
            f"found {sorted(distinct)}: {details}. A balanced assignment and a "
            "Bernoulli assignment are different estimators with different "
            "assignment variance, so their means and SDs cannot be quoted "
            "beside each other without labelling which design produced which. "
            "Re-run the odd artifacts under one design, or label them "
            "explicitly in the manuscript."
        )
    if expected is not None:
        if distinct and distinct != {expected}:
            raise ValueError(
                f"Central artifacts record selection_method {sorted(distinct)}, "
                f"but --expected-selection-method is {expected!r}."
            )
        if not distinct:
            print(
                f"WARNING: --expected-selection-method={expected!r} cannot be "
                "verified: no central artifact records selection_method. "
                f"{SELECTION_METHOD_PROVENANCE_FIX}"
            )
    if not present:
        # Until the 2026-08-13 re-run, every shipped artifact predated the
        # `selection_method` field and this branch could only warn: the design
        # had to be inferred from `shocks.PRESETS` rather than read off the
        # artifact. The re-run records it everywhere, so an artifact that
        # lacks it is now stale rather than merely old, and stale artifacts
        # are exactly what this pipeline exists to catch. Failing here is the
        # difference between a guard that runs and one that reports it could
        # not run.
        raise ValueError(
            "No central artifact records `selection_method`, so the "
            "assignment-design guard cannot run. Every artifact written since "
            "the re-run carries this field; one that does not was produced by "
            "an older build and must not be quoted beside current ones. These "
            "are built from `shocks.PRESETS`, which takes TradeShockScenario's "
            f"default ({SCENARIO_DEFAULT_SELECTION_METHOD!r}), whereas the "
            "50-draw submission design passes 'balanced' explicitly, and the "
            "manuscript compares the two. "
            f"{SELECTION_METHOD_PROVENANCE_FIX}"
        )
    elif len(present) < len(recorded):
        silent = sorted(set(recorded) - set(present))
        print(
            f"WARNING: {len(present)} of {len(recorded)} central artifacts record "
            f"`selection_method` (all {sorted(distinct)[0]!r}); silent: {silent}. "
            f"{SELECTION_METHOD_PROVENANCE_FIX}"
        )
    return recorded


def _fmt(value: float, scale: float = 1.0, digits: int = 1) -> str:
    return f"{value * scale:.{digits}f}"


def _wage_cut_endpoint(scenario: dict) -> tuple[float, float]:
    """Lowest and highest cushioning on the surface's wage-cut endpoint.

    The appendix quotes this range as evidence that the endpoint is stable
    across export-demand calibrations, so it must be read off the artifact
    Figure 1 is drawn from rather than transcribed from it. The two checks
    below pin the two things the sentence asserts: that the cells are the
    five-assignment ones, and that there is exactly one per calibration.
    """
    design = scenario["design"]
    if design["n_assignments_per_cell"] != SCENARIO_ASSIGNMENTS_PER_CELL:
        raise ValueError(
            f"{SCENARIO_TESTING}.json now uses "
            f"{design['n_assignments_per_cell']} assignments per cell; the "
            f"appendix describes {SCENARIO_ASSIGNMENTS_PER_CELL} common "
            "assignments. Update the prose and this constant together."
        )
    endpoint = [
        cell["cushioning_pct_mean"]
        for cell in scenario["cells"]
        if cell["displacement_share"] == WAGE_CUT_DISPLACEMENT_SHARE
    ]
    if len(endpoint) != len(design["elasticities"]):
        raise ValueError(
            "the wage-cut endpoint must hold one cell per export-demand "
            f"calibration: found {len(endpoint)} cells at displacement share "
            f"{WAGE_CUT_DISPLACEMENT_SHARE} against "
            f"{len(design['elasticities'])} elasticities"
        )
    return min(endpoint), max(endpoint)


def _exploratory_macros(results_dir: Path) -> dict[str, str]:
    """Macros for the appendix's exploratory reallocation, supply-chain and
    factorial exercises.

    Every value here is a small-assignment stress figure. The appendix says so
    in each section; this function's only job is to make sure the printed
    numbers come from the artifacts rather than from memory.
    """
    data = {}
    for name, expected in EXPLORATORY_DRAWS.items():
        item = _load(results_dir, name)
        if item["n_draws"] != expected:
            raise ValueError(
                f"{name}.json carries {item['n_draws']} assignments, but the "
                f"appendix describes it as a {expected}-assignment exercise. "
                "Update the prose and EXPLORATORY_DRAWS together."
            )
        data[name] = item

    low, high = _wage_cut_endpoint(_load(results_dir, SCENARIO_TESTING))
    macros = {
        "ScenarioWageEndpointMin": _fmt(low, 1.0, 1),
        "ScenarioWageEndpointMax": _fmt(high, 1.0, 1),
    }

    for name, prefix in REALLOCATION_PREFIXES.items():
        item = data[name]
        macros[f"{prefix}Gross"] = _fmt(
            _mean([d["gross_earnings_loss"] for d in item["draws"]]), 1 / 1e6, 0
        )
        macros[f"{prefix}Exchequer"] = _fmt(item["exchequer_cost_mean"], 1 / 1e6, 0)
        macros[f"{prefix}ExchequerSD"] = _fmt(item["exchequer_cost_sd"], 1 / 1e6, 0)
        macros[f"{prefix}Cushion"] = _fmt(item["cushioning_rate_mean"], 100, 1)
        macros[f"{prefix}CushionSD"] = _fmt(item["cushioning_rate_sd"], 100, 1)

    # The supply-chain pair predates the Universal Credit award-cache
    # correction and is deliberately frozen there (it records no
    # `selection_method`, and its cushioning rate is null). The appendix
    # discloses the vintage; these macros keep the LEVELS it prints tied to
    # the artifact, which is what drifted.
    displaced = data["supply_chain_displacement"]
    wage_cut = data["supply_chain_wage_cut"]
    macros.update(
        {
            "SupplyChainDisplacedWorkers": _fmt(
                displaced["displaced_weighted_mean"], 1 / 1_000, 1
            ),
            "SupplyChainDisplacedExchequer": _fmt(
                displaced["exchequer_cost_mean"], 1 / 1e6, 0
            ),
            "SupplyChainDisplacedExchequerSD": _fmt(
                displaced["exchequer_cost_sd"], 1 / 1e6, 0
            ),
            "SupplyChainDisplacedPoverty": _fmt(
                displaced["poverty_rate_change_bhc_mean"], 100, 3
            ),
            "SupplyChainWageExchequer": _fmt(
                wage_cut["exchequer_cost_mean"], 1 / 1e6, 0
            ),
        }
    )
    return macros


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--output", type=Path, default=Path("paper/generated_results.tex")
    )
    parser.add_argument("--expected-draws", type=int, default=100)
    parser.add_argument(
        "--expected-selection-method",
        default=None,
        help=(
            "assert every central artifact that records a record-selection "
            "design used this one (e.g. 'bernoulli' for the 100-draw legacy "
            "artifacts, 'balanced' for the submission design). Artifacts that "
            "do not record the field cannot be checked and are reported."
        ),
    )
    args = parser.parse_args()

    data = {name: _load(args.results_dir, name) for name in CENTRAL}
    anchor_data = {name: _load(args.results_dir, name) for name in ANCHORS}
    transition_data = {name: _load(args.results_dir, name) for name in REQUIRED_TRANSITION}
    optional_data = {}
    for name in OPTIONAL:
        if (args.results_dir / f"{name}.json").exists():
            optional_data[name] = _load(args.results_dir, name)
        else:
            print(
                f"WARNING: optional artifact {name}.json not found in "
                f"{args.results_dir}; skipping its macros. Generate it with "
                f"`make results` or `python analysis/run_scenarios.py "
                f"--n-draws 100 --scenarios {name}`."
            )
    compared = {**data, **anchor_data, **transition_data, **optional_data}
    draw_counts = {name: item["n_draws"] for name, item in compared.items()}
    if set(draw_counts.values()) != {args.expected_draws}:
        details = ", ".join(f"{k}={v}" for k, v in draw_counts.items())
        raise ValueError(
            f"Central artifacts must all use {args.expected_draws} draws: {details}"
        )
    # Same class of check on the OTHER half of the design. A shared draw count
    # does not make two artifacts comparable if one is a balanced assignment
    # and the other Bernoulli.
    check_selection_methods(compared, args.expected_selection_method)

    fd = data["full_tariff_displacement"]
    fw = data["full_tariff_wage_cut"]
    fi = data["full_tariff_inactivity"]
    ed = data["epd_displacement"]
    ew = data["epd_wage_cut"]
    ei = data["epd_inactivity"]
    md = data["measured_displacement"]
    mw = data["measured_wage_cut"]
    od = anchor_data["full_tariff_obr_low_displacement"]
    ow = anchor_data["full_tariff_obr_low_wage_cut"]
    full_draws = fd["draws"]
    epd_draws = ed["draws"]
    if len(full_draws) != len(epd_draws):
        raise ValueError("Full-tariff and EPD displacement draws must be paired")
    paired_workers = [
        full["displaced_weighted"] - epd["displaced_weighted"]
        for full, epd in zip(full_draws, epd_draws, strict=True)
    ]
    paired_gross = [
        full["gross_earnings_loss"] - epd["gross_earnings_loss"]
        for full, epd in zip(full_draws, epd_draws, strict=True)
    ]
    paired_exchequer = [
        full["exchequer_cost"] - epd["exchequer_cost"]
        for full, epd in zip(full_draws, epd_draws, strict=True)
    ]
    paired_poverty = [
        full["poverty_rate_change_bhc"] - epd["poverty_rate_change_bhc"]
        for full, epd in zip(full_draws, epd_draws, strict=True)
    ]

    macros = {
        "ProductionDraws": str(args.expected_draws),
        "FullDisplacedWorkers": _fmt(fd["displaced_weighted_mean"], 1 / 1_000, 1),
        "FullDisplacedGross": _fmt(
            sum(d["gross_earnings_loss"] for d in fd["draws"]) / len(fd["draws"]),
            1 / 1e6,
            0,
        ),
        "FullDisplacedGrossSD": _fmt(
            _sample_sd([d["gross_earnings_loss"] for d in fd["draws"]]),
            1 / 1e6,
            0,
        ),
        "FullDisplacedExchequer": _fmt(fd["exchequer_cost_mean"], 1 / 1e6, 0),
        "FullDisplacedExchequerSD": _fmt(fd["exchequer_cost_sd"], 1 / 1e6, 0),
        "FullDisplacedExchequerMCSE": _fmt(
            fd["exchequer_cost_mc_se"], 1 / 1e6, 0
        ),
        "FullDisplacedCushion": _fmt(fd["cushioning_rate_mean"], 100, 1),
        "FullDisplacedCushionSD": _fmt(fd["cushioning_rate_sd"], 100, 1),
        "FullDisplacedCushionMCSE": _fmt(
            fd["cushioning_rate_mc_se"], 100, 1
        ),
        "FullDisplacedPoverty": _fmt(fd["poverty_rate_change_bhc_mean"], 100, 3),
        "FullDisplacedPovertySD": _fmt(fd["poverty_rate_change_bhc_sd"], 100, 3),
        "FullWageExchequer": _fmt(fw["exchequer_cost_mean"], 1 / 1e6, 0),
        "FullWageGross": _fmt(
            _mean([d["gross_earnings_loss"] for d in fw["draws"]]), 1 / 1e6, 0
        ),
        "FullWageCushion": _fmt(fw["cushioning_rate_mean"], 100, 1),
        # Appendix reconciliation of the cushioning identity against the
        # Exchequer effect.  These were previously multiplied out by hand in
        # the prose and went stale when the underlying runs were refreshed,
        # so the products and wedges are generated here and the appendix now
        # quotes macros only.
        #
        # The identity is (1 - dY/dE) * dE = dE - dY, which must be averaged
        # PER DRAW.  Using mean(cushioning) * mean(gross) instead computes
        # E[c]E[g], and c and g are correlated across draws because a
        # small-gross draw carries an extreme ratio -- the same pathology the
        # bootstrap's ratio-of-pooled-sums estimator exists to avoid.  On the
        # displacement margin that error was worth GBP 6.5m on the offset and
        # 8 per cent on the wedge; the wage-cut margin is unaffected because
        # its gross loss is deterministic.
        "FullWageImpliedOffset": _fmt(_implied_offset(fw), 1 / 1e6, 0),
        "FullWageExchequerWedge": _fmt(
            fw["exchequer_cost_mean"] - _implied_offset(fw), 1 / 1e6, 0
        ),
        "FullDisplacedImpliedOffset": _fmt(_implied_offset(fd), 1 / 1e6, 0),
        "FullDisplacedExchequerWedge": _fmt(
            fd["exchequer_cost_mean"] - _implied_offset(fd), 1 / 1e6, 0
        ),
        "FullInactiveExchequer": _fmt(fi["exchequer_cost_mean"], 1 / 1e6, 0),
        "FullInactiveExchequerSD": _fmt(fi["exchequer_cost_sd"], 1 / 1e6, 0),
        "FullInactiveCushion": _fmt(fi["cushioning_rate_mean"], 100, 1),
        "FullInactiveCushionSD": _fmt(fi["cushioning_rate_sd"], 100, 1),
        "EPDDisplacedWorkers": _fmt(ed["displaced_weighted_mean"], 1 / 1_000, 1),
        "EPDDisplacedGross": _fmt(
            _mean([d["gross_earnings_loss"] for d in ed["draws"]]), 1 / 1e6, 0
        ),
        "EPDDisplacedGrossSD": _fmt(
            _sample_sd([d["gross_earnings_loss"] for d in ed["draws"]]),
            1 / 1e6,
            0,
        ),
        "EPDDisplacedExchequer": _fmt(ed["exchequer_cost_mean"], 1 / 1e6, 0),
        "EPDDisplacedExchequerSD": _fmt(ed["exchequer_cost_sd"], 1 / 1e6, 0),
        "EPDDisplacedCushion": _fmt(ed["cushioning_rate_mean"], 100, 1),
        "EPDDisplacedCushionSD": _fmt(ed["cushioning_rate_sd"], 100, 1),
        "EPDDisplacedPoverty": _fmt(ed["poverty_rate_change_bhc_mean"], 100, 3),
        "EPDDisplacedPovertySD": _fmt(ed["poverty_rate_change_bhc_sd"], 100, 3),
        "EPDWageExchequer": _fmt(ew["exchequer_cost_mean"], 1 / 1e6, 0),
        "EPDWageGross": _fmt(
            _mean([d["gross_earnings_loss"] for d in ew["draws"]]), 1 / 1e6, 0
        ),
        "EPDWageCushion": _fmt(ew["cushioning_rate_mean"], 100, 1),
        "EPDInactiveExchequer": _fmt(ei["exchequer_cost_mean"], 1 / 1e6, 0),
        "EPDInactiveExchequerSD": _fmt(ei["exchequer_cost_sd"], 1 / 1e6, 0),
        "EPDInactiveCushion": _fmt(ei["cushioning_rate_mean"], 100, 1),
        "EPDInactiveCushionSD": _fmt(ei["cushioning_rate_sd"], 100, 1),
        "MeasuredDisplacedWorkers": _fmt(md["displaced_weighted_mean"], 1 / 1_000, 1),
        "MeasuredDisplacedExchequer": _fmt(md["exchequer_cost_mean"], 1 / 1e6, 0),
        "MeasuredDisplacedExchequerSD": _fmt(md["exchequer_cost_sd"], 1 / 1e6, 0),
        "MeasuredDisplacedCushion": _fmt(md["cushioning_rate_mean"], 100, 1),
        "MeasuredDisplacedCushionSD": _fmt(md["cushioning_rate_sd"], 100, 1),
        "MeasuredWageExchequer": _fmt(mw["exchequer_cost_mean"], 1 / 1e6, 0),
        "MeasuredWageCushion": _fmt(mw["cushioning_rate_mean"], 100, 1),
        "OBRLowDisplacedWorkers": _fmt(
            od["displaced_weighted_mean"], 1 / 1_000, 1
        ),
        "OBRLowDisplacedGross": _fmt(
            _mean([d["gross_earnings_loss"] for d in od["draws"]]), 1 / 1e6, 0
        ),
        "OBRLowDisplacedGrossSD": _fmt(
            _sample_sd([d["gross_earnings_loss"] for d in od["draws"]]),
            1 / 1e6,
            0,
        ),
        "OBRLowDisplacedExchequer": _fmt(
            od["exchequer_cost_mean"], 1 / 1e6, 0
        ),
        "OBRLowDisplacedExchequerSD": _fmt(
            od["exchequer_cost_sd"], 1 / 1e6, 0
        ),
        "OBRLowDisplacedCushion": _fmt(
            od["cushioning_rate_mean"], 100, 1
        ),
        "OBRLowDisplacedCushionSD": _fmt(
            od["cushioning_rate_sd"], 100, 1
        ),
        "OBRLowDisplacedPoverty": _fmt(
            od["poverty_rate_change_bhc_mean"], 100, 3
        ),
        "OBRLowDisplacedPovertySD": _fmt(
            od["poverty_rate_change_bhc_sd"], 100, 3
        ),
        "OBRLowCushionValidDraws": str(od["cushioning_valid_draws"]),
        "OBRLowWageGross": _fmt(
            _mean([d["gross_earnings_loss"] for d in ow["draws"]]), 1 / 1e6, 0
        ),
        "OBRLowWageExchequer": _fmt(
            ow["exchequer_cost_mean"], 1 / 1e6, 0
        ),
        "OBRLowWageCushion": _fmt(ow["cushioning_rate_mean"], 100, 1),
        "EPDWorkerDifference": _fmt(_mean(paired_workers), 1.0, 0),
        "EPDWorkerDifferenceSD": _fmt(_sample_sd(paired_workers), 1.0, 0),
        "EPDWorkerDifferenceThousands": _fmt(
            _mean(paired_workers), 1 / 1_000, 1
        ),
        "EPDWorkerDifferenceSDThousands": _fmt(
            _sample_sd(paired_workers), 1 / 1_000, 1
        ),
        "EPDGrossDifference": _fmt(_mean(paired_gross), 1 / 1e6, 0),
        "EPDGrossDifferenceSD": _fmt(_sample_sd(paired_gross), 1 / 1e6, 0),
        "EPDExchequerDifference": _fmt(_mean(paired_exchequer), 1 / 1e6, 0),
        "EPDExchequerDifferenceSD": _fmt(_sample_sd(paired_exchequer), 1 / 1e6, 0),
        "EPDPovertyDifference": _fmt(_mean(paired_poverty), 100, 4),
        "EPDPovertyDifferenceSD": _fmt(_sample_sd(paired_poverty), 100, 4),
    }

    # Rent-sharing-calibrated mixed margin (optional; skipped with a warning
    # above when the artifacts have not been generated yet).
    _rentsharing_prefixes = {
        "full_tariff_rentsharing": "FullRentSharing",
        "epd_rentsharing": "EPDRentSharing",
    }
    for name, prefix in _rentsharing_prefixes.items():
        item = optional_data.get(name)
        if item is None:
            continue
        macros[f"{prefix}Exchequer"] = _fmt(item["exchequer_cost_mean"], 1 / 1e6, 0)
        macros[f"{prefix}ExchequerSD"] = _fmt(item["exchequer_cost_sd"], 1 / 1e6, 0)
        macros[f"{prefix}Cushion"] = _fmt(item["cushioning_rate_mean"], 100, 1)
        macros[f"{prefix}CushionSD"] = _fmt(item["cushioning_rate_sd"], 100, 1)
        macros[f"{prefix}Workers"] = _fmt(item["displaced_weighted_mean"], 1 / 1_000, 1)
        macros[f"{prefix}Poverty"] = _fmt(item["poverty_rate_change_bhc_mean"], 100, 3)

    _transition_prefixes = {
        "full_tariff_transition_central": "FullTransition",
        "epd_transition_central": "EPDTransition",
    }
    for name, prefix in _transition_prefixes.items():
        item = transition_data[name]
        macros[f"{prefix}Exchequer"] = _fmt(item["exchequer_cost_mean"], 1 / 1e6, 0)
        macros[f"{prefix}ExchequerSD"] = _fmt(item["exchequer_cost_sd"], 1 / 1e6, 0)
        macros[f"{prefix}Cushion"] = _fmt(item["cushioning_rate_mean"], 100, 1)
        macros[f"{prefix}CushionSD"] = _fmt(item["cushioning_rate_sd"], 100, 1)
        macros[f"{prefix}Gross"] = _fmt(_mean([d["gross_earnings_loss"] for d in item["draws"]]), 1 / 1e6, 0)
        macros[f"{prefix}Workers"] = _fmt(item.get("reallocated_weighted_mean", 0.0), 1 / 1_000, 1)

    macros.update(_exploratory_macros(args.results_dir))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    lines = ["% Generated by analysis/write_paper_results.py; do not edit by hand."]
    lines.extend(
        f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in macros.items()
    )
    args.output.write_text("\n".join(lines) + "\n")


def _sample_sd(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((value - mean) ** 2 for value in values) / (len(values) - 1)) ** 0.5


def _implied_offset(result: dict) -> float:
    """Household-side offset dE - dY, averaged over draws.

    The cushioning identity (1 - dY/dE) * dE collapses to dE - dY exactly, so
    the offset must be formed draw by draw and then averaged.  Averaging the
    ratio and the gross loss separately is not the same statistic when the
    two are correlated across draws, which they are on any stochastic margin.
    """
    draws = result["draws"]
    if not draws:
        raise ValueError("cannot compute the implied offset with no draws")
    return _mean(
        [d["gross_earnings_loss"] - d["net_disposable_loss"] for d in draws]
    )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


if __name__ == "__main__":
    main()
