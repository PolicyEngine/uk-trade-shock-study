"""Run all tariff/margin presets and write results to results/.

Usage: python analysis/run_scenarios.py [--data-dir DATA] [--period 2026]
       [--n-draws 100] [--scenarios full_tariff_displacement ...]
       python analysis/run_scenarios.py --e3     (the paper's E3 results)
       python analysis/run_scenarios.py --e3 --e3-part wage_cut   (wage cut only)

The paper's E3 files are produced by two commands:
    run_scenarios.py --n-draws 20 --scenarios full_tariff_displacement
        -> results/full_tariff_displacement_bernoulli.json (comparator)
    run_scenarios.py --e3
        -> results/full_tariff_displacement{,_no_jsa}.json,
           results/full_tariff_obr_low_displacement{,_no_jsa}.json,
           results/full_tariff{,_obr_low}_wage_cut.json
"""

import argparse
from pathlib import Path

from tariff_pipeline.runner import run_monte_carlo, write_result
from tariff_pipeline.shocks import (
    MIXED_CENTRAL_PRESETS,
    TRANSITION_PRESETS,
    PRESETS,
    RENT_SHARING_PRESETS,
    TradeShockScenario,
)
from tariff_pipeline.exposure import ELASTICITY_SCENARIOS

#: The MEASURED family: per-division shocks from the realised HMRC OTS
#: outturn (exposure.MEASURED_SCENARIO; built by build_measured_shocks.py)
#: rather than elasticity x tariff. One tariff scenario (the outturn embeds
#: the EPD), run under the displacement and wage-cut margins.
MEASURED_PRESETS = {
    f"measured_{margin}": TradeShockScenario(f"measured_{margin}", "measured", margin)
    for margin in ("displacement", "wage_cut")
}
# Literature-disciplined mixed sensitivity: {tariff}_rentsharing runs the mixed
# margin with displacement_share = 1 - RENT_SHARING_ELASTICITY = 0.85, so
# survivor wage cuts absorb 15% of each imposed sector wage-bill loss. A
# rent-sharing elasticity does not identify this decomposition one-for-one.
ALL_PRESETS = {
    **PRESETS,
    **MEASURED_PRESETS,
    **RENT_SHARING_PRESETS,
    **MIXED_CENTRAL_PRESETS,
    **TRANSITION_PRESETS,
}
OBR_LOW_PRESETS = {
    f"full_tariff_obr_low_{margin}": TradeShockScenario(
        f"full_tariff_obr_low_{margin}",
        "full_tariff",
        margin,
        elasticity=ELASTICITY_SCENARIOS["obr_low"],
    )
    for margin in ("displacement", "wage_cut")
}
ALL_PRESETS.update(OBR_LOW_PRESETS)

#: E3 (referee rerun). The displacement margin is evaluated by the
#: exact-expectation design (tariff_pipeline/expectation.py) with the New
#: Style JSA approximation ON in the central case and OFF as a variant. The
#: wage-cut margin displaces nobody and is deterministic apart from the UC
#: claiming re-draw, which is mixed exactly from an all-claim and a
#: none-claim simulation -- the same take-up treatment as the displacement
#: margin (expectation.wage_cut_expectation_result). It formerly averaged
#: three random claiming draws.
E3_PARTS = ("displacement", "wage_cut")
E3_ELASTICITIES = {
    "full_tariff": None,  # scenario default
    "full_tariff_obr_low": ELASTICITY_SCENARIOS["obr_low"],
}


#: Result names owned by the exact-expectation run (see ``main``).
E3_EXACT_OUTPUTS = ("full_tariff_displacement", "full_tariff_obr_low_displacement")


def _e3_displacement(prefix: str, elasticity, new_style_jsa: bool) -> TradeShockScenario:
    kwargs = {} if elasticity is None else {"elasticity": elasticity}
    name = f"{prefix}_displacement" + ("" if new_style_jsa else "_no_jsa")
    return TradeShockScenario(
        name, "full_tariff", "displacement", new_style_jsa=new_style_jsa, **kwargs
    )


def run_e3(
    period: int,
    results: Path,
    pass_cache: Path | None = None,
    parts: tuple[str, ...] = E3_PARTS,
) -> None:
    """Exact-expectation displacement and exact-take-up wage cut for E3."""
    import json
    import pickle

    from tariff_pipeline.expectation import (
        expectation_result,
        run_passes,
        validate_against_draw,
        wage_cut_expectation_result,
    )
    from tariff_pipeline.runner import _baseline_and_persons, _metrics, json_value

    dataset, baseline, persons = _baseline_and_persons(None, None, period)
    base_metrics = _metrics(baseline, period)
    equiv = baseline.calculate(
        "equiv_hbai_household_net_income", period=period, map_to="person"
    ).values
    region = baseline.calculate("region", period=period, map_to="person").values.astype(str)

    def write(payload, name):
        (results / f"{name}.json").write_text(
            json.dumps(json_value(payload), indent=2, allow_nan=False)
        )

    for new_style_jsa in (True, False) if "displacement" in parts else ():
        central = _e3_displacement("full_tariff", None, new_style_jsa)
        passes = run_passes(dataset, baseline, persons, central, period, base_metrics)
        if pass_cache is not None:
            pass_cache.mkdir(parents=True, exist_ok=True)
            with open(pass_cache / f"passes_jsa_{int(new_style_jsa)}.pkl", "wb") as f:
                pickle.dump(passes, f)
        validation = validate_against_draw(
            dataset, baseline, persons, central, passes, period, seed=0
        )
        print(f"[expectation] validation draw: {validation}")
        for prefix, elasticity in E3_ELASTICITIES.items():
            scenario = _e3_displacement(prefix, elasticity, new_style_jsa)
            out = expectation_result(
                passes, persons, scenario, base_metrics, equiv, region
            )
            out["validation_draw"] = validation
            write(out, scenario.name)
            print(
                f"{scenario.name}: gross £{out['gross_earnings_loss'] / 1e6:,.1f}m, "
                f"cushioning {100 * out['cushioning_rate']:.2f}% "
                f"(sampling SE {100 * out['cushioning_rate_sampling_se']:.2f}), "
                f"split of gross "
                f"{ {k: round(100 * v, 2) for k, v in out['cushion_component_shares']['share_of_gross_loss'].items()} }"
            )
        del passes

    wage_names = ("full_tariff_wage_cut", "full_tariff_obr_low_wage_cut")
    for name in wage_names if "wage_cut" in parts else ():
        out = wage_cut_expectation_result(
            dataset, baseline, persons, ALL_PRESETS[name], period, base_metrics,
            equiv, region,
        )
        write(out, name)
        print(
            f"{name}: cushioning {100 * out['cushioning_rate']:.2f}% "
            f"(all claim {100 * out['takeup_nodes']['1']['cushioning_rate']:.2f}, "
            f"none claim {100 * out['takeup_nodes']['0']['cushioning_rate']:.2f}; "
            f"{out['redrawn_benefit_units']} benefit units re-drawn), split of gross "
            f"{ {k: round(100 * v, 2) for k, v in out['cushion_component_shares']['share_of_gross_loss'].items()} }"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--period", type=int, default=2026)
    parser.add_argument("--n-draws", type=int, default=100)
    parser.add_argument("--scenarios", nargs="*", default=list(ALL_PRESETS))
    parser.add_argument(
        "--e3",
        action="store_true",
        help="run the E3 set: exact-expectation displacement (with and "
        "without New Style JSA, central and OBR-low elasticity) and the "
        "wage cut with the take-up re-draw mixed exactly; ignores "
        "--scenarios and --n-draws",
    )
    parser.add_argument(
        "--e3-part",
        nargs="*",
        choices=E3_PARTS,
        default=list(E3_PARTS),
        help="with --e3: which margins to (re)run",
    )
    parser.add_argument("--pass-cache", default=None, help="directory to pickle E3 passes")
    args = parser.parse_args()
    if args.e3:
        results = Path("results")
        results.mkdir(exist_ok=True)
        run_e3(
            args.period,
            results,
            Path(args.pass_cache) if args.pass_cache else None,
            tuple(args.e3_part),
        )
        return

    data = Path(args.data_dir)
    dataset = None  # certified Microcosm UK via policyengine.py
    # SIC comes from the h5's sic_industry_division variable; pass
    # adult_tab_path=... to run_monte_carlo to use the legacy adult.tab join.
    results = Path("results")
    results.mkdir(exist_ok=True)

    # Print the imposed aggregate gross earnings stress per scenario.
    from uk_trade_shock_study.microcosm import microsimulation

    import pandas as pd

    from tariff_pipeline.exposure import (
        sector_earnings_shocks,
        simulation_sic_division,
    )

    sim = microsimulation()
    sic = simulation_sic_division(sim, args.period)
    emp = sim.calculate("employment_income", period=args.period, map_to="person").values
    w = sim.calculate("person_weight", period=args.period, map_to="person").values
    for tariff in ("full_tariff", "epd", "measured"):
        shock = pd.Series(sic).map(sector_earnings_shocks(tariff)).fillna(0.0).to_numpy()
        loss = float((shock * emp * w).sum())
        print(f"[calibration] {tariff}: aggregate gross earnings loss £{loss / 1e9:.2f}bn/yr")

    for name in args.scenarios:
        result = run_monte_carlo(
            dataset, ALL_PRESETS[name], period=args.period, n_draws=args.n_draws
        )
        # results/{name}.json for the E3 displacement scenarios is the
        # exact-expectation estimate written by --e3; a Monte Carlo run of
        # the same preset (Bernoulli draws, no New Style JSA) is the
        # COMPARATOR and must not overwrite it.
        suffix = "_bernoulli" if name in E3_EXACT_OUTPUTS else ""
        write_result(result, results / f"{name}{suffix}.json")
        print(
            f"{name}: exchequer £{result.exchequer_cost_mean/1e9:.2f}bn "
            f"(SD {result.exchequer_cost_sd/1e9:.2f}), "
            f"poverty BHC {result.poverty_rate_change_bhc_mean*100:+.2f}pp, "
            f"gini {result.gini_change_mean:+.4f}"
        )


if __name__ == "__main__":
    main()
