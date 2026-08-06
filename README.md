# uk-trade-shock-study

How does the UK tax-benefit system respond to labour-income stress scenarios
calibrated to the 2025 US tariffs on UK goods? The project imposes a
reduced-form tariff-exposure-to-wage-bill bridge on FRS 2024-25 microdata and
runs it through PolicyEngine UK. It does not model the intervening effects on
prices, quantities, production, value added, productivity or labour demand.
Adjustment scenarios include **displacement**, **wage cuts**, a
**concentrated wage cut** factorial middle cell (the displacement draw's
exact worker-level losses without job loss), **inactivity**,
**reallocation**, a common-loss **temporary transition** path (lagged
re-employment plus survivor earnings cuts), and a factorial **mixed
wage/job-loss** family. An Economic
Prosperity Deal counterfactual
(full tariffs vs deal-mitigated: autos 25%→10% in-quota, conditional steel
relief, pharma exempt) prices the deal for households and the Exchequer.

Companion to [uk-ai-study](../uk-ai-study), whose conventions this repo
mirrors. Literature and scenario design: `tariff_paper_lit_review.md`.
The disposition of the referee audit is tracked in `REVISION_STATUS.md`.

## Pipeline

1. `analysis/build_trade_by_sic.py` — rebuilds US-export intensity by SIC
   division from HMRC uktradeinfo exports and ONS Annual Business Survey
   turnover. The packaged
   `uk_trade_shock_study/data/us_export_intensity_by_sic.csv` is the current
   real-data build; provenance and judgement calls are documented in the
   paper appendix and build script.
2. `analysis/download_data.py` — FRS microdata from PolicyEngine's Hugging
   Face repo (needs `HUGGING_FACE_TOKEN` with access to
   `policyengine/policyengine-uk-data`); lands in `data/` (gitignored).
3. `analysis/run_scenarios.py` — runs the {full_tariff, epd} ×
   {displacement, wage_cut, inactivity} presets with Monte Carlo draws
   (mean ± SD) and writes `results/*.json`.
4. `analysis/figures.py` — paper figures (PolicyEngine house style,
   `analysis/figstyle.py`).
5. `analysis/scenario_testing.py` — crosses export-demand calibration with
   the wage-cut/displacement mixture on common seeds and writes the scenario
   surface, cell data and draw-level artifact.
6. `analysis/hmrc_destination_event_study.py` — builds a balanced public HS4
   product panel and tests whether exports to the US changed relative to the
   same products sent to Canada, Japan and Australia, with weighting,
   control-destination, sample-start, placebo-date and tariff-intensity
   diagnostics.
7. `analysis/run_submission_scenarios.py` — runs the pre-specified primary
   design: {OBR-style 0.4, unit} × {wage cut, displacement} × {3, 6, 12 month
   duration-equivalent annual stress}. Displacement uses balanced repeated
   assignment and retains an independent Bernoulli comparator.
8. `analysis/write_submission_results.py` — validates all 50-draw submission
   artifacts, writes the primary manuscript table, paired margin contrasts
   and record-support diagnostics.
9. `analysis/factorial_decomposition.py` — runs the concentrated-wage-cut
   middle cell (the displacement draw's exact worker-level losses with no
   job loss, paired seeds) and decomposes the headline wage-cut vs
   displacement gap into a concentration/worker-selection step and an
   employment-state step, with a channel split on common seeds.
10. `analysis/leave_one_record_out.py` — removes each loss-contributing FRS
    household from both margins and recomputes the primary contrast exactly
    from the stored per-household bootstrap contributions.
11. `analysis/write_factorial_results.py` — validates both new artifacts and
    emits their manuscript macros.

The manuscript builds as a short main paper (`paper/main.tex`) plus a
standalone online supplement (`paper/supplement.tex`) carrying the HMRC
event-study detail, LFS imputation benchmarks, EPD results detail and the
exploratory sensitivity sections.

The legacy scenario suite uses 100 assignment draws. The compact submission
design uses 50 balanced draws; both specifications are declared explicitly.

`analysis/write_paper_results.py` checks that all central artifacts use the
same 100-draw production specification and generates
`paper/generated_results.tex`. The paper build fails rather than silently
mixing exploratory and production results. The corresponding submission
writer enforces 50 draws independently.

`make uncertainty-design` writes a seeded 500-draw Latin-hypercube design for
elasticity, wage-bill incidence, UC take-up, reallocation penalty, and the
displacement share. It is a parameter-sensitivity design, not a confidence
interval, and can be consumed by an expensive licensed-data run.

## Package

- `uk_trade_shock_study/exposure.py` — tariff schedule (both scenarios),
  US-export intensity, derived per-SIC earnings shocks, FRS SIC join.
- `analysis/download_hmrc_panel.py` — credential-free public HMRC
  product-by-destination monthly export-panel download.
- `analysis/download_bres_benchmarks.py` — credential-free open Nomis BRES
  manufacturing employment benchmarks (rounded aggregates, not microdata).
- `analysis/impute_lfs_to_frs.py` — locally estimates five-quarter LFS
  employment/wage-transition cells, aligns sector mass to public BRES, and
  attaches credibility-shrunk and aggregate-calibrated parameters to FRS
  employee records.
- `analysis/benchmark_lfs_qrf.py` — runs a regularised QRF robustness
  benchmark on all employed LFS adults for model shape, then calibrates its
  manufacturing level to the same direct LFS/BRES targets as the primary
  estimator. `make lfs-benchmarks LFS_TAB=/path/to/panel.tab` rebuilds both.
  These outputs are predictive imputations, not linked ASHE evidence or
  tariff-effect estimates.
- `uk_trade_shock_study/shocks.py` — pure and mixed adjustment-margin families;
  hard-errors if the employment_status transition fails to apply.
- `uk_trade_shock_study/runner.py` — PolicyEngine UK runs: disposable income,
  relative/absolute BHC + AHC poverty, Gini, decile/region breakdowns,
  Exchequer effect, Monte Carlo support.
- `uk_trade_shock_study/channels.py` — calibrated observable heterogeneity and
  a real-income price-channel interface; these require external estimates or
  expenditure data before causal use.
- `uk_trade_shock_study/uncertainty.py` — reproducible Latin-hypercube draws
  for structured parameter sensitivity (not confidence intervals).
- `uk_trade_shock_study/policy_counterfactuals.py` — transparent wage-insurance
  and targeted-transfer accounting helpers for subsequent PolicyEngine runs.

## Setup

```sh
uv sync --extra dev --frozen
make check         # synthetic tests + frozen-input manifest validation
```

`uv.lock` pins the Python 3.13 environment. CI runs the same lightweight
contract without licensed data. With the licensed/raw files listed in
`uk_trade_shock_study/data/input_manifest.json` present under `data/`, run
`make reproduce` to rebuild results, figures and the paper. The manifest
records source URLs, retrieval dates, vintages, exclusions and hashes; missing
licensed inputs are never downloaded or redistributed implicitly.

## Data requirements

- FRS 2024-25 h5 + adult.tab (licensed; via download_data.py) — gitignored.
- HMRC uktradeinfo country-by-commodity exports and ONS Annual Business Survey
  turnover are used to rebuild the packaged intensity table.

Public historical HMRC data require no credential. `make public-hmrc` builds
an interruptible, cached 2018--2026 monthly commodity-destination export panel
for the United States, Canada, Japan and Australia under `data/public/`.
`make trade-event-study` then rebuilds the public destination-panel benchmark.

The longitudinal LFS input is UKDS End User Licence data rather than Secure
Lab microdata. It must be downloaded by a registered user and supplied via
`LFS_TAB`; it is never copied into the repository. Secure ASHE is not required
for the implemented benchmark.

## Paper

`paper/main.tex` is the complete manuscript, including the data, mechanics,
Monte Carlo, sensitivity, and secondary-results appendices. It is the only
paper PDF produced by the build. All numerical values draw from validated
generated files.
The paper reports a static, partial-equilibrium, first-round fiscal-incidence
stress test conditional on imposed labour-income changes. It is not a causal
estimate of the tariffs' production, productivity, employment, macroeconomic
or total household-welfare effects. Licensed FRS inputs are not distributed.
