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
relief, pharma exempt) is retained in the code but **withdrawn as a
quantitative result**: the paired full-minus-EPD contrast is comparable to
or smaller than its own assignment dispersion on every quantity reported,
and it prices a change in a sector calibration whose own rank-order check
does not corroborate it. The main paper's results and policy sections state
the withdrawal and the detail lives in the online supplement as a
qualitative illustration of the counterfactual machinery only.

Companion to [uk-ai-study](../uk-ai-study), whose conventions this repo
mirrors. The disposition of successive referee rounds is tracked in
`REVISION_STATUS.md`.

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
   **Pinned by revision and verified by hash.** Upstream republishes
   `frs_2024_25.h5` under the same name on every data release, so a bare
   download does not reproduce these results — the file behind them
   (release 1.56.6, 2026-06-19) was replaced on 2026-07-21 and again on
   2026-07-26. The script fetches the revision recorded in
   `uk_trade_shock_study/data/input_manifest.json` and aborts if the bytes do
   not match its `sha256`. Use `--latest` to test against current upstream
   data; that is a robustness check, not a reproduction.
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
standalone online supplement (`paper/supplement.tex` +
`paper/sections/supplement_body.tex`). The main paper keeps the core
appendices (provenance, shock mechanics, Monte Carlo conventions, factorial
scenarios, duration scope, monthly-UC bounding); the supplement carries the
elasticity grid, reallocation sensitivity, observed-outturn stress, HMRC
destination-panel diagnostic, the EPD counterfactual (withdrawn as a
quantitative result), and the demographic, supply-chain and constituency
exercises. The LFS imputation benchmarks stay in the main methodology and
results sections.

The legacy scenario suite uses 100 assignment draws. The compact submission
design uses 50 balanced draws; both specifications are declared explicitly.

`analysis/write_paper_results.py` checks that all central artifacts use the
same 100-draw production specification and generates
`paper/generated_results.tex`. The paper build fails rather than silently
mixing exploratory and production results. The corresponding submission
writer enforces 50 draws independently.
`analysis/write_validation_macros.py` generates `paper/generated_validation.tex`
so the trade-benchmark figures quoted in the manuscript come from
`results/validation_sectors.json` rather than from prose; the customs totals
additionally require `results/trade_build_totals.json`, written by
`make inputs` from the raw HMRC/ONS inputs, and their macros are emitted only
when that artifact exists.

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
  Both per-record outputs carry FRS person records (`person_id`, age, gender,
  employment income, survey weight), so they are **gitignored and never
  distributed**. The manuscript reads
  `results/lfs_qrf_benchmark_summary.json`, an aggregate of weighted means
  and deciles that the same script writes alongside them.
  These outputs are predictive imputations, not linked ASHE evidence or
  tariff-effect estimates.
- `uk_trade_shock_study/shocks.py` — pure and mixed adjustment-margin families;
  hard-errors if the employment_status transition fails to apply. UC claiming
  is re-drawn post-shock under a declared `uc_takeup_scope`: `new_entitlement`
  (the default, and the specification behind every stored result) re-draws
  only benefit units moving from zero to positive entitlement, which turns
  out to be an **empty set** in production runs; `all_entitled` re-draws every
  changed entitled unit and is the scope in which the claiming assumption
  actually binds. `make takeup-entitled` runs both at the current calibration.
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
make check         # synthetic tests + manifest validation + regenerates all paper macros (drift gate)
```

`uv.lock` pins the Python 3.13 environment. CI runs the same lightweight
contract without licensed data. With the licensed/raw files listed in
`uk_trade_shock_study/data/input_manifest.json` present under `data/`, run
`make reproduce` to rebuild results, figures and the paper. The manifest
records source URLs, retrieval dates, vintages, exclusions and hashes; missing
licensed inputs are never downloaded or redistributed implicitly.

## Data requirements

- FRS 2024-25 h5 + adult.tab (licensed; via download_data.py) — gitignored.
  The h5 is pinned to upstream revision `5535b2f8` (policyengine-uk-data
  1.56.6) and hash-checked on download; see item 2 above for why the pin
  matters.
- `policyengine-uk==2.89.2`, as pinned in `pyproject.toml`. The simulated
  cells are version-sensitive: 2.90.0 shifts them by a few tenths of a point,
  which is small enough to look like noise and large enough to matter.
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

`paper/main.tex` is the short main manuscript (core appendices inline);
`paper/supplement.tex` is the standalone online supplement described above.
`make paper` builds both PDFs. All numerical values in both draw from
validated generated files.
The paper reports a static, partial-equilibrium, first-round fiscal-incidence
stress test conditional on imposed labour-income changes. It is not a causal
estimate of the tariffs' production, productivity, employment, macroeconomic
or total household-welfare effects. Licensed FRS inputs are not distributed.
