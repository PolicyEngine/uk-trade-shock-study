# uk-trade-shock-study

How does the UK tax-benefit system respond to labour-income stress scenarios
calibrated to the 2025 US tariffs on UK goods? The project imposes a
reduced-form tariff-exposure-to-wage-bill bridge on FRS 2024-25 microdata and
runs it through PolicyEngine UK. It does not model the intervening effects on
prices, quantities, production, value added, productivity or labour demand.
Adjustment scenarios include **displacement**, **wage cuts**, **inactivity**,
**reallocation**, and a factorial **mixed wage/job-loss** family. An Economic
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

## Package

- `uk_trade_shock_study/exposure.py` — tariff schedule (both scenarios),
  US-export intensity, derived per-SIC earnings shocks, FRS SIC join.
- `uk_trade_shock_study/shocks.py` — pure and mixed adjustment-margin families;
  hard-errors if the employment_status transition fails to apply.
- `uk_trade_shock_study/runner.py` — PolicyEngine UK runs: disposable income,
  relative/absolute BHC + AHC poverty, Gini, decile/region breakdowns,
  Exchequer effect, Monte Carlo support.

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

## Paper

`paper/main.tex` + `paper/sections/` (LaTeX conventions from uk-ai-study).
The paper reports a static, partial-equilibrium, first-round fiscal-incidence
stress test conditional on imposed labour-income changes. It is not a causal
estimate of the tariffs' production, productivity, employment, macroeconomic
or total household-welfare effects. Licensed FRS inputs are not distributed.
