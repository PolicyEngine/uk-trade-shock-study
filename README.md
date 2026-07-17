# uk-trade-shock-study

Who bears the 2025 US tariffs on UK goods after the tax-benefit system
responds? A PolicyEngine research project passing the tariff shock — entered
through **primitives** (tariff schedule × ONS/HMRC trade-by-SIC → sector-level
earnings shocks) — through PolicyEngine UK on FRS 2024-25 microdata, under
three adjustment-margin families: **displacement** (job loss), **wage cuts
with sectoral reallocation**, and **exit into inactivity** (older workers,
the Beatty–Fothergill UK margin). An Economic Prosperity Deal counterfactual
(full tariffs vs deal-mitigated: autos 25%→10% in-quota, conditional steel
relief, pharma exempt) prices the deal for households and the Exchequer.

Companion to [uk-ai-study](../uk-ai-study), whose conventions this repo
mirrors. Literature and scenario design: `tariff_paper_lit_review.md`.

## Pipeline

1. `analysis/build_trade_by_sic.py` — **stub**: build real US-export
   intensity by SIC division (the packaged
   `uk_trade_shock_study/data/us_export_intensity_by_sic.csv` is a
   placeholder; do not publish results from it).
2. `analysis/download_data.py` — FRS microdata from PolicyEngine's Hugging
   Face repo (needs `HUGGING_FACE_TOKEN` with access to
   `policyengine/policyengine-uk-data`); lands in `data/` (gitignored).
3. `analysis/run_scenarios.py` — runs the {full_tariff, epd} ×
   {displacement, wage_cut, inactivity} presets with Monte Carlo draws
   (mean ± SD) and writes `results/*.json`.
4. `analysis/figures.py` — paper figures (PolicyEngine house style,
   `analysis/figstyle.py`).

## Package

- `uk_trade_shock_study/exposure.py` — tariff schedule (both scenarios),
  US-export intensity, derived per-SIC earnings shocks, FRS SIC join.
- `uk_trade_shock_study/shocks.py` — the three adjustment-margin families;
  hard-errors if the employment_status transition fails to apply.
- `uk_trade_shock_study/runner.py` — PolicyEngine UK runs: disposable income,
  relative/absolute BHC + AHC poverty, Gini, decile/region breakdowns,
  Exchequer effect, Monte Carlo support.

## Setup

```sh
python -m venv .venv && . .venv/bin/activate
pip install -e . pytest
pytest            # unit tests run on synthetic tables, no FRS data needed
```

## Data requirements

- FRS 2024-25 h5 + adult.tab (licensed; via download_data.py) — gitignored.
- ONS UK trade country-by-commodity (US) and ONS output by SIC division for
  the real intensity table (TODO).

## Paper

`paper/main.tex` + `paper/sections/` (LaTeX conventions from uk-ai-study).
Literature section is drafted; results are structured placeholders pending
real trade data.
