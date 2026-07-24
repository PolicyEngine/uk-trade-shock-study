.PHONY: bootstrap test manifest check inputs results figures paper-values uncertainty-design paper reproduce

PYTHON := .venv/bin/python

bootstrap:
	uv sync --extra dev --frozen

test:
	$(PYTHON) -m pytest -q

manifest:
	$(PYTHON) analysis/validate_manifest.py

check: test manifest paper-values

inputs:
	$(PYTHON) analysis/build_trade_by_sic.py
	$(PYTHON) analysis/build_measured_shocks.py
	$(PYTHON) analysis/validate_manifest.py

results:
	$(PYTHON) analysis/run_scenarios.py --n-draws 100 --scenarios full_tariff_displacement full_tariff_wage_cut full_tariff_inactivity epd_displacement epd_wage_cut epd_inactivity measured_displacement measured_wage_cut
	$(PYTHON) analysis/scenario_testing.py
	$(PYTHON) analysis/sensitivity_grid.py
	$(PYTHON) analysis/takeup_sensitivity.py
	$(PYTHON) analysis/sensitivity_duration_takeup.py
	$(PYTHON) analysis/run_reallocation.py
	$(PYTHON) analysis/measured_cushioning.py
	$(PYTHON) analysis/mechanism_decomposition.py
	$(PYTHON) analysis/poverty_gap.py
	$(PYTHON) analysis/demographics.py
	$(PYTHON) analysis/supply_chain_scenario.py

figures:
	$(PYTHON) analysis/figures.py
	$(PYTHON) analysis/geo_impact.py
	$(PYTHON) analysis/geo_choropleth.py

uncertainty-design:
	$(PYTHON) analysis/write_uncertainty_design.py

paper-values:
	$(PYTHON) analysis/write_paper_results.py --expected-draws 100

paper: paper-values
	cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

reproduce: check inputs results figures paper
