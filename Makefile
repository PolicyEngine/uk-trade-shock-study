.PHONY: bootstrap test manifest check inputs results figures uncertainty-design paper reproduce

PYTHON := .venv/bin/python

bootstrap:
	uv sync --extra dev --frozen

test:
	$(PYTHON) -m pytest -q

manifest:
	$(PYTHON) analysis/validate_manifest.py

check: test manifest

inputs:
	$(PYTHON) analysis/build_trade_by_sic.py
	$(PYTHON) analysis/build_measured_shocks.py
	$(PYTHON) analysis/validate_manifest.py

results:
	$(PYTHON) analysis/run_scenarios.py --n-draws 500
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

paper:
	cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

reproduce: check inputs results figures paper
