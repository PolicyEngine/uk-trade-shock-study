# Who pays for trade shocks? -- reproduction targets.
# The pipeline lives in the uk_trade_shock_study package; generated macros
# land in results/ and are copied into paper/.

PYTHON := .venv/bin/python
PKG := uk_trade_shock_study
# Microdata: the certified Microcosm UK build pinned by policyengine[uk] in
# pyproject.toml, fetched on first use (needs HUGGING_FACE_TOKEN or a saved
# huggingface-cli login).

.PHONY: bootstrap e3 first-pass second-stage grid figures paper test reproduce

bootstrap:
	uv sync

# Episode 3 (US tariffs): earnings-channel runs, then the constants the
# first pass reads.
e3:
	PYTHONPATH=. $(PYTHON) analysis/run_scenarios.py --n-draws 20 --scenarios full_tariff_displacement
	PYTHONPATH=. $(PYTHON) analysis/run_scenarios.py --n-draws 3 --scenarios full_tariff_wage_cut
	PYTHONPATH=. $(PYTHON) analysis/emit_e3_constants.py

# First pass: public-data incidence for all five episodes (no licensed data).
first-pass:
	cd $(PKG) && ../$(PYTHON) incidence.py
	cd $(PKG) && ../$(PYTHON) make_figures.py

# Second stage: the energy and TCA episodes on the enhanced FRS.
second-stage:
	cd $(PKG) && ../$(PYTHON) second_stage_energy.py
	cd $(PKG) && ../$(PYTHON) vintage_and_tca.py
	cd $(PKG) && ../$(PYTHON) mpc_welfare.py
	cd $(PKG) && ../$(PYTHON) ceta_second_stage.py
	cd $(PKG) && ../$(PYTHON) rank_robustness.py

# Sensitivity grid, bootstrap and referee-round figures.
grid:
	cd $(PKG) && ../$(PYTHON) grid_energy.py
	cd $(PKG) && ../$(PYTHON) fig_grid.py
	cd $(PKG) && ../$(PYTHON) fig_extra.py

paper:
	cd paper && (tectonic -X compile main.tex || \
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex)

# Tests: pinned-input hashes, macro definitions, paper/results sync, and a
# first-pass reproduction check (public data only; no licensed microdata).
test:
	uv run pytest -q

reproduce: e3 first-pass second-stage grid paper
