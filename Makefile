# Who Bears the Trade Shocks? -- reproduction targets.
# The pipeline lives in the uk_trade_shock_study package; generated macros
# land in results/ and are copied into paper/.

PYTHON := .venv/bin/python
PKG := uk_trade_shock_study
DATASET ?= $(error set DATASET=<path to enhanced_frs .h5>)

.PHONY: bootstrap first-pass second-stage grid figures paper test reproduce

bootstrap:
	uv sync

# First pass: public-data incidence for all five episodes (no licensed data).
first-pass:
	cd $(PKG) && ../$(PYTHON) incidence.py
	cd $(PKG) && ../$(PYTHON) make_figures.py

# Second stage: the energy and TCA episodes on the enhanced FRS.
second-stage:
	cd $(PKG) && ../$(PYTHON) second_stage_energy.py --dataset $(DATASET)
	cd $(PKG) && ../$(PYTHON) vintage_and_tca.py --dataset $(DATASET)
	cd $(PKG) && ../$(PYTHON) mpc_welfare.py --dataset $(DATASET)

# Sensitivity grid, bootstrap and referee-round figures.
grid:
	cd $(PKG) && ../$(PYTHON) grid_energy.py --dataset $(DATASET)
	cd $(PKG) && ../$(PYTHON) fig_grid.py
	cd $(PKG) && ../$(PYTHON) fig_extra.py --dataset $(DATASET)

paper:
	cd paper && (tectonic -X compile main.tex || \
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex)

# Tests: pinned-input hashes, macro definitions, paper/results sync, and a
# first-pass reproduction check (public data only; no licensed microdata).
test:
	uv run pytest -q

reproduce: first-pass second-stage grid paper
