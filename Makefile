# Who pays for trade shocks? -- reproduction targets.
# The pipeline lives in the uk_trade_shock_study package; generated macros
# land in results/ and are copied into paper/.

PYTHON := .venv/bin/python
PKG := uk_trade_shock_study
# Microdata: the certified Microcosm UK build pinned by policyengine[uk] in
# pyproject.toml, fetched on first use (needs HUGGING_FACE_TOKEN or a saved
# huggingface-cli login).

.PHONY: bootstrap e3 first-pass second-stage deflated lcfs-validation grid figures paper test reproduce

bootstrap:
	uv sync --extra dev

# Episode 3 (US tariffs): earnings-channel runs, then the constants the
# first pass reads.
e3:
	PYTHONPATH=. $(PYTHON) analysis/run_scenarios.py --n-draws 20 --scenarios full_tariff_displacement
	PYTHONPATH=. $(PYTHON) analysis/run_scenarios.py --e3
	PYTHONPATH=. $(PYTHON) analysis/ashe_validation.py
	PYTHONPATH=. $(PYTHON) analysis/sensitivity_duration_takeup.py
	PYTHONPATH=. $(PYTHON) analysis/e3_uniform_and_income_shares.py
	PYTHONPATH=. $(PYTHON) analysis/e3_industry_table.py
	PYTHONPATH=. $(PYTHON) analysis/emit_e3_constants.py
	PYTHONPATH=. $(PYTHON) analysis/emit_e3_macros.py
	PYTHONPATH=. $(PYTHON) analysis/emit_data_macros.py

# First pass: public-data incidence for all four episodes (no licensed data).
first-pass:
	cd $(PKG) && ../$(PYTHON) incidence.py
	cd $(PKG) && ../$(PYTHON) make_figures.py

# Household stage: energy, food and India episodes on Microcosm UK.
second-stage:
	cd $(PKG) && ../$(PYTHON) second_stage_energy.py
	cd $(PKG) && ../$(PYTHON) second_stage_energy.py --exclude-synthetic --out ../results/second_stage_energy_nosyn.json
	cd $(PKG) && ../$(PYTHON) make_within_decile.py
	cd $(PKG) && ../$(PYTHON) vintage_and_tca.py
	cd $(PKG) && ../$(PYTHON) mpc_welfare.py
	cd $(PKG) && ../$(PYTHON) ceta_second_stage.py
	cd $(PKG) && ../$(PYTHON) rank_robustness.py
	cd $(PKG) && ../$(PYTHON) uprating_lag_reform.py
	cd $(PKG) && ../$(PYTHON) uprating_lag_reform.py --deflate

# Referee robustness: the energy stage with inputs deflated to the rulebook
# year by PolicyEngine UK's own projection indices, under 2023-24 and
# 2022-23 rules. Writes separate JSONs; the central run is untouched.
deflated:
	cd $(PKG) && ../$(PYTHON) second_stage_energy.py --deflate --out ../results/second_stage_energy_deflated.json
	cd $(PKG) && ../$(PYTHON) second_stage_energy.py --sim-year 2022 --out ../results/second_stage_energy_rules2022.json
	cd $(PKG) && ../$(PYTHON) second_stage_energy.py --sim-year 2022 --deflate --out ../results/second_stage_energy_rules2022_deflated.json
	cd $(PKG) && ../$(PYTHON) deflated_inputs.py

# LCFS validation of the within-decile spread. Needs the licensed LCFS
# household files (UK Data Service) at local paths outside the repository:
#   make lcfs-validation LCFS_2024=/path/to/dvhh_uk_anon_v3_2024.tab \
#                        LCFS_DIR=/path/to/folder/with/earlier/dvhh/dta
# Writes aggregates only.  With no path set, the committed LCFS aggregates
# are kept and realigned to the household stage's current burden basis, and
# the Microcosm column, macros and table_lcfs.tex are refreshed.
LCFS_2024 ?=
LCFS_DIR ?=
lcfs-validation:
	cd $(PKG) && ../$(PYTHON) lcfs_validation.py \
		$(if $(LCFS_2024),--lcfs-2024 $(LCFS_2024)) $(if $(LCFS_DIR),--lcfs-dir $(LCFS_DIR))

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

reproduce: e3 first-pass second-stage deflated lcfs-validation grid paper
