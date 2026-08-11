.PHONY: bootstrap test manifest check public-hmrc public-bres trade-event-study lfs-imputation lfs-qrf lfs-benchmarks inputs results submission-results figures paper-values uncertainty-design paper reproduce takeup-entitled bootstrap-summary

PYTHON := .venv/bin/python

bootstrap:
	uv sync --extra dev --frozen

test:
	$(PYTHON) -m pytest -q

manifest:
	$(PYTHON) analysis/validate_manifest.py

check: test manifest paper-values

# Re-runs BOTH take-up scopes at the current calibration.  The
# new-entitlement grid is inert (its re-draw set is empty); the all-entitled
# grid is the one that exercises the claiming margin, and the manuscript
# currently quotes a superseded-vintage bound for it.  Needs licensed FRS data.
takeup-entitled:
	$(PYTHON) analysis/referee_fixes.py --only takeup
	$(PYTHON) analysis/write_referee_macros.py

# Rebuilds results/bootstrap_uncertainty.json (including the ratio-of-pooled-sums
# estimator and the support diagnostics) from results/bootstrap_contributions.npz.
# Needs no FRS file and no PolicyEngine when that cache is present.
bootstrap-summary:
	$(PYTHON) analysis/bootstrap_uncertainty.py

public-hmrc:
	$(PYTHON) analysis/download_hmrc_panel.py

public-bres:
	$(PYTHON) analysis/download_bres_benchmarks.py

trade-event-study:
	$(PYTHON) analysis/hmrc_destination_event_study.py

lfs-imputation:
	@test -n "$(LFS_TAB)" || (echo "Set LFS_TAB=/path/to/five-quarter-LFS.tab" && exit 1)
	$(PYTHON) analysis/impute_lfs_to_frs.py --lfs-tab "$(LFS_TAB)"

lfs-qrf:
	@test -n "$(LFS_TAB)" || (echo "Set LFS_TAB=/path/to/five-quarter-LFS.tab" && exit 1)
	$(PYTHON) analysis/benchmark_lfs_qrf.py --lfs-tab "$(LFS_TAB)"

lfs-benchmarks: lfs-imputation lfs-qrf
	$(PYTHON) analysis/write_lfs_benchmark_results.py

inputs:
	$(PYTHON) analysis/build_trade_by_sic.py
	$(PYTHON) analysis/build_measured_shocks.py
	$(PYTHON) analysis/validate_manifest.py

results:
	$(PYTHON) analysis/run_scenarios.py --n-draws 100 --scenarios full_tariff_displacement full_tariff_wage_cut full_tariff_inactivity epd_displacement epd_wage_cut epd_inactivity measured_displacement measured_wage_cut full_tariff_rentsharing epd_rentsharing full_tariff_transition_central epd_transition_central full_tariff_obr_low_displacement full_tariff_obr_low_wage_cut
	$(PYTHON) analysis/scenario_testing.py
	$(PYTHON) analysis/sensitivity_grid.py
	$(PYTHON) analysis/takeup_sensitivity.py
	$(PYTHON) analysis/sensitivity_duration_takeup.py
	$(PYTHON) analysis/run_reallocation.py
	$(PYTHON) analysis/measured_cushioning.py
	$(PYTHON) analysis/mechanism_decomposition.py
	$(PYTHON) analysis/poverty_gap.py
	$(PYTHON) analysis/demographics.py
	$(PYTHON) analysis/run_lfs_selection_sensitivity.py
	$(PYTHON) analysis/supply_chain_scenario.py

submission-results:
	$(PYTHON) analysis/run_submission_scenarios.py --n-draws 50
	$(PYTHON) analysis/run_leave_one_sector_out.py --n-draws 20
	$(PYTHON) analysis/bootstrap_uncertainty.py
	$(PYTHON) analysis/factorial_decomposition.py --n-draws 50
	$(PYTHON) analysis/leave_one_record_out.py
	$(PYTHON) analysis/write_submission_results.py --expected-draws 50
	$(PYTHON) analysis/referee_fixes.py

figures:
	$(PYTHON) analysis/figures.py
	$(PYTHON) analysis/geo_impact.py
	$(PYTHON) analysis/geo_choropleth.py

uncertainty-design:
	$(PYTHON) analysis/write_uncertainty_design.py

paper-values:
	$(PYTHON) analysis/write_paper_results.py --expected-draws 100
	$(PYTHON) analysis/write_lfs_benchmark_results.py
	$(PYTHON) analysis/write_trade_benchmark_results.py
	$(PYTHON) analysis/write_lfs_selection_results.py
	$(PYTHON) analysis/write_submission_results.py --expected-draws 50
	$(PYTHON) analysis/write_factorial_results.py --expected-draws 50
	$(PYTHON) analysis/write_validation_macros.py
	# Pure statutory arithmetic; safe to recompute on every build.
	$(PYTHON) analysis/referee_fixes.py --only jsa
	$(PYTHON) analysis/write_referee_macros.py

paper: paper-values
	cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
	cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error supplement.tex

reproduce: check inputs results submission-results figures paper
