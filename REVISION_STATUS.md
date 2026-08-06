# Major-revision status

Updated 6 August 2026 against `REFEREE_AUDIT.md`, the journal-readiness
audit, and the August 2026 referee report (reject-and-resubmit).

## Completed for the August 2026 referee report

- **Factorial decomposition (referee major point 1).** A new
  `concentrated_wage_cut` margin imposes the displacement draw's exact
  worker-level losses (paired seeds, balanced assignment) with no
  employment-state change. At the unit 12-month stress the 10.0-point
  wage-cut-minus-displacement gap decomposes into 9.9 points from
  concentration/worker selection and 0.1 points from the employment state
  itself (OBR anchor: 11.4 vs 0.1). The channel split locates the
  concentration step in marginal-rate income-tax relief (+16.1pp),
  partially offset by UC (-1.8pp). The manuscript's causal language was
  revised accordingly: the gap is a loss-concentration effect within the
  modelled statutory schedule, and the near-zero state step explicitly
  measures the modelled annual rules (New Style JSA, monthly assessment,
  conditionality are unmodelled). Artifacts:
  `results/factorial_decomposition.json`,
  `results/submission/submission_{obr,unit}_12m_concentrated_wage_cut.json`.
- **Reframing (point 2).** New title (adjustment margins and tax-benefit
  cushioning; tariffs as the application), abstract rewritten around the
  factorial design with the calibration labelled a declared accounting
  bridge, not an estimated first stage.
- **Uncertainty presentation (point 3).** "Excludes zero" removed from the
  abstract, introduction, results and discussion; the household bootstrap is
  labelled a record-resampling sensitivity with an explicit list of what it
  does not capture; no significance-style claims remain.
- **Leave-one-record-out (point 4).** Exact LOO over all 194
  loss-contributing FRS households from the stored per-household
  contributions: the contrast stays in [10.3, 11.3]pp, always positive;
  max single-record shift 0.6pp. Artifact:
  `results/leave_one_record_out.json`.
- **Interior transition scenario (point 7).** Removed from the abstract and
  introduction headline; retained as one point on the response surface with
  its parameters labelled assumptions.
- **Number reconciliation (specific presentation problems).** All abstract
  numbers now come from the single 50-draw paired submission estimator
  (levels 43.9 / 34.0, paired-draw gap 10.0); an "estimator provenance"
  note in the results section states which estimator generates each number
  and distinguishes the 25-common-draw record-resampling contrast (10.7).
- **Restructuring (point 8).** The standalone online supplement is
  restored (`paper/supplement.tex` + `paper/sections/supplement_body.tex`):
  HMRC event-study design and estimates, LFS imputation designs and tables,
  EPD results detail, the exploratory grid/surface, reallocation, measured,
  demographic, supply-chain and constituency sections all moved out of the
  main paper.
- **UC mechanism language (point 6).** The discussion now separates
  statutory entitlement in PolicyEngine, entitlement under incomplete FRS
  information (capital gates unexercised), modelled take-up conventions,
  and actual post-displacement receipt.
- **Pension channel on the primary estimator.** The pension-gross
  sensitivity now runs on the balanced 50-draw unit 12-month design (was a
  25-seed Bernoulli comparator), so its from-values match the headline
  levels exactly: wage cut 43.9 -> 39.6, displacement 34.0 -> 29.1, gap
  10.0 -> 10.5pp. `analysis/referee_fixes.py --only pension` recomputes it
  standalone.
- Monthly modelling (point 5) remains outside the estimand (see below);
  the annualisation is presented as duration-equivalent stress with the
  parameter-based bounding appendix retained in the main paper.

## Completed in code and manuscript (July 2026 round)

- UC take-up is triggered symmetrically by newly positive post-shock
  entitlement across all margins.
- Existing LCWRA awards are capped and regression-tested.
- Displacement now uses balanced probability integration for the primary
  table, with Bernoulli assignments retained as an unbiased comparator.
  Record-support diagnostics disclose effective counts and maximum
  contributions.
- The April 2025 movement no longer identifies the central demand parameter.
  The declared scenario set is 0.4 (OBR-style low), 1.0 (central unit stress),
  2.0 (former April-based high case), and 3.0 (severe).
- Assignment SD and numerical Monte Carlo SE are stored separately.
- The measured family is labelled an observed-outturn stress scenario and its
  validation artifact reports both clipped downside and signed net exposure.
- The production-to-earnings parameter is described as wage-bill incidence;
  the legacy `passthrough` API spelling remains for compatibility.
- Three-, six- and twelve-month rows are explicitly duration-equivalent
  annual stresses. They match expected worker-month loss but do not claim to
  simulate partial-year unemployment spells.
- Input-output and local multiplier estimates are not compounded.
- Cash-income results prominently exclude consumer prices and total welfare.
- Regional results are labelled synthetic and no constituency rankings are
  reported.
- Input URLs, retrieval dates, vintages, exclusions and SHA-256 hashes are
  frozen in `uk_trade_shock_study/data/input_manifest.json`.
- Python 3.13 dependencies are locked with `uv.lock`; CI runs tests and
  manifest validation; `make reproduce` declares the full licensed-data build.
- Headline direct and measured scenarios use a common 100-assignment
  production specification.
- Headline manuscript values are generated from the JSON artifacts; the build
  fails if the ten declared production artifacts have mixed draw counts.
- Main-text benchmark checks now report the weak sector-level agreement with
  observed exports and the very small expected number of selected FRS records
  per displacement assignment.
- The 15/85 mixed margin is labelled a literature-disciplined sensitivity,
  not an empirical calibration obtained by mapping a rent-sharing elasticity
  one-for-one into an extensive/intensive loss share.
- Policy reforms not simulated in the paper are labelled candidate
  counterfactuals rather than recommendations supported by the estimates.
- PDF metadata, single-author acknowledgements, and submission declarations
  have been standardised.
- The OBR-style 0.4 anchor is co-equal with unit stress in the abstract,
  primary results table and discussion.
- Longitudinal LFS Study 9490 transitions are calibrated to FRS workers using
  BRES-adjusted industry composition. Cell, earnings-band and quantile-random-
  forest risk models are compared and propagated through the fiscal model.
- Worker-risk linkage uses the actual FRS person identifier and covers
  99.8--100% of manufacturing records. Within-industry calibration preserves
  the imposed expected wage-bill loss.
- A public HMRC product-by-destination panel provides an event-study benchmark
  with alternative controls, weights, start dates, trends and placebo dates.
  Its failed tariff-intensity falsification is reported, so it is not used to
  estimate the scenario elasticity.
- The primary contrast is pre-specified as the difference in statutory
  cushioning between earnings-equivalent wage-cut and displacement margins,
  conditional on declared sector losses.
- Balanced and Bernoulli integration agree closely at the unit 12-month
  stress, while balancing materially reduces allocation noise.
- Leave-one-division-out reruns cover all 23 exposed divisions. The primary
  contrast remains positive in every rerun, ranging from 7.6 to 11.3
  percentage points.
- The complete gate passes 100 tests, validates the frozen input manifest,
  regenerates all manuscript values and builds a visually inspected 28-page
  main paper plus a 14-page online supplement.

## Deliberately outside the current estimand

The following are not small code fixes and are not claimed as completed:

- a credible causal tariff first stage (the new descriptive destination panel
  does not pass the tariff-intensity falsification);
- linked firm/payroll estimation of labour incidence and re-employment paths;
- FRS complex-survey replicate/bootstrap uncertainty;
- monthly dynamics, New Style JSA and WCA timing;
- a household expenditure-price module;
- a coherent IO/CGE general-equilibrium closure.

The paper is consequently framed as a static first-round tax-benefit stress
test conditional on imposed labour-income scenarios. It is not presented as a
causal tariff estimate or a complete welfare analysis. Adding the items above
would constitute a new empirical project requiring data and modelling beyond
the repository's current inputs.

## Public-data acquisition completed

- A resumable, credential-free HMRC OTS downloader builds the
  product-by-destination monthly export panel for 2018--2026.
- Open Nomis BRES manufacturing employment benchmarks for 2015--2024 are
  downloaded reproducibly (disclosure-rounded aggregate estimates).
- Public ONS ABS turnover inputs were already in the trade-intensity builder.
- Longitudinal LFS Study 9490 is available under the UK Data Service End User
  Licence and is now integrated. Secure ASHE is not required for the current
  conditional estimand, but remains valuable for payroll-quality realised
  earnings-incidence validation.
