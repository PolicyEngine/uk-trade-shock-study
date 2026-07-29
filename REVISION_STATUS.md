# Major-revision status

Updated 29 July 2026 against `REFEREE_AUDIT.md` and the journal-readiness
audit.

## Completed in code and manuscript

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
