# Major-revision status

Updated 23 July 2026 against `REFEREE_AUDIT.md`.

## Completed in code and manuscript

- UC take-up is triggered symmetrically by newly positive post-shock
  entitlement across all margins.
- Existing LCWRA awards are capped and regression-tested.
- Displacement uses equal-probability Bernoulli sampling, making weighted
  headcount and wage-bill losses unbiased by sector; common seeds support
  paired comparisons.
- The April 2025 movement no longer identifies the central demand parameter.
  The declared scenario set is 0.4 (OBR-style low), 1.0 (central unit stress),
  2.0 (former April-based high case), and 3.0 (severe).
- Assignment SD and numerical Monte Carlo SE are stored separately.
- The measured family is labelled an observed-outturn stress scenario and its
  validation artifact reports both clipped downside and signed net exposure.
- The production-to-earnings parameter is described as wage-bill incidence;
  the legacy `passthrough` API spelling remains for compatibility.
- The internally inconsistent six-month annual-model hybrid was removed.
- Input-output and local multiplier estimates are not compounded.
- Cash-income results prominently exclude consumer prices and total welfare.
- Regional results are labelled synthetic and no constituency rankings are
  reported.
- Input URLs, retrieval dates, vintages, exclusions and SHA-256 hashes are
  frozen in `uk_trade_shock_study/data/input_manifest.json`.
- Python 3.13 dependencies are locked with `uv.lock`; CI runs tests and
  manifest validation; `make reproduce` declares the full licensed-data build.

## Deliberately outside the current estimand

The following are not small code fixes and are not claimed as completed:

- causal product-by-destination tariff estimation;
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
