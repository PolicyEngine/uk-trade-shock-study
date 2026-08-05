# Trade-shock paper redesign

## Target estimand

The paper will estimate conditional statutory fiscal incidence, not the causal
effect of the 2025 tariffs:

> Given a trade-exposure-calibrated aggregate labour-income loss, how do UK
> household disposable income, poverty, inequality and the Exchequer respond
> under alternative, UK-evidence-informed adjustment paths?

The tariff/exposure build supplies scenario anchors. PolicyEngine UK supplies
the tax-benefit mapping. No parameter is described as a causal tariff-to-
employment estimate unless supported by external evidence.

## Central scenario architecture

The primary design will separate three axes:

1. aggregate shock size (OBR-style 0.4 central anchor and unit/high stress);
2. adjustment path (temporary non-employment, re-employment with an earnings
   penalty, and modest incumbent wage/hours adjustment); and
3. policy/take-up assumptions (fixed policy in the main result, with take-up
   and reform counterfactuals in sensitivity analyses).

Pure wage-cut and pure displacement scenarios remain transparent bounds, not
the preferred description of the economy. The existing mixed/rentsharing and
reallocation implementations are candidate central paths, subject to UK
evidence and pension/timing checks.

## Required interpretation safeguards

- Three- and six-month annualised rows are not actual partial-year spells;
  they must be labelled as duration-equivalent stresses unless monthly spells
  are implemented.
- Pension contributions must be held fixed in a headline sensitivity, because
  reduced retirement saving is not current tax-benefit insurance.
- UC take-up, capital eligibility, inactivity/WCA passage and re-employment
  are scenario parameters, not identified estimates.
- FRS/LFS selection models describe plausible incidence shapes but do not
  identify tariff-specific worker selection.
- Assignment dispersion, survey sampling variation, and uncertainty in the
  trade-to-earnings bridge must be reported separately.

## Journal-facing structure

1. Motivation and contribution: incidence of trade-related labour-income loss.
2. Trade calibration and external evidence.
3. Adjustment-path scenarios and PolicyEngine implementation.
4. Main factorial results: fiscal, disposable-income and distributional
   incidence across paths and shock sizes.
5. Policy counterfactuals.
6. Robustness: worker selection, duration/re-employment, take-up, pensions,
   and Monte Carlo support.
7. Scope and limitations.

The failed HMRC tariff-intensity test is evidence against a causal first
stage and should be used to bound the claims, not hidden as validation.
