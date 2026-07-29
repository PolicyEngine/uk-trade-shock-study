# Journal-readiness and referee report

**Paper:** *The Fiscal Incidence of Trade-Related Labour-Income Shocks in the UK*

**Review date:** 29 July 2026

**Recommendation:** Submit only as a conditional microsimulation and
automatic-stabiliser paper; do not submit as causal tariff incidence

**Current confidence rating:** Reproducible and internally coherent, but not
ready for a causal tariff-incidence claim

## Editorial assessment

The project has a worthwhile question, unusually transparent code, a locked
environment, generated manuscript values, and a clear conditional estimand.
The tax-benefit accounting contribution is potentially publishable. The
revised paper is a credible conditional microsimulation submission candidate,
but it is not competitive as a causal tariff-incidence paper because its
tariff-to-wage-bill loss is imposed, the displacement calculation has thin FRS
support, and survey/first-stage uncertainty is incomplete.

The strongest viable paper from the present inputs is a methods and
automatic-stabiliser paper: conditional on a common sector wage-bill loss,
how do UK taxes and transfers differ when adjustment occurs through intensive
and extensive labour-market margins? A paper claiming the realised incidence
of the 2025 tariffs requires a new empirical first stage and additional data.

## Major findings, in priority order

1. **[Critical] The economic first stage is not identified.** Equation (1)
   imposes `tariff × US-export share × demand parameter × wage-bill incidence`.
   The reference demand parameter is 1 and wage-bill incidence is 1. Neither
   is estimated. The reference demand response is 2.5 times the OBR's 0.4
   elasticity benchmark and then assigns the full exposure to employee wages.
   The £886 million result is therefore a deliberately strong scenario input,
   not a central estimate.

2. **[Critical] Displacement incidence has thin survey support.** At the unit
   12-month stress the balanced design averages 9.8 affected records, only
   6.2 effective loss-contributing records and a 28.7% maximum-record loss
   share. At the OBR three-month scale these deteriorate to 1.3, 1.2 and
   91.9%, respectively. Balanced integration removes avoidable allocation
   noise but cannot add survey information. Small-scale, subgroup, poverty
   and regional estimates remain sensitivities rather than precise estimates.

3. **[High] The aggregate outturn match does not validate the model.** The
   predicted mapped-sector export fall is 12.57%, close to the observed 12.71%
   May 2025-February 2026 fall. But the top-ten sector rank correlation is
   0.00 and the all-division downside-shock correlation is only 0.58.
   Aerospace, electrical equipment and basic metals contradict the simple
   tariff ranking. The aggregate equality may be coincidental.

4. **[High] A rent-sharing elasticity cannot identify the wage/job-loss
   decomposition.** The earlier draft mapped a 0.15 incumbent-wage elasticity
   directly into a 15% survivor-wage-loss share and an 85% displacement share.
   These are different estimands. The repository now labels this a stylised,
   literature-disciplined sensitivity rather than an empirical calibration.

5. **[High] Statistical uncertainty remains incomplete.** Assignment SD and
   numerical Monte Carlo SE are correctly separated, but the analysis does
   not estimate FRS sampling variance, first-stage uncertainty, concordance
   uncertainty, model uncertainty or parameter probabilities. The reported
   SDs are not confidence intervals for an economic effect.

6. **[High] The polar margins are useful bounds, not observed adjustment
   processes.** Uniform within-SIC displacement ignores exporter wage premia
   and worker selection. Pure wage cuts impose a unit survivor-wage response.
   Reallocation destinations and penalties are cross-sectional. Inactivity
   immediately grants LCWRA to all selected workers aged 50 and over and omits
   WCA timing and New Style JSA.

7. **[High] The analysis is nominal cash-income incidence only.** Consumer
   prices, profits, pensions as deferred consumption, indirect taxes, imported
   inputs, exchange rates, retaliation, market switching and general
   equilibrium are outside the estimand. Poverty effects cannot be described
   as total household welfare effects.

8. **[Medium] The paper is longer and broader than its evidence base.**
   Constituency, demographic, reallocation, measured-outturn and supply-chain
   appendices are exploratory and can distract from the defensible core.
   A submission should shorten or move these to an online supplement.

## Calculation and plausibility checks

| Check | Result | Assessment |
|---|---:|---|
| Expected full-schedule wage loss | £885.75m | Reproduced from FRS earnings, weights and SIC shocks |
| Unit 12-month balanced displacement loss | £885m ± £6m | Matches the £885.75m target |
| Wage-cut cushioning identity | 43.93% | Internally reproduced |
| Unit 12-month displacement cushioning | 34.0% ± 4.7pp assignment SD | Conditional on sparse record support |
| Wage-cut Exchequer cost | £484.2m | Internally reproduced |
| Unit 12-month displacement Exchequer cost | £378m ± £37m | Bernoulli comparator is £377m |
| Primary cushioning contrast | 10.0pp | Leave-sector range 7.6--11.3pp; no reversal |
| Mapped customs exports | £52.5bn after exclusions | Plausible against ONS £59.3bn BOP goods exports, subject to coverage concepts |
| Predicted vs observed aggregate export fall | 12.57% vs 12.71% | Superficial match only |
| Predicted vs observed sector pattern | rank 0.00 top ten; correlation 0.58 overall | Does not validate sector mechanism |
| OBR demand benchmark | elasticity 0.4 | Reference value 1.0 is a severe stress relative to this anchor |
| Public destination-panel estimate | -31.9% weighted differential after May 2025 | Descriptive only: tariff-intensity falsification fails |
| LFS job-exit target | 8.96% | Calibrated from longitudinal Study 9490; only 177 manufacturing donors |
| LFS risk-model agreement | 0.06 cell/QRF correlation for exits | Material model uncertainty, now propagated |
| Risk-shaped displacement gross loss | £847m--£887m across selection models | Smaller model spread than £397m--£467m assignment SD |
| Tests and build | 100 tests; manifest valid; 28-page paper + 14-page supplement | Reproducibility contract is strong |

## Required research expansion for a stronger submission

### Submission blockers

1. Estimate or externally calibrate a defensible production-to-labour first
   stage. Minimum acceptable version: destination-product tariff exposure,
   pre-period trends, quantities and unit values, non-US destinations as
   controls, and explicit anticipation tests. Preferred version: linked
   employer-worker PAYE/ASHE or longitudinal worker evidence for earnings,
   hours, employment and re-employment.

2. **Partly completed:** primary results use balanced probability integration,
   retain Bernoulli comparators, disclose affected/effective record counts and
   maximum contributions, and include all-division leave-one-out tests.
   Richer survey support and calibration beyond SIC and broad age/earnings
   margins still require additional data.

3. Add survey uncertainty using the FRS design variables or an approved
   bootstrap/replicate-weight procedure. Report first-stage uncertainty,
   survey uncertainty, assignment integration error and parameter scenarios
   separately.

4. **Completed:** the OBR-style 0.4 case is co-equal with unit stress in the
   abstract and main table, and unit stress is not called a forecast or best
   estimate.

5. **Completed:** the primary estimand is the difference in statutory
   cushioning between earnings-equivalent intensive and extensive
   adjustments, conditional on declared sector losses.

### Valuable extensions

- Add monthly timing for unemployment, UC, WCA/LCWRA and New Style JSA.
- Build a household expenditure-price module and report real-income incidence
  separately from nominal cash income.
- Validate FRS sector employment and earnings against BRES, ASHE and ABS,
  including exporter wage-premium sensitivities.
- **Completed:** all 23 exposed divisions are tested leave-one-out and none
  reverses the primary contrast. A direct leave-one-record-out audit remains
  valuable.
- Report threshold-crossing counts and poverty-gap changes, not only tiny
  national headcount changes.
- Archive a blinded replication package and a computational appendix with
  software versions, run time, hashes and restricted-data instructions.

## Recommended submission strategy

The best current fit above the requested metric threshold is **The World
Economy** if the empirical trade first stage is added; its publisher reports a
2.4 Journal Impact Factor and a trade-policy remit. Without that first stage,
the paper is better reframed around social protection and automatic
stabilisers; **Journal of Social Policy** is a possible fit, with Cambridge
reporting a 5-year Impact Factor of 2.8, but it would require a stronger social
policy framing and credible uncertainty. Journal metrics change annually and
must be rechecked immediately before submission.

The present version is suitable for journal formatting and submission only
under the conditional microsimulation framing. A causal trade-journal claim
still has scientific-design and data blockers. Passing code does not convert
scenario inputs into identified effects.

## Changes made in this review

- Added the external magnitude and observed-outturn falsification checks to
  the main results.
- Disclosed the expected 9.7 selected FRS records per displacement assignment.
- Corrected the 15/85 mixed-margin interpretation throughout code and paper.
- Added an ONS source for the £59.3bn external trade benchmark.
- Added a regression test preventing return of the overstated empirical
  calibration language.
- Rebuilt the manuscript PDF and reran the repository checks.
- Added the public HMRC product-destination event study and its falsification
  suite without using the failed design to calibrate the headline shock.
- Added calibrated longitudinal LFS cell, earnings-band and QRF benchmarks and
  propagated alternative worker-risk selection through PolicyEngine.
- Fixed the FRS person-ID linkage after an audit caught an index-based fallback.
- Added co-equal OBR-style production scenarios and separated undefined
  zero-loss cushioning draws from valid summaries.
- Added a 14-scenario primary design spanning two anchors, three
  duration-equivalent stresses and two adjustment margins.
- Added balanced integration, Bernoulli comparators, record-support
  diagnostics and all-division leave-one-out tests.
- Pre-specified the margin cushioning contrast, shortened the main paper and
  moved exploratory material to an online supplement.
- Regenerated every table and value, passed 100 tests and visually inspected
  the rebuilt main paper and supplement.
