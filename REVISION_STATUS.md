# Major-revision status

Updated 17 August 2026. Records the disposition of the original journal
readiness audit, the August 2026 referee report (reject-and-resubmit), three
further rounds of independent review, and a full economics referee report.
The source memos for the earlier rounds have been removed from the
replication package; their dispositions are recorded below.

## Fifth-round review (17 August 2026, post-restoration)

Three independent reviews of the restored manuscript: a referee-style
economics review, a line-by-line numbers audit, and a replication-package
audit.

### Fixed in this round

- Methodology overstated balanced/Bernoulli agreement ("half a percentage
  point"; the macros give 36.6 vs 35.9 = 0.7pp). Now "within one percentage
  point", with both values quoted from macros.
- The shock-mechanics appendix described only Bernoulli selection as the
  mechanism; it now distinguishes the balanced primary estimator from the
  Bernoulli comparator.
- The abstract's $\pm$ is now labelled as assignment dispersion, the
  bootstrap point (7.6) is quoted beside its interval so the 7.3 headline and
  the interval's centre reconcile, and the same reconciliation (draw-subset
  drift, 25 common draws vs the 50-draw paired estimator) is stated in the
  results bootstrap paragraph.
- The zero-hours diagnostic quoted standard errors after $\pm$ without
  noting the exception to the assignment-SD convention, and called the hours
  channel (0.08, SE 0.03 — about 2.8 SEs from zero) "indistinguishable from
  zero". The sentence now labels the SEs and calls the hours channel
  statistically resolved but economically negligible; the convention
  exception is noted in the Monte Carlo appendix.
- "Roughly four percentage points" for the pension-gross drop is now "four
  to five" in the intro and discussion (artifact: wage −4.36, displacement
  −4.88).
- "The full comparator draw count" for the take-up grid was ambiguous (the
  grid uses 25 assignments; the Bernoulli comparator scenario has 50); the
  methodology and appendix now quote `\TakeupSeeds` directly.
- Known residual, not fixed: `\TakeupConventionSpread` (3.5) is the
  difference of the rounded endpoints (36.5 − 33.0); the unrounded artifact
  difference is 3.45, which rounds to 3.4. The writer should emit the
  unrounded difference; downstream claims ("about half the headline gap")
  hold either way.
- Package hygiene: the `.gitignore` rule that contradicted the tracked
  `paper/main.pdf` is replaced by an explicit tracked-PDF policy (rebuild in
  the same commit as any source change; CI does not compile the paper);
  a third party's local filesystem path in
  `results/lfs_imputed_frs_transition_parameters.diagnostics.json` is
  scrubbed to a basename; README's account of where the EPD/LFS material
  lives and what `make check` runs is corrected; a stale "supplement"
  comment in `write_trade_benchmark_results.py` now says "appendix".

### Audit claims checked and rejected

- The numbers audit alleged the take-up paragraph's "about half the headline
  gap" was a stale-vintage survivor, quoting a 7.0pp spread; the pinned
  macros give a 3.5pp spread (33.0 to 36.5) against the 7.3pp gap, so the
  sentence is correct and unchanged.
- The package audit alleged a stale `write_supplement_macros.py` /
  `generated_supplement.tex` remnant; neither file exists in the tree.

### Open items from the referee-style review (author decisions)

1. **New Style JSA is bounded but not simulated** (worth up to 5.0pp against
   the 7.3pp gap, loading entirely on the displacement side). Simulating it
   is mechanical statutory work and the reviewer's top demand.
2. **Redundancy pay is entirely absent** (direct tax-free cushioning up;
   UC capital-gating down). At minimum a JSA-style bound is needed.
3. **The LFS worker-selection sensitivity runs on the pre-award-cache-fix
   vintage** while shifting the gap by up to ~45 per cent of its size;
   re-run on the corrected pipeline.
4. **The interior transition path never exercises the extensive margin**
   (workers remain employed for PolicyEngine purposes); the abstract and
   intro quote its 41.8 per cent without that caveat.
5. **Framing**: the reviewer recommends leading with the adjustment-margin
   mechanism (including the Dolls et al. UK-ordering reversal under UC)
   rather than the trade calibration that fails its own falsification test.
   This conflicts with the deliberate 14 August restoration of the trade
   framing and is a decision for the author.
6. **Estimator**: with rules-based incidence and ~1,018 exposed records,
   exact probability-weighted enumeration per worker would remove assignment
   dispersion entirely; adopt or justify the Monte Carlo design.
7. Editorial: caveat-density (state each disclaimer once), policy-section
   content, literature trim, affected-household poverty in Table 1, the
   partial-WCA sensitivity for the inactivity family, ORCID at submission.
   Journal fit suggested: Fiscal Studies first, Review of Income and Wealth,
   International Journal of Microsimulation.

## Fourth-round disposition (17 August 2026)

The 14 August commits restored the 5 August manuscript (trade framing, title,
section order) on top of the corrected artifacts, then regenerated every
pinned macro. Earlier sections of this file therefore quote numbers and
describe manuscript decisions from superseded vintages; the current state is:

- **Headline numbers.** The pinned submission macros now give a
  wage-minus-displacement gap of **7.3pp** (wage cut 43.9, displacement
  36.6 ± 4.6), bootstrap contrast 7.6pp with 95 per cent interval
  5.8–10.7, factorial split 7.2 concentration + 0.1 state, pension-gross
  gap 7.3 → 7.8. References below to the 10.0pp gap and 34.0 displacement
  cushioning describe the earlier calibration vintage, not the manuscript.
- **Title and framing.** The referee-round reframing around adjustment
  margins (point 2 below) was deliberately reversed: the restored manuscript
  leads with the trade framing ("Labour-Market Adjustment and Fiscal
  Incidence of Trade Exposure in the UK") for fresh submission. A
  resubmission to the venue that requested the reframing would need to
  defend this choice in the response letter.
- **Supplement.** The standalone online supplement (point 8 below) was
  folded back into the main paper's appendix; `paper/supplement.tex` no
  longer exists. The Makefile and README now reflect this.
- **EPD counterfactual.** Withdrawn as a quantitative result in the results
  and policy sections themselves: each paired difference is comparable to or
  smaller than its assignment SD, and the underlying sector calibration
  fails its rank-order check. Retained as a qualitative illustration of the
  counterfactual machinery.
- **Uncertainty language.** The restored results section had reintroduced
  "the interval excludes zero"; the phrasing is now descriptive (the
  interval lies entirely above zero) with the record-resampling caveat
  retained, consistent with the point-3 disposition below.
- **Still open.** The drift hazards listed under the third round remain:
  the customs totals (£55,600m / £52,500m) and the FRS record counts
  (34,966 / 12,800 / largest person weight above 19 thousand) quoted in the
  methodology and appendix have no backing artifact in the repository and
  regenerate only from licensed inputs; the hardcoded LFS-selection pound
  figures in the results section are labelled vintage-consistent but typed.

## Third-round disposition (11 August 2026)

A full trace of every quantitative claim back to its artifact, plus an adversarial
review of the second-round code, found a new failure mode: **prose written against
the code rather than against the artifact**. The estimator was rewritten; the stored
event-study panel could not be regenerated without the HMRC data; and the manuscript
then described specifications that had never been run.

### Corrected

- **Claims of unrun analyses.** The methodology, results and supplement now state that
  the reported estimates omit April 2025 alone (via the generated
  `\HMRCAnticipationSpec`), and that the anticipation window, continuous-trade and
  Poisson specifications are implemented but absent from this run. The
  minimum-detectable-effect and high-minus-other test claims are labelled as hand
  arithmetic on stored standard errors.
- **The figure caption contradicted its own figure.** The stored PNG/CSV are normalised
  on January--March 2025 -- the front-running window -- while the caption claimed a 2024
  base. The caption is now generated from the plotted series itself
  (`\HMRCFigureBase`, `\HMRCFigureBaseIsAnticipation`), so it cannot drift again.
- **Directional errors.** The capital test and higher claiming push the gap in
  *opposite* directions; `results.tex` said they aligned. `policy.tex` claimed all three
  unmodelled gates reduce protection, when non-claiming and the absent New Style JSA
  mean the model *understates* it. Both corrected.
- **"Upper bound" over-claimed.** The headline cell is the second-smallest of six; the
  pension-gross concept widens the gap (10.0 to 10.5). The abstract now names what the
  bound is with respect to.
- **The channel split does not reconcile to the headline.** New macros report the
  five-seed sums and the shortfall (displacement 30.3 against 34.0; the deterministic
  wage cut reconciles exactly, isolating the cause as seed-subset drift). The Dolls et
  al. comparison now states its seed basis and stops absorbing the pension residual
  silently.
- **Undisclosed designs.** The leave-one-division-out audit's draw count is now emitted
  and quoted; the LFS selection family's own Bernoulli baseline gap is emitted and used
  instead of the balanced primary; the take-up grid's 33.0 is reconciled against the
  34.0 headline.
- **Vintage caveats** added everywhere a superseded epsilon = 2 number appears, including
  its seed-0 provenance.
- **pi = 1** self-contradiction between methodology and results resolved; the unsourced
  theta range withdrawn.
- **Response-letter narration** removed from the manuscript, except the take-up
  subsection where the superseded convention motivates the current one.

### Code defects fixed

The PPML solver declared convergence after a failed line search and adopted the
rejected step; rank-deficient and separated fits still emitted point estimates;
the continuous-trade restriction selected on post-treatment outcomes; the bootstrap's
partial-cache resume path was unreachable; several diagnostics defaulted to their own
alarm values. See the round-two report for the full list.

## Second-round referee report (11 August 2026)

Ten points were raised. Eight are closed in code and manuscript; two are
closed in code but need a licensed-data run to close in the manuscript.

### Closed

- **M1 take-up grid was vacuous.** The new-entitlement re-draw set is empty,
  so all four claiming conventions returned bit-identical results and the
  paper reported an inert experiment as robustness. `shocks.py` gains a
  documented `uc_takeup_scope` option (`new_entitlement`, preserving every
  stored result, and `all_entitled`, which exercises the margin that binds);
  `redraw_uc_takeup` now returns a re-draw-set diagnostic;
  `write_referee_macros.py` warns loudly when the grid is inert. The
  manuscript now states the grid is empty and reports the entitled-scope
  bound (`\TakeupEntitledSpread` = 7.0pp) as the claiming-margin sensitivity.
- **M2 LFS selection compared against a stale baseline.** The prose said 36.8
  per cent under uniform assignment; the artifact says 34.1. New macros
  (`\LFSUniformCushion`, `\LFSSelectionShiftMin/Max`, `\LFSSelectionGapMin/Max`)
  make the comparison generated rather than typed. Corrected reading: every
  LFS-shaped model raises displacement cushioning by 2.0-3.3 points and
  narrows the headline gap to 6.5-7.9 points. Reported as a first-order
  sensitivity, not as robustness.
- **M3 employment-state step is near-mechanically zero.** A `jsa_bounding`
  block quantifies the omission (New Style JSA worth up to 5.0 points of
  cushioning, ~50x the measured step). Intro, results and discussion now
  rest the contribution on the concentration channel.
- **M4 analytic benchmark.** The marginal-versus-average-rate derivation is
  stated before the simulation result, with the three reasons the simulation
  is not a restatement of it.
- **M5 Dolls et al. (2012) sign reversal.** New reconciliation subsection:
  the tax side agrees (23.1 vs 25.2 per cent) and the entire reversal sits in
  the out-of-work benefit channel (2.7 vs 16.3 per cent).
- **M6 pi = 1 mischaracterised.** Derivation added: pi = 1 is the
  constant-wage-share benchmark, not full wage incidence (which would need
  pi well above 3). The turnover-versus-GVA denominator offset is stated.
- **M7 Shapley claim withdrawn** from the manuscript and from
  `factorial_decomposition.json`'s design note: the diffuse-displacement
  coalition is undefined, so no Shapley value exists.
- **R1-R5 stale and hardcoded numbers.** `paper/generated_validation.tex` is
  new; the appendix reconciliation arithmetic is generated
  (`\FullWageImpliedOffset` etc.); `\TakeupSeeds` replaces the reused
  `\PensionSeeds`; `\SubmissionCushionDifferenceMax` now excludes the thin
  OBR three-month cell (12.8 -> 11.7) with `\SubmissionThinCellCushionDifference`
  for the appendix.

### Closed in code, pending a licensed-data run

- **M8 event study.** Anticipation-window exclusion (Jan-Apr 2025), figure
  re-normalisation to a 2024 base, continuous-trade and Poisson
  specifications, and per-group power diagnostics are implemented. The stored
  estimates predate them; the manuscript reports the caveats and does not
  quote unrun specifications.
- **R6/R7 bootstrap and inclusion probabilities.** The ratio-of-pooled-sums
  estimator, support-size count and the assignment-conditioning caveat are
  implemented; the per-record balanced-versus-Bernoulli inclusion diagnostic is
  `analysis/assignment_inclusion_diagnostic.py` (`make assignment-inclusion`).
  Both need `data/` present. The manuscript states the limitation rather than
  claiming a result.

### Known drift hazards not yet closed

- The customs totals (£55,600m / £52,500m) are not persisted until
  `make inputs` runs on raw HMRC/ONS inputs; `build_trade_by_sic.py` now
  writes `results/trade_build_totals.json` and
  `write_validation_macros.py` emits the macros only when it exists.
- `34,966` person records, `12,800` with employee earnings and the
  "above 19 thousand" largest person weight in the methodology section have
  no backing artifact.
- The `all_entitled` take-up grid quoted in the manuscript comes from the
  superseded epsilon = 2 calibration and is labelled as such. Re-running
  `make takeup-entitled` at the unit stress is the first outstanding item.

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
  fails if the declared production artifacts have mixed draw counts. The
  compared set in `analysis/write_paper_results.py` is the eight central
  scenarios plus the two OBR-low anchors, the two transition scenarios and the
  two rent-sharing artifacts when present — fourteen with the current results
  directory. The companion assignment-design check cannot yet run: the stored
  artifacts predate `MonteCarloResult.selection_method` and it warns instead.
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
- The complete gate passes the full synthetic test suite (328 passing,
  5 skipped as of 17 August 2026; 321/4 at the time of this round), validates
  the frozen input manifest, regenerates all manuscript values and builds a
  visually inspected manuscript (then a main paper plus an online supplement;
  now a single document — see the fourth-round disposition).

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
