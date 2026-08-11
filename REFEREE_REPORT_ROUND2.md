# Referee report — second round (consolidated)

**Manuscript:** "Adjustment Margins and Tax–Benefit Cushioning of Earnings Losses in the United Kingdom"
**Status:** revision of the version reviewed 11 August 2026
**Recommendation: major revision.**

I opened this round intending to recommend acceptance in principle. A full trace of every
quantitative claim back to its artifact changed that. The revision fixed the round-one
defects, but it introduced a new class of error that is more serious than anything in round
one: **the manuscript now describes estimator features that exist only in the code and
reports results from them that were never computed.** That has to be fixed before this can
go further.

---

## A. Claims of results that do not exist (blocking)

The event-study estimator was rewritten but the stored artifact
(`results/hmrc_destination_event_study.json`) predates the rewrite and cannot be regenerated
without the HMRC panel. The prose was written against the *code*, not against the *artifact*.
Three passages therefore report analyses that have never been run:

| Passage | Claim | Artifact reality |
|---|---|---|
| `methodology.tex` §3.2 | "omitting a January–April 2025 anticipation window" | The artifact has `anticipation_month_omitted: 202504`. Every reported estimate omits April only. |
| `results.tex`, `supplement_body.tex` | "the supplement now reports an anticipation-window-excluded specification alongside the original single-month omission" | No window-excluded fit exists. |
| `results.tex`, `methodology.tex`, `supplement_body.tex` | "continuous-trade and Poisson specifications are reported as checks" | No continuous-trade fit and no PPML fit exist in the artifact. |
| `supplement_body.tex` | "the diagnostics now record each group's … interval width and minimum detectable effect, together with a formal test of the high-minus-other difference" | None of those three fields exist in the artifact. |

The generated `\HMRCAnticipationSpec` macro renders, correctly, "April 2025 only (legacy
convention; re-run `make trade-event-study`…)" — so the paper contains a machine-generated
statement and three hand-written statements that contradict it. The macro is the one telling
the truth.

**Required:** either re-run the panel and let the numbers change, or rewrite all four passages
in the form "implemented in the estimator but not present in the run reported here". No
in-between. The supplement's minimum-detectable-effect claim is arithmetically correct
(2.80 × 0.211 = 0.591 log, against a 0.224 log estimate) but is computed nowhere and stored
nowhere, so it must be labelled as the referee's arithmetic or emitted as a macro.

## B. "The reported gap is an upper bound" is false as written

The abstract closes with it unqualified. Across the six main-table cells the gap is:

| Anchor | Months | Gap (pp) |
|---|---|---|
| OBR | 3 | 12.83 *(excluded from main table)* |
| OBR | 6 | 10.98 |
| OBR | 12 | 11.55 |
| UNIT | 3 | 11.70 |
| UNIT | 6 | 9.76 |
| **UNIT** | **12** | **9.96 ← the headline** |

The headline cell is the second-smallest of six. The pension-gross income concept *widens* the
gap (9.96 → 10.49). The bootstrap places it at 10.67 and leave-one-record-out at 10.25–11.31 —
both above the headline. The claim is defensible only with respect to worker selection, the
claiming convention, and the absent New Style JSA. Say that.

Two related directional errors, both introduced in this revision:

- **`results.tex`, take-up paragraph:** the gap is an upper bound "in the direction that
  non-claiming and the unmodelled UC capital test both push." They push opposite ways. Higher
  claiming raises displacement cushioning and *narrows* the gap; the capital test lowers it
  and *widens* the gap.
- **`policy.tex`:** after listing claiming (7.1 pts), the capital test, and absent New Style
  JSA (5.0 pts) — "Each of those points in the same direction, towards less protection than
  the headline figures imply." Two of the three point the other way. Modelled non-claiming and
  the JSA omission both mean the model *understates* protection. The two largest terms carry
  the opposite sign to the one claimed.

And a magnitude error: `intro.tex` says worker selection and the claiming convention "each
move the headline contrast by an amount comparable to a third of its size." Selection is about
a third (2.1–3.5 of 10.0); claiming is 7.1 of 10.0, i.e. **71 per cent**. The intro
contradicts itself two paragraphs earlier ("comparable to the headline gap") and contradicts
the discussion. The abstract's "bounds it comparably" understates it by a factor of ~2.4 and
should not imply the two are interchangeable — especially as the claiming figure is measured
at a different calibration and cannot be subtracted from the unit-stress gap at all.

## C. The Dolls et al. reconciliation compares a decomposition that does not add up

`factorial_decomposition.json`, unit anchor, 5 common seeds, displacement:

```
income_tax                    18.360
employee_national_insurance    4.723
universal_credit               2.993
other_benefits                -0.304
other_residual                 4.509
                              ------
SUM                           30.281      ← vs headline displacement cushioning 33.97
```

The wage-cut column sums to 43.931, exactly its headline — so the decomposition is exhaustive
and the 3.69-point shortfall is pure 5-seed drift on the stochastic margins, not a missing
channel. Consequences the paragraph does not state:

1. The implied total gap against Dolls is 41.5 − 30.3 = **11.2 points** from the channel
   numbers but 41.5 − 34.0 = **7.5 points** from the headline. Neither appears.
2. "The entire reversal sits in the out-of-work benefit channel" holds only after netting off
   the 4.5-point pension/salary-sacrifice residual, which has **no counterpart in Dolls'
   decomposition at all**: 2.1 (tax+NI) + 13.6 (benefits) − 4.5 (residual) = 11.2. The residual
   does a fifth of the reconciliation's work and goes unmentioned.
3. The word "precisely" is not earned.

The same drift breaks the factorial step: the 5-seed concentration step is **13.45 points**
against the reported 9.87 — a 36 per cent discrepancy that the existing one-sentence caveat
("do not reconcile exactly") badly understates. Re-run the channel split at 50 draws; it is
cheap and it removes the problem rather than documenting it.

## D. Undisclosed design and draw-count mixing

- **The leave-one-division-out audit runs at 20 draws.** It is cited in the introduction,
  results and discussion as robustness for the headline, and the paper's declared draw-count
  conventions are 50, 100, 25 and 5. Twenty appears nowhere. Disclose it.
- **The LFS selection gap mixes designs.** `\LFSSelectionGapMin/Max` are computed against the
  100-draw Bernoulli `full_tariff_displacement` (own gap 9.863), but the prose narrates the
  narrowing as being "from `\PrimaryCushionDifference`", the 50-draw balanced 9.958. The
  writer `write_lfs_selection_results.py` contains an explicit assertion forbidding exactly
  this mixing; the prose does it anyway. The discrepancy is only 0.1 points, but the principle
  matters given round one.
- **`\LFSUniformCushion` (34.1) is described as "the uniform benchmark that the rest of the
  paper uses."** The rest of the paper uses 34.0 — a different design and draw count.
- **The take-up grid reports 33.0** (Bernoulli, 25 seeds) inside a section whose headline for
  the same scenario is 34.0 (balanced, 50 draws). The difference is never explained, and the
  "Estimator provenance" paragraph that exists precisely to handle this case does not cover it.
- **`appendix.tex` still says the take-up grid "is scored at the full comparator draw count."**
  It is 25 against a comparator of 50.

## E. Internal contradictions

- **π = 1.** `methodology.tex` now derives that π = 1 is the constant-wage-share benchmark and
  explicitly "does not reach the full-incidence case." `results.tex` still says the reference
  "additionally assumes full wage-bill incidence." One of these was fixed in the revision and
  the other was missed. The results sentence should read "at the constant-wage-share benchmark
  π = 1."
- **The employment-state step appears as 0.1 and 0.2 within one paragraph** — the first is the
  50-draw step, the second the 5-seed UC channel. On the 0.2 basis the New Style JSA comparison
  is 25×, not the ~50× the paper states three times. State the seed basis in both places.
- **The appendix's shock mechanics describe the Bernoulli comparator** ("each employee … is
  independently selected with probability s_j") as though it were the primary estimator, which
  §3.4 says is balanced systematic assignment that "intentionally changes exact record-level
  marginal inclusion probabilities."
- **The appendix describes the `all_entitled` take-up scope** ("benefit units whose
  circumstances changed and that are entitled post-shock") as the implemented rule. The
  implemented default is `new_entitlement`, and §3.7 says `all_entitled` has not been run.
- **"The contrast is essentially unchanged"** under the pension-gross concept (`intro.tex`,
  `discussion.tex`) describes a 5.3 per cent widening that `results.tex` reports honestly as
  10.0 → 10.5.
- **"Roughly an eighth of measured cushioning on either margin"** for the pension residual:
  13.3 per cent under displacement but 8.6 per cent — nearer a twelfth — under the wage cut.
- **Discussion grammar** attaches "a fifth to a third of the contrast" to the *resulting gap*
  (6.5–7.9 points, which is 65–79 per cent of the contrast) rather than to the *narrowing*.

## F. Vintage caveats are attached to some numbers and not others

Everything sourced from `results/takeup_diagnosis.json` comes from the superseded ε = 2
high-case calibration, and the writer's own docstring says it "must therefore always be
labelled as such." Four places lack the label:

| Number | Location | Caveat |
|---|---|---|
| 7.1-point claiming spread | `policy.tex` | **missing** |
| 55.0 per cent zero-UC-from-non-take-up | `results.tex` take-up paragraph | **missing** — the "two caveats" sentence that follows attaches to the *spread*, not this share |
| same, as "over half the displaced by weight" | `results.tex` Dolls paragraph | **missing**, and it is offered there as an explanation of the reversal *at the unit stress* |
| 46.9 per cent post-shock take-up among the displaced | `methodology.tex` | **missing**; this is calibration-dependent, since it depends on who gets displaced |

The 55.0 per cent figure additionally comes from **seed 0 alone**, which is stated nowhere.

## G. Editorial

- **Eleven passages narrate what earlier drafts got wrong.** That belongs in the response
  letter. Keep only the take-up subsection, where the superseded convention is what motivates
  the current one.
- **Methodology (~4,600 words) now exceeds results (~4,400).** Move standing caveats to the
  supplement.
- **Title and second contribution overstate what survives.** The revision establishes that the
  employment-state channel is inert in this model. What the paper identifies is loss
  *concentration*. Also: the 0.2-point UC residual in the state step needs an explanation —
  both cells zero the same earnings for the same workers, so something is reading employment
  status — or should be reported as zero to numerical precision.
- **The tariff application is vestigial.** Nothing in the estimand depends on it, the sector
  calibration receives no corroboration, and the EPD section says a reader should take nothing
  quantitative from it. Cut the EPD counterfactual; reduce tariffs to the provenance of the
  sector weights.
- **θ ≈ 0.2–0.3, and hence π ≈ 3–5, is unsourced.** No artifact bears on it. ONS ABS supplies
  the turnover denominator already and also reports employment costs by division, so it is one
  join from being sourced. Otherwise soften to "well above 3."

## H. Defects in the new estimator code

An adversarial review of the revision's code found three further blocking items. I confirmed
the first two directly.

**H1 (blocking, confirmed). The published figure was never regenerated, and its caption is
false.** `results/figures/hmrc_destination_event_study.png` and
`hmrc_destination_event_study_monthly.csv` are unchanged in the diff. The CSV's
`normalised_gap` averages exactly 0.0000 over January–March 2025 and +0.1359 over 2024: it is
still normalised on the front-running window. `supplement_body.tex` now captions that same file
as "normalised to a 2024 base. The earlier January–March 2025 normalisation was withdrawn
because that window is itself the pre-tariff front-running surge." The figure the paper
includes is normalised on precisely the window the caption says was withdrawn, and is
mislabelled by 0.136 log points. This is section A's problem reaching the figures.

**H2 (blocking, confirmed by reading and by the reviewer's instrumented run).
`_ppml_irls` declares convergence after a *failed* line search and accepts the rejected step.**
At `analysis/hmrc_destination_event_study.py:312-322` there is no flag recording whether the
damping loop broke or exhausted; `beta, loglik, mu = candidate, candidate_loglik, candidate_mu`
executes unconditionally, so on exhaustion the candidate with the *worse* log-likelihood is
adopted. `damping` is then one halving past the candidate actually evaluated (~9.3e-10), which
satisfies `max(|damping*step|) < tolerance` for essentially any step — so a failed line search
reports `converged: True`. Compounding this, the acceptance tolerance `1e-10` is absolute,
while the concentrated log-likelihood is `y @ linear` with `y` in pounds; at HMRC magnitudes
one ulp exceeds the tolerance by orders of magnitude, making spurious rejection the normal case
near the optimum rather than an edge case. The reviewer reproduced a run reporting
`converged=True` immediately after a rejected, log-likelihood-worsening step.

The sandwich estimator itself is correct — verified against `sm.GLM` Poisson with explicit pair
dummies to rel ≈ 1e-14 on independent fixtures at £1e2–1e9 scales, with the right DOF
correction, and the repo's own 1e-8 test is genuine rather than rigged. The estimator is sound;
the solver bookkeeping is not.

**H3 (blocking). Rank-deficient and separated fits still emit point estimates.**
`fit_ppml_model` computes a `rank_deficient` flag and then returns `log_effect`,
`standard_error`, `p_value` and CIs built from `np.linalg.pinv(hessian)` anyway, contradicting
its own docstring. On data with a true effect of −0.30 the reviewer obtained −0.088 (p = 0.70),
−0.564 (p = 0.0007) and −6.8e-19 (p = nan) from three sample spans, all with
`converged: True, rank_deficient: True`. Under separation on `us_post` it returns
`log_effect = −100.54, standard_error = 1.25e-16, p_value = 0.0`; `separated_pairs_dropped`
catches only all-zero pairs. No consumer reads either flag, so "a 56 per cent decline,
p = 0.0007" could be quoted straight out of a numerically meaningless fit.

**H4 (medium-high). `continuous_trade_products` selects on post-treatment outcomes.**
`main()` calls it with `months=None`, requiring positive trade in every month of the whole
panel *including the post-policy period*. That drops exactly the products whose US flow went to
zero after the tariff — the extensive-margin exits the check exists to address — biasing the
robustness estimate toward zero by construction. The `months=` parameter exists and the unit
test exercises it; `main` never restricts to the pre-period.

**Lower-severity, all confirmed by the reviewer:** the bootstrap's partial-cache resume path is
dead code (a comprehension over all `SCENARIOS` raises `KeyError` before the new guards are
reached); `ANTICIPATION_WINDOW` and `ANTICIPATION_LEAD_MONTHS` are independently hardcoded with
no consistency assertion, so widening one silently breaks placebo symmetry; the
`all_entitled` block's `stale_baseline_flag` cell carries a `redraw_diagnostic` mislabelled
`uc_takeup_scope: "new_entitlement"`, and `\TakeupEntitledStale` is read out of exactly that
cell; `_summarise_diagnostics` defaults `n_redrawn` to `0`, which is the load-bearing
"grid is inert" signal, so schema drift would fabricate that finding; `\JSAWeeklyRate` came
from the hardcoded fallback rather than the PolicyEngine parameter tree, and since
`make paper-values` re-runs `--only jsa` on every build the value will flip depending on which
machine builds the paper; and `\TakeupEntitledSpread` renders 7.1 while the two macros quoted
in the same sentence differ by 7.0 (true value 7.083), which a reader will subtract.

**Categories where the reviewer found nothing:** the `new_entitlement` RNG stream and
`set_input`/cache-invalidation ordering are genuinely unchanged, so stored results still
reproduce; the `all_entitled` redraw set cannot touch unchanged units; the pooled bootstrap
estimator pools correctly and is genuinely resampled; month arithmetic and placebo window
construction have no off-by-one; the JSA day-to-week conversion and denominator are right.

## What is genuinely fixed

The macro layer is clean: all 166 generated macros used in the manuscript are defined, trace
to a writer and an artifact, and render the artifact's value. The appendix reconciliation, the
thin-cell exclusion, the `\TakeupSeeds`/`\PensionSeeds` separation, the withdrawn Shapley
claim, and the LFS baseline correction all hold. The build regenerates and the suite passes
(146 passed, 3 skipped). The problems above are all in prose that no macro covers, or in prose
that mislabels which artifact a macro came from — which is the failure mode the round-one
fixes were designed to prevent, arriving one level up.

## Priority

A is blocking and needs no new computation to fix honestly. B, C, E and F are corrections of
statements that are currently wrong. D needs disclosure. G is what would make this a good
paper rather than a defensible one.
