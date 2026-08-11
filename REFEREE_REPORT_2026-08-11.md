# Referee report

**Manuscript:** "Adjustment Margins and Tax–Benefit Cushioning of Earnings Losses in the United Kingdom"
**Repository:** PolicyEngine/uk-trade-shock-study (HEAD `5ff2d63`)
**Recommendation:** Major revision.

---

## Summary

The paper holds a fixed aggregate employee-earnings loss (~£886m/year, calibrated from the
2025 US tariff schedule through a declared accounting bridge) and varies its incidence across
FRS 2024–25 households in PolicyEngine UK: diffuse proportional wage cuts, a concentrated wage
cut imposing the displacement draw's exact worker-level losses without job loss, and
displacement. Cushioning (1 − ΔY/ΔE) is 43.9% for diffuse cuts and 34.0 ± 4.7% for
displacement; a factorial middle cell attributes 9.9 of the 10.0-point gap to loss
concentration and 0.1 points to the employment state itself.

The project is unusually disciplined about epistemic status, the code is clean and well
tested, and the honesty about the failed sector-level falsification is admirable. My concerns
are that (a) one of the two headline "robustness" results is an inert experiment presented as
evidence of insensitivity, (b) a sensitivity that *does* move the result by a third of the
headline gap is reported against a stale and incorrect baseline, and (c) the paper does not
engage the sign reversal against its own closest literature benchmark.

---

## Major points

### M1. The take-up "headline sensitivity" is vacuous, and the claiming margin that binds is frozen

Section 4.5 and §5.2 present the take-up grid as a headline robustness check: varying the
new-entitlement claiming rate over {0.55, 0.80, 1.00} and restoring the stale pre-shock flag
"leaves the unit-stress displacement cushioning unchanged to within 0.1 points."

The artifact shows something stronger and less reassuring. In `results/referee_fixes.json`
(`takeup_headline.displacement`), all four conventions return **bit-identical** values —
cushioning 0.32987738933694183 and Exchequer £348.44901376m under 0.55, 0.80, 1.00 *and* the
stale flag. Identical to 17 significant figures across a stochastic redraw means the redraw set
is empty in every seed: `redraw_uc_takeup` (`shocks.py:1011`) fires only for benefit units whose
potential award moves from zero to positive, and essentially no unit qualifies. The experiment
has no content. It is not evidence that cushioning is insensitive to claiming; it is evidence
that this particular parameter binds on nobody.

Meanwhile the claiming margin that *does* bind is held fixed by construction.
`results/takeup_diagnosis.json` reports that 55% of displaced workers (weighted) receive zero UC
because of the non-take-up flag, and that under the earlier convention — redrawing all changed
units — displacement cushioning ran from 0.3445 (stale flag) to 0.4153 at full take-up. That is
a **~7-point range, comparable to the entire 10-point headline gap.** The paper's own prior
referee audit (`REFEREE_AUDIT.md:50`) asked for previously-entitled non-claimants to be modelled
separately; the current design instead freezes them and reports the resulting null as robustness.

The paper does note this in one clause ("the residual convention that this grid cannot test..."),
but the framing inverts the finding's weight. Please:

1. Report the redraw-set diagnostic explicitly — number and survey weight of benefit units
   actually redrawn per seed. If it is zero, say so, and drop the claim that take-up was tested.
2. Run and report the sensitivity that binds: vary the claiming flag for all UC-entitled units
   containing a displaced worker (the `takeup_diagnosis.json` design), and report the resulting
   cushioning range as a headline uncertainty dimension alongside the 10-point gap.
3. Adjust §6 and the abstract accordingly. The honest statement is that the wage–displacement
   gap is bounded above by the modelled claiming convention and could be materially smaller
   under plausible alternatives.

*(Note also that `\PensionSeeds` is reused for the take-up draw count in §5.2 — the macro is
written from `pension_channel.n_seeds` = 50 at `write_referee_macros.py:56`, but
`takeup_headline.displacement.n_seeds` is 25. The paper states the wrong draw count.)*

### M2. The LFS worker-selection sensitivity is compared against a stale baseline, and it is not small

Section 5.4 and §6 state that LFS-shaped risk models give cushioning of "36.0–37.4 per cent
against **36.8** under uniform assignment," concluding the spread is smaller than any single
specification's assignment SD.

The uniform number is wrong. `results/full_tariff_displacement.json` (100 draws) gives
`cushioning_rate_mean` = **0.3407**, and `paper/generated_lfs_selection.tex` — generated from the
same artifact — prints 34.1 in the very table the sentence describes. The paper's own table and
its surrounding text disagree by 2.7 points. (36.8 is in fact the *cells* model's value, 0.3688.)

With the correct baseline the finding reverses in character: all three LFS-shaped selection
models raise displacement cushioning by **+1.9 to +3.3 points** over uniform assignment, which
shrinks the wage-minus-displacement gap from ~9.9 to roughly 6.5–7.9 points — i.e. worker
selection accounts for a fifth to a third of the headline contrast. That is one of the larger
sensitivities in the paper and is currently reported as evidence of robustness. This needs
correcting in §5.4, §6 and the abstract's framing, and the LFS selection range belongs in the
headline uncertainty discussion, not the supplement.

The same stale 36.8 propagates into `appendix.tex:34`, where the reconciliation computes
0.368 × £882m ≈ £325m; with the correct 0.341 the household-side offset is £301m and the wedge
against the £387m Exchequer cost is £86m — which is what the sentence's own stated "£87–95
million" range requires. The displayed arithmetic contradicts the displayed conclusion.

### M3. The "employment-state" factor is inert by construction, so the decomposition cannot bear the weight placed on it

The paper's second stated contribution is the factorial split: 9.9 points from concentration,
0.1 points from the employment state. But every institution in the UK system that conditions on
employment status is absent from the model, and the paper says so itself in four separate places:
New Style JSA "is not activated by the imposed transition" (§5.5), monthly assessment periods,
the five-week wait and conditionality are outside the annual model (§1, §5.2), and "work-search
conditionality carries no separate fiscal machinery" (§4.3). The concentrated cell and the
displacement cell zero the same earnings for the same workers; the only remaining difference is
a status label that no modelled formula reads. The residual 0.1-point state step is therefore
close to a mechanical zero, and the channel split confirms it (income tax and NI exactly zero by
construction; UC 0.2 points).

I do not think this invalidates the exercise, but it does mean the decomposition measures a
property of the model rather than of the UK system, and the current framing — "the answer is
one-sided," "moving those same workers into unemployment accounts for only the remaining 0.1
points" — reads as a substantive finding. Two things would fix this:

- State up front that the state-step estimate is bounded above by the model's coverage of
  status-contingent institutions, and give the reader an order of magnitude for what is missing.
  New Style JSA at £92.05/week for 182 days is roughly £2,400 per displaced worker; against a
  £48,272 mean loss that is ~5 points of cushioning on its own — fifty times the measured state
  step. A back-of-envelope calculation of this kind, in the text, would let readers price the
  omission.
- Consider whether the paper's real contribution is better stated as the *concentration* result
  alone, which is well identified within the model and genuinely interesting.

### M4. The concentration result should be shown against its analytic benchmark

The mechanism — a marginal-versus-average-rate effect under a progressive schedule — is
analytically predictable. For a diffuse marginal cut, cushioning ≈ the effective marginal
deduction rate; for a complete loss, it ≈ the average deduction rate, which is lower because of
the personal allowance. The channel split (income tax contributes 16.1 of the 9.9-point
concentration step, partially offset by −1.8 from UC) is exactly this.

The paper would be considerably stronger if it derived the closed-form benchmark for a
representative exposed worker (£48,272, 2026 parameters) and showed the simulation reproducing
it, then attributed the residual to household composition, UC interactions and pension
accounting. As written, a reader can reasonably ask whether the simulation is discovering
anything the tax schedule does not already imply. Showing that it matches — and then showing
where and why it *departs* — converts that objection into a validation.

### M5. Reversal against Dolls et al. (2012) is not addressed

§3.4 cites Dolls, Fuest and Peichl's UK cells as "the right comparator": income shock 0.352,
unemployment shock 0.415, and describes the paper's 43.9% / 34.0% as "of a similar order."
They are of a similar order but **the ordering is reversed** — Dolls et al. find the
unemployment shock stabilised *more*, this paper finds it stabilised *less*. That is the single
most informative external check available, and the paper passes over it.

The reversal is plausibly explained by the paper's own modelling choices, and saying so would be
a strength rather than a weakness: (i) Dolls et al.'s unemployment-shock stabilisation is 39%
benefit-driven (0.163 of 0.415), whereas here UC supplies only 3.0 of 34.0 points — 8.8%;
(ii) New Style JSA is absent; (iii) the frozen non-take-up flag zeroes UC for over half the
displaced (M1); (iv) exposed workers are high-earning (£48k mean), so out-of-work benefits are
small relative to the loss; (v) the UC household means test with partner earnings. Please add a
subsection reconciling the two, with the channel decomposition placed against theirs. If the
gap is mostly (ii)+(iii), that is important for how the headline should be read.

### M6. π = 1 is described as a severe assumption; under the stated bridge it is the neutral one

Equation (1) is s_j = τ_j x_j ε π with x_j = US goods exports ÷ ONS ABS division *turnover*, and
s_j interpreted as a fractional loss of the division's *employee wage bill*. Under that
construction, s_j = ε τ_j x_j with π = 1 is exactly the constant-labour-share benchmark: the
wage bill falls in the same proportion as output. Full incidence of the revenue loss on wages —
which is how §4.1 describes π = 1 ("assigns the full calibrated exposure to the employee wage
bill", "intentionally severe") — would instead require π ≈ 1/(labour share) ≈ 3–5 in
manufacturing. Conversely, using turnover rather than gross value added in the denominator of
x_j understates exposure, partly offsetting.

Neither error is fatal, but the paper repeatedly leans on π = 1 being conservative-in-the-severe-
direction, and that characterisation does not follow from the stated algebra. Please give the
one-line derivation, define π explicitly as the elasticity of the sector employee wage bill to
sector US-export revenue, state which denominator convention x_j uses and why, and re-describe
π = 1 as the proportional benchmark rather than an upper bound.

### M7. The Shapley claim is not available with the middle cell undefined

§4.3, §5.2 and `factorial_decomposition.json` all assert that the sequential path
diffuse → concentrated → displaced "coincides with the Shapley allocation over the two factors."
A Shapley value over two factors requires the value of both orderings, including the
diffuse-displacement coalition — which the paper correctly says is undefined because employment
is binary. With that coalition missing the Shapley value is not defined, so the decomposition
cannot coincide with it. The factors are also not symmetric (concentration is continuous, state
is binary). Either impose an explicit convention for the missing cell and show the result, or
simply drop the Shapley sentence — the sequential decomposition stands on its own as the only
feasible path, which is what the paper actually needs.

### M8. The HMRC event study: specification issues make the "−31.9%" hard to interpret

The falsification exercise is the right instinct, but three features of
`analysis/hmrc_destination_event_study.py` limit what can be concluded.

1. **Front-running contaminates the pre-period.** Only April 2025 is dropped
   (`ANTICIPATION_MONTH`, line 97). UK goods exports to the US surged through Q1 2025 ahead of
   the tariffs; that surge sits in the estimation pre-period and inflates the estimated
   post-decline. Worse, the plotted series is normalised to **January–March 2025**
   (lines 156–159) — the anticipation window itself — so Figure S1 measures the post-policy gap
   relative to the front-running peak. Please re-estimate excluding 2025Q1 (or the whole of
   2025 up to May) from the pre-period and re-normalise the figure to a 2024 base.
2. **log(1+V) on a zero-filled balanced panel.** Absent flows are set to zero (line 51) and the
   outcome is `log1p` (line 56), so the control term is (1/3)Σ log(1+V) with genuine zeros
   entering as 0. Extensive-margin movement in the small comparison destinations therefore maps
   mechanically into the gap. PPML on levels, or restricting to products with continuous trade
   in all four destinations, would be the standard fix and is a natural robustness column.
3. **The tariff-intensity falsification has essentially no power.** The steel/auto group has an
   effective product count of **7.5** (versus 99.6 for the rest), and its estimate is
   −20.0% with p = 0.29. The paper reports this as "the tariff-intensity ordering fails," but a
   test on 7.5 effective clusters cannot fail or pass — it is uninformative. The correct
   statement is that the design lacks power to discriminate, which is a weaker but defensible
   conclusion. Similarly, the sample-start sensitivity (−31.5%, −27.4%, −17.5% as the pre-period
   shortens) together with the borderline differential trend (p = 0.057) suggests the post dummy
   is partly absorbing a trend; this deserves a sentence.

The placebo results (2019–2023, all insignificant) are clean and should be kept prominently.

---

## Reproducibility and internal-consistency defects

These are specific and checkable; all were found by reading the repository against the
manuscript.

| # | Location | Issue |
|---|---|---|
| R1 | `sections/results.tex:245`, `sections/discussion.tex:64` | "36.8 per cent under uniform assignment" contradicts `generated_lfs_selection.tex` and `results/full_tariff_displacement.json` (34.1). Hardcoded, not macro-driven. |
| R2 | `sections/appendix.tex:34` | 0.368 × £882m ≈ £325m uses the same stale figure; the stated £87–95m wedge requires 0.341 (→ £301m). Internally inconsistent. |
| R3 | `analysis/write_referee_macros.py:56, 66` | `\PensionSeeds` (=50, from the pension block) is reused in §5.2 to describe the take-up draw count, which is 25 in `takeup_headline`. |
| R4 | `sections/discussion.tex:7` | The headline range `\SubmissionCushionDifferenceMin–Max` = 9.8–12.8 includes the OBR 3-month cell, which §5.1 excludes from the main table on thin-support grounds (1.2 effective records, largest record 91.9% of the loss). Excluding it gives 9.8–11.6. Use one rule consistently. |
| R5 | `sections/appendix.tex:4`, `sections/results.tex:219` | £55,600m / £52,500m / 12.57% / 12.71% / rank correlation 0.00 / 0.58 are hardcoded, contrary to the README's stated contract that "all numerical values draw from validated generated files." Promote to generated macros so the build fails if they drift — which is exactly how R1/R2 would have been caught. |
| R6 | `analysis/bootstrap_uncertainty.py:112–130` | Two things to state or fix: (i) the resample holds the *assignment* fixed — households are reweighted but the ~10 selected records are those of the original draw — so the interval is conditional on the original selection and understates true sampling variability; (ii) the statistic is a mean of per-draw ratios with near-zero denominators retained, which produces the visible right skew (point 10.7, SE 1.2, CI 9.1–13.9). Report a ratio-of-pooled-sums estimator alongside. |
| R7 | `uk_trade_shock_study/shocks.py:380–445` | The balanced sampler's best-of-N selection on a wage-bill-dominated score changes record-level inclusion probabilities. The paper says so but does not quantify it. Please report empirical inclusion probabilities under balanced vs Bernoulli for the top-weight exposed records — with the largest record supplying 28.7% of the loss on average, readers need to know whether it is selected at its nominal rate. |
| R8 | Environment | I could not execute `make check` locally: `uv` is not present and the pinned interpreter is Python 3.13 (local 3.14). CI covers this, and the lightweight contract is clearly documented, so this is a note rather than a criticism. |

---

## Minor points

1. **EPD section.** The paired full-minus-EPD Exchequer difference is £90m ± £98m — dispersion
   larger than the point estimate. §7.1 currently reads as though the deal has been priced.
   Either state plainly that the difference is not resolvable at this draw count, or raise the
   draw count for this contrast specifically.
2. **The tariff framing is now load-bearing in the wrong places.** After the retitling, the
   estimand no longer needs the tariff bridge — but the EPD counterfactual and the pharmaceutical
   discussion do, and the bridge has failed its own sector-level test. Consider moving §7.1–7.2
   wholly to the supplement, leaving the tariff schedule as what §4.1 says it is: a source of
   sector weights.
3. **Scale context.** £886m is roughly 0.03% of GDP and ~17,200 represented workers. One
   sentence early on placing the shock against total UK employee compensation would help readers
   calibrate the near-zero national poverty and Gini movements, which currently invite over-reading.
4. **The "other residual" channel** is 3.8–4.5 points of cushioning (≈13% of the displacement
   total) and is unnamed in the channel split. It appears to be the pension/salary-sacrifice
   outflow (matching the 4.4%/4.9% pension shares in `referee_fixes.json`). Name it in the table.
5. **Self-employment.** The shock is applied to employees only. Exposed divisions contain
   self-employed workers with no equivalent margin; worth one sentence in the scope limits.
6. **Concentrated cell coherence.** Workers in the middle cell hold baseline hours at zero pay.
   This is a deliberate accounting device, but say so explicitly — it is not a labour-market
   state anyone occupies, and a reader encountering "employed at baseline hours" with zero
   earnings will wonder about NMW and hours-conditional entitlements.
7. **Poverty precision.** Changes reported as 0.036 / 0.015 / 0.003 pp are below any meaningful
   resolution for the FRS. The caveat is present; consider simply reporting "<0.05 pp" and
   moving the decile profile forward, which carries the real distributional content.
8. **Abstract.** Given M1–M2, the abstract's "sensitivity analysis varies pensions, take-up,
   duration equivalents, adjustment shares and worker selection" overstates two of the five. The
   pension and duration analyses are solid; take-up and worker selection need the corrections above.

---

## What is good and should survive revision

- The primitives table in §4.1 and the discipline of labelling every object Data / Assumed /
  Computed is exemplary and rare. Keep it.
- Reporting the failed sector-rank falsification in the *introduction* rather than burying it is
  the right call and should not be softened.
- The record-support diagnostics (effective loss records, largest-record share, leave-one-record-out,
  leave-one-sector-out) are more than most microsimulation papers provide and correctly
  constrain the subgroup claims.
- The pension-gross recomputation is a well-designed check that answers a real objection cleanly.
- `redraw_uc_takeup`'s hard-error contract and the cache-flush verification
  (`shocks.py:1046`) are the kind of defensive engineering that makes a replication package
  trustworthy.
- The separation of assignment dispersion, numerical MC error, and record-resampling
  sensitivity — with no significance claim attached to any of them — is handled better than in
  most published microsimulation work.

## Priority for revision

M1 and M2 are the two that change what the paper concludes; both are correctable with runs the
repository can already do. M5 is the cheapest large gain in credibility. M3, M4, M6 and M7 are
framing and derivation fixes. M8 affects only the supplement.
