# Evidence-informed scenario design (August 2026)

This memo records the assumptions that can be defended with the data currently
in the repository. It is intended to guide the redesign around a mixed,
conditional fiscal-incidence estimand; it does not modify the simulation code.

## Recommended estimand

> Conditional on a specified export-exposure-calibrated employee earnings loss,
> how do alternative labour-market adjustment paths change household disposable
> income, poverty, inequality, and the Exchequer?

This is a statutory microsimulation exercise, not an estimate of the realised
tariff effect. The aggregate loss and its allocation across workers must remain
separate scenario axes.

## Central calibration and defensible ranges

| Axis | Central analysis | Sensitivity range | Evidence/interpretation |
|---|---:|---:|---|
| Export-demand elasticity | 0.4 (OBR-style lower anchor) | 0.4, 1.0, 2.0 | The project does not identify this coefficient; 1.0 is a unit stress, 2.0 an outturn-shaped upper case. Do not call any value estimated. |
| Employee earnings incidence | 1.0 conditional pass-through | 0.5, 1.0 | Incidence is a held-fixed accounting choice. 0.5 is a conservative sensitivity; no firm-level data identify the UK share. |
| Adjustment mixture | 70% temporary non-employment + 30% post/within-job earnings loss | displacement share 0.5--0.9; wage share 0.1--0.5 | UK/European worker evidence (De Lyon--Pessoa; Rud et al.; Dauth et al.) supports churn, temporary non-employment, and lower re-employment wages. A 15/85 point can be shown as a bound, but rent-sharing elasticities are not an identified aggregate split. |
| Non-employment duration | 3 months in the implemented organizing path | 0, 6 and 12 months | With current annual FRS, these are duration-equivalent earnings/hours stresses; no within-year benefit spell is simulated. Do not interpret them as observed spell durations. |
| Re-employment earnings penalty | 28.3% annual earnings | 14.0--28.3% | Current FRS destination calibration; 14% controls crudely for hours/age. Treat as descriptive, not causal. |
| Re-employment lag | 3 months | 0--6 months | A lag is needed to distinguish temporary unemployment from immediate sector switching; existing code supports this. |
| UC take-up among affected units | 0.80 | 0.55--0.95 (plus 1.0 statutory upper bound) | Stored FRS `would_claim_uc` is pre-shock and population-wide, not post-shock take-up. Redraw affected units, as current code does; report the interval as parameter sensitivity, not sampling uncertainty. |
| LCWRA among older inactive workers | 0.50 recommended for a future central inactivity case | 0--1.0 | WCA passage is not observed for the counterfactual. The implemented 1.0 case is an upper bound and is labelled as such in the paper. |
| Pension contributions | Hold fixed in headline | current HBAI treatment as sensitivity | Falling employee pension/salary-sacrifice payments otherwise appear as present-income cushioning. This is not tax-benefit insurance. |
| Household responses | Suppressed in headline | added-worker, self-employment, delayed-retirement accounting sensitivities | Irastorza-Fadrique--Levell--Parey documents these responses, but FRS/PolicyEngine does not identify their tariff-specific magnitudes. |

The central 70/30 split is a transparent prior, not an estimate. The paper
should display a response surface over the displacement share (0, .25, .50,
.70, .85, 1) and identify the 70/30 cell as the organizing calibration only if
the authors are comfortable defending that prior. If not, report the surface
without a single preferred cell and use the 50/50 cell for plots solely as a
midpoint.

## What the available data can and cannot identify

* FRS 2024--25 supplies household composition, incomes, benefits, employment
  status, SIC division, hours, pension contributions, and survey weights. It is
  suitable for the tax-benefit second stage and distributional outcomes.
* Longitudinal LFS has 177 manufacturing wave-1 donors in the present extract.
  The calibrated job-exit rate is 8.96%; this is an unconditional labour-market
  transition rate, not a tariff effect. The calibrated mean log wage change is
  +5.87%, so it must **not** be used as a trade-shock wage-loss estimate.
  Cell/QRF correlations are weak (about 0.06--0.08), so LFS heterogeneity is a
  sensitivity shape, not a primary incidence model.
* BRES provides rounded sector employment totals and can align sector mass, but
  not worker-level earnings or transitions.
* HMRC/ONS export data identify exposure weights and realised export falls. The
  destination-panel diagnostics should be presented as a failed causal first
  stage, not as evidence that tariffs generated the imposed worker losses.
* There is no linked employer-worker, ASHE, or firm-level production panel in
  the repository. Consequently, tariff-specific employment, wage, hours,
  re-employment, and take-up responses are not identified.

## Main-paper design

1. Use the 0.4 elasticity as the low/central trade calibration and show 1.0 and
   2.0 as transparent stress cases.
2. Make the mixed margin (temporary non-employment, re-employment penalty, and
   survivor earnings loss) the main scenario family; retain pure wage cut and
   pure displacement as bounds.
3. Show every result over the displacement-share grid and decompose tax, NI,
   UC/JSA, pensions, and other benefits.
4. Put the tariff calibration and all worker-selection models in a first-stage
   calibration section; state explicitly that PolicyEngine starts after the
   production-side response.
5. Treat assignment SDs and parameter ranges separately. They are not
   confidence intervals; survey-design and tariff-response uncertainty remain
   unestimated.

## Literature anchors already in `tariff_paper_lit_review.md`

The relevant UK evidence is De Lyon & Pessoa (worker earnings/employment), Rud
et al. (wage loss versus non-employment decomposition), Dauth et al. (service
reallocation with wage penalties), Beatty & Fothergill (benefit-financed UK
inactivity), and Irastorza-Fadrique, Levell & Parey (added-worker,
self-employment, and retirement responses). Dolls, Fuest & Peichl and Brewer &
Tasseva provide the automatic-stabiliser/EUROMOD and UKMOD stress-test
benchmarks. These papers discipline the scenario families; none identifies the
2025 tariff-to-worker mapping used here.
