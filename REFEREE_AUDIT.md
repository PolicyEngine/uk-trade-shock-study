# Referee and reproducibility audit: *Tariffs Abroad, Transfers at Home*

**Audit date:** 21 July 2026  
**Recommendation:** Major revision; not ready for external circulation in its current form.

> **Status note (23 July 2026):** This file preserves the original audit.
> Implemented remediations and deliberately out-of-scope empirical extensions
> are tracked in [`REVISION_STATUS.md`](REVISION_STATUS.md). Do not read the
> historical “remaining” language below as the state of the current branch.

## Remediation completed in this working revision

The findings below record the code and manuscript as first audited. This
working revision has since fixed the asymmetric UC take-up rule, LCWRA double
payment, worker-assignment sampler, duplicate mechanism constructor,
non-standard JSON `NaN` output, and mislabeled constituency-income units. It
regenerates the eight headline scenarios with 50 assignments, rebuilds all
dependent diagnostics and figures, and reconciles the manuscript to those
artifacts. The remaining major-revision items are research-design limitations:
the tariff first stage, wage-bill comparability across margins, survey and
parameter uncertainty, dynamic duration, prices, and general equilibrium.

## Technical summary

The project asks a valuable and unusually concrete question: how the UK tax-benefit system cushions a foreign tariff shock transmitted through workers' earnings. The codebase is more transparent than many applied microsimulation projects, and its deterministic random streams, transition read-back checks, decomposition files, and explicit scenario families are strong foundations.

The headline conclusions are not yet identified by the evidence, however. The central elasticity is calibrated from a single April 2025 month-on-month export fall that the paper elsewhere recognizes as contaminated by front-running. The main "reliability" comparison is partly imposed by code: displaced and reallocated households receive a fresh Universal Credit claim draw, while wage-cut households never do. The displacement sampler targets weighted headcount, not a matched wage-bill loss, so realized extensive- and intensive-margin shocks can differ substantially. Monte Carlo draw dispersion is then discussed as if it measured inferential uncertainty, although sampling, first-stage, parameter, survey, and model uncertainty are omitted.

The appropriate claim at this stage is therefore a **transparent static stress test of the direct labour-income channel**, not a causal estimate of the 2025 tariffs' household incidence or a complete welfare analysis. The project is publishable in concept, but the shock first stage, common treatment of benefit take-up, uncertainty framework, and replication contract need rebuilding.

## What can be retained

- The research question and comparison of adjustment margins are potentially original when stated narrowly: a UK statutory tax-benefit analysis of a partner-country tariff shock transmitted through labour income.
- PolicyEngine is a suitable engine for first-round tax-benefit accounting.
- The code separates random streams for displacement, reallocation destinations, LCWRA thinning, and UC take-up.
- Hard read-back checks reduce the risk that PolicyEngine silently rejects employment-status, industry, or claim-status inputs.
- Saved decompositions make the tax, National Insurance, UC, and residual channels auditable.
- The manuscript is generally candid about many limitations; several conclusions simply need to be brought back into line with those caveats.

## Findings requiring correction before circulation

### 1. UC take-up asymmetry mechanically supports the headline result — critical

`affected_mask` includes displacement, inactivity, and reallocation, but excludes wage cuts. If no listed flag is set, `redraw_uc_takeup` returns the baseline claim flag and explicitly declares wage-cut results invariant "by construction". See `uk_trade_shock_study/shocks.py`, lines 403–466. Consequently:

- a reallocated worker who remains employed receives a new claim propensity;
- a wage-cut household that becomes newly entitled, or whose award rises, retains its pre-shock flag;
- the abstract's contrast between "automatic" wage protection and "contingent" job-loss protection is partly a treatment-rule choice, not an empirical result.

**Required fix:** determine affected benefit units from the change in post-shock UC entitlement or award, not from margin labels. Apply common full take-up, no take-up, and evidence-based take-up scenarios to every margin. Separately model existing claimants, newly entitled units, and previously entitled non-claimants.

### 2. The central export elasticity is not causally identified — critical

The default elasticity uses roughly `24.7 / 12.8`, interpreting the April 2025 export fall relative to March as a tariff response (`uk_trade_shock_study/exposure.py`, lines 62–75). Yet the appendix excludes April from the measured family because it mixes the shock with the unwinding of first-quarter front-running. The same contaminated month cannot credibly identify the central structural parameter.

The calculation also combines values, prices, quantities, exchange rates, product composition, anticipation, global demand, and tariff treatment. A sensitivity grid does not repair a biased central estimate.

**Required fix:** make a literature-based prior or a product-by-destination causal estimate central. Use tariff changes at product-month level; distinguish quantities from unit values; test anticipation and pre-trends; compare US exports with the same products sent elsewhere and untariffed products sent to the US.

### 3. Extensive and intensive shocks are not reliably earnings-equivalent — high

The displacement routine consumes heterogeneous survey weights against a sector headcount quota (`shocks.py`, lines 167–200). It does not target earnings and does not prove equal first-order inclusion probabilities under unequal record weights. The wage-cut routine, by contrast, removes the exact sector shock times the weighted wage bill (`shocks.py`, lines 238–270).

The saved results demonstrate the practical problem: individual displacement draws can have gross losses far from the deterministic wage-cut loss. The paper's EPD example reports about £0.91bn versus £1.39bn, a difference of roughly 35%. Cushioning rates are composition-sensitive because the selected workers also differ in tax rate, family structure, and UC eligibility.

**Required fix:** either use a sampling design with prescribed inclusion probabilities and demonstrate wage-bill unbiasedness by sector, or calibrate each extensive-margin draw to a common gross loss. Report paired contrasts using identical seeds and draw counts.

### 4. Monte Carlo dispersion is not statistical uncertainty — high

The reported `± SD` measures variation across artificial worker assignments. It is not:

- uncertainty in the tariff response;
- FRS complex-survey sampling error;
- uncertainty in SIC/trade concordances or imputed industry;
- uncertainty in wage/employment pass-through;
- tax-benefit model or behavioural uncertainty.

For the simulated mean, numerical Monte Carlo error is closer to `SD / sqrt(n)`. Neither quantity is a confidence interval for the economic estimand.

**Required fix:** report, separately, assignment dispersion, Monte Carlo standard error, survey/bootstrap variance, first-stage coefficient uncertainty, and scenario/parameter uncertainty. Use paired differences for policy contrasts and avoid phrases such as "identified because within one SD."

### 5. The measured scenario is descriptive and one-sided — high

The measured family annualizes a ten-month export-value fall, sets expanding sectors to zero, attributes all declines to tariffs, and assumes full transmission to the wage bill. It therefore embeds global demand, exchange rates, sector trends, rerouting, prices, inventories, and other shocks, while clipping offsets from growing industries.

**Required fix:** label it an observed-outturn stress scenario, not a tariff-caused or "best" estimate. Report unclipped net and clipped downside versions. Do not annualize without duration alternatives, and separate values, prices, and quantities.

### 6. "Pass-through" conflates three distinct mechanisms — high

The parameter called pass-through is the share of an output/export-demand loss assigned to labour earnings. This is not the border-price pass-through studied by Amiti, Redding and Weinstein, Cavallo et al., or the USITC. The model currently compresses three bridges into one:

1. tariff to export price and quantity;
2. export quantity to domestic output/value added;
3. output/value added to wages, hours, employment, and profits.

**Required fix:** rename the current parameter `wage_bill_incidence` and model or bound all three bridges separately. Use domestic value-added content or ONS supply-use/TiVA information rather than treating customs exports divided by firm turnover as an immediate wage-bill share.

### 7. Existing LCWRA recipients can receive a duplicated element — high code defect

For newly flagged inactive workers, the code calculates the stored baseline benefit-unit LCWRA element and adds a full new annual element (`shocks.py`, lines 541–568). The helper caps the new add-on at one per benefit unit, but it does not cap `baseline element + add-on`. A benefit unit already receiving LCWRA can therefore exceed the statutory maximum.

**Required fix:** set the post-shock element to the maximum of the baseline element and one statutory element, or add an element only where none exists. Add regression tests for benefit units with an existing element and for two newly flagged members in one unit.

### 8. The inactivity scenario is an upper-bound accounting device, not a realistic transition — high

Every displaced worker aged 50 or older can immediately enter inactivity and receive LCWRA. The design omits health eligibility, the Work Capability Assessment, waiting periods, conditionality, New Style JSA, and dynamic claim histories. The paper acknowledges several of these, but still uses the scenario in substantive rankings.

**Required fix:** present it as a mechanical upper bound, or estimate transition and award probabilities from longitudinal/admin data. Include New Style JSA and time-to-award if the route remains central.

### 9. The reallocation margin is not an estimated reallocation process — high

The 28.3% penalty is a cross-sectional manufacturing-services earnings gap, not a causal mover penalty. Destinations are random among four service divisions, with no occupation, skill, region, tenure, or vacancy constraint. The destination SIC has no downstream fiscal effect in the current PolicyEngine calculation.

**Required fix:** estimate post-displacement earnings and employment hazards using longitudinal ASHE, PAYE, Understanding Society, or linked administrative evidence. Condition destinations on occupation, region, age, tenure, and prior earnings. Otherwise call this a stylized lower-loss earnings scenario.

### 10. The six-month unemployment sensitivity is internally incoherent — medium/high

The duration script restores half annual earnings but still assigns end-period unemployment and zero hours for the static annual tax-benefit calculation. That combines half-year wages with full-period unemployment status and can miscalculate UC and taxes.

**Required fix:** simulate monthly states or aggregate two internally coherent half-year simulations.

### 11. Network and local multipliers should not be multiplied mechanically — high

The paper combines a Leontief upstream output calculation with local employment multipliers and interprets the result as an all-channel GDP range. The components have different estimands, units, horizons, and possible overlap; local multipliers may already include supplier and induced effects.

**Required fix:** use one coherent IO/CGE/local-labour-market propagation model or stop at the clearly labelled direct effect. Remove the compounded GDP claim until that bridge exists.

### 12. Cash-income poverty is not complete household tariff incidence — high framing issue

The analysis holds consumer prices fixed. It therefore measures changes in nominal disposable income under a labour-income shock, not real income, consumption welfare, or total tariff incidence. Trade diversion could lower some UK import prices, while retaliation or fragmentation could raise others. Profit and pension channels also matter.

**Required fix:** either add a household expenditure-price module or state prominently that poverty and inequality results exclude cost-of-living effects. Avoid unqualified welfare language.

### 13. Several claims exceed the numerical evidence — medium/high

- "Poverty-certain" is inconsistent with an affected-household post-shock poverty rate of `38 ± 14%`; use "a large rise in poverty risk."
- "Regionally, not vertically, concentrated" is too strong when regional rankings are dominated by sparse FRS draws and constituency SIC is imputed. Use BRES/IDBR exposure and small-area calibration.
- Exact zero wage-cut poverty changes can be plausible for small dispersed cuts, but require threshold-crossing and poverty-gap validation.
- Small negative poverty changes in low-penalty reallocation scenarios should be presented as simulation/discrete-threshold noise unless robust.

### 14. Data vintage and provenance contradict each other — medium/high replication blocker

The README says the trade builder is a stub, the packaged intensity is a placeholder, and results should not be published. The paper and CSV say the data are real. The module docstring still says placeholder. The appendix says most ABS denominators are 2023, while the packaged CSV header says 2024 and the builder dynamically selects the latest available year.

**Required fix:** freeze a dated input manifest with URLs, retrieval timestamps, hashes, source vintages, exclusions, and per-division denominator years. Remove stale placeholder language only after the frozen build is reproduced.

### 15. The replication command is not currently one-step reproducible — medium

Plain `pytest -q` failed during collection with three `ModuleNotFoundError` errors in the default environment. The project requires Python 3.13, while that environment was Python 3.10. Running through the repository virtual environment/PYTHONPATH progressed, showing that this is partly an installation/environment issue, but there is no visible lockfile and the README's test command depends on completing editable installation first.

**Required fix:** provide a locked Python 3.13 environment, a single bootstrap/test command, test markers for expensive or data-dependent checks, and CI. Do not swallow every exception in `runner.py`'s absolute-poverty calculation; catch only a known missing-variable exception.

## Results plausibility and reconciliation

The headline magnitudes are arithmetically plausible as **model outputs**: a roughly £1.8bn gross earnings loss, £0.9–1.0bn Exchequer cost, and 39–41% cushioning can coexist because reduced income tax, National Insurance, and UC offset household losses. The saved decomposition is internally additive at seed 0. The near-zero national poverty effect is also possible because exposure is small nationally and concentrated above the bottom deciles.

Those magnitudes are not yet credible causal estimates because their first stage and labour-incidence bridge are assumed. Specific interpretation rules should be:

- `39% versus 41%` is a scenario comparison, not evidence that the margins are economically equally insured.
- `£0.92bn versus £0.99bn` is within assignment variability but that does not establish statistical equality.
- the rise from 4% to 38% among affected households is conditional on simulated selection and full-year job loss; it is not a population treatment effect;
- supply-chain amplification near 1.95 is an IO scenario, not validation of the direct model and not safely composable with local multipliers;
- nationally small poverty and Gini changes do not imply that the total tariff shock is mainly fiscal because prices, profits, partners' labour supply, investment, and general equilibrium are outside scope.

## Literature alignment

The paper aligns well with the automatic-stabilizer and worker-adjustment literatures, but its novelty language should be narrower. The closest comparators show that the intersection of trade shocks and household incidence is not empty:

- [Fajgelbaum and Khandelwal (2016)](https://www.nber.org/papers/w20331) connect trade prices, non-homothetic expenditure, and household welfare.
- [Borusyak and Jaravel (2021)](https://www.nber.org/papers/w28957) jointly decompose expenditure and earnings exposure.
- The [World Bank Household Impacts of Tariffs tool](https://www.worldbank.org/en/research/brief/hit) combines household consumption and income/production channels.
- The [OECD METRO-to-household-budget method](https://www.oecd.org/en/publications/mapping-trade-to-household-budget-survey_5fc6181b-en.html) maps CGE price changes through concordances to household compensating variation.
- [Amiti, Redding and Weinstein (2019)](https://pubs.aeaweb.org/doi/10.1257/jep.33.4.187), [Cavallo et al. (2021)](https://www.aeaweb.org/doi/10.1257/aeri.20190536), and the [USITC retrospective](https://www.usitc.gov/sites/default/files/publications/332/pub5405.pdf) distinguish tariff incidence at the border, exporter margins, retail prices, downstream users, and production.
- IFS work on [household food-tariff exposure](https://ifs.org.uk/publications/exposure-households-food-spending-tariff-changes-and-exchange-rate-movements) and [tariff abolition and consumer prices](https://ifs.org.uk/publications/customs-union-tariff-reductions-and-consumer-prices), together with [Breinlich et al. on Brexit prices](https://cep.lse.ac.uk/pubs/download/dp1667.pdf), provides UK household-price benchmarks.
- The [DBT trade-modelling expert review](https://www.gov.uk/government/publications/trade-modelling-review-expert-panel-report-and-recommendations/trade-modelling-review-expert-panel-report) recommends empirically grounded CGE cores, sector extensions, parameter ranges, ex-post validation, and reality checks.
- The [OBR March 2025 tariff scenarios](https://obr.uk/efo/economic-and-fiscal-outlook-march-2025/) use an export-demand elasticity of about 0.4 in a stylized 20-point tariff scenario, far below this paper's central 2.0 and therefore an important sensitivity benchmark.
- Bank of England firm evidence found [heterogeneous and generally limited direct effects, with greater exposure among US exporters](https://www.bankofengland.co.uk/agents-summary/2025/2025-q2/latest-results-from-the-decision-maker-panel-survey-2025-q2), supporting firm-level rather than uniform SIC shocks.
- Recent UK work on [household responses to trade shocks](https://ifs.org.uk/journals/household-responses-trade-shocks) emphasizes added-worker, retirement, and self-employment responses that the static model omits.

Replace claims such as "the intersection itself is empty" with: **to our knowledge, this is the first UK statutory tax-benefit stress test of the labour-income channel from the 2025 partner-country tariff shock that explicitly compares earnings-equivalent extensive and intensive adjustment margins.** Even that claim should be rechecked immediately before submission.

## How major research shops would strengthen the design

| Institution/style | Typical strength | Missing here |
|---|---|---|
| DBT / Armington-CGE | coherent trade, production, diversion, and sector equilibrium | causal/structural trade first stage and consistent closure |
| OECD METRO-HBS | sector prices mapped to household expenditure and welfare | consumer-price and real-income channel |
| World Bank HIT | transparent first-order consumer plus producer-income incidence | joint household incidence and adjustable pass-through stages |
| USITC retrospective | tariff-line econometrics for prices, imports, output, and downstream users | product-level ex-post identification and validation |
| OBR | macro scenario envelopes with explicit policy closures | consistent relation to macro assumptions and uncertainty |
| Bank of England | firm heterogeneity, trade uncertainty, monetary and price channels | firm-level exposure and general-equilibrium interpretation |

## A better project from scratch

### 1. Pre-specify separate estimands

Define the causal tariff effect on export quantity and price; the effect on firm payroll, employment, hours, and separations; household disposable-income and poverty effects; the EPD treatment effect; and one-year versus dynamic horizons. Do not let one scenario output stand in for all of them.

### 2. Build a tariff-line policy database

Record product-level rates by date, exemptions, quota treatment, country of origin, legal effective dates, pauses, and uncertainty. Freeze source snapshots and hashes. Treat autos, steel, and pharmaceuticals with their actual tariff-line/quota exposure rather than division-wide rates.

### 3. Estimate the trade first stage

Use monthly or transaction-level HMRC product-by-destination data from at least 2022–2026. Estimate an event study or triple difference using:

- changes in US tariff treatment by product;
- the same products exported to non-US destinations;
- untariffed products exported to the US;
- product, destination, calendar-time, and possibly firm fixed effects.

Test pre-trends and anticipation. Estimate quantities and unit values separately. Measure diversion rather than clipping export gains.

### 4. Estimate labour incidence rather than assuming it

Link trader identifiers to IDBR/BRES/PAYE/ASHE if access permits. Estimate payroll, employment, hours, hiring, separation, and wage responses by exposure, with firm and time controls. Model heterogeneity by occupation, age, sex, tenure, contract, region, and initial earnings. If linkage is unavailable, use disciplined literature priors and present wide scenario envelopes.

### 5. Propagate through one coherent network model

Use ONS supply-use/input-output accounts to map domestic value added and upstream effects. Either stop at partial equilibrium or use an Armington/CGE structure for substitution, diversion, imports, exports, prices, and factor returns. Do not multiply unrelated IO and local multipliers.

### 6. Map workers to households dynamically

Use linked admin data where possible; otherwise statistically match/reweight FRS and validate against Understanding Society. Simulate partner labour supply, added-worker responses, retirement/inactivity, self-employment, re-employment hazards, and mover earnings paths.

### 7. Apply a common tax-benefit protocol

Feed estimated transition distributions into PolicyEngine or UKMOD. Apply identical entitlement and claim rules across margins. Include UC take-up conditional on entitlement/award change, New Style JSA, capital where data permit, WCA eligibility and timing, and monthly state histories.

### 8. Add household consumer prices

Map sector/product price changes through COICOP to LCF household expenditure baskets. Report nominal disposable-income effects and real-income/compensating-variation effects separately. Include indirect taxes where relevant.

### 9. Propagate uncertainty honestly

Draw jointly from first-stage coefficients, labour-response estimates, pass-through parameters, concordance alternatives, survey replicate/bootstrap weights, and transition simulation. Report assignment dispersion and numerical Monte Carlo error separately. Use clustered/bootstrap inference and paired policy contrasts.

### 10. Validate out of sample

Hold out late-2025/2026 trade and payroll observations. Compare predicted sector rankings and magnitudes with ONS/HMRC outturns, BICS/DMP, SMMT, and administrative employment data. Publish pre-trends, placebo dates, coverage tables, crosswalk uncertainty, and calibration residuals.

## Minimum viable revision if the full rebuild is infeasible

1. Reframe the paper as a static stress test and remove causal/current-best-estimate language.
2. Replace the April calibration with literature-based low/central/high priors, including the OBR's lower elasticity benchmark.
3. Treat UC take-up symmetrically for all margins.
4. Fix LCWRA duplication and the duration scenario.
5. Calibrate every paired draw to the same gross wage-bill loss.
6. Report paired effects, Monte Carlo error, survey uncertainty, and parameter sensitivity separately.
7. Drop compounded multiplier/GDP claims and narrow regional claims.
8. Add a clear nominal-income-only caveat or a price-incidence appendix.
9. Freeze the data manifest, resolve 2023/2024 denominator inconsistencies, add a lockfile/CI, and make one command reproduce tests, tables, figures, and paper.
10. Rewrite the abstract only after regenerated results are stable.

## Verification performed

- Inspected all paper sections, the compiled 54-page PDF, package code, analysis scripts, tests, packaged CSVs, and saved JSON outputs.
- Confirmed all saved JSON files parse successfully.
- The LaTeX log showed no undefined references or citations in the inspected build; only non-substantive underline/underbar warnings appeared.
- Plain `pytest -q` failed collection with three package import errors in the default environment. Tests progressed under the repository environment/PYTHONPATH, but this audit does not certify a clean end-to-end rerun of licensed-data simulations.
- The licensed FRS source data and every external raw-data retrieval were not independently reacquired during this audit. Results are therefore checked for internal logic and saved-output consistency, not fully reproduced from raw sources.

## Overall assessment

**Needs revision.** The paper has a strong question, transparent code, and a potentially useful tax-benefit contribution. But the central empirical shock, the extensive/intensive comparison, and the uncertainty interpretation are not yet strong enough for a referee to treat the numerical results as research estimates. Fixing the code defects without redesigning the first stage would produce a cleaner stress test; a causal paper requires the product/destination and firm/labour design described above.
