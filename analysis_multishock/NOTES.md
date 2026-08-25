# Notes and assumptions — multi-shock household-incidence pipeline

Date run: 19 August 2026. One end-to-end script (`incidence.py`); all inputs
public, every raw file pinned by SHA256 (verified at each run), every reported
number generated into `out/results.json` and `out/generated_multishock.tex`
(`\newcommand` macros prefixed `Ms`, no digits in macro names). Declared first
stages are cited published estimates and are **not** re-estimated here.

## 1. Pinned raw inputs (`raw/`)

| File | Source | SHA256 (first/last 8) |
|---|---|---|
| `family_spending_wb1_fye2025.xlsx` | ONS Family Spending workbook 1, FYE2025 edition: `https://www.ons.gov.uk/file?uri=/peoplepopulationandcommunity/personalandhouseholdfinances/expenditure/datasets/familyspendingworkbook1detailedexpenditureandtrends/fye2025/workbook1detailedexpenditureandtrends.xlsx` | `af8d4a44…d8931076` |
| `family_spending_wb1_fye2022.xlsx` | Same dataset page, FYE2022 edition (pre-energy-shock spending vintage, Apr 2021–Mar 2022): `…/fye2022/workbook1detailedexpenditureandtrends.xlsx` | `549336c3…cb9a2e17` |
| `cpi_d7bt.csv` | ONS CPI all-items index (2015=100), series D7BT, dataset MM23: `https://www.ons.gov.uk/generator?format=csv&uri=/economy/inflationandpriceindices/timeseries/d7bt/mm23` (vintage released 19 Aug 2026) | `164f5f89…18e223e` |
| `dwp_benefit_rates_2022_23.pdf` | DWP "Benefit and pension rates 2022/2023": `https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/1036433/benefit-and-pension-rates-2022-2023.pdf` (carries the 2021/22 and 2022/23 columns) | `41559e15…6564f722` |
| `govuk_benefit_rates_2023_24.html` | gov.uk "Benefit and pension rates 2023 to 2024": `https://www.gov.uk/government/publications/benefit-and-pension-rates-2023-to-2024/benefit-and-pension-rates-2023-to-2024` (carries the 2022/23 and 2023/24 columns) | `57afb190…9446d733` |

The script additionally *verifies content*, not just hashes, for the statutory
UC rates: it greps the gov.uk page for `334.91`/`368.74` and the decompressed
PDF content streams for `324.84`/`334.91`, so the hard-coded chain
£324.84 → £334.91 → £368.74 is checked against pinned primary sources on
every run.

## 2. Household spending base (both vintages)

- Sheet **3.1E**: detailed household expenditure (£/week per household) by
  **equivalised disposable income decile group**, columns 5–14 = deciles
  lowest→highest, column 15 = all households. Rows are located by COICOP
  code/label (never fixed row index) because the layout shifts across
  editions.
- No suppressed (`..`) cells occur in any row this pipeline uses (all are
  COICOP aggregates); the parser would fail loudly if one appeared.
- Weighted households: FYE2022 2,850k per decile (28,500k total); FYE2025
  2,871k per decile (28,710k). Aggregates use each vintage's own counts.
- LCF spending under-records at the top of the distribution, so aggregate
  grossings are conservative relative to national accounts (same caveat as
  the sibling retaliation prototype).

## 3. Income denominators (ASSUMPTION block)

- "% of disposable income" = household-level weekly cost ÷ decile **mean
  equivalised** disposable income (£/wk). Mixing household costs with
  equivalised income overstates the ratio for multi-person households —
  flagged, same convention as the retaliation prototype.
- FYE2025 3.1E publishes decile means directly (used for E4).
- **FYE2022 3.1E publishes only decile lower boundaries.** Constructed decile
  means (`meta.fye2022_constructed_decile_income_gbp_wk`): interior deciles
  2–9 = boundary midpoints (validated on FYE2025, where midpoints match
  published means to within 1.84%); open-ended deciles 1 and 10 use FYE2025
  shape ratios mean₁/b₂ and mean₁₀/b₁₀. These ratios are invariant to a
  uniform equivalence-scale rebasing, which matters because…
- …ONS moved from the OECD-modified scale (FYE2022) to the *updated*
  OECD-modified scale (FYE2025): equivalised income *levels* are not
  comparable across the two vintages (boundaries jump far more than nominal
  income growth). Consequence: E1/E2 "% of income" (FYE2022 denominators) and
  E4 "% of income" (FYE2025 denominators) are internally consistent but not
  strictly comparable in level across episodes. "% of total spending" is
  clean everywhere and is the primary comparability metric (Module 3 uses it
  exclusively).

## 4. E1 — TCA food NTBs (declared first stage)

- Declared estimate: Brexit NTBs raised food & non-alcoholic beverage prices
  by **8%** (central; 6% sensitivity) cumulatively over Dec 2019–Mar 2023
  (~3.25 years). Source: Bakker, Datta, Davies & De Lyon, *CEP Brexit
  Analysis 18* (2023). Pass-through is embedded in the estimate — no dial.
- Spend base: FYE2022 COICOP 1 (£62.20/wk all households) — the closest
  downloadable vintage inside the window. ASSUMPTION: nominal FYE2022 spend
  proxies the window-average food-spend base.
- End-state annual cost per decile = spend × 8% × 52 (this is the "per-year"
  incidence used in Module 3).
- Cumulative per household ASSUMES a **linear phase-in** from 0 to 8% over
  the window (average effect = half of end-state): cumulative =
  end-state × 3.25 × 0.5.
- **Cross-check vs published £250/household cumulative**: our linear-ramp
  cumulative is £420 (ratio 1.68). The published figure implies an average
  effect ≈ 30% of end-state (`implied_average_effect_share` = 0.297), i.e.
  the estimated price path was heavily back-loaded into 2022–23, and CEP's
  spend base predates most of the 2022–23 food inflation. Reported as a
  cross-check diagnostic, not used to calibrate anything.

## 5. E2 — Energy 2022–23 (declared price vector)

- Declared cap levels (Ofgem, typical dual-fuel annual bill): winter 2021-22
  cap **£1,277**; Oct 2022 cap **£3,549**; Energy Price Guarantee **£2,500**.
- Price factors applied to spending, on FINANCIAL-YEAR mean caps (the
  point-to-point convention below is superseded, kept only as a labelled
  variant): base (1,138+1,277)/2 = 1,208; counterfactual
  (1,971+3,549)/2 = 2,760; realised (1,971+2,500)/2 = 2,236. Gross =
  **+128.6%**; net-of-EPG = **+85.1%**.
- Spend base: FYE2022 COICOP 4.4.1 electricity + 4.4.2 gas (other fuels
  4.4.3 excluded — not covered by the cap). FYE2022 (Apr 2021–Mar 2022) is
  the pre-shock vintage the task prescribes; no deflation needed. ASSUMPTION:
  that vintage's spend reflects the £1,277 base cap. In fact it straddles the
  £1,138 (summer 2021) and £1,277 (winter 2021-22) caps, so the base is
  slightly *under* £1,277-consistent levels → the £ shock is slightly
  understated, symmetrically in gross and net.
- ASSUMPTION: fixed quantities (no demand response), 100% pass-through of cap
  ratios to bills — an upper-bound price-incidence convention.
- EPG **discretionary cushioning share** on the FY basis =
  (2,760−2,236)/(2,760−1,208) = **33.8%**, constant across deciles by
  construction; the point-to-point cap ratio (3,549−2,500)/(3,549−1,277)
  = 46.2% is retained as a labelled variant and not used further.
- Cross-check (context anchor, not calibration): our 6-month aggregate
  cushion ≈ £7.8bn (FY-mean basis) vs OBR's £27bn EPG costing. Lower because OBR priced the
  EPG against the higher caps then expected for Jan–Jun 2023 (~£4,300+) over
  a longer window; same order of magnitude. £47bn = OBR total energy price
  subsidies anchor, also recorded.

## 6. E3 — US tariffs 2025 (zero consumer row + declared imports)

- The UK imposed no retaliatory tariffs, so the UK **consumer-price first
  stage is literally zero at every decile** (explicit zero row).
- Earnings side imported unchanged from the companion uk-trade-shock-study as
  declared constants: gross shock **£886m/yr**, displacement cushioning
  **36.6%**, wage-cut cushioning **43.9%** (macros `\MsUSTariff…`).

## 7. E4 — India CETA consumer duty reductions (declared first stage)

- Declared: DBT impact assessment Annex 9 Table 18, consumer duty reductions
  on final goods £180m/yr: textiles/clothing **£132.3m**, footwear
  **£13.2m**, food/beverages **£11.7m**.
- Category mapping ASSUMPTION: textiles/clothing → COICOP 3.1 Clothing
  (household textiles 5.2 excluded — understates the allocation breadth for
  the textiles sliver); footwear → 3.2; food/beverages → COICOP 1.
- Allocation: each category's £ gain distributed across deciles in proportion
  to that category's decile spending (FYE2025 3.1E), equal weighted
  households per decile; per-household gain = decile allocation ÷ 2,871k.
- Pass-through dial: 50% / 75% / 100% scaling of the duty saving reaching
  consumer prices.
- Progressivity verdict uses gain as % of total spending, bottom vs top
  decile, with a ±20% ratio tolerance band: bottom 0.0151% vs top 0.0154% →
  **roughly proportional** (ratio 0.98) — and economically negligible
  (£3–£10 per household per year at 100% pass-through).

## 8. E5 — CPTPP near-zero benchmark

- Declared: DBT central +£2.0bn GDP long-run (≈ +0.08%). Naive mean =
  £2.0bn / 28.4m households = **£70.4/household/yr**. Stated as a near-zero benchmark, not a near-zero benchmark test:
  a long-run GDP estimate is not a household price shock; no distributional
  structure is claimed.

## 9. Module 2 — UC uprating-lag rulebook arithmetic

- Statutory chain (single adult 25+, £/month): Apr 2021 **£324.84** →
  Oct 2021 removal of the £20/wk uplift (£86.67/month, **£1,040/yr**,
  reported as a discrete pre-shock cut) → Apr 2022 **£334.91** (+3.1%, the
  lagged Sep-2021 CPI) → Apr 2023 **£368.74** (+10.1%). Verified against the
  pinned DWP PDF and gov.uk page (see §1).
- Counterfactual "contemporaneous-CPI indexation" ASSUMPTION: the allowance
  maintains its April-2021 real value month by month —
  cf(m) = 324.84 × D7BT(m)/D7BT(Apr 2021). Shortfall(m) = cf(m) − 334.91 for
  each month Apr 2022–Mar 2023, summed.
- Result: **£393/yr shortfall ≈ 9.8% of the annual allowance** (monthly
  shortfall grows £19.14 → £45.40 across the year).
- Flat alternative (also reported): uprate once in Apr 2022 by the April-2022
  y/y CPI (9.0%) instead of the lagged 3.1% → £230/yr. The difference from
  the central number is within-year erosion.
- Representative-household arithmetic only (no microdata, no caseload
  weighting); this is the "rulebook vintage" seed result.

## 10. Module 3 — comparability conventions

- One row per episode: gross shock (labelled £bn/yr or £bn), sign, channel,
  bottom/top-decile burden or gain as % of total spending, verdict,
  cushioning where computable.
- Verdict thresholds: bottom/top ratio > 1.2 regressive (costs) /
  progressive (gains); < 0.8 the reverse; else roughly proportional.
- Normalised column: bottom-decile % of spending per £1bn of gross shock, so
  episodes of wildly different size are comparable (E1 0.162, E2 0.194,
  E4 0.084 %-points per £bn; E3 n/a — consumer side zero; E5 n/a — near-zero benchmark).
- E3's size is the *earnings-side* gross (£0.886bn/yr, companion study);
  its consumer row is zero, so no consumer normalisation is offered.

## 11. Determinism

Two consecutive runs produce byte-identical `out/results.json` and
`out/generated_multishock.tex` (`run1.sha` = `run2.sha`, recorded in the
pipeline root).

## Known gaps for a licensed-data / second-pass version

- LCFS microdata: within-decile heterogeneity, household types, regions,
  proper income denominators (unequivalised household income by decile).
- E1: replace the linear phase-in with the published monthly Brexit-NTB
  price path; deflate the spend base to Dec-2019 prices.
- E2: prepayment/standard-credit tariff differences, EBSS £400 rebate and
  other support (only EPG modelled), demand response.
- E4: DBT's own consumer-surplus incidence, VAT interactions, household
  textiles in COICOP 5.2.
- Module 2: full benefit-unit modelling (housing element, LHA freeze,
  deductions), caseload-weighted aggregate cost of the uprating lag.


## 9. Second stage (added in the referee rounds)

Scripts: `second_stage_energy.py` (energy episode through PolicyEngine UK
2.89.2; emits `out/generated_secondstage.tex`), `grid_energy.py`
(rebase x stack sensitivity grid, fine sweep, announced-path variant),
`fig_extra.py` (decomposition/paths figures, 999-rep household
bootstrap), `vintage_and_tca.py` (two-vintage rulebook run with all
input columns copied to 2022; TCA food second stage), `mpc_welfare.py`
(MPC-weighted demand, Atkinson welfare). All write generated `.tex` to
`out/` and copy to `../paper_multishock/`; paper/pipeline sync is
verifiable by diff.

Input dataset: `enhanced_frs_2023_24.h5` (policyengine-uk-data release,
2023 vintage; 53,508 household records; person weights 69.8m).
SHA256 prefix: `584ae33d80ca0431`. The QRF consumption imputation is upstream in
policyengine-uk-data; its provenance is outside this project and its
error is a declared limitation of the within-decile results.

Conventions shared with the paper: FY-mean cap basis; energy numerator
rebased to the FY2021-22 cap mean (0.606); expenditure denominator
rebased to the ONS FYE2022 mean total spend (GBP 27,503/yr); food
rebased to the ONS FYE2022 food base. Rates computed on unrounded
arrays.
