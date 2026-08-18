# TAA costing pipeline — sources and assumptions

First-pass annual costings of four candidate UK worker-side trade-adjustment
policies, plus four added analysis modules (wage-insurance parameter grid,
ISERBS cell, cyclical stress, self-financing break-even, demographic
composition — sections 9–12). One rerunnable script (`costing.py`,
dependencies: `openpyxl` + `xlrd`), mirroring the conventions of
[PolicyEngine/uk-trade-shock-study](https://github.com/PolicyEngine/uk-trade-shock-study):
pinned raw inputs verified by SHA256, all reported numbers generated (never
typed) into LaTeX `\newcommand` macros prefixed `Taa`.

Run: `.venv/bin/python costing.py` → `out/results.json`, `out/generated_taa.tex`.

All downloads made 18 August 2026.

## 1. Pinned raw inputs (`data/raw/`)

| File | Source URL | SHA256 |
|---|---|---|
| `red02aug2026.xlsx` | ONS RED02 "Redundancies by industry, age, sex and re-employment rates", August 2026 edition (published 18 Aug 2026): `https://www.ons.gov.uk/file?uri=/employmentandlabourmarket/peoplenotinwork/redundancies/datasets/redundanciesbyindustryagesexandreemploymentratesred02/current/red02aug2026.xlsx` | `f653c80c…93feb3` |
| `ashetable42025provisional.zip` (unzipped to `ashe_table4/`) | ONS ASHE Table 4 (industry by 2-digit SIC), 2025 provisional edition (released 23 Oct 2025): `https://www.ons.gov.uk/file?uri=/employmentandlabourmarket/peopleinwork/earningsandworkinghours/datasets/industry2digitsicashetable4/2025provisional/ashetable42025provisional.zip` | `581d23bb…cb1284` |
| `benefit_pension_rates_2026_27.pdf` | DWP "Proposed benefit and pension rates 2026/2027": `https://assets.publishing.service.gov.uk/media/69931706ceeaa48d377f6bd5/Benefit-and-pension-rates-2026-2027.pdf` | `c13111a6…eaec3c019` |
| `pathways_to_work_green_paper.html` | DWP "Pathways to Work: Reforming Benefits and Support to Get Britain Working" Green Paper (Mar 2025), gov.uk consultation page | `0af072c1…27f5a073` |
| `port_talbot_22m_press_release.html` | gov.uk press release "UK government provides £22m extra support for Port Talbot steelworkers and businesses" | `cb845995…3216bb581fd9` |
| `nao_british_steel_press_release.html` | NAO press release "Government spends £377 million in 9 months to keep British Steel's Scunthorpe site operating" (Mar 2026) | `2f0f0249…30668345e8` |
| `red01nsaaug2026.xls` | ONS RED01 (NSA) "Redundancies levels and rates (not seasonally adjusted)", August 2026 edition, people/men/women, monthly-rolling windows from Mar–May 1995: `https://www.ons.gov.uk/file?uri=/employmentandlabourmarket/peoplenotinwork/redundancies/datasets/redundancieslevelsandratesnotseasonallyadjustedred01nsa/current/red01nsaaug2026.xls` (downloaded 18 Aug 2026; legacy `.xls`, hence the added `xlrd` dependency) | `79a52829…22e75c2d` |

Full hashes are hard-coded in `costing.py` (`PINNED`) and verified on every run.

## 2. Redundancy flows (eligible-flow counts)

- Source: RED02, sheet `Industry - levels` (LFS, not seasonally adjusted,
  persons). Column `A-U` = all industries; column `C` = Manufacturing.
- RED02 publishes overlapping rolling 3-month windows. Annual flow = sum of
  the latest four **non-overlapping** quarters: Jul–Sep 2025, Oct–Dec 2025,
  Jan–Mar 2026, Apr–Jun 2026.
- Computed: **manufacturing 64,463/yr; all-industry 511,407/yr**
  (year Jul 2025–Jun 2026). Quarterly detail is in `results.json` and as
  `\TaaRedManuf…`/`\TaaRedAll…` macros.
- CAVEAT: LFS redundancy levels by industry are volatile at this cell size
  (ONS marks small cells as low-precision). Apr–Jun 2026 manufacturing
  (3,933) is far below the preceding three quarters (17.8k–23.2k); the
  manufacturing annual flow is therefore probably at the low end of its
  sampling range.

## 3. Earnings

- Source: ASHE 2025 provisional, Table 4.1a "Weekly pay – Gross", sheet
  `Full-Time` (full-time employees on adult rates, pay unaffected by
  absence, £/week).
- Whole economy ("ALL EMPLOYEES"): median £766.60, mean £920.80, lower
  quartile (p25) £583.50.
- Manufacturing (SIC C, "MANUFACTURING"): median £774.00, mean £891.20,
  lower quartile £594.10.
- ASSUMPTION: 2025 provisional earnings are used un-uprated as the
  2026-27 pre-displacement earnings base (no wage-growth uplift applied).

## 4. Statutory / policy parameters

- **New Style JSA, 25+**: £95.55/wk in 2026-27 (up from £92.05 in 2025-26;
  3.8% CPI uprating), statutory maximum 26 weeks. Source: DWP proposed
  benefit rates 2026/2027 PDF (pinned), section "JOBSEEKER'S ALLOWANCE →
  Contribution based JSA".
- **UC standard allowance, single 25+**: £424.90/month 2026-27 (up from
  £400.14; CPI 3.8% + 2.3% Universal Credit Act 2025 uplift). Same PDF,
  "UNIVERSAL CREDIT (monthly rates)".
- **Proposed "Unemployment Insurance" (UI)**: the Pathways to Work Green
  Paper (pinned HTML, paras 151–156) proposes replacing NS JSA and NS ESA
  with a single time-limited, non-means-tested UI paid "at the current
  higher rate of NS ESA" — £140.55/wk at 2025-26 rates. For consistency
  with 2026-27 costing we use the 2026-27 uprated equivalent:
  NS ESA personal allowance £95.55 + support component £50.35 =
  **£145.90/wk** (both from the pinned rates PDF).
- **UI duration**: the Green Paper consults on duration, "for example
  6–12 months". We model 6 / 9 / 12 months as low / central / high
  (9-month central is an ASSUMPTION within the consulted range).

## 5. Imported calibrations and labelled assumptions

- **Re-employment wage penalty**: low 15%, high 28% — imported from the
  companion study's FRS calibration (28.3% all-employee raw penalty,
  14.0% hours-controlled), NOT re-estimated here. Central 21.5% is the
  simple midpoint (ASSUMPTION).
- **Re-employment within a year**: 50 / 70 / 85% (low/central/high).
  ASSUMPTION bracketing UK displacement literature (UK studies typically
  find roughly half to four-fifths of displaced workers re-employed within
  a year, higher in tight labour markets).
- **Wage-insurance steady state**: duration factor 2.0 — two annual cohorts
  in payment simultaneously, no wage catch-up or attrition over the 2-year
  entitlement (ASSUMPTION; biases arm A up).
- **UI contribution-condition share**: 80 / 90 / 100% of displaced workers
  satisfy NI contribution conditions (ASSUMPTION). Arm B also assumes full
  take-up of the incremental entitlement and ignores means-tested (UC)
  offsets — both bias arm B up.
- **Retraining stipend**: flat £3,000 per trainee (ASSUMPTION); take-up
  20 / 40 / 60% — pure assumptions bracketing US TAA training take-up
  experience (historically roughly a quarter to a half of TAA-certified
  workers entered training).
- **Trade-exposed manufacturing memo**: 30% of manufacturing redundancies
  (ASSUMPTION; intended to be replaced by the companion study's
  exposed-division employment share — the public README does not report
  that share, so 30% is a placeholder anchor).
- **Arm D (UC redundancy-pay capital disregard)**: NOT credibly costable
  from public data. Implemented only as a MAXIMUM exposure bound:
  flow × 6 months of the UC single-25+ standard allowance (£424.90 × 6 =
  £2,549.40) × share plausibly capital-gated 20 / 35 / 50% — ALL
  SPECULATIVE. A real estimate needs the FRS microsimulation (redundancy
  pay distribution vs the £6k/£16k capital limits, household incomes,
  housing elements). Treat as an upper bound, not a costing.

## 6. Cross-check anchor

Ad-hoc steel support ≈ **£499m**: Port Talbot Transition Board £122m
(£102m UK Government + £20m Tata Steel; pinned gov.uk press release) +
British Steel Scunthorpe £377m (Apr 2025–Jan 2026; pinned NAO press
release, which projects £615m by Jun 2026 — the £377m is therefore a
lower bound of the Scunthorpe intervention).

## 7. Scenario convention

"Low/central/high" bundles all scenario dials in the cost-increasing
direction together (e.g. arm A low = 50% re-employment AND 15% penalty).
The brackets are therefore wide by construction — they are outer bounds,
not confidence intervals.

## 8. Known limitations

1. LFS redundancy counts miss displacements not reported as redundancy
   (end of temporary contracts, some insolvencies) and are sampling-noisy
   by industry.
2. Steady-state costing: no phase-in, no behavioural response (wage
   insurance may change search behaviour; UI extension may lengthen
   unemployment duration), no deadweight (payments to workers who would
   have re-employed at similar wages anyway).
3. Gross fiscal cost only: no clawback via income tax/NICs on wage
   insurance payments, no netting against UC spending displaced by UI.
4. ASHE full-time weekly earnings applied to all displaced workers
   (part-timers, who earn less, are ignored; biases costs up slightly).
5. Arm interactions ignored — the four arms are costed independently and
   the "TOTAL" simply adds them.

## 9. Wage-insurance parameter grid and ISERBS cell (added module)

- Factorial grid, **narrow** eligibility (manufacturing flow 64,463/yr, ASHE
  manufacturing mean £891.20/wk), central re-employment 70%:
  replacement rate {30, 50, 70%} × maximum duration {1, 2, 3 years} ×
  penalty {15, 21.5, 28%}. Cost formula identical to Arm A:
  flow × 0.70 × R × penalty × £891.20 × 52 × D (steady state = D annual
  cohorts in payment simultaneously, no catch-up/attrition — same
  up-biasing ASSUMPTION as Arm A).
- The 3×3 table at the central 21.5% penalty is emitted as
  `TaaGridR30D1` … `TaaGridR70D3` (£m/yr); `\TaaGridMin`/`\TaaGridMax`
  are the min/max over all 27 cells (min = 30%×1yr×15%; max = 70%×3yr×28%).
- NOTE on macro names: LaTeX control words cannot contain digits, so the
  nine grid macros are defined with
  `\expandafter\newcommand\csname TaaGridR30D1\endcsname{…}` and must be
  used as `\csname TaaGridR30D1\endcsname`. All other added macros are
  ordinary digit-free `\newcommand`s.
- **ISERBS-parameterised cell**: the 1995 civil-service Insurance Scheme
  for Early Retirement and Severance-style design — payment = max(0, 90%
  of previous earnings − new earnings), i.e. a top-up to 90% of prior pay,
  for 78 weeks (1.5 years), same flow and 70% re-employment. At the
  central 21.5% penalty the implied top-up rate is 90 − 78.5 = **11.5%**
  of prior pay. Arithmetic relation worth noting: this is close to Arm A's
  50% × 21.5% = **10.75%** of prior pay — the ISERBS design at these
  parameters is nearly the same transfer per week as Arm A, paid for 1.5
  rather than 2 years (hence `\TaaIserbsCost` £361m/yr ≈ ¾ × Arm A's
  £450m/yr × 11.5/10.75). Unlike Arm A, the ISERBS payment is earnings-
  linked ex post (max(0, ·) truncation); at the central-penalty mean-wage
  calculation the truncation never binds, so the formula is linear.

## 10. Cyclical/historical stress (added module)

- Annual series convention: calendar-year flow = sum of the four
  non-overlapping quarters Jan–Mar + Apr–Jun + Jul–Sep + Oct–Dec (mirrors
  section 2). Two sources:
  - **All-industry (and men/women) back to 1995**: pinned RED01 NSA
    (`All` sheet). Numeric data run Mar–May 1995 → Apr–Jun 2026; complete
    calendar years are **1996–2025** (1995 lacks Jan–Mar and is excluded
    from the median). Some period labels carry a trailing footnote digit
    (e.g. `Jan-Mar 20193` = Jan–Mar 2019 footnote 3); the parser strips it.
  - **Manufacturing**: RED02 `Industry - levels`, available from calendar
    2009 only.
- Stress flows (manufacturing, /yr):
  - **GFC peak**: RED01 confirms 2009 (940,525 all-industry) > 2008
    (660,211), so the peak calendar year lies inside RED02 coverage;
    manufacturing 2009 = **188,171** (direct RED02, no scaling). CAVEAT:
    RED02 industry detail starts Jan–Mar 2009, so a hypothetical
    Oct-2008-to-Sep-2009 window cannot be tested for manufacturing; on the
    all-industry series calendar 2009 is the peak year regardless.
  - **Covid peak**: 2020 (909,542) > 2021; manufacturing 2020 =
    **116,420** (direct RED02).
  - **Historical median**: median of RED01 all-industry annual flows
    1996–2025 = **565,794**, scaled to manufacturing by the CURRENT
    manufacturing share of all-industry redundancies (64,463 / 511,407 =
    12.6%) → **71,319** (ASSUMPTION: constant manufacturing share; the
    actual share was 20.0% in 2009 and 12.8% in 2020, so this understates
    manufacturing-heavy recessions).
- Costing: all four arms are **linear in the eligible flow**, so the
  stressed total programme cost = central narrow total (£770m/yr) × flow
  ratio. GFC ratio 188,171/64,463 = 2.9 (`\TaaFlowRatioGFC`). Cross-check
  consistency: RED01 latest-year total (Jul 2025–Jun 2026) = 511,407,
  identical to the RED02 all-industry flow of section 2.

## 11. Self-financing break-even (added module; statutory arithmetic only)

- Question: how many months of reduced non-employment per insured worker
  make Arm A (central, narrow) self-financing?
- Per-month fiscal value of one month less non-employment for a central
  manufacturing worker re-employed at the post-displacement wage
  (£891.20 × (1 − 0.215) = £699.59/wk, £36,379/yr):
  - Forgone-benefit saving: UC single-25+ standard allowance £424.90/mo
    + NS JSA £95.55/wk × 52/12 = £414.05/mo → £838.95/mo. CONVENTION:
    the marginal month is assumed to fall within the first 6 months of the
    claim (when NS JSA is payable); internally consistent since the
    resulting break-even (3.6 months) < 6. JSA and UC are treated as
    additive; in reality NS JSA counts pound-for-pound as unearned income
    against UC, so for UC-entitled claimants this DOUBLE-COUNTS up to the
    JSA amount — the benefit saving is an upper bound and the break-even
    months therefore a lower bound. For claimants without UC entitlement
    (savings/partner income) the JSA-only saving applies instead.
  - Tax + employee NICs on one month of re-employment earnings, simple
    2026-27 calculation (rates from gov.uk "Income Tax rates and Personal
    Allowances" and "Rates and allowances: National Insurance
    contributions"): personal allowance £12,570 (frozen), basic rate 20%,
    employee Class 1 main rate 8% above the primary threshold (aligned
    with the PA at £12,570; main rate 8% since 6 April 2024). Post-
    displacement pay £36,379 is below the higher-rate threshold, so
    basic-rate-only arithmetic is exact: (£36,379 − £12,570) × 28% =
    £6,666/yr → £555.54/mo.
  - Total `\TaaFiscalPerMonth` = £1,394/mo.
- Arm A per-worker annual cost (central) = 50% × 21.5% × £891.20 × 52 =
  £4,982/yr → break-even `\TaaBreakEvenMonths` = 4,982 / 1,394 =
  **3.6 months** of average non-employment reduction per insured worker
  per year of payment.
- ILLUSTRATIVE comparison: Hyman, Kovak and Leive, "Wage Insurance for
  Displaced Workers" (NBER Working Paper 32464, 2024; Federal Reserve Bank
  of New York Staff Report 1105) find, from a US TAA/ATAA age-50
  eligibility discontinuity, that wage-insurance eligibility raises
  short-run employment probabilities enough to shorten non-employment by
  several months within the first two years, and conclude the US programme
  was roughly self-financing through reduced UI benefits and higher tax
  receipts. Their setting (US benefit levels, 50+ workers, 50% replacement
  capped) differs from this one in most institutional details — the
  comparison indicates plausibility of the 3.6-month hurdle, not a
  forecast.

## 12. Demographic composition of the eligible flow (added module)

- Latest-year (Jul 2025 – Jun 2026, same four non-overlapping quarters as
  section 2) composition of **all-industry** redundancies:
  - Male share: RED01 NSA men/people = 274,286 / 511,407 = **53.6%**
    (`\TaaEligMaleShare`).
  - 50+ share: RED02 `Age - levels` 50+/all-16+ = 176,973 / 511,407 =
    **34.6%** (`\TaaEligFiftyPlusShare`).
- ASSUMPTION: neither pinned source splits age or sex by industry, so the
  all-industry composition proxies the manufacturing (narrow) eligible
  flow; manufacturing is plausibly more male and older than the
  all-industry average, so both shares are likely understated.
- Why it matters: the strongest causal evidence for wage insurance
  (Hyman–Kovak–Leive's US ATAA discontinuity) is identified at the age-50
  eligibility cut-off — with over a third of the UK eligible flow aged
  50+, the sub-population for which the evidence is strongest is a large
  share of the caseload.
