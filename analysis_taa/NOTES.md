# TAA costing pipeline — sources and assumptions

First-pass annual costings of four candidate UK worker-side trade-adjustment
policies. One rerunnable script (`costing.py`, dependency: `openpyxl` only),
mirroring the conventions of
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
