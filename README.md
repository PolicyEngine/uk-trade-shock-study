# Who Bears the Trade Shocks?

**Distributional incidence and fiscal cushioning of UK trade events,
2021–2026, through PolicyEngine UK.**

The paper (`paper/main.pdf`) traces five UK trade events — the EU customs
border, the 2022 energy price surge, US tariffs, and the India and CPTPP
agreements — to households. The two price-led shocks are run through the
statutory tax–benefit system on the enhanced Family Resources Survey.

## Layout

```
uk_trade_shock_study/   the pipeline package
  incidence.py            first pass: public-data incidence, all five events
  second_stage_energy.py  energy episode on the enhanced FRS
  vintage_and_tca.py      rulebook-vintage run + TCA food second stage
  grid_energy.py          rebase x stack sensitivity grid (+ path variant)
  fig_extra.py            decomposition/paths figures, household bootstrap
  mpc_welfare.py          MPC-weighted demand + Atkinson welfare
  make_figures.py         first-pass figures and generated tables
  figstyle.py             PolicyEngine figure schema (shared with the
                          sibling AI-shock study)
data/raw/               pinned public inputs (SHA256-verified per run)
results/                pipeline outputs: results.json + generated macros
paper/                  the manuscript; every number is a generated macro
NOTES.md                replication notes, conventions, dataset hashes
```

## Reproduce

```bash
make bootstrap                       # uv sync
make first-pass                      # public data only
make second-stage DATASET=<enhanced_frs .h5>
make grid DATASET=<enhanced_frs .h5>
make paper
```

The second stage requires the enhanced FRS from policyengine-uk-data
(licensed FRS via the UK Data Service upstream; see NOTES.md for the
dataset hash). All first-pass inputs are public.
