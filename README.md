# Who bears the trade shocks?

**Distributional incidence and fiscal cushioning of UK trade events,
2021–2026, through PolicyEngine UK.**

The paper (`paper/main.pdf`) traces five UK trade events (the EU customs
border, the 2022 energy price surge, US tariffs, and the India and CPTPP
agreements) to households. The two price-led shocks run through the
statutory tax and benefit system on the enhanced Family Resources Survey.

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
tests/                  public-data checks (hashes, macros, sync, first pass)
NOTES.md                replication notes, conventions, dataset hashes
```

## Reproduce

```bash
make bootstrap                       # uv sync
make test                            # public-data checks
make first-pass                      # public data only
make second-stage DATASET=<enhanced_frs .h5>
make grid DATASET=<enhanced_frs .h5>
make paper
```

## Data

The first pass uses only public inputs, pinned by SHA256 in
`uk_trade_shock_study/incidence.py` and listed in `NOTES.md`.

The second stage uses `enhanced_frs_2023_24.h5` from policyengine-uk-data
release 1.55.12 (Hugging Face repository
`policyengine/policyengine-uk-data-private`, revision
`4e25f9d6b67244340161098e76bb5e67148eb1e7`, SHA256
`584ae33d80ca0431254610a3f8254d132da73477d31966d6446282861ecae50d`). It is
derived from the Family Resources Survey, licensed through the UK Data
Service, so it cannot be redistributed here; access is through PolicyEngine's
data repository under the same licence. The model version is pinned to
policyengine-uk 2.89.2 in `uv.lock`.

## Licence

AGPL-3.0, the PolicyEngine convention (see `LICENSE`).
