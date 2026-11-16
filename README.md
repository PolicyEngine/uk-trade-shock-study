# Who pays for trade shocks?

**Distributional incidence and fiscal cushioning of UK trade events,
2021–2026, through PolicyEngine UK.**

The paper (`paper/main.pdf`) traces four UK trade events (the EU customs
border, the 2022 energy price surge, the 2025 US tariffs and the India
agreement) to households. Every microsimulation runs on policyengine.py
6.20.0 (policyengine-uk 2.174.1) with the certified Microcosm UK 2024-25
dataset.

## Layout

```
uk_trade_shock_study/   the pipeline package
  incidence.py            decile-table incidence from public data
  second_stage_energy.py  energy episode on Microcosm UK households
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
export HUGGING_FACE_TOKEN=...        # access to the certified Microcosm UK build
make second-stage                    # no DATASET argument needed
make grid
make paper
```

## Data

The first pass uses only public inputs, pinned by SHA256 in
`uk_trade_shock_study/incidence.py` and listed in `NOTES.md`.

The household-level stage, including the tariff (E3) earnings channel, uses
the certified Microcosm UK 2024-25 dataset, resolved by
`policyengine.uk.managed_microsimulation()` from the policyengine.py 6.20.0
release bundle (`policyengine[uk]==6.20.0` in `pyproject.toml`, which pins
policyengine-uk 2.174.1):

- build `microcosm-uk-2024-25-national-20261107T020434Z-eeb6093e`
- SHA256 `0a9b77db1c0dee2693cb2d47bad68da34cb662916eee4829942bcd45de581f25`
- 78,181 household records; household weights sum to 29.0m (2.1% above the
  ONS count of 28.4m); person weights sum to 69.0m

The dataset is derived from the Family Resources Survey, licensed through
the UK Data Service, so it cannot be redistributed here. It lives in a
private Hugging Face repository: set `HUGGING_FACE_TOKEN` (or run
`huggingface-cli login`) with an account that has access. Microcosm UK
carries inputs for 2024-25 only; the energy scripts copy every input into
2023 (and 2022 for the uprating comparison) before simulating.

## Licence

AGPL-3.0, the PolicyEngine convention (see `LICENSE`).
