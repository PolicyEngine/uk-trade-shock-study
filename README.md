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
uk_trade_shock_study/   household and decile-table pipeline
  microcosm.py            certified Microcosm UK loader (policyengine.py)
  incidence.py            decile-table incidence from public data
  second_stage_energy.py  energy episode on Microcosm UK households
  vintage_and_tca.py      policy-vintage run + TCA food on households
  uprating_lag_reform.py  parameters-only uprating-lag costing
  deflated_inputs.py      deflated-input robustness
  grid_energy.py          rebase x stack grid, price paths, seasonality
  energy_referee_checks.py  grid extremes, Council Tax rebate, CoL outturn
  ceta_second_stage.py    UK-India agreement on households
  rank_robustness.py      decile-ranking robustness
  mpc_welfare.py          MPC-weighted demand + Atkinson welfare
  lcfs_validation.py      within-decile spread against the LCFS (aggregates)
  fig_extra.py, fig_grid.py, make_figures.py, make_within_decile.py  figures
tariff_pipeline/        US-tariff earnings scenario (exposure, shocks,
                        exact-expectation displacement, runner)
analysis/               tariff runs and macro emitters (run_scenarios.py,
                        ashe_validation.py, sensitivity_duration_takeup.py,
                        emit_e3_*.py, emit_data_macros.py, validate_manifest.py)
data/raw/               pinned public inputs (SHA256-verified)
results/                pipeline outputs and generated macros
paper/                  the manuscript; every number is a generated macro
tests/                  pipeline, macro and manifest checks
NOTES.md                replication notes, conventions, dataset hashes
```

## Reproduce

```bash
make bootstrap                       # uv sync
make test                            # pipeline and macro checks
export HUGGING_FACE_TOKEN=...        # access to the certified Microcosm UK build
make e3                              # tariff scenario
make first-pass                      # decile-table analysis (public data)
make second-stage                    # household stage on Microcosm UK
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
