# KHAPE — Kelvin-Helmholtz Available Potential Energy

Computes Available Potential Energy (APE) from Kelvin-Helmholtz instability simulations using the Winters et al. (1995) sorting method.

## Pipeline overview

1. **Julia simulation** (`simulation.pbs`) — runs the KH instability on a GPU and writes NetCDF output
2. **Post-processing** — filters fields, sorts density, computes energy transfer and SFS budgets, split into two jobs:
   - `postprocessing/budgeting_filter.pbs` — filters fields at all scales (shared; runs once regardless of `FIXED_REF`)
   - `postprocessing/budgeting.pbs` — sorts density, computes all budget terms and plots (per `FIXED_REF` variant)
3. **Sweep** — parameter sweep over filter scales, split into two jobs:
   - `postprocessing/sweep_filter.pbs` — filters fields at all scales (shared; runs once regardless of `FIXED_REF`)
   - `postprocessing/sweep_transfer.pbs` — computes and plots energy transfer spectra (per `FIXED_REF` variant)

## Post-processing scripts (`postprocessing/`)

Scripts in `postprocessing/` follow a naming convention by purpose:

| Prefix | Purpose |
|--------|---------|
| `01_…` – `06_…` | **Numbered post-processing pipeline.** Sequentially filter fields, sort density, compute cross-scale energy transfer, and compute SFS KE/APE budgets from the raw simulation output. Each step reads the previous step's output. |
| `sweep1_…` – `sweep3_…` | **Parameter sweep pipeline** over filter scales: filter fields, compute cross-scale transfer at every scale, and plot transfer spectra. |
| `plot2_…`, `plot3_…`, `plot4_…` | **Paper figure scripts.** Produce the figures used in the manuscript (cross-scale transfer spectrum, SFS KE/APE budget time series, local-field snapshot panels). Output goes to `figures/`. |
| `anim1_…`, `S1_…`, `S2_…`, `S3_…` | **Supplementary material.** Animations (`anim*`, requires `ffmpeg`) and supplementary figures (`S1`–`S3`: Π hovmöllers, snapshot panels, sweep-spectrum figures). |
| `aux*` (under `src/`) | Shared utilities reused across the pipeline (data loading, Gaussian filtering, spatial derivatives, PE/KE budget terms, plotting helpers). |
| `00_get_budgets.sh`, `inv00_get_sweep.sh` | Local helpers that run the numbered pipeline or sweep pipeline end-to-end without PBS (see [Running locally](#running-locally-without-pbs)). |
| `*.pbs`, `submit_*.sh` | PBS job scripts and their wrappers (see [Submitting jobs](#submitting-jobs)). |

All Python scripts accept `--filename`, and most accept `--fixed-reference`, `--filter-scales`, and `--n-workers`. Run any script with `--help` for its full argument list.

## Setup

Create the conda environment for Python post-processing:

```bash
conda env create -f environment.yml   # creates env "py313"
conda activate py313
```

The Julia simulation uses the project's `Project.toml`/`Manifest.toml` — instantiate with `julia --project -e 'using Pkg; Pkg.instantiate()'` on first use.

## Submitting jobs

### File naming convention

| Extension | Role |
|-----------|------|
| `*.pbs`   | PBS job script — passed directly to `qsub`; do not run with `bash` |
| `submit_*.sh` | Wrapper script — constructs job names/log paths and calls `qsub`; this is what you invoke |

Always use the `submit_*.sh` wrappers rather than submitting `*.pbs` files directly — the wrappers ensure job names and log files reflect the run parameters.

Arguments are passed as `KEY=VALUE` pairs in any order. All arguments are optional and fall back to their defaults if omitted.

### Run everything (simulation + post-processing + sweep)

```bash
# Default resolution (Nz=2048), time-varying reference profile
bash submit_all_pbs.sh

# Custom resolution
bash submit_all_pbs.sh NZ=1024

# Custom resolution with fixed-in-time reference profile
bash submit_all_pbs.sh NZ=1024 FIXED_REF=1
```

Jobs are chained: `budgeting_filter` starts after simulation, `budgeting` starts after `budgeting_filter`, `sweep_filter` starts after `budgeting`, and `sweep_transfer` starts after `sweep_filter`. When `FIXED_REF=1`, the budgeting and sweep transfer jobs load the pre-sorted reference density from the preceding step.

### Run simulation only

```bash
# Default (Nz=1024)
bash submit_simulation.sh

# Custom resolution
bash submit_simulation.sh NZ=2048

# Also write the per-scale strain/stress tensor components (for online-vs-offline validation)
bash submit_simulation.sh NZ=2048 SAVE_TENSORS=1
```

`SAVE_TENSORS=1` passes `--save_tensors` to the Julia simulation, which additionally outputs the
resolved strain-rate (S̄ⁱʲ) and sub-filter stress (τⁱʲ) tensor components at each filter scale. These
are full 3D fields (off by default to keep production output lean) and are consumed only by the
validation scripts in `postprocessing/validation/`.

### Run a simulation + online-vs-offline validation

```bash
# Submit the simulation (with --save_tensors) then a chained validation job (default Nz=2048)
bash submit_validation_run.sh
bash submit_validation_run.sh NZ=1024
```

`submit_validation_run.sh` submits the simulation with `SAVE_TENSORS=1` and a `validation` job that
runs after it (`afterok`). The validation job recomputes the filtered fields, cross-scale KE transfer
Π_K, and the strain/stress tensors offline and compares them against the simulation's online
diagnostics (`postprocessing/validation/inv01`–`inv04`), writing comparison figures to `figures/` and
online-vs-offline animations to `animations/`.

### Run post-processing only

The budgeting pipeline is split into two PBS jobs to avoid race conditions when running both `FIXED_REF` variants simultaneously: the field-filtering step (`01`) runs once and is shared, while the density sort and budget steps (`02`–`06`) run separately per variant.

```bash
cd postprocessing
bash submit_budgeting.sh                          # default Nz=2048, FIXED_REF=0
bash submit_budgeting.sh NZ=1024
bash submit_budgeting.sh NZ=2048 FIXED_REF=1     # fixed-in-time reference profile
bash submit_budgeting.sh NZ=2048 FIXED_REF=both  # submit both variants; filter runs only once
```

`FIXED_REF=both` submits the filter job once and two budget jobs (one for each variant) that both depend on the single filter job.

The `FIXED_REF` argument controls how the reference (sorted) density profile is computed:
- `0` (default) — reference profile is recomputed at every time step
- `1` — reference profile is fixed to the `t=0` density field

Output files are suffixed with `_fixed_ref` when `FIXED_REF=1`.

### Run sweep only

The sweep is split into two PBS jobs to avoid race conditions when running both `FIXED_REF` variants simultaneously: the field-filtering step (`sweep1`) runs once and is shared, while the energy transfer and plotting steps (`sweep2`+`sweep3`) run separately per variant.

```bash
cd postprocessing
bash submit_sweep.sh                          # default Nz=2048, FIXED_REF=0
bash submit_sweep.sh NZ=4096
bash submit_sweep.sh NZ=2048 FIXED_REF=1     # fixed-in-time reference profile
bash submit_sweep.sh NZ=2048 FIXED_REF=both  # submit both variants; filter runs only once
```

`FIXED_REF=both` submits the filter job once and two transfer jobs (one for each variant) that both depend on the single filter job.

When `FIXED_REF=1`, the transfer job loads the pre-sorted reference density from `_sorted_density_fixed_ref.nc` (produced by the budgeting pipeline). Run budgeting with `FIXED_REF=1` before submitting the sweep with `FIXED_REF=1`.

## Running locally (without PBS)

For development on a workstation (no PBS scheduler), run the simulation and post-processing pipeline directly.

```bash
# Julia simulation (CPU, small grid)
julia --project -t 8 kelvin_helmholtz_instability.jl --Nz 512 --Ri 0.1 --stop_time 70

# Numbered post-processing pipeline (01–06) on an existing NetCDF file
cd postprocessing
bash 00_get_budgets.sh output/khi_Nz512_Ri0.10.nc --filter-scales 1 7
bash 00_get_budgets.sh output/khi_Nz512_Ri0.10.nc --filter-scales 1 7 --fixed-reference

# Sweep pipeline (sweep1–sweep3)
bash inv00_get_sweep.sh output/khi_Nz512_Ri0.10.nc
```

Set `N_WORKERS` to control Dask parallelism (default 1): `N_WORKERS=4 bash 00_get_budgets.sh ...`.

## Tests

The test suite checks SFS KE and APE budget closure (rms residual / min rms of terms < 10%) and expects post-processing output for `khi_Nz512_Ri0.10` in `postprocessing/output/`.

```bash
pytest tests/ -v -s                                  # time-varying reference (default)
pytest tests/ -v -s --ref-suffix _fixed_ref          # fixed reference variant
```

CI (`.github/workflows/test.yml`) runs the full chain — Julia simulation (Nz=512) → post-processing (both reference variants in parallel) → pytest → animation — on push to `main` and on PR comments starting with `test`.

## Logs

All job logs are written to the `logs/` subdirectory next to the submit script:
- `logs/<job_name>.log` — PBS stdout/stderr (written by PBS after job ends)
- `logs/<job_name>.out` — Python script output (written live via `tee`)

Job names follow the pattern `<stage>_Nz<NZ>_Ri0.10[_fixed_ref]`, e.g. `budgeting_Nz2048_Ri0.10_fixed_ref`, `sweep_filter_Nz2048_Ri0.10`, `sweep_transfer_Nz2048_Ri0.10_fixed_ref`.
