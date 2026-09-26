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

### Scale decomposition (`--reference`)

`03_energy_transfer.py`, `05_sfs_ape_budget.py` and `sweep2_energy_transfer.py` take `--reference {filtered,true}`, selecting the reference state the **resolved** reservoir is measured against:

| Value | Reference | Valid for |
|-------|-----------|-----------|
| `filtered` (default) | the vertically filtered profile ⟨ρ_*⟩ | any kernel, including one with vertical extent |
| `true` | the unfiltered ρ_* | horizontal kernels only (the `g_z = δ(z)` limit) |

The pipeline filters in x **and** z, so `filtered` is the correct choice: against the unfiltered ρ_* the resolved reservoir does not vanish for a fluid at rest and the sub-filter remainder goes negative over much of the domain. `true` is the earlier formulation, kept for comparison; it does not reproduce the earlier numbers, since every integral now excludes the z padding and `Π_A` and `ε_Aˢ` are computed offline. The simulation's online APE terms (`Π_A_ℓ<ℓ>`, `ε_As_ℓ<ℓ>` and the rest, under `--save_sorted`) are built against ⟨ρ_*⟩ too, on a coarse column; the budget computes every APE term offline under either value, and the online set is the validation cross-check. The z padding is also sized to the widest filter scale in use, rather than fixed at half the domain. See CLAUDE.md for the full account.

`--reference true` tags the outputs of 03–06 and `sweep2` with `_trueref`, so both references can sit side by side; 05 refuses 03/04 output built against the other one. The plotting scripts take the same flag to read the tagged files. `00_get_budgets.sh` forwards the flag, including to 06:

```bash
bash 00_get_budgets.sh output/khi_Nz512_Ri0.10.nc --filter-scales 1 7 --reference filtered
bash 00_get_budgets.sh output/khi_Nz512_Ri0.10.nc --filter-scales 1 7 --reference true
```

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

### Output locations

Two environment variables move the output off the repository, for example to scratch. Set them in the login environment, since PBS jobs do not inherit the submitting shell's variables, and give absolute paths.

| Variable | Default | Read by |
|----------|---------|---------|
| `KHAPE_OUTPUT_DIR` | `output/` | the simulation (where it writes), every post-processing PBS job (where they read the run), and the tests |
| `KHAPE_PP_OUTPUT` | `postprocessing/output/` | every post-processing script (through `src/aux00_utils.PP_OUTPUT`) and the tests |

### Run everything (simulation + post-processing + sweep, with optional validation and plots)

```bash
# Default resolution (Nz=2048), time-varying reference profile
bash submit_all_pbs.sh

# Custom resolution
bash submit_all_pbs.sh NZ=1024

# Custom resolution with fixed-in-time reference profile
bash submit_all_pbs.sh NZ=1024 FIXED_REF=1

# Add the online-vs-offline validation and/or the final plots (independently toggleable)
bash submit_all_pbs.sh VALIDATE=1            # + validation (figures + animations); runs the sim with --save_tensors and --save_sorted
bash submit_all_pbs.sh PLOTS=1               # + plot2/plot3/plot4 after sweep_transfer
bash submit_all_pbs.sh VALIDATE=1 PLOTS=1    # the whole pipeline
```

Jobs are chained: `budgeting_filter` starts after simulation, `budgeting` starts after `budgeting_filter`, `sweep_filter` starts after `budgeting`, and `sweep_transfer` starts after `sweep_filter`. When `FIXED_REF=1`, the budgeting and sweep transfer jobs load the pre-sorted reference density from the preceding step.

`SAVE_SORTED` defaults to `1`, so the simulation writes the sorted reference state and the online APE budget terms. The validation job (`inv06`–`inv10`) and the online panels animation use them; the offline budget does not, since `03` and `05` compute `Π_A` and `ε_Aˢ` offline against the exact ⟨ρ_*⟩. `SAVE_SORTED=0` gives smaller output and changes no budget number, and `VALIDATE=1` turns it back on.

Two optional stages are gated by flags (both default `0`, so the base behavior is simulation + post-processing + sweep):
- `VALIDATE=1` runs the simulation with `--save_tensors` (and with `--save_sorted`, even if `SAVE_SORTED=0`) and submits a parallel **validation** job (`postprocessing/validation/validation.pbs`) after the simulation, writing online-vs-offline comparison figures (`figures/validation/`) and animations (`animations/`).
- `PLOTS=1` submits a **plots** job (`postprocessing/plots.pbs`) after `sweep_transfer` that runs `plot2_transfer_spectrum.py`, `plot3_budgets.py`, and `plot4_panels.py`.

### Run simulation only

```bash
# Default (Nz=1024)
bash submit_simulation.sh

# Custom resolution
bash submit_simulation.sh NZ=2048

# Also write the per-scale strain/stress tensor components (for online-vs-offline validation)
bash submit_simulation.sh NZ=2048 SAVE_TENSORS=1

# Also write the Winters (1995) sorted reference state (for online-vs-offline validation)
bash submit_simulation.sh NZ=2048 SAVE_SORTED=1
```

`SAVE_TENSORS=1` passes `--save_tensors` to the Julia simulation, which additionally outputs the
resolved strain-rate (S̄ⁱʲ) and sub-filter stress (τⁱʲ) tensor components at each filter scale. These
are full 3D fields (off by default to keep production output lean) and are consumed only by the
validation scripts in `postprocessing/validation/`.

`SAVE_SORTED` (default **1**) passes `--save_sorted`, which additionally outputs the adiabatically sorted reference
state under each of the three Oceanostics sorting methods: the reference height `z✶_3dsort`
(`ThreeDimensionalSort`) and `z✶_heaviside` (`HeavisideIntegral`) as 3D fields on the model grid, and
the sorted column `z✶_1dsort` / `b✶_1dsort` (`VerticalSort`) on its own N = Nx·Ny·Nz vertical axis. It
also emits the online local available potential energy `E_a` (and its integral `∫E_a`) and the
sub-filter APE dissipation `ε_As_ℓ<ℓ>` at each online filter scale, which `05_sfs_ape_budget.py` reads
back instead of recomputing. All of these
go into the main output file, since one `NetCDFWriter` holds both grids; the resulting per-grid
dimension suffixing is undone at load time by the post-processing loader, so the rest of the pipeline
is unaffected. This is the online counterpart of what `02_sort_density.py` and the offline APE
computation produce; `inv06_compare_sorted_profiles.py`, `inv07_compare_local_ape.py` and
`inv08_compare_sfs_ape_dissipation.py` compare the two. Each method carries its own full-domain sort per
output, so this is off by default.

### Run a simulation + online-vs-offline validation

```bash
# Submit the simulation (with --save_tensors) then a chained validation job (default Nz=2048)
bash submit_validation_run.sh
bash submit_validation_run.sh NZ=1024
```

`submit_validation_run.sh` submits the simulation with `SAVE_TENSORS=1` and a `validation` job that
runs after it (`afterok`). The validation job recomputes the filtered fields, cross-scale KE transfer
Π_K, the strain/stress tensors, the SFS KE dissipation ε_Kˢ, the sorted reference state, the local APE,
and the sub-filter APE dissipation ε_Aˢ offline and
compares them against the simulation's online diagnostics (`postprocessing/validation/inv01`–`inv09`),
writing comparison figures to `figures/validation/` and online-vs-offline animations to `animations/`. Note that
`inv06`, `inv07`, `inv08` and `inv09` need a run with `SAVE_SORTED=1` (which `submit_all_pbs.sh VALIDATE=1` sets automatically).

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
bash submit_sweep.sh NZ=2048 EXTENSION=odd   # the whole sweep with b oddly reflected past the walls
```

`submit_sweep.sh` refuses any argument it does not know, so a misspelled key cannot silently rerun the default sweep over the production files.

#### Wall-extension test

The manuscript leaves the extension of b and b✶ past the walls free (§2) and uses the wall values (§4). `EXTENSION=odd` reflects b oddly about the wall value instead; every other field keeps the wall value, so the comparison changes the buoyancy rule alone. Its files carry an `_odd` tag. `extension_test.pbs` runs the odd rule at one scale, snapped to the nearest of the production sweep's 30, and `compare_extension.py` sets it against the production (edge) sweep:

```bash
cd postprocessing
qsub -v NZ=2048,SCALE=20 extension_test.pbs                               # odd rule at l=20 only (no wrapper yet)
python compare_extension.py --filename output/khi_Nz2048_Ri0.10.nc --filter-scale 20
python compare_extension.py --filename output/khi_Nz2048_Ri0.10.nc --all   # every scale, after EXTENSION=odd
```

A run over a subset of scales (`sweep1 --filter-scales`, as `extension_test.pbs` does) tags its files with the scales, e.g. `_sweep_l20_odd.nc`, so it never replaces a full sweep; `sweep2` takes the same `--filter-scales` to find it, and `compare_extension.py` prefers it for a single-scale comparison.

`FIXED_REF=both` submits the filter job once and two transfer jobs (one for each variant) that both depend on the single filter job.

When `FIXED_REF=1`, the transfer job builds its own frozen reference column: it sorts t=0 on the grid it loaded and broadcasts that row over the time axis. **The sweep does not need the budgeting pipeline to have run**, and does not read `_sorted_density_fixed_ref.nc`. It cannot: the column's z✶ are the padded grid's own heights, and the sweep pads to 4σ of its widest scale (ℓ=20) while `02_sort_density.py` pads to the budget scales — at Nz=2048, 2784 cells per side against 1024 — so the budgeting pipeline's column belongs to a different grid. Sorting t=0 costs one sort, and is bit-identical to `02`'s output whenever the two paddings do coincide.

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

The test suite checks SFS KE and APE budget closure (rms residual / min rms of terms < 10%) and expects the CI run, `khi_Nz1024_Ri0.10`, in `output/` with its post-processing output in `postprocessing/output/`. That name is set once, as `STEM` in `tests/conftest.py`; change it there to test a different run.

```bash
pytest tests/ -v -s                                  # time-varying reference (default)
pytest tests/ -v -s --ref-suffix _fixed_ref          # fixed reference variant
pytest tests/ -v -s --ref-suffix _trueref            # --reference true outputs
```

CI (`.github/workflows/test.yml`) runs the full chain — Julia simulation (Nz=1024) → post-processing (both reference variants in parallel) → pytest → animation — on push to `main` and on PR comments starting with `test`.

## Logs

All job logs are written to the `logs/` subdirectory next to the submit script:
- `logs/<job_name>.log` — PBS stdout/stderr (written by PBS after job ends)
- `logs/<job_name>.out` — Python script output (written live via `tee`)

Job names follow the pattern `<stage>_Nz<NZ>_Ri0.10[_fixed_ref]`, e.g. `budgeting_Nz2048_Ri0.10_fixed_ref`, `sweep_filter_Nz2048_Ri0.10`, `sweep_transfer_Nz2048_Ri0.10_fixed_ref`.
