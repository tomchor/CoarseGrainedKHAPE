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
```

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

The sweep is split into two PBS jobs to avoid race conditions when running both `FIXED_REF` variants simultaneously: the field-filtering step (`inv1`) runs once and is shared, while the energy transfer and plotting steps (`inv2`+`inv3`) run separately per variant.

```bash
cd postprocessing
bash submit_sweep.sh                          # default Nz=2048, FIXED_REF=0
bash submit_sweep.sh NZ=4096
bash submit_sweep.sh NZ=2048 FIXED_REF=1     # fixed-in-time reference profile
bash submit_sweep.sh NZ=2048 FIXED_REF=both  # submit both variants; filter runs only once
```

`FIXED_REF=both` submits the filter job once and two transfer jobs (one for each variant) that both depend on the single filter job.

When `FIXED_REF=1`, the transfer job loads the pre-sorted reference density from `_sorted_density_fixed_ref.nc` (produced by the budgeting pipeline). Run budgeting with `FIXED_REF=1` before submitting the sweep with `FIXED_REF=1`.

## Logs

All job logs are written to the `logs/` subdirectory next to the submit script:
- `logs/<job_name>.log` — PBS stdout/stderr (written by PBS after job ends)
- `logs/<job_name>.out` — Python script output (written live via `tee`)

Job names follow the pattern `<stage>_Nz<NZ>_Ri0.10[_fixed_ref]`, e.g. `budgeting_Nz2048_Ri0.10_fixed_ref`, `sweep_filter_Nz2048_Ri0.10`, `sweep_transfer_Nz2048_Ri0.10_fixed_ref`.
