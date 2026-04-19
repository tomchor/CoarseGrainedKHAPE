# KHAPE — Kelvin-Helmholtz Available Potential Energy

Computes Available Potential Energy (APE) from Kelvin-Helmholtz instability simulations using the Winters et al. (1995) sorting method.

## Pipeline overview

1. **Julia simulation** (`simulation.pbs`) — runs the KH instability on a GPU and writes NetCDF output
2. **Post-processing** (`postprocessing/budgeting.pbs`) — filters fields, computes energy transfer and SFS budgets
3. **Sweep** (`postprocessing/sweep.pbs`) — parameter sweep over filter scales

## Submitting jobs

### File naming convention

| Extension | Role |
|-----------|------|
| `*.pbs`   | PBS job script — passed directly to `qsub`; do not run with `bash` |
| `submit_*.sh` | Wrapper script — constructs job names/log paths and calls `qsub`; this is what you invoke |

Always use the `submit_*.sh` wrappers rather than submitting `*.pbs` files directly — the wrappers ensure job names and log files reflect the run parameters.

### Run everything (simulation + post-processing + sweep)

```bash
# Default resolution (Nz=2048), time-varying reference profile
bash submit_all_pbs.sh

# Custom resolution
bash submit_all_pbs.sh 1024

# Custom resolution with fixed-in-time reference profile
bash submit_all_pbs.sh 1024 1
```

Jobs are chained: post-processing only starts after the simulation succeeds, and the sweep only starts after post-processing succeeds.

### Run simulation only

```bash
# Default (Nz=1024)
bash submit_simulation.sh

# Custom resolution
bash submit_simulation.sh 2048
```

### Run post-processing only

```bash
# Default (Nz=2048), time-varying reference profile
cd postprocessing
bash submit_budgeting.sh

# Custom resolution
bash submit_budgeting.sh 1024

# With fixed-in-time reference profile
bash submit_budgeting.sh 1024 1
```

The `FIXED_REF` flag controls how the reference (sorted) density profile is computed:
- `0` (default) — reference profile is recomputed at every time step
- `1` — reference profile is fixed to the `t=0` density field

Output files are suffixed with `_fixed_ref` when `FIXED_REF=1`.

### Run sweep only

```bash
cd postprocessing
bash submit_sweep.sh        # default Nz=4096
bash submit_sweep.sh 2048
```

## Logs

All job logs are written to the `logs/` subdirectory next to the submit script:
- `logs/<job_name>.log` — PBS stdout/stderr (written by PBS after job ends)
- `logs/<job_name>.out` — Python script output (written live via `tee`)

Job names follow the pattern `budgeting_Nz<NZ>_Ri0.10[_fixed_ref]`.
