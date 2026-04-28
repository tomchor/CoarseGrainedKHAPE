# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

KHAPE (Kelvin-Helmholtz Available Potential Energy) computes Available Potential Energy (APE) from Kelvin-Helmholtz instability simulations using the Winters et al. (1995) sorting method. The pipeline is:
1. **Julia simulation** (Oceananigans.jl on GPU) -> NetCDF output
2. **Python post-processing** -> filter fields, sort density, compute energy transfer and SFS budgets, plot

GitHub remote: `git@github.com:tomchor/APE_calculations.git`

## Running the Code

### Full pipeline (simulation + post-processing + sweep)
```bash
bash submit_all_pbs.sh                        # default Nz=2048, FIXED_REF=0
bash submit_all_pbs.sh NZ=1024 FIXED_REF=1   # custom resolution, fixed reference
```
Jobs are chained via PBS `afterok` dependencies. Always use `submit_*.sh` wrappers, never submit `*.pbs` files directly.

### Simulation only
```bash
bash submit_simulation.sh NZ=2048
```
Account: `UMCP0028`, queue: `casper`, 1x A100, 8 cores, 64 GB RAM.

The Julia simulation accepts CLI args: `--Nz`, `--Ri`, `--stop_time`, `--Re0`, `--Pr`, `--U`. For local CPU development:
```bash
julia --project -t 8 kelvin_helmholtz_instability.jl
julia --project -t 8 kelvin_helmholtz_instability.jl --Nz 512 --Ri 0.1 --stop_time 70 --Re0 1e-3
```

### Post-processing only
```bash
cd postprocessing
bash submit_budgeting.sh NZ=2048 FIXED_REF=both   # runs both reference profile variants
bash submit_sweep.sh NZ=2048 FIXED_REF=both
```
`FIXED_REF=0` (default) recomputes the reference profile each timestep; `FIXED_REF=1` fixes it to t=0; `FIXED_REF=both` submits both variants sharing a single filter job.

### Local post-processing (no PBS)
```bash
cd postprocessing
bash 00_get_budgets.sh output/khi_Nz512_Ri0.10.nc --filter-scales 0.8 2
bash 00_get_budgets.sh output/khi_Nz512_Ri0.10.nc --filter-scales 0.8 2 --fixed-reference
```
Set `N_WORKERS` env var to control parallelism (default 1): `N_WORKERS=4 bash 00_get_budgets.sh ...`

### Running tests
```bash
pytest tests/ -v -s                                  # default (time-varying reference)
pytest tests/ -v -s --ref-suffix _fixed_ref          # fixed reference variant
```
Tests check SFS KE and APE budget closure: rms(residual)/min(rms(terms)) < 10%. They expect post-processing output in `postprocessing/output/` for `khi_Nz512_Ri0.10`.

### CI
GitHub Actions (`.github/workflows/test.yml`) runs on push to `main` and on PR comments starting with `test` (via `test_trigger.yml`). The CI pipeline: Julia simulation (Nz=512) -> post-processing (both reference variants in parallel) -> pytest -> animation generation. Uses `pip install -r tests/requirements.txt` (not conda).

### Python environment
```bash
conda env create -f environment.yml   # creates env "py313"
conda activate py313
```

## Architecture

### Post-processing pipeline (`postprocessing/`)

Sequential numbered scripts (01-06), each reading the previous step's output. `00_get_budgets.sh` runs them all in sequence:

| Script | Purpose |
|--------|---------|
| `01_filter_fields.py` | Gaussian-filter velocity and buoyancy at multiple length scales |
| `02_sort_density.py` | Sort density to compute reference state (Winters et al. 1995) |
| `03_energy_transfer.py` | Cross-scale KE and APE transfer terms (Pi_KE, Pi_APE) |
| `04_sfs_ke_budget.py` | Sub-filter-scale KE budget terms |
| `05_sfs_ape_budget.py` | Sub-filter-scale APE budget terms |
| `06_plot_budgets.py` | Plot budget time series |

`inv*` scripts are the sweep variant (parameter sweep over filter scales): `inv1` filters, `inv2` computes transfer, `inv3` plots spectra.

Standalone visualization scripts (not part of the numbered pipeline):
- `plot1_panels.py` -- 4-panel snapshot of local SFS budget fields
- `plot2_budgets.py` -- 2x2 panel of SFS KE and APE budget time series
- `plot3_plot_transfer_spectrum.py` -- cross-scale transfer spectra
- `anim1_panels.py` -- animated version of plot1 panels (requires ffmpeg)

Shared utilities:
- `aux00_utils.py` -- data loading (`load_dataset_and_grid`), filtering (`filter_fields`, `GaussianFilter`, `DaskParallelFilter`), domain padding (`_pad_domain_in_z`), spatial derivatives (`calculate_gradient`), tensor condensing (`condense_velocities`)
- `aux01_pe_functions.py` -- density sorting (`sorted_timeseries`), potential energy calculations (`local_potential_energies_timeseries`), APE budget terms (SFS flux tensor, cross-scale APE flux, SFS APE dissipation, reference-tendency correction R)
- `aux02_ke_functions.py` -- SFS stress tensor, strain rate tensor, cross-scale KE flux, SFS KE dissipation, full energy transfer pipeline (`calculate_energy_transfer`)
- `aux03_plotting.py` -- plotting helpers (`budget_colors`, `run_label`, `plot_sfs_budget`)

All post-processing scripts accept `--filename`, `--filter-scales`, `--n-workers`, `--fixed-reference` via argparse. Output goes to `postprocessing/output/`.

### Data flow between pipeline steps

The sorted density (`*_sorted_density.nc`) produced by step 02 is reused by steps 03, 05, and the sweep pipeline (`inv2`), avoiding redundant sorts. When `--fixed-reference` is used, output files are suffixed `_fixed_ref`. The sweep's `inv2` with `--fixed-reference` expects the sorted density from the budget pipeline's step 02 to already exist.

### Julia layer
- `kelvin_helmholtz_instability.jl` -- main simulation (Oceananigans.jl, WENO(5), adaptive timestep)
- `utils.jl` -- `closest_factor_number()` (FFT-friendly grid sizes), `show_gpu_status()`
- Setup: shear flow u(z)=tanh(z), stratification b(z)=Ri*tanh(z/0.25), default Ri=0.1
- Domain: Lx=Lz=10, Ly=5 (periodic x,y; bounded z)
- Output: `output/kelvin_helmholtz_instability_NxNyNz.nc`

### Key dependencies
- **Python**: `numpy`, `xarray`, `scipy`, `matplotlib`, `dask`, `gcm_filters`, `netcdf4`
- **Julia**: `Oceananigans`, `Oceanostics`, `CUDA`, `NCDatasets`, `CairoMakie`

## Physics Reference

- **TPE** = integral of g*rho*z dV  (total potential energy)
- **RPE** = minimum PE achievable by adiabatic rearrangement (from sorted reference state)
- **APE** = TPE - RPE  (available for conversion to KE)
- **Pi_KE**, **Pi_APE** -- cross-scale energy transfer (sub-filter to resolved)
- Physical constants: `g=9.81`, `rho_0=1025`

## Code Style

- Do not break a command/statement into multiple lines if it fits within 140 columns.
- Always delimit code sections with `#+++` on the opening line and `#---` on the closing line:
  ```python
  #+++ Section name
  ...code...
  #---
  ```

## Maintenance Rules

- **Always update `README.md` when the job submission scheme changes.** This includes: adding/removing/renaming PBS scripts or wrapper scripts, changing argument names or defaults, adding new pipeline stages, or changing job dependency chains.

## Notes

- Output files are excluded from git (`.nc`, `.mp4`, `.pdf`, `.png`, `.jld2`).
- Simulation output can reach 650 GB; scratch directory is `/glade/derecho/scratch/tomasc/khape/output/`.
- Logs: `logs/<job_name>.log` (PBS), `logs/<job_name>.out` (Python stdout via tee). Job names follow `<stage>_Nz<NZ>_Ri0.10[_fixed_ref]`.
