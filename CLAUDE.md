# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

KHAPE (Kelvin-Helmholtz Available Potential Energy) computes Available Potential Energy (APE) from Kelvin-Helmholtz instability simulations using the Winters et al. (1995) sorting method. The pipeline is:
1. **Julia simulation** (Oceananigans.jl on GPU) → NetCDF output
2. **Python post-processing** → APE/KE time series, filtering, plots

GitHub remote: `git@github.com:tomchor/APE_calculations.git`

## Running the Code

### Julia simulation (local CPU)
```bash
julia --project -t 8 kelvin_helmholtz_instability.jl
```

### Julia simulation (HPC — Derecho/Casper GPU)
```bash
qsub submit_kh_pbs.sh
```
Account: `UMCP0028`, queue: `casper`, 1× A100, 8 cores, 64 GB RAM, 23:59 walltime.

### Python APE analysis
```bash
python calculate_ape.py
```
Outputs `kelvin_helmholtz_ape.nc` with TPE, RPE, APE, KE time series.

## Architecture

### Julia layer (`kelvin_helmholtz_instability.jl`, `utils.jl`)
- **Framework**: Oceananigans.jl (incompressible Navier-Stokes, implicit LES)
- **Setup**: shear flow `u(z)=tanh(z)`, stratification `b(z)=0.025·tanh(z/0.25)`, Richardson number Ri=0.1
- **Numerics**: WENO(order=5) advection, adaptive timestepping
- **Domain**: Lx=Lz=10, Ly=5 (periodic x,y; bounded z); Nz=1024 on GPU
- **Output**: 3D NetCDF fields (u,v,w,b,pe,ω) at `output/kelvin_helmholtz_instability_NxNyNz.nc`
- `utils.jl` provides `closest_factor_number()` (FFT-friendly grid sizes) and `show_gpu_status()`

### Python layer (`ape_calculations.py`, `calculate_ape.py`, `ape_plots.py`)
- **`ape_calculations.py`**: Core library (~880 lines). Key functions:
  - `load_data()` — reads NetCDF, extracts grid, converts buoyancy → density
  - `vertical_sort_density_by_flattening()` / `vertical_sort_density_by_PDF()` — two methods to compute reference state
  - `local_potential_energies_timeseries()` / `integrated_potential_energies_timeseries()` — APE at each grid point and globally
  - `integrated_KE_timeseries()` — kinetic energy
  - Physical constants: `g=9.81`, `ρ0=1025`
- **`calculate_ape.py`**: Entry point. Loads data, validates energy conservation, applies Gaussian filtering via `gcm_filters` at σ = 0.1, 0.2, 0.4, 0.8, 1.6, saves results.
- **`ape_plots.py`**: Energy time-series plots (TPE/RPE, APE/KE, normalized budget).

### Key dependencies
- **Python**: `numpy`, `xarray`, `scipy`, `matplotlib`, `pynanigans`, `gcm_filters`
- **Julia**: `Oceananigans`, `Oceanostics`, `CUDA`, `NCDatasets`, `CairoMakie`/`GLMakie`

## Physics Reference

- **TPE** = ∭ g·ρ·z dV  (total potential energy)
- **RPE** = minimum PE achievable by adiabatic rearrangement (from sorted reference state)
- **APE** = TPE − RPE  (available for conversion to KE)

## Code Style

- Always delimit code sections with `#+++` on the opening line and `#---` on the closing line, e.g.:
  ```python
  #+++ Section name
  ...code...
  #---
  ```

## Maintenance Rules

- **Always update `README.md` when the job submission scheme changes.** This includes: adding/removing/renaming PBS scripts or wrapper scripts, changing argument names or defaults, adding new pipeline stages, or changing job dependency chains. The README is the authoritative reference for how to submit jobs.

## Notes

- Output files are excluded from git (`.nc`, `.mp4`, `.pdf`, `.png`, `.jld2`).
- The current branch `tc/subfilter-ape` focuses on subfilter-scale APE and multiscale energy decomposition.
- Simulation output can reach 650 GB; the scratch directory is `/glade/derecho/scratch/tomasc/khape/output/`.
