# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

KHAPE (Kelvin-Helmholtz Available Potential Energy) computes Available Potential Energy (APE) from Kelvin-Helmholtz instability simulations using the Winters et al. (1995) sorting method. The pipeline is:
1. **Julia simulation** (Oceananigans.jl on GPU) -> NetCDF output
2. **Python post-processing** -> filter fields, sort density, compute energy transfer and SFS budgets, plot

GitHub remote: `git@github.com:tomchor/CoarseGrainedKHAPE.git`

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

The Julia simulation accepts CLI args: `--Nz`, `--Ri`, `--stop_time`, `--Re0`, `--Pr`, `--U`, `--h`, `--perturbation_amplitude`, and `--save_tensors` (flag; also writes the per-scale strain/stress tensor components for online-vs-offline validation). For local CPU development:
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
bash 00_get_budgets.sh output/khi_Nz512_Ri0.10.nc --filter-scales 1 7
bash 00_get_budgets.sh output/khi_Nz512_Ri0.10.nc --filter-scales 1 7 --fixed-reference
```
Set `N_WORKERS` env var to control parallelism (default 1): `N_WORKERS=4 bash 00_get_budgets.sh ...`

### Running tests
```bash
pytest tests/ -v -s                                  # default (time-varying reference)
pytest tests/ -v -s --ref-suffix _fixed_ref          # fixed reference variant
pytest tests/test_gaussian_filter.py -v -s           # single file (filter unit tests, no pipeline output needed)
pytest "tests/test_budgets.py::test_ke_budget_residual" -v -s   # single test
```
`test_budgets.py` checks SFS KE and APE budget closure (rms(residual)/min(rms(terms)) < 10%) and **requires the post-processing output** in `postprocessing/output/` for `khi_Nz512_Ri0.10` (run `00_get_budgets.sh` first; it is parametrized per filter scale via `conftest.py`). `test_filter.py` / `test_gaussian_filter.py` are self-contained unit tests of the offline `GaussianFilter` and need no pipeline output.

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
| `03_energy_transfer.py` | Cross-scale APE transfer Π_A and APE↔KE exchange (Π_K is computed online — see Data flow) |
| `04_sfs_ke_budget.py` | Sub-filter-scale KE budget terms |
| `05_sfs_ape_budget.py` | Sub-filter-scale APE budget terms |
| `06_plot_budgets.py` | Plot budget time series |

`sweep*` scripts are the sweep variant (parameter sweep over filter scales): `sweep1_filter_fields.py` filters, `sweep2_energy_transfer.py` computes transfer, `sweep3_plot_transfer_spectrum.py` plots spectra.

`postprocessing/validation/` holds the online-vs-offline comparison scripts, each of which recomputes a quantity offline and compares it against the simulation's online output: `inv01_compare_filters.py` (filtered fields), `inv02_compare_ke_transfer.py` (Π_K), `inv03_compare_tensor.py` (the S̄/τ tensor components), and `inv05_compare_dissipation.py` (SFS KE dissipation ε_Kˢ). `inv04_animate_comparison.py` makes the animated online | offline | difference version for a chosen `--field` (Π_K, ε_Ks, or a filtered field). `validation.pbs` runs them all. Only the tensor-component comparison (`inv03`) needs a `--save_tensors` run; Π_K and ε_Kˢ are always written.

Standalone visualization scripts (not part of the numbered pipeline):
- `plot1_panels.py` -- 4-panel snapshot of local SFS budget fields
- `plot2_budgets.py` -- 2x2 panel of SFS KE and APE budget time series
- `plot3_plot_transfer_spectrum.py` -- cross-scale transfer spectra
- `anim1_panels.py` -- animated version of plot1 panels (requires ffmpeg)

Shared utilities:
- `aux00_utils.py` -- data loading (`load_dataset_and_grid`), filtering (`filter_fields`, `GaussianFilter`, `DaskParallelFilter`), domain padding (`_pad_domain_in_z`), spatial derivatives (`calculate_gradient`), tensor condensing (`condense_velocities`)
- `aux01_pe_functions.py` -- density sorting (`sorted_timeseries`), potential energy calculations (`local_potential_energies_timeseries`), APE budget terms (SFS flux tensor, cross-scale APE flux, SFS APE dissipation, reference-tendency correction R)
- `aux02_ke_functions.py` -- SFS stress tensor, strain rate tensor, cross-scale KE flux, SFS KE dissipation, full energy transfer pipeline (`calculate_energy_transfer`, with an `include_pi_k` flag to skip Π_K)
- `aux03_plotting.py` -- plotting helpers (`budget_colors`, `run_label`, `plot_sfs_budget`)

All post-processing scripts accept `--filename`, `--filter-scales`, `--n-workers`, `--fixed-reference` via argparse. Output goes to `postprocessing/output/`.

### Data flow between pipeline steps

The sorted density (`*_sorted_density.nc`) produced by step 02 is reused by steps 03, 05, and the sweep pipeline (`sweep2_energy_transfer.py`), avoiding redundant sorts. When `--fixed-reference` is used, output files are suffixed `_fixed_ref`. The sweep's `sweep2_energy_transfer.py` with `--fixed-reference` expects the sorted density from the budget pipeline's step 02 to already exist.

The cross-scale KE transfer **Π_K and the SFS KE dissipation ε_Kˢ are computed online** by the Julia simulation (`kelvin_helmholtz_instability.jl`, output as `Π_K_ℓ<ℓ>` / `ε_Ks_ℓ<ℓ>`). To avoid recomputing them offline, `03_energy_transfer.py` runs with `include_pi_k=False` (Π_A + exchange only) and `04_sfs_ke_budget.py` reads Π_K and ε_Kˢ directly from the simulation output (it still computes the SFS-KE density/tendency and the APE↔KE exchange offline — the exchange needs the sorted reference state). The budget filter scales must therefore match the simulation's online `filter_ℓs` (default `(1, 7)`); the offline-recompute paths are kept under `validation/` for cross-checking (`inv02` for Π_K, `inv05` for ε_Kˢ).

### Julia layer
- `kelvin_helmholtz_instability.jl` -- main simulation (Oceananigans.jl `NonhydrostaticModel`, `Centered(order=4)` advection, adaptive timestep via `TimeStepWizard`)
- `utils.jl` -- `closest_factor_number()` (FFT-friendly grid sizes), `show_gpu_status()`
- Setup: shear flow u(z)=U·tanh(z/h), stratification b(z)=B₀·tanh(z/h) with B₀=U²·Ri/h; perturbation seeded on w. Defaults U=1, Ri=0.1, h=1
- Domain: Lx=λ_max (the most-unstable KH wavelength ≈14.1h), Ly=λ_max/3, Lz=25h; topology (Periodic, Periodic, Bounded). `y_aspect_ratio=Inf` ⇒ **Ny=1**, so runs are effectively 2D in x–z (v ≡ 0)
- Output: `output/khi_Nz<Nz>_Ri<Ri>.nc` (3D fields, Float64, consecutive-iteration pairs for time derivatives) and `output/khi_Nz<Nz>_Ri<Ri>_2d.nc` (x–z slice, Float32)

#### Online cross-scale diagnostics
The simulation computes, at each scale in `filter_ℓs = (1, 7)`, the sub-filter quantities the offline pipeline would otherwise recompute in Python — so they are produced once, on the GPU, and read back later (see Data flow):
- Filtered fields (Oceanostics `GaussianFilter`), the cross-scale KE flux Πₖ = −τⁱʲ S̄ⁱʲ (`KineticEnergyCrossScaleFlux`), and the SFS KE dissipation ε_Kˢ = filter(ε) − ε̄ (where `ε` is the total viscous dissipation `KineticEnergyEquation.DissipationRate` and `ε̄` is `CoarseGrainedKineticEnergyDissipationRate`, the filtered-flow dissipation). For validation only, the resolved strain rate S̄ⁱʲ (`StrainRateTensor`) and sub-filter stress τⁱʲ = filter(uⁱuʲ) − ūⁱūʲ (`subfilter_stress_tensor`) components are also emitted.
- The Gaussian filter is configured to **match the offline filter exactly**: periodic x, edge-extended bounded z, stencil truncated at 4σ (matching scipy `gaussian_filter1d`'s default `truncate=4`; Oceanostics defaults to 2σ). Only i,j ∈ {1,3} are kept (2D x–z).
- `Π_K_ℓ<ℓ>` and `ε_Ks_ℓ<ℓ>` (and their volume integrals) are always written and read back by `04_sfs_ke_budget.py`; the individual S̄/τ components (`S11/S33/S13_ℓ<ℓ>`, `tau11/tau33/tau13_ℓ<ℓ>`) are gated behind `--save_tensors` and consumed only by `postprocessing/validation/`.
- Requires the Oceanostics `tc/sfs-dissipation` branch (PR #259), pinned via `repo-rev` in `Manifest.toml`, which provides `GaussianFilter`, `StrainRateTensor`, `subfilter_stress_tensor`, `KineticEnergyCrossScaleFlux`, and `CoarseGrainedKineticEnergyDissipationRate`.

### Key dependencies
- **Python**: `numpy`, `xarray`, `scipy`, `matplotlib`, `dask`, `gcm_filters`, `netcdf4`
- **Julia**: `Oceananigans`, `Oceanostics`, `CUDA`, `NCDatasets`, `CairoMakie`

## Physics Reference

- **TPE** = integral of g*rho*z dV  (total potential energy)
- **RPE** = minimum PE achievable by adiabatic rearrangement (from sorted reference state)
- **APE** = TPE - RPE  (available for conversion to KE)
- **Π_K**, **Π_A** -- cross-scale energy transfer (sub-filter to resolved)
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
