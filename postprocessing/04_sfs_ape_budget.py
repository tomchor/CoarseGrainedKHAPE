#!/usr/bin/env python
"""
Calculate SFS APE budget from Kelvin-Helmholtz simulation output
"""

#+++ Imports
import os
from pathlib import Path
import time
import xarray as xr
from dask.diagnostics.progress import ProgressBar
from aux00_utils import load_dataset_and_grid, condense_uw_velocities, integrate, make_gaussian_filter, load_energy_transfer
from aux01_pe_functions import (
    calculate_density_fields_from_buoyancy,
    calculate_b_r,
    local_potential_energies_timeseries,  # used for filtered density in loop
    calculate_sfs_ape_tendency,
    calculate_sfs_R_correction,
    calculate_sfs_ape_dissipation,
    calculate_ape_to_ke_exchange_term,
)
#---

#+++ Configuration
import argparse
parser = argparse.ArgumentParser(description="Calculate SFS APE budget from Kelvin-Helmholtz simulation output")
parser.add_argument("--filename", default="output/khi_Nz256_Ri0.10.nc",
                    help="Path to simulation NetCDF file")
parser.add_argument("--n-workers", type=int, default=18,
                    help="Number of CPU workers for APE sorting (ThreadPoolExecutor)")
args = parser.parse_args()
REPO_ROOT = Path(__file__).resolve().parent.parent
PP_OUTPUT = REPO_ROOT / "postprocessing" / "output"
filename = str(REPO_ROOT / args.filename) if not os.path.isabs(args.filename) else args.filename
n_workers = args.n_workers
#---

#+++ Load data and grid
print("\n" + "="*60)
print("Loading data and grid...")
t0 = time.time()
ds = load_dataset_and_grid(filename)
ds = ds.chunk({"time": 1})
print(f"Dataset loaded: {len(ds.time)} time steps  ({time.time()-t0:.1f}s)")
#---

#+++ Load filtered fields and pre-sorted density
print("\n" + "="*60)
print("Loading pre-filtered fields and sorted density...")

filtered_filename = str(PP_OUTPUT / (Path(filename).stem + "_filtered_velocities.nc"))
t0 = time.time()
ds_filt = xr.open_dataset(filtered_filename, decode_times=False).chunk({"time": 1})
filter_length_scales = ds_filt.filter_length_scale.values
filtered_dimensions = ["x_caa", "z_aac"]

ds = condense_uw_velocities(ds, indices=[1, 3])
ds_full = ds[["b", "dV", "LxLy", "uᵢ"]].copy()
print(f"  Pre-filtered fields loaded from: {filtered_filename}  ({time.time()-t0:.1f}s)")
print(f"  Filter length scales: {filter_length_scales}")
print(f"  Filter dimensions: x and z")

sorted_density_filename = str(PP_OUTPUT / (Path(filename).stem + "_sorted_density.nc"))
t0 = time.time()
ds_sorted = xr.open_dataset(sorted_density_filename, decode_times=False).chunk({"time": 1})
print(f"  Sorted density loaded from: {sorted_density_filename}  ({time.time()-t0:.1f}s)")
#---

#+++ Calculate scale-independent fields
print("\n" + "="*60)
print("Calculating scale-independent fields...")

t0 = time.time()
ds_full = calculate_density_fields_from_buoyancy(ds_full, buoyancy_name="b", density_name="ρ")
print(f"  ρ calculated  ({time.time()-t0:.1f}s)")

t0 = time.time()
full_local_pes = local_potential_energies_timeseries(ds_full, ds_sorted.rho_sorted, ds_sorted.dz_sorted,
                                                     density_name="ρ", n_workers=n_workers)
print(f"  full_local_pes calculated  ({time.time()-t0:.1f}s)")

t0 = time.time()
b_r = calculate_b_r(ds_full.ρ, full_local_pes.rho_sorted)
print(f"  b_r calculated  ({time.time()-t0:.1f}s)")
#---

#+++ Loop over filter scales and calculate budget terms
print("\n" + "="*60)
print("Calculating budget terms for each filter scale...")

energy_transfer = load_energy_transfer(filename)

dV = ds_full.dV
budget_list = []

for ℓ in filter_length_scales:
    print(f"\n--- filter_length_scale = {ℓ:.4f} ---")

    gaussian_filter = make_gaussian_filter(ℓ, ds)

    ds_filt_ℓ = ds_filt.sel(filter_length_scale=ℓ).drop_vars("filter_length_scale")
    ds_filt_ℓ["LxLy"] = ds["LxLy"]
    ds_filt_ℓ.attrs.update(ds.attrs)

    t0 = time.time()
    ds_filt_ℓ = calculate_density_fields_from_buoyancy(ds_filt_ℓ, buoyancy_name="b̄", density_name="ρ̄")
    print(f"  ρ̄ calculated  ({time.time()-t0:.1f}s)")

    t0 = time.time()
    filt_local_pes = local_potential_energies_timeseries(ds_filt_ℓ, full_local_pes.rho_sorted, full_local_pes.dz_sorted,
                                                         density_name="ρ̄", n_workers=n_workers)
    print(f"  filt_local_pes  ({time.time()-t0:.1f}s)")

    t0 = time.time()
    full_local_ape_filtered = gaussian_filter.apply(full_local_pes.ape, dims=filtered_dimensions)
    subfilter_local_ape = full_local_ape_filtered - filt_local_pes.ape
    print(f"  local APE filtered  ({time.time()-t0:.1f}s)")

    t0 = time.time()
    sfs_ape_dissipation = calculate_sfs_ape_dissipation(
        ds_full.ρ, full_local_pes.upsilon, filt_local_pes.upsilon, ds.κ, gaussian_filter,
        filter_dims=filtered_dimensions,
        filtered_density=ds_filt_ℓ.ρ̄,)
    print(f"  sfs_ape_dissipation  ({time.time()-t0:.1f}s)")

    t0 = time.time()
    b_r_filt = gaussian_filter.apply(b_r, dims=filtered_dimensions)
    ape_to_ke_exchange = calculate_ape_to_ke_exchange_term(
        ds_full["uᵢ"].sel(i=3),
        b_r,
        gaussian_filter,
        filter_dims=filtered_dimensions,
        filtered_w=ds_filt_ℓ["ūᵢ"].sel(i=3),
        filtered_b=b_r_filt,)
    print(f"  ape_to_ke_exchange  ({time.time()-t0:.1f}s)")

    t0 = time.time()
    R_s = calculate_sfs_R_correction(full_local_pes.rho_sorted, full_local_pes.z0, filt_local_pes.z0,
                                     full_local_pes.dz_sorted, gaussian_filter,
                                     filter_dims=filtered_dimensions)
    print(f"  R_s  ({time.time()-t0:.1f}s)")

    dAPE_dt = calculate_sfs_ape_tendency(subfilter_local_ape)

    int_dAPE_dt             = integrate(dAPE_dt, dV)
    int_sfs_ape_dissipation = integrate(sfs_ape_dissipation.reindex(time=dAPE_dt.time), dV)
    int_ape_to_ke_exchange  = integrate(ape_to_ke_exchange.reindex(time=dAPE_dt.time), dV)
    int_R_s                 = integrate(R_s.reindex(time=dAPE_dt.time), dV)

    Π_APE_ℓ     = energy_transfer["Π_APE"].sel(filter_length_scale=ℓ)
    int_Π_APE_ℓ = energy_transfer["∫Π_APE dV"].sel(filter_length_scale=ℓ)
    residual    = -int_dAPE_dt - int_ape_to_ke_exchange + int_Π_APE_ℓ.reindex(time=dAPE_dt.time) - int_sfs_ape_dissipation + int_R_s

    budget_ℓ = xr.Dataset({
        # Density fields
        "ρ̄": ds_filt_ℓ.ρ̄,
        # Reference heights
        "z₀(ρ)": full_local_pes.z0,
        "z₀(ρ̄)": filt_local_pes.z0,
        # Buoyancy displacement potentials
        "Υ": full_local_pes.upsilon,
        "Υˡ": filt_local_pes.upsilon,
        # Local APE fields
        "Ea(ρ, z)": full_local_pes.ape,
        "Ea(ρ̄, z)": filt_local_pes.ape,
        "Ēa(ρ, z)": full_local_ape_filtered,
        "Eaˢ(ρ, z)": subfilter_local_ape,
        # Local budget terms
        "∂ₜ SFS APE": dAPE_dt,
        "Π_APE": Π_APE_ℓ,
        "χₛ": sfs_ape_dissipation,
        "SFS KE->APE exchange": -ape_to_ke_exchange,
        "Rˢ": R_s,
        # Integrated budget terms
        "∫-∂ₜ SFS APE dV": -int_dAPE_dt,
        "∫Π_APE dV": int_Π_APE_ℓ,
        "∫-χₛ dV": -int_sfs_ape_dissipation,
        "∫(SFS KE->APE) dV": -int_ape_to_ke_exchange,
        "∫Rˢ dV": int_R_s,
        "residual_APE": residual,
    }).reindex(time=dAPE_dt.time)

    budget_list.append(budget_ℓ)

sfs_ape_budget_terms = xr.concat(budget_list, dim=xr.DataArray(filter_length_scales,
                                                               dims="filter_length_scale",
                                                               name="filter_length_scale"))
sfs_ape_budget_terms.attrs.update(ds.attrs)
# Scale-independent fields don't need filter_length_scale dimension
sfs_ape_budget_terms["ρ"] = ds_full.ρ
print("\nDone!")
#---

#+++ Save results
print("\n" + "="*60)
print("Saving results...")

output_filename = str(PP_OUTPUT / (Path(filename).stem + "_sfs_ape_budget.nc"))
with ProgressBar():
    sfs_ape_budget_terms.to_netcdf(output_filename)
print(f"\nResults saved to: {output_filename}")
#---
