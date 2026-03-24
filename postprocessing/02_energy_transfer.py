#!/usr/bin/env python
#+++ Imports
import os
from pathlib import Path
import time
import numpy as np
import xarray as xr
from dask.diagnostics.progress import ProgressBar
from aux00_utils import load_dataset_and_grid, condense_velocities, integrate, make_gaussian_filter
from aux01_pe_functions import (
    calculate_density_fields_from_buoyancy,
    local_potential_energies_timeseries,
    calculate_cross_scale_ape_flux,
)
from aux02_ke_functions import (
    calculate_sfs_stress_tensor,
    calculate_strain_tensor,
    calculate_cross_scale_ke_flux,
)
#---

#+++ Configuration
import argparse
parser = argparse.ArgumentParser(description="Calculate cross-scale KE and APE transfer terms")
parser.add_argument("--filename", default="output/khi_128x1x256.nc",
                    help="Path to simulation NetCDF file")
parser.add_argument("--n-workers", type=int, default=18,
                    help="Number of CPU workers for APE sorting (ThreadPoolExecutor)")
args = parser.parse_args()
REPO_ROOT = Path(__file__).resolve().parent.parent
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

#+++ Load filtered fields
print("\n" + "="*60)
print("Loading pre-filtered fields...")

filtered_dimensions = ["x_caa", "y_aca"]

ds = condense_velocities(ds, indices=[1, 2, 3])
ds_full = ds[["b", "dV", "LxLy", "uᵢ"]].copy()

filtered_filename = filename.replace(".nc", "_filtered_velocities.nc")
t0 = time.time()
ds_filt = xr.open_dataset(filtered_filename, decode_times=False).chunk({"time": 1})
filter_length_scales = ds_filt.filter_length_scale.values
filter_in_2d = int(ds_filt.attrs.get("filter_ndim", 2)) == 2
print(f"  Pre-filtered fields loaded from: {filtered_filename}  ({time.time()-t0:.1f}s)")
print(f"  Filter length scales: {filter_length_scales}")
print(f"  Filter dimensions: {'2D (x,y)' if filter_in_2d else '1D (x only)'}")
#---

#+++ Calculate scale-independent fields
print("\n" + "="*60)
print("Calculating scale-independent fields...")

t0 = time.time()
ds_full = calculate_density_fields_from_buoyancy(ds_full, buoyancy_name="b", density_name="ρ")
print(f"  ρ calculated  ({time.time()-t0:.1f}s)")

# Full strain tensor is scale-independent
t0 = time.time()
strain_rate_tensor = calculate_strain_tensor(ds_full["uᵢ"])
print(f"  Full strain tensor calculated  ({time.time()-t0:.1f}s)")
#---

#+++ Loop over filter scales
print("\n" + "="*60)
print("Calculating cross-scale transfer terms for each filter scale...")

dV = ds_full.dV
transfer_list = []

for ℓ in filter_length_scales:
    print(f"\n--- filter_length_scale = {ℓ:.4f} ---")

    gaussian_filter = make_gaussian_filter(ℓ, ds, filter_in_2d)

    ds_filt_ℓ = ds_filt.sel(filter_length_scale=ℓ).drop_vars("filter_length_scale")
    ds_filt_ℓ["LxLy"] = ds["LxLy"]
    ds_filt_ℓ.attrs.update(ds.attrs)

    # --- KE cross-scale transfer ---
    # τⁱʲ = filter(uⁱuʲ) - ūⁱūʲ
    t0 = time.time()
    sfs_stress_tensor = calculate_sfs_stress_tensor(ds_full["uᵢ"], gaussian_filter,
                                                    filter_dims=filtered_dimensions,
                                                    filtered_u_i=ds_filt_ℓ["ūᵢ"])
    strain_rate_tensor_l = calculate_strain_tensor(ds_filt_ℓ["ūᵢ"])
    # Π_KE = -τⁱʲ : S̄ⁱʲ
    Π_KE = calculate_cross_scale_ke_flux(sfs_stress_tensor, strain_rate_tensor_l)
    print(f"  Π_KE  ({time.time()-t0:.1f}s)")

    # --- APE cross-scale transfer ---
    # Compute ρ̄ and the large-scale reference state z₀(ρ̄) → Υˡ
    t0 = time.time()
    ds_filt_ℓ = calculate_density_fields_from_buoyancy(ds_filt_ℓ, buoyancy_name="b̄", density_name="ρ̄")
    filt_local_pes = local_potential_energies_timeseries(ds_filt_ℓ, density_name="ρ̄",
                                                         rho_to_sort=ds_full.ρ,
                                                         ape_method="precomputed_integral",
                                                         use_numpy_version=True, n_workers=n_workers)
    # Π_APE = -(filter(ρuᵢ) - ρ̄ūᵢ) · ∇Υˡ
    Π_APE = calculate_cross_scale_ape_flux(ds_full.ρ, ds_full["uᵢ"], filt_local_pes.upsilon,
                                            gaussian_filter, filter_dims=filtered_dimensions,
                                            filtered_density=ds_filt_ℓ.ρ̄,
                                            filtered_velocity_vector=ds_filt_ℓ["ūᵢ"])
    print(f"  Π_APE  ({time.time()-t0:.1f}s)")

    # --- Integrated transfer terms ---
    int_Π_KE  = integrate(Π_KE, dV)
    int_Π_APE = integrate(Π_APE, dV)

    transfer_ℓ = xr.Dataset({
        # Local transfer terms
        "Π_KE":  Π_KE,
        "Π_APE": Π_APE,
        # Integrated transfer terms
        "∫Π_KE dV":  int_Π_KE,
        "∫Π_APE dV": int_Π_APE,
    })
    transfer_list.append(transfer_ℓ)

scale_coord = xr.DataArray(filter_length_scales, dims="filter_length_scale",
                           name="filter_length_scale")
energy_transfer = xr.concat(transfer_list, dim=scale_coord)
print("\nDone!")
#---

#+++ Save results
print("\n" + "="*60)
print("Saving results...")

output_filename = filename.replace(".nc", "_energy_transfer.nc")
with ProgressBar():
    energy_transfer.to_netcdf(output_filename)
print(f"Results saved to: {output_filename}")
#---
