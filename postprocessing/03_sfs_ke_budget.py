#!/usr/bin/env python
#+++ Imports
import os
from pathlib import Path
import xarray as xr
from dask.diagnostics.progress import ProgressBar
from aux00_utils import load_dataset_and_grid, condense_uw_velocities, integrate, make_gaussian_filter, load_energy_transfer
from aux01_pe_functions import calculate_density_fields_from_buoyancy, calculate_b_r, calculate_b_r_simple, calculate_ape_to_ke_exchange_term
from aux02_ke_functions import (
    calculate_sfs_stress_tensor,
    calculate_strain_tensor,
    calculate_sfs_ke_dissipation,
    calculate_sfs_ke_tendency,
)
#---

#+++ Configuration
import argparse
parser = argparse.ArgumentParser(description="Calculate SFS KE budget from Kelvin-Helmholtz simulation output")
parser.add_argument("--filename", default="output/khi_Nz256_Ri0.10.nc",
                    help="Path to simulation NetCDF file")
args = parser.parse_args()
REPO_ROOT = Path(__file__).resolve().parent.parent
PP_OUTPUT = REPO_ROOT / "postprocessing" / "output"
filename = str(REPO_ROOT / args.filename) if not os.path.isabs(args.filename) else args.filename
#---

#+++ Load data and grid
print("\n" + "="*60)
print("Loading data and grid...")
ds = load_dataset_and_grid(filename)
ds = ds.chunk({"time": 1})
print(f"Dataset loaded: {len(ds.time)} time steps")
#---

#+++ Load filtered fields
print("\n" + "="*60)
print("Loading pre-filtered fields...")

filtered_filename = str(PP_OUTPUT / (Path(filename).stem + "_filtered_velocities.nc"))
ds_filt = xr.open_dataset(filtered_filename, decode_times=False).chunk({"time": 1})
filtered_dimensions = ["x_caa", "z_aac"]
filter_length_scales = ds_filt.filter_length_scale.values
tensor_dimensions = ("x_caa", "z_aac")

ds = condense_uw_velocities(ds, indices=[1, 3])
ds_full = ds[["b", "dV", "uᵢ"]].copy()

sorted_density_filename = str(PP_OUTPUT / (Path(filename).stem + "_sorted_density.nc"))
ds_sorted = xr.open_dataset(sorted_density_filename, decode_times=False).chunk({"time": 1})

print(f"Pre-filtered fields loaded from: {filtered_filename}")
print(f"Sorted density loaded from: {sorted_density_filename}")
print(f"Filter length scales: {filter_length_scales}")
print(f"Filter dimensions: x and z")
#---

#+++ Calculate density and relative buoyancy [scale-independent]
print("\n" + "="*60)
print("Calculating density and relative buoyancy...")
ds_full = calculate_density_fields_from_buoyancy(ds_full, buoyancy_name="b", density_name="ρ")
b_r = calculate_b_r(ds_full.ρ, ds_sorted.rho_sorted)
print("Done!")
#---

#+++ Calculate strain tensor of the full (unfiltered) flow  [scale-independent]
print("\n" + "="*60)
print("Calculating strain tensor of the full (unfiltered) flow...")
strain_rate_tensor = calculate_strain_tensor(ds_full["uᵢ"], dimensions=tensor_dimensions)
print("Done!")
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

    ds_filt_ℓ = ds_filt.sel(filter_length_scale=ℓ)

    # τⁱʲ = filter(uⁱ uʲ) - ūⁱ ūʲ   shape: (i, j, time, z, y, x)
    print("  SFS stress tensor...")
    sfs_stress_tensor = calculate_sfs_stress_tensor(ds_full["uᵢ"], gaussian_filter,
                                                    filter_dims=filtered_dimensions,
                                                    filtered_u_i=ds_filt_ℓ["ūᵢ"])
    i_vals = sfs_stress_tensor.coords["i"].values
    sfs_stress_tensor_trace = sum(sfs_stress_tensor.sel(i=k, j=k) for k in i_vals)
    sfs_ke_density = sfs_stress_tensor_trace / 2

    # ε<ℓ = 2ρ₀ν τ(S, S) = 2ρ₀ν Σᵢⱼ [ filter(Sⁱʲ Sⁱʲ) - filter(Sⁱʲ)² ]   [m² s⁻³]
    print("  SFS KE dissipation...")
    sfs_ke_dissipation = calculate_sfs_ke_dissipation(strain_rate_tensor, ds.ν, gaussian_filter,
                                                      filter_dims=filtered_dimensions)

    print("  APE->KE exchange term...")
    b_r_filt = gaussian_filter.apply(b_r, dims=filtered_dimensions)
    ape_to_ke_exchange = calculate_ape_to_ke_exchange_term(
        ds_full["uᵢ"].sel(i=3), # full w
        b_r,                    # relative buoyancy
        gaussian_filter,
        filter_dims=filtered_dimensions,
        filtered_w=ds_filt_ℓ["ūᵢ"].sel(i=3),
        filtered_b=b_r_filt,
    )

    # ∂KE_s/∂t   centred finite difference, staggered time grid
    dKE_dt = calculate_sfs_ke_tendency(sfs_ke_density)

    int_dKE_dt             = integrate(dKE_dt, dV)
    int_ape_to_ke_exchange = integrate(ape_to_ke_exchange.reindex(time=dKE_dt.time), dV)
    int_sfs_ke_dissipation = integrate(sfs_ke_dissipation.reindex(time=dKE_dt.time), dV)

    Π_KE_ℓ     = energy_transfer["Π_KE"].sel(filter_length_scale=ℓ)
    int_Π_KE_ℓ = energy_transfer["∫Π_KE dV"].sel(filter_length_scale=ℓ)
    residual   = -int_dKE_dt + int_Π_KE_ℓ.reindex(time=dKE_dt.time) + int_ape_to_ke_exchange - int_sfs_ke_dissipation

    budget_ℓ = xr.Dataset({
        # Local KE fields
        "KE_of_sfs_flow": sfs_ke_density,
        # Local budget terms
        "∂ₜ SFS KE": dKE_dt,
        "Π_KE": Π_KE_ℓ,
        "εₛ": sfs_ke_dissipation,
        "SFS APE->KE exchange": ape_to_ke_exchange,
        # Integrated budget terms
        "∫-∂ₜ SFS KE dV": -int_dKE_dt,
        "∫Π_KE dV": int_Π_KE_ℓ,
        "∫-εₛ dV": -int_sfs_ke_dissipation,
        "∫(SFS APE->KE) dV": int_ape_to_ke_exchange,
        "residual_KE": residual,
    }).reindex(time=dKE_dt.time)

    budget_list.append(budget_ℓ)

sfs_ke_budget_terms = xr.concat(budget_list, dim=xr.DataArray(filter_length_scales,
                                                              dims="filter_length_scale",
                                                              name="filter_length_scale"))
sfs_ke_budget_terms.attrs.update(ds.attrs)
print("\nDone!")
#---

#+++ Save results
print("\n" + "="*60)
print("Saving results...")

output_filename = str(PP_OUTPUT / (Path(filename).stem + "_sfs_ke_budget.nc"))
with ProgressBar():
    sfs_ke_budget_terms.to_netcdf(output_filename)
print(f"\nResults saved to: {output_filename}")
#---
