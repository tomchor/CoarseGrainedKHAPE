#!/usr/bin/env python
#+++ Imports
import os
from pathlib import Path
import xarray as xr
from dask.diagnostics.progress import ProgressBar
from src.aux00_utils import load_dataset_and_grid, condense_uw_velocities, integrate, make_gaussian_filter
from src.aux01_pe_functions import calculate_density_fields_from_buoyancy, calculate_b_r, calculate_b_r_simple, calculate_ape_to_ke_exchange_term
from src.aux02_ke_functions import (
    calculate_sfs_stress_tensor,
    calculate_sfs_ke_tendency,
)
#---

#+++ Configuration
import argparse
parser = argparse.ArgumentParser(description="Calculate SFS KE budget from Kelvin-Helmholtz simulation output")
parser.add_argument("--filename", default="output/khi_Nz2048_Ri0.10.nc", help="Path to simulation NetCDF file")
parser.add_argument("--fixed-reference", action="store_true", default=False, help="Load the fixed-in-time reference profile (produced by 01 with --fixed-reference)")
args = parser.parse_args()
print("\n" + "="*70 + f"\n  {Path(__file__).name}\n  " + "  ".join(f"{k}={v}" for k,v in vars(args).items()) + "\n" + "="*70)
REPO_ROOT = Path(__file__).resolve().parent.parent
PP_OUTPUT = REPO_ROOT / "postprocessing" / "output"
filename = str(REPO_ROOT / args.filename) if not os.path.isabs(args.filename) else args.filename
fixed_reference = args.fixed_reference
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
filter_scales = ds_filt.filter_scale.values
tensor_dimensions = ("x_caa", "z_aac")

ds = condense_uw_velocities(ds, indices=[1, 3])
ds_full = ds[["b", "dV", "uᵢ"]].copy()

ref_suffix = "_fixed_ref" if fixed_reference else ""
sorted_density_filename = str(PP_OUTPUT / (Path(filename).stem + f"_sorted_density{ref_suffix}.nc"))
ds_sorted = xr.open_dataset(sorted_density_filename, decode_times=False).chunk({"time": 1})

print(f"Pre-filtered fields loaded from: {filtered_filename}")
print(f"Sorted density loaded from: {sorted_density_filename}")
print(f"Filter length scales: {filter_scales}")
print(f"Filter dimensions: x and z")
#---

#+++ Calculate density and relative buoyancy [scale-independent]
print("\n" + "="*60)
print("Calculating density and relative buoyancy...")
ds_full = calculate_density_fields_from_buoyancy(ds_full, buoyancy_name="b", density_name="ρ")
b_r = calculate_b_r(ds_full.ρ, ds_sorted.rho_sorted)
print("Done!")
#---

#+++ Loop over filter scales and calculate budget terms
print("\n" + "="*60)
print("Calculating budget terms for each filter scale...")

# Π_K and ε_Kˢ are computed online by the simulation (no offline recompute). Map a filter scale to
# its sim-output variable name, matching the Julia Symbol("<var>_ℓ$(ℓ)"). Both are reference-
# independent, so the same online fields serve the time-varying and fixed-reference budgets.
def online_name(var, ℓ):
    return f"{var}_ℓ{int(ℓ)}" if float(ℓ) == int(ℓ) else f"{var}_ℓ{ℓ}"

dV = ds_full.dV
budget_list = []

for ℓ in filter_scales:
    print(f"\n--- filter_scale = {ℓ:.4f} ---")

    gaussian_filter = make_gaussian_filter(ℓ, ds)

    ds_filt_ℓ = ds_filt.sel(filter_scale=ℓ)

    # τⁱʲ = filter(uⁱ uʲ) - ūⁱ ūʲ   shape: (i, j, time, z, y, x)
    print("  SFS stress tensor...")
    sfs_stress_tensor = calculate_sfs_stress_tensor(ds_full["uᵢ"], gaussian_filter,
                                                    filter_dims=filtered_dimensions,
                                                    filtered_u_i=ds_filt_ℓ["ūᵢ"])
    i_vals = sfs_stress_tensor.coords["i"].values
    sfs_stress_tensor_trace = sum(sfs_stress_tensor.sel(i=k, j=k) for k in i_vals)
    sfs_ke_density = sfs_stress_tensor_trace / 2

    print("  APE->KE exchange term...")
    # b_r_l = -(g/ρ₀)(ρ̄ - ρ_ref): filtered density minus the unfiltered reference profile
    # (cf. filter(b_r) = -(g/ρ₀)(ρ̄ - filter(ρ_ref)), which filters the reference too)
    b_r_l = calculate_b_r(gaussian_filter.apply(ds_full.ρ, dims=filtered_dimensions), ds_sorted.rho_sorted)
    ape_to_ke_exchange = calculate_ape_to_ke_exchange_term(
        ds_full["uᵢ"].sel(i=3), # full w
        b_r,                    # relative buoyancy
        gaussian_filter,
        filter_dims=filtered_dimensions,
        filtered_w=ds_filt_ℓ["ūᵢ"].sel(i=3),
        filtered_b=b_r_l,
    )

    # ∂KE_s/∂t   centred finite difference, staggered time grid
    dKE_dt = calculate_sfs_ke_tendency(sfs_ke_density)

    int_dKE_dt             = integrate(dKE_dt, dV)
    int_ape_to_ke_exchange = integrate(ape_to_ke_exchange.reindex(time=dKE_dt.time), dV)

    # Π_K and ε_Kˢ from the online sim output, integrated on the budget (padded) grid so they are
    # consistent with the other terms (the padding region is ≈0 for both).
    for var in ("Π_K", "ε_Ks"):
        if online_name(var, ℓ) not in ds:
            raise KeyError(f"Online field '{online_name(var, ℓ)}' not in sim output; the budget "
                           f"filter scales must match the simulation's online filter_ℓs (got ℓ={ℓ}).")
    Π_K_ℓ              = ds[online_name("Π_K", ℓ)]
    sfs_ke_dissipation = ds[online_name("ε_Ks", ℓ)]
    int_Π_K_ℓ              = integrate(Π_K_ℓ, dV)
    int_sfs_ke_dissipation = integrate(sfs_ke_dissipation, dV)
    residual  = (-int_dKE_dt + int_Π_K_ℓ.reindex(time=dKE_dt.time) + int_ape_to_ke_exchange
                 - int_sfs_ke_dissipation.reindex(time=dKE_dt.time))

    budget_ℓ = xr.Dataset({
        # Local KE fields
        "KE_of_sfs_flow": sfs_ke_density,
        # Local budget terms
        "∂ₜ SFS KE": dKE_dt,
        "Π_K": Π_K_ℓ,
        "ε_Kˢ": sfs_ke_dissipation,
        "SFS APE->KE exchange": ape_to_ke_exchange,
        # Integrated budget terms
        "∫-∂ₜ SFS KE dV": -int_dKE_dt,
        "∫Π_K dV": int_Π_K_ℓ,
        "∫-ε_Kˢ dV": -int_sfs_ke_dissipation,
        "∫(SFS APE->KE) dV": int_ape_to_ke_exchange,
        "residual_K": residual,
    }).reindex(time=dKE_dt.time)

    budget_list.append(budget_ℓ)

sfs_ke_budget_terms = xr.concat(budget_list, dim=xr.DataArray(filter_scales,
                                                              dims="filter_scale",
                                                              name="filter_scale"))
sfs_ke_budget_terms.attrs.update(ds.attrs)
print("\nDone!")
#---

#+++ Save results
print("\n" + "="*60)
print("Saving results...")

integrated_vars = [v for v in sfs_ke_budget_terms.data_vars if v.startswith("∫") or "residual" in v]
local_vars      = [v for v in sfs_ke_budget_terms.data_vars if v not in integrated_vars]

fields_filename     = str(PP_OUTPUT / (Path(filename).stem + f"_sfs_ke_budget_fields{ref_suffix}.nc"))
integrated_filename = str(PP_OUTPUT / (Path(filename).stem + f"_sfs_ke_budget_integrated{ref_suffix}.nc"))

print("  Saving local fields...")
with ProgressBar(minimum=5, dt=5):
    sfs_ke_budget_terms[local_vars].to_netcdf(fields_filename)
print(f"  Fields saved to:     {fields_filename}")

print("  Saving integrated timeseries...")
with ProgressBar(minimum=5, dt=5):
    sfs_ke_budget_terms[integrated_vars].to_netcdf(integrated_filename)
print(f"  Integrated saved to: {integrated_filename}")
#---
