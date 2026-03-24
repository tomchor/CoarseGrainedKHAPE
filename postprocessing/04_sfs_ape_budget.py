#!/usr/bin/env python
"""
Calculate SFS APE budget from Kelvin-Helmholtz simulation output
"""

#+++ Imports
import os
from pathlib import Path
import time
import numpy as np
import xarray as xr
from dask.diagnostics.progress import ProgressBar
from aux00_utils import load_dataset_and_grid, condense_velocities, integrate, make_gaussian_filter
from aux03_plotting import budget_colors
from aux01_pe_functions import (
    calculate_density_fields_from_buoyancy,
    local_potential_energies_timeseries,
    calculate_sfs_ape_tendency,
    calculate_sfs_R_correction,
    calculate_sfs_ape_dissipation,
    calculate_ape_to_ke_exchange_term,
)
#---

#+++ Configuration
import argparse
parser = argparse.ArgumentParser(description="Calculate SFS APE budget from Kelvin-Helmholtz simulation output")
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

t0 = time.time()
full_local_pes = local_potential_energies_timeseries(ds_full, density_name="ρ", rho_to_sort=ds_full.ρ,
                                                     ape_method="precomputed_integral",
                                                     use_numpy_version=True, n_workers=n_workers)
print(f"  full_local_pes  ({time.time()-t0:.1f}s)")
#---

#+++ Loop over filter scales and calculate budget terms
print("\n" + "="*60)
print("Calculating budget terms for each filter scale...")

energy_transfer_filename = filename.replace(".nc", "_energy_transfer.nc")
energy_transfer = xr.open_dataset(energy_transfer_filename, decode_timedelta=False).chunk({"time": 1})

dV = ds_full.dV
budget_list = []

for ℓ in filter_length_scales:
    print(f"\n--- filter_length_scale = {ℓ:.4f} ---")

    gaussian_filter = make_gaussian_filter(ℓ, ds, filter_in_2d)

    ds_filt_ℓ = ds_filt.sel(filter_length_scale=ℓ).drop_vars("filter_length_scale")
    ds_filt_ℓ["LxLy"] = ds["LxLy"]
    ds_filt_ℓ.attrs.update(ds.attrs)

    t0 = time.time()
    ds_filt_ℓ = calculate_density_fields_from_buoyancy(ds_filt_ℓ, buoyancy_name="b̄", density_name="ρ̄")
    print(f"  ρ̄ calculated  ({time.time()-t0:.1f}s)")

    t0 = time.time()
    filt_local_pes = local_potential_energies_timeseries(ds_filt_ℓ, density_name="ρ̄",
                                                         rho_to_sort=ds_full.ρ,
                                                         ape_method="precomputed_integral",
                                                         use_numpy_version=True, n_workers=n_workers)
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
    ape_to_ke_exchange = calculate_ape_to_ke_exchange_term(
        ds_full["uᵢ"].sel(i=3),
        ds_full.b,
        gaussian_filter,
        filter_dims=filtered_dimensions,
        filtered_w=ds_filt_ℓ["ūᵢ"].sel(i=3),
        filtered_b=ds_filt_ℓ["b̄"],)
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
# Scale-independent fields don't need filter_length_scale dimension
sfs_ape_budget_terms["ρ"] = ds_full.ρ
print("\nDone!")
#---

#+++ Save results
print("\n" + "="*60)
print("Saving results...")

output_filename = filename.replace(".nc", "_sfs_ape_budget.nc")
with ProgressBar():
    sfs_ape_budget_terms.to_netcdf(output_filename)
print(f"\nResults saved to: {output_filename}")

# Reload from disk so plots read pre-computed data rather than re-triggering the dask graph
sfs_ape_budget_terms = xr.open_dataset(output_filename, decode_timedelta=False)
#---

#+++ Plot integrated budget terms
print("\n" + "="*60)
print("Creating plots...")
print("="*60)

import matplotlib.pyplot as plt

integrated_vars = {
    "∫-∂ₜ SFS APE dV":    budget_colors["tendency"],
    "∫Π_APE dV":           budget_colors["flux"],
    "∫-χₛ dV":             budget_colors["dissipation"],
    "∫(SFS KE->APE) dV":   budget_colors["exchange"],
    "∫Rˢ dV":              "C4",
    "residual_APE":         budget_colors["residual"],
}

for ℓ in filter_length_scales:
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for var, color in integrated_vars.items():
        sfs_ape_budget_terms[var].sel(filter_length_scale=ℓ).dropna("time").plot.line(
            ax=ax, x="time", label=var, color=color)
    ax.legend()
    ax.set_ylabel("Budget Terms [W or J s⁻¹]")
    ax.set_title(f"Integrated SFS APE Budget Terms  (ℓ = {ℓ:.4f})")
    ax.grid(True, alpha=0.3)
    plot_filename = str(REPO_ROOT / "figures" / os.path.basename(output_filename).replace(
        ".nc", f"_l{ℓ:.4f}.png"))
    fig.savefig(plot_filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved to: {plot_filename}")
#---
