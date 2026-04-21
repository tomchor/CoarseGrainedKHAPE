#!/usr/bin/env python
"""
Calculate SFS APE budget from Kelvin-Helmholtz simulation output
"""

#+++ Imports
import gc
import logging
import os
from pathlib import Path
import time
import xarray as xr
from dask.diagnostics.progress import ProgressBar
from aux00_utils import load_dataset_and_grid, condense_uw_velocities, integrate, make_gaussian_filter, load_energy_transfer
from aux01_pe_functions import (
    calculate_density_fields_from_buoyancy,
    local_potential_energies_timeseries,  # used for filtered density in loop
    calculate_sfs_ape_tendency,
    calculate_sfs_R_correction,
    calculate_sfs_ape_dissipation,
)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
print = logging.info
#---

#+++ Configuration
import argparse
parser = argparse.ArgumentParser(description="Calculate SFS APE budget from Kelvin-Helmholtz simulation output")
parser.add_argument("--filename", default="output/khi_Nz256_Ri0.10.nc",
                    help="Path to simulation NetCDF file")
parser.add_argument("--n-workers", type=int, default=18,
                    help="Number of CPU workers for APE sorting (ThreadPoolExecutor)")
parser.add_argument("--fixed-reference", action="store_true", default=False,
                    help="Load the fixed-in-time reference profile (produced by 01 with --fixed-reference)")
args = parser.parse_args()
print("\n" + "="*70 + f"\n  {Path(__file__).name}\n  " + "  ".join(f"{k}={v}" for k,v in vars(args).items()) + "\n" + "="*70)
REPO_ROOT = Path(__file__).resolve().parent.parent
PP_OUTPUT = REPO_ROOT / "postprocessing" / "output"
filename = str(REPO_ROOT / args.filename) if not os.path.isabs(args.filename) else args.filename
n_workers = args.n_workers
fixed_reference = args.fixed_reference
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

ref_suffix = "_fixed_ref" if fixed_reference else ""
sorted_density_filename = str(PP_OUTPUT / (Path(filename).stem + f"_sorted_density{ref_suffix}.nc"))
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

full_local_pes_checkpoint = PP_OUTPUT / (Path(filename).stem + f"_full_local_pes_checkpoint{ref_suffix}.nc")
if full_local_pes_checkpoint.exists():
    print(f"  Loading full_local_pes from checkpoint: {full_local_pes_checkpoint.name}")
    t0 = time.time()
    full_local_pes = xr.open_dataset(str(full_local_pes_checkpoint), decode_times=False).chunk({"time": 1})
    print(f"  full_local_pes loaded  ({time.time()-t0:.1f}s)")
else:
    t0 = time.time()
    full_local_pes = local_potential_energies_timeseries(ds_full, ds_sorted.rho_sorted, ds_sorted.dz_sorted,
                                                         density_name="ρ", n_workers=n_workers)
    print(f"  full_local_pes calculated  ({time.time()-t0:.1f}s)")
    print(f"  Saving full_local_pes checkpoint...")
    t0 = time.time()
    with ProgressBar(minimum=5, dt=5):
        full_local_pes.to_netcdf(str(full_local_pes_checkpoint))
    print(f"  Checkpoint saved  ({time.time()-t0:.1f}s)")
    del full_local_pes
    gc.collect()
    full_local_pes = xr.open_dataset(str(full_local_pes_checkpoint), decode_times=False).chunk({"time": 1})
    print(f"  full_local_pes reloaded lazily")
#---

#+++ Loop over filter scales and calculate budget terms
print("\n" + "="*60)
print("Calculating budget terms for each filter scale...")

energy_transfer = load_energy_transfer(filename, ref_suffix=ref_suffix)

ke_fields_filename     = str(PP_OUTPUT / (Path(filename).stem + f"_sfs_ke_budget_fields{ref_suffix}.nc"))
ke_integrated_filename = str(PP_OUTPUT / (Path(filename).stem + f"_sfs_ke_budget_integrated{ref_suffix}.nc"))
ke_budget = xr.merge([
    xr.open_dataset(ke_fields_filename,     decode_times=False).chunk({"time": 1}),
    xr.open_dataset(ke_integrated_filename, decode_times=False).chunk({"time": 1}),
])
print(f"  KE budget loaded from: {ke_fields_filename} + {ke_integrated_filename}")

dV = ds_full.dV
budget_list = []
checkpoint_files = [full_local_pes_checkpoint]

for ℓ in filter_length_scales:
    checkpoint_path = PP_OUTPUT / (Path(filename).stem + f"_sfs_ape_budget_checkpoint_l{ℓ:.4f}{ref_suffix}.nc")
    checkpoint_files.append(checkpoint_path)

    if checkpoint_path.exists():
        print(f"\n--- filter_length_scale = {ℓ:.4f} (loading from checkpoint) ---")
        budget_list.append(xr.open_dataset(str(checkpoint_path), decode_times=False).chunk({"time": 1}))
        continue

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

    # Read APE->KE exchange term from KE budget (avoid redundant recalculation)
    ape_to_ke_exchange     = ke_budget["SFS APE->KE exchange"].sel(filter_length_scale=ℓ)
    int_ape_to_ke_exchange = ke_budget["∫(SFS APE->KE) dV"].sel(filter_length_scale=ℓ)

    t0 = time.time()
    R_s = calculate_sfs_R_correction(full_local_pes.rho_sorted, full_local_pes.z0, filt_local_pes.z0,
                                     full_local_pes.dz_sorted, gaussian_filter,
                                     filter_dims=filtered_dimensions, n_workers=n_workers)
    print(f"  R_s  ({time.time()-t0:.1f}s)")

    dAPE_dt = calculate_sfs_ape_tendency(subfilter_local_ape)

    int_dAPE_dt             = integrate(dAPE_dt, dV)
    int_sfs_ape_dissipation = integrate(sfs_ape_dissipation.reindex(time=dAPE_dt.time), dV)
    int_R_s                 = integrate(R_s.reindex(time=dAPE_dt.time), dV)

    Π_APE_ℓ     = energy_transfer["Π_APE"].sel(filter_length_scale=ℓ)
    int_Π_APE_ℓ = energy_transfer["∫Π_APE dV"].sel(filter_length_scale=ℓ)
    residual    = -int_dAPE_dt - int_ape_to_ke_exchange.reindex(time=dAPE_dt.time) + int_Π_APE_ℓ.reindex(time=dAPE_dt.time) - int_sfs_ape_dissipation + int_R_s

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

    print(f"  Saving checkpoint...")
    t0 = time.time()
    with ProgressBar(minimum=5, dt=5):
        budget_ℓ.to_netcdf(str(checkpoint_path))
    print(f"  Checkpoint saved  ({time.time()-t0:.1f}s)")

    # Free memory before the next iteration
    del ds_filt_ℓ, filt_local_pes, full_local_ape_filtered, subfilter_local_ape
    del sfs_ape_dissipation, R_s, dAPE_dt, budget_ℓ
    del ape_to_ke_exchange, int_ape_to_ke_exchange
    del int_dAPE_dt, int_sfs_ape_dissipation, int_R_s
    del Π_APE_ℓ, int_Π_APE_ℓ, residual
    gc.collect()

    budget_list.append(xr.open_dataset(str(checkpoint_path), decode_times=False).chunk({"time": 1}))

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

integrated_vars = [v for v in sfs_ape_budget_terms.data_vars if v.startswith("∫") or "residual" in v]
local_vars      = [v for v in sfs_ape_budget_terms.data_vars if v not in integrated_vars]

fields_filename     = str(PP_OUTPUT / (Path(filename).stem + f"_sfs_ape_budget_fields{ref_suffix}.nc"))
integrated_filename = str(PP_OUTPUT / (Path(filename).stem + f"_sfs_ape_budget_integrated{ref_suffix}.nc"))

print("  Saving local fields...")
with ProgressBar(minimum=5, dt=5):
    sfs_ape_budget_terms[local_vars].to_netcdf(fields_filename)
print(f"  Fields saved to:     {fields_filename}")

print("  Saving integrated timeseries...")
with ProgressBar(minimum=5, dt=5):
    sfs_ape_budget_terms[integrated_vars].to_netcdf(integrated_filename)
print(f"  Integrated saved to: {integrated_filename}")

print("\nDeleting intermediate checkpoint files...")
for f in checkpoint_files:
    f.unlink(missing_ok=True)
    print(f"  Deleted: {f.name}")
#---
