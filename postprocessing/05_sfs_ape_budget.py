#!/usr/bin/env python
"""
Calculate SFS APE budget from baroclinic adjustment simulation output
"""

#+++ Imports
import gc
import logging
import os
from pathlib import Path
import time
import xarray as xr
from dask.diagnostics.progress import ProgressBar
from src.aux00_utils import load_dataset_and_grid, condense_velocities, integrate, make_gaussian_filter, load_energy_transfer, write_dataset
from src.aux01_pe_functions import (
    calculate_density_fields_from_buoyancy,
    local_potential_energies_timeseries,  # used for full_local_pes, and as a fallback for filtered density
    calculate_sfs_ape_tendency,
    calculate_sfs_R_correction,
    calculate_sfs_ape_dissipation,
)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
print = logging.info
#---

#+++ Configuration
import argparse
parser = argparse.ArgumentParser(description="Calculate SFS APE budget from baroclinic adjustment simulation output")
parser.add_argument("--filename", default="output/bci_Nx48_Ny48_Nz8.nc", help="Path to simulation NetCDF file")
parser.add_argument("--n-workers", type=int, default=18, help="Number of CPU workers for APE sorting (ThreadPoolExecutor)")
parser.add_argument("--fixed-reference", action="store_true", default=False, help="Load the fixed-in-time reference profile (produced by 01 with --fixed-reference)")
parser.add_argument("--write-mode", choices=["load", "synchronous"], default="load",
    help="How to avoid the dask-lazy .to_netcdf() write hang -- see write_dataset() in aux00_utils.py for "
         "what each mode does and the measured cost of 'synchronous' relative to 'load'. Only affects the "
         "per-scale checkpoint and final result writes below; the full_local_pes checkpoint is already fully "
         "eager by construction (see local_potential_energies_timeseries()) and unaffected by this flag.")
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

# --implicit (baroclinic_adjustment.jl) means closure=nothing: nu_h/nu_v/Pr-derived kh/kv below are
# identically zero, so ε_Aˢ (like ε_Kˢ in 04_sfs_ke_budget.py) is physically uninformative -- real
# dissipation is happening via WENO's own implicit numerics, invisible to a diagnostic that only reads an
# explicit closure's diffusivity. The domain-integrated ε_Aˢ is replaced below with a residual-based
# estimate (solves the budget for ε_Aˢ assuming every other term is correct); the local (spatial) ε_Aˢ
# field is left as-is (still the near-zero, uninformative explicit-closure value) since a local residual
# would also absorb real spatial transport/flux-divergence terms that only vanish upon domain integration,
# not pointwise -- there's no principled way to attribute those to "dissipation" at a single grid cell.
implicit = bool(ds.attrs.get("implicit", 0))
if implicit:
    print("  implicit=True (from simulation attrs): domain-integrated ε_Aˢ will be replaced with a "
          "residual-based estimate; the local (spatial) ε_Aˢ field stays at its uninformative explicit-closure value")
#---

#+++ Load filtered fields and pre-sorted density
print("\n" + "="*60)
print("Loading pre-filtered fields and sorted density...")

filtered_filename = str(PP_OUTPUT / (Path(filename).stem + "_filtered_velocities.nc"))
t0 = time.time()
ds_filt = xr.open_dataset(filtered_filename, decode_times=False).chunk({"time": 1})
filter_scales = ds_filt.filter_scale.values
filtered_dimensions = ["x_caa", "y_aca"]

# Diffusivities κh, κv: with the 'smagorinsky' closure the simulation writes a single diagnostic eddy
# diffusivity κₑ (spatially/temporally varying, isotropic), so the same field is used for both. With
# 'constant'/'scale_aware' closures, κh/κv are fixed scalars from the nu_h/nu_v & Pr global attributes.
# calculate_sfs_ape_dissipation() weights the horizontal and vertical parts of ∇ρ·∇Υ separately by κh,
# κv -- essential once the closure is anisotropic (κh ≫ κv for 'scale_aware'), since ∂ρ/∂z (dominated
# by the background stratification) would otherwise get multiplied by the wrong (much larger) κh.
if "κₑ" in ds:
    κh = κv = ds["κₑ"]
else:
    κh, κv = ds.attrs["nu_h"] / ds.attrs["Pr"], ds.attrs["nu_v"] / ds.attrs["Pr"]

ds = condense_velocities(ds, indices=(1, 2, 3))
ds_full = ds[["b", "dV", "LxLy", "uᵢ"]].copy()

print(f"  Pre-filtered fields loaded from: {filtered_filename}  ({time.time()-t0:.1f}s)")
print(f"  Filter length scales: {filter_scales}")
print(f"  Filter dimensions: x and y (horizontal)")

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

for ℓ in filter_scales:
    checkpoint_path = PP_OUTPUT / (Path(filename).stem + f"_sfs_ape_budget_checkpoint_l{ℓ:.4f}{ref_suffix}.nc")
    checkpoint_files.append(checkpoint_path)

    if checkpoint_path.exists():
        print(f"\n--- filter_scale = {ℓ:.4f} (loading from checkpoint) ---")
        budget_list.append(xr.open_dataset(str(checkpoint_path), decode_times=False).chunk({"time": 1}))
        continue

    print(f"\n--- filter_scale = {ℓ:.4f} ---")

    gaussian_filter = make_gaussian_filter(ℓ, ds)

    ds_filt_ℓ = ds_filt.sel(filter_scale=ℓ).drop_vars("filter_scale")
    ds_filt_ℓ["LxLy"] = ds["LxLy"]
    ds_filt_ℓ.attrs.update(ds.attrs)

    t0 = time.time()
    ds_filt_ℓ = calculate_density_fields_from_buoyancy(ds_filt_ℓ, buoyancy_name="b̄", density_name="ρ̄")
    print(f"  ρ̄ calculated  ({time.time()-t0:.1f}s)")

    # z₀(ρ̄)/Υˡ/Dˡ/Ea(ρ̄, z) (local_potential_energies_timeseries() on ds_filt_ℓ) are the same, expensive,
    # per-timestep computation that 03_energy_transfer.py's calculate_energy_transfer() already does
    # internally for this exact scale (same ds_filt_ℓ, same reference rho_sorted/dz_sorted) -- it normally
    # discards everything but Υˡ once Π_A is built, but with include_filt_local_pes=True it persists the
    # rest to energy_transfer too, so read them back here instead of recomputing. Falls back to computing
    # it directly if energy_transfer predates that flag (e.g. rerunning 05 alone against an older 03 output).
    if all(v in energy_transfer for v in ("z₀(ρ̄)", "Υˡ", "Dˡ", "Ea(ρ̄, z)")):
        filt_z0      = energy_transfer["z₀(ρ̄)"].sel(filter_scale=ℓ, method="nearest", tolerance=1e-6)
        filt_upsilon = energy_transfer["Υˡ"].sel(filter_scale=ℓ, method="nearest", tolerance=1e-6)
        filt_D       = energy_transfer["Dˡ"].sel(filter_scale=ℓ, method="nearest", tolerance=1e-6)
        filt_ape     = energy_transfer["Ea(ρ̄, z)"].sel(filter_scale=ℓ, method="nearest", tolerance=1e-6)
    else:
        print("  energy_transfer output predates include_filt_local_pes -- recomputing "
              "local_potential_energies_timeseries() on ds_filt_ℓ (rerun 03_energy_transfer.py to avoid this)")
        t0 = time.time()
        filt_local_pes = local_potential_energies_timeseries(ds_filt_ℓ, full_local_pes.rho_sorted, full_local_pes.dz_sorted,
                                                             density_name="ρ̄", n_workers=n_workers)
        print(f"  filt_local_pes  ({time.time()-t0:.1f}s)")
        filt_z0, filt_upsilon, filt_D, filt_ape = filt_local_pes.z0, filt_local_pes.upsilon, filt_local_pes.D, filt_local_pes.ape
        del filt_local_pes

    t0 = time.time()
    full_local_ape_filtered = gaussian_filter.apply(full_local_pes.ape, dims=filtered_dimensions)
    subfilter_local_ape = full_local_ape_filtered - filt_ape
    print(f"  local APE filtered  ({time.time()-t0:.1f}s)")

    t0 = time.time()
    # ∇Υ, ∇Υˡ are computed by differentiating the assembled Υ/Υˡ fields directly, using a 4th-order
    # stencil matching the simulation's own advection scheme -- see calculate_sfs_ape_dissipation()'s
    # docstring for why (reverted from the analytic D(ρ)-based reconstruction).
    sfs_ape_dissipation = calculate_sfs_ape_dissipation(
        ds_full.ρ, full_local_pes.upsilon, filt_upsilon, κh, κv, gaussian_filter,
        filter_dims=filtered_dimensions,
        filtered_density=ds_filt_ℓ.ρ̄,)
    print(f"  sfs_ape_dissipation  ({time.time()-t0:.1f}s)")

    # Read APE->KE exchange term from KE budget (avoid redundant recalculation)
    ape_to_ke_exchange     = ke_budget["SFS APE->KE exchange"].sel(filter_scale=ℓ, method="nearest", tolerance=1e-6)
    int_ape_to_ke_exchange = ke_budget["∫(SFS APE->KE) dV"].sel(filter_scale=ℓ, method="nearest", tolerance=1e-6)

    t0 = time.time()
    R_s = calculate_sfs_R_correction(full_local_pes.rho_sorted, full_local_pes.z0, filt_z0,
                                     full_local_pes.dz_sorted, gaussian_filter,
                                     filter_dims=filtered_dimensions, n_workers=n_workers)
    print(f"  R_s  ({time.time()-t0:.1f}s)")

    dAPE_dt = calculate_sfs_ape_tendency(subfilter_local_ape)

    int_dAPE_dt             = integrate(dAPE_dt, dV)
    int_sfs_ape_dissipation = integrate(sfs_ape_dissipation.reindex(time=dAPE_dt.time), dV)
    int_R_s                 = integrate(R_s.reindex(time=dAPE_dt.time), dV)

    Π_A_ℓ     = energy_transfer["Π_A"].sel(filter_scale=ℓ, method="nearest", tolerance=1e-6)
    int_Π_A_ℓ = energy_transfer["∫Π_A dV"].sel(filter_scale=ℓ, method="nearest", tolerance=1e-6)

    if implicit:
        # Solve the budget for ε_Aˢ assuming every other term (Π_A, ∂ₜE_A^s, exchange, Rˢ) is correct --
        # this is exactly the online (near-zero) ε_Aˢ's residual, redefined as the dissipation estimate
        # itself. The residual computed just below becomes ~0 by construction, a sanity check that the
        # substitution was applied correctly.
        int_sfs_ape_dissipation = (-int_dAPE_dt - int_ape_to_ke_exchange.reindex(time=dAPE_dt.time)
                                    + int_Π_A_ℓ.reindex(time=dAPE_dt.time) + int_R_s)

    residual  = -int_dAPE_dt - int_ape_to_ke_exchange.reindex(time=dAPE_dt.time) + int_Π_A_ℓ.reindex(time=dAPE_dt.time) - int_sfs_ape_dissipation + int_R_s

    budget_ℓ = xr.Dataset({
        # Density fields
        "ρ̄": ds_filt_ℓ.ρ̄,
        # Reference heights
        "z₀(ρ)": full_local_pes.z0,
        "z₀(ρ̄)": filt_z0,
        # Buoyancy displacement potentials
        "Υ": full_local_pes.upsilon,
        "Υˡ": filt_upsilon,
        "D": full_local_pes.D,
        "Dˡ": filt_D,
        # Local APE fields
        "Ea(ρ, z)": full_local_pes.ape,
        "Ea(ρ̄, z)": filt_ape,
        "Ēa(ρ, z)": full_local_ape_filtered,
        "Eaˢ(ρ, z)": subfilter_local_ape,
        # Local budget terms
        "∂ₜ SFS APE": dAPE_dt,
        "Π_A": Π_A_ℓ,
        "ε_Aˢ": sfs_ape_dissipation,
        "SFS KE->APE exchange": -ape_to_ke_exchange,
        "Rˢ": R_s,
        # Integrated budget terms
        "∫-∂ₜ SFS APE dV": -int_dAPE_dt,
        "∫Π_A dV": int_Π_A_ℓ,
        "∫-ε_Aˢ dV": -int_sfs_ape_dissipation,
        "∫(SFS KE->APE) dV": -int_ape_to_ke_exchange,
        "∫Rˢ dV": int_R_s,
        "residual_A": residual,
    }).reindex(time=dAPE_dt.time)

    if implicit:
        budget_ℓ["∫-ε_Aˢ dV"].attrs["method"] = "residual estimate (implicit LES): -∂ₜE_A^s - exchange + Π_A + Rˢ"
        budget_ℓ["residual_A"].attrs["method"] = "≈0 by construction: ε_Aˢ is defined as the residual estimate (implicit LES)"

    # budget_ℓ mixes lazy pieces throughout: full_local_pes.* (re-read from its own on-disk checkpoint above
    # -- open_dataset(...).chunk() is always lazy regardless of how the source data was originally computed),
    # filt_z0/filt_upsilon/filt_D/filt_ape (same, now read from energy_transfer's on-disk output instead of
    # local_potential_energies_timeseries() directly -- eager only in the energy_transfer-predates-the-flag
    # fallback above), and other lazy pieces (ρ̄, sfs_ape_dissipation, R_s, etc.) -- see write_dataset() in
    # aux00_utils.py for why that's an issue and what --write-mode does about it.
    print(f"  Computing and saving checkpoint (write-mode={args.write_mode})...")
    t0 = time.time()
    write_dataset(budget_ℓ, str(checkpoint_path), write_mode=args.write_mode)
    print(f"  Checkpoint computed and saved  ({time.time()-t0:.1f}s)")

    # Free memory before the next iteration
    del ds_filt_ℓ, filt_z0, filt_upsilon, filt_D, filt_ape, full_local_ape_filtered, subfilter_local_ape
    del sfs_ape_dissipation, R_s, dAPE_dt, budget_ℓ
    del ape_to_ke_exchange, int_ape_to_ke_exchange
    del int_dAPE_dt, int_sfs_ape_dissipation, int_R_s
    del Π_A_ℓ, int_Π_A_ℓ, residual
    gc.collect()

    budget_list.append(xr.open_dataset(str(checkpoint_path), decode_times=False).chunk({"time": 1}))

sfs_ape_budget_terms = xr.concat(budget_list, dim=xr.DataArray(filter_scales,
                                                               dims="filter_scale",
                                                               name="filter_scale"))
sfs_ape_budget_terms.attrs.update(ds.attrs)
# Scale-independent fields don't need filter_scale dimension
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

# sfs_ape_budget_terms is fully dask-lazy here too -- xr.concat() of per-scale checkpoint reloads (each
# open_dataset(...).chunk() is lazy regardless of the checkpoint's own data having been eager on disk) plus
# ds_full.ρ -- see write_dataset() in aux00_utils.py for why that's an issue and what --write-mode does about
# it. Writing each subset separately (rather than the whole Dataset at once) keeps peak memory the same as
# the two separate to_netcdf() calls already imply -- comparable to what a single filter scale's worth of
# local fields already costs above.
print(f"  Saving local fields (write-mode={args.write_mode})...")
write_dataset(sfs_ape_budget_terms[local_vars], fields_filename, write_mode=args.write_mode)
print(f"  Fields saved to:     {fields_filename}")

print(f"  Saving integrated timeseries (write-mode={args.write_mode})...")
write_dataset(sfs_ape_budget_terms[integrated_vars], integrated_filename, write_mode=args.write_mode)
print(f"  Integrated saved to: {integrated_filename}")

print("\nDeleting intermediate checkpoint files...")
for f in checkpoint_files:
    f.unlink(missing_ok=True)
    print(f"  Deleted: {f.name}")
#---
