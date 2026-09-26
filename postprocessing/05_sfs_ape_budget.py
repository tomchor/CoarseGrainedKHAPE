#!/usr/bin/env python
"""
Calculate SFS APE budget from Kelvin-Helmholtz simulation output
"""

#+++ Imports
import gc
import hashlib
import logging
import os
from pathlib import Path
import time
import xarray as xr
from dask.diagnostics.progress import ProgressBar
from src.aux00_utils import (PP_OUTPUT, pad_margin_for_run, load_dataset_and_grid, condense_uw_velocities, integrate, make_gaussian_filter,
                             load_energy_transfer, reference_suffix, check_same_padded_grid)
from src.aux01_pe_functions import (
    calculate_density_fields_from_buoyancy,
    local_potential_energies_timeseries,  # used for filtered density in loop
    calculate_sfs_ape_tendency,
    calculate_sfs_R_correction,
    calculate_sfs_ape_dissipation,
    filtered_reference_profile,
)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
print = logging.info
#---

#+++ Configuration
import argparse
parser = argparse.ArgumentParser(description="Calculate SFS APE budget from Kelvin-Helmholtz simulation output")
parser.add_argument("--filename", default="output/khi_Nz256_Ri0.10.nc", help="Path to simulation NetCDF file")
parser.add_argument("--n-workers", type=int, default=18, help="Number of CPU workers for APE sorting (ThreadPoolExecutor)")
parser.add_argument("--fixed-reference", action="store_true", default=False,
                    help="Load the fixed-in-time reference profile (produced by 01 with --fixed-reference)")
parser.add_argument("--reference", choices=["filtered", "true"], default="filtered",
                    help="Reference state the resolved reservoir is measured against. 'filtered' (default) uses the "
                         "vertically filtered profile ⟨ρ_*⟩, the scale decomposition valid for a kernel with vertical "
                         "extent. 'true' uses the unfiltered ρ_* for both reservoirs, the horizontal-filter limit, "
                         "the formulation the pipeline used before. It does not reproduce the earlier numbers: "
                         "integrals now exclude the z padding, and Π_A and ε_Aˢ are always computed offline.")
args = parser.parse_args()

print("\n" + "="*70 + f"\n  {Path(__file__).name}\n  " + "  ".join(f"{k}={v}" for k,v in vars(args).items()) + "\n" + "="*70)
REPO_ROOT = Path(__file__).resolve().parent.parent
filename = str(REPO_ROOT / args.filename) if not os.path.isabs(args.filename) else args.filename
n_workers = args.n_workers
fixed_reference = args.fixed_reference
filtered_reference = args.reference == "filtered"
#---

#+++ Load data and grid
print("\n" + "="*60)
print("Loading data and grid...")
t0 = time.time()
# Pad exactly as 01 did, so the sort and the budgets see the grid the fields were filtered on.
_filtered_fn = str(PP_OUTPUT / (Path(filename).stem + "_filtered_velocities.nc"))
ds = load_dataset_and_grid(filename, min_margin=pad_margin_for_run(_filtered_fn, required=True))
ds = ds.chunk({"time": 1})
print(f"Dataset loaded: {len(ds.time)} time steps  ({time.time()-t0:.1f}s)")
#---

#+++ Load filtered fields and pre-sorted density
print("\n" + "="*60)
print("Loading pre-filtered fields and sorted density...")

filtered_filename = str(PP_OUTPUT / (Path(filename).stem + "_filtered_velocities.nc"))
t0 = time.time()
ds_filt = xr.open_dataset(filtered_filename, decode_times=False).chunk({"time": 1})
filter_scales = ds_filt.filter_scale.values
filtered_dimensions = ["x_caa", "z_aac"]

ds = condense_uw_velocities(ds, indices=[1, 3])
ds_full = ds[["b", "dV", "dV_physical", "LxLy", "uᵢ"]].copy()
print(f"  Pre-filtered fields loaded from: {filtered_filename}  ({time.time()-t0:.1f}s)")
print(f"  Filter length scales: {filter_scales}")
print(f"  Filter dimensions: x and z")

ref_suffix = "_fixed_ref" if fixed_reference else ""
out_suffix = ref_suffix + reference_suffix(args.reference)   # 02's sort is shared by both references; this output is not
sorted_density_filename = str(PP_OUTPUT / (Path(filename).stem + f"_sorted_density{ref_suffix}.nc"))
t0 = time.time()
ds_sorted = xr.open_dataset(sorted_density_filename, decode_times=False).chunk({"time": 1})
check_same_padded_grid(ds, ds_sorted, Path(sorted_density_filename).name)   # sorted on this padded grid?
print(f"  Sorted density loaded from: {sorted_density_filename}  ({time.time()-t0:.1f}s)")
#---

#+++ Checkpoints
# A checkpoint is reused only if it was written for this run: the same simulation output (its global
# attributes carry the run's parameters and creation date), the same time axis, and the same padded grid
# (n_pad_z and z_extension, which load_dataset_and_grid adds to the same attributes). Anything else is
# discarded and recomputed. Without this, a job resumed after 01 was rerun with other filter scales, or after
# the simulation was rerun under the same name, reloaded the old full_local_pes -- whose sorted column feeds
# ⟨ρ_*⟩, the filtered lookup and Rˢ at every scale -- and mixed old per-scale budgets with new ones.
CHECKPOINT_ID = hashlib.sha1((repr(sorted((k, str(v)) for k, v in ds.attrs.items()))
                              + repr(ds.time.values.tolist())).encode()).hexdigest()


def load_checkpoint(path):
    """Open `path` if it was written for this run; otherwise delete it and return None, so it is recomputed."""
    if not path.exists():
        return None
    ckpt = xr.open_dataset(str(path), decode_times=False)
    if ckpt.attrs.get("checkpoint_id") == CHECKPOINT_ID:
        return ckpt.chunk({"time": 1})
    ckpt.close()
    print(f"  Discarding {path.name}: written for a different run, time axis or padded grid")
    path.unlink()
    return None


def save_checkpoint(dset, path):
    """Write `dset` to `path`, stamped with this run's identity and padded grid."""
    dset.attrs.update(checkpoint_id=CHECKPOINT_ID, n_pad_z=ds.attrs["n_pad_z"], z_extension=ds.attrs["z_extension"])
    with ProgressBar(minimum=5, dt=5):
        dset.to_netcdf(str(path))
#---

#+++ Calculate scale-independent fields
print("\n" + "="*60)
print("Calculating scale-independent fields...")

t0 = time.time()
ds_full = calculate_density_fields_from_buoyancy(ds_full, buoyancy_name="b", density_name="ρ")
print(f"  ρ calculated  ({time.time()-t0:.1f}s)")

# full_local_pes is the full field against ρ_*, the same under either --reference, so both share it.
full_local_pes_checkpoint = PP_OUTPUT / (Path(filename).stem + f"_full_local_pes_checkpoint{ref_suffix}.nc")
full_local_pes = load_checkpoint(full_local_pes_checkpoint)
if full_local_pes is not None:
    print(f"  full_local_pes loaded from checkpoint: {full_local_pes_checkpoint.name}")
else:
    t0 = time.time()
    full_local_pes = local_potential_energies_timeseries(ds_full, ds_sorted.rho_sorted, ds_sorted.dz_sorted,
                                                         density_name="ρ", n_workers=n_workers)
    print(f"  full_local_pes calculated  ({time.time()-t0:.1f}s)")
    print(f"  Saving full_local_pes checkpoint...")
    t0 = time.time()
    save_checkpoint(full_local_pes, full_local_pes_checkpoint)
    print(f"  Checkpoint saved  ({time.time()-t0:.1f}s)")
    del full_local_pes
    gc.collect()
    full_local_pes = xr.open_dataset(str(full_local_pes_checkpoint), decode_times=False).chunk({"time": 1})
    print(f"  full_local_pes reloaded lazily")
#---

#+++ Loop over filter scales and calculate budget terms
print("\n" + "="*60)
print("Calculating budget terms for each filter scale...")

energy_transfer = load_energy_transfer(filename, ref_suffix=out_suffix)

ke_fields_filename     = str(PP_OUTPUT / (Path(filename).stem + f"_sfs_ke_budget_fields{out_suffix}.nc"))
ke_integrated_filename = str(PP_OUTPUT / (Path(filename).stem + f"_sfs_ke_budget_integrated{out_suffix}.nc"))
ke_budget = xr.merge([
    xr.open_dataset(ke_fields_filename,     decode_times=False).chunk({"time": 1}),
    xr.open_dataset(ke_integrated_filename, decode_times=False).chunk({"time": 1}),
])
print(f"  KE budget loaded from: {ke_fields_filename} + {ke_integrated_filename}")

# Π_A (from 03) and the APE->KE exchange (from 04) must be measured against the reference this step uses,
# or the budget mixes two reference states. Output written before the attribute existed has none: rerun it.
for step, d in (("03", energy_transfer), ("04", ke_budget)):
    if d.attrs.get("ape_reference") != args.reference:
        raise ValueError(f"{step} output was built with ape_reference={d.attrs.get('ape_reference')!r}, but this run "
                         f"uses --reference {args.reference}; rerun 03 and 04 with --reference {args.reference}")

dV = ds_full.dV_physical   # padding carries no volume; see _pad_domain_in_z
budget_list = []
checkpoint_files = [full_local_pes_checkpoint]

for ℓ in filter_scales:
    checkpoint_path = PP_OUTPUT / (Path(filename).stem + f"_sfs_ape_budget_checkpoint_l{ℓ:.4f}{out_suffix}.nc")
    checkpoint_files.append(checkpoint_path)

    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint is not None:
        print(f"\n--- filter_scale = {ℓ:.4f} (loading from checkpoint) ---")
        budget_list.append(checkpoint)
        continue

    print(f"\n--- filter_scale = {ℓ:.4f} ---")

    gaussian_filter = make_gaussian_filter(ℓ, ds)

    ds_filt_ℓ = ds_filt.sel(filter_scale=ℓ).drop_vars("filter_scale")
    ds_filt_ℓ["LxLy"] = ds["LxLy"]
    ds_filt_ℓ.attrs.update(ds.attrs)

    t0 = time.time()
    ds_filt_ℓ = calculate_density_fields_from_buoyancy(ds_filt_ℓ, buoyancy_name="b̄", density_name="ρ̄")
    print(f"  ρ̄ calculated  ({time.time()-t0:.1f}s)")

    # The reference the resolved reservoir is measured against. With a kernel that has vertical extent the
    # unfiltered ρ_* is the wrong one: a fluid at rest filters to a profile that still carries APE against
    # ρ_*, so Eₐˡ does not vanish at rest and the remainder Eₐˢ goes negative. ⟨ρ_*⟩ — the rest state as the
    # filter sees it — makes both reservoirs vanish at rest and positive semi-definite (Eqs. 2.3-2.5). It is
    # scale-dependent, hence built inside this loop. `--reference true` restores the horizontal-limit path.
    if filtered_reference:
        t0 = time.time()
        ref_rho_sorted = filtered_reference_profile(full_local_pes.rho_sorted, full_local_pes.dz_sorted, ℓ,
                                                    frozen=fixed_reference)
        print(f"  ⟨ρ_*⟩ built (filtered reference)  ({time.time()-t0:.1f}s)")
    else:
        ref_rho_sorted = full_local_pes.rho_sorted
        print("  reference: unfiltered ρ_* (horizontal-filter limit)")

    t0 = time.time()
    filt_local_pes = local_potential_energies_timeseries(ds_filt_ℓ, ref_rho_sorted, full_local_pes.dz_sorted,
                                                         density_name="ρ̄", n_workers=n_workers)
    print(f"  filt_local_pes  ({time.time()-t0:.1f}s)")

    t0 = time.time()
    full_local_ape_filtered = gaussian_filter.apply(full_local_pes.ape, dims=filtered_dimensions)
    subfilter_local_ape = full_local_ape_filtered - filt_local_pes.ape
    print(f"  local APE filtered  ({time.time()-t0:.1f}s)")

    # ε_Aˢ is built on the filtered reference profile, which is exact only offline (the simulation
    # coarsens it), so it is computed here rather than read -- see 03 and filtered_reference_decisions.md.
    # The offline gradients are centred rather than face-paired, which cost 18.1% vs 2.0% of the dominant
    # term at Nz=128; measured at Nz=1024 the two agree to 0.99 and the residual moves 0.287% -> 0.320%,
    # so at production resolution consistency is worth more than the discretisation.
    t0 = time.time()
    sfs_ape_dissipation = calculate_sfs_ape_dissipation(
        ds_full.ρ, full_local_pes.upsilon, filt_local_pes.upsilon, ds.κ, gaussian_filter,
        filter_dims=filtered_dimensions,
        filtered_density=ds_filt_ℓ.ρ̄,)
    print(f"  sfs_ape_dissipation: offline, against {'⟨ρ_*⟩' if filtered_reference else 'the unfiltered ρ_*'}  "
      f"({time.time()-t0:.1f}s)")

    # Read APE->KE exchange term from KE budget (avoid redundant recalculation)
    ape_to_ke_exchange     = ke_budget["SFS APE->KE exchange"].sel(filter_scale=ℓ, method="nearest", tolerance=1e-6)
    int_ape_to_ke_exchange = ke_budget["∫(SFS APE->KE) dV"].sel(filter_scale=ℓ, method="nearest", tolerance=1e-6)

    t0 = time.time()
    R_s = calculate_sfs_R_correction(full_local_pes.rho_sorted, full_local_pes.z0, filt_local_pes.z0,
                                     full_local_pes.dz_sorted, gaussian_filter,
                                     filter_dims=filtered_dimensions, n_workers=n_workers,
                                     filt_rho_sorted=ref_rho_sorted if filtered_reference else None)
    print(f"  R_s  ({time.time()-t0:.1f}s)")

    dAPE_dt = calculate_sfs_ape_tendency(subfilter_local_ape)

    int_dAPE_dt             = integrate(dAPE_dt, dV)
    int_sfs_ape_dissipation = integrate(sfs_ape_dissipation.reindex(time=dAPE_dt.time), dV)
    int_R_s                 = integrate(R_s.reindex(time=dAPE_dt.time), dV)

    Π_A_ℓ     = energy_transfer["Π_A"].sel(filter_scale=ℓ, method="nearest", tolerance=1e-6)
    int_Π_A_ℓ = energy_transfer["∫Π_A dV"].sel(filter_scale=ℓ, method="nearest", tolerance=1e-6)
    residual  = (-int_dAPE_dt - int_ape_to_ke_exchange.reindex(time=dAPE_dt.time) + int_Π_A_ℓ.reindex(time=dAPE_dt.time)
                 - int_sfs_ape_dissipation + int_R_s)

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

    print(f"  Saving checkpoint...")
    t0 = time.time()
    save_checkpoint(budget_ℓ, checkpoint_path)
    print(f"  Checkpoint saved  ({time.time()-t0:.1f}s)")

    # Free memory before the next iteration
    del ds_filt_ℓ, filt_local_pes, full_local_ape_filtered, subfilter_local_ape, ref_rho_sorted
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
sfs_ape_budget_terms.attrs["ape_reference"] = "filtered" if filtered_reference else "true"

# Scale-independent fields don't need filter_scale dimension
sfs_ape_budget_terms["ρ"] = ds_full.ρ
print("\nDone!")
#---

#+++ Save results
print("\n" + "="*60)
print("Saving results...")

integrated_vars = [v for v in sfs_ape_budget_terms.data_vars if v.startswith("∫") or "residual" in v]
local_vars      = [v for v in sfs_ape_budget_terms.data_vars if v not in integrated_vars]

fields_filename     = str(PP_OUTPUT / (Path(filename).stem + f"_sfs_ape_budget_fields{out_suffix}.nc"))
integrated_filename = str(PP_OUTPUT / (Path(filename).stem + f"_sfs_ape_budget_integrated{out_suffix}.nc"))

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
