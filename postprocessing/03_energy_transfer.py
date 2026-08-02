#!/usr/bin/env python
#+++ Imports
import gc
import os
from pathlib import Path
import time
import xarray as xr
from src.aux00_utils import load_dataset_and_grid, write_dataset
from src.aux02_ke_functions import calculate_energy_transfer
#---

#+++ Configuration
import argparse
parser = argparse.ArgumentParser(description="Calculate cross-scale APE transfer (Π_A), APE↔KE exchange terms, and the cross-scale KE transfer (Π_K)")
parser.add_argument("--filename", default="output/bci_Nx48_Ny48_Nz8.nc", help="Path to simulation NetCDF file")
parser.add_argument("--n-workers", type=int, default=18, help="Number of CPU workers for APE sorting (ThreadPoolExecutor)")
parser.add_argument("--fixed-reference", action="store_true", default=False, help="Load the fixed-in-time reference profile (produced by 01 with --fixed-reference)")
parser.add_argument("--write-mode", choices=["load", "synchronous"], default="load",
    help="How to avoid the dask-lazy .to_netcdf() write hang -- see write_dataset() in aux00_utils.py for "
         "what each mode does and the measured cost of 'synchronous' relative to 'load'.")
args = parser.parse_args()

print("\n" + "="*70 + f"\n  {Path(__file__).name}\n  " + "  ".join(f"{k}={v}" for k,v in vars(args).items()) + "\n" + "="*70)
REPO_ROOT = Path(__file__).resolve().parent.parent
PP_OUTPUT = REPO_ROOT / "postprocessing" / "output"
filename = str(REPO_ROOT / args.filename) if not os.path.isabs(args.filename) else args.filename
fixed_reference = args.fixed_reference
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

#+++ Load pre-filtered fields and pre-sorted density
print("\n" + "="*60)
print("Loading pre-filtered fields and sorted density...")
t0 = time.time()
filtered_filename = str(PP_OUTPUT / (Path(filename).stem + "_filtered_velocities.nc"))
ds_filt = xr.open_dataset(filtered_filename, decode_times=False).chunk({"time": 1})
filter_scales = ds_filt.filter_scale.values
print(f"  Filtered fields loaded from: {filtered_filename}  ({time.time()-t0:.1f}s)")
print(f"  Filter length scales: {filter_scales}")
print(f"  Filter dimensions: x and y (horizontal)")

t0 = time.time()
ref_suffix = "_fixed_ref" if fixed_reference else ""
sorted_density_filename = str(PP_OUTPUT / (Path(filename).stem + f"_sorted_density{ref_suffix}.nc"))
ds_sorted = xr.open_dataset(sorted_density_filename, decode_times=False).chunk({"time": 1})
print(f"  Sorted density loaded from: {sorted_density_filename}  ({time.time()-t0:.1f}s)")
#---

#+++ Calculate cross-scale transfer terms (checkpointed per filter scale)
print("\n" + "="*60)
print("Calculating cross-scale transfer terms...")
# Π_K (cross-scale KE transfer, horizontal-only) is now computed online by the simulation itself
# (KineticEnergyCrossScaleFlux, restricted to dims=(1,2)) -- see 04_sfs_ke_budget.py, which reads it
# straight from the simulation NetCDF instead of from this script's output. Only the APE cross-scale
# transfer Π_A and the APE↔KE exchange are computed offline here.
#
# include_filt_local_pes=True: calculate_energy_transfer() already computes z₀(ρ̄)/Υˡ/Dˡ/Ea(ρ̄, z) internally
# per scale (via local_potential_energies_timeseries() on the filtered density) but normally discards all but
# Υˡ once Π_A is built. Persisting the rest here lets 05_sfs_ape_budget.py read them back from this script's
# output instead of computing local_potential_energies_timeseries() on the same ds_filt_ℓ a second time.
#
# Looped one scale at a time and checkpointed to disk -- same pattern as 05_sfs_ape_budget.py's per-scale
# loop -- so a scale's full lazy graph (and everything upstream of it) doesn't stay reachable for the rest
# of the script. calculate_energy_transfer() itself is unchanged (still called once per scale, just with a
# single-element filter_scales list): it's also used by sweep2_energy_transfer.py, so its own internals are
# left alone here rather than adding checkpointing inside the shared function.
transfer_list = []
checkpoint_files = []
for ℓ in filter_scales:
    checkpoint_path = PP_OUTPUT / (Path(filename).stem + f"_energy_transfer_checkpoint_l{ℓ:.4f}{ref_suffix}.nc")
    checkpoint_files.append(checkpoint_path)

    if checkpoint_path.exists():
        print(f"\n--- filter_scale = {ℓ:.4f} (loading from checkpoint) ---")
        transfer_list.append(xr.open_dataset(str(checkpoint_path), decode_times=False).chunk({"time": 1}))
        continue

    transfer_ℓ = calculate_energy_transfer(ds, [ℓ],
                                           ds_filt=ds_filt,
                                           rho_sorted=ds_sorted.rho_sorted,
                                           dz_sorted=ds_sorted.dz_sorted,
                                           n_workers=n_workers,
                                           include_pi_k=False,
                                           include_filt_local_pes=True)
    print(f"  Computing and saving checkpoint (write-mode={args.write_mode})...")
    t0 = time.time()
    write_dataset(transfer_ℓ, str(checkpoint_path), write_mode=args.write_mode)
    print(f"  Checkpoint computed and saved  ({time.time()-t0:.1f}s)")
    del transfer_ℓ
    gc.collect()
    transfer_list.append(xr.open_dataset(str(checkpoint_path), decode_times=False).chunk({"time": 1}))

energy_transfer = xr.concat(transfer_list, dim="filter_scale")
print("\nDone!")
#---

#+++ Save results
print("\n" + "="*60)
print("Saving results...")
energy_transfer.attrs.update(ds.attrs)
output_filename = str(PP_OUTPUT / (Path(filename).stem + f"_energy_transfer{ref_suffix}.nc"))
# energy_transfer is dask-lazy here too -- each piece re-read from its own on-disk checkpoint above
# (open_dataset(...).chunk() is always lazy regardless of the checkpoint's own data having been eager on
# disk) -- see write_dataset() in aux00_utils.py for why that's an issue and what --write-mode does about it.
print(f"  Computing and writing energy_transfer (write-mode={args.write_mode})...")
t0 = time.time()
write_dataset(energy_transfer, output_filename, write_mode=args.write_mode)
print(f"  energy_transfer computed and saved  ({time.time()-t0:.1f}s)")
print(f"Results saved to: {output_filename}")

print("\nDeleting intermediate checkpoint files...")
for f in checkpoint_files:
    f.unlink(missing_ok=True)
    print(f"  Deleted: {f.name}")
#---
