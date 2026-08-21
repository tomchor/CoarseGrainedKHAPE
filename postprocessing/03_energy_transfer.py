#!/usr/bin/env python
#+++ Imports
import os
from pathlib import Path
import time
import xarray as xr
from dask.diagnostics.progress import ProgressBar
from src.aux00_utils import load_dataset_and_grid
from src.aux02_ke_functions import calculate_energy_transfer
#---

#+++ Configuration
import argparse
parser = argparse.ArgumentParser(description="Calculate cross-scale APE transfer (Π_A) and APE↔KE exchange terms (Π_K is computed online)")
parser.add_argument("--filename", default="output/khi_Nz2048_Ri0.10.nc", help="Path to simulation NetCDF file")
parser.add_argument("--n-workers", type=int, default=18, help="Number of CPU workers for APE sorting (ThreadPoolExecutor)")
parser.add_argument("--fixed-reference", action="store_true", default=False, help="Load the fixed-in-time reference profile (produced by 01 with --fixed-reference)")
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
print(f"  Filter dimensions: x and z")

t0 = time.time()
ref_suffix = "_fixed_ref" if fixed_reference else ""
sorted_density_filename = str(PP_OUTPUT / (Path(filename).stem + f"_sorted_density{ref_suffix}.nc"))
ds_sorted = xr.open_dataset(sorted_density_filename, decode_times=False).chunk({"time": 1})
print(f"  Sorted density loaded from: {sorted_density_filename}  ({time.time()-t0:.1f}s)")
#---

#+++ Calculate cross-scale transfer terms
print("\n" + "="*60)
print("Calculating cross-scale transfer terms...")
# Π_K (cross-scale KE transfer) is computed online by the simulation, so it is skipped here
# (include_pi_k=False). Π_A is computed online too, but only usable for the time-varying reference:
# it is measured against the reference state, and the online one is always the sort of the *current*
# buoyancy, while --fixed-reference measures every other term against the frozen t=0 profile. So the
# online field is read there, and recomputed offline here otherwise — the same split
# `05_sfs_ape_budget.py` makes for ε_Aˢ. Reading it also skips the sort of the filtered density that
# Υˡ needs, which is the expensive half of this step.
def online_name(var, ℓ):
    return f"{var}_ℓ{int(ℓ)}" if float(ℓ) == int(ℓ) else f"{var}_ℓ{ℓ}"

# The online field is an optimisation, not a requirement: --save_sorted is off by default, so a
# production run may simply not have it, and the offline path has to keep working. Fall back rather
# than fail, and say which path was taken so a silent switch is visible in the log.
online_pi_a = None
if fixed_reference:
    print("  Π_A: recomputing offline (fixed reference)")
else:
    missing = [ℓ for ℓ in filter_scales if online_name("Π_A", ℓ) not in ds]
    if missing:
        print(f"  Π_A: recomputing offline (no online field for ℓ={missing}; the simulation was run "
              f"without --save_sorted, or with a different --filter_ls)")
    else:
        online_pi_a = {ℓ: ds[online_name("Π_A", ℓ)] for ℓ in filter_scales}
        print("  Π_A: reading the online fields (time-varying reference)")

energy_transfer = calculate_energy_transfer(ds, filter_scales,
                                            ds_filt=ds_filt,
                                            rho_sorted=ds_sorted.rho_sorted,
                                            dz_sorted=ds_sorted.dz_sorted,
                                            n_workers=n_workers,
                                            include_pi_k=False,
                                            online_pi_a=online_pi_a)
print("\nDone!")
#---

#+++ Save results
print("\n" + "="*60)
print("Saving results...")
energy_transfer.attrs.update(ds.attrs)
output_filename = str(PP_OUTPUT / (Path(filename).stem + f"_energy_transfer{ref_suffix}.nc"))
with ProgressBar(minimum=5, dt=5):
    energy_transfer.to_netcdf(output_filename)
print(f"Results saved to: {output_filename}")
#---
