#!/usr/bin/env python
#+++ Imports
import os
from pathlib import Path
import time
import xarray as xr
from dask.diagnostics.progress import ProgressBar
from src.aux00_utils import PP_OUTPUT, pad_margin_for_run, load_dataset_and_grid, reference_suffix
from src.aux02_ke_functions import calculate_energy_transfer
#---

#+++ Configuration
import argparse
parser = argparse.ArgumentParser(description="Calculate cross-scale APE transfer (Π_A) and APE↔KE exchange terms (Π_K is computed online)")
parser.add_argument("--filename", default="output/khi_Nz2048_Ri0.10.nc", help="Path to simulation NetCDF file")
parser.add_argument("--n-workers", type=int, default=18, help="Number of CPU workers for APE sorting (ThreadPoolExecutor)")
parser.add_argument("--fixed-reference", action="store_true", default=False, help="Load the fixed-in-time reference profile (produced by 01 with --fixed-reference)")
parser.add_argument("--reference", choices=["filtered", "true"], default="filtered",
                    help="Reference state the resolved scale is measured against. 'filtered' (default) uses the "
                         "vertically filtered profile ⟨ρ_*⟩, valid for a kernel with vertical extent. 'true' uses the "
                         "unfiltered ρ_*, the horizontal-filter limit the pipeline used before.")
args = parser.parse_args()

print("\n" + "="*70 + f"\n  {Path(__file__).name}\n  " + "  ".join(f"{k}={v}" for k,v in vars(args).items()) + "\n" + "="*70)
REPO_ROOT = Path(__file__).resolve().parent.parent
filename = str(REPO_ROOT / args.filename) if not os.path.isabs(args.filename) else args.filename
fixed_reference = args.fixed_reference
filtered_reference = args.reference == "filtered"
n_workers = args.n_workers
#---

#+++ Load data and grid
print("\n" + "="*60)
print("Loading data and grid...")
t0 = time.time()
# Pad exactly as 01 did, so the sort and the budgets see the grid the fields were filtered on.
_filtered_fn = str(PP_OUTPUT / (Path(filename).stem + "_filtered_velocities.nc"))
ds = load_dataset_and_grid(filename, min_margin=pad_margin_for_run(_filtered_fn))
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
out_suffix = ref_suffix + reference_suffix(args.reference)   # 02's sort is shared by both references; this output is not
sorted_density_filename = str(PP_OUTPUT / (Path(filename).stem + f"_sorted_density{ref_suffix}.nc"))
ds_sorted = xr.open_dataset(sorted_density_filename, decode_times=False).chunk({"time": 1})
print(f"  Sorted density loaded from: {sorted_density_filename}  ({time.time()-t0:.1f}s)")
#---

#+++ Calculate cross-scale transfer terms
print("\n" + "="*60)
print("Calculating cross-scale transfer terms...")
# Π_A is built on the filtered reference profile ⟨ρ_*⟩, and that profile is exact only here: the
# simulation still coarsens it onto a block-averaged column (see filtered_reference_decisions.md §3-§6).
# Two different constructions cannot both feed one budget, so every reference-dependent term -- Π_A, ε_Aˢ,
# Υ̃, L̃, S̃, Rˢ -- is computed offline against the one exact profile, and the simulation's versions are the
# independent cross-check that inv08/inv09/inv10 exist to make. Π_K and ε_Kˢ are unaffected: they are built
# from velocities alone and never touch the reference state, so 04 still reads them online.
online_pi_a = None
print("  Π_A: computing offline, against the exact (FFT) reference profile")

energy_transfer = calculate_energy_transfer(ds, filter_scales,
                                            ds_filt=ds_filt,
                                            rho_sorted=ds_sorted.rho_sorted,
                                            dz_sorted=ds_sorted.dz_sorted,
                                            n_workers=n_workers,
                                            include_pi_k=False,
                                            online_pi_a=online_pi_a,
                                            filtered_reference=filtered_reference)
print("\nDone!")
#---

#+++ Save results
print("\n" + "="*60)
print("Saving results...")
energy_transfer.attrs.update(ds.attrs)
energy_transfer.attrs["ape_reference"] = args.reference   # 05 checks this against its own --reference
output_filename = str(PP_OUTPUT / (Path(filename).stem + f"_energy_transfer{out_suffix}.nc"))
with ProgressBar(minimum=5, dt=5):
    energy_transfer.to_netcdf(output_filename)
print(f"Results saved to: {output_filename}")
#---
