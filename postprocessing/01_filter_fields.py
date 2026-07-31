#!/usr/bin/env python
#+++ Imports
import os
from pathlib import Path
import dask
from dask.diagnostics.progress import ProgressBar
from src.aux00_utils import load_dataset_and_grid, filter_fields
#---

#+++ Configuration
import argparse
parser = argparse.ArgumentParser(description="Filter velocity and buoyancy fields for SFS budgets")
parser.add_argument("--filename", default="output/bci_Nx48_Ny48_Nz8.nc", help="Path to simulation NetCDF file")
parser.add_argument("--filter-scales", type=float, nargs="+", default=None,
    help="Horizontal filter length scales (FWHM, in meters). Defaults to the simulation's own recorded "
         "filter_scales_m attribute (matching the online diagnostics) when present; falls back to "
         "50000 100000 for older files that predate that attribute. Pass explicitly to deliberately use "
         "different offline scales than the simulation's online ones.")
parser.add_argument("--write-mode", choices=["load", "synchronous"], default="load",
    help="How to avoid the dask-lazy .to_netcdf() write hang (see comment at the write step below). "
         "'load' (default, current production behavior): eagerly .load() the full filtered dataset into this "
         "process's memory before writing -- fast, but requires the whole dataset to fit in memory at once "
         "(OOMs at large resolutions). 'synchronous': force dask's single-threaded synchronous scheduler for "
         "just the write, which still streams/computes chunk-by-chunk (bounded memory) but avoids the "
         "multi-threaded-write hang. Untimed here -- see test_write_scheduler_timing.pbs for a controlled "
         "back-to-back comparison of the two.")
parser.add_argument("--output-suffix", default="",
    help="Appended to the output filename before .nc (e.g. '_load'/'_sync'), so a load-vs-synchronous timing "
         "comparison run on the same input doesn't have the second run overwrite the first's output. Empty "
         "by default, matching normal (single-mode) usage.")
args = parser.parse_args()

print("\n" + "="*70 + f"\n  {Path(__file__).name}\n  " + "  ".join(f"{k}={v}" for k,v in vars(args).items()) + "\n" + "="*70)
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

#+++ Resolve filter scales: explicit --filter-scales, else the simulation's own recorded attribute
# (keeps online and offline diagnostics describing the same physical scales by default), else the
# historical hardcoded default for files that predate the filter_scales_m attribute.
if args.filter_scales is not None:
    filter_scales = args.filter_scales
    print(f"  Filter scales (explicit --filter-scales): {filter_scales}")
elif "filter_scales_m" in ds.attrs:
    filter_scales = list(ds.attrs["filter_scales_m"])
    print(f"  Filter scales (from simulation's filter_scales_m attribute): {filter_scales}")
else:
    filter_scales = [50e3, 100e3]
    print(f"  Filter scales (no --filter-scales given and no filter_scales_m attribute found -- older file?): {filter_scales}")
#---

#+++ Filter velocity and buoyancy fields at each length scale
print("\n" + "="*60)
print("Filtering velocity and buoyancy fields in x and y (horizontal)...")
ds_filt = filter_fields(ds, filter_scales)
print("Done!")
#---

#+++ Save filtered fields
print("\n" + "="*60)
print("Saving filtered fields...")
output_filename = str(PP_OUTPUT / (Path(filename).stem + "_filtered_velocities" + args.output_suffix + ".nc"))
# ds_filt is still fully dask-lazy here (GaussianFilter.apply uses xr.apply_ufunc(dask="parallelized"), which
# stays lazy on this chunked input). Writing a lazy Dataset via .to_netcdf() under dask's default *threaded*
# scheduler computes it *during* the write, with multiple threads writing into the same HDF5 file handle -- a
# known hang risk, since the underlying HDF5 C library isn't reliably thread-safe for it (see
# local_potential_energies_timeseries() in aux01_pe_functions.py for a real observed stall of this kind).
# Two ways to avoid that (see --write-mode above); which one is faster in practice hasn't been measured, only
# assumed -- see test_write_scheduler_timing.pbs for a controlled comparison.
if args.write_mode == "load":
    print("  Computing filtered fields (forces the dask graph before the write)...")
    with ProgressBar(minimum=5, dt=5):
        ds_filt = ds_filt.load()
    ds_filt.to_netcdf(output_filename)
else:
    print("  Computing and writing filtered fields (synchronous scheduler, single-threaded)...")
    with dask.config.set(scheduler="synchronous"):
        with ProgressBar(minimum=5, dt=5):
            ds_filt.to_netcdf(output_filename)
print(f"Filtered fields saved to: {output_filename}")
#---
