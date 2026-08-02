#!/usr/bin/env python
#+++ Imports
import os
from pathlib import Path
import dask
import numpy as np
import xarray as xr
from dask.diagnostics.progress import ProgressBar
from src.aux00_utils import load_dataset_and_grid, condense_velocities, GaussianFilter
#---

#+++ Configuration
import argparse
parser = argparse.ArgumentParser(description="Filter velocity and buoyancy fields for cross-scale energy transfer sweep")
parser.add_argument("--filename", default="output/bci_Nx48_Ny48_Nz8.nc", help="Path to simulation NetCDF file")
parser.add_argument("--n-time-skip", type=int, default=1, help="Keep every n-th (consecutive) time step")
parser.add_argument("--scale-min", type=float, default=None, help="Smallest filter scale (FWHM, meters). Defaults to 2x the grid spacing, the smallest scale the horizontal Gaussian filter can meaningfully resolve.")
parser.add_argument("--scale-max", type=float, default=None, help="Largest filter scale (FWHM, meters). Defaults to 40%% of the domain width Lx, staying safely below the periodic half-domain.")
parser.add_argument("--n-scales", type=int, default=30, help="Number of log-spaced filter scales between --scale-min and --scale-max (default 30)")
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
ds = ds.chunk(dict(time=1))

i = np.arange(ds.sizes["time"])
n_time_skip = args.n_time_skip
ds = ds.isel(time=(i // 2) % n_time_skip == 0)
print(f"Dataset loaded: {len(ds.time)} time steps")

# Filter scales (FWHM, meters): log-spaced between a data-driven min/max unless overridden on the CLI, so
# the sweep's range adapts automatically to whichever resolution/domain the dataset actually has.
scale_min = args.scale_min if args.scale_min is not None else 2 * float(max(ds.Δx_caa.min(), ds.Δy_aca.min()))
scale_max = args.scale_max if args.scale_max is not None else 0.4 * np.asarray(ds.attrs["Lx"]).item()
filter_scales = np.geomspace(scale_min, scale_max, args.n_scales)
print(f"Filter scales: {scale_min/1e3:.1f}km to {scale_max/1e3:.1f}km ({args.n_scales} log-spaced steps)")
#---

#+++ Filter and save velocity/buoyancy fields, one filter scale at a time to bound memory
# filter_fields() (used by 01_filter_fields.py) computes every filter scale and concatenates them into one
# Dataset before returning -- fine for 01's 2 default scales, but this sweep's default of 30 scales means
# a single .load() of the concatenated result needs ~30x the memory of any one scale, all resident at once
# for the write (a 384x384x64 run OOM'd a 256GB node this way, dying at ~30% progress as dask accumulated
# more completed scales in memory). Instead, filter and write each scale to its own small tmp file, freeing
# memory before the next scale, then merge via dask's streaming to_netcdf (the same pattern already used
# for sweep2_energy_transfer.py's per-timestep writes) -- bounds peak memory to ~1 scale's worth (all
# times, one scale) instead of all n_scales.
print("\n" + "="*60)
print("Filtering and saving velocity/buoyancy fields, one scale at a time...")

ds = condense_velocities(ds, indices=(1, 2, 3))
output_filename = str(PP_OUTPUT / (Path(filename).stem + "_filtered_velocities_sweep.nc"))
tmp_dir = PP_OUTPUT / (Path(output_filename).stem + "_tmp")
tmp_dir.mkdir(exist_ok=True)
tmp_files = []

# dx_min/dy_min don't depend on ℓ -- compute once outside the loop (and outside the ProgressBar scope
# below) rather than via make_gaussian_filter(ℓ, ds) on every iteration. That call does two eager
# float(...) scalar computes on grid spacing; triggering them inside the per-scale ProgressBar block was
# measured to cost a flat ~5s (the dt=5 polling floor) *each*, purely for a scalar that's identical across
# every scale -- 2 x n_scales wasted ~5s floors for no reason.
dx_min = float(ds.Δx_caa.min())
dy_min = float(ds.Δy_aca.min())

with ProgressBar(minimum=5, dt=5):
    for scale_idx, ℓ in enumerate(filter_scales):
        print(f"  filter_scale = {ℓ:.4f}  ({scale_idx+1}/{len(filter_scales)})...")
        gf = GaussianFilter(ℓ, dx_min, dy_min)
        ds_filt_ℓ = xr.Dataset({
            "ūᵢ": gf.apply(ds["uᵢ"], dims=["x_caa", "y_aca"]),
            "b̄":  gf.apply(ds["b"],  dims=["x_caa", "y_aca"]),
        }).expand_dims(filter_scale=[ℓ]).load()
        tmp_f = str(tmp_dir / f"scale{scale_idx:03d}.nc")
        ds_filt_ℓ.to_netcdf(tmp_f)
        tmp_files.append(tmp_f)
        del ds_filt_ℓ
print("Done!")
#---

#+++ Merge per-scale files into the final filtered-fields output
print("\n" + "="*60)
print("Merging per-scale files...")
with xr.open_mfdataset(tmp_files, combine="by_coords", decode_timedelta=False,
                       parallel=False, chunks={"filter_scale": 1}) as merged:
    merged["dV"] = ds["dV"]
    merged.attrs.update(ds.attrs)
    merged.attrs["filter_dims"] = "x_caa,y_aca"
    write_job = merged.to_netcdf(output_filename, compute=False)
    # Forced single-threaded: computing write_job under dask's default *threaded* scheduler means multiple
    # threads write into the same HDF5 file handle -- the same hang write_dataset() (aux00_utils.py) exists
    # to avoid for 01/03/04/05, confirmed as a real 3+ hour stall here (frozen at 0% progress, not just slow).
    # No "load"-style alternative offered, unlike write_dataset(): the whole point of this per-scale temp-file
    # design is to never need the merged dataset fully in memory at once (see the comment above this loop).
    with dask.config.set(scheduler="synchronous"):
        with ProgressBar(minimum=5, dt=5):
            write_job.compute()
for f in tmp_files:
    os.remove(f)
tmp_dir.rmdir()
os.sync()
print(f"Filtered fields saved to: {output_filename}")
#---
