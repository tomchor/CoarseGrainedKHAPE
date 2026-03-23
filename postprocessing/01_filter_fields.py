#!/usr/bin/env python
#+++ Imports
import os
from pathlib import Path
import numpy as np
import xarray as xr
import gcm_filters
from dask.diagnostics.progress import ProgressBar
from aux00_utils import load_dataset_and_grid, condense_velocities
#---

#+++ Configuration
import argparse
parser = argparse.ArgumentParser(description="Filter velocity and buoyancy fields for KE budget")
parser.add_argument("--filename", default="output/khi_128x1x256.nc",
                    help="Path to simulation NetCDF file")
parser.add_argument("--use-gpu", action="store_true",
                    help="Apply filters on GPU via CuPy (requires cupy and cupy-xarray)")
args = parser.parse_args()
REPO_ROOT = Path(__file__).resolve().parent.parent
filename = str(REPO_ROOT / args.filename) if not os.path.isabs(args.filename) else args.filename
filter_length_scales = np.geomspace(0.1, 2, 4) # Length scales for filtering
#---

#+++ Load data and grid
print("\n" + "="*60)
print("Loading data and grid...")
ds = load_dataset_and_grid(filename)
ds = ds.chunk({"time": 1})
print(f"Dataset loaded: {len(ds.time)} time steps")
#---

#+++ Filter velocity and buoyancy fields at each length scale
print("\n" + "="*60)
print("Filtering velocity and buoyancy fields...")

filtered_dimensions = ["x_caa", "y_aca"]
dx_min = float(min(ds.Δx_caa.min(), ds.Δy_aca.min()))

ds = condense_velocities(ds, indices=[1, 2, 3])

if args.use_gpu:
    import cupy_xarray  # noqa: F401 — registers .as_cupy() / .as_numpy() on xarray
    print("GPU mode active (CuPy backend)")

    # Build all filters once
    filters = {
        ℓ: gcm_filters.Filter(
            filter_scale=ℓ * np.sqrt(12),
            dx_min=dx_min,
            filter_shape=gcm_filters.FilterShape.GAUSSIAN,
            grid_type=gcm_filters.GridType.REGULAR,
        )
        for ℓ in filter_length_scales
    }

    # Accumulate per-scale results across time steps
    results = {ℓ: {"ūᵢ": [], "b̄": []} for ℓ in filter_length_scales}
    n_times = len(ds.time)
    for t_idx in range(n_times):
        if t_idx % 10 == 0:
            print(f"  time step {t_idx}/{n_times}...")
        ds_t = ds.isel(time=t_idx).load().as_cupy()
        for ℓ, gf in filters.items():
            results[ℓ]["ūᵢ"].append(gf.apply(ds_t["uᵢ"], dims=filtered_dimensions).as_numpy())
            results[ℓ]["b̄"].append(gf.apply(ds_t["b"],   dims=filtered_dimensions).as_numpy())

    ds_filt_list = []
    for ℓ in filter_length_scales:
        ds_ℓ = xr.Dataset({
            "ūᵢ": xr.concat(results[ℓ]["ūᵢ"], dim=ds.time),
            "b̄":  xr.concat(results[ℓ]["b̄"],  dim=ds.time),
        })
        ds_filt_list.append(ds_ℓ)

else:
    ds_filt_list = []
    for ℓ in filter_length_scales:
        print(f"  filter_length_scale = {ℓ:.4f}...")
        gaussian_filter = gcm_filters.Filter(
            filter_scale=ℓ * np.sqrt(12),
            dx_min=dx_min,
            filter_shape=gcm_filters.FilterShape.GAUSSIAN,
            grid_type=gcm_filters.GridType.REGULAR,
        )
        ds_ℓ = xr.Dataset({
            "ūᵢ": gaussian_filter.apply(ds["uᵢ"], dims=filtered_dimensions),
            "b̄":  gaussian_filter.apply(ds["b"],  dims=filtered_dimensions),
        })
        ds_filt_list.append(ds_ℓ)

scale_coord = xr.DataArray(filter_length_scales, dims="filter_length_scale",
                            name="filter_length_scale")
ds_filt = xr.concat(ds_filt_list, dim=scale_coord)
ds_filt["dV"] = ds["dV"]  # scale-independent, no filter_length_scale dimension
print("Done!")
#---

#+++ Save filtered fields
print("\n" + "="*60)
print("Saving filtered fields...")

output_filename = filename.replace(".nc", "_filtered_velocities.nc")
with ProgressBar():
    ds_filt.to_netcdf(output_filename)
print(f"Filtered fields saved to: {output_filename}")
#---
