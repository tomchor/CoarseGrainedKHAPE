#!/usr/bin/env python
#+++ Imports
import os
from pathlib import Path
import numpy as np
from dask.diagnostics.progress import ProgressBar
from aux00_utils import load_dataset_and_grid, filter_fields
from aux01_pe_functions import calculate_density_fields_from_buoyancy, sorted_timeseries
#---

#+++ Configuration
import argparse
parser = argparse.ArgumentParser(description="Filter velocity and buoyancy fields for KE budget")
parser.add_argument("--filename", default="output/khi_90x1x256.nc",
                    help="Path to simulation NetCDF file")
parser.add_argument("--n-workers", type=int, default=18,
                    help="Number of CPU workers for density sorting (ThreadPoolExecutor)")
args = parser.parse_args()
REPO_ROOT = Path(__file__).resolve().parent.parent
PP_OUTPUT = REPO_ROOT / "postprocessing" / "output"
filename = str(REPO_ROOT / args.filename) if not os.path.isabs(args.filename) else args.filename
n_workers = args.n_workers
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
filter_in_2d = ds.sizes["x_caa"] > 1 and ds.sizes["y_aca"] > 1
print("\n" + "="*60)
if filter_in_2d:
    print("Filtering velocity and buoyancy fields in 2D (x and y)...")
else:
    print("Filtering velocity and buoyancy fields in 1D (x only)...")

ds_filt = filter_fields(ds, filter_length_scales, filter_in_2d=filter_in_2d)
print("Done!")
#---

#+++ Save filtered fields
print("\n" + "="*60)
print("Saving filtered fields...")

output_filename = str(PP_OUTPUT / (Path(filename).stem + "_filtered_velocities.nc"))
with ProgressBar():
    ds_filt.to_netcdf(output_filename)
print(f"Filtered fields saved to: {output_filename}")
#---

#+++ Sort unfiltered density and save reference state
print("\n" + "="*60)
print("Sorting unfiltered density...")

ds_for_sort = ds[["b", "dV", "LxLy"]].copy()
ds_for_sort.attrs.update(ds.attrs)
ds_for_sort = calculate_density_fields_from_buoyancy(ds_for_sort, buoyancy_name="b", density_name="ρ")

sorted_density = sorted_timeseries(ds_for_sort, field_to_sort="ρ", n_workers=n_workers)

sorted_density_filename = str(PP_OUTPUT / (Path(filename).stem + "_sorted_density.nc"))
with ProgressBar():
    sorted_density.to_netcdf(sorted_density_filename)
print(f"Sorted density saved to: {sorted_density_filename}")
#---