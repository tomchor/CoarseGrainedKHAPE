#!/usr/bin/env python
#+++ Imports
import os
from pathlib import Path
import numpy as np
from dask.diagnostics.progress import ProgressBar
from aux00_utils import load_dataset_and_grid, filter_fields
#---

#+++ Configuration
import argparse
parser = argparse.ArgumentParser(description="Filter velocity and buoyancy fields for cross-scale energy transfer sweep")
parser.add_argument("--filename", default="output/khi_Nz256_Ri0.10.nc",
                    help="Path to simulation NetCDF file")
args = parser.parse_args()
REPO_ROOT = Path(__file__).resolve().parent.parent
filename = str(REPO_ROOT / args.filename) if not os.path.isabs(args.filename) else args.filename
filter_length_scales = np.geomspace(0.05, 2, 25) # Length scales for filtering
#---

#+++ Load data and grid
print("\n" + "="*60)
print("Loading data and grid...")
ds = load_dataset_and_grid(filename)
ds = ds.chunk(dict(time=1, z_aac=10))

ds = ds.sel(time=[40, 50, 60], method="nearest")
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

output_filename = filename.replace(".nc", "_filtered_velocities_sweep.nc")
with ProgressBar():
    ds_filt.to_netcdf(output_filename)
print(f"Filtered fields saved to: {output_filename}")
#---
