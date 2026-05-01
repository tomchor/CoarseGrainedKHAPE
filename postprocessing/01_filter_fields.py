#!/usr/bin/env python
#+++ Imports
import os
from pathlib import Path
from dask.diagnostics.progress import ProgressBar
from aux00_utils import load_dataset_and_grid, filter_fields
#---

#+++ Configuration
import argparse
parser = argparse.ArgumentParser(description="Filter velocity and buoyancy fields for SFS budgets")
parser.add_argument("--filename", default="output/khi_Nz256_Ri0.10.nc", help="Path to simulation NetCDF file")
parser.add_argument("--filter-scales", type=float, nargs="+", default=[0.2, 0.5, 1, 2, 4, 7, 14], help="Filter length scales")
args = parser.parse_args()

print("\n" + "="*70 + f"\n  {Path(__file__).name}\n  " + "  ".join(f"{k}={v}" for k,v in vars(args).items()) + "\n" + "="*70)
REPO_ROOT = Path(__file__).resolve().parent.parent
PP_OUTPUT = REPO_ROOT / "postprocessing" / "output"
filename = str(REPO_ROOT / args.filename) if not os.path.isabs(args.filename) else args.filename
filter_scales = args.filter_scales
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
print("Filtering velocity and buoyancy fields in x and z...")
ds_filt = filter_fields(ds, filter_scales)
print("Done!")
#---

#+++ Save filtered fields
print("\n" + "="*60)
print("Saving filtered fields...")
output_filename = str(PP_OUTPUT / (Path(filename).stem + "_filtered_velocities.nc"))
with ProgressBar(minimum=5, dt=5):
    ds_filt.to_netcdf(output_filename)
print(f"Filtered fields saved to: {output_filename}")
#---
