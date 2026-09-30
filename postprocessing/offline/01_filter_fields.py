#!/usr/bin/env python
#+++ Imports
import os
from pathlib import Path
from dask.diagnostics.progress import ProgressBar
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # postprocessing/ on the path, for `src.*`
from src.aux00_utils import PP_OUTPUT, required_pad_margin, load_dataset_and_grid, filter_fields, upper_records
#---

#+++ Configuration
import argparse
parser = argparse.ArgumentParser(description="Filter velocity and buoyancy fields for SFS budgets")
parser.add_argument("--filename", default="output/khi_Nz256_Ri0.10.nc", help="Path to simulation NetCDF file")
parser.add_argument("--filter-scales", type=float, nargs="+", default=[1, 7],
                    help="Filter length scales (must match the simulation's online filter_ℓs, since the SFS KE budget reads Π_K from the "
                         "online output)")
args = parser.parse_args()

print("\n" + "="*70 + f"\n  {Path(__file__).name}\n  " + "  ".join(f"{k}={v}" for k,v in vars(args).items()) + "\n" + "="*70)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
filename = str(REPO_ROOT / args.filename) if not os.path.isabs(args.filename) else args.filename
filter_scales = args.filter_scales
#---

#+++ Load data and grid
print("\n" + "="*60)
print("Loading data and grid...")
ds = load_dataset_and_grid(filename, min_margin=required_pad_margin(filter_scales))
ds = ds.chunk({"time": 1})
print(f"Dataset loaded: {len(ds.time)} time steps")
# The tendencies of 04 and 05, and ∂ₜρ✶ inside R, are differences across the consecutive-iteration pair the
# simulation writes at each output with --offline_check. The online TimeDerivative needs no pair, so a run without
# the flag has one record per output time, and differencing those would span the whole output interval.
if not upper_records(ds.time.values).any():
    raise ValueError(f"{Path(filename).name} has no consecutive-iteration output pairs, which the offline pipeline "
                     "differences for its tendencies: rerun the simulation with --offline_check (OFFLINE_CHECK=1).")
#---

#+++ Filter velocity and buoyancy fields at each length scale
print("\n" + "="*60)
print("Filtering velocity and buoyancy fields in x and z...")
ds_filt = filter_fields(ds, filter_scales)
ds_filt.attrs["pad_margin"] = required_pad_margin(filter_scales)
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
