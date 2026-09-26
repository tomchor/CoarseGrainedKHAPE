#!/usr/bin/env python
#+++ Imports
import os
from pathlib import Path
import numpy as np
from dask.diagnostics.progress import ProgressBar
from src.aux00_utils import required_pad_margin, load_dataset_and_grid, filter_fields, extension_suffix
#---

#+++ Configuration
import argparse
parser = argparse.ArgumentParser(description="Filter velocity and buoyancy fields for cross-scale energy transfer sweep")
parser.add_argument("--filename", default="output/khi_Nz2048_Ri0.10.nc", help="Path to simulation NetCDF file")
parser.add_argument("--n-time-skip", type=int, default=1, help="Keep every n-th (consecutive) time step")
parser.add_argument("--filter-scales", type=float, nargs="+", default=None,
                    help="Filter scales to sweep (default: 30 points log-spaced over 0.02-20). Give a single "
                         "value to test one scale, e.g. --filter-scales 20")
parser.add_argument("--extension", choices=["edge", "odd"], default="edge",
                    help="How b and b_* are extended past the walls: 'edge' repeats the wall value (the "
                         "paper's section 4 choice), 'odd' reflects oddly about it. Both are admissible, so "
                         "any difference between them is an artifact of the choice rather than physics.")
args = parser.parse_args()

print("\\n" + "="*70 + f"\\n  {Path(__file__).name}\\n  " + "  ".join(f"{k}={v}" for k,v in vars(args).items()) + "\\n" + "="*70)
REPO_ROOT = Path(__file__).resolve().parent.parent
PP_OUTPUT = REPO_ROOT / "postprocessing" / "output"
filename = str(REPO_ROOT / args.filename) if not os.path.isabs(args.filename) else args.filename
filter_scales = (np.asarray(args.filter_scales, dtype=float) if args.filter_scales
                 else np.geomspace(0.02, 20, 30))  # Length scales for filtering
ext_suffix = extension_suffix(args.extension)
#---

#+++ Load data and grid
print("\n" + "="*60)
print("Loading data and grid...")
ds = load_dataset_and_grid(filename, min_margin=required_pad_margin(filter_scales),
                           extension=args.extension)
ds = ds.chunk(dict(time=1))

i = np.arange(ds.sizes["time"])
n_time_skip = args.n_time_skip
ds = ds.isel(time=(i // 2) % n_time_skip == 0)
print(f"Dataset loaded: {len(ds.time)} time steps")
#---

#+++ Filter velocity and buoyancy fields at each length scale
print("\n" + "="*60)
print("Filtering velocity and buoyancy fields in x and z...")

ds_filt = filter_fields(ds, filter_scales)
ds_filt.attrs["pad_margin"]  = required_pad_margin(filter_scales)
ds_filt.attrs["z_extension"] = args.extension   # sweep2 reads this back, as it does pad_margin
ds_filt.attrs["z_extension_vars"] = ds.attrs["z_extension_vars"]   # which fields the rule reached
print("Done!")
#---

#+++ Save filtered fields
print("\n" + "="*60)
print("Saving filtered fields...")

output_filename = str(PP_OUTPUT / (Path(filename).stem + f"_filtered_velocities_sweep{ext_suffix}.nc"))
with ProgressBar(minimum=5, dt=5):
    ds_filt.to_netcdf(output_filename)
os.sync()
print(f"Filtered fields saved to: {output_filename}")
#---
