#!/usr/bin/env python
#+++ Imports
import os
from pathlib import Path
from dask.diagnostics.progress import ProgressBar
from src.aux00_utils import load_dataset_and_grid
from src.aux01_pe_functions import calculate_density_fields_from_buoyancy, sorted_timeseries
#---

#+++ Configuration
import argparse
parser = argparse.ArgumentParser(description="Sort density and compute reference state for APE calculation")
parser.add_argument("--filename", default="output/khi_Nz256_Ri0.10.nc", help="Path to simulation NetCDF file")
parser.add_argument("--n-workers", type=int, default=18, help="Number of CPU workers for density sorting (ThreadPoolExecutor)")
parser.add_argument("--fixed-reference", action="store_true", default=False, help="Use the t=0 density field as a fixed-in-time reference profile")
args = parser.parse_args()

print("\n" + "="*70 + f"\n  {Path(__file__).name}\n  " + "  ".join(f"{k}={v}" for k,v in vars(args).items()) + "\n" + "="*70)
REPO_ROOT = Path(__file__).resolve().parent.parent
PP_OUTPUT = REPO_ROOT / "postprocessing" / "output"
filename = str(REPO_ROOT / args.filename) if not os.path.isabs(args.filename) else args.filename
n_workers = args.n_workers
fixed_reference = args.fixed_reference
#---

#+++ Load data and grid
print("\n" + "="*60)
print("Loading data and grid...")
ds = load_dataset_and_grid(filename)
ds = ds.chunk({"time": 1})
print(f"Dataset loaded: {len(ds.time)} time steps")
#---

#+++ Sort unfiltered density and save reference state
print("\n" + "="*60)
print("Sorting unfiltered density...")
ds_for_sort = ds[["b", "dV", "LxLy"]].copy()
ds_for_sort.attrs.update(ds.attrs)
ds_for_sort = calculate_density_fields_from_buoyancy(ds_for_sort, buoyancy_name="b", density_name="ρ")

sorted_density = sorted_timeseries(ds_for_sort, field_to_sort="ρ", n_workers=n_workers,
                                   fixed_reference=fixed_reference)
sorted_density.attrs.update(ds.attrs)

ref_suffix = "_fixed_ref" if fixed_reference else ""
sorted_density_filename = str(PP_OUTPUT / (Path(filename).stem + f"_sorted_density{ref_suffix}.zarr"))
sorted_density = sorted_density.chunk({d: (1 if d == "time" else -1) for d in sorted_density.dims})
with ProgressBar():
    sorted_density.to_zarr(sorted_density_filename, mode="w")
print(f"Sorted density saved to: {sorted_density_filename}")
#---
