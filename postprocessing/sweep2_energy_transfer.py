#!/usr/bin/env python
#+++ Imports
import os
from pathlib import Path
import time
import xarray as xr
from dask.diagnostics.progress import ProgressBar
from src.aux00_utils import pad_margin_for_run, load_dataset_and_grid
from src.aux02_ke_functions import calculate_energy_transfer
#---

#+++ Configuration
import argparse
parser = argparse.ArgumentParser(description="Calculate cross-scale KE and APE transfer terms")
parser.add_argument("--filename", default="output/khi_Nz1024_Ri0.10.nc", help="Path to simulation NetCDF file")
parser.add_argument("--n-workers", type=int, default=18, help="Number of CPU workers for APE sorting (ThreadPoolExecutor)")
parser.add_argument("--fixed-reference", action="store_true", default=False, help="Load the fixed-in-time reference profile (produced by 01 with --fixed-reference)")
parser.add_argument("--reference", choices=["filtered", "true"], default="filtered",
                    help="Reference state the resolved scale is measured against. 'filtered' (default) uses the "
                         "vertically filtered profile ⟨ρ_*⟩, valid for a kernel with vertical extent. 'true' uses the "
                         "unfiltered ρ_*, the horizontal-filter limit. The sweep spans filter scales, so ⟨ρ_*⟩ is "
                         "rebuilt at each one.")
args = parser.parse_args()

print("\n" + "="*70 + f"\n  {Path(__file__).name}\n  " + "  ".join(f"{k}={v}" for k,v in vars(args).items()) + "\n" + "="*70)
REPO_ROOT = Path(__file__).resolve().parent.parent
PP_OUTPUT = REPO_ROOT / "postprocessing" / "output"
filename = str(REPO_ROOT / args.filename) if not os.path.isabs(args.filename) else args.filename
fixed_reference = args.fixed_reference
filtered_reference = args.reference == "filtered"
n_workers = args.n_workers
chunks = dict(time=1)
ref_suffix = "_fixed_ref" if fixed_reference else ""
filtered_filename = str(PP_OUTPUT / (Path(filename).stem + "_filtered_velocities_sweep.nc"))
#---

#+++ Load data and grid
print("\n" + "="*60)
print("Loading data and grid...")
t0 = time.time()
# Pad exactly as sweep1 did, so the raw field and the filtered fields it is differenced against sit on
# one grid. The margin has to come from the *sweep's* filtered file, not 01's: the sweep spans ℓ up to 20,
# whose 4σ margin is ~3x what the budget scales need, so 01's margin would pad the raw field shallower
# than ds_filt and xarray would quietly align the two to their intersection rather than raising.
ds = load_dataset_and_grid(filename, min_margin=pad_margin_for_run(filtered_filename))
ds = ds.chunk(chunks)
print(f"Dataset loaded: {len(ds.time)} time steps  ({time.time()-t0:.1f}s)")
#---

#+++ Load pre-filtered fields
print("\n" + "="*60)
print("Loading pre-filtered fields...")
t0 = time.time()
ds_filt = xr.open_dataset(filtered_filename, decode_times=False).chunk(dict(time=1, filter_scale=1))
ds = ds.reindex(time=ds_filt.time).chunk(chunks)

filter_scales = ds_filt.filter_scale.values
print(f"  Loaded from: {filtered_filename}  ({time.time()-t0:.1f}s)")
print(f"  Filter length scales: {filter_scales}")
print(f"  Filter dimensions: x and z")
#---

#+++ Load sorted density (only when using fixed reference)
rho_sorted = dz_sorted = None
if fixed_reference:
    sorted_density_filename = str(PP_OUTPUT / (Path(filename).stem + f"_sorted_density{ref_suffix}.nc"))
    ds_sorted = xr.open_dataset(sorted_density_filename, decode_times=False).chunk(chunks)
    # 02 pads to 01's margin -- the budget scales -- so its column is the sort of a *different* grid from
    # the one loaded above whenever the sweep needs more room. z✶ is read straight off that column's own
    # heights, so a mismatch shifts every Υ̃ with nothing downstream to reveal it. Refuse rather than
    # produce that: a fixed-reference sweep needs its own sorted file, built on the sweep's padding.
    n_pad_sorted = ds_sorted.attrs.get("n_pad_z")
    if n_pad_sorted is not None and int(n_pad_sorted) != int(ds.attrs["n_pad_z"]):
        raise ValueError(
            f"{sorted_density_filename} was sorted on a grid padded with {int(n_pad_sorted)} cells per side, "
            f"but the sweep pads with {int(ds.attrs['n_pad_z'])} (margin {pad_margin_for_run(filtered_filename)}). "
            f"The reference column would not belong to the field it is used on. Re-run 02_sort_density.py "
            f"against the sweep's padding, or run the sweep without --fixed-reference.")
    ds_sorted = ds_sorted.reindex(time=ds_filt.time)
    rho_sorted = ds_sorted.rho_sorted
    dz_sorted  = ds_sorted.dz_sorted
    print(f"  Sorted density loaded from: {sorted_density_filename}")
#---

#+++ Calculate cross-scale transfer terms
print("\n" + "="*60)
print("Calculating cross-scale transfer terms...")
energy_transfer = calculate_energy_transfer(ds, filter_scales,
                                            ds_filt=ds_filt,
                                            rho_sorted=rho_sorted,
                                            dz_sorted=dz_sorted,
                                            n_workers=n_workers,
                                            filtered_reference=filtered_reference)
print("\nDone!")
#---

#+++ Save results
print("\n" + "="*60)
print("Saving results...")
energy_transfer.attrs.update(ds.attrs)
output_filename = str(PP_OUTPUT / (Path(filename).stem + f"_energy_transfer_sweep{ref_suffix}.nc"))
tmp_dir = PP_OUTPUT / (Path(output_filename).stem + "_tmp")
tmp_dir.mkdir(exist_ok=True)
tmp_files = []
with ProgressBar(minimum=5, dt=5):
    for i in range(energy_transfer.sizes["time"]):
        tmp_f = str(tmp_dir / f"t{i:04d}.nc")
        energy_transfer.isel(time=[i]).to_netcdf(tmp_f)
        tmp_files.append(tmp_f)
        print(f"  wrote time {i+1}/{energy_transfer.sizes['time']}")

print("Merging per-timestep files...")
# Stream via dask (no .load(): the merged dataset is hundreds of GB and won't fit in RAM).
with xr.open_mfdataset(tmp_files, combine="by_coords", decode_timedelta=False,
                       parallel=False, chunks={"time": 1}) as merged:
    write_job = merged.to_netcdf(output_filename, compute=False)
    with ProgressBar(minimum=5, dt=5):
        write_job.compute()
for f in tmp_files:
    os.remove(f)
tmp_dir.rmdir()
print(f"Results saved to: {output_filename}")
#---
