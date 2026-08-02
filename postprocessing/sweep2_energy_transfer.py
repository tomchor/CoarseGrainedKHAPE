#!/usr/bin/env python
#+++ Imports
import os
from pathlib import Path
import time
import dask
import xarray as xr
from dask.diagnostics.progress import ProgressBar
from src.aux00_utils import load_dataset_and_grid
from src.aux02_ke_functions import calculate_energy_transfer
#---

#+++ Configuration
import argparse
parser = argparse.ArgumentParser(description="Calculate cross-scale KE and APE transfer terms")
parser.add_argument("--filename", default="output/bci_Nx48_Ny48_Nz8.nc", help="Path to simulation NetCDF file")
parser.add_argument("--n-workers", type=int, default=18, help="Number of CPU workers for APE sorting (ThreadPoolExecutor)")
parser.add_argument("--fixed-reference", action="store_true", default=False, help="Load the fixed-in-time reference profile (produced by 01 with --fixed-reference)")
args = parser.parse_args()

print("\n" + "="*70 + f"\n  {Path(__file__).name}\n  " + "  ".join(f"{k}={v}" for k,v in vars(args).items()) + "\n" + "="*70)
REPO_ROOT = Path(__file__).resolve().parent.parent
PP_OUTPUT = REPO_ROOT / "postprocessing" / "output"
filename = str(REPO_ROOT / args.filename) if not os.path.isabs(args.filename) else args.filename
fixed_reference = args.fixed_reference
n_workers = args.n_workers
chunks = dict(time=1)
ref_suffix = "_fixed_ref" if fixed_reference else ""
#---

#+++ Load data and grid
print("\n" + "="*60)
print("Loading data and grid...")
t0 = time.time()
ds = load_dataset_and_grid(filename)
ds = ds.chunk(chunks)
print(f"Dataset loaded: {len(ds.time)} time steps  ({time.time()-t0:.1f}s)")
#---

#+++ Load pre-filtered fields
print("\n" + "="*60)
print("Loading pre-filtered fields...")
t0 = time.time()
filtered_filename = str(PP_OUTPUT / (Path(filename).stem + "_filtered_velocities_sweep.nc"))
ds_filt = xr.open_dataset(filtered_filename, decode_times=False).chunk(dict(time=1, filter_scale=1))
ds = ds.reindex(time=ds_filt.time).chunk(chunks)

filter_scales = ds_filt.filter_scale.values
print(f"  Loaded from: {filtered_filename}  ({time.time()-t0:.1f}s)")
print(f"  Filter length scales: {filter_scales}")
print(f"  Filter dimensions: x and y (horizontal)")
#---

#+++ Load sorted density (only when using fixed reference)
rho_sorted = dz_sorted = None
if fixed_reference:
    sorted_density_filename = str(PP_OUTPUT / (Path(filename).stem + f"_sorted_density{ref_suffix}.nc"))
    ds_sorted = xr.open_dataset(sorted_density_filename, decode_times=False).chunk(chunks)
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
                                            n_workers=n_workers)
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
# energy_transfer is fully dask-lazy (calculate_energy_transfer() never .load()s its own output); writing a
# lazy slice via .to_netcdf() computes it via dask's threaded scheduler *during* the write, with multiple
# threads writing into the same HDF5 file handle -- the same hang risk fixed for
# local_potential_energies_timeseries() in aux01_pe_functions.py. Loading one timestep at a time (not the
# whole energy_transfer up front -- that's the "hundreds of GB" case the comment below is about) keeps this
# bounded and makes each per-timestep write purely synchronous.
with ProgressBar(minimum=5, dt=5):
    for i in range(energy_transfer.sizes["time"]):
        tmp_f = str(tmp_dir / f"t{i:04d}.nc")
        energy_transfer.isel(time=[i]).load().to_netcdf(tmp_f)
        tmp_files.append(tmp_f)
        print(f"  wrote time {i+1}/{energy_transfer.sizes['time']}")

print("Merging per-timestep files...")
# Stream via dask (no .load(): the merged dataset is hundreds of GB and won't fit in RAM).
with xr.open_mfdataset(tmp_files, combine="by_coords", decode_timedelta=False,
                       parallel=False, chunks={"time": 1}) as merged:
    write_job = merged.to_netcdf(output_filename, compute=False)
    # Forced single-threaded: computing write_job under dask's default *threaded* scheduler means multiple
    # threads write into the same HDF5 file handle -- the same hang write_dataset() (aux00_utils.py) exists
    # to avoid for 01/03/04/05, confirmed as a real 3+ hour stall in sweep1's identical merge pattern (frozen
    # at 0% progress, not just slow). No "load"-style alternative offered, unlike write_dataset(): the whole
    # point of this per-timestep temp-file design is to never need the merged dataset fully in memory at once.
    with dask.config.set(scheduler="synchronous"):
        with ProgressBar(minimum=5, dt=5):
            write_job.compute()
for f in tmp_files:
    os.remove(f)
tmp_dir.rmdir()
print(f"Results saved to: {output_filename}")
#---
