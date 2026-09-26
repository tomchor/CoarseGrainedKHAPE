#!/usr/bin/env python
#+++ Imports
import os
from pathlib import Path
import time
import xarray as xr
from dask.diagnostics.progress import ProgressBar
from src.aux00_utils import (PP_OUTPUT, pad_margin_for_run, extension_for_run, extension_suffix, reference_suffix,
                            load_dataset_and_grid)
from src.aux01_pe_functions import calculate_density_fields_from_buoyancy, sorted_timeseries
from src.aux02_ke_functions import calculate_energy_transfer
#---

#+++ Configuration
import argparse
parser = argparse.ArgumentParser(description="Calculate cross-scale KE and APE transfer terms")
parser.add_argument("--filename", default="output/khi_Nz1024_Ri0.10.nc", help="Path to simulation NetCDF file")
parser.add_argument("--n-workers", type=int, default=18, help="Number of CPU workers for APE sorting (ThreadPoolExecutor)")
parser.add_argument("--fixed-reference", action="store_true", default=False, help="Load the fixed-in-time reference profile (produced by 01 with --fixed-reference)")
parser.add_argument("--extension", choices=["edge", "odd"], default="edge",
                    help="Which sweep1 run to read: the one filtered with wall-value extension ('edge', the "
                         "default) or with odd reflection ('odd'). Comparing the two measures how much the "
                         "extension choice moves the spectrum at large filter scale.")
parser.add_argument("--reference", choices=["filtered", "true"], default="filtered",
                    help="Reference state the resolved scale is measured against. 'filtered' (default) uses the "
                         "vertically filtered profile ⟨ρ_*⟩, valid for a kernel with vertical extent. 'true' uses the "
                         "unfiltered ρ_*, the horizontal-filter limit. The sweep spans filter scales, so ⟨ρ_*⟩ is "
                         "rebuilt at each one.")
args = parser.parse_args()

print("\n" + "="*70 + f"\n  {Path(__file__).name}\n  " + "  ".join(f"{k}={v}" for k,v in vars(args).items()) + "\n" + "="*70)
REPO_ROOT = Path(__file__).resolve().parent.parent
filename = str(REPO_ROOT / args.filename) if not os.path.isabs(args.filename) else args.filename
fixed_reference = args.fixed_reference
filtered_reference = args.reference == "filtered"
n_workers = args.n_workers
chunks = dict(time=1)
ref_suffix = "_fixed_ref" if fixed_reference else ""
# --extension picks *which* sweep1 run to read. The extension actually used is then taken from that file's
# own attributes, so the raw field here is extended exactly as sweep1 extended it -- the same contract
# `pad_margin` already has, and it catches a flag that disagrees with the file it names.
filtered_filename = str(PP_OUTPUT / (Path(filename).stem
                                     + f"_filtered_velocities_sweep{extension_suffix(args.extension)}.nc"))
extension  = extension_for_run(filtered_filename)
ext_suffix = extension_suffix(extension)
if extension != args.extension:
    raise ValueError(f"--extension {args.extension!r} but {Path(filtered_filename).name} records "
                     f"z_extension={extension!r}; rerun sweep1 with --extension {args.extension}")
#---

#+++ Load data and grid
print("\n" + "="*60)
print("Loading data and grid...")
t0 = time.time()
# Pad exactly as sweep1 did, so the raw field and the filtered fields it is differenced against sit on
# one grid. The margin has to come from the *sweep's* filtered file, not 01's: the sweep spans ℓ up to 20,
# whose 4σ margin is ~3x what the budget scales need, so 01's margin would pad the raw field shallower
# than ds_filt and xarray would quietly align the two to their intersection rather than raising.
ds = load_dataset_and_grid(filename, min_margin=pad_margin_for_run(filtered_filename, required=True),
                           extension=extension)
print(f"  wall extension: {extension!r} (from {Path(filtered_filename).name})")
ds = ds.chunk(chunks)
print(f"Dataset loaded: {len(ds.time)} time steps  ({time.time()-t0:.1f}s)")
#---

#+++ Load pre-filtered fields
print("\n" + "="*60)
print("Loading pre-filtered fields...")
t0 = time.time()
ds_filt = xr.open_dataset(filtered_filename, decode_times=False).chunk(dict(time=1, filter_scale=1))
ds = ds.reindex(time=ds_filt.time).chunk(chunks)
# A non-edge sweep1 file written before the rule was restricted to b also reflected u and w, while `ds`
# above is padded with the restricted rule, so the two would disagree in the padding. Refuse it.
if extension != "edge" and ds_filt.attrs.get("z_extension_vars") != ds.attrs["z_extension_vars"]:
    raise ValueError(f"{Path(filtered_filename).name} was extended with z_extension_vars="
                     f"{ds_filt.attrs.get('z_extension_vars')!r}, not {ds.attrs['z_extension_vars']!r}: it predates "
                     f"restricting --extension {extension} to the buoyancy. Rerun sweep1 with --extension {extension}.")

filter_scales = ds_filt.filter_scale.values
print(f"  Loaded from: {filtered_filename}  ({time.time()-t0:.1f}s)")
print(f"  Filter length scales: {filter_scales}")
print(f"  Filter dimensions: x and z")
#---

#+++ Build the frozen reference column (only when using fixed reference)
rho_sorted = dz_sorted = None
if fixed_reference:
    # Built here rather than read from 02's `_sorted_density_fixed_ref.nc`. The column's z✶ are the padded
    # grid's own heights, and the sweep pads to 4σ of ℓ=20 while 02 pads to the budget scales -- so 02's
    # column belongs to a different grid and cannot be reused: reading it would shift every displacement
    # with nothing downstream to reveal it. Nothing is lost by rebuilding -- the frozen reference is the
    # sort of t=0 alone, broadcast over the time axis, so this is one sort rather than one per output.
    t0 = time.time()
    print("\n" + "="*60)
    print("Sorting t=0 density for the frozen reference (on the sweep's own padded grid)...")
    # isel *before* sorting: `sorted_timeseries` does `ds[field].values`, which would otherwise pull the
    # whole padded density timeseries into RAM only to read row 0.
    ds_for_sort = ds[["b", "dV", "LxLy"]].isel(time=[0]).copy()
    ds_for_sort.attrs.update(ds.attrs)
    ds_for_sort = calculate_density_fields_from_buoyancy(ds_for_sort, buoyancy_name="b", density_name="ρ")
    sorted_t0 = sorted_timeseries(ds_for_sort, field_to_sort="ρ", n_workers=1, fixed_reference=True)
    sorted_density = sorted_t0.isel(time=0, drop=True).expand_dims(time=ds_filt.time).chunk(chunks)
    rho_sorted = sorted_density.rho_sorted
    dz_sorted  = sorted_density.dz_sorted
    print(f"  Reference column built on {ds.sizes['z_aac']} padded z cells (n_pad_z={int(ds.attrs['n_pad_z'])})  ({time.time()-t0:.1f}s)")
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
energy_transfer.attrs["ape_reference"] = args.reference
output_filename = str(PP_OUTPUT / (Path(filename).stem
                                   + f"_energy_transfer_sweep{ref_suffix}{reference_suffix(args.reference)}{ext_suffix}.nc"))
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
