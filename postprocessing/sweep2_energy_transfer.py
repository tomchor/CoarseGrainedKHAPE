#!/usr/bin/env python
#+++ Imports
import os
from pathlib import Path
import tempfile
import time
import numpy as np
import xarray as xr
from dask.diagnostics.progress import ProgressBar
from src.aux00_utils import (PP_OUTPUT, pad_margin_for_run, extension_for_run, halo_for_run, extension_suffix, load_dataset_and_grid,
                             scale_subset_tag)
from src.aux01_pe_functions import calculate_density_fields_from_buoyancy, sorted_timeseries
from src.aux02_ke_functions import calculate_energy_transfer
#---

#+++ Configuration
import argparse
parser = argparse.ArgumentParser(description="Calculate cross-scale KE and APE transfer terms")
parser.add_argument("--filename", default="output/khi_Nz1024_Ri0.10.nc", help="Path to simulation NetCDF file")
parser.add_argument("--n-workers", type=int, default=18, help="Number of CPU workers for APE sorting (ThreadPoolExecutor)")
parser.add_argument("--fixed-reference", action="store_true", default=False,
                    help="Load the fixed-in-time reference profile (produced by 01 with --fixed-reference)")
parser.add_argument("--extension", choices=["edge", "odd"], default="edge",
                    help="Which sweep1 run to read: the one filtered with wall-value extension ('edge', the "
                         "default) or with odd reflection ('odd'). Comparing the two measures how much the "
                         "extension choice moves the spectrum at large filter scale.")
parser.add_argument("--filter-scales", type=float, nargs="+", default=None,
                    help="The scales sweep1 was given with --filter-scales, if any: selects that run's tagged file "
                         "(e.g. _l20) and tags this output the same way. Omit for the full sweep.")
parser.add_argument("--keep-fields", action="store_true", default=False,
                    help="Also write the 4D fields (Π_K, Π_A, the SFS APE->KE exchange and w̄·b_rˡ), not just their "
                         "volume integrals. Every reader of the sweep uses only the integrals, and the fields are "
                         "about 860 GB at Nz=2048.")
split = parser.add_mutually_exclusive_group()
split.add_argument("--part", default=None, metavar="K/N",
                   help="Compute only the K-th of N contiguous, near-equal shares of the sweep's records (K = 1, ..., N) "
                        "and write them to a _part<K>of<N> file for --merge-parts to join. The memory grows with the "
                        "records a job holds, so this is how several jobs share a run too large for one node.")
split.add_argument("--merge-parts", type=int, default=None, metavar="N",
                   help="Compute nothing: join the N files of a `--part K/N` run into this sweep's output, after "
                        "checking that they hold each of the sweep's records once and in order, and delete them.")
args = parser.parse_args()

print("\n" + "="*70 + f"\n  {Path(__file__).name}\n  " + "  ".join(f"{k}={v}" for k,v in vars(args).items()) + "\n" + "="*70)
REPO_ROOT = Path(__file__).resolve().parent.parent
filename = str(REPO_ROOT / args.filename) if not os.path.isabs(args.filename) else args.filename
fixed_reference = args.fixed_reference
n_workers = args.n_workers
chunks = dict(time=1)
ref_suffix = "_fixed_ref" if fixed_reference else ""
scale_tag = scale_subset_tag(args.filter_scales)   # "" for the full sweep
# --extension picks *which* sweep1 run to read. The extension actually used is then taken from that file's
# own attributes, so the raw field here is extended exactly as sweep1 extended it -- the same contract
# `pad_margin` already has, and it catches a flag that disagrees with the file it names.
filtered_filename = str(PP_OUTPUT / (Path(filename).stem
                                     + f"_filtered_velocities_sweep{scale_tag}{extension_suffix(args.extension)}.nc"))
extension  = extension_for_run(filtered_filename)
ext_suffix = extension_suffix(extension)
if extension != args.extension:
    raise ValueError(f"--extension {args.extension!r} but {Path(filtered_filename).name} records "
                     f"z_extension={extension!r}; rerun sweep1 with --extension {args.extension}")
output_filename = str(PP_OUTPUT / (Path(filename).stem
                                   + f"_energy_transfer_sweep{ref_suffix}{scale_tag}{ext_suffix}.nc"))


def part_filename(k, n):
    return output_filename.removesuffix(".nc") + f"_part{k}of{n}.nc"


if args.part is not None:
    try:
        part_k, n_parts = (int(v) for v in args.part.split("/"))
    except ValueError:
        parser.error(f"--part takes K/N, e.g. 2/6, not {args.part!r}")
    if not 1 <= part_k <= n_parts:
        parser.error(f"--part {args.part}: K must be one of 1, ..., N")
#---

#+++ Join the parts of a split run (--merge-parts), and stop
if args.merge_parts is not None:
    part_files = [part_filename(k, args.merge_parts) for k in range(1, args.merge_parts + 1)]
    missing = [Path(f).name for f in part_files if not Path(f).exists()]
    if missing:
        raise FileNotFoundError(f"cannot join the parts: {missing} are not in {PP_OUTPUT}")
    with xr.open_dataset(filtered_filename, decode_times=False) as ds_filt:
        record_times = ds_filt.time.values
    with xr.open_mfdataset(part_files, combine="nested", concat_dim="time", decode_times=False, decode_timedelta=False,
                           parallel=False, chunks={"time": 1}) as merged:
        if not np.array_equal(merged.time.values, record_times):
            raise ValueError(f"the {args.merge_parts} parts hold {merged.sizes['time']} records, not the sweep's "
                             f"{len(record_times)} once each and in order: they are not one --part run of this sweep")
        for attr in ("sweep_part", "sweep_records"):
            merged.attrs.pop(attr, None)
        write_job = merged.to_netcdf(output_filename, compute=False)
        with ProgressBar(minimum=5, dt=5):
            write_job.compute()
    for f in part_files:
        os.remove(f)
    print(f"Joined {args.merge_parts} parts, {len(record_times)} records, into: {output_filename}")
    raise SystemExit(0)
#---

#+++ Load data and grid
print("\n" + "="*60)
print("Loading data and grid...")
t0 = time.time()
# Pad exactly as sweep1 did, so the raw field and the filtered fields it is differenced against sit on
# one grid. The margin has to come from the *sweep's* filtered file, not 01's: the sweep spans ℓ up to 20,
# whose 4σ margin is ~3x what the budget scales need, so 01's margin would pad the raw field shallower
# than ds_filt and xarray would quietly align the two to their intersection rather than raising. The halo
# comes from that file for the same reason: it says how much of the padding sweep1 kept in its arrays.
min_margin = pad_margin_for_run(filtered_filename, required=True)
halo = halo_for_run(filtered_filename)
ds = load_dataset_and_grid(filename, min_margin=min_margin, extension=extension, halo=halo)
print(f"  wall extension: {extension!r}, {'the whole padding' if halo is None else f'{halo} padded cell(s) per side'} in the arrays "
      f"(from {Path(filtered_filename).name})")
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
print(f"  Filter dimensions: x, y and z")
#---

#+++ This job's share of the records (--part)
# The frozen reference is the sort of the run's first record, whichever records this job computes, so its
# source is set aside before they are narrowed down.
first_record = ds[["b", "dV", "LxLy"]].isel(time=[0])
n_records = ds_filt.sizes["time"]
if args.part is not None:
    records = np.array_split(np.arange(n_records), n_parts)[part_k - 1]
    if records.size == 0:
        raise ValueError(f"--part {args.part}: the sweep has {n_records} records, fewer than its {n_parts} parts")
    ds_filt = ds_filt.isel(time=records)
    ds = ds.isel(time=records)
    print(f"\nPart {args.part}: records {records[0]}-{records[-1]} of 0-{n_records - 1} "
          f"(t = {float(ds_filt.time[0]):g} to {float(ds_filt.time[-1]):g})")
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
    ds_for_sort = first_record.copy()
    ds_for_sort.attrs.update(ds.attrs)
    ds_for_sort = calculate_density_fields_from_buoyancy(ds_for_sort, buoyancy_name="b", density_name="ρ")
    sorted_t0 = sorted_timeseries(ds_for_sort, field_to_sort="ρ", n_workers=1, fixed_reference=True)
    sorted_density = sorted_t0.isel(time=0, drop=True).expand_dims(time=ds_filt.time).chunk(chunks)
    rho_sorted = sorted_density.rho_sorted
    dz_sorted  = sorted_density.dz_sorted
    print(f"  Reference column built on {sorted_t0.sizes['z_1d_sorted']} slots: the padded domain, {int(ds.attrs['n_pad_z'])} padded z "
          f"cells per side in the arrays and {int(ds.attrs['n_pad_z_virtual'])} virtual  ({time.time()-t0:.1f}s)")
#---

#+++ Calculate cross-scale transfer terms
print("\n" + "="*60)
print("Calculating cross-scale transfer terms...")
energy_transfer = calculate_energy_transfer(ds, filter_scales,
                                            ds_filt=ds_filt,
                                            rho_sorted=rho_sorted,
                                            dz_sorted=dz_sorted,
                                            n_workers=n_workers,
                                            filtered_reference=True,
                                            frozen_reference=fixed_reference,
                                            integrals_only=not args.keep_fields)
print("\nDone!")
#---

#+++ Save results
print("\n" + "="*60)
print("Saving results...")
energy_transfer.attrs.update(ds.attrs)
energy_transfer.attrs["ape_reference"] = "filtered"
target_filename = output_filename
if args.part is not None:
    target_filename = part_filename(part_k, n_parts)
    energy_transfer.attrs.update(sweep_part=args.part, sweep_records=f"{records[0]}:{records[-1] + 1}")
if not args.keep_fields:
    # Every reader of the sweep (sweep3, plot4, compare_extension and the S scripts) uses only the ∫ integrals, which
    # calculate_energy_transfer has computed already: a few MB, written at once. The 4D fields behind them are kept only
    # when asked for: they are about 860 GB at Nz=2048 and 3.4 TB at Nz=4096.
    energy_transfer.to_netcdf(target_filename)
else:
    # A fresh directory per run: a shared one let concurrent runs delete each other's records, and a file left
    # by an interrupted run made the final rmdir fail after the output had been written.
    tmp_dir = Path(tempfile.mkdtemp(prefix=Path(target_filename).stem + "_tmp_", dir=PP_OUTPUT))
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
        write_job = merged.to_netcdf(target_filename, compute=False)
        with ProgressBar(minimum=5, dt=5):
            write_job.compute()
    for f in tmp_files:
        os.remove(f)
    tmp_dir.rmdir()
print(f"Results saved to: {target_filename}")
#---
