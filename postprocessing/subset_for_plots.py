#!/usr/bin/env python
# Cut a 3D run down to something that fits on a laptop, for plotting with X13 locally.
#
#   python postprocessing/subset_for_plots.py --filename output/khi_Nz1024_Ri0.10.nc --times 100 120 140 --scale 1
#
# Named neither plot*.py nor X*.py on purpose: it writes data, not a figure, so neither plots.pbs's glob
# nor the extra-figures family picks it up. Run it on the HPC, scp the result, point X13 at it.
#
# Four reductions, in the order they matter: keep only the fields asked for, only the records asked for,
# crop z to the active layer (Lz = 25h is mostly quiescent), and store float32. On an Nz=1024 run that is
# about 0.91 GB per field per record down to 0.145 GB, and far less once z is cropped.
#
# The output has to stay readable by the plotting scripts, which means two things a plain `to_netcdf`
# of a sliced dataset would get wrong:
#   - load_dataset_and_grid reconstructs the grid from the NetCDF *groups*, which a plain rewrite drops. Those
#     are copied here, with the z extent and size in `underlying_grid_reconstruction_kwargs` rewritten to
#     match the crop -- otherwise Lz, z_min/z_max and dV would describe the uncropped domain.
#   - `virtual_rank_files` is dropped. A merged file on the HPC carries it and check_rank_files refuses
#     the file when the rank files are absent, which they will be once it has been copied away.
#+++ Imports
import logging
import os
import shutil
from pathlib import Path
import numpy as np
import netCDF4
import xarray as xr
from src.aux00_utils import model_grid_suffix
#---

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
print = logging.info

#+++ Configuration
import argparse
parser = argparse.ArgumentParser(description="Write a small, local-plotting-sized subset of a 3D run")
parser.add_argument("--filename", required=True, help="Path to the full 3D simulation NetCDF file")
parser.add_argument("--output", default=None, help="Output path; default is <stem>_subset.nc beside the input")
parser.add_argument("--fields", default="u,v,w,b,wb_rs,Π_K,Π_A,ε_Ks,ε_As",
                    help="Comma-separated variables to keep. Bare budget names get the _ℓ<scale> suffix (see --scale). "
                         "u, v and w are what X13 derives Q, enstrophy and speed from, so keep them unless sure")
parser.add_argument("--scale", default="1", help="Filter scale ℓ whose per-scale budget terms to keep ('' to disable the suffixing)")
parser.add_argument("--times", type=float, nargs="+", default=None, help="Target times to keep (nearest record each); default every record")
parser.add_argument("--every", type=int, default=None, help="Instead of --times, keep every Nth record")
parser.add_argument("--zlim", type=float, default=5.0,
                    help="Keep |z| < zlim. Deliberately wider than the plotting scripts' own 4: they derive Q, enstrophy "
                         "and speed by centred differences, which need a cell on each side, so cropping here to exactly "
                         "what gets plotted leaves the derived fields one-sided on the crop boundary")
parser.add_argument("--dtype", default="float32", choices=["float32", "float64"], help="Storage type for the field data")
parser.add_argument("--complevel", type=int, default=4, help="zlib compression level, 0 to disable")
args = parser.parse_args()

print("\n" + "="*70 + f"\n  {Path(__file__).name}\n  " + "  ".join(f"{k}={v}" for k, v in vars(args).items()) + "\n" + "="*70)
REPO_ROOT = Path(__file__).resolve().parent.parent
filename = str(REPO_ROOT / args.filename) if not os.path.isabs(args.filename) else args.filename
outfile = args.output or str(Path(filename).with_name(Path(filename).stem + "_subset.nc"))
#---

#+++ Open and resolve the field names
ds = xr.open_dataset(filename, decode_times=False, chunks={"time": 1})
if model_grid_suffix(ds):
    raise SystemExit("this file holds more than one grid (a --save_sorted run). Subsetting the sorted column too is not "
                     "handled here; rerun the simulation without --save_sorted, or extend this script")


def resolve(name):
    """A bare budget name picks up the _ℓ<scale> suffix every per-scale term carries."""
    if name in ds:
        return name
    suffixed = f"{name}_ℓ{args.scale}" if args.scale else name
    if suffixed in ds:
        return suffixed
    raise SystemExit(f"neither {name!r} nor {suffixed!r} is in {Path(filename).name}.\nIt has: {sorted(ds.data_vars)}")


wanted = [resolve(f) for f in args.fields.split(",")]
print(f"Keeping {len(wanted)} fields: {', '.join(wanted)}")

# Every time-independent variable comes along: these are the grid's own (Δx_caa, Δy_aca, Δz_aac, ...), which
# load_dataset_and_grid builds dV from, so it fails on a file without them. They are 1D and cost nothing.
grid_vars = [v for v in ds.data_vars if "time" not in ds[v].dims and ds[v].size <= ds.sizes.get("z_aaf", 0) + 1]
wanted += [v for v in grid_vars if v not in wanted]
print(f"Keeping {len(grid_vars)} grid variables: {', '.join(grid_vars)}")
#---

#+++ Select records
if args.times is not None:
    idx = sorted({int(np.abs(ds.time.values - t).argmin()) for t in args.times})
elif args.every is not None:
    idx = list(range(0, ds.sizes["time"], args.every))
else:
    idx = list(range(ds.sizes["time"]))
out = ds[wanted].isel(time=idx)
print(f"Keeping {len(idx)} of {ds.sizes['time']} records: t = {np.round(out.time.values, 2).tolist()}")
#---

#+++ Crop z
# Both z axes are cropped: the centres by |z| < zlim, the faces to the ones that bound those cells, so a
# field on z_aaf (ω, Ri) stays consistent with the cells it straddles.
z_c = ds["z_aac"].values
keep = np.flatnonzero(np.abs(z_c) <= args.zlim)
if keep.size == 0:
    raise SystemExit(f"--zlim {args.zlim} keeps no cells; z runs {z_c.min():.2f} to {z_c.max():.2f}")
k0, k1 = int(keep[0]), int(keep[-1])
sel = {"z_aac": slice(k0, k1 + 1)}
if "z_aaf" in ds.dims:
    sel["z_aaf"] = slice(k0, k1 + 2)   # one more face than cells
out = out.isel({d: s for d, s in sel.items() if d in out.dims})
z_faces = ds["z_aaf"].values if "z_aaf" in ds else None
z_lo = float(z_faces[k0]) if z_faces is not None else float(z_c[k0] - 0.5 * (z_c[1] - z_c[0]))
z_hi = float(z_faces[k1 + 1]) if z_faces is not None else float(z_c[k1] + 0.5 * (z_c[1] - z_c[0]))
print(f"Cropping z to {k1 - k0 + 1} of {z_c.size} cells: faces {z_lo:.4f} to {z_hi:.4f}")
if args.zlim <= 4.0:
    print(f"  note: X13 defaults to plotting |z| < 4 and derives Q by centred differences, so a subset cropped to "
          f"{args.zlim} leaves them one-sided at the boundary. Plot with a smaller --zlim, or subset with a larger one.")
#---

#+++ Write
# Coordinates stay float64 -- they are tiny, and the grid arithmetic downstream reads them.
encoding = {}
for name, da in out.data_vars.items():
    enc = {"zlib": args.complevel > 0, "complevel": args.complevel}
    if args.dtype == "float32" and np.issubdtype(da.dtype, np.floating):
        enc["dtype"] = "float32"
    encoding[name] = enc

out.attrs = {k: v for k, v in ds.attrs.items() if k != "virtual_rank_files"}
out.attrs["subset_of"] = Path(filename).name
out.attrs["subset_zlim"] = args.zlim

tmp = outfile + ".tmp"
print(f"Writing {outfile} ...")
out.to_netcdf(tmp, encoding=encoding)
ds.close()

# Copy the grid groups, which xarray does not carry across, and make the model grid's z match the crop.
with netCDF4.Dataset(filename) as src, netCDF4.Dataset(tmp, "a") as dst:
    for name in ("underlying_grid_reconstruction_args", "underlying_grid_reconstruction_kwargs",
                 "grid_reconstruction_metadata"):
        if name not in src.groups:
            continue
        g_src, g_dst = src.groups[name], dst.createGroup(name)
        attrs = {k: g_src.getncattr(k) for k in g_src.ncattrs()}
        if name == "underlying_grid_reconstruction_kwargs":
            attrs["z"] = np.array([z_lo, z_hi])
            size = np.array(attrs["size"], dtype=int).copy()
            size[2] = k1 - k0 + 1
            attrs["size"] = size
        g_dst.setncatts(attrs)
    print(f"Copied grid groups; model grid z -> [{z_lo:.4f}, {z_hi:.4f}], Nz -> {k1 - k0 + 1}")

shutil.move(tmp, outfile)
before, after = Path(filename).stat().st_size, Path(outfile).stat().st_size
print(f"Done: {before/2**30:.2f} GiB -> {after/2**30:.3f} GiB  ({before/after:.0f}x smaller)")
print(f"Copy it over, then e.g.:\n"
      f"  julia --project postprocessing/X13_3d_volume.jl {Path(outfile).name} "
      f"--field Q,wb_rs,Π_K,Π_A,ε_Ks,ε_As --scale {args.scale} --mode isosurface")
#---
