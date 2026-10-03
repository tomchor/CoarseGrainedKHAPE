#!/usr/bin/env python
# Exploratory: a 3D cutaway of one snapshot, three orthogonal cutting planes in an mplot3d axes. Needs nothing
# beyond matplotlib, so it runs wherever the rest of the pipeline does and in a batch job with no display.
# The planes are mutually orthogonal and drawn back to front, which is the one case mplot3d's missing z-buffer
# does not spoil. For a true volume render or an isosurface see X13_3d_isosurface.jl (GLMakie).
#+++ Imports
import logging
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from src.aux00_utils import EXTRA_FIGURES, load_dataset_and_grid
from src.aux03_plotting import run_label
#---

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
print = logging.info

#+++ Configuration
import argparse
parser = argparse.ArgumentParser(description="3D cutaway snapshot: three orthogonal cutting planes through the domain")
parser.add_argument("--filename", default="output/khi_Nz512_Ri0.10.nc", help="Path to simulation NetCDF file")
parser.add_argument("--field", default="b", help="Variable to colour the planes with: any in the file (b, w, ε, Π_K_ℓ1, ...), or one of "
                                                 "the derived DERIVED keys, which the file cannot hold because its ω is spanwise only")
parser.add_argument("--time", type=float, default=None, help="Target time (nearest available); default is the last record")
parser.add_argument("--zlim", type=float, default=4.0, help="Crop to |z| < zlim: Lz = 25h is mostly quiescent and wrecks the aspect")
parser.add_argument("--planes", type=float, nargs=3, default=(0.0, 0.0, 0.25), metavar=("FX", "FY", "FZ"),
                    help="Where to cut, as a fraction of each axis from its low end (0 = far wall, 1 = near wall). The default "
                         "puts the horizontal cut off the b = 0 interface, where a buoyancy plane shows nothing")
parser.add_argument("--clim-percentile", type=float, default=99.0, help="Percentile of |data| used to set the colour limits")
parser.add_argument("--cmap", default=None, help="Colormap; default is RdBu_r for a signed field, magma_r for a positive one")
parser.add_argument("--elev", type=float, default=22, help="Camera elevation [deg]")
parser.add_argument("--azim", type=float, default=-55, help="Camera azimuth [deg]")
parser.add_argument("--dpi", type=int, default=200, help="Figure resolution")
args = parser.parse_args()

print("\n" + "="*70 + f"\n  {Path(__file__).name}\n  " + "  ".join(f"{k}={v}" for k, v in vars(args).items()) + "\n" + "="*70)
REPO_ROOT = Path(__file__).resolve().parent.parent
EXTRA_FIGURES.mkdir(exist_ok=True)
filename = str(REPO_ROOT / args.filename) if not os.path.isabs(args.filename) else args.filename
stem = Path(filename).stem
#---

#+++ Derived fields
# The output carries ω on (x_faa, y_aca, z_aaf), which is the *spanwise* component alone — enough for the 2D
# slice figures and useless for 3D structure. u, v and w are all written at cell centres, so the full gradient
# tensor is a plain centred difference on one grid, with no interpolation between staggered locations.
def _vorticity(ds_t):
    u, v, w = ds_t["u"], ds_t["v"], ds_t["w"]
    return (w.differentiate("y_aca") - v.differentiate("z_aac"),
            u.differentiate("z_aac") - w.differentiate("x_caa"),
            v.differentiate("x_caa") - u.differentiate("y_aca"))

def _enstrophy(ds_t):
    ωx, ωy, ωz = _vorticity(ds_t)
    return (ωx**2 + ωy**2 + ωz**2).rename("enstrophy")

def _q_criterion(ds_t):
    """Q = ½(|Ω|² − |S|²), the Hunt et al. (1988) vortex criterion: Q > 0 where rotation beats strain."""
    comp = {(a, b): ds_t[u].differentiate(d) for a, u in ((0, "u"), (1, "v"), (2, "w"))
            for b, d in ((0, "x_caa"), (1, "y_aca"), (2, "z_aac"))}
    S2 = sum((0.5 * (comp[i, j] + comp[j, i]))**2 for i in range(3) for j in range(3))
    Ω2 = sum((0.5 * (comp[i, j] - comp[j, i]))**2 for i in range(3) for j in range(3))
    return (0.5 * (Ω2 - S2)).rename("Q")

def _speed(ds_t):
    return np.sqrt(ds_t["u"]**2 + ds_t["v"]**2 + ds_t["w"]**2).rename("speed")

DERIVED = {"enstrophy": _enstrophy, "Q": _q_criterion, "speed": _speed}
#---

#+++ Load one snapshot
print("Loading simulation dataset...")
ds = load_dataset_and_grid(filename, pad=False).chunk({"time": 1})
if args.field not in ds and args.field not in DERIVED:
    raise SystemExit(f"{args.field!r} is neither in {Path(filename).name} nor one of {sorted(DERIVED)}; "
                     f"the file has {sorted(ds.data_vars)}")

t_sel = float(ds.time[-1]) if args.time is None else float(ds.time.sel(time=args.time, method="nearest"))
ds_t = ds.sel(time=t_sel, method="nearest")
da = DERIVED[args.field](ds_t) if args.field in DERIVED else ds_t[args.field]
# The budget fields live on centres; ω and Ri sit on faces. Name the axes from the field itself so both work.
x_name = next(d for d in da.dims if d.startswith("x_"))
y_name = next(d for d in da.dims if d.startswith("y_"))
z_name = next(d for d in da.dims if d.startswith("z_"))
da = da.sel({z_name: slice(-args.zlim, +args.zlim)}).load()
print(f"  {args.field} at t = {t_sel:.1f}, cropped to |z| < {args.zlim}: {dict(da.sizes)}")
#---

#+++ Colour scale
x, y, z = da[x_name].values, da[y_name].values, da[z_name].values
signed = float(da.min()) < 0 < float(da.max())
cmap = plt.get_cmap(args.cmap or ("RdBu_r" if signed else "magma_r"))
if signed:
    v = float(np.nanpercentile(np.abs(da.values), args.clim_percentile)) or 1.0
    norm = Normalize(vmin=-v, vmax=+v)
else:
    lo, hi = (float(np.nanpercentile(da.values, 100 - args.clim_percentile)), float(np.nanpercentile(da.values, args.clim_percentile)))
    norm = Normalize(vmin=lo, vmax=hi if hi > lo else lo + 1.0)
print(f"  colour limits: [{norm.vmin:.3g}, {norm.vmax:.3g}]  ({'signed' if signed else 'positive'}, {cmap.name})")
#---

#+++ The three cutting planes
# Each plane is a plot_surface whose facecolors carry the data. facecolors describes *cells*, so it takes the
# data with one row and column dropped; the coordinate mesh keeps its full extent as the cell corners.
def corners(c):
    """Cell edges from cell centres, so a plane of N cells is drawn over its true extent rather than N-1 of it."""
    mid = 0.5 * (c[:-1] + c[1:])
    return np.concatenate([[c[0] - (mid[0] - c[0])], mid, [c[-1] + (c[-1] - mid[-1])]])

xc, yc, zc = corners(x), corners(y), corners(z)
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d", computed_zorder=False)

fx, fy, fz = args.planes
i_cut, j_cut, k_cut = (min(int(f * n), n - 1) for f, n in zip((fx, fy, fz), (len(x), len(y), len(z))))
print(f"  cutting at x = {x[i_cut]:.2f}, y = {y[j_cut]:.2f}, z = {z[k_cut]:.2f}")

X, Z = np.meshgrid(xc, zc, indexing="ij")
ax.plot_surface(X, np.full_like(X, yc[j_cut]), Z, rstride=1, cstride=1, shade=False,
                facecolors=cmap(norm(da.isel({y_name: j_cut}).transpose(x_name, z_name).values)))

Y, Z = np.meshgrid(yc, zc, indexing="ij")
ax.plot_surface(np.full_like(Y, xc[i_cut]), Y, Z, rstride=1, cstride=1, shade=False,
                facecolors=cmap(norm(da.isel({x_name: i_cut}).transpose(y_name, z_name).values)))

X, Y = np.meshgrid(xc, yc, indexing="ij")
ax.plot_surface(X, Y, np.full_like(X, zc[k_cut]), rstride=1, cstride=1, shade=False,
                facecolors=cmap(norm(da.isel({z_name: k_cut}).transpose(x_name, y_name).values)))
#---

#+++ Axes, colourbar, save
ax.set_box_aspect((xc[-1] - xc[0], yc[-1] - yc[0], zc[-1] - zc[0]))
ax.view_init(elev=args.elev, azim=args.azim)
ax.set_xlim(xc[0], xc[-1]); ax.set_ylim(yc[0], yc[-1]); ax.set_zlim(zc[0], zc[-1])
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")

cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.55, pad=0.08)
cb.set_label(args.field)
label = run_label(ds.attrs)
ax.set_title(f"{args.field}   t = {t_sel:.1f}" + (f"\n{label}" if label else ""), fontsize=11)

outfile = str(EXTRA_FIGURES / f"{stem}_3d_{args.field}_t{t_sel:.1f}.png")
fig.savefig(outfile, dpi=args.dpi, bbox_inches="tight")
print(f"Figure saved to: {outfile}")
#---
