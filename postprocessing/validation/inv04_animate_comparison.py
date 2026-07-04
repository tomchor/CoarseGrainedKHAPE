#!/usr/bin/env python
#+++ Imports
import logging
import os
import sys
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # postprocessing/ on path for `src.*`
from src.aux00_utils import (load_dataset_and_grid, make_gaussian_filter, condense_uw_velocities)
from src.aux02_ke_functions import (calculate_sfs_stress_tensor, calculate_strain_tensor,
                                    calculate_cross_scale_ke_flux, calculate_sfs_ke_dissipation)
from src.aux03_plotting import run_label
#---

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
print = logging.info

#+++ Configuration
import argparse
parser = argparse.ArgumentParser(description="Animate online (simulation-time) vs offline (post-processed) diagnostics over time: "
                                             "one row per filter scale, columns = online | offline | difference.")
parser.add_argument("--filename", default="output/khi_Nz256_Ri0.10.nc", help="Path to simulation NetCDF file")
parser.add_argument("--filter-scales", type=float, nargs="+", default=[1, 7], help="Filter ℓ (FWHM) values matching the online widths")
parser.add_argument("--field", default="Π_K", choices=["Π_K", "ε_Ks", "u", "w", "b"],
                    help="Quantity to compare: cross-scale KE transfer Π_K (default), SFS KE dissipation ε_Ks, or a filtered field")
parser.add_argument("--z-window", type=float, default=6.0, help="Half-height of the z window (in units of h) shown in the maps")
parser.add_argument("--fps", type=int, default=12, help="Frames per second")
parser.add_argument("--dpi", type=int, default=150, help="DPI for output video")
parser.add_argument("--clim-percentile", type=float, default=99, help="Percentile of |data| used to set symmetric color limits")
parser.add_argument("--max-frames", type=int, default=None, help="Cap on number of animation frames (subsamples in time; for quick tests)")
args = parser.parse_args()

print("\n" + "="*70 + f"\n  {Path(__file__).name}\n  " + "  ".join(f"{k}={v}" for k, v in vars(args).items()) + "\n" + "="*70)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # validation/ → postprocessing/ → repo root
ANIMATIONS = REPO_ROOT / "animations"
ANIMATIONS.mkdir(exist_ok=True)
filename = str(REPO_ROOT / args.filename) if not os.path.isabs(args.filename) else args.filename
stem = Path(filename).stem
field = args.field
field_tag = {"Π_K": "Pi_K", "ε_Ks": "eps_Ks", "u": "u", "w": "w", "b": "b"}[field]


def online_name(name, ℓ):
    """Online output variable name for filter scale ℓ (matches the Julia Symbol naming)."""
    ℓ_tag = int(ℓ) if ℓ == int(ℓ) else ℓ
    return f"{name}_ℓ{ℓ_tag}"
#---

#+++ Load dataset, recover the original (unpadded, windowed) z extent
# load_dataset_and_grid pads z to 2× (edge values); this pads both the online variables and the
# fields used to recompute them offline, so they share a grid. We recover the original z extent from
# the grid group and additionally crop to ±z_window for display.
print("Loading simulation data...")
ds = load_dataset_and_grid(filename)
ds = ds.chunk({"time": 1})

grid = xr.open_dataset(filename, group="underlying_grid_reconstruction_kwargs")
z0, z1 = float(grid.z.min()), float(grid.z.max())          # original (unpadded) z faces
zw = args.z_window
z_lo, z_hi = max(z0, -zw), min(z1, zw)
window = dict(z_aac=slice(z_lo, z_hi))

# Select (optionally subsampled) animation times.
times_all = ds.time.values
if args.max_frames is not None and len(times_all) > args.max_frames:
    idx = np.linspace(0, len(times_all) - 1, args.max_frames, dtype=int)
    times = times_all[idx]
else:
    times = times_all
print(f"Animating {len(times)} frames (of {len(times_all)} available)")
#---

#+++ Build online + offline cubes for each filter scale and materialize the cropped window
# For Π_K we recompute the full offline pipeline (filter → τ, S̄ → Π_K = −τⁱʲ S̄ⁱʲ); for a plain
# filtered field we just apply the offline Gaussian filter. Both are cropped to the display window
# and loaded into memory so the animation can index numpy arrays directly.
filtered_dimensions = ["x_caa", "z_aac"]
tensor_dimensions   = ("x_caa", "z_aac")

if field in ("Π_K", "ε_Ks"):
    uᵢ = condense_uw_velocities(ds, indices=(1, 3))["uᵢ"]   # drops u,w from ds; keeps everything else
if field == "ε_Ks":
    S = calculate_strain_tensor(uᵢ, dimensions=tensor_dimensions)   # full-flow strain (scale-independent)

panels = {}   # ℓ -> dict(online=, offline=, diff=)
print(f"Building online/offline cubes for field '{field}'...")
for ℓ in args.filter_scales:
    on_name = online_name(field, ℓ)
    if on_name not in ds:
        print(f"  WARNING: online field '{on_name}' not in dataset, skipping ℓ={ℓ}")
        continue

    gf = make_gaussian_filter(ℓ, ds)
    if field == "Π_K":
        ūᵢ = gf.apply(uᵢ, dims=filtered_dimensions)
        τ  = calculate_sfs_stress_tensor(uᵢ, gf, filter_dims=filtered_dimensions, filtered_u_i=ūᵢ)
        S̄  = calculate_strain_tensor(ūᵢ, dimensions=tensor_dimensions)
        off = calculate_cross_scale_ke_flux(τ, S̄)
    elif field == "ε_Ks":
        off = calculate_sfs_ke_dissipation(S, ds.ν, gf, filter_dims=filtered_dimensions)
    else:
        off = gf.apply(ds[field], dims=filtered_dimensions)

    on = ds[on_name]
    sel = dict(time=times)
    on  = on.sel(**sel).sel(**window).squeeze(drop=True).compute()
    off = off.sel(**sel).sel(**window).squeeze(drop=True).compute()
    panels[ℓ] = dict(online=on, offline=off, diff=(on - off))
    print(f"  ℓ = {ℓ:>4g}: online + offline cubes ready ({dict(on.sizes)})")

scales = [ℓ for ℓ in args.filter_scales if ℓ in panels]
if not scales:
    raise SystemExit(f"No online '{field}' fields found — run the simulation with the online diagnostics first.")
#---

#+++ Color limits (symmetric, shared between online & offline within a scale; diff autoscaled)
pct = args.clim_percentile
clim = {}
for ℓ in scales:
    p = panels[ℓ]
    vmax = max(float(np.nanpercentile(np.abs(p["online"].values), pct)),
               float(np.nanpercentile(np.abs(p["offline"].values), pct)))
    vmax = vmax if vmax > 0 else 1.0
    dmax = float(np.nanpercentile(np.abs(p["diff"].values), pct))
    dmax = dmax if dmax > 0 else vmax * 1e-6
    clim[ℓ] = dict(vmax=vmax, dmax=dmax)
#---

#+++ Per-scale, time-mean match metric (printed for confidence)
print(f"\nTime-mean match for '{field}' (rms over the windowed domain):")
for ℓ in scales:
    p = panels[ℓ]
    rms_diff   = float(np.sqrt(np.nanmean(p["diff"].values**2)))
    rms_online = float(np.sqrt(np.nanmean(p["online"].values**2)))
    rel = rms_diff / rms_online if rms_online > 0 else float("inf")
    print(f"  ℓ={ℓ:>4g}: rms(diff)/rms(online) = {rel:.2e},  max|diff| = {float(np.nanmax(np.abs(p['diff'].values))):.2e}")
#---

#+++ Set up the figure (one row per scale × 3 columns)
label = run_label(ds.attrs)
n_scales = len(scales)
x = panels[scales[0]]["online"]["x_caa"].values
z = panels[scales[0]]["online"]["z_aac"].values

fig, axes = plt.subplots(n_scales, 3, figsize=(15, 3.6 * n_scales), constrained_layout=True, squeeze=False)
col_titles = [f"Online {field}", f"Offline {field}", "Difference (online − offline)"]


def frame_array(da, frame):
    return da.isel(time=frame).transpose("z_aac", "x_caa").values


positive = field == "ε_Ks"   # ε_Kˢ is non-negative: sequential map, zero-based limits for online/offline
meshes = []
for i, ℓ in enumerate(scales):
    p = panels[ℓ]
    vmax, dmax = clim[ℓ]["vmax"], clim[ℓ]["dmax"]
    fmap, fmin = ("magma", 0) if positive else ("RdBu_r", -vmax)
    specs = [("online", fmin, vmax, fmap), ("offline", fmin, vmax, fmap), ("diff", -dmax, dmax, "RdBu_r")]
    row_meshes = []
    for k, (key, vmin, vmx, cmap) in enumerate(specs):
        ax = axes[i, k]
        im = ax.pcolormesh(x, z, frame_array(p[key], 0), cmap=cmap, vmin=vmin, vmax=vmx,
                           shading="nearest", rasterized=True)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        if i == 0:
            ax.set_title(col_titles[k], fontsize=12)
        ax.set_ylim(z_lo, z_hi)
        ax.set_aspect("equal")
        ax.set_xlabel("x" if i == n_scales - 1 else "")
        ax.tick_params(labelbottom=(i == n_scales - 1), labelleft=(k == 0))
        row_meshes.append(im)
    axes[i, 0].set_ylabel(f"ℓ = {ℓ:g}\nz", fontsize=12)
    meshes.append(row_meshes)

suptitle_base = f"Online vs offline {field}"
if label:
    suptitle_base += f"   {label}"
suptitle = fig.suptitle(f"{suptitle_base}   t = {times[0]:.1f}", fontsize=13)
#---

#+++ Animate and save
def update(frame):
    if frame % 10 == 0:
        print(f"  Frame {frame+1}/{len(times)}  (t = {times[frame]:.1f})")
    for i, ℓ in enumerate(scales):
        p = panels[ℓ]
        for im, key in zip(meshes[i], ("online", "offline", "diff")):
            im.set_array(frame_array(p[key], frame).ravel())
    suptitle.set_text(f"{suptitle_base}   t = {times[frame]:.1f}")
    return [im for row in meshes for im in row] + [suptitle]


outfile = str(ANIMATIONS / f"{stem}_{field_tag}_online_offline_comparison.mp4")
print(f"\nRecording {len(times)} frames at {args.fps} fps...")
writer = FFMpegWriter(fps=args.fps, metadata=dict(title=f"Online vs offline {field}"))
anim = FuncAnimation(fig, update, frames=len(times), blit=False, cache_frame_data=False)
anim.save(outfile, writer=writer, dpi=args.dpi)
del anim
plt.close(fig)
print(f"Animation saved to: {outfile}")
#---
