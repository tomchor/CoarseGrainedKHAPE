#!/usr/bin/env python
#+++ Imports
import logging
import os
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation, FFMpegWriter
from aux00_utils import load_dataset_and_grid
from aux03_plotting import run_label, budget_colors
#---

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
print = logging.info

#+++ Configuration
import argparse
parser = argparse.ArgumentParser(description="Animate 2×3 panels: vorticity, buoyancy, and SFS budget fields")
parser.add_argument("--filename", default="output/khi_Nz2048_Ri0.10.nc", help="Path to simulation NetCDF file")
parser.add_argument("--filter-scale", type=float, default=0.4, help="Target filter length scale")
parser.add_argument("--clim-percentile", type=float, default=99, help="Percentile of |data| used to set symmetric color limits")
parser.add_argument("--zlim", type=float, default=3.5, help="Vertical extent for z-axis (symmetric around 0)")
parser.add_argument("--fps", type=int, default=12, help="Frames per second")
parser.add_argument("--dpi", type=int, default=150, help="DPI for output video")
parser.add_argument("--fixed-reference", action="store_true", default=False, help="Load the fixed-in-time reference profile outputs")
args = parser.parse_args()
print("\n" + "="*70 + f"\n  {Path(__file__).name}\n  " + "  ".join(f"{k}={v}" for k,v in vars(args).items()) + "\n" + "="*70)

REPO_ROOT = Path(__file__).resolve().parent.parent
PP_OUTPUT = REPO_ROOT / "postprocessing" / "output"
FIGURES   = REPO_ROOT / "figures"
ANIMATIONS = REPO_ROOT / "animations"
ANIMATIONS.mkdir(exist_ok=True)
filename = str(REPO_ROOT / args.filename) if not os.path.isabs(args.filename) else args.filename
stem = Path(filename).stem
filename_2d = filename.replace(".nc", "_2d.nc")
ref_suffix = "_fixed_ref" if args.fixed_reference else ""
#---

#+++ Load datasets
print("Loading 2D simulation output...")
ds_2d = xr.open_dataset(filename_2d, decode_times=False)
ds_2d = ds_2d.sel(z_aac=slice(-args.zlim, args.zlim), z_aaf=slice(-args.zlim, args.zlim))

print("Loading KE and APE budget fields...")
ke_budget = xr.open_dataset(str(PP_OUTPUT / f"{stem}_sfs_ke_budget_fields{ref_suffix}.nc"), decode_times=False)
ape_budget = xr.open_dataset(str(PP_OUTPUT / f"{stem}_sfs_ape_budget_fields{ref_suffix}.nc"), decode_times=False)
ke_budget = ke_budget.sel(z_aac=slice(-args.zlim, args.zlim))
ape_budget = ape_budget.sel(z_aac=slice(-args.zlim, args.zlim))

ℓ_sel = float(ke_budget.filter_scale.sel(filter_scale=args.filter_scale, method="nearest"))
print(f"Selected filter scale: ℓ = {ℓ_sel:.4f}  (requested {args.filter_scale})")
ke_budget = ke_budget.sel(filter_scale=ℓ_sel)
ape_budget = ape_budget.sel(filter_scale=ℓ_sel)

print("Loading integrated budgets...")
ke_int = xr.open_dataset(str(PP_OUTPUT / f"{stem}_sfs_ke_budget_integrated{ref_suffix}.nc"), decode_timedelta=False)
ape_int = xr.open_dataset(str(PP_OUTPUT / f"{stem}_sfs_ape_budget_integrated{ref_suffix}.nc"), decode_timedelta=False)
ke_int = ke_int.sel(filter_scale=ℓ_sel, method="nearest")
ape_int = ape_int.sel(filter_scale=ℓ_sel, method="nearest")
#---

#+++ Reindex 2D data to budget time coordinate
print("Reindexing 2D data to budget time coordinate...")
ds_2d = ds_2d.reindex(time=ke_budget.time, method="nearest")
times = ke_budget.time.values
print(f"Aligned {len(times)} time steps")
#---

#+++ Extract coordinate arrays
x = ke_budget["x_caa"].values
z = ke_budget["z_aac"].values
#---

#+++ Compute global color limits from a few representative snapshots
print("Computing color limits...")
sample_idx = np.linspace(0, len(times) - 1, min(10, len(times)), dtype=int)

def global_clim_symmetric(da, sample_idx, pct):
    vals = np.concatenate([np.abs(da.isel(time=i).values).ravel() for i in sample_idx])
    return np.nanpercentile(vals, pct)

def global_clim_positive(da, sample_idx, pct):
    vals = np.concatenate([da.isel(time=i).values.ravel() for i in sample_idx])
    return np.nanpercentile(vals, pct)

pct = args.clim_percentile

omega_vmax = global_clim_symmetric(ds_2d["ω"].squeeze("y_aca", drop=True), sample_idx, pct)
b_vmax = global_clim_symmetric(ds_2d["b"].squeeze("y_aca", drop=True), sample_idx, pct)

pi_ke_vmax = max(global_clim_symmetric(ke_budget["Π_KE"].squeeze("y_aca"), sample_idx, pct),
                 global_clim_symmetric(ke_budget["SFS APE->KE exchange"].squeeze("y_aca"), sample_idx, pct))
pi_ape_vmax = global_clim_symmetric(ape_budget["Π_APE"].squeeze("y_aca"), sample_idx, pct)
chi_s_vmax = global_clim_positive(ape_budget["χₛ"].squeeze("y_aca"), sample_idx, pct)

print(f"  ω: ±{omega_vmax:.3e},  b: ±{b_vmax:.3e}")
print(f"  Π_KE/exchange: ±{pi_ke_vmax:.3e},  Π_APE: ±{pi_ape_vmax:.3e},  χₛ: 0–{chi_s_vmax:.3e}")
#---

#+++ Helper functions
def get_xz_dims(ds, var):
    """Return (x_dim_name, z_dim_name, x_coords, z_coords) for a variable."""
    da = ds[var]
    xdim = next(d for d in da.dims if "x" in d)
    zdim = next(d for d in da.dims if "z" in d)
    return xdim, zdim, da[xdim].values, da[zdim].values

def get_frame(ds, var, xdim, zdim, idx):
    """Return 2D array in (nz, nx) order for pcolormesh."""
    return ds[var].isel(time=idx).squeeze().transpose(zdim, xdim).values
#---

#+++ Set up figure with GridSpec (2 snapshot rows + 2 budget rows)
print("Setting up figure...")
fig = plt.figure(figsize=(16, 13))
gs = gridspec.GridSpec(4, 3, figure=fig, height_ratios=[1, 1, 0.7, 0.7],
                       hspace=0.22, wspace=0.05, left=0.06, right=0.99, top=0.95, bottom=0.05)

snapshot_axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(2)])
ax_ke_budget = fig.add_subplot(gs[2, :])
ax_ape_budget = fig.add_subplot(gs[3, :])
#---

#+++ Snapshot panels (rows 0–1)
panel_specs = [
    (0, 0, ds_2d,      "ω",                    r"Vorticity ($\omega$)",                   "RdBu_r",  -omega_vmax, omega_vmax),
    (0, 1, ds_2d,      "b",                    r"Buoyancy ($b$)",                         "RdBu_r",  -b_vmax,     b_vmax),
    (0, 2, ke_budget,  "Π_KE",                 r"$\Pi_{KE}$ (cross-scale KE flux)",       "RdBu_r",  -pi_ke_vmax, pi_ke_vmax),
    (1, 0, ape_budget, "Π_APE",                r"$\Pi_{APE}$ (cross-scale APE flux)",     "RdBu_r",  -pi_ape_vmax, pi_ape_vmax),
    (1, 1, ke_budget,  "SFS APE->KE exchange", r"Small-scale APE$\to$KE exchange",        "RdBu_r",  -pi_ke_vmax, pi_ke_vmax),
    (1, 2, ape_budget, "χₛ",                   r"$\chi_s$ (small-scale APE dissipation)",  "inferno", 0,          chi_s_vmax),
]

meshes = []
panel_dims = []
for row, col, ds, var, title, cmap, vmin, vmax in panel_specs:
    ax = snapshot_axes[row, col]
    xdim, zdim, xc, zc = get_xz_dims(ds, var)
    panel_dims.append((xdim, zdim))
    data0 = get_frame(ds, var, xdim, zdim, 0)
    im = ax.pcolormesh(xc, zc, data0, cmap=cmap, vmin=vmin, vmax=vmax, shading="nearest", rasterized=True)
    fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02, aspect=20)
    ax.set_title(title, fontsize=11)
    ax.set_ylim(-args.zlim, args.zlim)
    ax.set_aspect("auto")
    ax.set_ylabel("z" if col == 0 else "")
    ax.set_xlabel("")
    ax.tick_params(labelbottom=(row == 1))
    meshes.append(im)

for ax, letter in zip(snapshot_axes.flat, "abcdef"):
    ax.text(0.02, 0.97, f"({letter})", transform=ax.transAxes, fontsize=12, fontweight="bold", va="top", ha="left",
            bbox=dict(facecolor="white", edgecolor="none", pad=1.5, alpha=0.8))
#---

#+++ Budget time-series panels (rows 2–3)
ke_terms = {
    r"$-\partial_t$ SFS KE":  ("∫-∂ₜ SFS KE dV",    budget_colors["tendency"]),
    r"$\Pi_{KE}$":             ("∫Π_KE dV",           budget_colors["flux"]),
    r"$-\varepsilon_s$":       ("∫-εₛ dV",            budget_colors["dissipation"]),
    r"SFS APE $\to$ KE":       ("∫(SFS APE->KE) dV",  budget_colors["exchange"]),
    r"residual":               ("residual_KE",         budget_colors["residual"]),
}
ape_terms = {
    r"$-\partial_t$ SFS APE":  ("∫-∂ₜ SFS APE dV",   budget_colors["tendency"]),
    r"$\Pi_{APE}$":            ("∫Π_APE dV",          budget_colors["flux"]),
    r"$-\chi_s$":              ("∫-χₛ dV",            budget_colors["dissipation"]),
    r"SFS KE $\to$ APE":       ("∫(SFS KE->APE) dV",  budget_colors["exchange"]),
    r"$R^s$":                  ("∫Rˢ dV",             "C4"),
    r"residual":               ("residual_APE",        budget_colors["residual"]),
}

for ax, budget_ds, terms, ylabel in [
    (ax_ke_budget,  ke_int,  ke_terms,  "SFS KE budget [W]"),
    (ax_ape_budget, ape_int, ape_terms, "SFS APE budget [W]"),
]:
    for label_str, (var, color) in terms.items():
        data = budget_ds[var].dropna("time")
        ls = "--" if "residual" in var else "-"
        lw = 1.0 if "residual" in var else 1.5
        ax.plot(data.time, data.values, label=label_str, color=color, ls=ls, lw=lw)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, alpha=0.3, lw=0.5)
    ax.legend(fontsize=10, loc="upper right", frameon=True, fancybox=True)

ax_ke_budget.tick_params(labelbottom=False)
ax_ape_budget.set_xlabel("Time", fontsize=12)

ymin = min(ax_ke_budget.get_ylim()[0], ax_ape_budget.get_ylim()[0])
ymax = max(ax_ke_budget.get_ylim()[1], ax_ape_budget.get_ylim()[1])
ax_ke_budget.set_ylim(ymin, ymax)
ax_ape_budget.set_ylim(ymin, ymax)

ke_vline = ax_ke_budget.axvline(times[0], color="k", ls=":", lw=1.5, alpha=0.7)
ape_vline = ax_ape_budget.axvline(times[0], color="k", ls=":", lw=1.5, alpha=0.7)

ax_ke_budget.text(0.01, 0.95, "(g)", transform=ax_ke_budget.transAxes, fontsize=12, fontweight="bold", va="top", ha="left",
                  bbox=dict(facecolor="white", edgecolor="none", pad=1.5, alpha=0.8))
ax_ape_budget.text(0.01, 0.95, "(h)", transform=ax_ape_budget.transAxes, fontsize=12, fontweight="bold", va="top", ha="left",
                   bbox=dict(facecolor="white", edgecolor="none", pad=1.5, alpha=0.8))
#---

#+++ Suptitle
label = run_label(ke_budget.attrs)
suptitle_base = f"ℓ = {ℓ_sel:.4f}"
if label:
    suptitle_base = f"{label},  {suptitle_base}"
suptitle = fig.suptitle(f"{suptitle_base},  t = {times[0]:.1f}", fontsize=14)
#---

#+++ Animation update function
def update(frame):
    if frame % 10 == 0:
        print(f"  Frame {frame+1}/{len(times)}  (t = {times[frame]:.1f})")
    for mesh, (row, col, ds, var, *_), (xdim, zdim) in zip(meshes, panel_specs, panel_dims):
        data = get_frame(ds, var, xdim, zdim, frame)
        mesh.set_array(data.ravel())
    ke_vline.set_xdata([times[frame], times[frame]])
    ape_vline.set_xdata([times[frame], times[frame]])
    suptitle.set_text(f"{suptitle_base},  t = {times[frame]:.1f}")
    return meshes + [ke_vline, ape_vline, suptitle]
#---

#+++ Save animation
outfile = str(ANIMATIONS / f"{stem}_panels_l{ℓ_sel:.4f}{ref_suffix}.mp4")
print(f"Recording {len(times)} frames at {args.fps} fps...")
writer = FFMpegWriter(fps=args.fps, metadata=dict(title=f"SFS budgets ℓ={ℓ_sel:.4f}"))
anim = FuncAnimation(fig, update, frames=len(times), blit=False, cache_frame_data=False)
anim.save(outfile, writer=writer, dpi=args.dpi)
del anim
plt.close(fig)
print(f"Animation saved to: {outfile}")
#---
