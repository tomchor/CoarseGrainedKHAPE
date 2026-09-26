#!/usr/bin/env python
#+++ Imports
import logging
import os
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from src.aux00_utils import PP_OUTPUT, load_dataset_and_grid, reference_suffix
#---

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
print = logging.info

#+++ Configuration
import argparse
parser = argparse.ArgumentParser(description="2x3 panel snapshot: vorticity/buoyancy in col 1, total/SFS energy in cols 2-3 (KE top row, APE bottom row)")
parser.add_argument("--filename", default="output/khi_Nz2048_Ri0.10.nc", help="Path to simulation NetCDF file")
parser.add_argument("--time", type=float, default=50, help="Target time for snapshot (nearest available will be used)")
parser.add_argument("--filter-scale", type=float, default=1.0, help="Target filter length scale (nearest available will be used)")
parser.add_argument("--clim-percentile", type=float, default=99.5, help="Percentile of |data| used to set symmetric color limits")
parser.add_argument("--reference", choices=["filtered", "true"], default="filtered",
                    help="Read the output built with this --reference (03-05 and sweep2 tag the 'true' ones _trueref)")
args = parser.parse_args()

print("\n" + "="*70 + f"\n  {Path(__file__).name}\n  " + "  ".join(f"{k}={v}" for k,v in vars(args).items()) + "\n" + "="*70)
REPO_ROOT = Path(__file__).resolve().parent.parent
FIGURES   = REPO_ROOT / "figures"
FIGURES.mkdir(exist_ok=True)
filename = str(REPO_ROOT / args.filename) if not os.path.isabs(args.filename) else args.filename
stem = Path(filename).stem
ref_suffix = reference_suffix(args.reference)   # adds _trueref for --reference true
#---

#+++ Load datasets
print("Loading simulation dataset and budgets...")
ds = load_dataset_and_grid(filename)
ke_budget  = xr.open_dataset(str(PP_OUTPUT / f"{stem}_sfs_ke_budget_fields{ref_suffix}.nc"),  decode_times=False)
ape_budget = xr.open_dataset(str(PP_OUTPUT / f"{stem}_sfs_ape_budget_fields{ref_suffix}.nc"), decode_times=False)

ds         = ds.sel(z_aac=slice(-4, +4))
ke_budget  = ke_budget.sel(z_aac=slice(-4, +4))
ape_budget = ape_budget.sel(z_aac=slice(-4, +4))
print("Done.")
#---

#+++ Select nearest time and filter scale
t_sel = float(ke_budget.time.sel(time=args.time, method="nearest").values)
ℓ_sel = float(ke_budget.filter_scale.sel(filter_scale=args.filter_scale, method="nearest").values)
print(f"Selected time = {t_sel:.3f}  (requested {args.time})")
print(f"Selected ℓ    = {ℓ_sel:.4f}  (requested {args.filter_scale})")
#---

#+++ Extract fields
print("Extracting fields...")
sel = dict(time=t_sel, filter_scale=ℓ_sel, method="nearest")
sel_t = dict(time=t_sel, method="nearest")

ω = ds["ω"].sel(**sel_t).squeeze()
b = ds["b"].sel(**sel_t).squeeze()

u_c = ds["u"].sel(**sel_t).squeeze()
w_c = ds["w"].sel(**sel_t).squeeze()
KE_total = 0.5 * (u_c**2 + w_c**2)

KE_sfs    = ke_budget["KE_of_sfs_flow"].sel(**sel).squeeze()
APE_total = ape_budget["Ea(ρ, z)"].sel(**sel).squeeze()
APE_sfs   = ape_budget["Eaˢ(ρ, z)"].sel(**sel).squeeze()
print("Done.")
#---

#+++ Buoyancy contours for overlay
bx_dim = next(d for d in b.dims if "x" in d)
bz_dim = next(d for d in b.dims if "z" in d)
bdata  = b.transpose(bx_dim, bz_dim).values.T
bx     = b[bx_dim].values
bz     = b[bz_dim].values
blevels = np.linspace(np.nanpercentile(bdata, 2), np.nanpercentile(bdata, 98), 12)
#---

#+++ Helper to get (x, z, data) from a 2D field with arbitrary staggered dim names
def _xzdata(field):
    x_dim = next(d for d in field.dims if "x" in d)
    z_dim = next(d for d in field.dims if "z" in d)
    return field[x_dim].values, field[z_dim].values, field.transpose(x_dim, z_dim).values.T
#---

#+++ Plot
print("Plotting...")
fig, axes = plt.subplots(2, 3, figsize=(15, 6.3), constrained_layout=True, gridspec_kw=dict(wspace=0, hspace=0))

# Panels: (row, col) -> (field, title, kind)
#   kind in {"diverging", "buoyancy", "positive"}
panels = {
    (0, 0): (ω,         r"$\omega$ (vorticity)",                                 "diverging"),
    (0, 1): (KE_total,  r"Total KE: $\frac{1}{2}(u^2+w^2)$",                     "positive"),
    (0, 2): (KE_sfs,    r"SFS KE: $\frac{1}{2}|\mathbf{u}-\bar{\mathbf{u}}|^2$", "positive"),
    (1, 0): (b,         r"$b$ (buoyancy)",                                       "buoyancy"),
    (1, 1): (APE_total, r"Total APE: $E_a(\rho, z)$",                            "positive"),
    (1, 2): (APE_sfs,   r"SFS APE: $E_a^s(\rho, z)$",                            "positive"),
}

# Symmetric clim for vorticity, percentile-based clim for positive fields
ω_vmax = np.nanpercentile(np.abs(_xzdata(ω)[2]),   args.clim_percentile)
b_vmin = np.nanpercentile(_xzdata(b)[2], 100 - args.clim_percentile)
b_vmax = np.nanpercentile(_xzdata(b)[2], args.clim_percentile)

for (row, col), (field, title, kind) in panels.items():
    ax = axes[row, col]
    x, z, data = _xzdata(field)

    if kind == "diverging":
        cmap = "RdBu_r"
        vmin, vmax = -ω_vmax, ω_vmax
        contour_color = "k"
        title_facecolor = "white"
        title_color = "black"
        tick_color = "black"
    elif kind == "buoyancy":
        cmap = "RdBu_r"
        vmin, vmax = b_vmin, b_vmax
        contour_color = "k"
        title_facecolor = "white"
        title_color = "black"
        tick_color = "black"
    else:  # positive (energies)
        cmap = "inferno"
        vmin = 0
        vmax = np.nanpercentile(data, args.clim_percentile)
        contour_color = "white"
        title_facecolor = "black"
        title_color = "white"
        tick_color = "white"

    im = ax.pcolormesh(x, z, data, cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True)
    cax = ax.inset_axes([0.2, 0.09, 0.6, 0.03])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", extend="both")
    cb.locator = MaxNLocator(nbins=4)
    cb.update_ticks()
    cax.tick_params(colors=tick_color)
    for spine in cax.spines.values():
        spine.set_edgecolor(tick_color)

    ax.contour(bx, bz, bdata, levels=blevels, colors=contour_color, linewidths=0.6, alpha=0.5)

    ax.text(0.5, 0.97, title, transform=ax.transAxes, fontsize=11, ha="center", va="top", color=title_color, bbox=dict(facecolor=title_facecolor, edgecolor="none", pad=2, alpha=0.6))
    ax.set_title("")
    ax.set_ylim(-4, +4)
    ax.set_aspect("equal")
    ax.set_yticks([-3, -1, 1, 3])

for row in range(2):
    axes[row, 0].set_ylabel("z")
    for col in range(1, 3):
        axes[row, col].set_ylabel("")
        axes[row, col].tick_params(labelleft=False, left=False)

for col in range(3):
    axes[0, col].set_xlabel("")
    axes[0, col].tick_params(labelbottom=False, bottom=False)
    axes[1, col].set_xlabel("x")

for ax, letter in zip(axes.flat, "abcdef"):
    ax.text(0.02, 0.97, f"({letter})", transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top", ha="left",
            bbox=dict(facecolor="white", edgecolor="none", pad=1.5))

outfile = str(FIGURES / f"{stem}_panels2x3_t{t_sel:.1f}_l{ℓ_sel:.4f}{ref_suffix}.png")
fig.savefig(outfile, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Figure saved to: {outfile}")
#---
