#!/usr/bin/env python
#+++ Imports
import logging
import os
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from aux00_utils import load_dataset_and_grid
from aux03_plotting import run_label
#---

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
print = logging.info

#+++ Configuration
import argparse
parser = argparse.ArgumentParser(description="Plot 4-panel snapshot of local SFS budget terms")
parser.add_argument("--filename", default="output/khi_Nz2048_Ri0.10.nc", help="Path to simulation NetCDF file")
parser.add_argument("--time", type=float, default=50, help="Target time for snapshot (nearest available will be used)")
parser.add_argument("--filter-scale", type=float, default=0.5, help="Target filter length scale (nearest available will be used)")
parser.add_argument("--clim-percentile", type=float, default=99.5, help="Percentile of |data| used to set symmetric color limits")
args = parser.parse_args()

print("\n" + "="*70 + f"\n  {Path(__file__).name}\n  " + "  ".join(f"{k}={v}" for k,v in vars(args).items()) + "\n" + "="*70)
REPO_ROOT = Path(__file__).resolve().parent.parent
PP_OUTPUT = REPO_ROOT / "postprocessing" / "output"
FIGURES   = REPO_ROOT / "figures"
FIGURES.mkdir(exist_ok=True)
filename = str(REPO_ROOT / args.filename) if not os.path.isabs(args.filename) else args.filename
stem = Path(filename).stem
#---

#+++ Load budgets
print("Loading KE and APE budgets...")
ke_budget  = xr.open_dataset(str(PP_OUTPUT / f"{stem}_sfs_ke_budget_fields.nc"),  decode_times=False)
ape_budget = xr.open_dataset(str(PP_OUTPUT / f"{stem}_sfs_ape_budget_fields.nc"), decode_times=False)

ke_budget = ke_budget.sel(z_aac=slice(-4, +4))
ape_budget = ape_budget.sel(z_aac=slice(-4, +4))
print("Done.")
#---

#+++ Select nearest time and filter scale
t_sel = float(ke_budget.time.sel(time=args.time, method="nearest").values)
ℓ_sel = float(ke_budget.filter_scale.sel(filter_scale=args.filter_scale, method="nearest").values)
print(f"Selected time = {t_sel:.3f}  (requested {args.time})")
print(f"Selected ℓ   = {ℓ_sel:.4f}  (requested {args.filter_scale})")
#---

#+++ Load fields at selected time and filter scale
print("Selecting fields...")
sel = dict(time=t_sel, filter_scale=ℓ_sel, method="nearest")

pi_ke    = ke_budget["Π_KE"].sel(**sel).squeeze()               # cross-scale KE flux
exchange = ke_budget["SFS APE->KE exchange"].sel(**sel).squeeze() # APE->KE exchange
pi_ape   = ape_budget["Π_APE"].sel(**sel).squeeze()             # cross-scale APE flux
chi_s    = ape_budget["χₛ"].sel(**sel).squeeze()                # APE dissipation
#---

#+++ Load buoyancy field for contours
print("Loading buoyancy field...")
ds = load_dataset_and_grid(filename)
b = ds["b"].sel(time=t_sel, method="nearest").squeeze()
print("Done.")
#---

#+++ Plot
print("Plotting...")
panels = [
    (pi_ke,    r"$\Pi_{KE}$  (cross-scale KE flux)"),
    (pi_ape,   r"$\Pi_{APE}$  (cross-scale APE flux)"),
    (exchange, r"Small-scale APE$\to$KE exchange"),
    (chi_s,    r"$\chi_s$  (Small-scale APE dissipation)"),
]

fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True, gridspec_kw=dict(wspace=0, hspace=0))

x_dim = next(d for d in pi_ke.dims if "x" in d)
z_dim = next(d for d in pi_ke.dims if "z" in d)
x = pi_ke[x_dim].values
z = pi_ke[z_dim].values

bx_dim = next(d for d in b.dims if "x" in d)
bz_dim = next(d for d in b.dims if "z" in d)
bdata   = b.transpose(bx_dim, bz_dim).values.T   # → (nz, nx)
bx      = b[bx_dim].values
bz      = b[bz_dim].values
blevels = np.linspace(np.nanpercentile(bdata, 2), np.nanpercentile(bdata, 98), 12)

# Shared clim for cross-scale KE flux and APE->KE conversion
pi_ke_data   = pi_ke.transpose(x_dim, z_dim).values.T
exchange_data = exchange.transpose(x_dim, z_dim).values.T
pi_ke_vmax = max(np.nanpercentile(np.abs(pi_ke_data),    args.clim_percentile),
                 np.nanpercentile(np.abs(exchange_data), args.clim_percentile))

# Independent clim for cross-scale APE flux
pi_ape_data = pi_ape.transpose(x_dim, z_dim).values.T
pi_ape_vmax = np.nanpercentile(np.abs(pi_ape_data), args.clim_percentile)

for ax, (field, title) in zip(axes.flat, panels):
    data = field.transpose(x_dim, z_dim).values.T  # → (nz, nx)

    is_dissipation = field is chi_s
    if is_dissipation:
        cmap = "inferno"
        vmin = 0
        vmax = np.nanpercentile(data, args.clim_percentile)
    else:
        cmap = "RdBu_r"
        vmin, vmax = -pi_ke_vmax, pi_ke_vmax

    im = ax.pcolormesh(x, z, data, cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True)
    cax = ax.inset_axes([0.2, 0.09, 0.6, 0.03])
    tick_color = "white" if is_dissipation else "black"
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cax.tick_params(colors=tick_color)
    for spine in cax.spines.values():
        spine.set_edgecolor(tick_color)

    contour_color = "white" if is_dissipation else "k"
    ax.contour(bx, bz, bdata, levels=blevels, colors=contour_color, linewidths=0.6, alpha=0.5)

    title_color = "white" if is_dissipation else "black"
    ax.text(0.5, 0.97, title, transform=ax.transAxes, fontsize=11, ha="center", va="top", color=title_color, bbox=dict(facecolor="black" if is_dissipation else "white", edgecolor="none", pad=2, alpha=0.6))
    ax.set_title("")
    ax.set_ylim(-4, +4)
    ax.set_aspect("equal")

for ax in axes.flat:
    ax.set_yticks([-3, -1, 1, 3])

for row in range(2):
    axes[row, 0].set_ylabel("z")
    axes[row, 1].set_ylabel("")
    axes[row, 1].tick_params(labelleft=False, left=False)

for col in range(2):
    axes[0, col].set_xlabel("")
    axes[0, col].tick_params(labelbottom=False, bottom=False)
    axes[1, col].set_xlabel("x")

for ax, letter in zip(axes.flat, "abcd"):
    ax.text(0.02, 0.97, f"({letter})", transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top", ha="left",
            bbox=dict(facecolor="white", edgecolor="none", pad=1.5))

outfile = str(FIGURES / f"{stem}_panels_t{t_sel:.1f}_l{ℓ_sel:.4f}.png")
fig.savefig(outfile, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Figure saved to: {outfile}")
#---
