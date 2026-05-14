#!/usr/bin/env python
#+++ Imports
import logging
import os
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from src.aux00_utils import load_dataset_and_grid
#---

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
print = logging.info

#+++ Configuration
import argparse
parser = argparse.ArgumentParser(description="Plot 4-panel thumbnail of local SFS budget terms (no labels)")
parser.add_argument("--filename", default="output/khi_Nz2048_Ri0.10.nc", help="Path to simulation NetCDF file")
parser.add_argument("--time", type=float, default=85, help="Target time for snapshot (nearest available will be used)")
parser.add_argument("--filter-scale", type=float, default=None, help="Target filter length scale (nearest available will be used; defaults to the smallest available)")
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
ℓ_target = float(ke_budget.filter_scale.min().values) if args.filter_scale is None else args.filter_scale
ℓ_sel = float(ke_budget.filter_scale.sel(filter_scale=ℓ_target, method="nearest").values)
print(f"Selected time = {t_sel:.3f}  (requested {args.time})")
print(f"Selected ℓ   = {ℓ_sel:.4f}  (requested {args.filter_scale})")
#---

#+++ Load fields at selected time and filter scale
print("Selecting fields...")
sel = dict(time=t_sel, filter_scale=ℓ_sel, method="nearest")

ε_Kˢ     = ke_budget["ε_Kˢ"].sel(**sel).squeeze()
Π_K      = ke_budget["Π_K"].sel(**sel).squeeze()
Π_A      = ape_budget["Π_A"].sel(**sel).squeeze()
ε_Aˢ     = ape_budget["ε_Aˢ"].sel(**sel).squeeze()
#---

#+++ Load buoyancy field for contours
print("Loading buoyancy field...")
ds = load_dataset_and_grid(filename)
b = ds["b"].sel(time=t_sel, method="nearest").squeeze()
print("Done.")
#---

#+++ Plot
print("Plotting...")
panels = [ε_Kˢ, Π_K, Π_A, ε_Aˢ]

x_dim = next(d for d in Π_K.dims if "x" in d)
z_dim = next(d for d in Π_K.dims if "z" in d)
x = Π_K[x_dim].values
z = Π_K[z_dim].values

cm = 1 / 2.54
fig_height_cm = 5
fig_width_cm = 1.2 * fig_height_cm
fig, axes = plt.subplots(2, 2, figsize=(fig_width_cm * cm, fig_height_cm * cm))
fig.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)

bx_dim = next(d for d in b.dims if "x" in d)
bz_dim = next(d for d in b.dims if "z" in d)
bdata   = b.transpose(bx_dim, bz_dim).values.T
bx      = b[bx_dim].values
bz      = b[bz_dim].values
blevels = np.linspace(np.nanpercentile(bdata, 2), np.nanpercentile(bdata, 98), 12)

Π_K_data = Π_K.transpose(x_dim, z_dim).values.T
Π_A_data = Π_A.transpose(x_dim, z_dim).values.T
Π_K_vmax = 0.5 * np.nanpercentile(np.abs(Π_K_data), args.clim_percentile)
Π_A_vmax = np.nanpercentile(np.abs(Π_A_data), args.clim_percentile)

z_half = min(float(z.max()), -float(z.min()))  # symmetric available z extent
x_mid = 0.5 * (float(x.min()) + float(x.max()))
quadrants = [
    (float(x.min()), x_mid,           0.0,    +z_half),  # top-left
    (x_mid,          float(x.max()),  0.0,    +z_half),  # top-right
    (float(x.min()), x_mid,          -z_half, 0.0),      # bottom-left
    (x_mid,          float(x.max()), -z_half, 0.0),      # bottom-right
]

for ax, field, (xlo, xhi, zlo, zhi) in zip(axes.flat, panels, quadrants):
    data = field.transpose(x_dim, z_dim).values.T

    is_dissipation = field is ε_Kˢ or field is ε_Aˢ
    if is_dissipation:
        cmap = "inferno"
        vmin = 0
        vmax = np.nanpercentile(data, args.clim_percentile)
    else:
        cmap = "RdBu_r"
        vfield_max = Π_K_vmax if field is Π_K else Π_A_vmax
        vmin, vmax = -vfield_max, vfield_max

    ax.pcolormesh(x, z, data, cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True)

    ax.set_xlim(xlo, xhi)
    ax.set_ylim(zlo, zhi)
    ax.set_aspect("auto")  # let each panel fill its 1.2:1 cell; data gets stretched
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_visible(False)

outfile = str(FIGURES / f"{stem}_thumbnail_t{t_sel:.1f}_l{ℓ_sel:.4f}.jpg")
fig.savefig(outfile, dpi=3000, pad_inches=0)
print(f"Figure saved to: {outfile}")
#---
