#!/usr/bin/env python
"""Mid-depth snapshot of buoyancy, Rossby number, and cross-scale KE/APE flux at one filter scale."""

#+++ Imports
import os
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
#---

#+++ Configuration
import argparse
parser = argparse.ArgumentParser(description="Plot a 2x2 mid-depth snapshot: buoyancy, Rossby number ζ/f, cross-scale KE flux Πₖ, cross-scale APE flux Π_A")
parser.add_argument("--filename", default="output/bci_Nx48_Ny48_Nz8.nc", help="Path to simulation NetCDF file")
parser.add_argument("--filter-scale", type=float, default=None, help="Target filter length scale in meters (nearest available; defaults to the smallest available)")
parser.add_argument("--time", type=float, default=None, help="Target time in days (nearest available; defaults to the last available)")
parser.add_argument("--z", type=float, default=-500.0, help="Target depth in meters (nearest available cell center; default -500, mid-depth)")
parser.add_argument("--clim-percentile", type=float, default=99.5, help="Percentile of |data| used to set symmetric color limits")
args = parser.parse_args()
print("\n" + "="*70 + f"\n  {Path(__file__).name}\n  " + "  ".join(f"{k}={v}" for k,v in vars(args).items()) + "\n" + "="*70)

REPO_ROOT = Path(__file__).resolve().parent.parent
PP_OUTPUT = REPO_ROOT / "postprocessing" / "output"
FIGURES = REPO_ROOT / "figures"
FIGURES.mkdir(exist_ok=True)
filename = str(REPO_ROOT / args.filename) if not os.path.isabs(args.filename) else args.filename
stem = Path(filename).stem
#---

#+++ Orientation fix: some pipeline fields (Π_A, the KE<->APE exchange term) are stored with dims
# (..., x, y) instead of (..., y, x) like most fields -- always transpose to (..., y_dim, x_dim) first.
# See anim3_panels.py/plot5_vorticity_strain_flux.py for the same pattern.
def fix_orientation(da):
    y_dim = next(d for d in da.dims if d.startswith("y"))
    x_dim = next(d for d in da.dims if d.startswith("x"))
    other_dims = [d for d in da.dims if d not in (y_dim, x_dim)]
    return da.transpose(*other_dims, y_dim, x_dim)

def coriolis_f(attrs):
    """Coriolis parameter from this run's own attrs (for the ζ/f Rossby-number normalization)."""
    Omega_earth = 7.2921159e-5
    return 2 * Omega_earth * np.sin(np.radians(attrs["latitude"]))
#---

#+++ Open datasets and pick filter scale/time/z
print("Opening datasets...")
ds_raw = xr.open_dataset(filename, decode_times=False)
ke_fields  = xr.open_dataset(PP_OUTPUT / f"{stem}_sfs_ke_budget_fields.nc",  decode_times=False)
ape_fields = xr.open_dataset(PP_OUTPUT / f"{stem}_sfs_ape_budget_fields.nc", decode_times=False)

ℓ_target = args.filter_scale if args.filter_scale is not None else float(ke_fields.filter_scale.min())
ℓ = float(ke_fields.filter_scale.sel(filter_scale=ℓ_target, method="nearest"))
ℓ_km = int(round(ℓ / 1000))

t_target = args.time * 86400 if args.time is not None else float(ke_fields.time.max())
t_sel = float(ke_fields.time.sel(time=t_target, method="nearest"))
t_days = t_sel / 86400

x_km = ds_raw.x_caa.values / 1e3
y_km = ds_raw.y_aca.values / 1e3
f_cor = coriolis_f(ds_raw.attrs)
print(f"ℓ = {ℓ:.4f} m ({ℓ_km} km), t = {t_days:.2f} days")
#---

#+++ Select fields at the target time/z/filter scale
t_sel_raw = float(ds_raw.time.sel(time=t_sel, method="nearest"))
b = fix_orientation(ds_raw["b"]).sel(time=t_sel_raw, method="nearest").sel(z_aac=args.z, method="nearest")
zeta_norm = fix_orientation(ds_raw["ζ"]).sel(time=t_sel_raw, method="nearest").sel(z_aac=args.z, method="nearest") / f_cor
z_sel = float(b.z_aac)

Pi_K = fix_orientation(ke_fields["Π_K"].sel(filter_scale=ℓ, time=t_sel, method="nearest")).sel(z_aac=args.z, method="nearest")
Pi_A = fix_orientation(ape_fields["Π_A"].sel(filter_scale=ℓ, time=t_sel, method="nearest")).sel(z_aac=args.z, method="nearest")
print(f"z = {z_sel:.1f} m")
#---

#+++ Plot
print("Building figure...")
fig, axes = plt.subplots(2, 2, figsize=(11, 10), constrained_layout=True)

def plot_field(ax, field, title, cmap="RdBu_r"):
    vmax = np.nanpercentile(np.abs(field.values), args.clim_percentile)
    im = ax.pcolormesh(x_km, y_km, field.values, cmap=cmap, vmin=-vmax, vmax=vmax, shading="auto")
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("x [km]")
    ax.set_ylabel("y [km]")
    fig.colorbar(im, ax=ax, shrink=0.85)

plot_field(axes[0,0], b,         f"buoyancy b\nt={t_days:.1f}d, z={z_sel:.0f}m")
plot_field(axes[0,1], zeta_norm, f"Rossby number ζ/f\nt={t_days:.1f}d, z={z_sel:.0f}m")
plot_field(axes[1,0], Pi_K,      f"cross-scale KE flux Πₖ (ℓ={ℓ_km}km)\nt={t_days:.1f}d, z={z_sel:.0f}m")
plot_field(axes[1,1], Pi_A,      f"cross-scale APE flux Π_A (ℓ={ℓ_km}km)\nt={t_days:.1f}d, z={z_sel:.0f}m")

fig.suptitle(f"{stem}: z={z_sel:.0f}m snapshots", fontsize=14)

z_m = int(round(z_sel))
outfile = FIGURES / f"{stem}_middepth_snapshots_l{ℓ_km}km_z{z_m}m_t{t_days:.0f}d.pdf"
fig.savefig(outfile, dpi=150, bbox_inches="tight")
print(f"Saved: {outfile}")
#---
