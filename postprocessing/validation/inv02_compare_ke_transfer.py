#!/usr/bin/env python
#+++ Imports
import logging
import os
import sys
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # postprocessing/ on path for `src.*`
from src.aux00_utils import (load_dataset_and_grid, make_gaussian_filter,
                             condense_uw_velocities, integrate)
from src.aux02_ke_functions import (calculate_sfs_stress_tensor,
                                     calculate_strain_tensor,
                                     calculate_cross_scale_ke_flux)
from src.aux03_plotting import run_label
#---

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
print = logging.info

#+++ Configuration
import argparse
parser = argparse.ArgumentParser(description="Compare online (simulation-time) vs offline (post-processed) cross-scale KE transfer Π_K")
parser.add_argument("--filename", default="output/khi_Nz256_Ri0.10.nc", help="Path to simulation NetCDF file")
parser.add_argument("--filter-scales", type=float, nargs="+", default=[1, 7], help="Filter ℓ (FWHM) values matching the online cross_scale_ke_flux widths")
parser.add_argument("--time", type=float, default=None, help="Target time for the snapshot maps (default: midpoint of simulation)")
parser.add_argument("--z-window", type=float, default=6.0, help="Half-height of the z window (in units of h) shown in the snapshot maps")
args = parser.parse_args()

print("\n" + "="*70 + f"\n  {Path(__file__).name}\n  " + "  ".join(f"{k}={v}" for k, v in vars(args).items()) + "\n" + "="*70)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # validation/ → postprocessing/ → repo root
FIGURES = REPO_ROOT / "figures"
FIGURES.mkdir(exist_ok=True)
filename = str(REPO_ROOT / args.filename) if not os.path.isabs(args.filename) else args.filename
stem = Path(filename).stem


def online_name(ℓ):
    """Online output variable name for filter scale ℓ (matches the Julia Symbol("Π_K_ℓ$(ℓ)"))."""
    return f"Π_K_ℓ{int(ℓ)}" if ℓ == int(ℓ) else f"Π_K_ℓ{ℓ}"
#---

#+++ Load dataset
# load_dataset_and_grid pads the z domain to 2× (edge values); this pads *both* the online Π_K
# variables read from the file and the fields used to recompute Π_K offline, so the two live on the
# same grid. We recover the original (unpadded) z extent from the grid group to drop the padding
# before any comparison (the padded region is ≈0 for Π_K and would otherwise dilute the metrics).
print("Loading simulation data...")
ds = load_dataset_and_grid(filename)
ds = ds.chunk({"time": 1})

grid = xr.open_dataset(filename, group="underlying_grid_reconstruction_kwargs")
z0, z1 = float(grid.z.min()), float(grid.z.max())   # original (unpadded) z faces
in_domain = dict(z_aac=slice(z0, z1))               # selects the original cell centers

if args.time is None:
    args.time = float(ds.time.values[len(ds.time) // 2])
t_sel = float(ds.time.sel(time=args.time, method="nearest").values)
print(f"Selected snapshot time = {t_sel:.3f}  (requested {args.time})")
#---

#+++ Recompute Π_K offline (mirrors the KE part of 03_energy_transfer.py)
filtered_dimensions = ["x_caa", "z_aac"]
tensor_dimensions   = ("x_caa", "z_aac")

uᵢ = condense_uw_velocities(ds, indices=(1, 3))["uᵢ"]   # also leaves ds.uᵢ available, keeps ds.dV etc.

print("Recomputing Π_K offline at each filter scale...")
offline = {}
for ℓ in args.filter_scales:
    gf = make_gaussian_filter(ℓ, ds)
    ūᵢ = gf.apply(uᵢ, dims=filtered_dimensions)
    # τⁱʲ = filter(uⁱuʲ) - ūⁱūʲ  ;  S̄ⁱʲ = ½(∂ūⁱ/∂xʲ + ∂ūʲ/∂xⁱ)  ;  Π_K = -τⁱʲ S̄ⁱʲ
    τ  = calculate_sfs_stress_tensor(uᵢ, gf, filter_dims=filtered_dimensions, filtered_u_i=ūᵢ)
    S̄  = calculate_strain_tensor(ūᵢ, dimensions=tensor_dimensions)
    offline[ℓ] = calculate_cross_scale_ke_flux(τ, S̄).rename(online_name(ℓ))
    print(f"  ℓ = {ℓ:>4}: offline Π_K computed")
#---

#+++ Load the online Π_K fields / integrals
online = {}
online_int = {}
for ℓ in args.filter_scales:
    name = online_name(ℓ)
    if name not in ds:
        print(f"  WARNING: online field '{name}' not in dataset, skipping ℓ={ℓ}")
        continue
    online[ℓ] = ds[name]
    int_name = f"{name}_int"
    online_int[ℓ] = ds[int_name].squeeze(drop=True) if int_name in ds else None

filter_scales = [ℓ for ℓ in args.filter_scales if ℓ in online]
if not filter_scales:
    raise SystemExit("No online Π_K fields found — run the simulation with the online KE-transfer diagnostic first.")
#---

#+++ Snapshot maps: online | offline | difference  (one row per filter scale)
label = run_label(ds.attrs)
zw = args.z_window
n_scales = len(filter_scales)
fig, axes = plt.subplots(n_scales, 3, figsize=(15, 3.6 * n_scales), constrained_layout=True, squeeze=False)

print("\nSnapshot comparison (bulk rms over the original domain):")
for i, ℓ in enumerate(filter_scales):
    on  = online[ℓ].sel(time=t_sel, method="nearest").sel(**in_domain).squeeze()
    off = offline[ℓ].sel(time=t_sel, method="nearest").sel(**in_domain).squeeze()
    diff = (on - off).compute()
    on = on.compute(); off = off.compute()

    vmax = max(float(np.nanpercentile(np.abs(on.values), 99)), float(np.nanpercentile(np.abs(off.values), 99)))
    vmax = vmax if vmax > 0 else 1.0
    kw = dict(x="x_caa", y="z_aac", add_colorbar=True, cmap="RdBu_r", vmin=-vmax, vmax=vmax)

    on.plot(ax=axes[i, 0], **kw);  axes[i, 0].set_title(f"Online Π_K (ℓ={ℓ:g})")
    off.plot(ax=axes[i, 1], **kw); axes[i, 1].set_title(f"Offline Π_K (ℓ={ℓ:g})")
    diff.plot(ax=axes[i, 2], x="x_caa", y="z_aac", add_colorbar=True, cmap="RdBu_r", robust=True)
    axes[i, 2].set_title("Difference (online − offline)")

    for k in range(3):
        axes[i, k].set_ylim(-zw, zw)
        axes[i, k].set_aspect("equal")
    axes[i, 0].set_ylabel(f"ℓ = {ℓ:g}", fontsize=13)

    rms_diff   = float(np.sqrt(np.nanmean(diff.values**2)))
    rms_online = float(np.sqrt(np.nanmean(on.values**2)))
    rel = rms_diff / rms_online if rms_online > 0 else float("inf")
    print(f"  ℓ={ℓ:>4g}: rms(diff)/rms(online) = {rel:.2e},  max|diff| = {float(np.nanmax(np.abs(diff.values))):.2e}"
          f",  rms(online) = {rms_online:.2e}")

suptitle = f"Online vs offline cross-scale KE transfer Π_K   t = {t_sel:.1f}"
if label:
    suptitle += f"   {label}"
fig.suptitle(suptitle, fontsize=13, y=1.01)
outfile = str(FIGURES / f"{stem}_ke_transfer_comparison_maps_t{t_sel:.1f}.png")
fig.savefig(outfile, dpi=150, bbox_inches="tight")
print(f"Snapshot figure saved to: {outfile}")
#---

#+++ Volume-integrated transfer ∫Π_K dV vs time
# offline: integrate the recomputed field over the original domain
# online : both the field integrated the same way (sanity check) and the Integral output written online
dV_dom = ds.dV.sel(**in_domain)
fig2, ax2 = plt.subplots(1, n_scales, figsize=(7 * n_scales, 4.2), constrained_layout=True, squeeze=False)

print("\nVolume-integrated transfer ∫Π_K dV (time-mean relative difference):")
for i, ℓ in enumerate(filter_scales):
    off_int    = integrate(offline[ℓ].sel(**in_domain), dV_dom).compute()
    on_int_fld = integrate(online[ℓ].sel(**in_domain),  dV_dom).compute()

    a = ax2[0, i]
    off_int.plot(ax=a, x="time", label="offline (recomputed)", color="k", lw=2)
    on_int_fld.plot(ax=a, x="time", label="online (∫ of field)", color="tab:red", ls="--", lw=2)
    if online_int.get(ℓ) is not None:
        online_int[ℓ].compute().plot(ax=a, x="time", label="online (Integral output)", color="tab:orange", ls=":", lw=2)

    a.set_title(f"ℓ = {ℓ:g}")
    a.set_ylabel("∫ Π_K dV")
    a.axhline(0, color="0.6", lw=0.8)
    a.legend(fontsize=9)

    denom = float(np.sqrt(np.nanmean(off_int.values**2)))
    rel = float(np.sqrt(np.nanmean((on_int_fld.values - off_int.values)**2))) / denom if denom > 0 else float("inf")
    print(f"  ℓ={ℓ:>4g}: rms(online−offline)/rms(offline) = {rel:.2e}")

suptitle = "Online vs offline volume-integrated KE transfer ∫Π_K dV"
if label:
    suptitle += f"   {label}"
fig2.suptitle(suptitle, fontsize=13, y=1.03)
outfile2 = str(FIGURES / f"{stem}_ke_transfer_comparison_integral.png")
fig2.savefig(outfile2, dpi=150, bbox_inches="tight")
print(f"Integral figure saved to: {outfile2}")
#---
