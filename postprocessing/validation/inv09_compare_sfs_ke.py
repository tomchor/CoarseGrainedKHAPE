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
from aux_check import add_tolerance_arg, set_tolerance, check, finalize
from src.aux00_utils import load_dataset_and_grid, make_gaussian_filter, condense_uw_velocities, integrate, open_grid_group
from src.aux02_ke_functions import calculate_sfs_stress_tensor
from src.aux03_plotting import run_label
#---

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
print = logging.info

#+++ Configuration
import argparse
parser = argparse.ArgumentParser(description="Compare online (simulation-time) vs offline (post-processed) sub-filter KE Kˢ")
parser.add_argument("--filename", default="output/khi_Nz256_Ri0.10.nc", help="Path to simulation NetCDF file")
parser.add_argument("--filter-scales", type=float, nargs="+", default=[1, 7], help="Filter ℓ (FWHM) values matching the online filter_ℓs")
parser.add_argument("--time", type=float, default=None, help="Target time for the snapshot maps (default: midpoint of simulation)")
parser.add_argument("--z-window", type=float, default=6.0, help="Half-height of the z window (in units of h) shown in the snapshot maps")
add_tolerance_arg(parser)
args = parser.parse_args()
set_tolerance(args.tolerance)

print("\n" + "="*70 + f"\n  {Path(__file__).name}\n  " + "  ".join(f"{k}={v}" for k, v in vars(args).items()) + "\n" + "="*70)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # validation/ → postprocessing/ → repo root
# Validation figures go in their own subdirectory so they are separable from the budget and
# paper figures in `figures/` — CI uploads this directory as its own artifact.
FIGURES = REPO_ROOT / "figures" / "validation"
FIGURES.mkdir(parents=True, exist_ok=True)
filename = str(REPO_ROOT / args.filename) if not os.path.isabs(args.filename) else args.filename
stem = Path(filename).stem


def online_name(ℓ):
    """Online output variable name for filter scale ℓ (matches the Julia Symbol("K_s_ℓ$(ℓ)"))."""
    return f"K_s_ℓ{int(ℓ)}" if ℓ == int(ℓ) else f"K_s_ℓ{ℓ}"
#---

#+++ Load dataset
# load_dataset_and_grid pads the z domain to 2× (edge values); this pads *both* the online Kˢ
# variables read from the file and the fields used to recompute Kˢ offline, so the two live on the
# same grid. We recover the original (unpadded) z extent from the grid group to drop the padding
# before any comparison (the padded region is ≈0 for Kˢ and would otherwise dilute the metrics).
print("Loading simulation data...")
ds = load_dataset_and_grid(filename)
ds = ds.chunk({"time": 1})

grid = open_grid_group(filename)   # handles the _gridN naming a --save_sorted run introduces
z0, z1 = float(grid.z.min()), float(grid.z.max())   # original (unpadded) z faces
in_domain = dict(z_aac=slice(z0, z1))               # selects the original cell centers

if args.time is None:
    args.time = float(ds.time.values[len(ds.time) // 2])
t_sel = float(ds.time.sel(time=args.time, method="nearest").values)
print(f"Selected snapshot time = {t_sel:.3f}  (requested {args.time})")
#---

#+++ Recompute Kˢ offline (the term 04_sfs_ke_budget.py computes as ½τⁱⁱ)
# Kˢ = filter(K) - Kˡ = ½ τⁱⁱ, the trace of the sub-filter stress τⁱʲ = filter(uⁱuʲ) - ūⁱūʲ. The online
# `SubFilterKineticEnergy` assembles it as filter(½uᵢuᵢ) - ½ūᵢūᵢ over all three components, while the
# offline tensor is built for i, j ∈ {1, 3}; these runs have Ny = 1 and v ≡ 0, so τ₂₂ = 0 and the two
# sums agree.
filtered_dimensions = ["x_caa", "z_aac"]

uᵢ = condense_uw_velocities(ds, indices=(1, 3))["uᵢ"]   # keeps ds.dV / online fields available

print("Recomputing Kˢ offline at each filter scale...")
offline = {}
for ℓ in args.filter_scales:
    gf = make_gaussian_filter(ℓ, ds)
    τ = calculate_sfs_stress_tensor(uᵢ, gf, filter_dims=filtered_dimensions)
    trace = sum(τ.sel(i=k, j=k) for k in τ.coords["i"].values)
    offline[ℓ] = (trace / 2).rename(online_name(ℓ))
    print(f"  ℓ = {ℓ:>4}: offline Kˢ computed")
#---

#+++ Load the online Kˢ fields / integrals
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
    raise SystemExit("No online Kˢ fields found — rerun the simulation with the online SubFilterKineticEnergy diagnostic.")
#---

#+++ Snapshot maps: online | offline | difference  (one row per filter scale; Kˢ ≥ 0)
label = run_label(ds.attrs)
zw = args.z_window
n_scales = len(filter_scales)
# One figure per script: the map rows plus a final row subdivided into the integral panels.
fig = plt.figure(figsize=(15, 3.6 * n_scales + 4.6), constrained_layout=True)
_gs = fig.add_gridspec(n_scales + 1, 3, height_ratios=[3.6] * n_scales + [4.6])
axes = np.empty((n_scales, 3), dtype=object)
for _i in range(n_scales):
    for _k in range(3):
        axes[_i, _k] = fig.add_subplot(_gs[_i, _k])
_gs_int = _gs[n_scales, :].subgridspec(1, n_scales)
ax2 = np.empty((1, n_scales), dtype=object)
for _i in range(n_scales):
    ax2[0, _i] = fig.add_subplot(_gs_int[0, _i])

print("\nSnapshot comparison (bulk rms over the original domain):")
for i, ℓ in enumerate(filter_scales):
    on  = online[ℓ].sel(time=t_sel, method="nearest").sel(**in_domain).squeeze().compute()
    off = offline[ℓ].sel(time=t_sel, method="nearest").sel(**in_domain).squeeze().compute()
    diff = (on - off).compute()

    vmax = max(float(np.nanpercentile(np.abs(on.values), 99)), float(np.nanpercentile(np.abs(off.values), 99)))
    vmax = vmax if vmax > 0 else 1.0
    kw = dict(x="x_caa", y="z_aac", add_colorbar=True, cmap="magma", vmin=0, vmax=vmax)   # Kˢ ≥ 0

    on.plot(ax=axes[i, 0], **kw);  axes[i, 0].set_title(f"Online Kˢ (ℓ={ℓ:g})")
    off.plot(ax=axes[i, 1], **kw); axes[i, 1].set_title(f"Offline Kˢ (ℓ={ℓ:g})")
    diff.plot(ax=axes[i, 2], x="x_caa", y="z_aac", add_colorbar=True, cmap="RdBu_r", robust=True)
    axes[i, 2].set_title("Difference (online − offline)")

    for k in range(3):
        axes[i, k].set_ylim(-zw, zw)
        axes[i, k].set_aspect("equal")
    axes[i, 0].set_ylabel(f"ℓ = {ℓ:g}", fontsize=13)

    rms_diff   = float(np.sqrt(np.nanmean(diff.values**2)))
    rms_online = float(np.sqrt(np.nanmean(on.values**2)))
    rel = rms_diff / rms_online if rms_online > 0 else float("inf")
    check(rel, f"  ℓ={ℓ:>4g}: rms(diff)/rms(online) = {rel:.2e},  max|diff| = {float(np.nanmax(np.abs(diff.values))):.2e}"
               f",  rms(online) = {rms_online:.2e}", print)

#---

#+++ Volume-integrated sub-filter KE ∫Kˢ dV vs time
dV_dom = ds.dV.sel(**in_domain)

print("\nVolume-integrated sub-filter KE ∫Kˢ dV (time-mean relative difference):")
for i, ℓ in enumerate(filter_scales):
    off_int    = integrate(offline[ℓ].sel(**in_domain), dV_dom).compute()
    on_int_fld = integrate(online[ℓ].sel(**in_domain),  dV_dom).compute()

    a = ax2[0, i]
    off_int.plot(ax=a, x="time", label="offline (recomputed)", color="k", lw=2)
    on_int_fld.plot(ax=a, x="time", label="online (∫ of field)", color="tab:red", ls="--", lw=2)
    if online_int.get(ℓ) is not None:
        online_int[ℓ].compute().plot(ax=a, x="time", label="online (Integral output)", color="tab:orange", ls=":", lw=2)

    a.set_title(f"ℓ = {ℓ:g}")
    a.set_ylabel("∫ Kˢ dV")
    a.legend(fontsize=9)

    denom = float(np.sqrt(np.nanmean(off_int.values**2)))
    rel = float(np.sqrt(np.nanmean((on_int_fld.values - off_int.values)**2))) / denom if denom > 0 else float("inf")
    check(rel, f"  ℓ={ℓ:>4g}: rms(online−offline)/rms(offline) = {rel:.2e}", print)

suptitle = f"Online vs offline sub-filter KE Kˢ   maps at t = {t_sel:.1f}, volume integrals over the run"
if label:
    suptitle += f"   {label}"
fig.suptitle(suptitle, fontsize=13)
outfile = str(FIGURES / f"{stem}_sfs_ke_comparison_t{t_sel:.1f}.png")
fig.savefig(outfile, dpi=150, bbox_inches="tight")
print(f"\nFigure saved to: {outfile}")
#---

finalize(print)
