#!/usr/bin/env python
#+++ Imports
import logging
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # postprocessing/ on path for `src.*`
from src.aux00_utils import load_dataset_and_grid, make_gaussian_filter
from src.aux03_plotting import run_label
#---

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
print = logging.info

#+++ Configuration
import argparse
parser = argparse.ArgumentParser(description="Compare online (simulation-time) vs offline (post-processed) Gaussian filters")
parser.add_argument("--filename", default="output/khi_Nz256_Ri0.10.nc", help="Path to simulation NetCDF file")
parser.add_argument("--filter-scales", type=float, nargs="+", default=[1, 7], help="Filter ℓ (FWHM) values matching the online GaussianFilter widths")
parser.add_argument("--time", type=float, default=None, help="Target time for snapshot (default: midpoint of simulation)")
args = parser.parse_args()

print("\n" + "="*70 + f"\n  {Path(__file__).name}\n  " + "  ".join(f"{k}={v}" for k, v in vars(args).items()) + "\n" + "="*70)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # validation/ → postprocessing/ → repo root
FIGURES = REPO_ROOT / "figures"
FIGURES.mkdir(exist_ok=True)
filename = str(REPO_ROOT / args.filename) if not os.path.isabs(args.filename) else args.filename
stem = Path(filename).stem
#---

#+++ Load dataset
print("Loading simulation data...")
ds = load_dataset_and_grid(filename)
if args.time is None:
    args.time = float(ds.time.values[len(ds.time) // 2])
t_sel = float(ds.time.sel(time=args.time, method="nearest").values)
print(f"Selected time = {t_sel:.3f}  (requested {args.time})")
ds_snap = ds.sel(time=t_sel, method="nearest").squeeze()
#---

#+++ Compute offline-filtered fields and compare with online
# Both the online GaussianFilter (Oceanostics) and the offline one (aux00_utils)
# now use ℓ = FWHM as the filter length scale parameter.
field_names = ["u", "w", "b"]

print("Computing offline-filtered fields...")
results = {}
for ℓ in args.filter_scales:
    gf = make_gaussian_filter(ℓ, ds_snap)
    for name in field_names:
        online_var = f"{name}_ℓ{int(ℓ)}" if ℓ == int(ℓ) else f"{name}_ℓ{ℓ}"
        if online_var not in ds_snap:
            print(f"  WARNING: '{online_var}' not found in dataset, skipping")
            continue
        online = ds_snap[online_var]
        offline = gf.apply(ds_snap[name], dims=["x_caa", "z_aac"])
        results[(name, ℓ)] = dict(online=online, offline=offline, diff=online - offline)
print(f"  Computed {len(results)} field comparisons")
#---

#+++ Plot: one figure per field, one row per filter scale, 3 columns (online | offline | difference)
z_lim = (-4, 4)
n_scales = len(args.filter_scales)
label = run_label(ds.attrs)

print("\nComparison summary:")
for name in field_names:
    if not any((name, ℓ) in results for ℓ in args.filter_scales):
        continue

    fig, axes = plt.subplots(n_scales, 3, figsize=(15, 3.5 * n_scales), constrained_layout=True, squeeze=False)

    for i, ℓ in enumerate(args.filter_scales):
        key = (name, ℓ)
        if key not in results:
            for k in range(3):
                axes[i, k].set_visible(False)
            continue
        r = results[key]

        ℓ_str = int(ℓ) if ℓ == int(ℓ) else ℓ
        vmax = max(float(np.nanpercentile(np.abs(r["online"].values), 98)), float(np.nanpercentile(np.abs(r["offline"].values), 98)))
        kw = dict(x="x_caa", y="z_aac", add_colorbar=True, cmap="RdBu_r", vmin=-vmax, vmax=vmax)

        r["online"].plot(ax=axes[i, 0], **kw)
        axes[i, 0].set_title(f"Online {name}_ℓ{ℓ_str}")

        r["offline"].plot(ax=axes[i, 1], **kw)
        axes[i, 1].set_title(f"Offline {name} (ℓ={ℓ_str})")

        r["diff"].plot(ax=axes[i, 2], x="x_caa", y="z_aac", add_colorbar=True, cmap="RdBu_r", robust=True)
        axes[i, 2].set_title("Difference")

        for k in range(3):
            axes[i, k].set_ylim(*z_lim)
            axes[i, k].set_aspect("equal")

        axes[i, 0].set_ylabel(f"ℓ = {ℓ_str}", fontsize=13)

        diff = r["diff"].values
        online_vals = r["online"].values
        rms_diff = np.sqrt(np.nanmean(diff**2))
        rms_online = np.sqrt(np.nanmean(online_vals**2))
        rel_err = rms_diff / rms_online if rms_online > 0 else float("inf")
        max_abs = np.nanmax(np.abs(diff))
        print(f"  {name}_ℓ{ℓ_str}: rms(diff)/rms(online) = {rel_err:.2e}, max|diff| = {max_abs:.2e}")

    suptitle = f"Online vs offline Gaussian filter: {name}   t = {t_sel:.1f}"
    if label:
        suptitle += f"   {label}"
    fig.suptitle(suptitle, fontsize=13, y=1.01)

    outfile = str(FIGURES / f"{stem}_filter_comparison_{name}_t{t_sel:.1f}.png")
    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    print(f"  Figure saved to: {outfile}")
#---
