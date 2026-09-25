#!/usr/bin/env python
#+++ Imports
import os
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from src.aux03_plotting import run_label, collapse_time_pairs
#---

#+++ Configuration
import argparse
parser = argparse.ArgumentParser(description="Hovmollers of the total cross-scale transfer and its two components, on a shared 1/ℓ axis")
parser.add_argument("--filename", default="output/khi_Nz2048_Ri0.10.nc", help="Path to simulation NetCDF file (used to derive energy transfer filename)")
parser.add_argument("--fixed-reference", action="store_true", default=False, help="Load output produced with the fixed-in-time reference profile")
parser.add_argument("--extension", default="edge", help="Which wall-extension run to plot ('edge' is the default sweep)")
parser.add_argument("--max-time", type=float, default=140.0, help="Latest time included, in both the average and the Hovmollers")
parser.add_argument("--linthresh", type=float, default=1e-2, help="Linear threshold of the symmetric-log colour scale")
parser.add_argument("--fig-height", type=float, default=7.2, help="Figure height in inches; lower compresses the time axis")
parser.add_argument("--hov-ratio", type=float, default=1.0, help="Height of the component rows relative to the total row")
args = parser.parse_args()

print("\n" + "="*70 + f"\n  {Path(__file__).name}\n  " + "  ".join(f"{k}={v}" for k, v in vars(args).items()) + "\n" + "="*70)
REPO_ROOT = Path(__file__).resolve().parent.parent
PP_OUTPUT = REPO_ROOT / "postprocessing" / "output"
FIGURES   = REPO_ROOT / "figures"
FIGURES.mkdir(exist_ok=True)
filename = str(REPO_ROOT / args.filename) if not os.path.isabs(args.filename) else args.filename
ref_suffix = "_fixed_ref" if args.fixed_reference else ""
ext_suffix = "" if args.extension == "edge" else f"_{args.extension}"
#---

#+++ Load
print("Loading energy transfer data...")
input_filename = str(PP_OUTPUT / (Path(filename).stem + f"_energy_transfer_sweep{ref_suffix}{ext_suffix}.nc"))
et = collapse_time_pairs(xr.open_dataset(input_filename, decode_timedelta=False)).chunk(dict(time=1))
et = et.sel(time=slice(None, args.max_time)).sortby("filter_scale")
t0, t1 = float(et.time.min()), float(et.time.max())
print(f"  Loaded: {input_filename}")
print(f"  Filter scales: {et.filter_scale.values}")
print(f"  Time range: [{t0:.2f}, {t1:.2f}]")

# Pi_K and Pi_A are the two halves of the energy crossing the filter scale, so their sum is the net
# cascade -- the quantity whose sign says whether a scale is forward or inverse overall, and which two
# components of opposite sign can hide. It leads, with the components below it for attribution.
pi_K, pi_A = et["∫Π_K dV"], et["∫Π_A dV"]
pi_T = pi_K + pi_A
inv = 1.0 / et.filter_scale.values           # the shared x axis of all three rows
#---

#+++ Colour scale
# One symmetric-log scale across both Hovmollers, so the two rows are directly comparable and a colour
# means the same rate in each. Symmetric log rather than linear because the transfer spans decades and
# changes sign; `linthresh` sets where it stops being logarithmic so zero is representable.
vmax = float(np.nanmax(np.abs(np.concatenate([pi_T.values.ravel(), pi_K.values.ravel(), pi_A.values.ravel()]))))
norm = SymLogNorm(linthresh=args.linthresh, linscale=1.0, vmin=-vmax, vmax=vmax, base=10)
C_PI_K, C_PI_A = "#2166ac", "#d6604d"
#---

#+++ Figure
# All three rows are Hovmollers on one 1/ℓ axis and one colour scale, so a colour means the same rate in
# every panel and the total can be compared with its parts by eye. Time runs up the y axis in each.
fig, axes = plt.subplots(3, 1, figsize=(7.0, args.fig_height), constrained_layout=True, sharex=True,
                         gridspec_kw=dict(height_ratios=[1.0, args.hov_ratio, args.hov_ratio]))
ax_T, ax_K, ax_A = axes

C_PI_K, C_PI_A, C_PI_T = "#2166ac", "#d6604d", "#000000"
for ax, da, name, color in [(ax_T, pi_T, r"$\Pi_K + \Pi_A$", C_PI_T),
                            (ax_K, pi_K, r"$\Pi_K$",          C_PI_K),
                            (ax_A, pi_A, r"$\Pi_A$",          C_PI_A)]:
    pcm = ax.pcolormesh(inv, da.time.values, da.transpose("time", "filter_scale").values,
                        norm=norm, cmap="RdBu_r", shading="auto", rasterized=True)
    ax.set_xscale("log")
    ax.set_ylabel("Time")
    ax.grid(True, alpha=0.2)
    ax.text(0.015, 0.96, name, transform=ax.transAxes, fontsize=12, ha="left", va="top",
            color=color, bbox=dict(facecolor="white", edgecolor="none", pad=2.5, alpha=0.85))

ax_A.set_xlabel("Inverse of filter scale 1/ℓ")
ax_top = ax_T.secondary_xaxis("top", functions=(lambda x: 1 / x, lambda x: 1 / x))
ax_top.set_xlabel("Filter scale ℓ")

cbar = fig.colorbar(pcm, ax=axes.tolist(), orientation="vertical", extend="both",
                    fraction=0.032, pad=0.015)
cbar.set_label("Cross-scale transfer")

label = run_label(et.attrs)
ax_T.text(0.98, 0.04, ",  ".join(filter(None, [label, f"$t \\in [{t0:.0f}, {t1:.0f}]$"])),
          transform=ax_T.transAxes, fontsize=9, ha="right", va="bottom",
          bbox=dict(facecolor="white", edgecolor="none", pad=2, alpha=0.85))

# Each row carries a term label at the far left, so the letter is nudged clear of it -- further on the
# total row, whose label is the widest.
for ax, letter, x in [(ax_T, "a", 0.225), (ax_K, "b", 0.105), (ax_A, "c", 0.105)]:
    ax.text(x, 0.96, f"({letter})", transform=ax.transAxes, fontsize=11, fontweight="bold",
            ha="left", va="top")
#---

plot_filename = str(FIGURES / os.path.basename(input_filename)
                    .replace("energy_transfer_sweep", "S6_spectrum_hovmoller").replace(".nc", ".pdf"))
fig.savefig(plot_filename, dpi=150, bbox_inches="tight")
print(f"Plot saved to: {plot_filename}")
