#!/usr/bin/env python
#+++ Imports
import os
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import AsinhNorm
from src.aux03_plotting import collapse_time_pairs
#---

#+++ Configuration
import argparse
parser = argparse.ArgumentParser(description="Hovmollers of the total cross-scale transfer and its two components, on a shared 1/ℓ axis")
parser.add_argument("--filename", default="output/khi_Nz2048_Ri0.10.nc", help="Path to simulation NetCDF file (used to derive energy transfer filename)")
parser.add_argument("--fixed-reference", action="store_true", default=False, help="Load output produced with the fixed-in-time reference profile")
parser.add_argument("--extension", default="edge", help="Which wall-extension run to plot ('edge' is the default sweep)")
parser.add_argument("--max-time", type=float, default=140.0, help="Latest time included, in both the average and the Hovmollers")
parser.add_argument("--linear-width", type=float, default=1e-2,
                    help="Width of the near-linear region of the colour scale; the mapping is logarithmic "
                         "beyond it, with a smooth transition.")
parser.add_argument("--cmap", default="coolwarm", help="Diverging colormap for the transfer")
parser.add_argument("--orient", choices=["scale-x", "time-x"], default="scale-x",
                    help="Which variable runs along x. 'scale-x' puts 1/l on x with time up the y axis; "
                         "'time-x' flips them, the conventional Hovmoller orientation. The flipped version "
                         "is written to its own file, so the two can sit side by side.")
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
# One scale across all three rows, so a colour means the same rate in every panel.
#
# The transfer spans about three decades and changes sign, so a linear scale would show the peak and
# leave the rest blank. asinh rather than symlog for the compression: symlog is piecewise, with a real
# kink where its linear region meets its logarithmic one, and that kink shows up in the image as a band
# of near-constant colour at |value| ~ linthresh -- a feature of the threshold, not of the flow. asinh is
# the same idea, linear near zero and logarithmic far from it, but smooth everywhere.
vmax = float(np.nanmax(np.abs(np.concatenate([pi_T.values.ravel(), pi_K.values.ravel(), pi_A.values.ravel()]))))
norm = AsinhNorm(linear_width=args.linear_width, vmin=-vmax, vmax=vmax)
#---

#+++ Figure
# All three rows are Hovmollers on one 1/ℓ axis and one colour scale, so a colour means the same rate in
# every panel and the total can be compared with its parts by eye. Time runs up the y axis in each.
fig, axes = plt.subplots(3, 1, figsize=(7.0, args.fig_height), constrained_layout=True,
                         sharex=True, sharey=(args.orient == "time-x"),
                         gridspec_kw=dict(height_ratios=[1.0, args.hov_ratio, args.hov_ratio]))
ax_T, ax_K, ax_A = axes

time_x = args.orient == "time-x"
for ax, da, letter, name in [(ax_T, pi_T, "a", r"$\Pi_K + \Pi_A$"),
                             (ax_K, pi_K, "b", r"$\Pi_K$"),
                             (ax_A, pi_A, "c", r"$\Pi_A$")]:
    # A light ground behind the cells: these colormaps put near-white at zero, so on a white page a
    # quiescent region would be indistinguishable from no data at all.
    ax.set_facecolor("#e9e9e9")
    if time_x:
        pcm = ax.pcolormesh(da.time.values, inv, da.transpose("filter_scale", "time").values,
                            norm=norm, cmap=args.cmap, shading="auto", rasterized=True)
        ax.set_yscale("log")
        ax.set_ylabel(r"$1/\ell$")
    else:
        pcm = ax.pcolormesh(inv, da.time.values, da.transpose("time", "filter_scale").values,
                            norm=norm, cmap=args.cmap, shading="auto", rasterized=True)
        ax.set_xscale("log")
        ax.set_ylabel(r"$t$")
    ax.grid(True, alpha=0.15, color="k")
    # Inside the panel, upper right: the top edge of (a) is then free for the ℓ axis, and no row spends
    # vertical space on a title. Boxed, since it sits over data.
    ax.text(0.985, 0.955, f"({letter})  {name}", transform=ax.transAxes, fontsize=11,
            ha="right", va="top", bbox=dict(facecolor="white", edgecolor="none", pad=2.5, alpha=0.85))

# Both conventions on the figure: the data is plotted against 1/ℓ, but the text discusses scales as ℓ.
# It goes on the far edge of (a), which the in-panel labels leave free.
if time_x:
    ax_A.set_xlabel(r"$t$")
    ax_ell = ax_T.secondary_yaxis("right", functions=(lambda x: 1 / x, lambda x: 1 / x))
    ax_ell.set_ylabel(r"filter scale $\ell$")
else:
    ax_A.set_xlabel(r"$1/\ell$")
    ax_ell = ax_T.secondary_xaxis("top", functions=(lambda x: 1 / x, lambda x: 1 / x))
    ax_ell.set_xlabel(r"filter scale $\ell$")

# The two scales the budgets are shown at, marked on the total so the detailed figures can be located
# against the cascade. Labelled, since two bare lines on one panel would be ambiguous.
for ℓ_mark in [7, 1]:
    if time_x:
        ax_T.axhline(1.0 / ℓ_mark, color="k", lw=0.9, ls="--", alpha=0.55)
        ax_T.text(0.012, 1.0 / ℓ_mark, f"$\\ell={ℓ_mark}$", transform=ax_T.get_yaxis_transform(),
                  fontsize=8, ha="left", va="bottom",
                  bbox=dict(facecolor="white", edgecolor="none", pad=1.2, alpha=0.8))
    else:
        ax_T.axvline(1.0 / ℓ_mark, color="k", lw=0.9, ls="--", alpha=0.55)
        ax_T.text(1.0 / ℓ_mark, 0.012, f"$\\ell={ℓ_mark}$", transform=ax_T.get_xaxis_transform(),
                  fontsize=8, ha="left", va="bottom", rotation=90,
                  bbox=dict(facecolor="white", edgecolor="none", pad=1.2, alpha=0.8))

cbar = fig.colorbar(pcm, ax=axes.tolist(), orientation="vertical", extend="both",
                    fraction=0.032, pad=0.015)
cbar.set_label(r"$\int \Pi\, \mathrm{d}V$")

#---

plot_filename = str(FIGURES / os.path.basename(input_filename)
                    .replace("energy_transfer_sweep",
                             "S6_spectrum_hovmoller" + ("_timex" if time_x else "")).replace(".nc", ".pdf"))
fig.savefig(plot_filename, dpi=150, bbox_inches="tight")
print(f"Plot saved to: {plot_filename}")
