#!/usr/bin/env python
#+++ Imports
import os
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from src.aux03_plotting import run_label
#---

#+++ Configuration
import argparse
parser = argparse.ArgumentParser(description="Three-row sweep figure: transfer spectrum (top) over Hovmollers of Π_K and Π_A, all on a shared 1/ℓ axis")
parser.add_argument("--filename", default="output/khi_Nz2048_Ri0.10.nc", help="Path to simulation NetCDF file (used to derive energy transfer filename)")
parser.add_argument("--fixed-reference", action="store_true", default=False, help="Load output produced with the fixed-in-time reference profile")
parser.add_argument("--extension", default="edge", help="Which wall-extension run to plot ('edge' is the default sweep)")
parser.add_argument("--max-time", type=float, default=140.0, help="Latest time included, in both the average and the Hovmollers")
parser.add_argument("--linthresh", type=float, default=1e-2, help="Linear threshold of the symmetric-log colour scale")
parser.add_argument("--fig-height", type=float, default=7.2, help="Figure height in inches; lower compresses the time axis")
parser.add_argument("--hov-ratio", type=float, default=0.95, help="Height of each Hovmoller row relative to the spectrum row")
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
et = xr.open_dataset(input_filename, decode_timedelta=False).chunk(dict(time=1))
et = et.sel(time=slice(None, args.max_time)).sortby("filter_scale")
t0, t1 = float(et.time.min()), float(et.time.max())
print(f"  Loaded: {input_filename}")
print(f"  Filter scales: {et.filter_scale.values}")
print(f"  Time range: [{t0:.2f}, {t1:.2f}]")

pi_K, pi_A = et["∫Π_K dV"], et["∫Π_A dV"]
inv = 1.0 / et.filter_scale.values           # the shared x axis of all three rows
#---

#+++ Colour scale
# One symmetric-log scale across both Hovmollers, so the two rows are directly comparable and a colour
# means the same rate in each. Symmetric log rather than linear because the transfer spans decades and
# changes sign; `linthresh` sets where it stops being logarithmic so zero is representable.
vmax = float(np.nanmax(np.abs(np.concatenate([pi_K.values.ravel(), pi_A.values.ravel()]))))
norm = SymLogNorm(linthresh=args.linthresh, linscale=1.0, vmin=-vmax, vmax=vmax, base=10)
C_PI_K, C_PI_A = "#2166ac", "#d6604d"
#---

#+++ Figure
# Shared x across all three rows, so a feature in the spectrum lines up vertically with the times that
# produced it. The Hovmoller rows are deliberately not tall: their y axis is time, which is read for where
# features sit rather than off a fine scale, so height spent there buys little. --fig-height and
# --hov-ratio tune it without editing the script.
fig, axes = plt.subplots(3, 1, figsize=(7.0, args.fig_height), constrained_layout=True, sharex=True,
                         gridspec_kw=dict(height_ratios=[1.0, args.hov_ratio, args.hov_ratio]))
ax_sp, ax_K, ax_A = axes

#+++ Row 1: the time-averaged spectrum
for var, color, lab in [("∫Π_K dV", C_PI_K, r"$\Pi_K$"), ("∫Π_A dV", C_PI_A, r"$\Pi_A$")]:
    ax_sp.plot(inv, et[var].mean("time").values, color=color, lw=1.8, label=lab)
ax_sp.axhline(0, color="k", lw=0.8, ls="--")
ax_sp.set_yscale("symlog", linthresh=args.linthresh)
ax_sp.set_ylabel("Volume-integrated rate")
ax_sp.legend(loc="best", fontsize=9, framealpha=0.9)
ax_sp.grid(True, alpha=0.3)
# Filter scale on top, since 1/ℓ is what the x axis actually is
ax_top = ax_sp.secondary_xaxis("top", functions=(lambda x: 1 / x, lambda x: 1 / x))
ax_top.set_xlabel("Filter scale ℓ")
label = run_label(et.attrs)
ax_sp.text(0.98, 0.04, ",  ".join(filter(None, [label, f"$t \\in [{t0:.0f}, {t1:.0f}]$"])),
           transform=ax_sp.transAxes, fontsize=9, ha="right", va="bottom",
           bbox=dict(facecolor="white", edgecolor="none", pad=2, alpha=0.85))
#---

#+++ Rows 2-3: Hovmollers, transposed so 1/ℓ is the x axis and time runs up the y axis
for ax, da, name, color in [(ax_K, pi_K, r"$\Pi_K$", C_PI_K), (ax_A, pi_A, r"$\Pi_A$", C_PI_A)]:
    pcm = ax.pcolormesh(inv, da.time.values, da.transpose("time", "filter_scale").values,
                        norm=norm, cmap="RdBu_r", shading="auto", rasterized=True)
    ax.set_ylabel("Time")
    ax.grid(True, alpha=0.2)
    ax.text(0.015, 0.96, name, transform=ax.transAxes, fontsize=12, ha="left", va="top",
            color=color, bbox=dict(facecolor="white", edgecolor="none", pad=2.5, alpha=0.85))

ax_A.set_xlabel("Inverse of filter scale 1/ℓ")
for ax in axes:
    ax.set_xscale("log")

cbar = fig.colorbar(pcm, ax=(ax_K, ax_A), orientation="vertical", extend="both", fraction=0.035, pad=0.015)
cbar.set_label("Cross-scale transfer")

# The Hovmoller rows already carry a term label at the far left, so their letter is nudged right of it.
for ax, letter, x in [(ax_sp, "a", 0.015), (ax_K, "b", 0.105), (ax_A, "c", 0.105)]:
    ax.text(x, 0.96, f"({letter})", transform=ax.transAxes, fontsize=11, fontweight="bold",
            ha="left", va="top")
#---

plot_filename = str(FIGURES / os.path.basename(input_filename)
                    .replace("energy_transfer_sweep", "S6_spectrum_hovmoller").replace(".nc", ".pdf"))
fig.savefig(plot_filename, dpi=150, bbox_inches="tight")
print(f"Plot saved to: {plot_filename}")
