#!/usr/bin/env python
#+++ Imports
import os
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from src.aux00_utils import PP_OUTPUT
from src.aux03_plotting import run_label
#---

#+++ Configuration
import argparse
parser = argparse.ArgumentParser(description="Cross-scale transfer spectra, with shading at +-1 standard deviation in time")
parser.add_argument("--filename", default="output/khi_Nz2048_Ri0.10.nc", help="Path to simulation NetCDF file (used to derive energy transfer filename)")
parser.add_argument("--fixed-reference", action="store_true", default=False, help="Load output produced with the fixed-in-time reference profile")
parser.add_argument("--extension", default="edge", help="Which wall-extension run to plot ('edge' is the default sweep)")
parser.add_argument("--min-time", type=float, default=0.0,
                    help="Earliest time included. The run is non-stationary -- the transfer grows from ~0 as the "
                         "instability develops -- so over the full record the standard deviation mostly measures "
                         "that growth. Set this past the onset to make the spread describe variability instead.")
parser.add_argument("--max-time", type=float, default=140.0, help="Latest time included in the mean and the spread")
parser.add_argument("--linthresh", type=float, default=1e-2, help="Linear threshold of the symmetric-log y axis")
parser.add_argument("--fig-height", type=float, default=3.8, help="Figure height in inches")
args = parser.parse_args()

print("\n" + "="*70 + f"\n  {Path(__file__).name}\n  " + "  ".join(f"{k}={v}" for k, v in vars(args).items()) + "\n" + "="*70)
REPO_ROOT = Path(__file__).resolve().parent.parent
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
et = et.sel(time=slice(args.min_time, args.max_time)).sortby("filter_scale")
t0, t1 = float(et.time.min()), float(et.time.max())
inv = 1.0 / et.filter_scale.values
print(f"  Loaded: {input_filename}")
print(f"  Filter scales: {et.filter_scale.values}")
print(f"  Time range: [{t0:.2f}, {t1:.2f}]  ({et.sizes['time']} records)")
#---

#+++ Plot
# The band is +-1 standard deviation of the term across time, not a standard error: it says how much the
# transfer at a given scale varies over the run, which is the physically interesting spread. It is wide
# wherever the instability is still developing, and that width is the point rather than noise.
C = {"∫Π_K dV": "#2166ac", "∫Π_A dV": "#d6604d"}
LABEL = {"∫Π_K dV": r"$\Pi_K$", "∫Π_A dV": r"$\Pi_A$"}

fig, ax = plt.subplots(figsize=(7.0, args.fig_height), constrained_layout=True)

for var in ["∫Π_K dV", "∫Π_A dV"]:
    m = et[var].mean("time").values
    s = et[var].std("time").values
    ax.fill_between(inv, m - s, m + s, color=C[var], alpha=0.20, linewidth=0)
    ax.plot(inv, m, color=C[var], lw=1.8, label=LABEL[var])

ax.axhline(0, color="k", lw=0.8, ls="--")
for ℓ in [1, 7]:
    ax.axvline(1.0 / ℓ, color="k", lw=0.8, ls="--", alpha=0.4)
ax.set_xscale("log")
ax.set_yscale("symlog", linthresh=args.linthresh)
ax.grid(True, alpha=0.3)
ax.set_xlabel("Inverse of filter scale 1/ℓ")
ax.set_ylabel("Volume-integrated rate")
ax.legend(loc="best", fontsize=9, framealpha=0.9)
ax_top = ax.secondary_xaxis("top", functions=(lambda x: 1 / x, lambda x: 1 / x))
ax_top.set_xlabel("Filter scale ℓ")

label = run_label(et.attrs)
ax.text(0.98, 0.04, ",  ".join(filter(None, [label, f"$t \\in [{t0:.0f}, {t1:.0f}]$", r"shading: $\pm 1\,\sigma$ in time"])),
        transform=ax.transAxes, fontsize=9, ha="right", va="bottom",
        bbox=dict(facecolor="white", edgecolor="none", pad=2, alpha=0.85))
#---

plot_filename = str(FIGURES / os.path.basename(input_filename)
                    .replace("energy_transfer_sweep", "S7_spectrum_spread").replace(".nc", ".pdf"))
fig.savefig(plot_filename, dpi=150, bbox_inches="tight")
print(f"Plot saved to: {plot_filename}")
