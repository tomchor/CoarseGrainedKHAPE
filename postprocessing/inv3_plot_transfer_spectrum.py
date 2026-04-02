#!/usr/bin/env python
#+++ Imports
import os
from pathlib import Path
import xarray as xr
import matplotlib.pyplot as plt
from aux03_plotting import run_label
#---

#+++ Configuration
import argparse
parser = argparse.ArgumentParser(description="Plot cross-scale KE and APE transfer spectra")
parser.add_argument("--filename", default="output/khi_Nz256_Ri0.10.nc",
                    help="Path to simulation NetCDF file (used to derive energy transfer filename)")
args = parser.parse_args()
REPO_ROOT = Path(__file__).resolve().parent.parent
filename = str(REPO_ROOT / args.filename) if not os.path.isabs(args.filename) else args.filename
#---

#+++ Load energy transfer data
print("Loading energy transfer data...")
input_filename = filename.replace(".nc", "_energy_transfer_sweep.nc")
et = xr.open_dataset(input_filename, decode_timedelta=False)

# Add 1/ℓ as a non-dimension coordinate so plot.line can use it as the x axis
et = et.assign_coords(inv_scale=("filter_length_scale", 1.0 / et.filter_length_scale.values))
et["inv_scale"].attrs = {"long_name": "1/ℓ", "units": "m⁻¹"}
print(f"  Loaded: {input_filename}")
print(f"  Time steps: {len(et.time)}   Filter scales: {len(et.filter_length_scale)}")
#---

#+++ Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True, sharey=True)

for ax, var in zip(axes, ["∫Π_KE dV", "∫Π_APE dV"]):
    et[var].plot.line(x="inv_scale", hue="time", ax=ax)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1e-4)
    ax.grid(True, alpha=0.3)
    ax2 = ax.secondary_xaxis("top", functions=(lambda x: 1/x, lambda x: 1/x))
    ax2.set_xlabel("ℓ  [m]")

axes[0].set_title("KE cross-scale transfer spectrum")
axes[1].set_title("APE cross-scale transfer spectrum")
label = run_label(et.attrs)
if label:
    fig.suptitle(label, fontsize=11)

plot_filename = str(REPO_ROOT / "figures" / os.path.basename(input_filename).replace(".nc", ".png"))
fig.savefig(plot_filename, dpi=150, bbox_inches="tight")
print(f"Plot saved to: {plot_filename}")
#---
