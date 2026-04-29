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
parser.add_argument("--filename", default="output/khi_Nz2048_Ri0.10.nc", help="Path to simulation NetCDF file (used to derive energy transfer filename)")
parser.add_argument("--fixed-reference", action="store_true", default=False, help="Load output produced with the fixed-in-time reference profile")
args = parser.parse_args()

print("\n" + "="*70 + f"\n  {Path(__file__).name}\n  " + "  ".join(f"{k}={v}" for k,v in vars(args).items()) + "\n" + "="*70)
REPO_ROOT = Path(__file__).resolve().parent.parent
PP_OUTPUT = REPO_ROOT / "postprocessing" / "output"
filename = str(REPO_ROOT / args.filename) if not os.path.isabs(args.filename) else args.filename
ref_suffix = "_fixed_ref" if args.fixed_reference else ""
#---

#+++ Load energy transfer data
print("Loading energy transfer data...")
input_filename = str(PP_OUTPUT / (Path(filename).stem + f"_energy_transfer_sweep{ref_suffix}.nc"))
et = xr.open_dataset(input_filename, decode_timedelta=False)
et = et.sel(time=50, method="nearest")

# Add 1/ℓ as a non-dimension coordinate so plot.line can use it as the x axis
et = et.assign_coords(inv_scale=("filter_length_scale", 1.0 / et.filter_length_scale.values))
et["inv_scale"].attrs = {"long_name": "Inverse of filter scale 1/ℓ",}
print(f"  Loaded: {input_filename}")
print(f"  Time step: {et.time.values}   Filter scales: {et.filter_length_scale.values}")
#---

#+++ Plot
fig, ax = plt.subplots(figsize=(6, 3.5), constrained_layout=True)

for var, color, label_str in [("∫Π_KE dV", "#2166ac", r"$\Pi_{KE}$"), ("∫Π_APE dV", "#d6604d", r"$\Pi_{APE}$")]:
    ax.plot(et.inv_scale, et[var].values, color=color, label=label_str)
ax.axhline(0, color="k", lw=0.8, ls="--")
for ℓ in [0.5, 7]:
    ax.axvline(1.0 / ℓ, color="k", lw=0.8, ls="--")
ax.set_xscale("log")
ax.set_yscale("symlog", linthresh=1e-2)
ax.grid(True, alpha=0.3)
ax.set_xlabel("Inverse of filter scale 1/ℓ")
ax.legend()
ax2 = ax.secondary_xaxis("top", functions=(lambda x: 1/x, lambda x: 1/x))
ax2.set_xlabel("Filter scale ℓ")

info_parts = []
label = run_label(et.attrs)
if label:
    info_parts.append(label)
info_parts.append(f"$t = {float(et.time.values):.0f}$")
ax.text(0.98, 0.04, ",  ".join(info_parts), transform=ax.transAxes, fontsize=10, ha="right", va="bottom", bbox=dict(facecolor="white", edgecolor="none", pad=2, alpha=0.8))

plot_filename = str(REPO_ROOT / "figures" / os.path.basename(input_filename).replace("energy_transfer_sweep", "cross-scale_transfer_spectrum").replace(".nc", ".png"))
fig.savefig(plot_filename, dpi=150, bbox_inches="tight")
print(f"Plot saved to: {plot_filename}")
#---
