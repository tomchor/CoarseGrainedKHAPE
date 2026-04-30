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

# Add 1/ℓ as a non-dimension coordinate so plot.line can use it as the x axis
et = et.assign_coords(inv_scale=("filter_length_scale", 1.0 / et.filter_length_scale.values))
et["inv_scale"].attrs = {"long_name": "1/ℓ", "units": "m⁻¹"}
print(f"  Loaded: {input_filename}")
print(f"  Time steps: {len(et.time)}   Filter scales: {len(et.filter_length_scale)}")
#---

#+++ Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True, sharey=True)

vmax = float(max(abs(et["∫Π_KE dV"]).max(), abs(et["∫Π_APE dV"]).max()))
for ax, var in zip(axes, ["∫Π_KE dV", "∫Π_APE dV"]):
    et[var].plot.pcolormesh(x="time", y="filter_length_scale", ax=ax,
                            cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                            norm=plt.matplotlib.colors.SymLogNorm(linthresh=1e-2, vmin=-vmax, vmax=vmax))
    ax.set_yscale("log")
    ax.set_ylabel("ℓ  [m]")

axes[0].set_title("KE cross-scale transfer (Hovmöller)")
axes[1].set_title("APE cross-scale transfer (Hovmöller)")
label = run_label(et.attrs)
if label:
    fig.suptitle(label, fontsize=11)

plot_filename = str(REPO_ROOT / "figures" / os.path.basename(input_filename).replace(".nc", ".png"))
fig.savefig(plot_filename, dpi=150, bbox_inches="tight")
print(f"Plot saved to: {plot_filename}")
#---
