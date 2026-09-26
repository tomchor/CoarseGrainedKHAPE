#!/usr/bin/env python
#+++ Imports
import os
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from src.aux00_utils import PP_OUTPUT, reference_suffix
from src.aux03_plotting import run_label
#---

#+++ Configuration
import argparse
parser = argparse.ArgumentParser(description="Hovmöller plots (time vs filter scale) of Π_K and Π_A")
parser.add_argument("--filename", default="output/khi_Nz2048_Ri0.10.nc", help="Path to simulation NetCDF file (used to derive energy transfer filename)")
parser.add_argument("--fixed-reference", action="store_true", default=False, help="Load output produced with the fixed-in-time reference profile")
parser.add_argument("--reference", choices=["filtered", "true"], default="filtered",
                    help="Read the output built with this --reference (03-05 and sweep2 tag the 'true' ones _trueref)")
parser.add_argument("--max-time", type=float, default=140.0, help="Latest time included in the hovmöller")
parser.add_argument("--linthresh", type=float, default=1e-2, help="Linear threshold for symmetric log color scale")
args = parser.parse_args()

print("\n" + "="*70 + f"\n  {Path(__file__).name}\n  " + "  ".join(f"{k}={v}" for k,v in vars(args).items()) + "\n" + "="*70)
REPO_ROOT = Path(__file__).resolve().parent.parent
filename = str(REPO_ROOT / args.filename) if not os.path.isabs(args.filename) else args.filename
ref_suffix = ("_fixed_ref" if args.fixed_reference else "") + reference_suffix(args.reference)   # adds _trueref for --reference true
#---

#+++ Load energy transfer data
print("Loading energy transfer data...")
input_filename = str(PP_OUTPUT / (Path(filename).stem + f"_energy_transfer_sweep{ref_suffix}.nc"))
et = xr.open_dataset(input_filename, decode_timedelta=False).chunk(dict(time=1))
et = et.sel(time=slice(None, args.max_time))
print(f"  Loaded: {input_filename}")
print(f"  Filter scales: {et.filter_scale.values}")
print(f"  Time range: [{float(et.time.min()):.2f}, {float(et.time.max()):.2f}]")
#---

#+++ Build shared symmetric-log color scale across both panels
pi_K = et["∫Π_K dV"]
pi_A = et["∫Π_A dV"]
vmax = float(np.nanmax(np.abs(np.concatenate([pi_K.values.ravel(), pi_A.values.ravel()]))))
norm = SymLogNorm(linthresh=args.linthresh, linscale=1.0, vmin=-vmax, vmax=vmax, base=10)
cmap = "RdBu_r"
#---

#+++ Plot
fig, (ax_K, ax_A) = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True, sharey=True)

for ax, da, title in [(ax_K, pi_K, r"$\Pi_K$"), (ax_A, pi_A, r"$\Pi_A$")]:
    pcm = ax.pcolormesh(da.time.values, da.filter_scale.values, da.transpose("filter_scale", "time").values,
                        norm=norm, cmap=cmap, shading="auto", rasterized=True)
    ax.set_yscale("log")
    ax.set_xlabel("Time")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

ax_K.set_ylabel("Filter scale ℓ")

cbar = fig.colorbar(pcm, ax=(ax_K, ax_A), orientation="vertical", extend="both")
cbar.set_label(r"Cross-scale transfer")

label = run_label(et.attrs)
if label:
    fig.suptitle(label)
#---

#+++ Save
figures_dir = REPO_ROOT / "figures"
figures_dir.mkdir(exist_ok=True)
plot_filename = str(figures_dir / os.path.basename(input_filename).replace("energy_transfer_sweep", "hovmoller_PiK_PiA").replace(".nc", ".pdf"))
fig.savefig(plot_filename, dpi=150, bbox_inches="tight")
print(f"Plot saved to: {plot_filename}")
#---
