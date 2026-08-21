#!/usr/bin/env python
#+++ Imports
import os
from pathlib import Path
import xarray as xr
import matplotlib.pyplot as plt
from src.aux03_plotting import run_label
#---

#+++ Configuration
import argparse
parser = argparse.ArgumentParser(description="Two-panel sweep figure: cross-scale transfer & SFS APE->KE exchange spectra (top); ℓ-derivative of the SFS exchange (bottom)")
parser.add_argument("--filename", default="output/khi_Nz2048_Ri0.10.nc", help="Path to simulation NetCDF file (used to derive energy transfer filename)")
parser.add_argument("--fixed-reference", action="store_true", default=False, help="Load output produced with the fixed-in-time reference profile")
def str2bool(s):
    if s.lower() in ("true", "1", "yes"):  return True
    if s.lower() in ("false", "0", "no"):  return False
    raise argparse.ArgumentTypeError(f"Expected boolean, got {s!r}")
parser.add_argument("--time", type=float, default=40, help="Snapshot time (nearest available will be used; ignored if --time-average true)")
parser.add_argument("--time-average", type=str2bool, default=True, metavar="BOOL", help="Average transfer terms over the whole time range (true/false)")
parser.add_argument("--max-average-time", type=float, default=140.0, help="Latest time included when averaging (only used if --time-average true)")
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
et = xr.open_dataset(input_filename, decode_timedelta=False).chunk(dict(time=1))
if args.time_average:
    et = et.sel(time=slice(None, args.max_average_time))
    t0, t1 = float(et.time.min()), float(et.time.max())
    et = et.mean("time", keep_attrs=True)
    time_label = f"$t \\in [{t0:.0f}, {t1:.0f}]$"
else:
    et = et.sel(time=args.time, method="nearest")
    time_label = f"$t = {float(et.time.values):.0f}$"

et = et.assign_coords(inv_scale=("filter_scale", 1.0 / et.filter_scale.values))
et["inv_scale"].attrs = {"long_name": "Inverse of filter scale 1/ℓ",}
print(f"  Loaded: {input_filename}")
print(f"  Filter scales: {et.filter_scale.values}")
#---

#+++ Color palette (shared between panels)
C_PI_K   = "#2166ac"  # blue
C_PI_A   = "#d6604d"  # red
C_SFS    = "#1b7837"  # green
LW       = 1.8
MK       = "o"
MS       = 4
#---

#+++ Figure
fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(7, 8), constrained_layout=True, sharex=True)

#+++ Top panel: cross-scale transfer (Π_K, Π_A) and the SFS APE->KE exchange
for var, color, label_str in [
    ("∫Π_K dV",           C_PI_K,  r"$\Pi_K$  (cross-scale KE flux)"),
    ("∫Π_A dV",           C_PI_A,  r"$\Pi_A$  (cross-scale APE flux)"),
    ("∫(SFS APE->KE) dV", C_SFS,   r"SFS APE$\to$KE exchange: $\overline{w\,b_r} - \bar w\,b_r^\ell$"),
]:
    ax_top.plot(et.inv_scale, et[var].values, color=color, lw=LW, label=label_str)

ax_top.axhline(0, color="k", lw=0.8, ls="--")
for ℓ in [1, 7]:
    ax_top.axvline(1.0 / ℓ, color="k", lw=0.8, ls="--", alpha=0.4)
ax_top.set_xscale("log")
ax_top.set_yscale("symlog", linthresh=1e-2)
ax_top.grid(True, alpha=0.3)
ax_top.set_ylabel("Volume-integrated rate")
ax_top.set_title("Cross-scale transfer and APE↔KE exchange spectra")
ax_top.legend(loc="best", fontsize=9, framealpha=0.9)
ax_top2 = ax_top.secondary_xaxis("top", functions=(lambda x: 1/x, lambda x: 1/x))
ax_top2.set_xlabel("Filter scale ℓ")

info_parts = []
label = run_label(et.attrs)
if label:
    info_parts.append(label)
info_parts.append(time_label)
ax_top.text(0.98, 0.04, ",  ".join(info_parts), transform=ax_top.transAxes, fontsize=10,
            ha="right", va="bottom", bbox=dict(facecolor="white", edgecolor="none", pad=2, alpha=0.85))
#---

#+++ Bottom panel: ℓ-derivative of the SFS exchange term
d_sfs = et["∫(SFS APE->KE) dV"].differentiate("filter_scale")

ax_bot.plot(et.inv_scale, d_sfs.values, color=C_SFS, lw=LW,
            label=r"$\partial_\ell \int(\overline{w\,b_r}-\bar w\,b_r^\ell)\,dV$  (SFS exchange)")

ax_bot.axhline(0, color="k", lw=0.8, ls="--")
for ℓ in [1, 7]:
    ax_bot.axvline(1.0 / ℓ, color="k", lw=0.8, ls="--", alpha=0.4)
ax_bot.set_xscale("log")
ax_bot.set_yscale("symlog", linthresh=1e-2)
ax_bot.grid(True, alpha=0.3)
ax_bot.set_xlabel("Inverse of filter scale 1/ℓ")
ax_bot.set_ylabel(r"$\partial_\ell$ of exchange rate")
ax_bot.set_title("Filter-scale derivative of the SFS APE↔KE exchange")
ax_bot.legend(loc="best", fontsize=9, framealpha=0.9)
#---

plot_filename = str(REPO_ROOT / "figures" / os.path.basename(input_filename).replace("energy_transfer_sweep", "S3_sweep").replace(".nc", ".pdf"))
fig.savefig(plot_filename, dpi=150, bbox_inches="tight")
print(f"Plot saved to: {plot_filename}")
