#!/usr/bin/env python
#+++ Imports
import os
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from src.aux03_plotting import run_label
#---

#+++ Configuration
import argparse
parser = argparse.ArgumentParser(description="The filter scale at which each cross-scale transfer changes sign, against time")
parser.add_argument("--filename", default="output/khi_Nz2048_Ri0.10.nc")
parser.add_argument("--fixed-reference", action="store_true", default=False)
parser.add_argument("--extension", default="edge")
parser.add_argument("--max-time", type=float, default=140.0)
parser.add_argument("--min-amplitude", type=float, default=0.02,
                    help="Skip times whose peak transfer is below this fraction of the record's peak: before the "
                         "instability develops the spectrum is round-off and its sign changes are meaningless.")
args = parser.parse_args()

print("\n" + "="*70 + f"\n  {Path(__file__).name}\n  " + "  ".join(f"{k}={v}" for k, v in vars(args).items()) + "\n" + "="*70)
REPO_ROOT = Path(__file__).resolve().parent.parent
PP_OUTPUT = REPO_ROOT / "postprocessing" / "output"
FIGURES = REPO_ROOT / "figures"; FIGURES.mkdir(exist_ok=True)
filename = str(REPO_ROOT / args.filename) if not os.path.isabs(args.filename) else args.filename
ref_suffix = "_fixed_ref" if args.fixed_reference else ""
ext_suffix = "" if args.extension == "edge" else f"_{args.extension}"
#---

#+++ Load
input_filename = str(PP_OUTPUT / (Path(filename).stem + f"_energy_transfer_sweep{ref_suffix}{ext_suffix}.nc"))
et = xr.open_dataset(input_filename, decode_timedelta=False).sel(time=slice(None, args.max_time)).sortby("filter_scale")
ℓ = et.filter_scale.values
t = et.time.values
fields = {r"$\Pi_K$": ("#2166ac", et["∫Π_K dV"].transpose("time", "filter_scale").values),
          r"$\Pi_A$": ("#d6604d", et["∫Π_A dV"].transpose("time", "filter_scale").values)}
print(f"  Loaded: {input_filename}\n  {len(t)} times, {len(ℓ)} scales")
#---

#+++ Crossings
# The structural claim in the text is about *where* each transfer reverses direction, not its magnitude.
# That is one number per time per term, so plotting it directly says the same thing as a Hovmoller with far
# less ink -- and it shows whether the crossover migrates as the billow grows, which an average cannot.
# Crossings are found by linear interpolation in log(l), since the scales are log-spaced.
def crossings(row):
    s = np.sign(row)
    out = []
    for i in np.where(np.diff(s) != 0)[0]:
        y0, y1 = row[i], row[i + 1]
        if y1 == y0:
            continue
        w = y0 / (y0 - y1)                      # fraction of the interval to the zero
        out.append(np.exp(np.log(ℓ[i]) + w * (np.log(ℓ[i + 1]) - np.log(ℓ[i]))))
    return out

fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
peak = max(np.abs(M).max() for _, M in fields.values())

for name, (color, M) in fields.items():
    amp = np.abs(M).max(axis=1)
    xs, ys, first = [], [], True
    for k, row in enumerate(M):
        if amp[k] < args.min_amplitude * peak:
            continue
        for c in crossings(row):
            xs.append(t[k]); ys.append(c)
    ax.scatter(xs, ys, s=22, color=color, label=name, alpha=0.85, edgecolor="none")
    print(f"  {name}: {len(xs)} sign changes over {len(set(xs))} times")

for g in [1, 7]:
    ax.axhline(g, color="k", lw=0.8, ls="--", alpha=0.4)
ax.set_yscale("log")
ax.set_xlabel("Time")
ax.set_ylabel("Filter scale ℓ of sign change")
ax.grid(True, alpha=0.3)
ax.legend(loc="best", fontsize=9, framealpha=0.9)
label = run_label(et.attrs)
if label:
    ax.text(0.98, 0.04, label, transform=ax.transAxes, fontsize=9, ha="right", va="bottom",
            bbox=dict(facecolor="white", edgecolor="none", pad=2, alpha=0.85))
#---

plot_filename = str(FIGURES / os.path.basename(input_filename)
                    .replace("energy_transfer_sweep", "S10_crossover").replace(".nc", ".pdf"))
fig.savefig(plot_filename, dpi=150, bbox_inches="tight")
print(f"Plot saved to: {plot_filename}")
