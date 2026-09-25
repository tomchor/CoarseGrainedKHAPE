#!/usr/bin/env python
#+++ Imports
import os
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from src.aux03_plotting import run_label, collapse_time_pairs
#---

#+++ Configuration
import argparse
parser = argparse.ArgumentParser(description="Cross-scale transfer spectra resolved by phase of the instability, instead of averaged over the run")
parser.add_argument("--filename", default="output/khi_Nz2048_Ri0.10.nc")
parser.add_argument("--fixed-reference", action="store_true", default=False)
parser.add_argument("--extension", default="edge")
parser.add_argument("--max-time", type=float, default=140.0)
parser.add_argument("--linthresh", type=float, default=1e-2)
parser.add_argument("--phase-edges", type=float, nargs="+", default=None,
                    help="Times bounding the phases. Default: placed from the transfer amplitude itself "
                         "-- 10%% and 50%% of its peak on the rise, the peak, and the end of the record.")
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
et = collapse_time_pairs(xr.open_dataset(input_filename, decode_timedelta=False)).sel(time=slice(None, args.max_time)).sortby("filter_scale")
inv = 1.0 / et.filter_scale.values
t = et.time.values
K = et["∫Π_K dV"].transpose("time", "filter_scale").values
A = et["∫Π_A dV"].transpose("time", "filter_scale").values
print(f"  Loaded: {input_filename}\n  {len(t)} times over [{t.min():.1f}, {t.max():.1f}], {len(inv)} scales")
#---

#+++ Phases
# A time-averaged spectrum is the wrong summary here: the spectrum changes *shape* as the billow develops,
# not just amplitude (normalising each time by its own peak does not reduce the scatter). So the record is
# cut into phases and each is shown as its own curve -- a state rather than an average over states.
# Edges are placed from the transfer amplitude itself so they track the instability rather than the clock.
amp = np.abs(K).max(axis=1)
if args.phase_edges:
    edges = np.array(sorted(args.phase_edges), dtype=float)
    if edges[0] > t.min():  edges = np.r_[t.min(), edges]
    if edges[-1] < t.max(): edges = np.r_[edges, t.max()]
else:
    ipk = int(np.argmax(amp))
    rise = amp[:ipk + 1]
    def first_at(frac):
        j = np.argmax(rise >= frac * amp[ipk])
        return float(t[j])
    edges = np.array([t.min(), first_at(0.10), first_at(0.50), float(t[ipk]), t.max()])
    edges = np.unique(edges)
print("  phases: " + ",  ".join(f"[{a:.0f}, {b:.0f}]" for a, b in zip(edges[:-1], edges[1:])))
#---

#+++ Plot
# Phases are ordered, so they get a sequential ramp (light = early, dark = late) rather than categorical
# hues: the reader should be able to see the direction of evolution without consulting the legend.
fig, (ax_K, ax_A, ax_T) = plt.subplots(3, 1, figsize=(7.0, 9.2), constrained_layout=True, sharex=True)
# Pi_K and Pi_A keep the hues they carry elsewhere in the paper; the total is neutral, since it is a
# derived sum rather than a third measured term. Each ramp runs light (early) to dark (late).
ramps = {"K": plt.get_cmap("Blues"), "A": plt.get_cmap("Reds"), "T": plt.get_cmap("Greys")}
spans = {"K": (0.30, 0.92), "A": (0.30, 0.92), "T": (0.40, 0.95)}
n = len(edges) - 1

# Pi_K and Pi_A are the two halves of the energy crossing the filter scale, so their sum is the net
# cascade -- the quantity whose sign says whether a scale is forward or inverse overall, and which
# components of opposite sign can hide.
for ax, M, key, name in [(ax_K, K, "K", r"$\Pi_K$"), (ax_A, A, "A", r"$\Pi_A$"),
                         (ax_T, K + A, "T", r"$\Pi_K + \Pi_A$")]:
    for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        sel = (t >= a) & (t <= b)
        if not sel.any():
            continue
        lo, hi = spans[key]
        ax.plot(inv, M[sel].mean(axis=0), lw=1.8,
                color=ramps[key](lo + (hi - lo) * i / max(n - 1, 1)),
                label=f"$t \\in [{a:.0f}, {b:.0f}]$")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    for ℓ in [1, 7]:
        ax.axvline(1.0 / ℓ, color="k", lw=0.8, ls="--", alpha=0.35)
    ax.set_xscale("log"); ax.set_yscale("symlog", linthresh=args.linthresh)
    ax.grid(True, alpha=0.3)
    ax.set_ylabel("Volume-integrated rate")
    ax.legend(loc="best", fontsize=8, framealpha=0.9, title=name, title_fontsize=9)

ax_T.set_xlabel("Inverse of filter scale 1/ℓ")
ax_top = ax_K.secondary_xaxis("top", functions=(lambda x: 1 / x, lambda x: 1 / x))
ax_top.set_xlabel("Filter scale ℓ")
label = run_label(et.attrs)
if label:
    ax_K.text(0.98, 0.04, label, transform=ax_K.transAxes, fontsize=9, ha="right", va="bottom",
              bbox=dict(facecolor="white", edgecolor="none", pad=2, alpha=0.85))
for ax, letter in [(ax_K, "a"), (ax_A, "b"), (ax_T, "c")]:
    ax.text(0.015, 0.96, f"({letter})", transform=ax.transAxes, fontsize=11, fontweight="bold", ha="left", va="top")
#---

plot_filename = str(FIGURES / os.path.basename(input_filename)
                    .replace("energy_transfer_sweep", "S8_spectrum_phases").replace(".nc", ".pdf"))
fig.savefig(plot_filename, dpi=150, bbox_inches="tight")
print(f"Plot saved to: {plot_filename}")
