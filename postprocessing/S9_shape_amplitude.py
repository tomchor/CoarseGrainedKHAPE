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
parser = argparse.ArgumentParser(description="Cross-scale transfer split into shape (normalised Hovmoller) and amplitude (side trace)")
parser.add_argument("--filename", default="output/khi_Nz2048_Ri0.10.nc")
parser.add_argument("--fixed-reference", action="store_true", default=False)
parser.add_argument("--extension", default="edge")
parser.add_argument("--max-time", type=float, default=140.0)
parser.add_argument("--fig-height", type=float, default=6.8)
parser.add_argument("--min-amplitude", type=float, default=0.02,
                    help="Blank times whose peak transfer is below this fraction of the record's peak. Dividing "
                         "by a near-zero peak turns round-off into saturated colour, so the pre-instability rows "
                         "would otherwise read as strong structure.")
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
inv = 1.0 / et.filter_scale.values
t = et.time.values
fields = {r"$\Pi_K$": et["∫Π_K dV"].transpose("time", "filter_scale").values,
          r"$\Pi_A$": et["∫Π_A dV"].transpose("time", "filter_scale").values}
print(f"  Loaded: {input_filename}\n  {len(t)} times, {len(inv)} scales")
#---

#+++ Plot
# The transfer is amplitude x shape, and the two vary over very different ranges: the amplitude grows by
# ~10^3 as the instability develops, while the shape -- where in scale the transfer is positive or negative
# -- is the physics. A single colour scale has to span both and ends up showing mostly the growth. So each
# time row is divided by its own peak, leaving the colour to carry shape on a plain linear [-1, 1] scale,
# and the amplitude it was divided by is drawn alongside on a log axis. Nothing is discarded.
fig, axes = plt.subplots(2, 2, figsize=(7.8, args.fig_height), constrained_layout=True,
                         sharey=True, gridspec_kw=dict(width_ratios=[3.2, 1.0]))

peak = max(np.abs(M).max() for M in fields.values())
for row, (name, M) in enumerate(fields.items()):
    amp = np.abs(M).max(axis=1)
    shape = M / np.maximum(amp, 1e-300)[:, None]
    shape[amp < args.min_amplitude * peak] = np.nan     # see --min-amplitude
    ax, ax_amp = axes[row]

    pcm = ax.pcolormesh(inv, t, shape, cmap="RdBu_r", vmin=-1, vmax=1, shading="auto", rasterized=True)
    ax.set_xscale("log"); ax.set_ylabel("Time"); ax.grid(True, alpha=0.2)
    ax.text(0.015, 0.97, name, transform=ax.transAxes, fontsize=12, ha="left", va="top",
            bbox=dict(facecolor="white", edgecolor="none", pad=2.5, alpha=0.85))
    # zero crossings of the shape, which is where the cascade reverses direction
    ax.contour(inv, t, shape, levels=[0], colors="k", linewidths=0.8, alpha=0.6)

    ax_amp.plot(amp, t, color="k", lw=1.5)
    ax_amp.axvline(args.min_amplitude * peak, color="k", lw=0.8, ls=":", alpha=0.6)
    ax_amp.set_xscale("log"); ax_amp.grid(True, alpha=0.3)
    # Shared limits: the two terms differ by orders of magnitude early on, and separate autoscaled axes
    # would hide that by drawing both as full-width curves.
    ax_amp.set_xlim(args.min_amplitude * peak / 30, peak * 2)
    ax_amp.set_xlabel(r"$\max_\ell |\Pi|$")

axes[1, 0].set_xlabel("Inverse of filter scale 1/ℓ")
ax_top = axes[0, 0].secondary_xaxis("top", functions=(lambda x: 1 / x, lambda x: 1 / x))
ax_top.set_xlabel("Filter scale ℓ")

cbar = fig.colorbar(pcm, ax=axes[:, 0].tolist(), orientation="vertical", location="left",
                    fraction=0.05, pad=0.02, ticks=[-1, -0.5, 0, 0.5, 1])
cbar.set_label(r"Transfer / $\max_\ell|\Pi|$ at that time")

label = run_label(et.attrs)
if label:
    axes[0, 1].set_title(label, fontsize=8)
for (r, c), letter in zip([(0, 0), (0, 1), (1, 0), (1, 1)], "abcd"):
    axes[r, c].text(0.03 if c else 0.12, 0.97, f"({letter})", transform=axes[r, c].transAxes,
                    fontsize=10, fontweight="bold", ha="left", va="top")
#---

plot_filename = str(FIGURES / os.path.basename(input_filename)
                    .replace("energy_transfer_sweep", "S9_shape_amplitude").replace(".nc", ".pdf"))
fig.savefig(plot_filename, dpi=150, bbox_inches="tight")
print(f"Plot saved to: {plot_filename}")
