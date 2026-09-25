#!/usr/bin/env python
"""Test whether the cross-scale transfer evolves self-similarly across filter scales.

The transfer switches on progressively later at smaller scales, so a spectrum averaged over any fixed
time window mixes scales that are at different points in their own evolution -- which is why phase
averaging turned out to be sensitive to where the boundaries were placed. If instead each scale has a
characteristic onset t*(l) and duration tau(l), and the evolution is otherwise the same shape, then

    Pi(l, t) / max_t|Pi(l, .)|   against   s = (t - t*(l)) / tau(l)

collapses onto one curve, and the scale dependence is fully described by those two functions. If it does
not collapse, the evolution is not self-similar and no reduction over time is faithful -- the Hovmoller
is then the honest presentation.
"""
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
parser = argparse.ArgumentParser(description="Self-similar collapse test for the cross-scale transfer spectra")
parser.add_argument("--filename", default="output/khi_Nz2048_Ri0.10.nc")
parser.add_argument("--fixed-reference", action="store_true", default=False)
parser.add_argument("--extension", default="edge")
parser.add_argument("--max-time", type=float, default=140.0)
parser.add_argument("--threshold", type=float, default=0.2,
                    help="Fraction of a scale's own peak |Pi| defining its onset and duration")
parser.add_argument("--ell-range", type=float, nargs=2, default=[0.03, 3.5],
                    help="Filter scales to include. The largest scales do not follow the downscale front -- "
                         "they carry the roll-up and breakdown of the billow itself -- so they are excluded "
                         "by default. Pass a wider range to see them fail.")
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
et = collapse_time_pairs(xr.open_dataset(input_filename, decode_timedelta=False))
et = et.sel(time=slice(None, args.max_time)).sortby("filter_scale")
sel = (et.filter_scale >= args.ell_range[0]) & (et.filter_scale <= args.ell_range[1])
et = et.isel(filter_scale=np.where(sel.values)[0])
t = et.time.values
L = et.filter_scale.values
print(f"  Loaded: {input_filename}\n  {len(t)} times, {len(L)} scales in l = [{L.min():.3g}, {L.max():.3g}]")
#---

#+++ Per-scale onset and duration
def onset_width(series, t, frac):
    """(t*, tau, peak) from the first and last crossing of `frac` x the series' own peak."""
    a = np.abs(series)
    pk = a.max()
    if pk <= 0 or not np.isfinite(pk):
        return np.nan, np.nan, np.nan
    over = a >= frac * pk
    if not over.any():
        return np.nan, np.nan, pk
    i0, i1 = np.argmax(over), len(over) - 1 - np.argmax(over[::-1])
    # linear interpolation of the first crossing; the last is often the record end, so tau is a lower bound
    if i0 > 0:
        y0, y1 = a[i0-1], a[i0]
        t0 = t[i0-1] + (frac*pk - y0)/(y1 - y0)*(t[i0] - t[i0-1]) if y1 != y0 else t[i0]
    else:
        t0 = t[i0]
    return t0, max(t[i1] - t0, np.diff(t).mean()), pk

terms = {r"$\Pi_K$": et["∫Π_K dV"].transpose("filter_scale", "time").values,
         r"$\Pi_A$": et["∫Π_A dV"].transpose("filter_scale", "time").values}

stats, curves = {}, {}
for name, M in terms.items():
    ts, tau, pk = np.array([onset_width(M[i], t, args.threshold) for i in range(len(L))]).T
    stats[name] = (ts, tau, pk)
    curves[name] = M / np.where(pk[:, None] > 0, pk[:, None], 1.0)
#---

#+++ Collapse quality
# Interpolate every scale onto a common rescaled time and measure how far apart the curves stay. The
# curves are normalised to unit peak, so the spread is directly interpretable: ~0 is a collapse, ~0.5 is
# no relationship. Reported over the bulk of the event rather than the tails, where all curves are ~0.
S_GRID = np.linspace(-0.5, 2.0, 120)
quality = {}
for name, M in terms.items():
    ts, tau, pk = stats[name]
    rows = []
    for i in range(len(L)):
        if not np.isfinite(ts[i]) or not np.isfinite(tau[i]) or tau[i] <= 0:
            continue
        s = (t - ts[i]) / tau[i]
        rows.append(np.interp(S_GRID, s, curves[name][i], left=np.nan, right=np.nan))
    R = np.array(rows)
    n_valid = np.sum(np.isfinite(R), axis=0)
    with np.errstate(invalid="ignore"):
        spread = np.where(n_valid >= 2, np.nanstd(np.where(np.isfinite(R), R, np.nan), axis=0), np.nan)
    core = (S_GRID > -0.2) & (S_GRID < 1.2)
    quality[name] = (R, float(np.nanmedian(spread[core])))
    print(f"  {name}: {R.shape[0]} scales collapsed, median spread over the event = {quality[name][1]:.3f}"
          f"   (0 = perfect collapse, ~0.5 = none)")
#---

#+++ Plot
fig, axes = plt.subplots(3, 2, figsize=(10.5, 10.0), constrained_layout=True)
cmap = plt.get_cmap("viridis")
norm_l = (np.log10(L) - np.log10(L).min()) / max(float(np.ptp(np.log10(L))), 1e-12)

for col, (name, M) in enumerate(terms.items()):
    ts, tau, pk = stats[name]
    ax0, ax1, ax2 = axes[0, col], axes[1, col], axes[2, col]

    # (row 1) the two scaling functions
    ax0.plot(1/L, ts,  "o-", ms=3, color="#2166ac", label=r"onset $t^*(\ell)$")
    ax0.plot(1/L, tau, "s-", ms=3, color="#d6604d", label=r"duration $\tau(\ell)$")
    ok = np.isfinite(ts)
    if ok.sum() > 3:
        p = np.polyfit(np.log10(1/L[ok]), ts[ok], 1)
        ax0.plot(1/L[ok], np.polyval(p, np.log10(1/L[ok])), "k--", lw=1,
                 label=f"$t^*$ fit: {p[0]:+.0f}/decade")
    ax0.set_xscale("log"); ax0.set_xlabel(r"$1/\ell$"); ax0.set_ylabel("Time")
    ax0.set_title(f"{name}: scaling functions", fontsize=10)
    ax0.legend(fontsize=8); ax0.grid(True, alpha=0.3)

    # (row 2) before: normalised by peak only
    for i in range(len(L)):
        ax1.plot(t, curves[name][i], color=cmap(norm_l[i]), lw=1, alpha=0.8)
    ax1.set_xlabel("Time"); ax1.set_ylabel(r"$\Pi\,/\,\max_t|\Pi|$")
    ax1.set_title("before: amplitude normalised only", fontsize=10); ax1.grid(True, alpha=0.3)

    # (row 3) after: rescaled time
    R, q = quality[name]
    for i, row in enumerate(R):
        ax2.plot(S_GRID, row, color=cmap(norm_l[min(i, len(norm_l)-1)]), lw=1, alpha=0.8)
    ax2.plot(S_GRID, np.nanmean(R, axis=0), "k-", lw=2.2, label="mean")
    ax2.set_xlabel(r"$(t - t^*(\ell))\,/\,\tau(\ell)$"); ax2.set_ylabel(r"$\Pi\,/\,\max_t|\Pi|$")
    ax2.set_title(f"after: rescaled  (spread {q:.3f})", fontsize=10)
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(np.log10(L).min(), np.log10(L).max()))
cb = fig.colorbar(sm, ax=axes[:, :].ravel().tolist(), fraction=0.02, pad=0.01)
cb.set_label(r"$\log_{10}\ell$")
label = run_label(et.attrs)
if label:
    fig.suptitle(label, fontsize=10)
#---

plot_filename = str(FIGURES / os.path.basename(input_filename)
                    .replace("energy_transfer_sweep", "S11_selfsimilar").replace(".nc", ".pdf"))
fig.savefig(plot_filename, dpi=150, bbox_inches="tight")
print(f"Plot saved to: {plot_filename}")
