#!/usr/bin/env python
"""Compare a sweep computed with two different wall extensions, at one filter scale.

The manuscript (§2) leaves the extension of b and b_* past a wall free, requiring only that the extended
profile stay monotonic and that extending a fluid at rest add no APE, and names two admissible rules:
repeating the wall value (§4's choice) and odd reflection about it. Anything that differs between them is
an artifact of the choice rather than physics, so the size of that difference is the honest uncertainty on
the sweep at that scale -- which matters most at large ℓ, where the kernel reaches past the walls.
"""
import argparse
import os
from pathlib import Path
import numpy as np
import xarray as xr

parser = argparse.ArgumentParser(description="Compare sweep transfer terms between two wall extensions")
parser.add_argument("--filename", default="output/khi_Nz2048_Ri0.10.nc", help="Path to simulation NetCDF file")
parser.add_argument("--filter-scale", type=float, default=20.0, help="Filter scale to compare at")
parser.add_argument("--all", action="store_true", help="Compare at every scale the two runs share, not just one")
parser.add_argument("--fixed-reference", action="store_true", default=False)
parser.add_argument("--extension", default="odd", help="The non-default extension to compare against 'edge'")
args = parser.parse_args()

REPO_ROOT = Path(__file__).resolve().parent.parent
PP_OUTPUT = REPO_ROOT / "postprocessing" / "output"
stem = Path(args.filename).stem
ref_suffix = "_fixed_ref" if args.fixed_reference else ""

def load(suffix):
    path = PP_OUTPUT / f"{stem}_energy_transfer_sweep{ref_suffix}{suffix}.nc"
    if not path.exists():
        raise SystemExit(f"missing {path}\nRun extension_test.pbs first (and the production sweep for 'edge').")
    ds = xr.open_dataset(str(path), decode_timedelta=False)
    # Output made before the rule was restricted to the buoyancy also reflected u and w, which moves ∫Π_K
    # by ~30% on its own; comparing against it measures the velocity extension, not the buoyancy one.
    if suffix and ds.attrs.get("z_extension_vars") is None:
        raise SystemExit(f"{path.name} predates restricting the extension to b (no z_extension_vars "
                         f"attribute); rerun extension_test.pbs")
    return ds

a = load("")                      # edge: the production sweep
b = load(f"_{args.extension}")    # the alternative rule

if args.all:
    # Scales present in both runs. Matched by value rather than by index: a single-scale odd run and the
    # 30-point production sweep share only their common points.
    scales = [s for s in a.filter_scale.values
              if np.isclose(b.filter_scale.values, s, rtol=1e-6).any()]
    if not scales:
        raise SystemExit("the two runs share no filter scales")
else:
    scales = [float(a.filter_scale.sel(filter_scale=args.filter_scale, method="nearest").values)]

# The two runs must cover the same times, or the comparison is of time sampling rather than of extension.
t = np.intersect1d(a.time.values, b.time.values)
if t.size == 0:
    raise SystemExit("no times in common: was the odd run given the same --n-time-skip as the production sweep?")
if t.size < a.sizes["time"] or t.size < b.sizes["time"]:
    print(f"note: comparing on {t.size} shared times (edge has {a.sizes['time']}, "
          f"{args.extension} has {b.sizes['time']})")
a, b = a.sel(time=t), b.sel(time=t)

terms = [x for x in a.data_vars if x.startswith("∫") and x in b]

if args.all:
    print(f"\nedge vs {args.extension} extension, {t.size} shared times."
          f"  Each cell is |mean(odd) - mean(edge)| / |mean(edge)|.\n")
    print(f"{'l':>8}  " + "  ".join(f"{v:>20}" for v in terms))
    for s in scales:
        row = []
        for v in terms:
            A = np.asarray(a[v].sel(filter_scale=s, method="nearest").values, float).ravel()
            B = np.asarray(b[v].sel(filter_scale=s, method="nearest").values, float).ravel()
            mA, mB = A.mean(), B.mean()
            row.append(f"{abs(mB - mA) / abs(mA):>19.1%} " if abs(mA) > 0 else f"{'--':>20}")
        print(f"{s:8.3f}  " + "  ".join(row))
    print("\nThe scale at which these stop being small is the honest upper limit of the sweep.")
else:
    ℓ = scales[0]
    print(f"\nl = {ℓ:.4g},  {t.size} times,  edge vs {args.extension} extension\n")
    print(f"{'term':<24} {'edge (t-mean)':>14} {args.extension + ' (t-mean)':>14} {'rel. diff':>11} {'max|d|/rms':>11}")
    for v in terms:
        A = np.asarray(a[v].sel(filter_scale=ℓ, method="nearest").values, float).ravel()
        B = np.asarray(b[v].sel(filter_scale=ℓ, method="nearest").values, float).ravel()
        mA, mB = A.mean(), B.mean()
        rms = np.sqrt((A ** 2).mean())
        rel = abs(mB - mA) / abs(mA) if abs(mA) > 0 else np.inf
        mx = np.abs(B - A).max() / rms if rms > 0 else np.inf
        print(f"{v:<24} {mA:14.5e} {mB:14.5e} {rel:10.2%} {mx:11.2%}")
    print("\nLarge values mean the extension rule, not the flow, is setting the answer at this scale.")
