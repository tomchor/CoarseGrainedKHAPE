#!/usr/bin/env python
"""
Check that the SFS KE and APE budgets close using only the simulation's online terms.

Every term of both budgets is now written by the simulation, so closure can be checked without the
offline pipeline at all:

    KE:   residual_K = -∂ₜKˢ + Π_K - ε_Kˢ + τ(w, b_r)
    APE:  residual_A = -∂ₜEₐˢ - τ(w, b_r) + Π_A - ε_Aˢ + Rˢ

The metric is the one `tests/test_budgets.py` uses offline, rms(residual) / min over terms of
rms(term), so the two are directly comparable. With `--offline-stem` the offline budget files are read
too and both residuals are reported side by side, which is the comparison that matters: the online
budget is worth having only if it closes at least as well as the offline one.

Every term is evaluated at the output time except the tendencies, which Oceananigans' `TimeDerivative`
centres at tⁿ - Δt/2 for one model timestep Δt. That is why the tendencies are registered on
`IterationInterval(1)` rather than on the writer's schedule: over one step the offset is negligible
against the output interval, while differencing on the writer's schedule would centre them half an
output interval early and would not close.

The first output of a run is skipped: a `TimeDerivative` is zero until its operand has been evaluated
twice, so the tendency there is not yet a derivative.
"""
#+++ Imports
import logging
import os
import sys
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # postprocessing/ on path for `src.*`
from aux_check import add_tolerance_arg, set_tolerance, check, finalize
from src.aux00_utils import model_grid_suffix, strip_grid_suffix
from src.aux03_plotting import run_label
#---

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
print = logging.info

#+++ Configuration
import argparse
parser = argparse.ArgumentParser(description="Check that the online SFS KE and APE budgets close")
parser.add_argument("--filename", default="output/khi_Nz256_Ri0.10.nc", help="Simulation NetCDF file (run with --save_sorted)")
parser.add_argument("--filter-scales", type=float, nargs="+", default=[1, 7], help="Filter ℓ (FWHM) values matching the online filter_ℓs")
parser.add_argument("--skip", type=int, default=2, help="Leading outputs to drop (default 2: the first ConsecutiveIterations pair)")
parser.add_argument("--offline-stem", default=None, help="Stem of the offline budget files in postprocessing/output/, for a side-by-side residual")
add_tolerance_arg(parser)
args = parser.parse_args()
set_tolerance(args.tolerance)

print("\n" + "="*70 + f"\n  {Path(__file__).name}\n  " + "  ".join(f"{k}={v}" for k, v in vars(args).items()) + "\n" + "="*70)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FIGURES = REPO_ROOT / "figures" / "validation"
FIGURES.mkdir(parents=True, exist_ok=True)
PP_OUTPUT = REPO_ROOT / "postprocessing" / "output"
filename = str(REPO_ROOT / args.filename) if not os.path.isabs(args.filename) else args.filename
stem = Path(filename).stem


def tag(ℓ):
    return f"{int(ℓ)}" if float(ℓ) == int(ℓ) else f"{ℓ}"


def rms(a):
    return float(np.sqrt(np.nanmean(np.asarray(a, dtype=float) ** 2)))


def relative_residual(residual, terms):
    """rms(residual) / min over terms of rms(term), matching tests/test_budgets.py."""
    nonzero = [rms(t) for t in terms if rms(t) > 0]
    if not nonzero:
        return float("inf")
    return rms(residual) / min(nonzero)
#---

#+++ Load the online integrals
ds = xr.open_dataset(filename, decode_times=False)
ds = strip_grid_suffix(ds, model_grid_suffix(ds))

# The writer runs on ConsecutiveIterations, so outputs come in pairs. The first pair is dropped: a
# TimeDerivative is zero until its operand has been evaluated twice, so the derivative at the first
# output is not yet a derivative at all, and the one at the second spans the initialisation transient
# (∫Eₐˢ goes from zero to its working value within the first fraction of a time unit). Neither is a
# statement about the budget, and both would dominate an rms over the run.
ds = ds.isel(time=slice(args.skip, None))

BUDGETS = {
    "KE": dict(
        tendency = "dKs_dt_ℓ{}_int",
        terms = {"∫Π_K dV": ("Π_K_ℓ{}_int", +1),
                 "∫-ε_Kˢ dV": ("ε_Ks_ℓ{}_int", -1),
                 "∫τ(w,b_r) dV": ("wb_rs_ℓ{}_int", +1)},
    ),
    "APE": dict(
        tendency = "dEas_dt_ℓ{}_int",
        terms = {"∫-τ(w,b_r) dV": ("wb_rs_ℓ{}_int", -1),
                 "∫Π_A dV": ("Π_A_ℓ{}_int", +1),
                 "∫-ε_Aˢ dV": ("ε_As_ℓ{}_int", -1),
                 "∫Rˢ dV": ("R_s_ℓ{}_int", +1)},
    ),
}
#---

#+++ Residuals
label = run_label(ds.attrs)
results = {}

for name, spec in BUDGETS.items():
    for ℓ in args.filter_scales:
        t = tag(ℓ)
        missing = [v.format(t) for v in [spec["tendency"], *(n for n, _ in spec["terms"].values())] if v.format(t) not in ds]
        if missing:
            raise SystemExit(f"Online budget terms missing from {filename}: {missing}. Rerun the simulation "
                             f"with --save_sorted and a matching --filter_ls.")

        dt = ds[spec["tendency"].format(t)].squeeze(drop=True).values
        named = {k: sign * ds[v.format(t)].squeeze(drop=True).values for k, (v, sign) in spec["terms"].items()}
        residual = -dt + sum(named.values())

        all_terms = [-dt, *named.values()]
        rel = relative_residual(residual, all_terms)
        results[(name, ℓ)] = (dict([(f"∫-∂ₜ{'Kˢ' if name == 'KE' else 'Eₐˢ'} dV", -dt)], **named), residual, rel)

        print("\n" + "="*70)
        print(f"  {name} budget, online only  (ℓ = {ℓ:g})")
        print("="*70)
        print(f"  {'term':<24}  rms(term)")
        print(f"  {'-'*24}  {'-'*12}")
        for k, v in results[(name, ℓ)][0].items():
            print(f"  {k:<24}  {rms(v):.4e}")
        print(f"  {'residual':<24}  {rms(residual):.4e}")
        check(rel, f"  {'residual / min(terms)':<24}  {rel:.3%}", print)
#---

#+++ Side-by-side with the offline residual
if args.offline_stem:
    print("\n" + "="*70)
    print("  Online vs offline relative residual")
    print("="*70)
    OFFLINE = {"KE": ("sfs_ke_budget_integrated", "residual_K",
                      ["∫-∂ₜ SFS KE dV", "∫Π_K dV", "∫-ε_Kˢ dV", "∫(SFS APE->KE) dV"]),
               "APE": ("sfs_ape_budget_integrated", "residual_A",
                       ["∫-∂ₜ SFS APE dV", "∫Π_A dV", "∫-ε_Aˢ dV", "∫(SFS KE->APE) dV", "∫R_s dV"])}
    for name, (suffix, res_var, term_vars) in OFFLINE.items():
        path = PP_OUTPUT / f"{args.offline_stem}_{suffix}.nc"
        if not path.exists():
            print(f"  {name}: offline file not found ({path.name}), skipping")
            continue
        off = xr.open_dataset(path, decode_timedelta=False)
        for ℓ in args.filter_scales:
            o = off.sel(filter_scale=ℓ, method="nearest")
            have = [v for v in term_vars if v in o]
            off_rel = relative_residual(o[res_var].values, [o[v].values for v in have])
            on_rel = results[(name, ℓ)][2]
            verdict = "online <= offline" if on_rel <= off_rel * 1.0 else "ONLINE WORSE"
            print(f"  {name} ℓ={ℓ:g}:  online {on_rel:.3%}   offline {off_rel:.3%}   [{verdict}]")
#---

#+++ Figure
n = len(BUDGETS) * len(args.filter_scales)
fig, axes = plt.subplots(len(BUDGETS), len(args.filter_scales), figsize=(7 * len(args.filter_scales), 4.2 * len(BUDGETS)),
                         constrained_layout=True, squeeze=False)
for r, name in enumerate(BUDGETS):
    for c, ℓ in enumerate(args.filter_scales):
        terms, residual, rel = results[(name, ℓ)]
        ax = axes[r, c]
        t = ds.time.values
        for k, v in terms.items():
            ax.plot(t, v, lw=1.6, label=k)
        ax.plot(t, residual, "k--", lw=2, label="residual")
        ax.set(xlabel="time", ylabel=f"{name} budget", title=f"{name}, ℓ = {ℓ:g}   (residual/min = {rel:.2%})")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
fig.suptitle("Online SFS budgets" + (f"   {label}" if label else ""))
out = FIGURES / f"inv10_online_budget_closure_{stem}.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved {out}")
#---

finalize(print)
