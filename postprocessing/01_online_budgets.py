#!/usr/bin/env python
"""
Write the integrated SFS KE and APE budgets from the terms the simulation computes online.

Every term of both sub-filter budgets is written by `kelvin_helmholtz_instability.jl` at each of its
`--filter_ls` scales: Kˢ, ∂ₜKˢ, Π_K, ε_Kˢ, τ(w,b_r) for the KE budget and S̃, ∂ₜS̃, Π_A, ε_Aˢ, Rˢ (with the
same τ) for the APE one, each as a 3D field and as a volume integral. `online_budgets` (src/aux04_online_budgets.py)
assembles them under the variable names the offline pipeline used to write, so nothing downstream knows which
pipeline produced them, and this script writes the volume integrals and the residual to the two files the tests,
`02_plot_budgets.py` and `anim1_panels.py` consume:

    <stem>_sfs_ke_budget_integrated.nc    <stem>_sfs_ape_budget_integrated.nc

The 3D fields are not copied: they are in the simulation output already, and whoever needs them calls
`online_budgets` on it (the tests, `plot5_budgets.py`, `plot6_panels.py`, `X2_panels.py`, `X4_thumbnail.py`) or
on the x–z slices of the `_2d.nc` file (`anim1_panels.py`).

Only the records whose `TimeDerivative` spans a time step are kept: every record but the first (iteration 0,
where the derivative has had one evaluation and reads zero) or, when the simulation ran with `--offline_check`
and wrote consecutive-iteration pairs, the upper record of each pair, whose derivative is the single-step
difference across it, the same difference the offline pipeline forms, so the budget is stated at the same
instants. The offline pipeline itself lives in `offline/` and runs only as the CI cross-check
(`pytest --offline-check`).
"""
#+++ Imports
import os
from pathlib import Path
from src.aux00_utils import PP_OUTPUT, load_dataset_and_grid
from src.aux04_online_budgets import online_budgets, integrated_variables
#---

#+++ Configuration
import argparse
parser = argparse.ArgumentParser(description="Write the integrated SFS KE and APE budgets from the simulation's online terms")
parser.add_argument("--filename", default="output/khi_Nz1024_Ri0.10.nc", help="Path to simulation NetCDF file")
parser.add_argument("--filter-scales", type=float, nargs="+", default=None,
                    help="Filter scales ℓ to assemble; each must be among the simulation's online --filter_ls (default: all of them)")
parser.add_argument("--records", choices=["differenced", "all"], default="differenced",
                    help="Keep the records whose TimeDerivative spans a time step (default: the upper member of each pair of an "
                         "--offline_check run, else every record but the first), or every record")
args = parser.parse_args()
print("\n" + "="*70 + f"\n  {Path(__file__).name}\n  " + "  ".join(f"{k}={v}" for k, v in vars(args).items()) + "\n" + "="*70)
REPO_ROOT = Path(__file__).resolve().parent.parent
filename = str(REPO_ROOT / args.filename) if not os.path.isabs(args.filename) else args.filename
stem = Path(filename).stem
#---

#+++ Load the online output on its own grid and assemble both budgets
print("\n" + "="*60)
print("Loading the simulation output (unpadded)...")
ds = load_dataset_and_grid(filename, pad=False).chunk({"time": 1})
ke_budget, ape_budget = online_budgets(ds, filter_scales=args.filter_scales, records=args.records)
print(f"  {len(ke_budget.time)} records kept out of {len(ds.time)}")
#---

#+++ Write the two integrated budget files
def save(budget, kind):
    integrated_filename = str(PP_OUTPUT / f"{stem}_sfs_{kind}_budget_integrated.nc")
    print(f"  Saving {kind.upper()} integrated time series → {integrated_filename}")
    budget[integrated_variables(budget)].load().to_netcdf(integrated_filename)


print("\n" + "="*60)
print("Writing the SFS KE budget...")
save(ke_budget, "ke")
print("\n" + "="*60)
print("Writing the SFS APE budget...")
save(ape_budget, "ape")
print("\nDone!")
#---
