#!/usr/bin/env python
#+++ Imports
import os
from pathlib import Path
import matplotlib.pyplot as plt
from src.aux00_utils import load_dataset_and_grid
from src.aux03_plotting import run_label
from src.aux04_online_budgets import online_budgets
#---

#+++ Configuration
import argparse
parser = argparse.ArgumentParser(description="Mixing efficiency η = ε_Aˢ/(ε_Aˢ + ε_Kˢ) against time, one line per filter scale")
parser.add_argument("--filename", default="output/khi_Nz1024_Ri0.10.nc",
                    help="Path to simulation NetCDF file (the dissipations are assembled from its online terms)")
parser.add_argument("--filter-scales", type=float, nargs="+", default=[7, 1], help="Filter length scales, one line each")
parser.add_argument("--max-time", type=float, default=None,
                    help="Right edge of the time axis; default is the end of the record set")
args = parser.parse_args()

print("\n" + "="*70 + f"\n  {Path(__file__).name}\n  " + "  ".join(f"{k}={v}" for k, v in vars(args).items()) + "\n" + "="*70)
REPO_ROOT = Path(__file__).resolve().parent.parent
FIGURES   = REPO_ROOT / "figures"
FIGURES.mkdir(exist_ok=True)
filename = str(REPO_ROOT / args.filename) if not os.path.isabs(args.filename) else args.filename
stem = Path(filename).stem
#---

#+++ Assemble the budgets and pull out the two sinks
# Same source as plot5: `online_budgets` is what 01_online_budgets.py writes, so this runs straight off the
# simulation output with no post-processing. The budgets carry the dissipations with the sign they enter the
# budget with -- as sinks -- so they are negated back to the dissipation rates themselves.
print("Assembling the SFS budgets from the simulation output...")
ds = load_dataset_and_grid(filename, pad=False).chunk({"time": 1})
ke_budget, ape_budget = online_budgets(ds)

def sink(budget, var, ℓ):
    """The dissipation at scale ℓ, as a positive rate, over the records where the budget is defined."""
    return -budget[var].sel(filter_scale=ℓ, method="nearest").dropna("time").isel(time=slice(1, None))
#---

#+++ Plot
print("Plotting...")
fig, ax = plt.subplots(figsize=(7.5, 4.6), constrained_layout=True)

for ℓ, color in zip(args.filter_scales, ["C0", "C3", "C2", "C1"]):
    actual = float(ke_budget.filter_scale.sel(filter_scale=ℓ, method="nearest"))
    ε_K = sink(ke_budget,  "∫-ε_Kˢ dV", ℓ)
    ε_A = sink(ape_budget, "∫-ε_Aˢ dV", ℓ)
    ε_A = ε_A.reindex(time=ε_K.time)            # the two budgets drop the same records, but do not assume it
    # t = 0 alone is dropped: both sinks are identically zero there, so the ratio is 0/0. Everywhere else
    # the curve is drawn as it comes, including where ε_Aˢ -- which is not sign-definite, unlike ε_Kˢ --
    # drives the denominator through zero and the ratio with it.
    η = (ε_A / (ε_A + ε_K)).where(ε_A.time > 0)

    ax.plot(η.time, η.values, color=color, lw=1.8, label=f"$\\ell = {actual:g}$")
    print(f"  ℓ = {actual:g}: η over t > 0 spans [{float(η.min()):.3g}, {float(η.max()):.3g}]")

ax.axhline(0.0, color="k", lw=0.8, ls="--")
ax.set_xlim(0, args.max_time if args.max_time is not None else float(ke_budget.time.max()))
ax.set_xlabel("Time", fontsize=13)
ax.set_ylabel(r"$\varepsilon_A^s / (\varepsilon_A^s + \varepsilon_K^s)$", fontsize=13)
ax.set_title("Sub-filter mixing efficiency", fontsize=13)
ax.grid(True, alpha=0.3, lw=0.5)
ax.legend(fontsize=12, frameon=False)

label = run_label(ke_budget.attrs)
if label:
    ax.text(0.98, 0.04, label, transform=ax.transAxes, fontsize=10, ha="right", va="bottom",
            bbox=dict(facecolor="white", edgecolor="none", pad=2, alpha=0.85))

outfile = str(FIGURES / f"{stem}_mixing_efficiency.pdf")
fig.savefig(outfile, dpi=200, bbox_inches="tight")
print(f"Figure saved to: {outfile}")
#---
