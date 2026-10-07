#!/usr/bin/env python
#+++ Imports
import os
from pathlib import Path
import matplotlib.pyplot as plt
from src.aux00_utils import EXTRA_FIGURES, load_dataset_and_grid
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
EXTRA_FIGURES.mkdir(exist_ok=True)
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
# Three rows sharing a time axis: the sinks themselves, the ratio they form at each scale, and the ratio of
# each sink between scales. Colour carries the filter scale in the first two and line style the quantity,
# so they index the same way; the third compares scales, so there colour is dropped and style alone carries
# the quantity.
print("Plotting...")
n_scales = len(args.filter_scales)
cross_scale = n_scales == 2
if not cross_scale:
    print(f"  note: the cross-scale ratio panel needs exactly two scales; {n_scales} given, so it is omitted")

nrows = 3 if cross_scale else 2
fig, axes = plt.subplots(nrows, 1, figsize=(7.5, 3.6 * nrows), constrained_layout=True, sharex=True)
ax_e, ax = axes[0], axes[1]

series = {}
for ℓ, color in zip(args.filter_scales, ["C0", "C3", "C2", "C1"]):
    actual = float(ke_budget.filter_scale.sel(filter_scale=ℓ, method="nearest"))
    ε_K = sink(ke_budget,  "∫-ε_Kˢ dV", ℓ)
    ε_A = sink(ape_budget, "∫-ε_Aˢ dV", ℓ)
    ε_A = ε_A.reindex(time=ε_K.time)            # the two budgets drop the same records, but do not assume it
    series[actual] = (ε_K, ε_A)

    ax_e.plot(ε_K.time, ε_K.values, color=color, lw=1.8, ls="-",
              label=f"$\\varepsilon_K^s$, $\\ell = {actual:g}$")
    ax_e.plot(ε_A.time, ε_A.values, color=color, lw=1.8, ls="--",
              label=f"$\\varepsilon_A^s$, $\\ell = {actual:g}$")

    # t = 0 alone is dropped: both sinks are identically zero there, so the ratio is 0/0. Everywhere else
    # the curve is drawn as it comes, including where ε_Aˢ -- which is not sign-definite, unlike ε_Kˢ --
    # drives the denominator through zero and the ratio with it.
    η = (ε_A / (ε_A + ε_K)).where(ε_A.time > 0)

    ax.plot(η.time, η.values, color=color, lw=1.8, label=f"$\\ell = {actual:g}$")
    print(f"  ℓ = {actual:g}: ∫ε_Kˢ dV peaks at {float(ε_K.max()):.3g}, ∫ε_Aˢ dV at {float(ε_A.max()):.3g}; "
          f"η over t > 0 spans [{float(η.min()):.3g}, {float(η.max()):.3g}]")

ax_e.axhline(0.0, color="k", lw=0.8, ls=":")
ax_e.set_ylabel(r"$\int \varepsilon^s \, dV$", fontsize=13)
ax_e.set_title("Sub-filter dissipation and mixing efficiency", fontsize=13)
ax_e.grid(True, alpha=0.3, lw=0.5)
ax_e.legend(fontsize=10, frameon=False, ncol=2)

ax.axhline(0.0, color="k", lw=0.8, ls="--")
ax.set_ylabel(r"$\varepsilon_A^s / (\varepsilon_A^s + \varepsilon_K^s)$", fontsize=13)
ax.grid(True, alpha=0.3, lw=0.5)
ax.legend(fontsize=12, frameon=False)

#+++ Cross-scale ratio
# Each sink at the first scale over the same sink at the second, in the order --filter-scales gives them,
# so the default 7 1 reads as "how much more is dissipated at the coarse scale than the fine one". t = 0 is
# dropped as above, both sinks being identically zero there and the ratio 0/0.
if cross_scale:
    ax_r = axes[2]
    ℓ_num, ℓ_den = (float(ke_budget.filter_scale.sel(filter_scale=ℓ, method="nearest")) for ℓ in args.filter_scales)
    for (idx, name, style) in ((0, r"\varepsilon_K^s", "-"), (1, r"\varepsilon_A^s", "--")):
        r = (series[ℓ_num][idx] / series[ℓ_den][idx]).where(series[ℓ_num][idx].time > 0)
        ax_r.plot(r.time, r.values, color="k", lw=1.8, ls=style, label=f"${name}$")
        print(f"  {name.replace(chr(92),''):>16}: ℓ={ℓ_num:g}/ℓ={ℓ_den:g} spans [{float(r.min()):.3g}, {float(r.max()):.3g}]")
    ax_r.axhline(1.0, color="k", lw=1.0, ls="--")   # equal dissipation at the two scales
    ax_r.set_ylabel(f"$\\ell = {ℓ_num:g}\\ /\\ \\ell = {ℓ_den:g}$", fontsize=13)
    ax_r.grid(True, alpha=0.3, lw=0.5)
    ax_r.legend(fontsize=12, frameon=False)
#---

axes[-1].set_xlim(0, args.max_time if args.max_time is not None else float(ke_budget.time.max()))
axes[-1].set_xlabel("Time", fontsize=13)

label = run_label(ke_budget.attrs)
if label:
    ax.text(0.98, 0.04, label, transform=ax.transAxes, fontsize=10, ha="right", va="bottom",
            bbox=dict(facecolor="white", edgecolor="none", pad=2, alpha=0.85))

outfile = str(EXTRA_FIGURES / f"{stem}_mixing_efficiency.pdf")
fig.savefig(outfile, dpi=200, bbox_inches="tight")
print(f"Figure saved to: {outfile}")
#---
