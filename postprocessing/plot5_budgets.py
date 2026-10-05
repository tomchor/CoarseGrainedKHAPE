#!/usr/bin/env python
#+++ Imports
import os
from pathlib import Path
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from src.aux00_utils import load_dataset_and_grid
from src.aux03_plotting import budget_colors, run_label
from src.aux04_online_budgets import online_budgets
#---

#+++ Configuration
import argparse
parser = argparse.ArgumentParser(description="Plot 2x2 panel of SFS KE and APE budgets at two filter scales")
parser.add_argument("--filename", default="output/khi_Nz2048_Ri0.10.nc",
                    help="Path to simulation NetCDF file (the budgets are assembled from its online terms)")
parser.add_argument("--filter-scales", type=float, nargs=2, default=[7, 1], help="Two filter length scales for left and right columns")
parser.add_argument("--max-time", type=float, default=None,
                    help="Right edge of the time axis; default is the end of the record set. The data always runs to "
                         "the last record with a closed TimeDerivative window, whatever this is set to")
parser.add_argument("--tendency-sign", choices=["negative", "positive"], default="positive",
                    help="Plot -∂ₜE (negative — sums to zero with other terms) or ∂ₜE (positive)")
args = parser.parse_args()

print("\n" + "="*70 + f"\n  {Path(__file__).name}\n  " + "  ".join(f"{k}={v}" for k,v in vars(args).items()) + "\n" + "="*70)
REPO_ROOT = Path(__file__).resolve().parent.parent
FIGURES   = REPO_ROOT / "figures"
FIGURES.mkdir(exist_ok=True)
filename = str(REPO_ROOT / args.filename) if not os.path.isabs(args.filename) else args.filename
stem = Path(filename).stem
#---

#+++ Assemble the budgets from the simulation's online terms
# Straight from the simulation output, so this runs as soon as the Julia job has finished: `online_budgets` is what
# 01_online_budgets.py writes to the budget files, and only the volume integrals are read here.
print("Assembling the SFS KE and APE budgets from the simulation output...")
ds = load_dataset_and_grid(filename, pad=False).chunk({"time": 1})
ke_budget, ape_budget = online_budgets(ds)
print(f"  Filter scales available: {ke_budget.filter_scale.values}")
#---

#+++ Define budget terms (shared colors across all panels)
positive_tendency = args.tendency_sign == "positive"
tendency_sign = -1 if positive_tendency else 1
ke_tendency_label  = r"$\partial_t E_K^s$"  if positive_tendency else r"$-\partial_t E_K^s$"
ape_tendency_label = r"$\partial_t E_A^s$"  if positive_tendency else r"$-\partial_t E_A^s$"

ke_terms = {
    ke_tendency_label:        ("∫-∂ₜ SFS KE dV",    budget_colors["tendency"]),
    r"$\Pi_K$":               ("∫Π_K dV",           budget_colors["flux"]),
    r"$-\varepsilon_K^s$":    ("∫-ε_Kˢ dV",         budget_colors["dissipation"]),
    r"$E_A^s \to E_K^s$":     ("∫(SFS APE->KE) dV", budget_colors["exchange"]),
}
ape_terms = {
    ape_tendency_label:      ("∫-∂ₜ SFS APE dV",    budget_colors["tendency"]),
    r"$\Pi_A$":              ("∫Π_A dV",            budget_colors["flux"]),
    r"$-\varepsilon_A^s$":   ("∫-ε_Aˢ dV",          budget_colors["dissipation"]),
    r"$E_K^s \to E_A^s$":    ("∫(SFS KE->APE) dV",  budget_colors["exchange"]),
    r"$R^s$":                ("∫Rˢ dV",             "C4"),
}
#---

#+++ Plot 2×2 figure
print("Creating 2×2 budget panel plot...")
fig, axes = plt.subplots(2, 2, figsize=(14, 7), constrained_layout=True)

budget_configs = [
    (0, ke_budget,  ke_terms,  "residual_K",  "SFS KE budget terms"),
    (1, ape_budget, ape_terms, "residual_A", "SFS APE budget terms"),
]

for row, budget, terms, residual_var, row_title in budget_configs:
    for col, ℓ in enumerate(args.filter_scales):
        ax = axes[row, col]
        for label, (var, color) in terms.items():
            data = budget[var].sel(filter_scale=ℓ, method="nearest").dropna("time").isel(time=slice(1, None))
            sign = tendency_sign if var.startswith("∫-∂ₜ") else 1
            ax.plot(data.time, sign * data.values, label=label, color=color, lw=1.5)
        residual = budget[residual_var].sel(filter_scale=ℓ, method="nearest").dropna("time").isel(time=slice(1, None))
        ax.plot(residual.time, residual.values, color="k", ls="--", lw=1.0, zorder=0)

        if col == 0:
            ax.set_ylabel(row_title, fontsize=13)
        else:
            ax.set_ylabel("")
        if row == 1:
            ax.set_xlabel("Time", fontsize=13)
        else:
            ax.set_xlabel("")
            ax.tick_params(labelbottom=False)
        # Tight to the run: 0 to the last output time, with no matplotlib margin either side. The data
        # itself starts one record in and ends at the last closed TimeDerivative window (t = 198 on a run
        # to 200), so the small gaps at the ends are real and say where the budget is defined.
        ax.set_xlim(0, args.max_time if args.max_time is not None else float(ke_budget.time.max()))
        ax.grid(True, alpha=0.3, lw=0.5)
        ax.set_title("")

for col, ℓ in enumerate(args.filter_scales):
    actual_ℓ = float(ke_budget.filter_scale.sel(filter_scale=ℓ, method="nearest"))
    axes[0, col].set_title(f"$\\ell = {actual_ℓ:.1f}$", fontsize=14)
#---

#+++ Share y-axis within each column
# Both legends sit in the right-hand column, so that column gets headroom above the data for them to
# occupy. Expanding the column rather than one panel keeps the within-column sharing the loop exists for.
LEGEND_HEADROOM = 0.33   # of the data span, enough for the five-entry APE legend
for col in range(2):
    ymin = min(axes[row, col].get_ylim()[0] for row in range(2))
    ymax = max(axes[row, col].get_ylim()[1] for row in range(2))
    if col == 1:
        ymax += LEGEND_HEADROOM * (ymax - ymin)
    for row in range(2):
        axes[row, col].set_ylim(ymin, ymax)
#---

#+++ Legend and labels
ke_handles, ke_labels = axes[0, 1].get_legend_handles_labels()
ape_handles, ape_labels = axes[1, 1].get_legend_handles_labels()
axes[0, 1].legend(ke_handles, ke_labels, fontsize=13, loc="upper right", frameon=True, fancybox=True, framealpha=0.1)
# Five entries over two columns, with one of them alone in the first column and the rest in the second.
# The tendency takes the solo slot: it is the term the other four sum to, so it reads as the left-hand
# side of the budget rather than as one more flux.
solo_idx    = ape_labels.index(ape_tendency_label)
solo_handle = ape_handles.pop(solo_idx)
solo_label  = ape_labels.pop(solo_idx)
blank = Line2D([], [], linestyle="None")
n_pad = len(ape_handles) - 1
ape_handles = [solo_handle] + [blank] * n_pad + ape_handles
ape_labels  = [solo_label]  + [""]    * n_pad + ape_labels
axes[1, 1].legend(ape_handles, ape_labels, fontsize=13, loc="upper right", frameon=True, fancybox=True, framealpha=0.1, ncol=2)

for ax, letter in zip(axes.flat, "abcd"):
    ax.text(0.02, 0.97, f"({letter})", transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top", ha="left",
            bbox=dict(facecolor="white", edgecolor="none", pad=1.5, alpha=0.8))

label = run_label(ke_budget.attrs)
info_parts = []
if label:
    info_parts.append(label)
if info_parts:
    axes[0, 0].text(0.98, 0.97, "  ".join(info_parts), transform=axes[0, 0].transAxes, fontsize=11, ha="right", va="top",
                    bbox=dict(facecolor="white", edgecolor="none", pad=2, alpha=0.8))
#---

#+++ Save
outfile = str(FIGURES / f"{stem}_sfs_budgets_2x2.pdf")
fig.savefig(outfile, dpi=200, bbox_inches="tight")
print(f"Figure saved to: {outfile}")
#---
