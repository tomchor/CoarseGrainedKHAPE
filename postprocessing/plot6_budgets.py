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

# Order here sets both the plot order and the legend order. The tendency leads, as the term the others sum
# to, and the dissipation closes: it is the only sink, so the row reads sources first and the sink last.
ke_terms = {
    ke_tendency_label:        ("∫-∂ₜ SFS KE dV",    budget_colors["tendency"]),
    r"$\Pi_K$":               ("∫Π_K dV",           budget_colors["flux"]),
    r"$E_A^s \to E_K^s$":     ("∫(SFS APE->KE) dV", budget_colors["exchange"]),
    r"$-\varepsilon_K^s$":    ("∫-ε_Kˢ dV",         budget_colors["dissipation"]),
}
ape_terms = {
    ape_tendency_label:      ("∫-∂ₜ SFS APE dV",    budget_colors["tendency"]),
    r"$\Pi_A$":              ("∫Π_A dV",            budget_colors["flux"]),
    r"$E_K^s \to E_A^s$":    ("∫(SFS KE->APE) dV",  budget_colors["exchange"]),
    r"$R^s$":                ("∫Rˢ dV",             "C4"),
    r"$-\varepsilon_A^s$":   ("∫-ε_Aˢ dV",          budget_colors["dissipation"]),
}
#---

#+++ Plot 2×2 figure
print("Creating 2×2 budget panel plot...")
fig, axes = plt.subplots(2, 2, figsize=(14, 7.6), constrained_layout=True)
# Room between the rows for the KE legend, which sits under row 0. At the default spacing it has
# nowhere to go and overlaps the tops of row 1's panels.
fig.get_layout_engine().set(hspace=0.16)

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
for col in range(2):
    ymin = min(axes[row, col].get_ylim()[0] for row in range(2))
    ymax = max(axes[row, col].get_ylim()[1] for row in range(2))
    for row in range(2):
        axes[row, col].set_ylim(ymin, ymax)
#---

#+++ Legend and labels
# One horizontal legend under each row, outside the axes. Inside the panels they collided with the data at
# Nz=1024 and could only be kept clear by adding headroom, which is whitespace spent on the legend rather
# than on the curves. Out here the axes keep their own limits.
#
# They are anchored to the left-hand axes of the row and pushed past its right edge, so each legend centres
# on the row rather than on its first panel. Row 1's sits lower, clearing the "Time" label beneath it.
ke_handles, ke_labels   = axes[0, 1].get_legend_handles_labels()
ape_handles, ape_labels = axes[1, 1].get_legend_handles_labels()

# The tendency leads: it is the term the others sum to, so the legend reads in the order the budget does.
solo_idx = ape_labels.index(ape_tendency_label)
ape_handles.insert(0, ape_handles.pop(solo_idx))
ape_labels.insert(0, ape_labels.pop(solo_idx))

# Centred on the row, which means centred between the left edge of its first panel and the right edge of
# its last -- not on the figure, whose left margin carries the y label, and not a guessed offset in axes
# coordinates, which depends on the inter-column gap. constrained_layout only resolves those positions at
# draw time, so the figure is drawn once and then frozen: adding legends afterwards would otherwise make
# the engine re-run on save and shift the axes out from under them.
fig.canvas.draw()
fig.set_layout_engine("none")

# `drop` is how far below the row's axes the legend's top sits, in figure fraction. Row 1's clears the
# "Time" label beneath it; row 0's has only the inter-row gap to sit in, and sits high in it so it reads as
# belonging to the row above rather than floating between the two.
for row, (handles, labels, drop) in enumerate([(ke_handles, ke_labels, 0.015),
                                               (ape_handles, ape_labels, 0.060)]):
    left, right = axes[row, 0].get_position(), axes[row, -1].get_position()
    fig.legend(handles, labels, fontsize=13, ncol=len(labels), frameon=False,
               loc="upper center", bbox_transform=fig.transFigure,
               bbox_to_anchor=((left.x0 + right.x1) / 2, left.y0 - drop))

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
