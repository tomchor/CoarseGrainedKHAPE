#!/usr/bin/env python
#+++ Imports
import os
from pathlib import Path
import xarray as xr
import matplotlib.pyplot as plt
from aux03_plotting import budget_colors, run_label
#---

#+++ Configuration
import argparse
parser = argparse.ArgumentParser(description="Plot SFS KE and APE budget terms from saved budget files")
parser.add_argument("--filename", default="output/khi_Nz1024_Ri0.10.nc",
                    help="Path to simulation NetCDF file (used to derive budget filenames)")
parser.add_argument("--fixed-reference", action="store_true", default=False,
                    help="Load the fixed-in-time reference profile outputs (produced by pipeline with --fixed-reference)")
args = parser.parse_args()
print("\n" + "="*70 + f"\n  {Path(__file__).name}\n  " + "  ".join(f"{k}={v}" for k,v in vars(args).items()) + "\n" + "="*70)
REPO_ROOT = Path(__file__).resolve().parent.parent
PP_OUTPUT = REPO_ROOT / "postprocessing" / "output"
filename = str(REPO_ROOT / args.filename) if not os.path.isabs(args.filename) else args.filename
fixed_reference = args.fixed_reference
ref_suffix = "_fixed_ref" if fixed_reference else ""
#---

#+++ Load budget data
print("Loading budget data...")
ke_filename  = str(PP_OUTPUT / (Path(filename).stem + f"_sfs_ke_budget_integrated{ref_suffix}.nc"))
ape_filename = str(PP_OUTPUT / (Path(filename).stem + f"_sfs_ape_budget_integrated{ref_suffix}.nc"))
ke_budget  = xr.open_dataset(ke_filename,  decode_timedelta=False)
ape_budget = xr.open_dataset(ape_filename, decode_timedelta=False)
filter_length_scales = ke_budget.filter_length_scale.values
print(f"  KE  budget: {ke_filename}")
print(f"  APE budget: {ape_filename}")
print(f"  Filter scales: {filter_length_scales}")
#---

#+++ Define budget terms and colors
ke_vars = {
    "∫-∂ₜ SFS KE dV":    budget_colors["tendency"],
    "∫Π_KE dV":          budget_colors["flux"],
    "∫-εₛ dV":           budget_colors["dissipation"],
    "∫(SFS APE->KE) dV": budget_colors["exchange"],
    "residual_KE":        budget_colors["residual"],
}
ape_vars = {
    "∫-∂ₜ SFS APE dV":  budget_colors["tendency"],
    "∫Π_APE dV":         budget_colors["flux"],
    "∫-χₛ dV":           budget_colors["dissipation"],
    "∫(SFS KE->APE) dV": budget_colors["exchange"],
    "∫Rˢ dV":            "C4",
    "residual_APE":       budget_colors["residual"],
}
#---

#+++ Plot one figure per filter scale (KE and APE side by side)
print("\nCreating plots...")
label = run_label(ke_budget.attrs)

for ℓ in filter_length_scales:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)

    for row in range(2):
        for ax, budget, vars_dict, title in [
            (axes[row, 0], ke_budget,  ke_vars,  "Integrated SFS KE Budget"),
            (axes[row, 1], ape_budget, ape_vars, "Integrated SFS APE Budget"),
        ]:
            for var, color in vars_dict.items():
                data = budget[var].sel(filter_length_scale=ℓ).dropna("time")
                lbl = f"{var}  [max|·| = {float(abs(data).max()):.2e}]" if "residual" in var else var
                data.plot.line(ax=ax, x="time", label=lbl, color=color)
            ax.legend(fontsize=8)
            ax.set_ylabel("Budget Terms [W or J s⁻¹]")
            ax.set_title(f"{title}  (ℓ = {ℓ:.4f})")
            ax.grid(True, alpha=0.3)

    # Share y-axis scale across both columns in the bottom row only
    ymin = min(axes[1, 0].get_ylim()[0], axes[1, 1].get_ylim()[0])
    ymax = max(axes[1, 0].get_ylim()[1], axes[1, 1].get_ylim()[1])
    axes[1, 0].set_ylim(ymin, ymax)
    axes[1, 1].set_ylim(ymin, ymax)

    if label and fixed_reference:
        fig.suptitle(f"{label} (fixed reference)", fontsize=11)
    elif label:
        fig.suptitle(label, fontsize=11)
    elif fixed_reference:
        fig.suptitle("(fixed reference)", fontsize=11)

    plot_filename = str(
        REPO_ROOT / "figures" / (Path(filename).stem + f"_sfs_budgets_l{ℓ:.4f}{ref_suffix}.png")
    )
    fig.savefig(plot_filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved to: {plot_filename}")
#---
