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
parser.add_argument("--filename", default="output/khi_90x1x256.nc",
                    help="Path to simulation NetCDF file (used to derive budget filenames)")
args = parser.parse_args()
REPO_ROOT = Path(__file__).resolve().parent.parent
PP_OUTPUT = REPO_ROOT / "postprocessing" / "output"
filename = str(REPO_ROOT / args.filename) if not os.path.isabs(args.filename) else args.filename
#---

#+++ Load budget data
print("Loading budget data...")
ke_filename  = str(PP_OUTPUT / (Path(filename).stem + "_sfs_ke_budget.nc"))
ape_filename = str(PP_OUTPUT / (Path(filename).stem + "_sfs_ape_budget.nc"))
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
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)

    for ax, budget, vars_dict, title in [
        (axes[0], ke_budget,  ke_vars,  "Integrated SFS KE Budget"),
        (axes[1], ape_budget, ape_vars, "Integrated SFS APE Budget"),
    ]:
        for var, color in vars_dict.items():
            budget[var].sel(filter_length_scale=ℓ).dropna("time").plot.line(
                ax=ax, x="time", label=var, color=color)
        ax.legend(fontsize=8)
        ax.set_ylabel("Budget Terms [W or J s⁻¹]")
        ax.set_title(f"{title}  (ℓ = {ℓ:.4f})")
        ax.grid(True, alpha=0.3)

    if label:
        fig.suptitle(label, fontsize=11)

    plot_filename = str(REPO_ROOT / "figures" / (Path(filename).stem + f"_sfs_budgets_l{ℓ:.4f}.png"))
    fig.savefig(plot_filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved to: {plot_filename}")
#---
