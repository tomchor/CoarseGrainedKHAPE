#!/usr/bin/env python
#+++ Imports
import os
from pathlib import Path
import xarray as xr
import matplotlib.pyplot as plt
from src.aux00_utils import PP_OUTPUT
from src.aux03_plotting import budget_colors, run_label
#---

#+++ Configuration
import argparse
parser = argparse.ArgumentParser(description="Plot the integrated SFS KE and APE budgets written by 01_online_budgets.py")
parser.add_argument("--filename", default="output/khi_Nz1024_Ri0.10.nc",
                    help="Path to simulation NetCDF file (used to derive budget filenames)")
args = parser.parse_args()
print("\n" + "="*70 + f"\n  {Path(__file__).name}\n  " + "  ".join(f"{k}={v}" for k,v in vars(args).items()) + "\n" + "="*70)
REPO_ROOT = Path(__file__).resolve().parent.parent
filename = str(REPO_ROOT / args.filename) if not os.path.isabs(args.filename) else args.filename
#---

#+++ Load budget data
print("Loading budget data...")
ke_filename  = str(PP_OUTPUT / (Path(filename).stem + f"_sfs_ke_budget_integrated.nc"))
ape_filename = str(PP_OUTPUT / (Path(filename).stem + f"_sfs_ape_budget_integrated.nc"))
ke_budget  = xr.open_dataset(ke_filename,  decode_timedelta=False)
ape_budget = xr.open_dataset(ape_filename, decode_timedelta=False)
filter_scales = ke_budget.filter_scale.values
print(f"  KE  budget: {ke_filename}")
print(f"  APE budget: {ape_filename}")
print(f"  Filter scales: {filter_scales}")
#---

#+++ Define budget terms and colors
ke_vars = {
    "∫-∂ₜ SFS KE dV":    budget_colors["tendency"],
    "∫Π_K dV":           budget_colors["flux"],
    "∫-ε_Kˢ dV":         budget_colors["dissipation"],
    "∫(SFS APE->KE) dV": budget_colors["exchange"],
    "residual_K":         budget_colors["residual"],
}
ape_vars = {
    "∫-∂ₜ SFS APE dV":  budget_colors["tendency"],
    "∫Π_A dV":           budget_colors["flux"],
    "∫-ε_Aˢ dV":         budget_colors["dissipation"],
    "∫(SFS KE->APE) dV": budget_colors["exchange"],
    "∫Rˢ dV":            "C4",
    "residual_A":         budget_colors["residual"],
}
#---

#+++ Plot one figure per filter scale (KE on top, APE on bottom)
print("\nCreating plots...")
label = run_label(ke_budget.attrs)

for ℓ in filter_scales:
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), constrained_layout=True)

    for ax, budget, vars_dict, title in [
        (axes[0], ke_budget,  ke_vars,  "Integrated SFS KE Budget"),
        (axes[1], ape_budget, ape_vars, "Integrated SFS APE Budget"),
    ]:
        for var, color in vars_dict.items():
            data = budget[var].sel(filter_scale=ℓ).dropna("time")
            lbl = f"{var}  [max|·| = {float(abs(data).max()):.2e}]" if "residual" in var else var
            data.plot.line(ax=ax, x="time", label=lbl, color=color)
        ax.legend(fontsize=8)
        ax.set_ylabel("Budget Terms [W or J s⁻¹]")
        ax.set_title(f"{title}  (ℓ = {ℓ:.4f})")
        ax.grid(True, alpha=0.3)

    ymin = min(axes[0].get_ylim()[0], axes[1].get_ylim()[0])
    ymax = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])
    axes[0].set_ylim(ymin, ymax)
    axes[1].set_ylim(ymin, ymax)

    if label:
        fig.suptitle(label, fontsize=11)

    plot_filename = str(REPO_ROOT / "figures" / (Path(filename).stem + f"_sfs_budgets_l{ℓ:.4f}.png"))
    fig.savefig(plot_filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved to: {plot_filename}")
#---
