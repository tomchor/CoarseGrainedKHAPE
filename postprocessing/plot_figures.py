#+++ Imports
import os
from pathlib import Path
import xarray as xr
import matplotlib.pyplot as plt
from aux03_plotting import budget_colors, run_label
#---

#+++ Configuration
import argparse
parser = argparse.ArgumentParser(description="Plot local quantities from SFS budget output files")
parser.add_argument("--filename", default="output/khi_Nz256_Ri0.10.nc",
                    help="Path to base simulation NetCDF file (budget files are derived from this)")
parser.add_argument("--time-stride", type=int, default=5,
                    help="Stride for time selection in plots")
args = parser.parse_args()
REPO_ROOT = Path(__file__).resolve().parent.parent
filename = str(REPO_ROOT / args.filename) if not os.path.isabs(args.filename) else args.filename
time_stride = args.time_stride
#---

#+++ Load budget datasets
ke_budget_file = filename.replace(".nc", "_sfs_ke_budget.nc")
ape_budget_file = filename.replace(".nc", "_sfs_ape_budget.nc")

print(f"Loading {ke_budget_file}...")
ds_ke = xr.open_dataset(ke_budget_file, decode_times=False)
print(f"Loading {ape_budget_file}...")
ds_ape = xr.open_dataset(ape_budget_file, decode_times=False)
#---

#+++ Plot local KE budget terms
print("\nPlotting local KE budget terms...")

ke_local_vars = ["KE_of_sfs_flow", "∂ₜ SFS KE", "Π_KE", "εₛ", "SFS APE->KE exchange"]

label = run_label(ds_ke.attrs)
for var in ke_local_vars:
    ds_ke[var].isel(y_aca=0, time=slice(None, None, time_stride)).plot(
        x="x_caa", col="time", col_wrap=5)
    if label:
        plt.gcf().suptitle(label, fontsize=10, y=1.01)
    plot_filename = str(REPO_ROOT / "figures" / (os.path.basename(ke_budget_file).replace(".nc", f"_{var}.png").replace("/", "_")))
    plt.savefig(plot_filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {plot_filename}")
#---

#+++ Plot integrated KE and APE budgets
print("\nPlotting integrated KE and APE budgets...")

ke_integrated_vars = {
    "∫-∂ₜ SFS KE dV":   budget_colors["tendency"],
    "∫Π_KE dV":          budget_colors["flux"],
    "∫-εₛ dV":           budget_colors["dissipation"],
    "∫(SFS APE->KE) dV": budget_colors["exchange"],
    "residual_KE":        budget_colors["residual"],
}
ape_integrated_vars = {
    "∫-∂ₜ SFS APE dV":   budget_colors["tendency"],
    "∫Π_APE dV":          budget_colors["flux"],
    "∫-χₛ dV":            budget_colors["dissipation"],
    "∫(SFS KE->APE) dV":  budget_colors["exchange"],
    "∫Rˢ dV":             "C4",
    "residual_APE":        budget_colors["residual"],
}

fig, (ax_ke, ax_ape) = plt.subplots(2, 1, figsize=(10, 10), constrained_layout=True)

for var, color in ke_integrated_vars.items():
    ds_ke[var].dropna("time").plot.line(ax=ax_ke, x="time", label=var, color=color)
ax_ke.legend()
ax_ke.set_ylabel("Budget Terms [W or J s⁻¹]")
label = run_label(ds_ke.attrs)
subtitle = f"\n{label}" if label else ""
ax_ke.set_title(f"Integrated SFS KE Budget Terms{subtitle}")
ax_ke.grid(True, alpha=0.3)

for var, color in ape_integrated_vars.items():
    ds_ape[var].dropna("time").plot.line(ax=ax_ape, x="time", label=var, color=color)
ax_ape.legend()
ax_ape.set_ylabel("Budget Terms [W or J s⁻¹]")
ax_ape.set_title(f"Integrated SFS APE Budget Terms{subtitle}")
ax_ape.grid(True, alpha=0.3)

plot_filename = str(REPO_ROOT / "figures" / os.path.basename(filename).replace(".nc", "_sfs_budgets.png"))
fig.savefig(plot_filename, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved {plot_filename}")
#---

#+++ Plot local APE budget terms
print("\nPlotting local APE budget terms...")

ape_local_vars = ["Eaˢ(ρ, z)", "∂ₜ SFS APE", "Π_APE", "χₛ", "SFS KE->APE exchange", "Rˢ"]

label_ape = run_label(ds_ape.attrs)
for var in ape_local_vars:
    ds_ape[var].isel(y_aca=0, time=slice(None, None, time_stride)).plot(
        x="x_caa", col="time", col_wrap=5)
    if label_ape:
        plt.gcf().suptitle(label_ape, fontsize=10, y=1.01)
    plot_filename = str(REPO_ROOT / "figures" / (os.path.basename(ape_budget_file).replace(".nc", f"_{var}.png").replace("/", "_")))
    plt.savefig(plot_filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {plot_filename}")
#---
