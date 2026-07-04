#!/usr/bin/env python
#+++ Imports
import logging
import os
import sys
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # postprocessing/ on path for `src.*`
from src.aux00_utils import load_dataset_and_grid, make_gaussian_filter, condense_uw_velocities
from src.aux02_ke_functions import calculate_strain_tensor, calculate_sfs_stress_tensor
from src.aux03_plotting import run_label
#---

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
print = logging.info

#+++ Configuration
import argparse
parser = argparse.ArgumentParser(description="Compare an online vs offline tensor (resolved strain rate S̄ⁱʲ or sub-filter stress τⁱʲ) used in the cross-scale KE transfer")
parser.add_argument("--filename", default="output/khi_Nz128_Ri0.10.nc", help="Path to simulation NetCDF file (must be a run with --save_tensors)")
parser.add_argument("--tensor", choices=["strain", "stress"], default="strain", help="Which tensor to compare")
parser.add_argument("--filter-scale", type=float, default=7, help="Filter ℓ (FWHM) of the online tensor output to compare against")
parser.add_argument("--time", type=float, default=None, help="Target time for the snapshot (default: midpoint of simulation)")
parser.add_argument("--z-window", type=float, default=6.0, help="Half-height of the z window (in units of h) shown in the maps")
args = parser.parse_args()

print("\n" + "="*70 + f"\n  {Path(__file__).name}\n  " + "  ".join(f"{k}={v}" for k, v in vars(args).items()) + "\n" + "="*70)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # validation/ → postprocessing/ → repo root
FIGURES = REPO_ROOT / "figures"
FIGURES.mkdir(exist_ok=True)
filename = str(REPO_ROOT / args.filename) if not os.path.isabs(args.filename) else args.filename
stem = Path(filename).stem
ℓ = args.filter_scale
ℓ_tag = int(ℓ) if ℓ == int(ℓ) else ℓ

# Tensor metadata: (online variable key, math label) for the three independent i,j ∈ {1,3} components.
COMPONENTS = {
    "strain": [("S11", "S̄₁₁ = ∂ū/∂x"), ("S33", "S̄₃₃ = ∂w̄/∂z"), ("S13", "S̄₁₃ = ½(∂ū/∂z + ∂w̄/∂x)")],
    "stress": [("tau11", "τ₁₁ = filter(u u) − ū ū"), ("tau33", "τ₃₃ = filter(w w) − w̄ w̄"), ("tau13", "τ₁₃ = filter(u w) − ū w̄")],
}[args.tensor]
TENSOR_SYMBOL = {"strain": "S̄ⁱʲ", "stress": "τⁱʲ"}[args.tensor]
#---

#+++ Load dataset and select snapshot
# load_dataset_and_grid pads z to 2× (edge values), which pads both the online tensor variables and
# the fields used to recompute the tensor offline, so they share a grid. We recover the original
# (unpadded) z extent from the grid group to drop the padding before comparing.
print("Loading simulation data...")
ds = load_dataset_and_grid(filename)
grid = xr.open_dataset(filename, group="underlying_grid_reconstruction_kwargs")
z0, z1 = float(grid.z.min()), float(grid.z.max())
in_domain = dict(z_aac=slice(z0, z1))

if args.time is None:
    args.time = float(ds.time.values[len(ds.time) // 2])
t_sel = float(ds.time.sel(time=args.time, method="nearest").values)
print(f"Selected snapshot time = {t_sel:.3f}  (requested {args.time})")
ds_t = ds.sel(time=t_sel, method="nearest")

online_key0 = f"{COMPONENTS[0][0]}_ℓ{ℓ_tag}"
if online_key0 not in ds_t:
    raise SystemExit(f"Online tensor field '{online_key0}' not in dataset — re-run the simulation with --save_tensors.")
#---

#+++ Recompute the tensor offline at this snapshot (mirrors 03_energy_transfer.py)
filtered_dimensions = ["x_caa", "z_aac"]
tensor_dimensions   = ("x_caa", "z_aac")

uᵢ = condense_uw_velocities(ds_t, indices=(1, 3))["uᵢ"].load()
gf = make_gaussian_filter(ℓ, ds_t)
ūᵢ = gf.apply(uᵢ, dims=filtered_dimensions)

print(f"Recomputing offline {args.tensor} tensor ({TENSOR_SYMBOL}) at ℓ = {ℓ_tag} ...")
if args.tensor == "strain":
    T = calculate_strain_tensor(ūᵢ, dimensions=tensor_dimensions)
else:
    T = calculate_sfs_stress_tensor(uᵢ, gf, filter_dims=filtered_dimensions, filtered_u_i=ūᵢ)

# (i, j) component selector for the offline tensor: S11/τ11 ↔ (1,1), S33/τ33 ↔ (3,3), S13/τ13 ↔ (1,3)
_ij = {"S11": (1, 1), "S33": (3, 3), "S13": (1, 3), "tau11": (1, 1), "tau33": (3, 3), "tau13": (1, 3)}
#---

#+++ One figure: rows = the three components, columns = online | offline | difference
label = run_label(ds.attrs)
zw = args.z_window
fig, axes = plt.subplots(len(COMPONENTS), 3, figsize=(15, 3.6 * len(COMPONENTS)), constrained_layout=True, squeeze=False)

print(f"\n{args.tensor.capitalize()} tensor comparison (bulk rms over the original domain):")
for row, (key, math_label) in enumerate(COMPONENTS):
    on  = ds_t[f"{key}_ℓ{ℓ_tag}"].sel(**in_domain).squeeze().compute()
    i, j = _ij[key]
    off = T.sel(i=i, j=j, drop=True).sel(**in_domain).squeeze().compute()
    diff = (on - off).compute()

    vmax = max(float(np.nanpercentile(np.abs(on.values), 99)), float(np.nanpercentile(np.abs(off.values), 99)))
    vmax = vmax if vmax > 0 else 1.0
    kw = dict(x="x_caa", y="z_aac", add_colorbar=True, cmap="RdBu_r", vmin=-vmax, vmax=vmax)

    on.plot(ax=axes[row, 0], **kw);  axes[row, 0].set_title(f"Online {math_label}")
    off.plot(ax=axes[row, 1], **kw); axes[row, 1].set_title("Offline")
    diff.plot(ax=axes[row, 2], x="x_caa", y="z_aac", add_colorbar=True, cmap="RdBu_r", robust=True)
    axes[row, 2].set_title("Difference (online − offline)")

    for k in range(3):
        axes[row, k].set_ylim(-zw, zw)
        axes[row, k].set_aspect("equal")
    axes[row, 0].set_ylabel(key, fontsize=13)

    rms_diff   = float(np.sqrt(np.nanmean(diff.values**2)))
    rms_online = float(np.sqrt(np.nanmean(on.values**2)))
    rel = rms_diff / rms_online if rms_online > 0 else float("inf")
    print(f"  {key:>5}: rms(diff)/rms(online) = {rel:.2e},  max|diff| = {float(np.nanmax(np.abs(diff.values))):.2e}")

suptitle = f"Online vs offline {args.tensor} tensor {TENSOR_SYMBOL}   ℓ = {ℓ_tag}   t = {t_sel:.1f}"
if label:
    suptitle += f"   {label}"
fig.suptitle(suptitle, fontsize=13, y=1.01)
outfile = str(FIGURES / f"{stem}_{args.tensor}_tensor_comparison_l{ℓ_tag}_t{t_sel:.1f}.png")
fig.savefig(outfile, dpi=150, bbox_inches="tight")
print(f"Figure saved to: {outfile}")
#---
