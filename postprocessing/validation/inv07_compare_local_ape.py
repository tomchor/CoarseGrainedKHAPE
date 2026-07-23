#!/usr/bin/env python
"""
Compare the online (Oceanostics, simulation-time) local available potential energy against the offline
(Python post-processing) one.

`AvailablePotentialEnergy` (Oceanostics PR #274) computes the Holliday & McIntyre (1981) local APE
density

    Eₐ(b, z) = ∫_{z✶}^{z} [b✶(z̃) - b] dz̃ ,   equivalently   (g/ρ₀) ∫_{z✶}^{z} [ρ - ρ✶(z̃)] dz̃ ,

the positive-definite work to bring a parcel from its reference height z✶ to where it sits. This is
exactly the `ape` field the offline pipeline builds in `local_potential_energies_timeseries`: with
b = g(ρ₀-ρ)/ρ₀ the buoyancy and density forms are identical, per unit mass (m² s⁻²), no ρ₀/sign
conversion. So this is a like-for-like check of the same quantity computed two ways.

The simulation must have been run with `--save_sorted`, which writes the online local field `E_a` and
its integral `∫E_a` (both built from the ThreeDimensionalSort z✶). This script recomputes the offline
`ape` from the sorted reference state and compares:

  * the local field  E_a(x, z)   at a snapshot time  — rms(online - offline) / rms(online),
  * the integral     ∫E_a dV      over the whole run — rms(online - offline) / rms(offline).

Two structural differences from `inv06`, both deliberate:

  * The offline sort is run only on the *unpadded* (true) domain, the like-for-like control for the
    online sort. The padding effect is already quantified by `inv06`; repeating it here would only
    restate it.
  * The online z✶ ranks cells (ThreeDimensionalSort) while the offline z₀ is a nearest-density lookup,
    so the two disagree over tied buoyancies exactly as in `inv06`. Eₐ is near zero there for both (a
    parcel in uniform fluid sits at its own reference height), so the difference concentrates in the
    stratified, actively mixing interior — which is where the APE actually lives.

With `--tolerance` the script exits nonzero if either relative difference exceeds it (see
`aux_check.py`); without it, it only reports. `tests/test_online_vs_offline.py` runs it in CI.
"""
#+++ Imports
import logging
import os
import sys
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # postprocessing/ on path for `src.*`
from aux_check import add_tolerance_arg, set_tolerance, check, finalize
from src.aux00_utils import load_dataset_and_grid, integrate, open_grid_group, model_grid_suffix, strip_grid_suffix
from src.aux01_pe_functions import (calculate_density_fields_from_buoyancy, sorted_timeseries,
                                    local_potential_energies_timeseries)
from src.aux03_plotting import run_label
#---

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
print = logging.info

#+++ Configuration
import argparse
parser = argparse.ArgumentParser(description="Compare online vs offline local available potential energy Eₐ")
parser.add_argument("--filename", default="output/khi_Nz256_Ri0.10.nc", help="Path to simulation NetCDF file (run with --save_sorted)")
parser.add_argument("--time", type=float, default=None, help="Target time for the snapshot maps (default: midpoint of simulation)")
parser.add_argument("--z-window", type=float, default=None, help="Half-height of the z window shown in the snapshot maps (default: the full domain)")
parser.add_argument("--n-workers", type=int, default=1, help="Thread-pool workers for the offline sort and APE")
add_tolerance_arg(parser)
args = parser.parse_args()
set_tolerance(args.tolerance)

print("\n" + "="*70 + f"\n  {Path(__file__).name}\n  " + "  ".join(f"{k}={v}" for k, v in vars(args).items()) + "\n" + "="*70)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # validation/ → postprocessing/ → repo root
FIGURES = REPO_ROOT / "figures"
FIGURES.mkdir(exist_ok=True)
filename = str(REPO_ROOT / args.filename) if not os.path.isabs(args.filename) else args.filename
stem = Path(filename).stem
#---

#+++ Load the padded dataset and the online fields
print("Loading simulation data...")
ds = load_dataset_and_grid(filename)          # z-padded; also strips the model grid's _gridN suffix
ds = ds.chunk({"time": 1})

# The true (unpadded) domain, read from the grid group: load_dataset_and_grid overwrites the z_min/z_max
# attributes with the *padded* extent, so the slice has to come from the grid, as in inv06.
grid = open_grid_group(filename)   # handles the _gridN naming a --save_sorted run introduces
z_bot, z_top = float(grid.z.min()), float(grid.z.max())
in_domain = dict(z_aac=slice(z_bot, z_top))   # the original cell centers, dropping load_dataset_and_grid's padding

for v in ("E_a", "∫E_a"):
    if v not in ds:
        raise SystemExit(f"Online field '{v}' not in {filename} — rerun the simulation with --save_sorted "
                         f"(needs the Oceanostics local-APE support, PR #274).")

online_E_a = ds["E_a"].sel(**in_domain)             # local field, on the model grid
online_int = ds["∫E_a"].squeeze(drop=True)          # already volume-integrated online

if args.time is None:
    args.time = float(ds.time.values[len(ds.time) // 2])
t_sel = float(ds.time.sel(time=args.time, method="nearest").values)
print(f"Selected snapshot time = {t_sel:.3f}  (requested {args.time})")
#---

#+++ Recompute the offline local APE on the unpadded domain (the like-for-like control)
# Rebuild the grid variables sorted_timeseries needs from the raw file, without load_dataset_and_grid's
# padding. strip_grid_suffix undoes the multi-grid dimension renaming a --save_sorted run introduces.
print("Building the unpadded dataset and recomputing the offline local APE...")
ds_raw = xr.open_dataset(filename, decode_times=False, chunks={}).chunk({"time": 1})
ds_raw = strip_grid_suffix(ds_raw, model_grid_suffix(ds_raw))
ds_raw["dV"]   = ds_raw.Δx_caa * ds_raw.Δy_aca * ds_raw.Δz_aac
# Domain extent as a 0-d scalar (last − first): `float(np.diff(grid.x))` would call float() on a
# 1-element *1-d* array, which newer numpy rejects ("only 0-dimensional arrays can be converted…").
ds_raw["LxLy"] = float(grid.x[-1] - grid.x[0]) * float(grid.y[-1] - grid.y[0])
ds_raw.attrs["z_min"] = z_bot
ds_raw.attrs["z_max"] = z_top

ds_rho = ds_raw[["b", "dV", "LxLy"]].copy()
ds_rho.attrs.update(ds_raw.attrs)
ds_rho = calculate_density_fields_from_buoyancy(ds_rho, buoyancy_name="b", density_name="ρ")
sorted_state = sorted_timeseries(ds_rho, field_to_sort="ρ", n_workers=args.n_workers, verbose_level=0)
offline = local_potential_energies_timeseries(ds_rho, sorted_state.rho_sorted, sorted_state.dz_sorted,
                                              density_name="ρ", n_workers=args.n_workers, verbose_level=0)
offline_ape = offline["ape"]                        # (time, x, y, z), same definition as online E_a
dV = ds_raw["dV"]
#---

#+++ Metrics
print("\n" + "="*70)
print("  Local APE field  Eₐ(x, z)   [rms(online - offline) / rms(online), at the snapshot time]")
print("="*70)
on_snap  = online_E_a.sel(time=t_sel, method="nearest").squeeze().compute()
off_snap = offline_ape.sel(time=t_sel, method="nearest").squeeze().compute()
diff = on_snap - off_snap
rms_online = float(np.sqrt(np.nanmean(on_snap.values**2)))
rms_field  = float(np.sqrt(np.nanmean(diff.values**2))) / rms_online if rms_online > 0 else float("inf")
check(rms_field, f"    Eₐ field: rms(diff)/rms(online) = {rms_field:.3e},  "
                 f"max|diff| = {float(np.nanmax(np.abs(diff.values))):.3e},  rms(online) = {rms_online:.3e}", print)

print("\n" + "="*70)
print("  Integrated APE  ∫Eₐ dV   [rms(online - offline) / rms(offline), over the run]")
print("="*70)
# The online integral samples the reference at cell centers; the offline one integrates the same local
# field with the same dV, so this is a direct check of the volume integral rather than of the continuum
# split (which the docstring notes differs from TPE - BPE at second order in Δz).
offline_int = integrate(offline_ape, dV).squeeze().compute()
online_int_c = online_int.reindex(time=offline_int.time, method="nearest").compute()
denom = float(np.sqrt(np.nanmean(offline_int.values**2)))
rms_int = float(np.sqrt(np.nanmean((online_int_c.values - offline_int.values)**2))) / denom if denom > 0 else float("inf")
check(rms_int, f"    ∫Eₐ dV: rms(online - offline)/rms(offline) = {rms_int:.3e}", print)
#---

#+++ Figures
label = run_label(ds.attrs)
zw = args.z_window

# Field maps: online | offline | difference
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
vmax = max(float(np.nanpercentile(np.abs(on_snap.values), 99)), float(np.nanpercentile(np.abs(off_snap.values), 99)))
vmax = vmax if vmax > 0 else 1.0
kw = dict(x="x_caa", y="z_aac", add_colorbar=True, cmap="magma_r", vmin=0, vmax=vmax)
on_snap.plot(ax=axes[0], **kw);  axes[0].set_title(f"Online Eₐ")
off_snap.plot(ax=axes[1], **kw); axes[1].set_title(f"Offline Eₐ")
diff.plot(ax=axes[2], x="x_caa", y="z_aac", add_colorbar=True, cmap="RdBu_r", robust=True)
axes[2].set_title("Difference (online − offline)")
for a in axes:
    if zw is not None:
        a.set_ylim(-zw, zw)
    a.set_aspect("equal")
fig.suptitle(f"Online vs offline local APE Eₐ   t = {t_sel:.1f}" + (f"   {label}" if label else ""))
out1 = FIGURES / f"inv07_local_ape_maps_{stem}_t{t_sel:.1f}.png"
fig.savefig(out1, dpi=150, bbox_inches="tight")
print(f"\nSaved {out1}")

# Integral time series
fig2, ax = plt.subplots(figsize=(7, 4.2), constrained_layout=True)
ax.plot(offline_int.time, offline_int, lw=2.5, label="offline")
ax.plot(online_int_c.time, online_int_c, "--", lw=1.6, label="online")
ax.set(xlabel="time", ylabel="∫Eₐ dV", title="Volume-integrated local APE")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
fig2.suptitle(f"Online vs offline ∫Eₐ dV" + (f"   {label}" if label else ""))
out2 = FIGURES / f"inv07_local_ape_integral_{stem}.png"
fig2.savefig(out2, dpi=150, bbox_inches="tight")
print(f"Saved {out2}")
#---

finalize(print)
