#!/usr/bin/env python
"""
Compare the online (Oceanostics, simulation-time) buoyancy displacement potential Υ and total APE
dissipation rate ε_A against the offline (Python post-processing) ones.

The simulation computes, at each output (see `online_ape_dissipation.jl`),

    Υ   = z✶ - z            the buoyancy displacement potential, and
    ε_A = κ ∂ᵢb ∂ᵢΥ         the rate at which diffusion destroys available potential energy,

the sink of the local APE equation of Wenegrat, Chor & Barkan (2026) (their Eqs. 11 and 14, where it
appears as -ε_A). Offline the same pair is built in density form by `local_potential_energies_timeseries`
and `calculate_ape_dissipation`, so this is a like-for-like check of two implementations of the same
quantity — up to two differences that are conventions, not errors:

  * **Units.** The offline Υ is the paper's Eq. (7), Υ = g(z - z✶(ρ))/ρ₀, written for density; the
    online one is its buoyancy form Υ = z✶ - z. They differ by exactly -g/ρ₀, since b = g(ρ₀-ρ)/ρ₀
    makes ∇ρ = -(ρ₀/g)∇b. This script converts the offline Υ to the buoyancy form before comparing.
    Both sign flips cancel in the contraction, so ε_A is the *same number* in either form and is
    compared directly.

  * **Ties.** The online z✶ is Winters Eq. (11) (`HeavisideIntegral`), which puts every cell of a run
    of equal buoyancy at the mid-height of the layer that run fills; the offline z₀ is a
    nearest-density lookup, which lands on the run's *bottom* slot instead. Where a whole z level is
    exactly horizontally uniform it is one tied run of Nx·Ny cells spanning exactly one cell height,
    and the two Υ differ there by the constant Δz/2 - Δz/(2·Nx·Ny). At t = 0 that is every level, so
    the mean difference is exactly that offset; once the perturbation and advection have broken the
    symmetry it survives only in the thin bands against the walls, where the initial tanh has
    saturated to within roundoff. The mean difference is printed next to the predicted offset so it is
    clear which regime a given snapshot is in. Being a constant it drops out of ∇Υ, and so out of ε_A
    entirely — one reason to compare ε_A rather than infer it from Υ.

The remaining ε_A difference is discretization, and it is one-sided: the online form pairs the two
factors on the face where both differences live and interpolates the product to the cell center, while
the offline `calculate_gradient` takes centered derivatives at the center and multiplies those. On a
grid-scale-sharp buoyancy interface the centered derivative filters exactly the correlation this
product is made of, so the offline ∫ε_A runs low — by ~16% at Nz=192, concentrated in the late,
strongly mixing phase, and shrinking with resolution. The online kernel was checked to reproduce the
conservative form it intends to compute to machine precision.

As in `inv07`, the offline sort is run on the *unpadded* (true) domain, the like-for-like control for
the online sort; the padding effect is quantified once, in `inv06`.

With `--tolerance` the script exits nonzero if any relative difference exceeds it (see `aux_check.py`);
without it, it only reports. `tests/test_online_vs_offline.py` runs it in CI.
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
from src.aux01_pe_functions import (g, ρ0, calculate_density_fields_from_buoyancy, sorted_timeseries,
                                    local_potential_energies_timeseries, calculate_ape_dissipation)
from src.aux03_plotting import run_label
#---

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
print = logging.info

#+++ Configuration
import argparse
parser = argparse.ArgumentParser(description="Compare online vs offline Υ and APE dissipation ε_A")
parser.add_argument("--filename", default="output/khi_Nz256_Ri0.10.nc", help="Path to simulation NetCDF file (run with --save_sorted)")
parser.add_argument("--time", type=float, default=None, help="Target time for the snapshot maps (default: midpoint of simulation)")
parser.add_argument("--z-window", type=float, default=None, help="Half-height of the z window shown in the snapshot maps (default: the full domain)")
parser.add_argument("--n-workers", type=int, default=1, help="Thread-pool workers for the offline sort and APE")
add_tolerance_arg(parser)
args = parser.parse_args()
set_tolerance(args.tolerance)

print("\n" + "="*70 + f"\n  {Path(__file__).name}\n  " + "  ".join(f"{k}={v}" for k, v in vars(args).items()) + "\n" + "="*70)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # validation/ → postprocessing/ → repo root
FIGURES = REPO_ROOT / "figures" / "validation"
FIGURES.mkdir(parents=True, exist_ok=True)
filename = str(REPO_ROOT / args.filename) if not os.path.isabs(args.filename) else args.filename
stem = Path(filename).stem
#---

#+++ Load the padded dataset and the online fields
print("Loading simulation data...")
ds = load_dataset_and_grid(filename)          # z-padded; also strips the model grid's _gridN suffix
ds = ds.chunk({"time": 1})

# The true (unpadded) domain, read from the grid group: load_dataset_and_grid overwrites the z_min/z_max
# attributes with the *padded* extent, so the slice has to come from the grid, as in inv06/inv07.
grid = open_grid_group(filename)   # handles the _gridN naming a --save_sorted run introduces
z_bot, z_top = float(grid.z.min()), float(grid.z.max())
in_domain = dict(z_aac=slice(z_bot, z_top))   # the original cell centers, dropping load_dataset_and_grid's padding

for v in ("Υ", "ε_A", "∫ε_A"):
    if v not in ds:
        raise SystemExit(f"Online field '{v}' not in {filename} — rerun the simulation with --save_sorted "
                         f"(needs online_ape_dissipation.jl).")

online_upsilon = ds["Υ"].sel(**in_domain)         # buoyancy form, z✶ - z, on the model grid
online_eps_A   = ds["ε_A"].sel(**in_domain)
online_int     = ds["∫ε_A"].squeeze(drop=True)    # already volume-integrated online

if args.time is None:
    args.time = float(ds.time.values[len(ds.time) // 2])
t_sel = float(ds.time.sel(time=args.time, method="nearest").values)
print(f"Selected snapshot time = {t_sel:.3f}  (requested {args.time})")
#---

#+++ Recompute Υ and ε_A offline on the unpadded domain (the like-for-like control)
print("Building the unpadded dataset and recomputing the offline Υ and ε_A...")
ds_raw = xr.open_dataset(filename, decode_times=False, chunks={}).chunk({"time": 1})
ds_raw = strip_grid_suffix(ds_raw, model_grid_suffix(ds_raw))
ds_raw["dV"]   = ds_raw.Δx_caa * ds_raw.Δy_aca * ds_raw.Δz_aac
# Domain extent as a 0-d scalar (last − first): `float(np.diff(grid.x))` would call float() on a
# 1-element *1-d* array, which newer numpy rejects ("only 0-dimensional arrays can be converted…").
ds_raw["LxLy"] = float(grid.x[-1] - grid.x[0]) * float(grid.y[-1] - grid.y[0])
ds_raw.attrs["z_min"] = z_bot
ds_raw.attrs["z_max"] = z_top

κ = float(ds_raw.attrs["κ"])
print(f"  Diffusivity κ = {κ:.4e}")

ds_rho = ds_raw[["b", "dV", "LxLy"]].copy()
ds_rho.attrs.update(ds_raw.attrs)
ds_rho = calculate_density_fields_from_buoyancy(ds_rho, buoyancy_name="b", density_name="ρ")
sorted_state = sorted_timeseries(ds_rho, field_to_sort="ρ", n_workers=args.n_workers, verbose_level=0)
offline = local_potential_energies_timeseries(ds_rho, sorted_state.rho_sorted, sorted_state.dz_sorted,
                                              density_name="ρ", n_workers=args.n_workers, verbose_level=0)

# Υ in the paper's density form, and the same field converted to the buoyancy form the online
# diagnostic writes: Υ_b = -(ρ₀/g) Υ = z₀ - z.
offline_upsilon_rho = offline["upsilon"]
offline_upsilon     = (-(ρ0 / g) * offline_upsilon_rho).rename("Υ")
offline_eps_A       = calculate_ape_dissipation(ds_rho["ρ"], offline_upsilon_rho, κ).rename("ε_A")
dV = ds_raw["dV"]
#---

#+++ Metrics
print("\n" + "="*70)
print("  Displacement potential  Υ = z✶ - z   [rms(online - offline) / rms(online), at the snapshot time]")
print("="*70)
ups_on  = online_upsilon.sel(time=t_sel, method="nearest").squeeze().compute()
ups_off = offline_upsilon.sel(time=t_sel, method="nearest").squeeze().compute()
ups_diff = ups_on - ups_off
rms_ups_online = float(np.sqrt(np.nanmean(ups_on.values**2)))
rms_ups = float(np.sqrt(np.nanmean(ups_diff.values**2))) / rms_ups_online if rms_ups_online > 0 else float("inf")
check(rms_ups, f"    Υ field: rms(diff)/rms(online) = {rms_ups:.3e},  "
               f"max|diff| = {float(np.nanmax(np.abs(ups_diff.values))):.3e},  rms(online) = {rms_ups_online:.3e}", print)

# Reported, not asserted: the difference over *exactly* horizontally uniform fluid is the tie
# convention, a run of Nx·Ny cells filling one cell height whose mid-height (online) and bottom slot
# (offline) differ by this much. At t = 0 the mean lands on it exactly; later it should be far below
# it, since exact ties survive only in the bands against the walls where the initial tanh saturated.
Δz = float(ds_raw.Δz_aac.mean())
N_h = ds_raw.sizes["x_caa"] * ds_raw.sizes["y_aca"]
tie_offset = Δz / 2 - Δz / (2 * N_h)
print(f"    mean(online - offline) = {float(np.nanmean(ups_diff.values)):+.3e}   "
      f"vs the predicted tie offset Δz/2 - Δz/(2·Nx·Ny) = {tie_offset:+.3e}")

print("\n" + "="*70)
print("  APE dissipation  ε_A(x, z)   [rms(online - offline) / rms(online), at the snapshot time]")
print("="*70)
eps_on  = online_eps_A.sel(time=t_sel, method="nearest").squeeze().compute()
eps_off = offline_eps_A.sel(time=t_sel, method="nearest").squeeze().compute()
eps_diff = eps_on - eps_off
rms_eps_online = float(np.sqrt(np.nanmean(eps_on.values**2)))
rms_eps = float(np.sqrt(np.nanmean(eps_diff.values**2))) / rms_eps_online if rms_eps_online > 0 else float("inf")
check(rms_eps, f"    ε_A field: rms(diff)/rms(online) = {rms_eps:.3e},  "
               f"max|diff| = {float(np.nanmax(np.abs(eps_diff.values))):.3e},  rms(online) = {rms_eps_online:.3e}", print)

print("\n" + "="*70)
print("  Integrated dissipation  ∫ε_A dV   [rms(online - offline) / rms(offline), over the run]")
print("="*70)
offline_int = integrate(offline_eps_A, dV).squeeze().compute()
online_int_c = online_int.reindex(time=offline_int.time, method="nearest").compute()
denom = float(np.sqrt(np.nanmean(offline_int.values**2)))
rms_int = float(np.sqrt(np.nanmean((online_int_c.values - offline_int.values)**2))) / denom if denom > 0 else float("inf")
check(rms_int, f"    ∫ε_A dV: rms(online - offline)/rms(offline) = {rms_int:.3e}", print)
#---

#+++ Figures
label = run_label(ds.attrs)
zw = args.z_window


def _maps(on, off, diff, name, cmap, out_name, symmetric):
    """online | offline | difference, on a shared scale for the first two."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    vmax = max(float(np.nanpercentile(np.abs(on.values), 99)), float(np.nanpercentile(np.abs(off.values), 99)))
    vmax = vmax if vmax > 0 else 1.0
    kw = dict(x="x_caa", y="z_aac", add_colorbar=True, cmap=cmap,
              vmin=-vmax if symmetric else 0, vmax=vmax)
    on.plot(ax=axes[0], **kw);  axes[0].set_title(f"Online {name}")
    off.plot(ax=axes[1], **kw); axes[1].set_title(f"Offline {name}")
    diff.plot(ax=axes[2], x="x_caa", y="z_aac", add_colorbar=True, cmap="RdBu_r", robust=True)
    axes[2].set_title("Difference (online − offline)")
    for a in axes:
        if zw is not None:
            a.set_ylim(-zw, zw)
        a.set_aspect("equal")
    fig.suptitle(f"Online vs offline {name}   t = {t_sel:.1f}" + (f"   {label}" if label else ""))
    out = FIGURES / out_name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved {out}")


# ε_A gets a diverging map too: it is not sign-definite pointwise, since the reference state's own
# diffusion (-κ∂b/∂z) can outweigh the diapycnal mixing term locally.
_maps(ups_on, ups_off, ups_diff, "Υ = z✶ − z", "RdBu_r", f"inv08_upsilon_maps_{stem}_t{t_sel:.1f}.png", symmetric=True)
_maps(eps_on, eps_off, eps_diff, "ε_A", "RdBu_r", f"inv08_ape_dissipation_maps_{stem}_t{t_sel:.1f}.png", symmetric=True)

# Integral time series
fig, ax = plt.subplots(figsize=(7, 4.2), constrained_layout=True)
ax.plot(offline_int.time, offline_int, lw=2.5, label="offline")
ax.plot(online_int_c.time, online_int_c, "--", lw=1.6, label="online")
ax.set(xlabel="time", ylabel="∫ε_A dV", title="Volume-integrated APE dissipation")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
fig.suptitle("Online vs offline ∫ε_A dV" + (f"   {label}" if label else ""))
out = FIGURES / f"inv08_ape_dissipation_integral_{stem}.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}")
#---

finalize(print)
