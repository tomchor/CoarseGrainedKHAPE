#!/usr/bin/env python
"""
Compare the online (Oceanostics, simulation-time) sub-filter available potential energy Eₐˢ against the
offline (Python post-processing) one, at every filter scale.

The simulation computes, at each scale ℓ in its online `filter_ℓs` and each output,

    Eₐˢ = filter(eₐ) - eₐˡ ,   eₐ = eₐ(b, z) ,   eₐˡ = eₐ(b̄, z) ,   b̄ = filter(b)

(Oceanostics' `SubFilterAvailablePotentialEnergy`), the energy whose budget Eₐˢ is the diffusive sink
of. Both halves are measured against one shared reference profile, the sort of the full buoyancy, which
is what makes the difference a decomposition rather than a difference of two unrelated quantities.

Offline this is `filter(ape) - ape(ρ̄)` built from `local_potential_energies_timeseries`, exactly as
`05_sfs_ape_budget.py` builds its `subfilter_local_ape`. Two differences between the two, neither an
error:

  * **Ties.** The online reference heights come from a `ProfileLookup`, which puts a run of equal
    buoyancy at the mid-height of the band it fills, where the offline z₀ takes the run's bottom slot.
    Eₐ is near zero over tied fluid for both, so this concentrates in the boundary bands.

  * **The lookup itself.** Online, both states are matched into the shared profile by binary search;
    offline, `local_potential_energies_timeseries` does a nearest-density search of the filtered density
    into the full field's sorted profile. Those are the same operation up to the tie convention above.

Unlike the dissipation, Eₐˢ involves no derivatives, so it carries none of the conservative-vs-centered
discretization gap that dominates the Eₐˢ comparison.

The offline sort runs on the *unpadded* (true) domain, the like-for-like control for the online one, as
in `inv07`; the padding effect is quantified once, in `inv06`.

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
from src.aux00_utils import (load_dataset_and_grid, integrate, make_gaussian_filter, open_grid_group,
                             model_grid_suffix, strip_grid_suffix)
from src.aux01_pe_functions import (calculate_density_fields_from_buoyancy, sorted_timeseries,
                                    local_potential_energies_timeseries)
from src.aux03_plotting import run_label
#---

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
print = logging.info

#+++ Configuration
import argparse
parser = argparse.ArgumentParser(description="Compare online vs offline sub-filter APE Eₐˢ")
parser.add_argument("--filename", default="output/khi_Nz256_Ri0.10.nc", help="Path to simulation NetCDF file (run with --save_sorted)")
parser.add_argument("--filter-scales", type=float, nargs="+", default=[1, 7], help="Filter ℓ (FWHM) values matching the online filter_ℓs")
parser.add_argument("--time", type=float, default=None, help="Target time for the snapshot maps (default: midpoint of simulation)")
parser.add_argument("--z-window", type=float, default=6.0, help="Half-height of the z window shown in the snapshot maps (default: 6h; None for the full domain)")
parser.add_argument("--n-workers", type=int, default=1, help="Thread-pool workers for the offline sorts and APE")
add_tolerance_arg(parser)
args = parser.parse_args()
set_tolerance(args.tolerance)

print("\n" + "="*70 + f"\n  {Path(__file__).name}\n  " + "  ".join(f"{k}={v}" for k, v in vars(args).items()) + "\n" + "="*70)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # validation/ → postprocessing/ → repo root
FIGURES = REPO_ROOT / "figures" / "validation"
FIGURES.mkdir(parents=True, exist_ok=True)
filename = str(REPO_ROOT / args.filename) if not os.path.isabs(args.filename) else args.filename
stem = Path(filename).stem
FILTER_DIMS = ["x_caa", "z_aac"]


def online_name(ℓ, suffix=""):
    """Online output variable for scale ℓ, matching the Julia Symbol("E_as_ℓ$(ℓ)")."""
    tag = f"{int(ℓ)}" if float(ℓ) == int(ℓ) else f"{ℓ}"
    return f"E_as_ℓ{tag}{suffix}"
#---

#+++ Load the padded dataset and the online fields
print("Loading simulation data...")
ds = load_dataset_and_grid(filename)          # z-padded; also strips the model grid's _gridN suffix
ds = ds.chunk({"time": 1})

# The true (unpadded) domain, read from the grid group, as in inv07.
grid = open_grid_group(filename)
z_bot, z_top = float(grid.z.min()), float(grid.z.max())
in_domain = dict(z_aac=slice(z_bot, z_top))

for ℓ in args.filter_scales:
    for name in (online_name(ℓ), online_name(ℓ, "_int")):
        if name not in ds:
            raise SystemExit(f"Online field '{name}' not in {filename} — rerun the simulation with "
                             f"--save_sorted and a matching --filter_ls (needs the Oceanostics "
                             f"`SubFilterAvailablePotentialEnergy`).")

if args.time is None:
    args.time = float(ds.time.values[len(ds.time) // 2])
t_sel = float(ds.time.sel(time=args.time, method="nearest").values)
print(f"Selected snapshot time = {t_sel:.3f}  (requested {args.time})")
#---

#+++ Rebuild the unpadded state and sort it once (shared by every filter scale)
print("Building the unpadded dataset and sorting the full buoyancy...")
ds_raw = xr.open_dataset(filename, decode_times=False, chunks={}).chunk({"time": 1})
ds_raw = strip_grid_suffix(ds_raw, model_grid_suffix(ds_raw))
ds_raw["dV"]   = ds_raw.Δx_caa * ds_raw.Δy_aca * ds_raw.Δz_aac
ds_raw["LxLy"] = float(grid.x[-1] - grid.x[0]) * float(grid.y[-1] - grid.y[0])
ds_raw.attrs["z_min"] = z_bot
ds_raw.attrs["z_max"] = z_top

dV = ds_raw["dV"]

ds_rho = ds_raw[["b", "dV", "LxLy"]].copy()
ds_rho.attrs.update(ds_raw.attrs)
ds_rho = calculate_density_fields_from_buoyancy(ds_rho, buoyancy_name="b", density_name="ρ")

# One sort of the full field, shared by every scale — which is also what the online diagnostics do,
# since they are all handed the same `VerticalSort` column to look up into.
sorted_state = sorted_timeseries(ds_rho, field_to_sort="ρ", n_workers=args.n_workers, verbose_level=0)
full_local_pes = local_potential_energies_timeseries(ds_rho, sorted_state.rho_sorted, sorted_state.dz_sorted,
                                                     density_name="ρ", n_workers=args.n_workers, verbose_level=0)
#---

#+++ Per-scale comparison
label = run_label(ds.attrs)
zw = args.z_window
integrals = {}

# One figure: a row of maps per filter scale, then the volume integrals spanning the row below.
n_scales = len(args.filter_scales)
fig = plt.figure(figsize=(15, 4.2 * n_scales + 4.6), constrained_layout=True)
_gs = fig.add_gridspec(n_scales + 1, 3, height_ratios=[4.2] * n_scales + [4.6])
map_axes = [[fig.add_subplot(_gs[i, k]) for k in range(3)] for i in range(n_scales)]

for row, ℓ in enumerate(args.filter_scales):
    print("\n" + "="*70)
    print(f"  filter scale ℓ = {ℓ:g}")
    print("="*70)

    gf = make_gaussian_filter(ℓ, ds_raw)

    # ρ̄ from the filtered buoyancy, exactly as the budget pipeline builds it (01 filters b, then 05
    # converts), and Υˡ from looking that filtered density up in the *full* field's sorted profile.
    ds_filt = xr.Dataset({"b̄": gf.apply(ds_rho["b"], dims=FILTER_DIMS)})
    ds_filt.attrs.update(ds_raw.attrs)
    ds_filt = calculate_density_fields_from_buoyancy(ds_filt, buoyancy_name="b̄", density_name="ρ̄")
    filt_local_pes = local_potential_energies_timeseries(ds_filt, sorted_state.rho_sorted, sorted_state.dz_sorted,
                                                         density_name="ρ̄", n_workers=args.n_workers, verbose_level=0)

    offline = (gf.apply(full_local_pes.ape, dims=FILTER_DIMS) - filt_local_pes.ape).rename("Eₐˢ")
    online = ds[online_name(ℓ)].sel(**in_domain)

    on_snap  = online.sel(time=t_sel, method="nearest").squeeze().compute()
    off_snap = offline.sel(time=t_sel, method="nearest").squeeze().compute()
    diff = on_snap - off_snap
    rms_online = float(np.sqrt(np.nanmean(on_snap.values**2)))
    rms_field = float(np.sqrt(np.nanmean(diff.values**2))) / rms_online if rms_online > 0 else float("inf")
    check(rms_field, f"    Eₐˢ field (ℓ={ℓ:g}): rms(diff)/rms(online) = {rms_field:.3e},  "
                     f"max|diff| = {float(np.nanmax(np.abs(diff.values))):.3e},  rms(online) = {rms_online:.3e}", print)

    offline_int  = integrate(offline, dV).squeeze().compute()
    online_int   = ds[online_name(ℓ, "_int")].squeeze(drop=True).reindex(time=offline_int.time, method="nearest").compute()
    denom = float(np.sqrt(np.nanmean(offline_int.values**2)))
    rms_int = float(np.sqrt(np.nanmean((online_int.values - offline_int.values)**2))) / denom if denom > 0 else float("inf")
    check(rms_int, f"    ∫Eₐˢ dV (ℓ={ℓ:g}): rms(online - offline)/rms(offline) = {rms_int:.3e}", print)
    integrals[ℓ] = (offline_int, online_int)

    # Maps: online | offline | difference, into this scale's row
    axes = map_axes[row]
    vmax = max(float(np.nanpercentile(np.abs(on_snap.values), 99)), float(np.nanpercentile(np.abs(off_snap.values), 99)))
    vmax = vmax if vmax > 0 else 1.0
    kw = dict(x="x_caa", y="z_aac", add_colorbar=True, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    on_snap.plot(ax=axes[0], **kw);  axes[0].set_title(f"Online Eₐˢ (ℓ={ℓ:g})")
    off_snap.plot(ax=axes[1], **kw); axes[1].set_title(f"Offline Eₐˢ (ℓ={ℓ:g})")
    diff.plot(ax=axes[2], x="x_caa", y="z_aac", add_colorbar=True, cmap="RdBu_r", robust=True)
    axes[2].set_title("Difference (online − offline)")
    for a in axes:
        if zw is not None:
            a.set_ylim(-zw, zw)
        a.set_aspect("equal")
    axes[0].set_ylabel(f"ℓ = {ℓ:g}", fontsize=13)
#---

#+++ Integral time series, all scales, in the figure's last row
ax = fig.add_subplot(_gs[n_scales, :])
for i, (ℓ, (offline_int, online_int)) in enumerate(integrals.items()):
    ax.plot(offline_int.time, offline_int, lw=2.5, color=f"C{i}", label=f"offline  ℓ={ℓ:g}")
    ax.plot(online_int.time, online_int, "--", lw=1.6, color=f"C{i}", label=f"online  ℓ={ℓ:g}")
ax.set(xlabel="time", ylabel="∫Eₐˢ dV", title="Volume-integrated sub-filter APE")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

fig.suptitle(f"Online vs offline Eₐˢ   maps at t = {t_sel:.1f}, volume integrals over the run"
             + (f"   {label}" if label else ""))
out = FIGURES / f"inv10_sfs_ape_{stem}_t{t_sel:.1f}.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved {out}")
#---

finalize(print)
