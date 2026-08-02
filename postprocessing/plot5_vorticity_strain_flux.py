#!/usr/bin/env python
"""Cross-scale KE/APE flux conditioned on filtered-field vorticity and strain (Balwada et al. 2021 style)."""

#+++ Imports
import os
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import binned_statistic_2d
from src.aux00_utils import load_dataset_and_grid
#---

#+++ Configuration
import argparse
parser = argparse.ArgumentParser(description="Conditional-mean and net-contribution maps of Πₖ, Π_A, and Πₖ+Π_A in filtered vorticity-strain space")
parser.add_argument("--filename", default="output/bci_Nx48_Ny48_Nz8.nc", help="Path to simulation NetCDF file")
parser.add_argument("--filter-scale", type=float, default=None, help="Target filter length scale in meters (nearest available; defaults to the smallest available)")
parser.add_argument("--time-min", type=float, default=None, help="Start of the time range (days, inclusive; defaults to the earliest available time)")
parser.add_argument("--time-max", type=float, default=None, help="End of the time range (days, inclusive; defaults to the latest available time)")
parser.add_argument("--z", type=float, default=-500.0, help="Target depth in meters (nearest available cell center; default -500, mid-depth)")
parser.add_argument("--n-bins", type=int, default=40, help="Number of bins per axis for the vorticity-strain JPDF (default 40)")
parser.add_argument("--min-count", type=int, default=5, help="Bins with fewer than this many points are masked out in the conditional-mean/net panels (default 5)")
parser.add_argument("--clim-percentile", type=float, default=99.0, help="Percentile of |data| used to set symmetric color limits for flux panels")
parser.add_argument("--percentiles", type=float, nargs="+", default=[50, 90, 99], help="JPDF highest-density-region percentiles to contour on each flux panel (default 50 90 99)")
args = parser.parse_args()
print("\n" + "="*70 + f"\n  {Path(__file__).name}\n  " + "  ".join(f"{k}={v}" for k,v in vars(args).items()) + "\n" + "="*70)

REPO_ROOT = Path(__file__).resolve().parent.parent
PP_OUTPUT = REPO_ROOT / "postprocessing" / "output"
FIGURES = REPO_ROOT / "figures"
FIGURES.mkdir(exist_ok=True)
filename = str(REPO_ROOT / args.filename) if not os.path.isabs(args.filename) else args.filename
stem = Path(filename).stem
#---

#+++ Orientation fix: some pipeline fields (Π_A, the KE<->APE exchange term, and ūᵢ here) are stored with
# dims (..., x, y) instead of (..., y, x) like most fields -- plotting/differentiating without accounting
# for this silently produces rotated or transposed results. Always transpose to (..., y_dim, x_dim) first.
def fix_orientation(da):
    y_dim = next(d for d in da.dims if d.startswith("y"))
    x_dim = next(d for d in da.dims if d.startswith("x"))
    other_dims = [d for d in da.dims if d not in (y_dim, x_dim)]
    return da.transpose(*other_dims, y_dim, x_dim)
#---

#+++ Load filtered velocities, pick filter scale/time/z, compute filtered vorticity and strain
print("Loading filtered velocities...")
filt = xr.open_dataset(PP_OUTPUT / f"{stem}_filtered_velocities.nc", decode_times=False)

ℓ_target = args.filter_scale if args.filter_scale is not None else float(filt.filter_scale.min())
ℓ = float(filt.filter_scale.sel(filter_scale=ℓ_target, method="nearest"))
ℓ_km = int(round(ℓ / 1000))

t_min_sec = args.time_min * 86400 if args.time_min is not None else float(filt.time.min())
t_max_sec = args.time_max * 86400 if args.time_max is not None else float(filt.time.max())
filt_t = filt.sel(time=slice(t_min_sec, t_max_sec))
n_times = filt_t.sizes["time"]
if n_times == 0:
    raise ValueError(f"No available times in [{t_min_sec/86400:.2f}, {t_max_sec/86400:.2f}] days")
t_days_actual = filt_t.time.values / 86400
t_min_days, t_max_days = float(t_days_actual.min()), float(t_days_actual.max())

ubar = fix_orientation(filt_t["ūᵢ"].sel(i=1, filter_scale=ℓ, method="nearest", drop=True)).sel(z_aac=args.z, method="nearest")
vbar = fix_orientation(filt_t["ūᵢ"].sel(i=2, filter_scale=ℓ, method="nearest", drop=True)).sel(z_aac=args.z, method="nearest")
z_sel = float(ubar.z_aac)
print(f"ℓ = {ℓ:.4f} m ({ℓ_km} km), t = {t_min_days:.2f}-{t_max_days:.2f} days ({n_times} snapshots), z = {z_sel:.1f} m")

sigma_n  = ubar.differentiate("x_caa") - vbar.differentiate("y_aca")
sigma_s  = vbar.differentiate("x_caa") + ubar.differentiate("y_aca")
zeta_bar = vbar.differentiate("x_caa") - ubar.differentiate("y_aca")
sigma_bar = np.sqrt(sigma_n**2 + sigma_s**2)

Omega_earth = 7.2921159e-5
f0 = 2 * Omega_earth * np.sin(np.radians(filt.attrs["latitude"]))
zeta_norm  = (zeta_bar / f0).values
sigma_norm = (sigma_bar / abs(f0)).values
print(f"f0 = {f0:.4e} s^-1")
#---

#+++ Load Πₖ, Π_A at the same ℓ, t, z; form the total
print("Loading Πₖ, Π_A...")
ke_fields  = xr.open_dataset(PP_OUTPUT / f"{stem}_sfs_ke_budget_fields.nc",  decode_times=False)
ape_fields = xr.open_dataset(PP_OUTPUT / f"{stem}_sfs_ape_budget_fields.nc", decode_times=False)

# Each of filt_t's own times is matched to its nearest neighbor in the flux files independently (vectorized
# nearest-time selection, one match per snapshot) rather than a single shared lookup -- the three files can
# have slightly different time grids/roundoff, same reasoning as sweep2's ds.reindex(time=ds_filt.time).
Pi_K = fix_orientation(ke_fields["Π_K"].sel(filter_scale=ℓ, time=filt_t.time, method="nearest")).sel(z_aac=args.z, method="nearest").values
Pi_A = fix_orientation(ape_fields["Π_A"].sel(filter_scale=ℓ, time=filt_t.time, method="nearest")).sel(z_aac=args.z, method="nearest").values
Pi_total = Pi_K + Pi_A
#---

#+++ Area weights (for the JPDF) and flattening
# Δx_caa/Δy_aca live on the raw sim file (not the filtered-velocities file), added by load_dataset_and_grid
ds_grid = load_dataset_and_grid(filename)
dx_1d = ds_grid.Δx_caa.values
dy_1d = ds_grid.Δy_aca.values
area_2d = np.outer(dy_1d, dx_1d)  # matches (y_aca, x_caa) after fix_orientation

# Broadcast the (time-independent) area weights across the time dimension so every included snapshot
# contributes one full domain's worth of area-weighted samples to the JPDF/conditional means below --
# snapshots are weighted equally regardless of the (possibly uneven) Δt between them, not by Δt itself,
# consistent with how this codebase already treats multi-snapshot time series elsewhere (e.g. sweep3's
# time-mean/std, which likewise treats each saved snapshot as one equally-weighted sample).
area_3d = np.broadcast_to(area_2d, zeta_norm.shape)
A_total = area_3d.sum()

zeta_flat  = zeta_norm.ravel()
sigma_flat = sigma_norm.ravel()
area_flat  = area_3d.ravel()
Pi_K_flat, Pi_A_flat, Pi_total_flat = Pi_K.ravel(), Pi_A.ravel(), Pi_total.ravel()
#---

#+++ Bin edges (data-driven range) and JPDF
zeta_max  = np.nanpercentile(np.abs(zeta_flat), 99.5)
sigma_max = np.nanpercentile(sigma_flat, 99.5)
zeta_edges  = np.linspace(-zeta_max, zeta_max, args.n_bins + 1)
sigma_edges = np.linspace(0, sigma_max, args.n_bins + 1)

jpdf_counts, _, _ = np.histogram2d(zeta_flat, sigma_flat, bins=[zeta_edges, sigma_edges], weights=area_flat)
dζ = zeta_edges[1] - zeta_edges[0]
dσ = sigma_edges[1] - sigma_edges[0]
jpdf = jpdf_counts / A_total / dζ / dσ

counts, _, _, _ = binned_statistic_2d(zeta_flat, sigma_flat, None, statistic="count", bins=[zeta_edges, sigma_edges])
sparse_mask = counts < args.min_count
#---

#+++ Conditional means and net contributions
def cond_mean_and_net(values_flat):
    mean, _, _, _ = binned_statistic_2d(zeta_flat, sigma_flat, values_flat, statistic="mean", bins=[zeta_edges, sigma_edges])
    mean = np.where(sparse_mask, np.nan, mean)
    net = mean * jpdf
    return mean, net

Pi_K_mean, Pi_K_net = cond_mean_and_net(Pi_K_flat)
Pi_A_mean, Pi_A_net = cond_mean_and_net(Pi_A_flat)
Pi_total_mean, Pi_total_net = cond_mean_and_net(Pi_total_flat)
#---

#+++ Region (AVD/CVD/SD) flux fractions, using the net-contribution field
zeta_centers = 0.5 * (zeta_edges[:-1] + zeta_edges[1:])
sigma_centers = 0.5 * (sigma_edges[:-1] + sigma_edges[1:])
ZZ, SS = np.meshgrid(zeta_centers, sigma_centers, indexing="ij")
sd_mask, avd_mask, cvd_mask = SS >= np.abs(ZZ), (ZZ < 0) & (SS < np.abs(ZZ)), (ZZ > 0) & (SS < np.abs(ZZ))

def region_fractions(net):
    total = np.nansum(net)
    return {r: 100 * np.nansum(net[m]) / total for r, m in [("SD", sd_mask), ("AVD", avd_mask), ("CVD", cvd_mask)]}

fractions = {"Πₖ": region_fractions(Pi_K_net), "Π_A": region_fractions(Pi_A_net), "Πₖ+Π_A": region_fractions(Pi_total_net)}
for name, frac in fractions.items():
    print(f"  {name}: SD={frac['SD']:.1f}%  AVD={frac['AVD']:.1f}%  CVD={frac['CVD']:.1f}%")
#---

#+++ JPDF highest-density-region contour levels: the level of P such that the region {P > level} contains
# a given percentile of the total probability mass (a "percentile contour", as in Balwada et al.'s gray
# contours) -- NOT a fixed absolute-probability threshold, since that wouldn't compare meaningfully across
# filter scales/resolutions where the JPDF's overall magnitude differs.
def percentile_levels(density, mass, percentiles):
    """Return {percentile: density level} such that {density > level} contains that percentile of the mass."""
    order = np.argsort(density.ravel())[::-1]
    sorted_density = density.ravel()[order]
    cum_mass = np.cumsum(mass.ravel()[order])
    total = cum_mass[-1]
    levels = {}
    for p in percentiles:
        idx = min(np.searchsorted(cum_mass, p / 100 * total), len(sorted_density) - 1)
        levels[p] = sorted_density[idx]
    return levels

mass = jpdf * dζ * dσ
level_by_percentile = percentile_levels(jpdf, mass, args.percentiles)
# ax.contour requires levels in ascending order; keep the percentile<->level correspondence via this dict
# rather than re-deriving it from sort order (silently swaps labels if percentiles aren't already sorted).
contour_levels = sorted(level_by_percentile.values())
level_labels = {lvl: f"{p:g}%" for p, lvl in level_by_percentile.items()}
print(f"  JPDF percentile levels -> {level_by_percentile}")
#---

#+++ Plot
print("Building figure...")
fig = plt.figure(figsize=(14, 9), constrained_layout=True)
gs = fig.add_gridspec(2, 3)

flux_panels = [("Πₖ", Pi_K_mean, Pi_K_net), ("Π_A", Pi_A_mean, Pi_A_net), ("Πₖ+Π_A", Pi_total_mean, Pi_total_net)]
axes_mean = [fig.add_subplot(gs[0, c]) for c in range(3)]
axes_net  = [fig.add_subplot(gs[1, c]) for c in range(3)]

def add_sd_lines(ax):
    zmax = zeta_edges[-1]
    z = np.linspace(-zmax, zmax, 200)
    ax.plot(z, np.abs(z), "--", color="gray", lw=1)

_LINESTYLES = ["dotted", "dashed", "solid", "dashdot"]

def add_jpdf_contours(ax):
    # One contour call per level (not a single multi-level call) so each percentile gets its own linestyle --
    # inline clabel was rejected: a single JPDF level from one noisy snapshot often breaks into several
    # disconnected pieces, so clabel stamped the same "50%" text repeatedly across the panel. A shared
    # figure-level legend (built once, below) avoids that clutter entirely.
    for i, lvl in enumerate(contour_levels):
        ax.contour(zeta_centers, sigma_centers, jpdf.T, levels=[lvl], colors="0.25", linewidths=1.0,
                   linestyles=_LINESTYLES[i % len(_LINESTYLES)])

for ax, (name, mean, _) in zip(axes_mean, flux_panels):
    vmax = np.nanpercentile(np.abs(mean), args.clim_percentile)
    im = ax.pcolormesh(zeta_edges, sigma_edges, mean.T, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="flat")
    add_sd_lines(ax)
    add_jpdf_contours(ax)
    ax.set_title(f"{name} conditional mean\nSD={fractions[name]['SD']:.0f}% AVD={fractions[name]['AVD']:.0f}% CVD={fractions[name]['CVD']:.0f}%", fontsize=10)
    ax.set_xlabel(r"$\bar\zeta / f_0$")
    ax.set_ylabel(r"$\bar\sigma / |f_0|$")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

for ax, (name, _, net) in zip(axes_net, flux_panels):
    vmax = np.nanpercentile(np.abs(net), args.clim_percentile)
    im = ax.pcolormesh(zeta_edges, sigma_edges, net.T, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="flat")
    add_sd_lines(ax)
    add_jpdf_contours(ax)
    ax.set_title(f"{name} net contribution", fontsize=10)
    ax.set_xlabel(r"$\bar\zeta / f_0$")
    ax.set_ylabel(r"$\bar\sigma / |f_0|$")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

legend_handles = [Line2D([0], [0], color="0.25", lw=1.2, linestyle=_LINESTYLES[i % len(_LINESTYLES)],
                        label=f"JPDF {level_labels[lvl]} HDR") for i, lvl in enumerate(contour_levels)]
fig.legend(handles=legend_handles, loc="upper center", ncol=len(legend_handles), fontsize=9,
           frameon=False, bbox_to_anchor=(0.5, 1.06))

fig.suptitle(f"{stem}, ℓ={ℓ_km}km, t={t_min_days:.1f}-{t_max_days:.1f}d ({n_times} snapshots), z={z_sel:.0f}m", fontsize=13, y=1.1)

z_m = int(round(z_sel))
outfile = FIGURES / f"{stem}_vorticity_strain_flux_l{ℓ_km}km_z{z_m}m_t{t_min_days:.0f}-{t_max_days:.0f}d.pdf"
fig.savefig(outfile, dpi=150, bbox_inches="tight")
print(f"Saved: {outfile}")
#---
