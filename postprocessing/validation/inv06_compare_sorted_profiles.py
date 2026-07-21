#!/usr/bin/env python
"""
Compare the online (Oceanostics, simulation-time) Winters et al. (1995) sorted reference state
against the offline (Python post-processing) one, for all three Oceanostics sorting methods.

The simulation must have been run with `--save_sorted`, which writes

    z✶_3dsort              ThreeDimensionalSort   z✶ on the model grid, tied cells take consecutive slots
    z✶_heaviside           HeavisideIntegral      z✶ on the model grid, tied cells share their layer mid-height
    z✶_1dsort, b✶_1dsort   OneDimensionalSort     the sorted column, on its own N-cell vertical axis

all into the same output file. Since the column is on a different grid from the model fields, the
writer suffixes every dimension name (`z_aac` -> `z_aac_grid1` for the model, `_grid2` for the column)
and prefixes the grid metadata groups; `load_dataset_and_grid` strips the model grid's suffix so the
rest of the pipeline is unaffected (see `strip_grid_suffix` in `aux00_utils`).

Only the column is a reference *profile* as written. For the two model-grid methods the buoyancy that
pairs with z✶ is the model's own `b`, so their profiles are recovered by ordering the (z✶, b) pairs by
z✶ — what `profile_from_model_grid` does below, mirroring the lock_release example in the Oceanostics
PR. All three are then compared against the offline sort (`sorted_timeseries` + the nearest-density z₀
lookup of `local_potential_energies_timeseries`), which is run twice:

  * **padded**   — on the z-padded domain, which is what the production pipeline actually sorts, since
                   `load_dataset_and_grid` doubles the domain height with edge values at load time.
  * **unpadded** — on the true domain, which is the like-for-like control for the online sort. The
                   online sort never sees the padding.

Splitting those two isolates the padding effect from any genuine online-vs-offline difference.

The three methods describe the same reference state and must agree on every volume integral, so the
RPE time series is the sharpest check; they legitimately disagree cell by cell wherever buoyancy ties.
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
from src.aux00_utils import load_dataset_and_grid, integrate, open_grid_group, model_grid_suffix, strip_grid_suffix
from src.aux01_pe_functions import calculate_density_fields_from_buoyancy, sorted_timeseries, g, ρ0
from src.aux03_plotting import run_label
#---

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
print = logging.info

#+++ Configuration
import argparse
parser = argparse.ArgumentParser(description="Compare online vs offline Winters (1995) sorted reference state, for all three sorting methods")
parser.add_argument("--filename", default="output/khi_Nz256_Ri0.10.nc", help="Path to simulation NetCDF file (run with --save_sorted)")
parser.add_argument("--time", type=float, default=None, help="Target time for the snapshot maps (default: midpoint of simulation)")
parser.add_argument("--z-window", type=float, default=None, help="Half-height of the z window shown in the snapshot maps (default: the full domain — the tie and padding effects this script is about live at the saturated top and bottom, so cropping to the shear layer hides them)")
parser.add_argument("--n-workers", type=int, default=1, help="Thread-pool workers for the offline sort")
args = parser.parse_args()

print("\n" + "="*70 + f"\n  {Path(__file__).name}\n  " + "  ".join(f"{k}={v}" for k, v in vars(args).items()) + "\n" + "="*70)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # validation/ → postprocessing/ → repo root
FIGURES = REPO_ROOT / "figures"
FIGURES.mkdir(exist_ok=True)
filename = str(REPO_ROOT / args.filename) if not os.path.isabs(args.filename) else args.filename
stem = Path(filename).stem
#---

#+++ Buoyancy <-> density
# The offline pipeline sorts density (heaviest first); Oceanostics sorts buoyancy (densest = lowest b
# first). The two orderings are identical, since ρ = ρ₀(1 - b/g) is monotonically decreasing in b.
def b_to_rho(b):
    return ρ0 * (1 - b / g)

def rho_to_b(rho):
    return g * (1 - rho / ρ0)
#---

#+++ Load the padded dataset (what the production pipeline sees) and recover the true domain
print("Loading simulation data...")
ds = load_dataset_and_grid(filename)          # z-padded to 2x height with edge values
ds = ds.chunk({"time": 1})

grid = open_grid_group(filename)   # handles the _gridN naming a --save_sorted run introduces
z_bot, z_top = float(grid.z.min()), float(grid.z.max())   # original (unpadded) z faces
in_domain = dict(z_aac=slice(z_bot, z_top))               # selects the original cell centers

for v in ("z✶_3dsort", "z✶_heaviside"):
    if v not in ds:
        raise SystemExit(f"Online field '{v}' not in {filename} — rerun the simulation with --save_sorted.")

# The sorted column shares the output file with the model-grid fields. Because they are on different
# grids the writer suffixes every dimension (`z_aac` -> `z_aac_grid1` for the model, `_grid2` for the
# column); `load_dataset_and_grid` strips the model's suffix, so the column's variables are still in
# `ds` carrying theirs. Read them from the raw file to keep the padded/unpadded bookkeeping clear.
ds_col = xr.open_dataset(filename, decode_times=False)
for v in ("z✶_1dsort", "b✶_1dsort"):
    if v not in ds_col:
        raise SystemExit(f"Online field '{v}' not in {filename} — rerun the simulation with --save_sorted.")
print(f"Sorted column read from {Path(filename).name}, on its own axis: {ds_col['b✶_1dsort'].dims}")

if args.time is None:
    args.time = float(ds.time.values[len(ds.time) // 2])
t_sel = float(ds.time.sel(time=args.time, method="nearest").values)
print(f"Selected snapshot time = {t_sel:.3f}  (requested {args.time})")
#---

#+++ Offline sort, on the padded and on the unpadded domain
def offline_sort(ds_in, label):
    """Run the production offline sort (`sorted_timeseries`) and the nearest-density z₀ lookup on `ds_in`."""
    print(f"  offline sort [{label}]: {ds_in.sizes['x_caa']}x{ds_in.sizes['y_aca']}x{ds_in.sizes['z_aac']} cells "
          f"x {ds_in.sizes['time']} times")
    dsr = ds_in[["b", "dV", "LxLy"]].copy()
    dsr.attrs.update(ds_in.attrs)
    dsr = calculate_density_fields_from_buoyancy(dsr, buoyancy_name="b", density_name="ρ")
    srt = sorted_timeseries(dsr, field_to_sort="ρ", n_workers=args.n_workers, verbose_level=0)
    return dsr, srt


def offline_z0(rho, rho_sorted, z_sorted):
    """z₀ = z_*(ρ): nearest-density lookup in the sorted profile, exactly as `_process_single_timestep` does."""
    rho_flat = np.asarray(rho).ravel()
    rho_s    = np.asarray(rho_sorted)
    z_s      = np.asarray(z_sorted)

    idx        = np.clip(np.searchsorted(-rho_s, -rho_flat, side="left"), 0, len(rho_s) - 1)
    idx_left   = np.maximum(idx - 1, 0)
    dist_left  = np.abs(rho_s[idx_left] - rho_flat)
    dist_right = np.abs(rho_s[idx]      - rho_flat)
    best       = np.where(dist_left < dist_right, idx_left, idx)
    return z_s[best].reshape(np.shape(rho))


# The unpadded dataset: reopen the raw file and rebuild the few grid variables `sorted_timeseries` needs,
# without going through load_dataset_and_grid (which always pads).
print("Building the unpadded dataset (the like-for-like control for the online sort)...")
ds_raw = xr.open_dataset(filename, decode_times=False, chunks={}).chunk({"time": 1})
# Same normalization load_dataset_and_grid does, minus the padding: strip the model grid's `_gridN`
# suffix so the plain names below resolve. A no-op unless the run used --save_sorted.
ds_raw = strip_grid_suffix(ds_raw, model_grid_suffix(ds_raw))
ds_raw.attrs.update(ds.attrs)
ds_raw["dV"]    = ds_raw.Δx_caa * ds_raw.Δy_aca * ds_raw.Δz_aac
ds_raw["LxLy"]  = float(np.diff(grid.x)) * float(np.diff(grid.y))
ds_raw.attrs["z_min"] = z_bot
ds_raw.attrs["z_max"] = z_top

print("Running the offline sort...")
ds_pad_rho,   srt_pad   = offline_sort(ds,     "padded")
ds_unpad_rho, srt_unpad = offline_sort(ds_raw, "unpadded")
#---

#+++ Assemble the sorted profiles, as buoyancy on a common footing
# Online: b✶ on the sorted column. Offline: rho_sorted on z_1d_sorted -> convert to buoyancy.
b_col = ds_col["b✶_1dsort"].squeeze(drop=True)
z_col = ds_col["z✶_1dsort"].squeeze(drop=True)
col_dim = [d for d in b_col.dims if d != "time"][0]
b_col = b_col.assign_coords({col_dim: np.asarray(z_col.isel(time=0))}).rename({col_dim: "z✶"})


def profile_from_model_grid(z_star_field, b_field):
    """Recover the reference profile b✶(z✶) from a model-grid method.

    `reference_buoyancy` is the model's own `b` for `ThreeDimensionalSort` and `HeavisideIntegral`,
    since z✶ and b already pair up cell by cell there. Ordering those pairs by z✶ gives the same
    profile `OneDimensionalSort` stores directly — which is exactly what the lock_release example in
    the Oceanostics PR does to plot all three methods on the same axes.
    """
    z = np.asarray(z_star_field).ravel()
    b = np.asarray(b_field).ravel()
    order = np.argsort(z)
    return xr.DataArray(b[order], dims="z✶", coords={"z✶": z[order]})


# All three online methods as profiles, at the snapshot time, plus the two offline sorts.
profiles = {
    "online (OneDimensionalSort)": b_col,
    "offline (unpadded)": rho_to_b(srt_unpad.rho_sorted).rename({"z_1d_sorted": "z✶"}),
    "offline (padded)":   rho_to_b(srt_pad.rho_sorted).rename({"z_1d_sorted": "z✶"}),
}
print("\nSorted-profile lengths:  " + "   ".join(f"{k}: N={v.sizes['z✶']}" for k, v in profiles.items()))

# Invariant: the online column is a *permutation* of the model's own buoyancy at the same time, so
# sorting the raw field must reproduce it exactly. This is what catches b✶ going stale. Oceanostics
# fills the column as a side effect of sorting z✶, and an earlier revision of the branch returned it as
# a bare `Field` that `compute!` ignored, so an output writer kept emitting the *previous* sort's
# profile — no error, entirely plausible-looking output. `reference_buoyancy` now wraps it so it
# recomputes itself, and this assertion guards against that regressing. Bit-for-bit is the right
# tolerance: both sides sort identical Float64 data, so any real difference is a bug, not roundoff.
b_online = b_col.sel(time=t_sel, method="nearest").values
b_raw    = np.sort(ds_raw["b"].sel(time=t_sel, method="nearest").values.ravel())
stale    = np.abs(b_online - b_raw).max()
if stale > 0:
    lag = {float(t): float(np.abs(b_online - np.sort(ds_raw["b"].sel(time=t).values.ravel())).max())
           for t in ds_raw.time.values}
    best_t, best_d = min(lag.items(), key=lambda kv: kv[1])
    raise SystemExit(
        f"Online sorted profile b✶ is not a permutation of b at t={t_sel:g} (max|diff| = {stale:.3e}).\n"
        f"  Closest match is t={best_t:g} (max|diff| = {best_d:.3e}).\n"
        f"  If that is an *earlier* time, b✶ is stale — the sort that fills it did not run before the "
        f"writer read it. Check that `reference_buoyancy` still returns a self-recomputing field.")
print(f"Invariant OK: online b✶ is an exact permutation of b at t={t_sel:.3f}")
#---

#+++ Assemble the reference-height fields z✶(x, z), on the true domain
z_star = {
    "online (ThreeDimensionalSort)": ds["z✶_3dsort"].sel(**in_domain),
    "online (HeavisideIntegral)":    ds["z✶_heaviside"].sel(**in_domain),
}

for label, (dsr, srt, sel) in {
    "offline (unpadded)": (ds_unpad_rho, srt_unpad, {}),
    "offline (padded)":   (ds_pad_rho,   srt_pad,   in_domain),
}.items():
    # The lookup is cheap next to the sort, so do it for every time: the RPE time series below needs it.
    rho = dsr["ρ"].transpose("time", ...)
    z_s = srt.rho_sorted.z_1d_sorted.values
    z0 = np.stack([offline_z0(rho.isel(time=i).values, srt.rho_sorted.isel(time=i).values, z_s)
                   for i in range(rho.sizes["time"])])
    z_star[label] = xr.DataArray(z0, dims=rho.dims, coords=rho.coords).sel(**sel)
#---

#+++ Metrics
Lz = z_top - z_bot
print("\n" + "="*70)
print("  Sorted profile b✶(z✶)   [rms difference vs the online column, in units of B₀]")
print("="*70)
B0 = float(ds.attrs.get("B₀", 1.0))
ref_prof = profiles["online (OneDimensionalSort)"].sel(time=t_sel, method="nearest")

# The two model-grid methods become profiles by pairing their z✶ with the model's own b and ordering.
# Done here rather than above because it needs `z_star`, and only at the snapshot time (these are the
# comparison's subject, not a time series). They must reproduce the column: same set of cells, same
# buoyancies, just indexed by cell instead of by rank.
b_true_snap = ds["b"].sel(**in_domain).sel(time=t_sel, method="nearest")
for _lbl, _key in (("online (ThreeDimensionalSort)", "online (ThreeDimensionalSort)"),
                   ("online (HeavisideIntegral)",    "online (HeavisideIntegral)")):
    profiles[_lbl] = profile_from_model_grid(z_star[_key].sel(time=t_sel, method="nearest"), b_true_snap)


def compare_profiles(a, b):
    """Difference between two sorted profiles, elementwise when they share a z✶ axis, else interpolated.

    The unpadded offline sort produces exactly the online column's axis (same N, same cumulative-volume
    heights), so interpolating it onto a third grid would report interpolation error rather than a real
    difference. The padded sort has 2N cells over 2·Lz and has to be interpolated.
    """
    za, zb = a["z✶"].values, b["z✶"].values
    if za.shape == zb.shape and np.allclose(za, zb, rtol=0, atol=1e-9 * Lz):
        return a.values - b.values, "elementwise"
    common = np.linspace(z_bot, z_top, min(a.sizes["z✶"], b.sizes["z✶"]))
    return np.interp(common, zb, b.values) - np.interp(common, za, a.values), "interpolated"


REF_PROFILE = "online (OneDimensionalSort)"
for label, prof in profiles.items():
    if label == REF_PROFILE:
        continue
    p = prof.sel(time=t_sel, method="nearest") if "time" in prof.dims else prof
    d, how = compare_profiles(ref_prof, p)
    print(f"    {label:<30}  rms = {np.sqrt(np.mean(d**2))/B0:.3e}   max = {np.max(np.abs(d))/B0:.3e}   [{how}]")

print("\n" + "="*70)
print("  Reference height z✶(x, z)   [pairwise rms difference, in units of Lz]")
print("="*70)
labels = list(z_star)
snap = {k: v.sel(time=t_sel, method="nearest").squeeze().compute() for k, v in z_star.items()}
print(f"    {'':<32}" + "".join(f"{l:>30}" for l in labels))
for a in labels:
    row = f"    {a:<32}"
    for b in labels:
        d = float(np.sqrt(np.mean((snap[a].values - snap[b].values) ** 2))) / Lz
        row += f"{d:>30.3e}"
    print(row)

# Where the disagreement lives. The methods can only differ where cells share a buoyancy, and in this
# setup that is the quiescent fluid above and below the shear layer, where b has saturated at ±B₀. Away
# from those ties the reference height is uniquely determined, so every method must agree there.
b_snap    = ds["b"].sel(**in_domain).sel(time=t_sel, method="nearest").squeeze().values
saturated = np.abs(b_snap) > 0.99 * B0
print(f"\n    Split by buoyancy ties  ({100*saturated.mean():.1f}% of cells have |b| > 0.99·B₀, i.e. are tied):")
print(f"    {'':<32}{'tied fluid':>20}{'stratified interior':>24}")
for a in labels:
    if a == "online (ThreeDimensionalSort)":
        continue
    d = (snap[a].values - snap["online (ThreeDimensionalSort)"].values) / Lz
    rms_tie = float(np.sqrt(np.mean(d[saturated] ** 2))) if saturated.any() else np.nan
    rms_int = float(np.sqrt(np.mean(d[~saturated] ** 2))) if (~saturated).any() else np.nan
    print(f"    {a:<32}{rms_tie:>20.3e}{rms_int:>24.3e}")

# OneDimensionalSort answers on the sorted column rather than the model grid, so it has no cell-by-cell
# entry in the table above. It is still tied to the other two exactly: ThreeDimensionalSort gives each
# cell the height of its own slot in that same column, so collecting its z✶ values and sorting them must
# reproduce the column verbatim. That is the whole content of "the two methods describe the same sorted
# state, on different grids", and it is a bit-for-bit check rather than a tolerance.
print("\n" + "="*70)
print("  OneDimensionalSort vs ThreeDimensionalSort   [same sorted column, indexed by rank vs by cell]")
print("="*70)
z_col_t   = np.sort(z_col.sel(time=t_sel, method="nearest").values.ravel())
z_3d_sort = np.sort(snap["online (ThreeDimensionalSort)"].values.ravel())
if z_col_t.shape != z_3d_sort.shape:
    print(f"    SKIPPED: column has N={z_col_t.size} but the model grid has {z_3d_sort.size} cells")
else:
    d = np.abs(z_col_t - z_3d_sort)
    print(f"    sorted(z✶_3dsort) vs the 1D column:  max|diff| = {d.max():.3e}  ({d.max()/Lz:.3e} · Lz)")
    print(f"    {'exact match' if d.max() == 0 else 'MISMATCH — the two are not the same sorted column'}")

# The integral every method must agree on: RPE = ∫ ρ g z✶ dV / ρ₀  ( = -∫ b z✶ dV, the Winters E_b).
print("\n" + "="*70)
print("  Background potential energy  ∫E_b dV = -∫ b z✶ dV   [relative to the online 3D sort]")
print("="*70)
b_true = ds["b"].sel(**in_domain)
dV     = ds["dV"].sel(**in_domain)
Eb = {k: float(integrate(-b_true.sel(time=t_sel, method="nearest") * v.sel(time=t_sel, method="nearest"),
                         dV).squeeze()) for k, v in z_star.items()}
Eb_ref = Eb["online (ThreeDimensionalSort)"]
for k, v in Eb.items():
    print(f"    {k:<32} ∫E_b = {v: .6e}   rel. diff = {(v - Eb_ref)/abs(Eb_ref): .3e}")
#---

#+++ Figure 1: the sorted profiles
label_run = run_label(ds.attrs)
fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)

for lbl, prof in profiles.items():
    p = prof.sel(time=t_sel, method="nearest") if "time" in prof.dims else prof
    axes[0].plot(p.values / B0, p["z✶"].values, label=f"{lbl}  (N={p.sizes['z✶']})",
                 lw=2.5 if lbl.startswith("online") else 1.4,
                 ls="-" if lbl.startswith("online") else "--")
axes[0].set(xlabel="b✶ / B₀", ylabel="z✶", title="Sorted buoyancy profile")
axes[0].axhspan(z_bot, z_top, color="k", alpha=0.05, zorder=0)
axes[0].legend(fontsize=8)
axes[0].grid(alpha=0.3)

for lbl, prof in profiles.items():
    if lbl == REF_PROFILE:
        continue
    p = prof.sel(time=t_sel, method="nearest") if "time" in prof.dims else prof
    d, how = compare_profiles(ref_prof, p)
    z_axis = ref_prof["z✶"].values if how == "elementwise" else np.linspace(z_bot, z_top, len(d))
    axes[1].plot(d / B0, z_axis, lw=1.4, label=f"{lbl} [{how}]")
axes[1].set(xlabel="(other − 1D-sort column) b✶ / B₀", ylabel="z✶", title="Difference from the online column")
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.3)

fig.suptitle(f"Sorted reference profile — online vs offline   ({label_run},  t = {t_sel:.2f})")
out1 = FIGURES / f"inv06_sorted_profile_{stem}.png"
fig.savefig(out1, dpi=150)
print(f"\nSaved {out1}")
#---

#+++ Figure 2: the reference-height maps
zw = args.z_window
n = len(labels)
fig, axes = plt.subplots(2, n, figsize=(4.2 * n, 8), constrained_layout=True, squeeze=False)

# Scale the z✶ maps to the true domain. z✶ is a height, so this is its natural range; the padded
# offline sort assigns heights outside it, and letting those saturate the colormap is the point.
for i, lbl in enumerate(labels):
    snap[lbl].plot(ax=axes[0, i], x="x_caa", y="z_aac", vmin=z_bot, vmax=z_top, cmap="viridis", add_colorbar=True)
    axes[0, i].set_title(f"z✶ — {lbl}", fontsize=9)

    d = (snap[lbl] - snap["online (ThreeDimensionalSort)"]) / Lz
    # Not `robust=True`: the differences are concentrated in the tied, saturated fluid near the top and
    # bottom, which is exactly what percentile clipping would throw away.
    m = float(np.abs(d).max()) or 1.0
    d.plot(ax=axes[1, i], x="x_caa", y="z_aac", cmap="RdBu_r", vmin=-m, vmax=m, add_colorbar=True)
    axes[1, i].set_title("(− online 3D sort) / Lz", fontsize=9)

    if zw is not None:
        for r in range(2):
            axes[r, i].set_ylim(-zw, zw)

fig.suptitle(f"Winters reference height z✶ — online vs offline   ({label_run},  t = {t_sel:.2f})")
out2 = FIGURES / f"inv06_reference_height_{stem}.png"
fig.savefig(out2, dpi=150)
print(f"Saved {out2}")
#---

#+++ Figure 3: RPE time series (the integral all methods must agree on)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
Eb_t = {k: integrate(-b_true * v, dV).squeeze().compute() for k, v in z_star.items()}
ref_t = Eb_t["online (ThreeDimensionalSort)"]
for lbl, series in Eb_t.items():
    axes[0].plot(series.time, series, label=lbl, lw=2.5 if lbl.startswith("online") else 1.4,
                 ls="-" if lbl.startswith("online") else "--")
    axes[1].plot(series.time, (series - ref_t) / np.abs(ref_t), label=lbl, lw=1.4)
axes[0].set(xlabel="time", ylabel="∫E_b dV", title="Background potential energy")
axes[1].set(xlabel="time", ylabel="relative difference", title="Relative to the online 3D sort")
for ax in axes:
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

fig.suptitle(f"Background potential energy — online vs offline   ({label_run})")
out3 = FIGURES / f"inv06_background_pe_{stem}.png"
fig.savefig(out3, dpi=150)
print(f"Saved {out3}")
#---
