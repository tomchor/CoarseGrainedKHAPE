#!/usr/bin/env python
#+++ Imports
import logging
import os
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from src.aux00_utils import PP_OUTPUT, pad_margin_for_run, load_dataset_and_grid, check_same_padded_grid
from src.aux01_pe_functions import calculate_density_fields_from_buoyancy, calculate_b_r
from src.aux03_plotting import run_label
#---

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
print = logging.info

#+++ Configuration
import argparse
parser = argparse.ArgumentParser(description="Time-evolution snapshots: buoyancy b (top row) and relative buoyancy b_r (bottom row), one column per time")
parser.add_argument("--filename", default="output/khi_Nz2048_Ri0.10.nc", help="Path to simulation NetCDF file")
parser.add_argument("--times", type=float, nargs="+", default=[20, 50, 80], help="Snapshot times, one column each (nearest available is used)")
parser.add_argument("--fixed-reference", action="store_true", default=False, help="Use the fixed-in-time reference profile produced by 02 with --fixed-reference")
parser.add_argument("--zlim", type=float, default=4.0, help="Half-height of the plotted z window")
parser.add_argument("--clim-percentile", type=float, default=99.5, help="Percentile of |data| used to set symmetric color limits")
args = parser.parse_args()

print("\n" + "="*70 + f"\n  {Path(__file__).name}\n  " + "  ".join(f"{k}={v}" for k, v in vars(args).items()) + "\n" + "="*70)
REPO_ROOT = Path(__file__).resolve().parent.parent
FIGURES   = REPO_ROOT / "figures"
FIGURES.mkdir(exist_ok=True)
filename = str(REPO_ROOT / args.filename) if not os.path.isabs(args.filename) else args.filename
stem = Path(filename).stem
ref_suffix = "_fixed_ref" if args.fixed_reference else ""
#---

#+++ Load the simulation and the sorted reference profile
# b_r = -(g/ρ₀)(ρ - ρ_*(z)) is built against the *unfiltered* ρ_*, so unlike the resolved b_rˡ it carries
# no filter scale and no filtered-reference subtlety: one field per time, valid for any ℓ.
print("Loading simulation dataset...")
# Pad as 02 did, so ρ_*'s own z grid is the one b_r interpolates it onto.
_filtered_fn = str(PP_OUTPUT / f"{stem}_filtered_velocities.nc")
ds = load_dataset_and_grid(filename, min_margin=pad_margin_for_run(_filtered_fn, required=True))

t_sel = [float(ds.time.sel(time=t, method="nearest").values) for t in args.times]
for want, got in zip(args.times, t_sel):
    print(f"  t = {got:.3f}  (requested {want})")
if len(set(t_sel)) < len(t_sel):     # two requests on one record would duplicate the time axis
    t_sel = list(dict.fromkeys(t_sel))
    print(f"  note: some requested times share a record; plotting {len(t_sel)} distinct times")
ds = ds.sel(time=t_sel)

ds_b = ds[["b", "dV", "LxLy"]].copy()
ds_b.attrs.update(ds.attrs)
ds_b = calculate_density_fields_from_buoyancy(ds_b, buoyancy_name="b", density_name="ρ")

sorted_filename = str(PP_OUTPUT / f"{stem}_sorted_density{ref_suffix}.nc")
print(f"Loading sorted reference profile: {sorted_filename}")
ds_sorted = xr.open_dataset(sorted_filename, decode_times=False)
check_same_padded_grid(ds, ds_sorted, Path(sorted_filename).name)   # sorted on this padded grid?
rho_sorted = ds_sorted.rho_sorted.sel(time=t_sel, method="nearest")
drift = np.abs(rho_sorted.time.values - np.asarray(t_sel)).max()
if drift > 1e-6:
    print(f"  note: nearest sorted-profile times differ from the snapshot times by up to {drift:.3g}")
rho_sorted = rho_sorted.assign_coords(time=ds_b.time)   # align exactly; the offsets are reported above

print("Computing b_r...")
b_r = calculate_b_r(ds_b.ρ, rho_sorted)

zsl = slice(-args.zlim, +args.zlim)
b_fields   = [ds_b.b.sel(time=t).sel(z_aac=zsl).squeeze()  for t in t_sel]
b_r_fields = [b_r.sel(time=t).sel(z_aac=zsl).squeeze()     for t in t_sel]
print("Done.")
#---

#+++ Helper to get (x, z, data) from a 2D field with arbitrary staggered dim names
def _xzdata(field):
    x_dim = next(d for d in field.dims if "x" in d)
    z_dim = next(d for d in field.dims if "z" in d)
    return field[x_dim].values, field[z_dim].values, field.transpose(x_dim, z_dim).values.T
#---

#+++ Colour limits
# One scale per row, pooled over every column. Giving each panel its own limits would renormalise each
# snapshot and hide exactly what this figure is for -- how the field changes between the three times.
# Both quantities are signed and centred on zero, so the limits are symmetric and the midpoint neutral.
def _sym_clim(fields):
    v = max(np.nanpercentile(np.abs(_xzdata(f)[2]), args.clim_percentile) for f in fields)
    return -v, +v

b_vmin,   b_vmax   = _sym_clim(b_fields)
b_r_vmin, b_r_vmax = _sym_clim(b_r_fields)
print(f"  b   colour range: [{b_vmin:+.3e}, {b_vmax:+.3e}]")
print(f"  b_r colour range: [{b_r_vmin:+.3e}, {b_r_vmax:+.3e}]")

# Buoyancy contours overlaid on both rows, at levels fixed across every panel so the overlay is itself
# comparable between times (per-panel levels would trace different isopycnals in each column).
blevels = np.linspace(b_vmin, b_vmax, 12)
#---

#+++ Plot
print("Plotting...")
ncol = len(t_sel)
# Extra width for the row colourbars, which sit outside the panels and take their space from the figure.
fig, axes = plt.subplots(2, ncol, figsize=(5.0 * ncol + 0.9, 6.3), constrained_layout=True,
                         gridspec_kw=dict(wspace=0, hspace=0), squeeze=False)

rows = [
    (b_fields,   r"$b$ (buoyancy)",            "RdBu_r", b_vmin,   b_vmax),
    (b_r_fields, r"$b_r$ (relative buoyancy)", "PuOr_r", b_r_vmin, b_r_vmax),
]

for row, (fields, row_label, cmap, vmin, vmax) in enumerate(rows):
    im_row = None
    for col, field in enumerate(fields):
        ax = axes[row, col]
        x, z, data = _xzdata(field)
        im_row = ax.pcolormesh(x, z, data, cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True)

        bx, bz, bdata = _xzdata(b_fields[col])
        ax.contour(bx, bz, bdata, levels=blevels, colors="k", linewidths=0.6, alpha=0.5)

        # One colourbar per row: the scale is shared, so three identical bars would be three times the
        # ink for the same information. It goes in the last column, inset as in S2_panels.
        if col == 0:
            ax.text(0.5, 0.97, row_label, transform=ax.transAxes, fontsize=11, ha="center", va="top",
                    color="black", bbox=dict(facecolor="white", edgecolor="none", pad=2, alpha=0.6))
        if row == 0:
            ax.set_title(f"$t = {t_sel[col]:.0f}$", fontsize=12)

        ax.set_ylim(-args.zlim, +args.zlim)
        ax.set_aspect("equal")
        yticks = [v for v in (-3, -1, 1, 3) if abs(v) <= args.zlim]   # ticks past --zlim would widen the axis
        if yticks:
            ax.set_yticks(yticks)

    # One vertical bar per row, off the end of the row. Outside the panels it needs no backing patch and
    # cannot cover data, and spanning the row states plainly that the scale is shared along it.
    cb = fig.colorbar(im_row, ax=list(axes[row, :]), location="right", extend="both",
                      fraction=0.018, pad=0.01, aspect=22)
    cb.locator = MaxNLocator(nbins=5)
    cb.update_ticks()
    cb.ax.tick_params(labelsize=9)

for row in range(2):
    axes[row, 0].set_ylabel("z")
    for col in range(1, ncol):
        axes[row, col].set_ylabel("")
        axes[row, col].tick_params(labelleft=False, left=False)

for col in range(ncol):
    axes[0, col].set_xlabel("")
    axes[0, col].tick_params(labelbottom=False, bottom=False)
    axes[1, col].set_xlabel("x")

for ax, letter in zip(axes.flat, "abcdefghijkl"):
    ax.text(0.02, 0.97, f"({letter})", transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top", ha="left",
            bbox=dict(facecolor="white", edgecolor="none", pad=1.5))

label = run_label(ds.attrs)
if label:
    axes[1, -1].text(0.98, 0.04, label, transform=axes[1, -1].transAxes, fontsize=10, ha="right", va="bottom",
                     bbox=dict(facecolor="white", edgecolor="none", pad=2, alpha=0.85))

t_tag = "-".join(f"{t:.0f}" for t in t_sel)
# PDF, as the other paper figures (S3, plot2). The pcolormesh layers are rasterized above, so the file
# stays small while the contours, axes and text remain vector.
outfile = str(FIGURES / f"{stem}_b_br_snapshots_t{t_tag}{ref_suffix}.pdf")
fig.savefig(outfile, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Figure saved to: {outfile}")
#---
