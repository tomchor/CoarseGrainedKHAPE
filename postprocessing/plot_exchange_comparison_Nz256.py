"""
Compare the SFS APE->KE exchange term as computed on `main` vs `tc/test-bl`.

The two branches differ only in the `filtered_b` argument of the exchange:
    main       : filtered_b = filter(b_r)            = -(g/ρ0)(ρ̄ - <ρ_ref>)
    tc/test-bl : filtered_b = b_r_l                  = -(g/ρ0)(ρ̄ -  ρ_ref )
Everything else (w, b_r, w̄, the Gaussian filter) is identical, so both are
computed here from the SAME Nz=256 simulation output.

Outputs (kept, not deleted):
  output/khi_Nz256_Ri0.10_exchange_comparison.nc      -- fields + integrals
  output/exchange_comparison_Nz256_snapshot.png       -- x-z snapshot: main | tc/test-bl | diff
  output/exchange_comparison_Nz256_integrated.png     -- ∫exchange dV time series
"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent   # the postprocessing/ directory
REPO = HERE.parent                       # repository root
sys.path.insert(0, str(HERE))
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.aux00_utils import load_dataset_and_grid, condense_uw_velocities, integrate, make_gaussian_filter
from src.aux01_pe_functions import (calculate_density_fields_from_buoyancy, sorted_timeseries,
                                    calculate_b_r, calculate_ape_to_ke_exchange_term)

filename = str(REPO / "output" / "khi_Nz256_Ri0.10.nc")
OUTDIR = HERE / "output"
OUTDIR.mkdir(parents=True, exist_ok=True)
filtered_dimensions = ["x_caa", "z_aac"]
scales = [1.0, 7.0]
times = [10, 30, 50]          # times to analyse; everything downstream is computed only for these
OUT = str(OUTDIR / "khi_Nz256_Ri0.10_exchange_comparison.nc")

#+++ Load, pick times, then compute (only the selected times are ever processed)
ds = load_dataset_and_grid(filename).chunk({"time": 1})
ds = condense_uw_velocities(ds, indices=[1, 3])
ds = ds.sel(time=times, method="nearest")
print("selected times:", list(np.round(ds.time.values, 2)))
print("z_aac: n=%d  range=[%.2f, %.2f]" % (ds.sizes["z_aac"], float(ds.z_aac.min()), float(ds.z_aac.max())))

ds_for_sort = ds[["b", "dV", "LxLy"]].copy()
ds_for_sort = calculate_density_fields_from_buoyancy(ds_for_sort, buoyancy_name="b", density_name="ρ")
print("Sorting reference density (%d timesteps)..." % ds.sizes["time"])
rho_sorted = sorted_timeseries(ds_for_sort, field_to_sort="ρ", n_workers=4, verbose_level=0).rho_sorted

ds_full = ds[["b", "dV", "uᵢ"]].copy()
ds_full = calculate_density_fields_from_buoyancy(ds_full, buoyancy_name="b", density_name="ρ")
b_r    = calculate_b_r(ds_full.ρ, rho_sorted)
w_full = ds_full["uᵢ"].sel(i=3)
dV     = ds_full.dV
#---

#+++ Compute both exchange definitions at each scale
data = {}
for ℓ in scales:
    print(f"--- ℓ = {ℓ} ---")
    gf      = make_gaussian_filter(ℓ, ds)
    w_bar   = gf.apply(w_full, dims=filtered_dimensions)
    b_r_bar = gf.apply(b_r, dims=filtered_dimensions)                                    # main filtered_b
    b_r_l   = calculate_b_r(gf.apply(ds_full.ρ, dims=filtered_dimensions), rho_sorted)   # tc/test-bl filtered_b

    exch_main = calculate_ape_to_ke_exchange_term(w_full, b_r, gf, filter_dims=filtered_dimensions,
                                                  filtered_w=w_bar, filtered_b=b_r_bar).compute()
    exch_test = calculate_ape_to_ke_exchange_term(w_full, b_r, gf, filter_dims=filtered_dimensions,
                                                  filtered_w=w_bar, filtered_b=b_r_l).compute()
    data[ℓ] = dict(main=exch_main.squeeze("y_aca", drop=True), test=exch_test.squeeze("y_aca", drop=True),
                   int_main=integrate(exch_main, dV).compute(),
                   int_test=integrate(exch_test, dV).compute())
#---

#+++ Save fields + integrals
out = xr.Dataset()
for ℓ in scales:
    tag = f"l{ℓ:.0f}"
    out[f"exchange_main_{tag}"] = data[ℓ]["main"]
    out[f"exchange_testbl_{tag}"] = data[ℓ]["test"]
    out[f"int_exchange_main_{tag}"] = data[ℓ]["int_main"]
    out[f"int_exchange_testbl_{tag}"] = data[ℓ]["int_test"]
out.attrs["description"] = "SFS APE->KE exchange: main=filter(b_r), tc/test-bl=b_r_l"
out.to_netcdf(OUT)
print(f"saved {OUT}")
#---

#+++ Snapshot figures: one per selected time (scales × main | tc/test-bl | difference)
zsel = dict(z_aac=slice(-12.5, 12.5))
for time in times:
    tval = float(data[1.0]["main"].time.sel(time=time, method="nearest"))
    fig, axs = plt.subplots(len(scales), 3, figsize=(13, 3.4 * len(scales)), constrained_layout=True)
    for r, ℓ in enumerate(scales):
        m = data[ℓ]["main"].sel(time=time, method="nearest").sel(**zsel)
        t = data[ℓ]["test"].sel(time=time, method="nearest").sel(**zsel)
        d = t - m
        X = m.x_caa.values; Z = m.z_aac.values
        vmax = float(np.nanpercentile(np.abs(np.concatenate([m.values.ravel(), t.values.ravel()])), 99))
        dmax = float(np.nanpercentile(np.abs(d.values), 99)) or vmax
        for c, (field, title, vm) in enumerate([(m, "main:  filter(b_r)", vmax),
                                                (t, "tc/test-bl:  b_r_l", vmax),
                                                (d, "difference (test − main)", dmax)]):
            ax = axs[r, c]
            im = ax.pcolormesh(X, Z, field.T, cmap="RdBu_r", vmin=-vm, vmax=vm, shading="auto", rasterized=True)
            fig.colorbar(im, ax=ax, shrink=0.9)
            ax.set_title(f"ℓ={ℓ:.0f}   {title}", fontsize=10)
            ax.set_xlabel("x"); ax.set_ylabel("z" if c == 0 else "")
    fig.suptitle(f"SFS APE→KE exchange term, Nz=256, t={tval:.1f}", fontsize=12)
    fname = f"{OUTDIR}/exchange_comparison_Nz256_snapshot_t{int(round(tval))}.png"
    fig.savefig(fname, dpi=140)
    print("saved", fname)
#---

#+++ Integrated time series: shows the volume integral is (nearly) identical
fig2, axs2 = plt.subplots(1, len(scales), figsize=(11, 4), constrained_layout=True)
for c, ℓ in enumerate(scales):
    ax = axs2[c]
    tt = data[ℓ]["int_main"].time.values
    ax.plot(tt, data[ℓ]["int_main"].values, "-o",  lw=2,        label="main: filter(b_r)")
    ax.plot(tt, data[ℓ]["int_test"].values, "--x", lw=2, ms=10, label="tc/test-bl: b_r_l")
    ax.set_title(f"ℓ={ℓ:.0f}"); ax.set_xlabel("time"); ax.axhline(0, color="k", lw=0.5)
    if c == 0: ax.set_ylabel("∫ exchange dV")
    ax.legend(fontsize=8)
fig2.suptitle("Volume-integrated SFS APE→KE exchange (Nz=256)", fontsize=12)
fig2.savefig(f"{OUTDIR}/exchange_comparison_Nz256_integrated.png", dpi=140)
print("saved output/exchange_comparison_Nz256_integrated.png")

# quick numeric summary
for ℓ in scales:
    a = data[ℓ]["int_main"].values; b = data[ℓ]["int_test"].values
    loc = ((data[ℓ]["test"] - data[ℓ]["main"]) ** 2).mean() ** 0.5
    print(f"ℓ={ℓ:.0f}:  rms(∫main)={np.sqrt(np.mean(a**2)):.3e}  "
          f"rms(∫diff)={np.sqrt(np.mean((a-b)**2)):.3e}  rms(local diff)={float(loc):.3e}")
#---
