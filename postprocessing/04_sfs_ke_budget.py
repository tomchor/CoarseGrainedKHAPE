#!/usr/bin/env python
#+++ Imports
import os
from pathlib import Path
import xarray as xr
from src.aux00_utils import load_dataset_and_grid, condense_velocities, integrate, make_gaussian_filter, write_dataset
from src.aux01_pe_functions import calculate_density_fields_from_buoyancy, calculate_b_r, calculate_b_r_simple, calculate_ape_to_ke_exchange_term
from src.aux02_ke_functions import (
    calculate_sfs_stress_tensor,
    calculate_sfs_ke_tendency,
)
#---

#+++ Configuration
import argparse
parser = argparse.ArgumentParser(description="Calculate SFS KE budget from baroclinic adjustment simulation output")
parser.add_argument("--filename", default="output/bci_Nx48_Ny48_Nz8.nc", help="Path to simulation NetCDF file")
parser.add_argument("--fixed-reference", action="store_true", default=False, help="Load the fixed-in-time reference profile (produced by 01 with --fixed-reference)")
parser.add_argument("--write-mode", choices=["load", "synchronous"], default="load",
    help="How to avoid the dask-lazy .to_netcdf() write hang -- see write_dataset() in aux00_utils.py for "
         "what each mode does and the measured cost of 'synchronous' relative to 'load'.")
args = parser.parse_args()
print("\n" + "="*70 + f"\n  {Path(__file__).name}\n  " + "  ".join(f"{k}={v}" for k,v in vars(args).items()) + "\n" + "="*70)
REPO_ROOT = Path(__file__).resolve().parent.parent
PP_OUTPUT = REPO_ROOT / "postprocessing" / "output"
filename = str(REPO_ROOT / args.filename) if not os.path.isabs(args.filename) else args.filename
fixed_reference = args.fixed_reference
#---

#+++ Load data and grid
print("\n" + "="*60)
print("Loading data and grid...")
ds = load_dataset_and_grid(filename)
ds = ds.chunk({"time": 1})
print(f"Dataset loaded: {len(ds.time)} time steps")

# --implicit (baroclinic_adjustment.jl) means closure=nothing: the online ε_Kˢ read below is computed from
# the model's explicit viscous fluxes, which are identically zero here -- real dissipation is happening via
# WENO's own implicit numerics, invisible to that diagnostic. The domain-integrated ε_Kˢ is replaced below
# with a residual-based estimate (solves the budget for ε_Kˢ assuming every other term is correct); the
# local (spatial) ε_Kˢ field is left as-is (still the near-zero, uninformative online value) since a local
# residual would also absorb real spatial transport/flux-divergence terms that only vanish upon domain
# integration, not pointwise -- there's no principled way to attribute those to "dissipation" at a single
# grid cell. See 05_sfs_ape_budget.py for the same treatment on the APE side.
implicit = bool(ds.attrs.get("implicit", 0))
if implicit:
    print("  implicit=True (from simulation attrs): domain-integrated ε_Kˢ will be replaced with a "
          "residual-based estimate; the local (spatial) ε_Kˢ field stays at its uninformative online value")

# --bottom_drag (baroclinic_adjustment.jl): adds a quadratic bottom-drag sink to the SFS KE budget.
# τ̄_x,ℓ/τ̄_y,ℓ (filtered bottom stress) and overline{τ·u_b}_ℓ (filtered pointwise drag work) are computed
# online (same gaussian_filters as Πₖ/ε_Kˢ) and written to a separate _bottom.nc file (indices=(:,:,1) --
# only the bottom z-level is physically meaningful, matching the existing _surface.nc convention).
bottom_drag = bool(ds.attrs.get("bottom_drag", 0))
if bottom_drag:
    bottom_filename = str(Path(filename).parent / (Path(filename).stem + "_bottom.nc"))
    ds_bottom = xr.open_dataset(bottom_filename, decode_times=False).chunk({"time": 1}).isel(z_aac=0)
    print(f"  bottom_drag=True (from simulation attrs): loaded bottom-boundary fields from {bottom_filename}")
#---

#+++ Load filtered fields
print("\n" + "="*60)
print("Loading pre-filtered fields...")

filtered_filename = str(PP_OUTPUT / (Path(filename).stem + "_filtered_velocities.nc"))
ds_filt = xr.open_dataset(filtered_filename, decode_times=False).chunk({"time": 1})
filtered_dimensions = ["x_caa", "y_aca"]
filter_scales = ds_filt.filter_scale.values

ds = condense_velocities(ds, indices=(1, 2, 3))
ds_full = ds[["b", "dV", "uᵢ"]].copy()

ref_suffix = "_fixed_ref" if fixed_reference else ""
sorted_density_filename = str(PP_OUTPUT / (Path(filename).stem + f"_sorted_density{ref_suffix}.nc"))
ds_sorted = xr.open_dataset(sorted_density_filename, decode_times=False).chunk({"time": 1})

print(f"Pre-filtered fields loaded from: {filtered_filename}")
print(f"Sorted density loaded from: {sorted_density_filename}")
print(f"Filter length scales: {filter_scales}")
print(f"Filter dimensions: x and y (horizontal)")
#---

#+++ Calculate density and relative buoyancy [scale-independent]
print("\n" + "="*60)
print("Calculating density and relative buoyancy...")
ds_full = calculate_density_fields_from_buoyancy(ds_full, buoyancy_name="b", density_name="ρ")
b_r = calculate_b_r(ds_full.ρ, ds_sorted.rho_sorted)
print("Done!")
#---

#+++ Loop over filter scales and calculate budget terms
print("\n" + "="*60)
print("Calculating budget terms for each filter scale...")

# Π_K and ε_Kˢ are both computed online by the simulation now (KineticEnergyCrossScaleFlux,
# SubFilterKineticEnergyDissipationRate -- see baroclinic_adjustment.jl), read straight from `ds` below
# rather than computed offline. Both are reference-independent (they don't depend on the density sort),
# so the same values serve the time-varying and fixed-reference budgets.
dV = ds_full.dV
dA = ds.Δx_caa * ds.Δy_aca  # for the bottom-drag work terms, which are boundary (area), not volume, integrals -- ds_full is a subset (b, dV, uᵢ only) that drops Δx_caa/Δy_aca
budget_list = []

for ℓ in filter_scales:
    print(f"\n--- filter_scale = {ℓ:.4f} ---")

    gaussian_filter = make_gaussian_filter(ℓ, ds)

    ds_filt_ℓ = ds_filt.sel(filter_scale=ℓ)
    u_i_full     = ds_full["uᵢ"].sel(i=[1, 2, 3])
    u_i_bar_full = ds_filt_ℓ["ūᵢ"].sel(i=[1, 2, 3])

    # τⁱʲ = filter(uⁱ uʲ) - ūⁱ ūʲ   shape: (i, j, time, z, y, x) -- full 3D (i,j ∈ {1,2,3}): w is a
    # genuine prognostic variable in this NonhydrostaticModel, with its own dissipative dynamics, so
    # there's no reason to exclude it from the SFS KE density the way the old HydrostaticFreeSurfaceModel
    # setup had to (see baroclinic_adjustment.jl and CLAUDE.md's Architecture notes).
    print("  SFS stress tensor...")
    sfs_stress_tensor = calculate_sfs_stress_tensor(u_i_full, gaussian_filter,
                                                    filter_dims=filtered_dimensions,
                                                    filtered_u_i=u_i_bar_full)
    i_vals = sfs_stress_tensor.coords["i"].values
    sfs_stress_tensor_trace = sum(sfs_stress_tensor.sel(i=k, j=k) for k in i_vals)
    sfs_ke_density = sfs_stress_tensor_trace / 2

    print("  APE->KE exchange term...")
    b_r_filt = gaussian_filter.apply(b_r, dims=filtered_dimensions)
    ape_to_ke_exchange = calculate_ape_to_ke_exchange_term(
        ds_full["uᵢ"].sel(i=3), # full w
        b_r,                    # relative buoyancy
        gaussian_filter,
        filter_dims=filtered_dimensions,
        filtered_w=ds_filt_ℓ["ūᵢ"].sel(i=3),
        filtered_b=b_r_filt,
    )

    # ∂KE_s/∂t   centred finite difference, staggered time grid
    dKE_dt = calculate_sfs_ke_tendency(sfs_ke_density)

    int_dKE_dt             = integrate(dKE_dt, dV)
    int_ape_to_ke_exchange = integrate(ape_to_ke_exchange.reindex(time=dKE_dt.time), dV)

    print("  Π_K and ε_Kˢ (computed online by the simulation)...")
    ℓ_km = int(round(ℓ / 1000))
    Π_K_ℓ = ds[f"Π_K_ℓ{ℓ_km}km"]
    sfs_ke_dissipation = ds[f"ε_Kˢ_ℓ{ℓ_km}km"]
    int_Π_K_ℓ              = integrate(Π_K_ℓ, dV)
    int_sfs_ke_dissipation = integrate(sfs_ke_dissipation, dV)

    if bottom_drag:
        # Large-scale bottom drag work: -(τ̄·ū_b). SFS bottom drag work: -(overline{τ·u_b} - τ̄·ū_b).
        # τ̄_x,ℓ/τ̄_y,ℓ/overline{τ·u_b}_ℓ are the online-filtered primitives (Oceanostics' GaussianFilter);
        # ū_b,ℓ/v̄_b,ℓ reuse the offline-filtered u_ℓ/v_ℓ already loaded above (ds_filt_ℓ), sliced at the
        # bottom z-level -- no new computation needed there. Mixing an online-filtered term (τ̄) with an
        # offline-filtered one (ū_b) here is the same pre-existing online/offline filter discrepancy this
        # pipeline already has elsewhere (Πₖ vs Π_A/exchange), not a new inconsistency. Computed before the
        # `if implicit:` block below (rather than after, as originally written) so that block can fold
        # int_bottom_drag_work_SFS into its residual-based ε_Kˢ estimate when both flags are active.
        print("  Bottom drag work terms...")
        τx_b_bar   = ds_bottom[f"τx_b_ℓ{ℓ_km}km"]
        τy_b_bar   = ds_bottom[f"τy_b_ℓ{ℓ_km}km"]
        τu_b_bar   = ds_bottom[f"τu_b_ℓ{ℓ_km}km"]
        u_b_bar    = ds_filt_ℓ["ūᵢ"].sel(i=1).isel(z_aac=0)
        v_b_bar    = ds_filt_ℓ["ūᵢ"].sel(i=2).isel(z_aac=0)

        bottom_drag_work_LS  = τx_b_bar * u_b_bar + τy_b_bar * v_b_bar
        bottom_drag_work_SFS = τu_b_bar.reindex(time=bottom_drag_work_LS.time) - bottom_drag_work_LS

        int_bottom_drag_work_LS  = integrate(bottom_drag_work_LS,  dA, dims=("x_caa", "y_aca"))
        int_bottom_drag_work_SFS = integrate(bottom_drag_work_SFS, dA, dims=("x_caa", "y_aca"))

    if implicit:
        # Solve the budget for ε_Kˢ assuming every other term (Πₖ, ∂ₜE_K^s, exchange, and -- when
        # --bottom_drag is also active -- the independently-diagnosed bottom-drag SFS sink) is correct.
        # Without --bottom_drag this is exactly the online (near-zero) ε_Kˢ's residual, redefined as the
        # dissipation estimate itself; with it, the bottom-drag term is a real, separately-diagnosed
        # physical sink (not numerical dissipation), so it must be subtracted out here too, or it would
        # leak into the final residual below instead of being absorbed into ε_Kˢ like everything else. The
        # residual computed just below becomes ~0 by construction either way, a sanity check that the
        # substitution was applied correctly.
        int_sfs_ke_dissipation = int_Π_K_ℓ.reindex(time=dKE_dt.time) + int_ape_to_ke_exchange - int_dKE_dt
        if bottom_drag:
            int_sfs_ke_dissipation = int_sfs_ke_dissipation - int_bottom_drag_work_SFS.reindex(time=dKE_dt.time)

    residual  = (-int_dKE_dt + int_Π_K_ℓ.reindex(time=dKE_dt.time) + int_ape_to_ke_exchange
                 - int_sfs_ke_dissipation.reindex(time=dKE_dt.time))
    if bottom_drag:
        residual = residual - int_bottom_drag_work_SFS.reindex(time=dKE_dt.time)

    budget_ℓ = xr.Dataset({
        # Local KE fields
        "KE_of_sfs_flow": sfs_ke_density,
        # Local budget terms
        "∂ₜ SFS KE": dKE_dt,
        "Π_K": Π_K_ℓ,
        "ε_Kˢ": sfs_ke_dissipation,
        "SFS APE->KE exchange": ape_to_ke_exchange,
        # Integrated budget terms
        "∫-∂ₜ SFS KE dV": -int_dKE_dt,
        "∫Π_K dV": int_Π_K_ℓ,
        "∫-ε_Kˢ dV": -int_sfs_ke_dissipation,
        "∫(SFS APE->KE) dV": int_ape_to_ke_exchange,
        "residual_K": residual,
    }).reindex(time=dKE_dt.time)

    if implicit:
        method = "residual estimate (implicit LES): Πₖ + exchange - ∂ₜE_K^s"
        if bottom_drag:
            method += " - (bottom drag work, SFS)"
        budget_ℓ["∫-ε_Kˢ dV"].attrs["method"] = method
        budget_ℓ["residual_K"].attrs["method"] = "≈0 by construction: ε_Kˢ is defined as the residual estimate (implicit LES)"

    if bottom_drag:
        # Large-scale term is a standalone diagnostic (not wired into any budget equation here -- there's
        # no full large-scale/filtered KE budget assembly in this pipeline yet). SFS term is already
        # folded into residual_K above; recorded here too for visibility/plotting.
        budget_ℓ["bottom drag work (LS)"]  = bottom_drag_work_LS
        budget_ℓ["bottom drag work (SFS)"] = bottom_drag_work_SFS
        budget_ℓ["∫-(bottom drag work, LS) dA"]  = -int_bottom_drag_work_LS
        budget_ℓ["∫-(bottom drag work, SFS) dA"] = -int_bottom_drag_work_SFS

    budget_list.append(budget_ℓ)

sfs_ke_budget_terms = xr.concat(budget_list, dim=xr.DataArray(filter_scales,
                                                              dims="filter_scale",
                                                              name="filter_scale"))
sfs_ke_budget_terms.attrs.update(ds.attrs)
print("\nDone!")
#---

#+++ Save results
print("\n" + "="*60)
print("Saving results...")

integrated_vars = [v for v in sfs_ke_budget_terms.data_vars if v.startswith("∫") or "residual" in v]
local_vars      = [v for v in sfs_ke_budget_terms.data_vars if v not in integrated_vars]

fields_filename     = str(PP_OUTPUT / (Path(filename).stem + f"_sfs_ke_budget_fields{ref_suffix}.nc"))
integrated_filename = str(PP_OUTPUT / (Path(filename).stem + f"_sfs_ke_budget_integrated{ref_suffix}.nc"))

# sfs_ke_budget_terms is still fully dask-lazy at this point (KE_of_sfs_flow, Π_K/ε_Kˢ read from the chunked
# simulation file, and every integral, none of it .load()'d yet) -- see write_dataset() in aux00_utils.py for
# why that's an issue and what --write-mode does about it. Writing each subset separately (rather than the
# whole Dataset at once) keeps peak memory the same as the two separate to_netcdf() calls already imply.
print(f"  Saving local fields (write-mode={args.write_mode})...")
write_dataset(sfs_ke_budget_terms[local_vars], fields_filename, write_mode=args.write_mode)
print(f"  Fields saved to:     {fields_filename}")

print(f"  Saving integrated timeseries (write-mode={args.write_mode})...")
write_dataset(sfs_ke_budget_terms[integrated_vars], integrated_filename, write_mode=args.write_mode)
print(f"  Integrated saved to: {integrated_filename}")
#---
