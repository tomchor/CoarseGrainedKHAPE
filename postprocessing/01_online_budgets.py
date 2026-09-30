#!/usr/bin/env python
"""
Assemble the SFS KE and APE budget files from the terms the simulation computes online.

Every term of both sub-filter budgets is written by `kelvin_helmholtz_instability.jl` at each of its
`--filter_ls` scales: Kˢ, ∂ₜKˢ, Π_K, ε_Kˢ, τ(w,b_r) for the KE budget and S̃, ∂ₜS̃, Π_A, ε_Aˢ, Rˢ (with the
same τ) for the APE one, each as a 3D field and as a volume integral. This script reads them and writes
the four budget files the plotting scripts and the tests consume, under the variable names the offline
pipeline used to write, so nothing downstream knows which pipeline produced them:

    <stem>_sfs_ke_budget_fields.nc    <stem>_sfs_ke_budget_integrated.nc
    <stem>_sfs_ape_budget_fields.nc   <stem>_sfs_ape_budget_integrated.nc

The fields are on the simulation's own grid (no z padding, `n_pad_z = 0`), and only the records whose
`TimeDerivative` spans a time step are kept: every record but the first (iteration 0, where the derivative has
had one evaluation and reads zero) or, when the simulation ran with `--offline_check` and wrote consecutive-
iteration pairs, the upper record of each pair, whose derivative is the single-step difference across it, the
same difference the offline pipeline forms, so the budget is stated at the same instants. The offline pipeline
itself lives in `offline/` and runs only as the CI cross-check (`pytest --offline-check`).
"""
#+++ Imports
import os
import re
from pathlib import Path
import numpy as np
import xarray as xr
from dask.diagnostics.progress import ProgressBar
from src.aux00_utils import PP_OUTPUT, load_dataset_and_grid, upper_records
#---

#+++ Configuration
import argparse
parser = argparse.ArgumentParser(description="Assemble the SFS KE and APE budgets from the simulation's online terms")
parser.add_argument("--filename", default="output/khi_Nz1024_Ri0.10.nc", help="Path to simulation NetCDF file")
parser.add_argument("--filter-scales", type=float, nargs="+", default=None,
                    help="Filter scales ℓ to assemble; each must be among the simulation's online --filter_ls (default: all of them)")
parser.add_argument("--records", choices=["differenced", "all"], default="differenced",
                    help="Keep the records whose TimeDerivative spans a time step (default: the upper member of each pair of an "
                         "--offline_check run, else every record but the first), or every record")
args = parser.parse_args()
print("\n" + "="*70 + f"\n  {Path(__file__).name}\n  " + "  ".join(f"{k}={v}" for k, v in vars(args).items()) + "\n" + "="*70)
REPO_ROOT = Path(__file__).resolve().parent.parent
filename = str(REPO_ROOT / args.filename) if not os.path.isabs(args.filename) else args.filename
stem = Path(filename).stem
#---

#+++ Online variable names
# The Julia writer names each term Symbol("<var>_ℓ$(ℓ)"), with ℓ printed as an Int when it is one.
def online_tag(ℓ):
    return f"{int(ℓ)}" if float(ℓ) == int(ℓ) else f"{ℓ}"

def online_name(var, ℓ, suffix=""):
    return f"{var}_ℓ{online_tag(ℓ)}{suffix}"

# (budget variable name, online variable, sign). The integrals carry the sign the budget gives the term.
KE_FIELDS = [("KE_of_sfs_flow",       "K_s",    +1),
             ("∂ₜ SFS KE",            "dKs_dt", +1),
             ("Π_K",                  "Π_K",    +1),
             ("ε_Kˢ",                 "ε_Ks",   +1),
             ("SFS APE->KE exchange", "wb_rs",  +1)]
KE_INTEGRALS = [("∫-∂ₜ SFS KE dV",    "dKs_dt", -1),
                ("∫Π_K dV",           "Π_K",    +1),
                ("∫-ε_Kˢ dV",         "ε_Ks",   -1),
                ("∫(SFS APE->KE) dV", "wb_rs",  +1)]
APE_FIELDS = [("Ea(ρ, z)",             "E_a",     +1),   # scale-independent, written once
              ("Ea(ρ̄, z)",             "L",       +1),   # L̃, the resolved reservoir against ⟨b✶⟩
              ("Ēa(ρ, z)",             "Ea_flt",  +1),   # Ē_A, the filtered full-field APE
              ("Eaˢ(ρ, z)",            "E_as",    +1),   # S̃ = Ē_A - L̃
              ("∂ₜ SFS APE",           "dEas_dt", +1),
              ("Π_A",                  "Π_A",     +1),
              ("ε_Aˢ",                 "ε_As",    +1),
              ("SFS KE->APE exchange", "wb_rs",   -1),
              ("Rˢ",                   "R_s",     +1)]
APE_INTEGRALS = [("∫-∂ₜ SFS APE dV",  "dEas_dt", -1),
                 ("∫Π_A dV",          "Π_A",     +1),
                 ("∫-ε_Aˢ dV",        "ε_As",    -1),
                 ("∫(SFS KE->APE) dV", "wb_rs",  -1),
                 ("∫Rˢ dV",           "R_s",     +1)]
SCALE_INDEPENDENT = {"E_a"}
# Written by the simulation since the budget went online; a file from before then lacks them, and the
# budget is assembled without the two halves of S̃ (the tests that read them skip).
OPTIONAL = {"E_a", "L", "Ea_flt"}
#---

#+++ Load the online output on its own grid
print("\n" + "="*60)
print("Loading the simulation output (unpadded)...")
ds = load_dataset_and_grid(filename, pad=False).chunk({"time": 1})
time = ds.time.values

if args.records == "differenced":
    # A TimeDerivative reads zero until its operand has been evaluated twice, so iteration 0's record states no
    # budget. With --offline_check the output comes in ConsecutiveIterations pairs, (tⁿ, tⁿ⁺¹) at every
    # TimeInterval, and the second of each is kept, whose derivative spans the pair; otherwise every record but the first.
    is_upper = upper_records(time)
    if is_upper.any():
        print("  Paired output (--offline_check): keeping the upper record of each consecutive-iteration pair")
        ds = ds.isel(time=np.where(is_upper)[0])
    else:
        print("  Unpaired output: keeping every record but the first")
        ds = ds.isel(time=slice(1, None))
print(f"  {len(ds.time)} records kept out of {len(time)}")

online_scales = sorted({float(m.group(1)) for v in ds.data_vars
                        for m in [re.fullmatch(r"Π_K_ℓ(.+?)(?:_int)?", str(v))] if m and not str(v).endswith("_int")})
filter_scales = online_scales if args.filter_scales is None else [float(ℓ) for ℓ in args.filter_scales]
missing_scales = [ℓ for ℓ in filter_scales if ℓ not in online_scales]
if missing_scales:
    raise KeyError(f"Online terms for ℓ={missing_scales} not in {filename}: the simulation's --filter_ls were "
                   f"{online_scales}. The budget scales must be among them.")
print(f"  Filter scales: {filter_scales}  (online: {online_scales})")
#---

#+++ Assemble one dataset per scale, then concatenate along filter_scale
def take(var, ℓ, sign, integral=False):
    """The online term `var` at scale ℓ (or the scale-independent one), with the budget's sign, or None if absent."""
    name = var if var in SCALE_INDEPENDENT else online_name(var, ℓ, "_int" if integral else "")
    if name not in ds:
        if var in OPTIONAL and not integral:
            print(f"  note: '{name}' not in the simulation output; the budget is assembled without it")
            return None
        raise KeyError(f"Online term '{name}' not in {filename}; rerun the simulation with the current "
                       f"kelvin_helmholtz_instability.jl (every budget term is written unconditionally).")
    da = ds[name]
    if not integral:
        da = da.transpose("time", "y_aca", "x_caa", "z_aac")
    return sign * da if sign != 1 else da


def assemble(fields, integrals, residual_name):
    per_scale = []
    for ℓ in filter_scales:
        terms = {}
        for out_name, var, sign in fields:
            da = take(var, ℓ, sign)
            if da is not None:
                terms[out_name] = da
        ints = {out_name: take(var, ℓ, sign, integral=True) for out_name, var, sign in integrals}
        ints[residual_name] = sum(ints.values())
        per_scale.append(xr.Dataset({**terms, **ints}))
    scale = xr.DataArray(filter_scales, dims="filter_scale", name="filter_scale")
    budget = xr.concat(per_scale, dim=scale)
    budget.attrs.update({k: v for k, v in ds.attrs.items()})
    budget.attrs.update(ape_reference="filtered", budget_source="online", online_records=args.records,
                        n_pad_z=0, z_extension="none")
    return budget


def save(budget, kind):
    integrated_vars = [v for v in budget.data_vars if v.startswith("∫") or "residual" in v]
    local_vars      = [v for v in budget.data_vars if v not in integrated_vars]
    fields_filename     = str(PP_OUTPUT / f"{stem}_sfs_{kind}_budget_fields.nc")
    integrated_filename = str(PP_OUTPUT / f"{stem}_sfs_{kind}_budget_integrated.nc")
    print(f"  Saving {kind.upper()} integrated time series → {integrated_filename}")
    budget[integrated_vars].load().to_netcdf(integrated_filename)
    print(f"  Saving {kind.upper()} local fields → {fields_filename}")
    with ProgressBar(minimum=5, dt=5):
        budget[local_vars].to_netcdf(fields_filename)


print("\n" + "="*60)
print("Assembling the SFS KE budget...")
save(assemble(KE_FIELDS, KE_INTEGRALS, "residual_K"), "ke")
print("\n" + "="*60)
print("Assembling the SFS APE budget...")
save(assemble(APE_FIELDS, APE_INTEGRALS, "residual_A"), "ape")
print("\nDone!")
#---
