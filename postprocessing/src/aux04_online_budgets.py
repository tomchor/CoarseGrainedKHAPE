"""
The SFS KE and APE budgets, assembled from the terms the simulation computes online.

`kelvin_helmholtz_instability.jl` writes every term of both sub-filter budgets at each of its `--filter_ls` scales, as
a 3D field and as a volume integral, under per-scale names (`Π_K_ℓ7`, `Π_K_ℓ7_int`, ...). `online_budgets` turns a
loaded run into the two budgets under the variable names the offline pipeline used to write, with the scales stacked
along a `filter_scale` dimension, the budget's sign on every integral and the residual formed, so nothing downstream
knows which pipeline produced them. It is lazy (dask), so a caller that wants one slice pays for one slice.

`01_online_budgets.py` writes the integrals it returns to the two integrated budget files that the tests,
`02_plot_budgets.py` and `anim1_panels.py` read. The 3D terms are never copied: the tests, `plot5_budgets.py`,
`plot6_panels.py`, `X2_panels.py` and `X4_thumbnail.py` call it on the simulation file directly, and `anim1_panels.py`
on the x–z slices of the `_2d.nc` file.
"""
#+++ Imports
import re
import xarray as xr
from src.aux00_utils import output_flag
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

#+++ Records and scales
def differenced_records(ds):
    """The records whose TimeDerivative spans a time step.

    A TimeDerivative reads zero until its operand has been evaluated twice, so iteration 0's record states no budget.
    With --offline_check (the file's `offline_check` attribute) the output comes in ConsecutiveIterations pairs,
    (tⁿ, tⁿ⁺¹) at every TimeInterval, and the second of each is kept, whose derivative spans the pair; otherwise every
    record but the first.
    """
    paired = output_flag(ds, "offline_check")
    print("  " + ("Paired output (offline_check=1): keeping the upper record of each consecutive-iteration pair" if paired
                  else "Unpaired output (offline_check=0): keeping every record but the first"))
    return ds.isel(time=slice(1, None, 2 if paired else 1))


def online_filter_scales(ds):
    """The filter scales the simulation wrote its budget terms at, read off the `Π_K_ℓ<ℓ>` names."""
    return sorted({float(m.group(1)) for v in ds.data_vars
                   for m in [re.fullmatch(r"Π_K_ℓ(.+?)(?:_int)?", str(v))] if m and not str(v).endswith("_int")})
#---

#+++ Assembly
def _take(ds, var, ℓ, sign, integral=False):
    """The online term `var` at scale ℓ (or the scale-independent one), with the budget's sign, or None if absent."""
    name = var if var in SCALE_INDEPENDENT else online_name(var, ℓ, "_int" if integral else "")
    if name not in ds:
        if var in OPTIONAL and not integral:
            print(f"  note: '{name}' not in this file; the budget is assembled without it")
            return None
        raise KeyError(f"Online term '{name}' not in the simulation output; rerun the simulation with the current "
                       f"kelvin_helmholtz_instability.jl (every budget term is written unconditionally).")
    da = ds[name]
    if not integral:
        da = da.transpose("time", "y_aca", "x_caa", "z_aac")
    return sign * da if sign != 1 else da


def _assemble(ds, fields, integrals, residual_name, filter_scales, records):
    per_scale = []
    for ℓ in filter_scales:
        terms = {}
        for out_name, var, sign in fields:
            da = _take(ds, var, ℓ, sign)
            if da is not None:
                terms[out_name] = da
        ints = {out_name: _take(ds, var, ℓ, sign, integral=True) for out_name, var, sign in integrals}
        ints[residual_name] = sum(ints.values())
        per_scale.append(xr.Dataset({**terms, **ints}))
    scale = xr.DataArray(filter_scales, dims="filter_scale", name="filter_scale")
    budget = xr.concat(per_scale, dim=scale)
    budget.attrs.update({k: v for k, v in ds.attrs.items()})
    budget.attrs.update(ape_reference="filtered", budget_source="online", online_records=records, n_pad_z=0, z_extension="none")
    return budget


def online_budgets(ds, filter_scales=None, records="differenced"):
    """The SFS KE and APE budgets of a run, as two lazy Datasets holding the 3D terms and the integrals together.

    `ds` is the simulation output as `load_dataset_and_grid(filename, pad=False)` returns it. `filter_scales` must be
    among the scales the simulation wrote (default: all of them). `records` keeps the differenced records (default,
    see `differenced_records`) or "all" of them. `integrated_variables` names the integrals and the residual.

    The `_2d.nc` slice file works too, with `records="all"`: it holds the same terms on one x–z plane (bar `E_a` and
    the two halves of S̃, which only the 3D writer has), and its records are never paired, whatever `offline_check` says.
    """
    if records not in ("differenced", "all"):
        raise ValueError(f"records must be 'differenced' or 'all', not {records!r}")
    if records == "differenced":
        ds = differenced_records(ds)
    online_scales = online_filter_scales(ds)
    filter_scales = online_scales if filter_scales is None else [float(ℓ) for ℓ in filter_scales]
    missing = [ℓ for ℓ in filter_scales if ℓ not in online_scales]
    if missing:
        raise KeyError(f"Online terms for ℓ={missing} not in the simulation output: its --filter_ls were {online_scales}. "
                       "The budget scales must be among them.")
    print(f"  Filter scales: {filter_scales}  (online: {online_scales})")
    ke  = _assemble(ds, KE_FIELDS,  KE_INTEGRALS,  "residual_K", filter_scales, records)
    ape = _assemble(ds, APE_FIELDS, APE_INTEGRALS, "residual_A", filter_scales, records)
    return ke, ape


def integrated_variables(budget):
    """The names of the volume integrals and the residual: the content of the `_integrated.nc` files."""
    return [v for v in budget.data_vars if v.startswith("∫") or "residual" in v]
#---
