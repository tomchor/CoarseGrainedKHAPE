"""
The frozen reference column belongs to the grid it was sorted on.

`sorted_timeseries` returns a column with one equal-volume slot per cell of the field it sorted, and
z✶ is read straight off that column's own heights. A column is therefore only meaningful on the grid
it came from — and the budget pipeline and the sweep do not share a grid. `required_pad_margin` sizes
the z padding to 4σ of the widest filter in the run, so `01_filter_fields.py` (ℓ ≤ 7) and
`sweep1_filter_fields.py` (ℓ up to 20) pad to different heights, by roughly 2.9x.

`sweep2_energy_transfer.py --fixed-reference` used to read `02_sort_density.py`'s column, which is
built on the budget padding. The two can never agree, so the guard added to catch the mismatch fired
unconditionally and every fixed-reference sweep aborted at the last stage of the chain. The fix is for
sweep2 to build its own column on the grid it loaded; these tests pin the properties that makes
correct, and the reason the old cross-pipeline read could not be.

Self-contained: no simulation or post-processing output.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent.parent / "postprocessing"))
from src.aux00_utils import _pad_domain_in_z, required_pad_margin, make_gaussian_filter, sweep_halo
from src.aux01_pe_functions import sorted_timeseries
from src.aux02_ke_functions import calculate_energy_transfer

sys.path.insert(0, str(Path(__file__).parent))
from test_jensen import make_dataset

#+++ The two shipped filter-scale sets
# Kept in step with `01_filter_fields.py` (--filter-scales default) and the hardcoded sweep in
# `sweep1_filter_fields.py`. If either changes, the margins below move with it; the tests assert the
# *relationship* between them, not absolute numbers, so only a change that makes the two sets span the
# same widest scale would need attention here -- and that is exactly the case the first test guards.
BUDGET_SCALES = [1, 7]
SWEEP_SCALES = np.geomspace(0.02, 20, 30)

# The production domain height, so the Nz//2 floor sits where it really does relative to the margins.
LZ = 25.0
#---


#+++ Fixtures
def _grid_dataset(n_times=1, Nx=16, Nz=64, Lx=4.0, Lz=LZ, h=2.0):
    """`make_dataset` plus the spacing variables `_pad_domain_in_z` needs to rebuild dV.

    y is the singleton axis the runs use, so Δy = 1 and dV = Δx·Δz, matching what `make_dataset` writes.
    """
    ds, dx, dz = make_dataset(Nx=Nx, Nz=Nz, Lx=Lx, Lz=Lz, h=h)
    if n_times > 1:
        # Distinct fields per time, so a reference frozen at t=0 is visibly *not* the sort of a later one.
        frames = [ds.isel(time=[0]).assign(ρ=lambda d, k=k: d.ρ + 0.01 * k * np.sin(3 * d.x_caa))
                  for k in range(n_times)]
        for k, f in enumerate(frames):
            f["time"] = [float(k)]
        # data_vars="minimal": only ρ carries a time axis. dV and LxLy are grid quantities and must not
        # gain one (`_pad_domain_in_z` rebuilds dV anyway, but the intent should not depend on that).
        ds = xr.concat(frames, dim="time", data_vars="minimal")
        ds.attrs = frames[0].attrs
    ds["Δx_caa"] = xr.DataArray(np.full(ds.sizes["x_caa"], dx), dims=["x_caa"], coords={"x_caa": ds.x_caa})
    ds["Δy_aca"] = xr.DataArray(np.full(ds.sizes["y_aca"], 1.0), dims=["y_aca"], coords={"y_aca": ds.y_aca})
    ds["Δz_aac"] = xr.DataArray(np.full(ds.sizes["z_aac"], dz), dims=["z_aac"], coords={"z_aac": ds.z_aac})
    return ds


def _frozen_column(ds):
    """The frozen reference column, built the way `sweep2_energy_transfer.py --fixed-reference` builds it."""
    return sorted_timeseries(ds, field_to_sort="ρ", n_workers=1, fixed_reference=True, verbose_level=0)
#---


#+++ Tests
def test_budget_and_sweep_paddings_cannot_coincide():
    """The two pipelines pad to different heights, so neither one's column is valid on the other's grid.

    This is the premise the whole fix rests on. If it ever stops holding, the cross-pipeline read that
    used to be here would become legitimate again -- and, more to the point, the mismatch guard that
    used to abort every fixed-reference sweep would stop firing for a different reason than the fix.
    """
    budget, sweep = required_pad_margin(BUDGET_SCALES), required_pad_margin(SWEEP_SCALES)
    print(f"\nrequired_pad_margin: budget(ℓ≤7) = {budget:.4f}   sweep(ℓ≤20) = {sweep:.4f}   ratio = {sweep/budget:.2f}")
    assert sweep > budget, (f"the sweep's margin ({sweep:.4f}) no longer exceeds the budget's ({budget:.4f}); "
                            f"the two pipelines would share a grid and these tests need rethinking")

    ds = _grid_dataset()
    n_pad = {}
    for name, margin in (("budget", budget), ("sweep", sweep)):
        n_pad[name] = int(_pad_domain_in_z(ds, min_margin=margin).attrs["n_pad_z"])
    print(f"  at Nz={ds.sizes['z_aac']}, Lz={LZ}:  n_pad_z budget = {n_pad['budget']}   sweep = {n_pad['sweep']}")
    assert n_pad["sweep"] > n_pad["budget"], (
        f"both pipelines padded to {n_pad['budget']} cells per side. The sweep's widest scale needs "
        f"{sweep:.4f} of margin against the budget's {budget:.4f}, so this should not happen unless the "
        f"Nz//2 floor swallowed both -- in which case raise Nz in `_grid_dataset`.")


@pytest.mark.parametrize("which", ["budget", "sweep"])
def test_column_has_one_slot_per_cell_of_its_own_grid(which):
    """N = Nx·Ny·Nz_padded, and the column spans that padded domain.

    This is what makes a column grid-specific: its slots *are* the cells, and z✶ is a position along
    it. Handing it to a field on a different grid silently reinterprets every height.
    """
    margin = required_pad_margin(BUDGET_SCALES if which == "budget" else SWEEP_SCALES)
    ds = _pad_domain_in_z(_grid_dataset(), min_margin=margin)
    column = _frozen_column(ds)

    expected = ds.sizes["x_caa"] * ds.sizes["y_aca"] * ds.sizes["z_aac"]
    z_name = column.rho_sorted.dims[-1]
    z = column[z_name].values
    print(f"\n{which}: n_pad_z={ds.attrs['n_pad_z']}  Nz_padded={ds.sizes['z_aac']}  slots={column.sizes[z_name]} "
          f"(expected {expected})  z ∈ [{z.min():.3f}, {z.max():.3f}]  domain z ∈ [{ds.attrs['z_min']:.3f}, "
          f"{ds.attrs['z_max']:.3f}]")

    assert column.sizes[z_name] == expected
    assert ds.attrs["z_min"] <= z.min() and z.max() <= ds.attrs["z_max"]
    # Slot heights are equal-volume, so they must sum to the padded domain height.
    assert np.isclose(float(column.dz_sorted.isel(time=0).sum()), ds.attrs["Lz"], rtol=1e-10)


def test_budget_column_is_not_usable_on_the_sweep_grid():
    """The bug, stated as a property: 02's column has the wrong number of slots for the sweep's grid.

    `sweep2` used to read it anyway. Nothing downstream would have revealed the error -- z✶ is just an
    index into a column, so a column of the wrong length shifts every displacement and stays finite.
    """
    ds_budget = _pad_domain_in_z(_grid_dataset(), min_margin=required_pad_margin(BUDGET_SCALES))
    ds_sweep = _pad_domain_in_z(_grid_dataset(), min_margin=required_pad_margin(SWEEP_SCALES))
    col_budget, col_sweep = _frozen_column(ds_budget), _frozen_column(ds_sweep)

    z_name = col_budget.rho_sorted.dims[-1]
    n_budget, n_sweep = col_budget.sizes[z_name], col_sweep.sizes[z_name]
    print(f"\n02's column: {n_budget} slots   sweep's grid needs: {n_sweep}   "
          f"z ranges [{col_budget[z_name].values.min():.3f}, {col_budget[z_name].values.max():.3f}] vs "
          f"[{col_sweep[z_name].values.min():.3f}, {col_sweep[z_name].values.max():.3f}]")
    assert n_budget != n_sweep, "02's column happens to fit the sweep's grid; see the premise test above"
    assert col_budget[z_name].values.min() > col_sweep[z_name].values.min()


def test_sweep_construction_matches_the_full_timeseries_sort():
    """Sorting t=0 alone and broadcasting == sorting the whole series with `fixed_reference=True`.

    `sweep2` takes `isel(time=[0])` before sorting, because `sorted_timeseries` does `ds[field].values`
    and would otherwise pull the entire padded density timeseries into RAM to read row 0. That is an
    optimisation only if it is exact, which is what this checks -- bit-identically, since both paths
    sort the same array with the same routine.
    """
    ds = _pad_domain_in_z(_grid_dataset(n_times=4), min_margin=required_pad_margin(SWEEP_SCALES))

    full = _frozen_column(ds)                                     # what 02 writes: sort t=0, repeat the row
    one = _frozen_column(ds.isel(time=[0]))                       # what sweep2 does
    broadcast = one.isel(time=0, drop=True).expand_dims(time=ds.time)

    for name in ("rho_sorted", "dz_sorted"):
        a = np.asarray(broadcast[name].transpose(*full[name].dims).values, float)
        b = np.asarray(full[name].values, float)
        print(f"\n  {name}: shape {a.shape} vs {b.shape}   max|diff| = {np.abs(a - b).max():.3e}")
        assert a.shape == b.shape
        assert np.array_equal(a, b), f"{name} differs between the broadcast and full-series constructions"

    # And the reference really is frozen: t=0 is genuinely not the sort of the later, different fields.
    varying = sorted_timeseries(ds, field_to_sort="ρ", n_workers=1, fixed_reference=False, verbose_level=0)
    drift = float(np.abs(varying.rho_sorted.isel(time=-1) - full.rho_sorted.isel(time=-1)).max())
    print(f"  frozen vs time-varying at the last step: max|diff| = {drift:.3e}")
    assert drift > 0, ("the frozen and time-varying columns agree at the last step, so the fixture's field "
                       "does not evolve and the freezing is untested")
#---


#+++ The sweep leaves the padding out of its arrays
# Under the edge extension the sweep loads its fields with one padded cell per side (`sweep_halo`) and leaves the rest
# of the padding virtual: the filter's `nearest` mode repeats the wall value past the end of an array, which is what
# the padding holds, and only the sorted column still needs the padded fluid. These tests hold that to the fully
# padded computation.
def _physical_plus_halo(ds_full, ds_halo):
    """The cells of the fully padded dataset that the halo one keeps."""
    n = ds_full.attrs["n_pad_z"] - ds_halo.attrs["n_pad_z"]
    return slice(n, ds_full.sizes["z_aac"] - n)


def _flow_dataset(n_times=2, Nx=16, Nz=64, Lx=4.0, U=1.0, h=0.5):
    """u, w and b of a sheared, stratified layer with a wave on it, far enough from the walls that b is flat there to
    the last bit, as it is in the runs (tanh(Lz/2h) is 1 in double precision)."""
    ds = _grid_dataset(Nx=Nx, Nz=Nz, Lx=Lx, h=h).drop_vars("ρ").drop_vars("time")
    X, Z = np.meshgrid(ds.x_caa.values, ds.z_aac.values, indexing="ij")
    interior = np.exp(-(Z / 3.0) ** 2)
    fields = dict(u=[], w=[], b=[])
    for k in range(n_times):
        wave = np.sin(2 * np.pi * X / Lx + k) + 0.5 * np.sin(6 * np.pi * X / Lx)
        fields["u"].append(U * np.tanh(Z / h) + 0.2 * np.cos(2 * np.pi * X / Lx + k) * interior)
        fields["w"].append(0.1 * wave * interior)                       # in phase with the displacement, so w·b_r has a mean
        fields["b"].append(0.1 * np.tanh((Z + 0.3 * wave * interior) / h))
    dims = ("time", "x_caa", "y_aca", "z_aac")
    coords = dict(time=np.arange(n_times, dtype=float), x_caa=ds.x_caa, y_aca=ds.y_aca, z_aac=ds.z_aac)
    for name, frames in fields.items():
        ds[name] = xr.DataArray(np.stack(frames)[:, :, None, :], dims=dims, coords=coords)
    return ds


@pytest.mark.parametrize("fixed", [True, False])
def test_virtual_padding_sorts_to_the_padded_column(fixed):
    """The column is the sort of the whole padded field, whether the padding is in the arrays or not."""
    margin = required_pad_margin(SWEEP_SCALES)
    ds_full = _pad_domain_in_z(_grid_dataset(n_times=3), min_margin=margin)
    ds_halo = _pad_domain_in_z(_grid_dataset(n_times=3), min_margin=margin, halo=sweep_halo("edge"))
    assert ds_halo.attrs["n_pad_z"] + ds_halo.attrs["n_pad_z_virtual"] == ds_full.attrs["n_pad_z"]
    assert all(ds_halo.attrs[k] == ds_full.attrs[k] for k in ("z_min", "z_max", "Lz"))
    assert ds_halo.sizes["z_aac"] == ds_full.sizes["z_aac"] - 2 * ds_halo.attrs["n_pad_z_virtual"]

    kept = _physical_plus_halo(ds_full, ds_halo)
    assert np.array_equal(ds_halo.z_aac.values, ds_full.z_aac.values[kept])
    assert np.array_equal(ds_halo["ρ"].values, ds_full["ρ"].isel(z_aac=kept).values)
    assert float(ds_halo.dV_physical.sum()) == float(ds_full.dV_physical.sum())

    kw = dict(field_to_sort="ρ", n_workers=1, fixed_reference=fixed, verbose_level=0)
    full, halo = sorted_timeseries(ds_full, **kw), sorted_timeseries(ds_halo, **kw)
    nz_full, nz_halo = ds_full.sizes["z_aac"], ds_halo.sizes["z_aac"]
    print(f"\n  fixed_reference={fixed}: {full.sizes['z_1d_sorted']} slots from {nz_full} and from {nz_halo} z levels")
    for name in ("rho_sorted", "dz_sorted"):
        assert np.array_equal(full[name].values, halo[name].values), f"{name} depends on whether the padding is in the arrays"
    assert np.array_equal(full.z_1d_sorted.values, halo.z_1d_sorted.values)


def test_only_the_edge_extension_can_be_left_virtual():
    """`nearest` repeats the wall value; an odd reflection is a different extension and has to be in the arrays."""
    assert sweep_halo("odd") is None
    with pytest.raises(ValueError, match="edge extension"):
        _pad_domain_in_z(_grid_dataset(), min_margin=required_pad_margin(SWEEP_SCALES), extension="odd", halo=1)


@pytest.mark.parametrize("ell", [0.5, 3.0, 20.0])
def test_filter_without_the_padding_matches_on_the_kept_cells(ell):
    """A field that repeats its wall value past the walls filters to the same numbers, bit for bit, with the padding in
    the array and with the filter's `nearest` mode standing in for it, at scales narrower and wider than the domain."""
    margin = required_pad_margin(SWEEP_SCALES)
    ds_full = _pad_domain_in_z(_flow_dataset(), min_margin=margin)
    ds_halo = _pad_domain_in_z(_flow_dataset(), min_margin=margin, halo=sweep_halo("edge"))
    kept = _physical_plus_halo(ds_full, ds_halo)
    gf = make_gaussian_filter(ell, ds_full)
    for field in (ds_full.u, ds_full.b, ds_full.u * ds_full.w):
        name = field.name or "u·w"
        full = gf.apply(field, dims=["x_caa", "z_aac"]).isel(z_aac=kept).transpose("time", "x_caa", "y_aca", "z_aac")
        halo = gf.apply(field.isel(z_aac=kept), dims=["x_caa", "z_aac"]).transpose("time", "x_caa", "y_aca", "z_aac")
        assert np.array_equal(full.values, halo.values), f"filter({name}) at ℓ={ell} changes when the padding leaves the array"


def test_transfer_integrals_do_not_need_the_padding():
    """∫Π_K, ∫Π_A and both halves of the APE->KE conversion from fields that keep one padded cell, against the fully
    padded computation, to roundoff: the padded cells add exact zeros to the integrals, in a different order."""
    scales = [0.5, 3.0, 20.0]
    margin = required_pad_margin(scales)
    kw = dict(ds_filt=None, n_workers=1, filtered_reference=True)
    full = calculate_energy_transfer(_pad_domain_in_z(_flow_dataset(), min_margin=margin), scales, **kw)
    ds_halo = _pad_domain_in_z(_flow_dataset(), min_margin=margin, halo=sweep_halo("edge"))
    halo = calculate_energy_transfer(ds_halo, scales, **kw)
    integrals = [v for v in full.data_vars if v.startswith("∫")]
    assert len(integrals) == 4
    for name in integrals:
        a, b = full[name].values, halo[name].values
        rel = float(np.abs(a - b).max() / np.abs(a).max())
        print(f"\n  {name:<20} max|padded| = {np.abs(a).max():.3e}   max|Δ|/max|padded| = {rel:.1e}")
        assert np.abs(a).max() > 1e-6, f"{name} is ~0 on this field, so the comparison says nothing"
        assert rel < 1e-12, f"{name} changes by {rel:.1e} when the padding is left out of the arrays"
#---
