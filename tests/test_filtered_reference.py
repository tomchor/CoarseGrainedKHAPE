"""
The filtered-reference scale decomposition, on the same synthetic field `test_jensen.py` uses.

`test_jensen.py` establishes that the sub-filter APE built as a plain remainder against the *unfiltered*
reference,

    Eₐˢ = filter(eₐ(b, z)) - eₐ(b̄, z)        [both against b✶]

has no fixed sign once the filter acts in z: eₐ(·, z) is convex in b at fixed z, but a kernel with
vertical extent averages a different convex function at every height, so Jensen does not carry. That is
measured there as a fact about the remainder construction.

It is not, however, a fact about the physics. Wenegrat, Chor & Barkan resolve it by measuring the
resolved reservoir against the rest state *as the filter sees it* — the vertically filtered profile
⟨b✶⟩ — rather than against b✶:

    Ē_A = L̃ + S̃,    L̃ = Ẽ_A(b̄, z)  [against ⟨b✶⟩],    S̃ = Ē_A - L̃

Both terms are then non-negative and both vanish for a fluid at rest, for any kernel. L̃ ≥ 0 because it
is an APE measured against a genuine monotone reference profile; S̃ ≥ 0 because it is a stencil average
of eₐ ≥ 0 once the stencil has been moved to the coarse field's own resting level. The splitting level
ζ̃ = z̃✶(b̄) is the unique stationary point of the split, so any other choice — the unfiltered b✶ included
— moves energy out of the resolved reservoir and can drive it negative.

These tests pin that down on the synthetic field: the same stratification, the same x-z filter, the same
machinery, changing only the reference profile the resolved reservoir is measured against. They need no
simulation or post-processing output.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent.parent / "postprocessing"))
from src.aux00_utils import GaussianFilter
from src.aux01_pe_functions import (filtered_reference_profile, local_potential_energies_timeseries,
                                    sorted_timeseries)

sys.path.insert(0, str(Path(__file__).parent))
from test_jensen import make_dataset, report

#+++ Thresholds
# Both reservoirs are non-negative by construction, so only the discretization may push them below zero:
# the nearest-density z✶ lookup places a filtered buoyancy that falls between two slots at the nearer one,
# which perturbs the integration limit by up to half a slot. `test_positivity.py` allows the same 1e-3 of
# rms for exactly this reason on the local APE, and these are the same lookup on the same column.
POSITIVITY_TOL = 1e-3

# S̃ is measured as Ē_A - L̃, which is exact — but only where filter(z) = z. `GaussianFilter` extends the
# bounded z axis with its edge value (mode="nearest"), so within a stencil of a wall the kernel is
# effectively one-sided, filter(z) is pulled toward the wall, and the τ(z, b) that the identity carries
# picks up a spurious contribution. On test_jensen.py's shorter column (the `near_wall` fixture below) the
# whole violation lives in the outermost cells: min/rms is -0.11, -0.14 and -0.07 at ℓ = 2, 4 and 8 cells,
# all of it in levels 0, 30 and 31. The production pipeline does not run into this — `_pad_domain_in_z`
# pads the domain with edge values at load time, precisely so the physical domain sits a long way from the
# filter's edge — so the band is excluded here rather than tested.
#
# The band is the filter's own stencil half-width, and may not exceed it: a wider band would exclude cells
# the filter's edge cannot reach, which is a way of passing the positivity tests rather than a reason.
# `test_edge_cells_are_the_only_place_the_identity_fails` checks both.
KERNEL_TRUNCATE = 4.0    # scipy's gaussian_filter1d truncation, used by GaussianFilter and matched online
EDGE_BAND_SIGMAS = 4.0

# At rest L̃ and S̃ vanish against ⟨ρ✶⟩ up to the discrete column: ~2e-8 on this field, against a max|L| of
# 7e-6, 2.6e-5 and 1.2e-4 measured against ρ✶ at 2, 4 and 8 cells (ratios 3.3e-3, 8.7e-4 and 1.7e-4).
# 1e-2 of the unfiltered value keeps a threefold margin at the narrowest filter.
REST_TOL = 1e-2

# The remainder construction's failure, reproduced here so the contrast is measured in one place rather
# than inferred across files. test_jensen.py uses the same floor.
VIOLATION_FLOOR = -0.1

FILTER_SCALES_IN_CELLS = [2, 4, 8]
FILTER_DIMS_XZ = ["x_caa", "z_aac"]
#---

#+++ Fixture
# Twice the height of the `test_jensen.py` default, so that excluding a 4σ band at each wall still leaves a
# substantial interior at the widest filter. The extra height is stratification the tanh has saturated,
# which is also what `_pad_domain_in_z` manufactures in the real pipeline.
DOMAIN = dict(Nx=64, Nz=64, Lx=4.0, Lz=4.0)


def sorted_field(**kwargs):
    """The field, its sorted reference state, and the local APE of the full field — sorted once."""
    ds, dx, dz = make_dataset(**kwargs)
    sorted_state = sorted_timeseries(ds, field_to_sort="ρ", n_workers=1, verbose_level=0)
    full = local_potential_energies_timeseries(ds, sorted_state.rho_sorted, sorted_state.dz_sorted,
                                               density_name="ρ", verbose_level=0, n_workers=1)
    return ds, dx, dz, sorted_state, full


@pytest.fixture(scope="module")
def synthetic():
    """The wave-displaced column on the tall domain, whose walls sit in saturated stratification."""
    return sorted_field(**DOMAIN)


@pytest.fixture(scope="module")
def at_rest():
    """The same tall column with no wave and no noise: horizontally uniform and stable, a fluid at rest."""
    return sorted_field(**DOMAIN, amplitude=0.0, noise=0.0)


@pytest.fixture(scope="module")
def near_wall():
    """test_jensen.py's default column, half the height, whose stratification still varies within a stencil
    of the walls, so the filter's one-sided edge treatment does break Ē_A = L̃ + S̃ there."""
    return sorted_field()


def decompose(synthetic, ell, dims, filtered_reference):
    """Return (L, S, filtered total) at scale `ell`, measuring the resolved reservoir against ⟨ρ✶⟩ when
    `filtered_reference`, and against the unfiltered ρ✶ otherwise. Everything else is held fixed, so the
    two paths differ in the reference profile and nothing else."""
    ds, dx, dz, sorted_state, full = synthetic
    gf = GaussianFilter(ell, dx_min=dx, dz_min=dz)
    ds_filtered = ds.assign(ρ̄=gf.apply(ds.ρ, dims=dims))

    reference = (filtered_reference_profile(sorted_state.rho_sorted, sorted_state.dz_sorted, ell)
                 if filtered_reference else sorted_state.rho_sorted)
    resolved = local_potential_energies_timeseries(ds_filtered, reference, sorted_state.dz_sorted,
                                                   density_name="ρ̄", verbose_level=0, n_workers=1)
    total = gf.apply(full.ape, dims=dims)
    for name, da in (("L̃", resolved.ape), ("Ē_A", total)):   # the min/max/rms reductions below skip NaN
        assert bool(np.isfinite(da).all()), f"{name} has non-finite values at l={ell:.4f}"
    return resolved.ape, total - resolved.ape, total


def interior(da, ell, dz, z_name="z_aac"):
    """Drop a 4σ band at each z wall — the region where the filter's edge extension makes filter(z) ≠ z
    and the Ē_A - L̃ identity therefore stops holding. Returns the field unchanged along every other axis."""
    from src.aux00_utils import _FWHM_TO_SIGMA
    band = int(np.ceil(EDGE_BAND_SIGMAS * ell * _FWHM_TO_SIGMA / dz))
    n = da.sizes[z_name]
    assert 2 * band < n - 2, f"a {band}-cell band at each wall leaves nothing of a {n}-cell axis"
    return da.isel({z_name: slice(band, n - band)})
#---

#+++ Tests
@pytest.mark.parametrize("cells", FILTER_SCALES_IN_CELLS)
def test_subfilter_ape_nonnegative_under_filtered_reference(synthetic, cells):
    """S̃ ≥ 0 under the pipeline's x-z filter — the claim the whole construction exists to deliver.

    This is the same field, filter and machinery under which `test_jensen.py` measures a violation of
    -0.8 to -2.4 x rms; the only change is which profile the resolved reservoir is measured against."""
    _, dx, dz, _, _ = synthetic
    ell = cells * dx
    print(f"\nFiltered reference, x-z filter  (l={ell:.4f} = {cells} cells)")
    _, S, _ = decompose(synthetic, ell, FILTER_DIMS_XZ, filtered_reference=True)
    report(S, "S~ (whole domain)")
    relative = report(interior(S, ell, dz), "S~ (interior)")
    assert relative > -POSITIVITY_TOL, (f"S̃ went negative under the filtered reference: min = {relative:.3e} x rms, "
                                        f"tolerance {-POSITIVITY_TOL:.0e} x rms. The construction of Eq. (2.5) is "
                                        f"positive semi-definite for any kernel, so this is either a bug in ⟨ρ✶⟩ or "
                                        f"a lookup artefact larger than the tolerance allows.")


@pytest.mark.parametrize("cells", FILTER_SCALES_IN_CELLS)
def test_resolved_ape_nonnegative_under_filtered_reference(synthetic, cells):
    """L̃ ≥ 0: the resolved reservoir is an APE measured against a genuine monotone reference profile."""
    _, dx, dz, _, _ = synthetic
    ell = cells * dx
    print(f"\nResolved reservoir, x-z filter  (l={ell:.4f} = {cells} cells)")
    L, _, _ = decompose(synthetic, ell, FILTER_DIMS_XZ, filtered_reference=True)
    relative = report(interior(L, ell, dz), "L~ (interior)")
    assert relative > -POSITIVITY_TOL, (f"L̃ went negative: min = {relative:.3e} x rms, tolerance "
                                        f"{-POSITIVITY_TOL:.0e} x rms.")


@pytest.mark.parametrize("cells", FILTER_SCALES_IN_CELLS)
def test_both_reservoirs_vanish_at_rest(at_rest, cells):
    """At rest, L̃ and S̃ vanish against ⟨ρ✶⟩, while L measured against the unfiltered ρ✶ does not.

    This is the property the construction exists for (Eq. 2.3). A fluid at rest filters to ⟨ρ✶⟩(z) itself, so
    the resolved reservoir measured against ⟨ρ✶⟩ is empty, and S̃ = Ē_A - L̃ with Ē_A = 0 is empty too. Measured
    against ρ✶ the filtered column still carries APE, which sets the scale: a ⟨ρ✶⟩ that is not the profile
    the filter makes of the resting column fails here. (L̃ + S̃ = Ē_A is not tested: S̃ is defined that way.)"""
    _, dx, _, _, _ = at_rest
    ell = cells * dx
    L, S, total = decompose(at_rest, ell, FILTER_DIMS_XZ, filtered_reference=True)
    L_unfiltered, _, _ = decompose(at_rest, ell, FILTER_DIMS_XZ, filtered_reference=False)
    size = lambda da: float(np.abs(da).max())
    print(f"\nAt rest  (l={ell:.4f} = {cells} cells)   max|Ea_bar| = {size(total):.2e}   max|L~| = {size(L):.2e}   "
          f"max|S~| = {size(S):.2e}   max|L| against rho* = {size(L_unfiltered):.2e}")
    assert size(L_unfiltered) > 0, "against the unfiltered ρ✶ a filtered column at rest should still carry APE"
    for name, da in (("L̃", L), ("S̃", S)):
        assert size(da) <= REST_TOL * size(L_unfiltered), (
            f"{name} does not vanish at rest: max|{name}| = {size(da):.3e} against {size(L_unfiltered):.3e} for L "
            f"measured against ρ✶. ⟨ρ✶⟩ is not the profile the filter makes of the resting column.")


@pytest.mark.parametrize("cells", FILTER_SCALES_IN_CELLS)
def test_unfiltered_reference_still_breaks(synthetic, cells):
    """The contrast, on the same field: against the unfiltered ρ✶ the sub-filter reservoir goes negative.

    Measured here so the two constructions are compared under one fixture. If this ever stops failing,
    the synthetic field has lost the vertical structure that makes the test meaningful, and the result
    above would be vacuous rather than informative."""
    _, dx, dz, _, _ = synthetic
    ell = cells * dx
    print(f"\nUnfiltered reference, x-z filter  (l={ell:.4f} = {cells} cells)")
    _, S, _ = decompose(synthetic, ell, FILTER_DIMS_XZ, filtered_reference=False)
    relative = report(interior(S, ell, dz), "Eas (interior)")
    assert relative < VIOLATION_FLOOR, (f"Eₐˢ stayed non-negative against the unfiltered ρ✶ (min = {relative:.3e} x "
                                        f"rms). The synthetic field may have lost its vertical structure, which "
                                        f"would make the filtered-reference result above vacuous.")


@pytest.mark.parametrize("cells", FILTER_SCALES_IN_CELLS)
def test_edge_cells_are_the_only_place_the_identity_fails(near_wall, cells):
    """The excluded band is the filter's edge and nothing more.

    S̃ ≥ 0 is guaranteed wherever S̃ = Ē_A - L̃ holds, and that identity needs filter(z) = z, which edge
    extension breaks within a stencil of a wall. On a column whose stratification reaches its walls the
    identity does fail there, so this checks three things: that it does (otherwise the test is vacuous),
    that every negative cell lies within the kernel's own reach of a wall, and that EDGE_BAND_SIGMAS is no
    wider than that reach, so it cannot become a way of excluding genuine interior violations."""
    assert EDGE_BAND_SIGMAS <= KERNEL_TRUNCATE, (f"EDGE_BAND_SIGMAS = {EDGE_BAND_SIGMAS} excludes cells beyond "
                                                f"the kernel's {KERNEL_TRUNCATE}σ reach, which its edge cannot affect")
    _, dx, dz, _, _ = near_wall
    ell = cells * dx
    _, S, _ = decompose(near_wall, ell, FILTER_DIMS_XZ, filtered_reference=True)
    S0 = S.isel(time=0, y_aca=0).transpose("x_caa", "z_aac")
    rms = float(np.sqrt((S0**2).mean()))
    negative_k = np.where((S0.values < -POSITIVITY_TOL * rms).any(axis=0))[0]

    from src.aux00_utils import _FWHM_TO_SIGMA
    band = int(np.ceil(KERNEL_TRUNCATE * ell * _FWHM_TO_SIGMA / dz))
    n = S0.sizes["z_aac"]
    in_band = [int(k) for k in negative_k if k < band or k >= n - band]
    stray = [int(k) for k in negative_k if band <= k < n - band]
    print(f"\nEdge confinement  (l={ell:.4f} = {cells} cells)   band = {band} cells of {n}   "
          f"negative levels = {negative_k.tolist()}")
    assert in_band, (f"S̃ has no negative cell near the walls of the near-wall column at l={ell:.4f}, so this "
                     f"test no longer exercises the edge failure it exists to confine.")
    assert not stray, (f"S̃ is negative at z-levels {stray}, which lie outside the {band}-cell reach of the "
                       f"kernel. The identity Ē_A = L̃ + S̃ should hold there, so this is a real violation "
                       f"rather than the filter's edge treatment.")
#---
