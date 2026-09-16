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
# picks up a spurious contribution. Measured on this field the whole violation lives in the first and last
# cell: interior minima are +0.003, +0.026 and +0.25 x rms at ℓ = 2, 4 and 8 cells, against -0.11, -0.14
# and -0.07 in the edge cells. The production pipeline does not run into this — `_pad_domain_in_z` doubles
# the domain height with edge values at load time, precisely so the physical domain sits a long way from
# the filter's edge — so the band is excluded here rather than tested. `test_edge_cells_are_the_only_...`
# below pins the exclusion to the edge so it cannot quietly widen into a way of passing.
#
# The band is the filter's own stencil half-width: scipy truncates the Gaussian at 4σ, which the online
# filter matches (see CLAUDE.md on `matched_filter`).
EDGE_BAND_SIGMAS = 4.0

# L̃ + S̃ = Ē_A holds identically — S̃ is *defined* as the difference here — so this checks only that the
# reassembly carries no NaNs and loses nothing to float64 cancellation.
CLOSURE_TOL = 1e-12

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


@pytest.fixture(scope="module")
def synthetic():
    """The field, its sorted reference state, and the local APE of the full field — sorted once."""
    ds, dx, dz = make_dataset(**DOMAIN)
    sorted_state = sorted_timeseries(ds, field_to_sort="ρ", n_workers=1, verbose_level=0)
    full = local_potential_energies_timeseries(ds, sorted_state.rho_sorted, sorted_state.dz_sorted,
                                               density_name="ρ", verbose_level=0, n_workers=1)
    return ds, dx, dz, sorted_state, full


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
def test_decomposition_is_exact(synthetic, cells):
    """L̃ + S̃ = Ē_A, so the split moves energy between reservoirs without creating or destroying any."""
    _, dx, _, _, _ = synthetic
    ell = cells * dx
    L, S, total = decompose(synthetic, ell, FILTER_DIMS_XZ, filtered_reference=True)
    residual = float(np.abs((L + S - total)).max())
    scale = float(np.sqrt((total**2).mean()))
    print(f"\nClosure  (l={ell:.4f})   max|L~ + S~ - Ea_bar| = {residual:.3e}   rms(Ea_bar) = {scale:.3e}")
    assert residual < CLOSURE_TOL * scale, f"L̃ + S̃ does not reassemble Ē_A: {residual:.3e} vs rms {scale:.3e}"


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
def test_edge_cells_are_the_only_place_the_identity_fails(synthetic, cells):
    """The excluded band is the filter's edge and nothing more.

    S̃ ≥ 0 is guaranteed wherever S̃ = Ē_A - L̃ holds, and that identity needs filter(z) = z, which edge
    extension breaks within a stencil of a wall. This checks the failure really is confined there: every
    negative cell must lie in the excluded band. Without it, widening EDGE_BAND_SIGMAS would be a way to
    make the positivity tests pass by excluding genuine interior violations."""
    _, dx, dz, _, _ = synthetic
    ell = cells * dx
    _, S, _ = decompose(synthetic, ell, FILTER_DIMS_XZ, filtered_reference=True)
    S0 = S.isel(time=0, y_aca=0)
    rms = float(np.sqrt((S0**2).mean()))
    negative_k = np.where((S0.values < -POSITIVITY_TOL * rms).any(axis=0))[0]

    from src.aux00_utils import _FWHM_TO_SIGMA
    band = int(np.ceil(EDGE_BAND_SIGMAS * ell * _FWHM_TO_SIGMA / dz))
    n = S0.sizes["z_aac"]
    stray = [int(k) for k in negative_k if band <= k < n - band]
    print(f"\nEdge confinement  (l={ell:.4f} = {cells} cells)   band = {band} cells of {n}   "
          f"negative levels = {negative_k.tolist()}")
    assert not stray, (f"S̃ is negative at z-levels {stray}, which lie outside the {band}-cell edge band. "
                       f"The identity Ē_A = L̃ + S̃ should hold there, so this is a real violation rather "
                       f"than the filter's edge treatment.")
#---
