"""
Jensen's inequality for the sub-filter APE, on a synthetic field.

The local APE eₐ(b, z) = ∫_{z✶(b)}^{z}[b✶(z̃) - b] dz̃ is convex in b at fixed z:

    ∂eₐ/∂b = z✶ - z        (the b✶(z✶) = b term cancels)
    ∂²eₐ/∂b² = ∂z✶/∂b ≥ 0  (b✶ increases with height, so z✶ increases with b)

Jensen then gives filter(eₐ) ≥ eₐ(b̄) — that is, Eₐˢ = filter(eₐ) - eₐˡ ≥ 0 pointwise — for any filter
whose weights are non-negative and sum to one, which the Gaussian filter's are. The catch is the
"at fixed z": Jensen averages one convex function, and eₐ(·, z) is a different function at every
height. A filter acting only in x therefore satisfies the bound, and one acting in z averages a family
of functions and need not.

That is the whole reason `test_positivity.py` checks the local APE and the SFS KE but not Eₐˢ: the
pipeline filters in x and z (`filtered_dimensions` in `05_sfs_ape_budget.py`), so its Eₐˢ is genuinely
of either sign. Both halves are checked here, on a stratification displaced by a wave in x:

  * with a horizontal filter the bound must hold — this is the pointwise Jensen test, and it exercises
    the repo's own `GaussianFilter` and APE machinery rather than the abstract inequality;
  * with the pipeline's x-z filter it must break. That second assertion is not a bug being enshrined.
    It is the measurement showing the negative Eₐˢ in the budget comes from filtering across z and
    nothing else, and it would fail if the filter were ever narrowed to the horizontal — which would
    change what the sub-filter budget means and should not pass quietly.

Self-contained: no simulation or post-processing output needed.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent.parent / "postprocessing"))
from src.aux00_utils import GaussianFilter
from src.aux01_pe_functions import local_potential_energies_timeseries, sorted_timeseries

#+++ Thresholds
# Horizontal filter: Jensen is exact, so only roundoff is allowed. The discrete z✶ lookup does not
# spoil it — measured minima are strictly positive, at +8e-06 to +1e-04 of rms(Eₐˢ).
JENSEN_TOL = 1e-9

# x-z filter: measured min/rms is -0.80, -2.0 and -2.4 at ℓ = 2, 4 and 8 grid cells, with 16-48% of the
# domain negative. A tenth of the rms is far below the smallest of those and far above anything a
# lookup or roundoff artefact could produce.
VIOLATION_FLOOR = -0.1

FILTER_SCALES_IN_CELLS = [2, 4, 8]
#---

#+++ Synthetic stratification
def make_dataset(Nx=64, Nz=32, Lx=4.0, Lz=2.0, h=0.3, amplitude=0.25, noise=0.05, seed=0):
    """A stably stratified column displaced by a wave in x, plus noise, so that eₐ > 0 with structure
    on both axes. Built to the layout the offline sort expects: ρ(time, x, y, z), cell volumes dV, the
    horizontal area LxLy, and a z_min attribute."""
    dx, dz = Lx / Nx, Lz / Nz
    x = (np.arange(Nx) + 0.5) * dx
    z = (np.arange(Nz) + 0.5) * dz - Lz / 2
    X, Z = np.meshgrid(x, z, indexing="ij")

    rng = np.random.default_rng(seed)
    displacement = amplitude * np.sin(2 * np.pi * X / Lx) + noise * rng.standard_normal(X.shape)
    ρ = 1025.0 - 2.0 * np.tanh((Z + displacement) / h)

    dims   = ("time", "x_caa", "y_aca", "z_aac")
    coords = dict(time=[0.0], x_caa=x, y_aca=[0.0], z_aac=z)
    ds = xr.Dataset(dict(ρ=xr.DataArray(ρ[None, :, None, :], dims=dims, coords=coords)))
    ds["dV"]   = xr.DataArray(np.full((Nx, 1, Nz), dx * dz), dims=dims[1:], coords={d: coords[d] for d in dims[1:]})
    ds["LxLy"] = xr.DataArray(Lx)
    ds.attrs["z_min"] = float(z[0] - dz / 2)
    return ds, dx, dz


@pytest.fixture(scope="module")
def synthetic():
    """The field, its sorted reference state, and the local APE of the full field. Sorted once: every
    Eₐˢ below is measured against this same reference profile, which is what makes the two APEs two
    values of one convex function rather than two unrelated quantities."""
    ds, dx, dz = make_dataset()
    sorted_state = sorted_timeseries(ds, field_to_sort="ρ", n_workers=1, verbose_level=0)
    full = local_potential_energies_timeseries(ds, sorted_state.rho_sorted, sorted_state.dz_sorted,
                                               density_name="ρ", verbose_level=0, n_workers=1)
    return ds, dx, dz, sorted_state, full
#---

#+++ Sub-filter APE and its statistics
def subfilter_ape(synthetic, ell, dims):
    """Eₐˢ = filter(eₐ(ρ)) - eₐ(ρ̄), both terms against the full field's reference profile.

    `dims` is the [x, z] pair `GaussianFilter.apply` convolves over in turn. Passing the singleton y
    axis in the second slot makes that pass a no-op, which is how a horizontal-only filter is spelled
    here (the same idiom `calculate_sfs_R_correction` uses by default)."""
    ds, dx, dz, sorted_state, full = synthetic
    gf = GaussianFilter(ell, dx_min=dx, dz_min=dz)
    ds_filtered = ds.assign(ρ̄=gf.apply(ds.ρ, dims=dims))
    filtered = local_potential_energies_timeseries(ds_filtered, sorted_state.rho_sorted, sorted_state.dz_sorted,
                                                   density_name="ρ̄", verbose_level=0, n_workers=1)
    return gf.apply(full.ape, dims=dims) - filtered.ape


def report(Eas, label):
    """min(Eₐˢ)/rms(Eₐˢ), printed with the raw numbers behind it."""
    values = Eas.values.ravel()
    minimum, rms = float(values.min()), float(np.sqrt(np.mean(values**2)))
    relative = minimum / rms
    print(f"  {label:<22}  min={minimum:+.4e}  rms={rms:.4e}  min/rms={relative:+.3e}  "
          f"frac(<0)={float((values < 0).mean()):.3e}")
    return relative
#---

#+++ Tests
@pytest.mark.parametrize("cells", FILTER_SCALES_IN_CELLS)
def test_sfs_ape_nonnegative_under_horizontal_filter(synthetic, cells):
    """Filtering in x alone keeps every kernel weight at one height, so Jensen bounds Eₐˢ below by zero."""
    _, dx, _, _, _ = synthetic
    ell = cells * dx
    print(f"\nJensen bound, horizontal filter  (l={ell:.4f} = {cells} cells)")
    relative = report(subfilter_ape(synthetic, ell, ["x_caa", "y_aca"]), "Eaˢ (x only)")
    assert relative > -JENSEN_TOL, (f"Eₐˢ is negative under a horizontal filter, which violates Jensen: "
                                    f"min = {relative:.3e} x rms, tolerance is {-JENSEN_TOL:.0e} x rms")


@pytest.mark.parametrize("cells", FILTER_SCALES_IN_CELLS)
def test_vertical_filtering_breaks_the_jensen_bound(synthetic, cells):
    """The pipeline's x-z filter averages eₐ(·, z) across heights, so the same bound does not hold."""
    _, dx, _, _, _ = synthetic
    ell = cells * dx
    print(f"\nJensen bound, x-z filter  (l={ell:.4f} = {cells} cells)")
    relative = report(subfilter_ape(synthetic, ell, ["x_caa", "z_aac"]), "Eaˢ (x and z)")
    assert relative < VIOLATION_FLOOR, (f"Eₐˢ stayed non-negative under an x-z filter (min = {relative:.3e} x rms). "
                                        f"Either the filter no longer acts in z, in which case Eₐˢ has a sign and "
                                        f"the budget's interpretation changes, or the synthetic field lost its "
                                        f"vertical structure.")
#---
