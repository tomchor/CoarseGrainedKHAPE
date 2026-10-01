"""Unit tests for GaussianFilter FWHM convention.

Filtering a Dirac impulse should yield a Gaussian whose full-width at
half-maximum equals the requested filter scale ℓ.
"""
import sys
from pathlib import Path
import numpy as np
import xarray as xr
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "postprocessing"))
from src.aux00_utils import GaussianFilter, _FWHM_TO_SIGMA, _filter_field, _filter_passes


def _measure_fwhm(values, coords):
    """FWHM of a positive 1-D profile by linear interpolation at half-maximum."""
    peak = float(np.max(values))
    half = peak / 2.0
    above = values >= half
    crossings = np.where(np.diff(above.astype(int)))[0]
    if len(crossings) < 2:
        return None
    # left half-max crossing (rising edge: False→True at index i)
    i = crossings[0]
    x_left = np.interp(half,
                       [values[i], values[i + 1]],
                       [coords[i], coords[i + 1]])
    # right half-max crossing (falling edge: True→False at index j)
    j = crossings[-1]
    x_right = np.interp(half,
                        [values[j + 1], values[j]],
                        [coords[j + 1], coords[j]])
    return float(x_right - x_left)


@pytest.mark.parametrize("ell", [0.4, 1.0, 2.0])
def test_filter_fwhm_x(ell):
    """Impulse filtered in x (periodic BC) has FWHM = ell."""
    N, L = 2048, 40.0
    dx = L / N
    x = np.arange(N) * dx

    impulse = np.zeros((N, 1))
    impulse[N // 2, 0] = 1.0
    da = xr.DataArray(impulse, dims=["x_caa", "z_aac"],
                      coords={"x_caa": x, "z_aac": [0.0]})

    gf = GaussianFilter(ell, {"x_caa": dx, "z_aac": dx})
    filtered = gf.apply(da, dims=["x_caa", "z_aac"])

    profile = filtered.isel(z_aac=0).values
    fwhm = _measure_fwhm(profile, x)

    assert fwhm is not None, "Could not find half-max crossings in x profile"
    assert abs(fwhm - ell) / ell < 0.01, f"x-FWHM = {fwhm:.4f}, expected {ell:.4f}"


@pytest.mark.parametrize("ell", [0.4, 1.0, 2.0])
def test_filter_fwhm_z(ell):
    """Impulse filtered in z (bounded BC, away from walls) has FWHM = ell."""
    N, L = 2048, 40.0
    dz = L / N
    z = np.arange(N) * dz

    impulse = np.zeros((1, N))
    impulse[0, N // 2] = 1.0
    da = xr.DataArray(impulse, dims=["x_caa", "z_aac"],
                      coords={"x_caa": [0.0], "z_aac": z})

    gf = GaussianFilter(ell, {"x_caa": dz, "z_aac": dz})
    filtered = gf.apply(da, dims=["x_caa", "z_aac"])

    profile = filtered.isel(x_caa=0).values
    fwhm = _measure_fwhm(profile, z)

    assert fwhm is not None, "Could not find half-max crossings in z profile"
    assert abs(fwhm - ell) / ell < 0.01, f"z-FWHM = {fwhm:.4f}, expected {ell:.4f}"


#+++ The whole-field path gives the plain passes bit for bit, whatever the z padding
# `_filter_field` filters one plane of each run of identical planes at the ends of z (the padding `_pad_domain_in_z`
# adds) and copies it. It must not change a single bit, so that the filtered fields do not depend on how the filter
# got to them.
RNG = np.random.default_rng(0)


def _padded(shape, n_lo, n_hi, mode="edge"):
    """A random field with `n_lo` and `n_hi` planes of z padding, repeated wall planes unless `mode` says otherwise."""
    return np.pad(RNG.standard_normal(shape), [(0, 0)] * (len(shape) - 1) + [(n_lo, n_hi)], mode=mode)


def _xyz_passes(r_x, r_y, r_z):
    return [(-3, 0.3 * r_x, r_x, "wrap"), (-2, 0.3 * r_y, r_y, "wrap"), (-1, 0.3 * r_z, r_z, "nearest")]


@pytest.mark.parametrize("n_lo,n_hi", [(0, 0), (1, 1), (3, 0), (0, 7), (12, 12), (40, 25)])
@pytest.mark.parametrize("r_z", [1, 5, 20, 60])
def test_padding_shortcut_is_exact(n_lo, n_hi, r_z):
    """Padding narrower and wider than the stencil, on one side or both: identical to filtering every plane."""
    a = _padded((10, 6, 16), n_lo, n_hi)
    passes = _xyz_passes(4, 6, r_z)
    assert np.array_equal(_filter_field(a, passes), _filter_passes(a, passes))


@pytest.mark.parametrize("a,passes", [
    (_padded((3, 10, 6, 16), 12, 12),                 _xyz_passes(4, 6, 5)),                            # several records in one block
    (_padded((10, 6, 16), 12, 12, mode="reflect"),    _xyz_passes(4, 6, 5)),                            # padding that is not constant
    (np.broadcast_to(RNG.standard_normal((10, 6, 1)), (10, 6, 30)).copy(), _xyz_passes(4, 6, 5)),      # no z dependence at all
    (_padded((10, 6, 16), 12, 12),                    [(-1, 1.5, 5, "nearest")]),                       # the bounded pass alone
    (_padded((10, 6, 16), 12, 12),                    [(-1, 1.5, 5, "nearest"), (-3, 1.2, 4, "wrap")]), # the bounded pass first
    (_padded((10, 6, 16), 12, 12).astype("float32"),  _xyz_passes(4, 6, 5)),
], ids=["records", "reflected", "constant", "z-only", "z-first", "float32"])
def test_padding_shortcut_is_exact_in_the_odd_cases(a, passes):
    out = _filter_field(a, passes)
    assert out.dtype == a.dtype and np.array_equal(out, _filter_passes(a, passes))


@pytest.mark.parametrize("ell", [0.3, 2.0, 30.0])
@pytest.mark.parametrize("chunks", [None, {"time": 1, "z_aac": 9, "y_aca": 2, "x_caa": 5}])
def test_apply_is_the_three_scipy_passes(ell, chunks):
    """`apply`, on numpy and on a dask array chunked along every dimension, against the passes written out: wrap in x and
    y, nearest in z, the stencil cut at 4σ and at one period."""
    from scipy.ndimage import gaussian_filter1d
    spacing = {"x_caa": 0.5, "y_aca": 0.5, "z_aac": 0.25}
    da = xr.DataArray(_padded((2, 16, 6, 12), 10, 10).transpose(0, 3, 2, 1), dims=["time", "z_aac", "y_aca", "x_caa"])
    expected = da.transpose("time", "x_caa", "y_aca", "z_aac").values
    for axis, (dim, mode) in enumerate([("x_caa", "wrap"), ("y_aca", "wrap"), ("z_aac", "nearest")], start=1):
        sigma = ell * _FWHM_TO_SIGMA / spacing[dim]
        radius = max(1, int(4 * sigma + 0.5))
        radius = min(radius, da.sizes[dim]) if mode == "wrap" else radius
        expected = gaussian_filter1d(expected, sigma=sigma, axis=axis, mode=mode, radius=radius)

    filtered = GaussianFilter(ell, spacing).apply(da if chunks is None else da.chunk(chunks), dims=("x_caa", "y_aca", "z_aac"))
    assert filtered.dims == ("time", "x_caa", "y_aca", "z_aac")
    assert np.array_equal(filtered.values, expected)
#---
