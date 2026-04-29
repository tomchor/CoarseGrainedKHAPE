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
from aux00_utils import GaussianFilter


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

    gf = GaussianFilter(ell, dx_min=dx, dz_min=dx)
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

    gf = GaussianFilter(ell, dx_min=dz, dz_min=dz)
    filtered = gf.apply(da, dims=["x_caa", "z_aac"])

    profile = filtered.isel(x_caa=0).values
    fwhm = _measure_fwhm(profile, z)

    assert fwhm is not None, "Could not find half-max crossings in z profile"
    assert abs(fwhm - ell) / ell < 0.01, f"z-FWHM = {fwhm:.4f}, expected {ell:.4f}"
