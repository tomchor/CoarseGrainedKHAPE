"""
Test that the Gaussian filter reproduces the analytical response to a 2D Dirac delta.

GaussianFilter takes the FWHM (ℓ) and derives σ = ℓ / (2√(2 ln 2)) internally.
Filtering a point impulse δ(x)δ(z) should yield the 2D Gaussian kernel:
  G(x, z) = 1/(2πσ²) · exp(-(x² + z²) / (2σ²))
"""

import sys
from pathlib import Path
import numpy as np
import xarray as xr
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "postprocessing"))
from aux00_utils import GaussianFilter

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"

#+++ Parameters
Nx, Nz = 512, 512
Lx, Lz = 14.0, 14.0
dx, dz = Lx / Nx, Lz / Nz
FILTER_SCALE = 1.0  # FWHM passed to GaussianFilter
_FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
SIGMA = FILTER_SCALE * _FWHM_TO_SIGMA  # actual Gaussian σ used internally
RTOL = 0.02
ATOL = 0.01  # ~6% of peak; covers tails where filtered rounds to zero
#---

#+++ Fixtures
@pytest.fixture(scope="module")
def filtered_delta():
    x = np.linspace(-Lx/2 + dx/2, Lx/2 - dx/2, Nx)
    z = np.linspace(-Lz/2 + dz/2, Lz/2 - dz/2, Nz)
    data = np.zeros((Nx, Nz))
    data[Nx // 2, Nz // 2] = 1.0 / (dx * dz)
    da = xr.DataArray(data, dims=["x_caa", "z_aac"], coords={"x_caa": x, "z_aac": z})
    gf = GaussianFilter(FILTER_SCALE, dx, dz)
    return gf.apply(da, dims=["x_caa", "z_aac"])
#---

#+++ Analytical reference
def analytical_2d_gaussian(x, z, ℓ):
    return np.exp(-(x**2 + z**2) / (2 * ℓ**2)) / (2 * np.pi * ℓ**2)
#---

#+++ Tests
def test_transect_z0(filtered_delta):
    """Transect at z=0: filtered vs analytical Gaussian in x."""
    x = filtered_delta.x_caa.values
    filtered = filtered_delta.sel(z_aac=0, method="nearest").values
    expected = analytical_2d_gaussian(x, 0, SIGMA)
    np.testing.assert_allclose(filtered, expected, rtol=RTOL, atol=ATOL)


def test_transect_x0(filtered_delta):
    """Transect at x=0: filtered vs analytical Gaussian in z."""
    z = filtered_delta.z_aac.values
    filtered = filtered_delta.sel(x_caa=0, method="nearest").values
    expected = analytical_2d_gaussian(0, z, SIGMA)
    np.testing.assert_allclose(filtered, expected, rtol=RTOL, atol=ATOL)


def test_peak_value(filtered_delta):
    """Peak value at origin should match 1/(2πσ²)."""
    peak = float(filtered_delta.sel(x_caa=0, z_aac=0, method="nearest"))
    expected = 1.0 / (2 * np.pi * SIGMA**2)
    assert abs(peak - expected) / expected < RTOL


def test_integral(filtered_delta):
    """Integral over the domain should be ≈ 1 (delta was normalised)."""
    integral = float((filtered_delta * dx * dz).sum())
    assert abs(integral - 1.0) < 0.01
#---

#+++ Figure generation (runs after tests, always)
@pytest.fixture(scope="module", autouse=True)
def save_figure(filtered_delta):
    yield
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = filtered_delta.x_caa.values
    z = filtered_delta.z_aac.values
    ℓ = SIGMA
    da_orig = xr.DataArray(np.zeros((Nx, Nz)), dims=["x_caa", "z_aac"], coords={"x_caa": x, "z_aac": z})
    da_orig.values[Nx // 2, Nz // 2] = 1.0 / (dx * dz)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    kw = dict(x="x_caa", y="z_aac", add_colorbar=True)
    da_orig.plot(ax=axes[0, 0], **kw)
    axes[0, 0].set_title("Original (Dirac delta)")
    filtered_delta.plot(ax=axes[0, 1], **kw)
    axes[0, 1].set_title(f"Filtered (FWHM = {FILTER_SCALE}, σ = {ℓ:.4f})")
    for ax in axes[0, :]:
        ax.set_aspect("equal")

    filtered_x0 = filtered_delta.sel(x_caa=0, method="nearest").values
    analytical_x0 = analytical_2d_gaussian(0, z, ℓ)
    axes[1, 0].plot(z, filtered_x0, label="Filtered")
    axes[1, 0].plot(z, analytical_x0, "--", label=r"$\frac{1}{2\pi\ell^2} e^{-z^2 / 2\ell^2}$")
    axes[1, 0].set(title="Transect at x = 0", xlabel="z", ylabel="Amplitude")

    filtered_z0 = filtered_delta.sel(z_aac=0, method="nearest").values
    analytical_z0 = analytical_2d_gaussian(x, 0, ℓ)
    axes[1, 1].plot(x, filtered_z0, label="Filtered")
    axes[1, 1].plot(x, analytical_z0, "--", label=r"$\frac{1}{2\pi\ell^2} e^{-x^2 / 2\ell^2}$")
    axes[1, 1].set(title="Transect at z = 0", xlabel="x", ylabel="Amplitude")

    for ax in axes[1, :]:
        ax.legend()

    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "test_gaussian_filter.png", dpi=150)
    plt.close(fig)
#---
