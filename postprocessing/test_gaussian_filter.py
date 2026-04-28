#!/usr/bin/env python
#+++ Imports
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from aux00_utils import GaussianFilter
#---

#+++ Parameters
Nx, Nz = 512, 128
Lx, Lz = 14.0, 25.0
dx, dz = Lx / Nx, Lz / Nz
filter_scale = 1.0
#---

#+++ Create synthetic DataArray with a Dirac delta at x=0
x = np.linspace(-Lx/2 + dx/2, Lx/2 - dx/2, Nx)
z = np.linspace(-Lz/2 + dz/2, Lz/2 - dz/2, Nz)
data = np.zeros((Nx, Nz))
mid_x, mid_z = Nx // 2, Nz // 2
data[mid_x, mid_z] = 1.0 / (dx * dz)  # normalize so integral over x,z ≈ 1
da = xr.DataArray(data, dims=["x_caa", "z_aac"], coords={"x_caa": x, "z_aac": z})
#---

#+++ Apply the Gaussian filter
gf = GaussianFilter(filter_scale, dx, dz)
da_filtered = gf.apply(da, dims=["x_caa", "z_aac"])
#---

#+++ Plot: heatmaps + transects with analytical expectations
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Top row: heatmaps
kw = dict(x="x_caa", y="z_aac", add_colorbar=True)
da.plot(ax=axes[0, 0], **kw)
axes[0, 0].set_title("Original (Dirac delta)")

da_filtered.plot(ax=axes[0, 1], **kw)
axes[0, 1].set_title(f"Filtered (ℓ = {filter_scale})")

for ax in axes[0, :]:
    ax.set_aspect("equal")

# Bottom-left: transect at x=0 (filtered profile vs z)
filtered_x0 = da_filtered.sel(x_caa=0, method="nearest").values
ℓ = filter_scale
analytical_x0 = np.exp(-z**2 / (2 * ℓ**2)) / (2 * np.pi * ℓ**2)
axes[1, 0].plot(z, filtered_x0, label="Filtered")
axes[1, 0].plot(z, analytical_x0, "--", label=r"$\frac{1}{2\pi\ell^2} e^{-z^2 / 2\ell^2}$")
axes[1, 0].set(title="Transect at x = 0", xlabel="z", ylabel="Amplitude")

# Bottom-right: transect at z=0 (filtered profile vs x)
filtered_z0 = da_filtered.sel(z_aac=0, method="nearest").values
analytical_z0 = np.exp(-x**2 / (2 * ℓ**2)) / (2 * np.pi * ℓ**2)
axes[1, 1].plot(x, filtered_z0, label="Filtered")
axes[1, 1].plot(x, analytical_z0, "--", label=r"$\frac{1}{2\pi\ell^2} e^{-x^2 / 2\ell^2}$")
axes[1, 1].set(title="Transect at z = 0", xlabel="x", ylabel="Amplitude")

for ax in axes[1, :]:
    # ax.set_ylim(0, None)
    ax.legend()

fig.tight_layout()
fig.savefig("output/test_gaussian_filter.png", dpi=150)
print("Saved to output/test_gaussian_filter.png")
plt.show()
#---
