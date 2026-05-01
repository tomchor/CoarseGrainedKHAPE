#!/usr/bin/env python
#+++ Imports
import os
from pathlib import Path
import numpy as np
import xarray as xr
from aux00_utils import load_dataset_and_grid, make_gaussian_filter
#---

#+++ Configuration
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--filename", default="output/khi_Nz256_Ri0.10.nc")
args = parser.parse_args()
REPO_ROOT = Path(__file__).resolve().parent.parent
filename = str(REPO_ROOT / args.filename) if not os.path.isabs(args.filename) else args.filename
filter_scales = np.geomspace(0.1, 8, 5)
#---

#+++ Load data and grid
ds = load_dataset_and_grid(filename)
#---

#+++ Create circle DataArray (radius 2, centered at x=2, z=2)
x, z = ds.x_caa, ds.z_aac
ds["circle"] = xr.where((x - 3)**2 + (z - 5)**2 <= 4, 1.0, 0.0)
#---

#+++ Filter at each length scale
scale_coord = xr.DataArray(filter_scales, dims="filter_scale")
ds_filt = xr.concat(
    [make_gaussian_filter(ℓ, ds).apply(ds["circle"], dims=["x_caa", "z_aac"])
     for ℓ in filter_scales],
    dim=scale_coord,
).to_dataset(name="circle_filtered")
print(ds_filt)
#---

from matplotlib import pyplot as plt
g = ds_filt.circle_filtered.plot(col="filter_scale", x="x_caa",)

# Set "data aspect ratio" to 1 for each subplot (axes)
for ax in np.ravel(g.axes):
    ax.set_aspect('equal', adjustable='box')

plt.show()