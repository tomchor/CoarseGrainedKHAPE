#!/usr/bin/env python
import os
# Disable HDF5 advisory file locking — required for parallel writes on Lustre (Derecho/GPFS)
# and when multiple dask worker processes open the same HDF5/NetCDF4 file concurrently.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
#+++ Imports
from pathlib import Path
import numpy as np
import xarray as xr
from dask.distributed import Client, LocalCluster, progress
from aux00_utils import load_dataset_and_grid, condense_velocities, make_gaussian_filter
#---

#+++ Configuration
import argparse
parser = argparse.ArgumentParser(description="Filter velocity and buoyancy fields for KE budget")
parser.add_argument("--filename", default="output/khi_128x1x256.nc",
                    help="Path to simulation NetCDF file")
parser.add_argument("--n-workers", type=int, default=6,
                    help="Number of dask workers in LocalCluster")
parser.add_argument("--threads-per-worker", type=int, default=3,
                    help="Threads per dask worker")
#---

if __name__ == "__main__":
    args = parser.parse_args()
    REPO_ROOT = Path(__file__).resolve().parent.parent
    filename = str(REPO_ROOT / args.filename) if not os.path.isabs(args.filename) else args.filename
    filter_length_scales = np.geomspace(0.1, 2, 4) # Length scales for filtering

    #+++ Start dask cluster
    print("\n" + "="*60)
    print("Starting dask LocalCluster...")
    cluster = LocalCluster(n_workers=args.n_workers, threads_per_worker=args.threads_per_worker)
    client = Client(cluster)
    print(f"  Workers: {args.n_workers}  threads/worker: {args.threads_per_worker}  "
          f"(total CPUs: {args.n_workers * args.threads_per_worker})")
    print(f"  Dashboard: {client.dashboard_link}")
    #---

    #+++ Load data and grid
    print("\n" + "="*60)
    print("Loading data and grid...")
    ds = load_dataset_and_grid(filename)
    # Chunk only along time: each time step is one independent task.
    # x/y must stay whole (filter operates along those dims).
    # z is a batch dimension that gcm_filters handles internally via numpy — splitting
    # it would multiply an already complex task graph without improving performance.
    ds = ds.chunk({"time": 1})
    print(f"Dataset loaded: {len(ds.time)} time steps")
    #---

    #+++ Filter velocity and buoyancy fields at each length scale
    filter_in_2d = ds.sizes["x_caa"] > 1 and ds.sizes["y_aca"] > 1
    print("\n" + "="*60)
    if filter_in_2d:
        print("Filtering velocity and buoyancy fields in 2D (x and y)...")
    else:
        print("Filtering velocity and buoyancy fields in 1D (x only)...")

    ds = condense_velocities(ds, indices=[1, 2, 3])

    ds_filt_list = []
    for ℓ in filter_length_scales:
        print(f"  filter_length_scale = {ℓ:.4f}...")
        gf = make_gaussian_filter(ℓ, ds, filter_in_2d)
        ds_filt_list.append(xr.Dataset({
            "ūᵢ": gf.apply(ds["uᵢ"], dims=["x_caa", "y_aca"]),
            "b̄":  gf.apply(ds["b"],  dims=["x_caa", "y_aca"]),
        }))

    scale_coord = xr.DataArray(filter_length_scales, dims="filter_length_scale",
                                name="filter_length_scale")
    ds_filt = xr.concat(ds_filt_list, dim=scale_coord)
    ds_filt["dV"] = ds["dV"]  # scale-independent, no filter_length_scale dimension
    ds_filt.attrs["filter_ndim"] = 2 if filter_in_2d else 1
    print("Done building lazy task graph — all filter scales will be computed in parallel")
    #---

    #+++ Save filtered fields
    print("\n" + "="*60)
    print("Saving filtered fields...")

    output_filename = filename.replace(".nc", "_filtered_velocities.nc")
    write = ds_filt.to_netcdf(output_filename, compute=False)
    future = client.compute(write)
    progress(future)
    future.result()
    print(f"Filtered fields saved to: {output_filename}")
    #---

    #+++ Shutdown cluster
    client.close()
    cluster.close()
    #---
