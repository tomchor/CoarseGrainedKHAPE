import os
from pathlib import Path
import numpy as np
import xarray as xr
import gcm_filters

PP_OUTPUT = Path(__file__).resolve().parent / "output"

#+++ Integrations and sums
def integrate(da, dV, dims=("x_caa", "y_aca", "z_aac")):
    """Integrate a DataArray over spatial dimensions"""
    return (da * dV).sum(dims)
#---

#+++ Load data
def load_dataset_and_grid(filename):
    """
    Load the simulation output and grid information

    Parameters
    ----------
    filename : str
        Path to the NetCDF file

    Returns
    -------
    ds : xr.Dataset
        Dataset with grid information added as attributes and variables
    """
    print(f"Loading data from {filename}...")
    ds = xr.open_dataset(filename, decode_times=False)
    grid = xr.open_dataset(filename, group="underlying_grid_reconstruction_kwargs")

    # Add grid extent as attributes
    ds.attrs["Lx"] = np.diff(grid.x)
    ds.attrs["Ly"] = np.diff(grid.y)
    ds.attrs["Lz"] = np.diff(grid.z)

    ds.attrs["x_min"] = grid.x.min()
    ds.attrs["x_max"] = grid.x.max()
    ds.attrs["y_min"] = grid.y.min()
    ds.attrs["y_max"] = grid.y.max()
    ds.attrs["z_min"] = grid.z.min()
    ds.attrs["z_max"] = grid.z.max()

    # Add volume and area variables
    ds["dV"] = ds.Δx_caa * ds.Δy_aca * ds.Δz_aac
    ds["LxLy"] = ds.Lx * ds.Ly

    return ds
#---

#+++ Condensing operations for Datasets
def condense(ds, vlist, varname, dimname="i", indices=(1, 2, 3)):
    """
    Condense variables in `vlist` into one variable named `varname`.
    In the process, individual variables in `vlist` are removed from `ds`.
    """
    ds[varname] = ds[vlist].to_array(dim=dimname).assign_coords({dimname : list(indices)})
    ds = ds.drop(vlist)
    return ds

def condense_velocities(ds, dimname="i", indices=(1, 2, 3)):
    """Condense velocity components into tensor form"""
    return condense(ds, ["u", "v", "w"], "uᵢ", dimname=dimname, indices=indices)

def condense_uw_velocities(ds, dimname="i", indices=(1, 3)):
    """Condense u and w velocity components into tensor form (for 2D simulations)"""
    return condense(ds, ["u", "w"], "uᵢ", dimname=dimname, indices=indices)
#---

#+++ Spatial derivatives
def calculate_gradient(scalar, output_name="grad_scalar", dimensions=("x_caa", "y_aca", "z_aac"), dimname="i", indices=(1, 2, 3)):
    """
    Calculate the gradient of a scalar field

    Each component ∂scalar/∂xᵢ is computed via xr.DataArray.differentiate along
    the corresponding spatial coordinate, then all components are condensed into
    a single DataArray with an extra index dimension using condense().

    Parameters
    ----------
    scalar : xr.DataArray
        Scalar field to differentiate
    output_name : str, optional
        Name for the output DataArray. Defaults to "grad_scalar"
    dimname : str, optional
        Name of the new index dimension, default "i"
    indices : list, optional
        Index values along the new dimension. Defaults to [1, 2, ..., N].

    Returns
    -------
    xr.DataArray
        Gradient components stacked along a new `dimname` dimension,
        with the same spatial dimensions as scalar
    """
    vlist = []
    for dim in dimensions:
        if dim in scalar.dims and scalar.sizes[dim] > 1:
            vlist.append(scalar.differentiate(dim))
        else:
            vlist.append(xr.zeros_like(scalar))

    aux_ds = xr.Dataset()
    for i, da in enumerate(vlist):
        aux_ds[str(i+1)] = da
    aux_ds = condense(aux_ds, list(aux_ds.data_vars.keys()), output_name, dimname=dimname, indices=indices)
    return aux_ds[output_name]
#---

#+++ Gaussian filter (unified 1D / 2D)
class GaussianFilter:
    """Unified Gaussian filter with a gcm_filters-compatible .apply() interface.

    In 2D mode: wraps gcm_filters.Filter (REGULAR grid).
    In 1D mode: applies scipy gaussian_filter1d along dims[0] with mode='wrap'
                (suitable for periodic x with y=1 simulations).

    Scale convention is consistent with gcm_filters: filter_scale = ℓ * sqrt(12),
    so the real-space Gaussian sigma = ℓ, and sigma in grid units = ℓ / dx_min.
    """
    def __init__(self, ℓ, dx_min, filter_in_2d=True):
        self.filter_in_2d = filter_in_2d
        if filter_in_2d:
            self._filter = gcm_filters.Filter(
                filter_scale=ℓ * np.sqrt(12),
                dx_min=dx_min,
                filter_shape=gcm_filters.FilterShape.GAUSSIAN,
                grid_type=gcm_filters.GridType.REGULAR,
            )
        else:
            self._sigma_grid = ℓ / dx_min

    def apply(self, da, dims):
        if self.filter_in_2d:
            return self._filter.apply(da, dims=dims)
        from scipy.ndimage import gaussian_filter1d
        return xr.apply_ufunc(
            gaussian_filter1d, da,
            input_core_dims=[[dims[0]]],
            output_core_dims=[[dims[0]]],
            kwargs={"sigma": self._sigma_grid, "axis": -1, "mode": "wrap"},
            dask="parallelized",
            output_dtypes=[da.dtype],
        )

    def __getattr__(self, name):
        if self.filter_in_2d:
            return getattr(self._filter, name)
        raise AttributeError(f"GaussianFilter has no attribute '{name}' in 1D mode")


def make_gaussian_filter(ℓ, ds, filter_in_2d):
    """Return a GaussianFilter for length scale ℓ using grid spacing from ds.

    Parameters
    ----------
    ℓ : float
        Filter length scale in physical units.
    ds : xr.Dataset
        Simulation dataset (must contain Δx_caa and Δy_aca).
    filter_in_2d : bool
        If True, filter in x and y (gcm_filters). If False, filter in x only (scipy).
    """
    if filter_in_2d:
        dx_min = float(min(ds.Δx_caa.min(), ds.Δy_aca.min()))
    else:
        dx_min = float(ds.Δx_caa.min())
    return GaussianFilter(ℓ, dx_min, filter_in_2d=filter_in_2d)


def filter_fields(ds, filter_length_scales, filter_in_2d=True):
    """Filter velocity and buoyancy fields at each length scale.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with velocity components (u,v,w or u,w) and buoyancy b.
    filter_length_scales : array-like
        Physical length scales at which to apply the filter.
    filter_in_2d : bool
        If True, filter in x and y (3D simulation). If False, filter in x only
        (xz simulation with y_aca=1).

    Returns
    -------
    ds_filt : xr.Dataset
        Dataset with filtered fields ūᵢ and b̄ at each filter_length_scale,
        plus dV (scale-independent) and the filter_ndim global attribute.
    """
    if filter_in_2d:
        ds = condense_velocities(ds, indices=(1, 2, 3))
    else:
        ds = condense_uw_velocities(ds, indices=(1, 3))

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
    ds_filt["dV"] = ds["dV"]
    ds_filt.attrs["filter_ndim"] = 2 if filter_in_2d else 1
    return ds_filt
#---

#+++ Dask-parallel filter wrapper
class DaskParallelFilter:
    """
    Thin proxy around gcm_filters.Filter that automatically chunks the input
    along the time dimension and computes with a thread pool, giving ~N×
    speedup where N is the number of available cores.

    All attributes other than `apply` are forwarded to the wrapped filter.
    """
    def __init__(self, filter_obj, chunk_size=1, n_workers=None):
        self._filter  = filter_obj
        self._chunk   = chunk_size
        self._workers = n_workers or os.cpu_count()
        print(f"  Using {self._workers} CPU workers")

    def apply(self, da, dims):
        if "time" in da.dims and da.sizes.get("time", 1) > 1:
            lazy = self._filter.apply(da.chunk({"time": self._chunk}), dims=dims)
            return lazy.compute(scheduler="threads", num_workers=self._workers)
        return self._filter.apply(da, dims=dims)

    def __getattr__(self, name):
        return getattr(self._filter, name)
#---

#+++ Pre-computed result loaders
def load_energy_transfer(filename):
    """Load the *_energy_transfer.nc file produced by 02_energy_transfer.py."""
    et_filename = str(PP_OUTPUT / (Path(filename).stem + "_energy_transfer.nc"))
    return xr.open_dataset(et_filename, decode_timedelta=False).chunk({"time": 1})
#---