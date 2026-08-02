import os
from pathlib import Path
import numpy as np
import xarray as xr
import dask
from dask.diagnostics.progress import ProgressBar

PP_OUTPUT = Path(__file__).resolve().parent.parent / "output"

#+++ Integrations and sums
def integrate(da, dV, dims=("x_caa", "y_aca", "z_aac")):
    """Integrate a DataArray over spatial dimensions"""
    return (da * dV).sum(dims)
#---

#+++ Write a (possibly still dask-lazy) Dataset to netCDF
def write_dataset(ds, path, write_mode="load"):
    """Write ds to path, avoiding the dask-lazy .to_netcdf() write hang: writing a Dataset that still has
    unevaluated dask arrays computes them via dask's default *threaded* scheduler *during* the write, with
    multiple threads writing into the same HDF5 file handle -- a known hang risk, since the underlying HDF5
    C library isn't reliably thread-safe for it (see local_potential_energies_timeseries() in
    aux01_pe_functions.py for a real observed stall of this kind).

    write_mode="load" (default): eagerly .load() the full Dataset into memory before writing -- fast, but
    requires the whole thing to fit in memory at once (OOMs at large resolutions, e.g. 512x512x128).
    write_mode="synchronous": force dask's single-threaded synchronous scheduler for just the write, which
    still streams/computes chunk-by-chunk (bounded memory, like the original streaming design) but avoids
    the multi-threaded-write hang. Measured ~30% slower than "load" for 01_filter_fields.py at 256x256x64
    (see test_write_scheduler_timing.pbs) -- a real but modest cost next to a job that otherwise OOMs.
    """
    if write_mode == "load":
        with ProgressBar(minimum=5, dt=5):
            ds = ds.load()
        ds.to_netcdf(path)
    elif write_mode == "synchronous":
        with dask.config.set(scheduler="synchronous"):
            with ProgressBar(minimum=5, dt=5):
                ds.to_netcdf(path)
    else:
        raise ValueError(f"write_mode must be 'load' or 'synchronous', got {write_mode!r}")
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
        Dataset with grid information added as attributes and variables.
    """
    print(f"Loading data from {filename}...")
    ds = xr.open_dataset(filename, decode_times=False, chunks={})
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
    ds = ds.drop_vars(vlist)
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


#+++ Gaussian filter (horizontal: x and y, both periodic)
# FWHM = 2√(2 ln 2) · σ  →  σ = FWHM / (2√(2 ln 2))
_FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))

class GaussianFilter:
    """Gaussian filter over the horizontal directions x and y (both periodic).

    Two sequential 1D scipy Gaussian convolutions, each with mode='wrap' (periodic
    BC), matching the doubly-periodic-horizontal channel topology. The vertical
    (z) direction is never filtered -- horizontal scales span the mesoscale/
    submesoscale range this budget targets, while z has its own distinct
    structure (stratification, boundary layers) that shouldn't be smoothed over.

    ℓ is the FWHM of the kernel; σ = ℓ · _FWHM_TO_SIGMA is derived internally,
    independently per direction (dx and dy need not be equal).
    """
    def __init__(self, ℓ, dx_min, dy_min):
        self._sigma_x = ℓ * _FWHM_TO_SIGMA / dx_min
        self._sigma_y = ℓ * _FWHM_TO_SIGMA / dy_min

    def apply(self, da, dims):
        """Apply filter in dims[0] (x, periodic) then dims[1] (y, periodic).

        Parameters
        ----------
        da : xr.DataArray
        dims : list of str
            [x_dim, y_dim], e.g. ['x_caa', 'y_aca']
        """
        from scipy.ndimage import gaussian_filter1d
        x_dim, y_dim = dims
        da_x = xr.apply_ufunc(
            gaussian_filter1d, da,
            input_core_dims=[[x_dim]],
            output_core_dims=[[x_dim]],
            kwargs={"sigma": self._sigma_x, "axis": -1, "mode": "wrap"},
            dask="parallelized",
            output_dtypes=[da.dtype],
            dask_gufunc_kwargs={"allow_rechunk": True},
        )
        return xr.apply_ufunc(
            gaussian_filter1d, da_x,
            input_core_dims=[[y_dim]],
            output_core_dims=[[y_dim]],
            kwargs={"sigma": self._sigma_y, "axis": -1, "mode": "wrap"},
            dask="parallelized",
            output_dtypes=[da_x.dtype],
            dask_gufunc_kwargs={"allow_rechunk": True},
        )


def make_gaussian_filter(ℓ, ds):
    """Return a GaussianFilter for FWHM ℓ using grid spacing from ds.

    Parameters
    ----------
    ℓ : float
        Filter length scale (FWHM) in physical units.
    ds : xr.Dataset
        Simulation dataset (must contain Δx_caa and Δy_aca).
    """
    dx_min = float(ds.Δx_caa.min())
    dy_min = float(ds.Δy_aca.min())
    return GaussianFilter(ℓ, dx_min, dy_min)


def filter_fields(ds, filter_scales):
    """Filter velocity and buoyancy fields at each length scale, horizontally (x, y).

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with velocity components (u, v, w) and buoyancy b.
    filter_scales : array-like
        Filter length scales (FWHM) in physical units.

    Returns
    -------
    ds_filt : xr.Dataset
        Dataset with filtered fields ūᵢ (all 3 components -- w̄ is needed by the
        APE<->KE exchange term even though the KE cross-scale flux itself is
        restricted to horizontal components downstream) and b̄ at each
        filter_scale, plus dV (scale-independent).
    """
    ds = condense_velocities(ds, indices=(1, 2, 3))

    ds_filt_list = []
    for ℓ in filter_scales:
        print(f"  filter_scale = {ℓ:.4f}...")
        gf = make_gaussian_filter(ℓ, ds)
        ds_filt_list.append(xr.Dataset({
            "ūᵢ": gf.apply(ds["uᵢ"], dims=["x_caa", "y_aca"]),
            "b̄":  gf.apply(ds["b"],  dims=["x_caa", "y_aca"]),
        }))

    scale_coord = xr.DataArray(filter_scales, dims="filter_scale",
                               name="filter_scale")
    ds_filt = xr.concat(ds_filt_list, dim=scale_coord)
    ds_filt["dV"] = ds["dV"]
    ds_filt.attrs.update(ds.attrs)
    ds_filt.attrs["filter_dims"] = "x_caa,y_aca"
    return ds_filt
#---

#+++ Dask-parallel filter wrapper
class DaskParallelFilter:
    """
    Thin proxy around a GaussianFilter that automatically chunks the input
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
def load_energy_transfer(filename, ref_suffix=""):
    """Load the *_energy_transfer.nc file produced by 03_energy_transfer.py."""
    et_filename = str(PP_OUTPUT / (Path(filename).stem + f"_energy_transfer{ref_suffix}.nc"))
    return xr.open_dataset(et_filename, decode_timedelta=False).chunk({"time": 1})
#---