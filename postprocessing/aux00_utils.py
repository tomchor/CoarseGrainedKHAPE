import os
from pathlib import Path
import numpy as np
import xarray as xr

PP_OUTPUT = Path(__file__).resolve().parent / "output"

#+++ Integrations and sums
def integrate(da, dV, dims=("x_caa", "y_aca", "z_aac")):
    """Integrate a DataArray over spatial dimensions"""
    return (da * dV).sum(dims)
#---

#+++ Load data
def _pad_domain_in_z(ds):
    """Extend the z domain to twice its original height using edge-value padding.

    Adds Nz//2 cells at the bottom (each filled with that field's bottom boundary
    value) and Nz//2 cells at the top (each filled with the top boundary value),
    doubling the domain height. Assumes a uniform z grid. Δz_aac is extended with
    the same constant dz; dV and z-extent attributes are recomputed.
    """
    Nz     = ds.sizes["z_aac"]
    Nz_pad = Nz // 2
    dz     = float(ds.Δz_aac.isel(z_aac=0))

    z_orig = ds.z_aac.values
    z_bot  = z_orig[0]  - np.arange(Nz_pad, 0, -1) * dz
    z_top  = z_orig[-1] + np.arange(1, Nz_pad + 1) * dz
    z_new  = np.concatenate([z_bot, z_orig, z_top])

    new_vars = {}
    for name, da in ds.data_vars.items():
        if name in {"Δz_aac", "dV"} or "z_aac" not in da.dims:
            new_vars[name] = da
            continue
        # Use actual boundary values: multiply a zero slab by 0 then add boundary scalar
        bot_slab = (da.isel(z_aac=slice(None, Nz_pad)) * 0
                    + da.isel(z_aac=0)).assign_coords(z_aac=z_bot)
        top_slab = (da.isel(z_aac=slice(-Nz_pad, None)) * 0
                    + da.isel(z_aac=-1)).assign_coords(z_aac=z_top)
        new_vars[name] = xr.concat([bot_slab, da, top_slab], dim="z_aac")

    new_vars["Δz_aac"] = xr.DataArray(
        np.full(len(z_new), dz), dims=["z_aac"],
        coords={"z_aac": z_new}, attrs=ds["Δz_aac"].attrs,
    )

    other_coords = {k: v for k, v in ds.coords.items() if k != "z_aac"}
    ds_new = xr.Dataset(new_vars, coords={**other_coords, "z_aac": z_new}, attrs=ds.attrs)
    ds_new["dV"] = ds_new.Δx_caa * ds_new.Δy_aca * ds_new.Δz_aac

    ds_new.attrs["z_min"] = float(z_new[0])  - dz / 2
    ds_new.attrs["z_max"] = float(z_new[-1]) + dz / 2
    ds_new.attrs["Lz"]    = ds_new.attrs["z_max"] - ds_new.attrs["z_min"]

    return ds_new


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
        Dataset with grid information added as attributes and variables,
        with the z domain extended to 2x its original height by padding each
        field with its bottom/top edge values and extending z coordinates by
        the uniform grid spacing.
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

    # Pad domain in z (double height using boundary values of each field)
    ds = _pad_domain_in_z(ds)

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

#+++ Gaussian filter (x: periodic, z: bounded)
# FWHM = 2√(2 ln 2) · σ  →  σ = FWHM / (2√(2 ln 2))
_FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))

class GaussianFilter:
    """Gaussian filter in x (periodic) and z (bounded) directions.

    Two sequential 1D scipy Gaussian convolutions:
      - x: mode='wrap'    — periodic BC
      - z: mode='nearest' — extends with boundary value beyond domain walls

    ℓ is the FWHM of the kernel; σ = ℓ · _FWHM_TO_SIGMA is derived internally.
    """
    def __init__(self, ℓ, dx_min, dz_min):
        self._sigma_x = ℓ * _FWHM_TO_SIGMA / dx_min
        self._sigma_z = ℓ * _FWHM_TO_SIGMA / dz_min

    def apply(self, da, dims):
        """Apply filter in dims[0] (x, periodic) then dims[1] (z, bounded).

        Parameters
        ----------
        da : xr.DataArray
        dims : list of str
            [x_dim, z_dim], e.g. ['x_caa', 'z_aac']
        """
        from scipy.ndimage import gaussian_filter1d
        x_dim, z_dim = dims
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
            input_core_dims=[[z_dim]],
            output_core_dims=[[z_dim]],
            kwargs={"sigma": self._sigma_z, "axis": -1, "mode": "nearest"},
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
        Simulation dataset (must contain Δx_caa and Δz_aac).
    """
    dx_min = float(ds.Δx_caa.min())
    dz_min = float(ds.Δz_aac.min())
    return GaussianFilter(ℓ, dx_min, dz_min)


def filter_fields(ds, filter_length_scales):
    """Filter velocity and buoyancy fields at each length scale in x and z.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with velocity components (u, w) and buoyancy b.
    filter_length_scales : array-like
        Filter length scales (FWHM) in physical units.

    Returns
    -------
    ds_filt : xr.Dataset
        Dataset with filtered fields ūᵢ and b̄ at each filter_length_scale,
        plus dV (scale-independent).
    """
    ds = condense_uw_velocities(ds, indices=(1, 3))

    ds_filt_list = []
    for ℓ in filter_length_scales:
        print(f"  filter_length_scale = {ℓ:.4f}...")
        gf = make_gaussian_filter(ℓ, ds)
        ds_filt_list.append(xr.Dataset({
            "ūᵢ": gf.apply(ds["uᵢ"], dims=["x_caa", "z_aac"]),
            "b̄":  gf.apply(ds["b"],  dims=["x_caa", "z_aac"]),
        }))

    scale_coord = xr.DataArray(filter_length_scales, dims="filter_length_scale",
                               name="filter_length_scale")
    ds_filt = xr.concat(ds_filt_list, dim=scale_coord)
    ds_filt["dV"] = ds["dV"]
    ds_filt.attrs.update(ds.attrs)
    ds_filt.attrs["filter_dims"] = "x_caa,z_aac"
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