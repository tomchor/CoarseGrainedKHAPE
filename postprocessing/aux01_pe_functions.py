"""
Potential energy calculation functions for Available Potential Energy (APE) analysis

This module contains functions for calculating TPE, RPE, and APE using the sorting method
following Winters et al. (1995).
"""

import numpy as np
import xarray as xr
from scipy.integrate import cumulative_trapezoid
import warnings
import concurrent.futures
from aux00_utils import integrate, calculate_gradient

# Physical constants
g = 9.81  # gravitational acceleration [m/s^2]
ρ0 = 1025  # reference density [kg/m^3]

#+++ Calculate density fields (preprocessing step)
def calculate_density_fields_from_buoyancy(ds, ρ_ref=ρ0, buoyancy_name="b", density_name="rho", compute_density_z=False):
    """
    Calculate density and related fields from buoyancy

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing buoyancy field
    ρ_ref : float, optional
        Reference density [kg/m^3], default is global ρ0
    buoyancy_name : str, optional
        Name of the buoyancy field in the dataset, default is "b"
    density_name : str, optional
        Name for the calculated density field, default is "rho"
    compute_density_z : bool, optional
        If True, also compute {density_name}_z = ρ₀·z + pe/g (≡ ρ·z) for
        cross-checking TPE via the online pe field. Default is False.

    Returns
    -------
    ds : xr.Dataset
        Dataset with added fields: {density_name}, Z, and optionally {density_name}_z
    """
    # Convert buoyancy to density
    # b = g * (ρ0 - rho) / ρ0  =>  rho = ρ0 * (1 - b/g)
    ds[density_name] = ρ_ref * (1 - ds[buoyancy_name] / g)
    if compute_density_z:
        ds[f"{density_name}_z"] = (ρ_ref * ds.z_aac + ds.pe / g)  # pe = -b*z

    # Add coordinate arrays
    if "z_aac" in ds.coords:
        ds["Z"] = ds[density_name] * 0 + ds.z_aac
    else:
        print("Warning: z_aac coordinate not found, trying to infer from data")

    return ds
#---

#+++ Vertical sort density by flattening, sorting, and reshaping
def vertical_sort_density_by_flattening(rho, dV, LxLy, test=False, z_min=0, Lz=None):
    """
    Sort the density field to obtain the reference state

    This creates a sorted density field by flattening, sorting, and reshaping.
    The sorted field represents the reference state with minimum potential energy.

    Parameters
    ----------
    rho : xr.DataArray
        3D density field (x, y, z)
    dV : xr.DataArray
        Volume of each grid cell (x, y, z)
    LxLy : xr.DataArray or float
        Horizontal area (Lx * Ly)
    test : bool
        Whether to test the sorting
    Lz : float, optional
        Total vertical extent for testing

    Returns
    -------
    vertically_sorted_ds : xr.Dataset
        1D sorted dataset with variables: rho_1d_sorted, dz_1d_sorted,
        sort_indices_1d — all indexed by the virtual z_1d_sorted coordinate
        (cell centres in the sorted reference state, running bottom to top).
    threed_sorted_ds : xr.Dataset
        3D sorted fields in the original grid shape: rho_3d_sorted,
        z_3d_sorted, sort_indices_3d.
    """
    rho_1d = np.ravel(rho.copy(), order="C")

    dz_flat = dV / LxLy # 3D DataArray with the same shape as rho
    dz_flat_1d = np.ravel(dz_flat.values, order="C")
    if test and Lz is not None:
        assert(np.isclose(dz_flat.sum().values, Lz))

    # Get the permutation indices used to sort rho_1d
    sort_indices = np.argsort(-rho_1d) # descending order since this is density

    # Sort dz_flat using the same permutation
    dz_flat_1d_sorted = dz_flat_1d[sort_indices]
    rho_1d_sorted = rho_1d[sort_indices]
    z_1d_sorted = np.cumsum(dz_flat_1d_sorted) + z_min + dz_flat_1d_sorted[0]/2

    # Reshape z_1d_sorted and rho_1d_sorted back into the original shape used by rho
    z_3d_sorted = z_1d_sorted.reshape(rho.shape, order="C")
    rho_3d_sorted = rho_1d_sorted.reshape(rho.shape, order="C")
    sort_indices_3d = sort_indices.reshape(rho.shape, order="C")

    # Put vertically sorted data into a Dataset
    sort_indices_1d = xr.DataArray(sort_indices, dims="z_1d_sorted", coords=dict(z_1d_sorted=z_1d_sorted))
    rho_1d_sorted = xr.DataArray(rho_1d_sorted, dims="z_1d_sorted", coords=dict(z_1d_sorted=z_1d_sorted))
    dz_1d_sorted = xr.DataArray(dz_flat_1d_sorted, dims="z_1d_sorted", coords=dict(z_1d_sorted=z_1d_sorted))
    vertically_sorted_ds = xr.Dataset(dict(rho_1d_sorted=rho_1d_sorted, dz_1d_sorted=dz_1d_sorted, sort_indices_1d=sort_indices_1d))

    # Put 3D sorted data into a Dataset
    sort_indices_3d = xr.DataArray(sort_indices_3d, dims=rho.dims, coords=rho.coords)
    z_3d_sorted = xr.DataArray(z_3d_sorted, dims=rho.dims, coords=rho.coords)
    rho_3d_sorted = xr.DataArray(rho_3d_sorted, dims=rho.dims, coords=rho.coords)
    threed_sorted_ds = xr.Dataset(dict(rho_3d_sorted=rho_3d_sorted, z_3d_sorted=z_3d_sorted, sort_indices_3d=sort_indices_3d))

    if test:
        rho_3d_reshaped = rho_1d.reshape(rho.shape, order="C")
        assert(np.all(rho == rho_3d_reshaped))

    return vertically_sorted_ds, threed_sorted_ds
#---

#+++ Vertical sort density by the PDF method
def vertical_sort_density_by_PDF(rho, Lz, nbins=1000):
    ρ_1d = np.ravel(rho.values, order="C")
    ε = 1e-3
    bin_edges = np.linspace(ρ_1d.min()-ε, ρ_1d.max()+ε, nbins+1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        P_rho, bins_count = np.histogram(ρ_1d, bins=bin_edges, density=True)
    bin_centers = 0.5*(bin_edges[1:] + bin_edges[:-1])
    Z_r = Lz * cumulative_trapezoid(P_rho[::-1], x=bin_centers[::-1], initial=0)
    Z_r = Z_r - Z_r.mean()
    return xr.DataArray(Z_r, dims="z_1d_sorted", coords=dict(z_1d_sorted=bin_centers))
#---

#+++ Calculate TPE
def local_TPE(rho, z_name="z_aac"):
    """
    Calculate local Total Potential Energy density

    Parameters
    ----------
    rho : xr.DataArray
        Density field

    Returns
    -------
    xr.DataArray
        Local TPE density: g * rho * z / ρ0  [m² s⁻²]
    """
    return g * rho * rho[z_name] / ρ0

def integrated_total_potential_energy(rho, dV=None, ds=None, z_name="z_aac"):
    """
    Calculate volume-integrated Total Potential Energy (TPE)

    Parameters
    ----------
    rho : xr.DataArray
        Density field
    dV : xr.DataArray, optional
        Volume element
    ds : xr.Dataset, optional
        Dataset containing dV

    Returns
    -------
    float
        Integrated TPE
    """
    if dV is None:
        if ds is not None:
            dV = ds.dV
        else:
            raise ValueError("Either dV or ds must be provided")
    tpe_local = local_TPE(rho, z_name=z_name)
    return integrate(tpe_local, dV)
#---

#+++ Calculate reference state using sorting method
def calculate_reference_potential_energy_profile(rho, dV, LxLy, test=False, z_min=0, Lz=None, sorting_method="vertically_flattened"):
    """
    Compute the adiabatically sorted reference-state density profile

    Sorts rho to obtain the minimum-PE reference state and returns it as a
    1D Dataset indexed by virtual z in the sorted state.

    Parameters
    ----------
    rho : xr.DataArray
        Density field
    dV : xr.DataArray
        Volume element
    LxLy : float
        Horizontal area
    test : bool
        Whether to run tests
    z_min : float
        Minimum z coordinate
    Lz : float, optional
        Total vertical extent for testing

    Returns
    -------
    vertically_sorted_ds : xr.Dataset
        Dataset containing sorted 1D density and coordinate
    """
    # Sort the density field to get reference state
    if sorting_method == "vertically_flattened":
        vertically_sorted_ds = vertical_sort_density_by_flattening(rho, dV, LxLy, test=test, z_min=z_min, Lz=Lz)[0]

        if test:
            assert(all(np.diff(vertically_sorted_ds.rho_1d_sorted) <= 0))
            assert(all(np.diff(vertically_sorted_ds.z_1d_sorted) > 0))
            if Lz is not None:
                assert(np.isclose(np.sum(vertically_sorted_ds.dz_1d_sorted).values, Lz))

    elif sorting_method == "PDF":
        vertically_sorted_ds = vertical_sort_density_by_PDF(rho, Lz, nbins=1000)
    else:
        raise ValueError(f"Invalid sorting method: {sorting_method}")

    return vertically_sorted_ds

def integrated_reference_potential_energy(vertically_sorted_ds, LxLy):
    """
    Calculate volume-integrated Reference Potential Energy (RPE)

    Parameters
    ----------
    vertically_sorted_ds : xr.Dataset
        Sorted density dataset from calculate_reference_potential_energy_profile
    LxLy : float
        Horizontal area

    Returns
    -------
    float
        Integrated RPE
    """
    dV_flat_1d_sorted = vertically_sorted_ds.dz_1d_sorted * LxLy
    return g * np.sum(vertically_sorted_ds.rho_1d_sorted * vertically_sorted_ds.z_1d_sorted * dV_flat_1d_sorted)

def integrated_potential_energies(ds, time_idx, test=False, sorting_method="vertically_flattened",
                                  density_name="rho", dV_name="dV", LxLy_name="LxLy",
                                  z_min_name="z_min", Lz_name="Lz", z_name="z_aac",
                                  rho_to_sort=None):
    """
    Calculate volume-integrated potential energies (APE, TPE, RPE)

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing simulation data
    time_idx : int
        Time index
    test : bool
        Whether to run tests
    sorting_method : str
        Method for sorting density
    density_name : str
        Name of density field in dataset
    dV_name : str
        Name of volume element field
    LxLy_name : str
        Name of horizontal area field/attribute
    z_min_name : str
        Name of minimum z attribute
    Lz_name : str
        Name of vertical extent attribute
    z_name : str
        Name of vertical coordinate
    rho_to_sort : xr.DataArray, optional
        Density field to use for sorting. If it has a "time" dimension it will
        be sliced at time_idx. If None, the density from the dataset is used.

    Returns
    -------
    tuple
        (APE, TPE, RPE) - all volume-integrated scalars
    """
    rho = ds[density_name].isel(time=time_idx)
    dV = ds[dV_name]
    LxLy = ds[LxLy_name]
    z_min = ds.attrs[z_min_name] if isinstance(z_min_name, str) else z_min_name
    Lz = ds.attrs[Lz_name] if isinstance(Lz_name, str) else Lz_name

    TPE = integrated_total_potential_energy(rho, dV=dV, z_name=z_name)
    _rho_to_sort = rho_to_sort.isel(time=time_idx) if (rho_to_sort is not None and "time" in rho_to_sort.dims) else (rho_to_sort if rho_to_sort is not None else rho)
    vertically_sorted_ds = calculate_reference_potential_energy_profile(
        _rho_to_sort, dV, LxLy, test=test, z_min=z_min, Lz=Lz, sorting_method=sorting_method
    )
    RPE = integrated_reference_potential_energy(vertically_sorted_ds, LxLy if isinstance(LxLy, float) else LxLy.values)
    APE = TPE - RPE
    return APE, TPE, RPE
#---

#+++ Integrated APE, TPE, RPE time series calculations
def integrated_potential_energies_timeseries(ds, test=False, verbose_level=1, sorting_method="vertically_flattened",
                                             density_name="rho", dV_name="dV", LxLy_name="LxLy",
                                             z_min_name="z_min", Lz_name="Lz", z_name="z_aac",
                                             rho_to_sort=None):
    """
    Calculate volume-integrated potential energies for all time steps

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing simulation data
    test : bool
        Whether to run tests
    verbose_level : int
        Verbosity level
    sorting_method : str
        Method for sorting density
    density_name : str
        Name of density field in dataset
    dV_name : str
        Name of volume element field
    LxLy_name : str
        Name of horizontal area field/attribute
    z_min_name : str
        Name of minimum z attribute
    Lz_name : str
        Name of vertical extent attribute
    z_name : str
        Name of vertical coordinate
    rho_to_sort : xr.DataArray, optional
        Density field to use for sorting instead of the dataset density. If it
        has a "time" dimension it will be sliced per step. If None, the density
        from the dataset is used.

    Returns
    -------
    potential_energies_ds : xr.Dataset
        Dataset containing:
        - APE: 1D DataArray (time) with APE values
        - TPE: 1D DataArray (time) with TPE values
        - RPE: 1D DataArray (time) with RPE values
    """
    if verbose_level > 0: print("Calculating potential energies time series...")

    n_times = len(ds.time)
    APE = np.zeros(n_times)
    TPE = np.zeros(n_times)
    RPE = np.zeros(n_times)

    for i in range(n_times):
        if verbose_level > 0: print(f"  Processing time step {i+1}/{n_times}", end="\r")
        APE[i], TPE[i], RPE[i] = integrated_potential_energies(
            ds, i, test=test, sorting_method=sorting_method,
            density_name=density_name, dV_name=dV_name, LxLy_name=LxLy_name,
            z_min_name=z_min_name, Lz_name=Lz_name, z_name=z_name,
            rho_to_sort=rho_to_sort,
        )

    if verbose_level > 0: print("\nDone!")

    potential_energies_ds = xr.Dataset(dict(APE=("time", APE), TPE=("time", TPE), RPE=("time", RPE)))
    potential_energies_ds["time"] = ds.time
    return potential_energies_ds
#---

#+++ Create inverse lookup table for fast z_0 retrieval (if needed)
def _create_inverse_sort_lookup(vertically_sorted_ds, verbose=False):
    """
    Create inverse lookup table for fast z_0 coordinate retrieval

    This function creates a mapping from density indices to their positions in the
    sorted array and extracts z coordinates for fast access. This avoids the slow
    .where() operation when looking up reference z coordinates.

    Parameters
    ----------
    vertically_sorted_ds : xr.Dataset
        Dataset containing sort_indices_1d and z_1d_sorted

    Returns
    -------
    inverse_sort_indices : np.ndarray
        Array mapping density_index -> position in sorted array (O(1) lookup)
    z_1d_sorted_values : np.ndarray
        Z coordinate values in sorted order for fast indexing
    """
    if verbose: print("Creating inverse lookup table for fast z_0 retrieval...", end="")
    inverse_sort_indices = np.empty(len(vertically_sorted_ds.sort_indices_1d), dtype=int)
    for i, idx in enumerate(vertically_sorted_ds.sort_indices_1d.values):
        inverse_sort_indices[int(idx)] = i
    z_1d_sorted_values = vertically_sorted_ds.z_1d_sorted.values
    if verbose: print("Done!")

    return inverse_sort_indices, z_1d_sorted_values
#---

#+++ z_0 lookup helpers
def _find_z0_xarray(ρ, z, ρ_sorted_profile):
    """
    Find the reference height z_0 for a single point (xarray inputs)

    z_0 is the height in the sorted reference state where density equals ρ,
    choosing the one closest to the actual height z. If ρ is not present
    exactly in the sorted profile (e.g. a filtered density not in the full
    sort), the position of the nearest density value is returned instead.

    Parameters
    ----------
    ρ : float
        Density at the point
    z : float
        Physical z coordinate at the point
    ρ_sorted_profile : xr.DataArray
        1D sorted density profile indexed by z_1d_sorted

    Returns
    -------
    z_0 : xr.DataArray
        Reference height (scalar DataArray with z_1d_sorted coordinate)
    """
    z_possibilities = ρ_sorted_profile.where(ρ_sorted_profile == ρ, drop=True).z_1d_sorted
    if len(z_possibilities) > 0:
        return z_possibilities[abs(z_possibilities - z).argmin()]
    idx_min = int(np.argmin(np.abs(ρ_sorted_profile.values - ρ)))
    return ρ_sorted_profile.z_1d_sorted[idx_min]


def _find_z0_numpy(ρ, z, ρ_sorted_array, z_sorted_array):
    """
    Find the reference height z_0 for a single point (numpy inputs)

    z_0 is the height in the sorted reference state where density equals ρ,
    choosing the one closest to the actual height z. If ρ is not present
    exactly in the sorted profile (e.g. a filtered density not in the full
    sort), the position of the nearest density value is returned instead.

    Parameters
    ----------
    ρ : float
        Density at the point
    z : float
        Physical z coordinate at the point
    ρ_sorted_array : np.ndarray
        1D sorted density profile values
    z_sorted_array : np.ndarray
        1D z coordinates corresponding to ρ_sorted_array

    Returns
    -------
    z_0 : float
        Reference height
    """
    mask = ρ_sorted_array == ρ
    if mask.any():
        z_possibilities = z_sorted_array[mask]
        return z_possibilities[np.abs(z_possibilities - z).argmin()]
    return z_sorted_array[np.argmin(np.abs(ρ_sorted_array - ρ))]
#---

#+++ Reference height (z_0) field
def calculate_z0_field_xarray(rho, vertically_sorted_ds, z_name="z_aac"):
    """
    Calculate the reference height z_0 at every grid point (xarray version)

    For each point in the 3D density field, z_0 is the height in the sorted
    reference state occupied by a fluid parcel of that density, choosing the
    candidate closest to the parcel's actual height z. Uses _find_z0_xarray
    internally via xr.apply_ufunc.

    Parameters
    ----------
    rho : xr.DataArray
        3D density field (x, y, z)
    vertically_sorted_ds : xr.Dataset
        Sorted reference-state dataset containing rho_1d_sorted indexed by
        z_1d_sorted
    z_name : str, optional
        Name of the vertical coordinate in rho, default "z_aac"

    Returns
    -------
    z0_field : xr.DataArray
        3D field of reference heights z_0, same shape and coordinates as rho
    """
    ρ_sorted_profile = vertically_sorted_ds.rho_1d_sorted
    z_broadcast = xr.zeros_like(rho) + rho[z_name]
    return xr.apply_ufunc(
        _find_z0_xarray,
        rho,
        z_broadcast,
        vectorize=True,
        dask="allowed",
        kwargs=dict(ρ_sorted_profile=ρ_sorted_profile),
    )


def calculate_z0_field_numpy(rho, vertically_sorted_ds, z_name="z_aac"):
    """
    Calculate the reference height z_0 at every grid point (numpy version)

    For each point in the 3D density field, z_0 is the height in the sorted
    reference state occupied by a fluid parcel of that density, choosing the
    candidate closest to the parcel's actual height z. Uses _find_z0_numpy
    internally via xr.apply_ufunc.

    Parameters
    ----------
    rho : xr.DataArray
        3D density field (x, y, z)
    vertically_sorted_ds : xr.Dataset
        Sorted reference-state dataset containing rho_1d_sorted and
        dz_1d_sorted indexed by z_1d_sorted
    z_name : str, optional
        Name of the vertical coordinate in rho, default "z_aac"

    Returns
    -------
    z0_field : xr.DataArray
        3D field of reference heights z_0, same shape and coordinates as rho
    """
    ρ_sorted_array = vertically_sorted_ds.rho_1d_sorted.values
    z_sorted_array = vertically_sorted_ds.z_1d_sorted.values
    z_broadcast = xr.zeros_like(rho) + rho[z_name]
    return xr.apply_ufunc(
        _find_z0_numpy,
        rho,
        z_broadcast,
        vectorize=True,
        dask="allowed",
        kwargs=dict(ρ_sorted_array=ρ_sorted_array, z_sorted_array=z_sorted_array),
    )


def calculate_z0_from_sort_indices(rho, vertically_sorted_ds):
    """
    O(N) z_0 calculation via inverse sort permutation.

    Valid only when rho is the *same* field that was sorted to produce
    vertically_sorted_ds (i.e. rho_to_sort == rho).  Builds the inverse
    permutation from sort_indices_1d in a single pass so there is no
    per-point Python overhead.

    Parameters
    ----------
    rho : xr.DataArray
        3D density field — must be the field that was sorted.
    vertically_sorted_ds : xr.Dataset
        Output of vertical_sort_density_by_flattening, must contain
        sort_indices_1d and z_1d_sorted.

    Returns
    -------
    z0_field : xr.DataArray
        3D field of reference heights z_0, same shape/coords as rho.
    """
    sort_idx = vertically_sorted_ds.sort_indices_1d.values  # (N,) int
    z_sorted = vertically_sorted_ds.z_1d_sorted.values      # (N,) float

    n = len(sort_idx)
    inverse = np.empty(n, dtype=np.intp)
    inverse[sort_idx] = np.arange(n, dtype=np.intp)

    z0_flat = z_sorted[inverse]
    return xr.DataArray(z0_flat.reshape(rho.shape), dims=rho.dims, coords=rho.coords)


def calculate_z0_field_vectorized(rho, vertically_sorted_ds, z_name="z_aac"):
    """
    Fully-vectorized z_0 calculation using binary search (no Python loops).

    Suitable for any density field, including filtered density not present
    in the sorted profile.  Uses np.searchsorted on the negated (ascending)
    sorted array to handle the descending density ordering.

    Parameters
    ----------
    rho : xr.DataArray
        3D density field (may differ from the sorted field).
    vertically_sorted_ds : xr.Dataset
        Sorted reference-state dataset containing rho_1d_sorted and
        z_1d_sorted.
    z_name : str, optional
        Name of the vertical coordinate in rho, default "z_aac".

    Returns
    -------
    z0_field : xr.DataArray
        3D field of reference heights z_0, same shape/coords as rho.
    """
    rho_sorted = vertically_sorted_ds.rho_1d_sorted.values  # descending
    z_sorted   = vertically_sorted_ds.z_1d_sorted.values

    rho_flat = rho.values.ravel()

    # searchsorted requires ascending; negate both arrays
    neg_rho_sorted = -rho_sorted
    neg_rho_flat   = -rho_flat

    idx = np.searchsorted(neg_rho_sorted, neg_rho_flat, side="left")
    idx = np.clip(idx, 0, len(rho_sorted) - 1)

    # Compare with left neighbour to find nearest density
    idx_left  = np.maximum(idx - 1, 0)
    dist_left = np.abs(rho_sorted[idx_left] - rho_flat)
    dist_right = np.abs(rho_sorted[idx]     - rho_flat)
    best_idx  = np.where(dist_left < dist_right, idx_left, idx)

    z0_flat = z_sorted[best_idx]
    return xr.DataArray(z0_flat.reshape(rho.shape), dims=rho.dims, coords=rho.coords)


def calculate_Upsilon(z0_field, rho, z_name="z_aac"):
    """
    Calculate the buoyancy displacement potential Υ = g (z - z_0) / ρ0

    Υ represents the work done per unit mass in displacing a parcel from its
    reference height z_0 to its actual height z, against the background
    buoyancy gradient.

    Parameters
    ----------
    z0_field : xr.DataArray
        3D field of reference heights z_0, which is where the density rho is at
        neutral buoyancy in the sorted reference state (using the original non-filtered
        density for the sorting). Same shape as rho.
    rho : xr.DataArray
        Density field, used to supply the z coordinate
    z_name : str, optional
        Name of the vertical coordinate in rho, default "z_aac"

    Returns
    -------
    xr.DataArray
        3D field of Υ values [m² s⁻²], same shape as rho
    """
    return g * (rho[z_name] - z0_field) / ρ0
#---

#+++ Local APE calculations using on-the-fly integral method
def _local_APE_on_the_fly_integral_xarray(ρ, z, z_0, vertically_sorted_ds):
    """
    Compute APE for a single point using summation method (xarray inputs)

    This function is designed to be called by xr.apply_ufunc with vectorize=True

    Parameters
    ----------
    ρ : float
        Density at the point (scalar, passed by apply_ufunc with vectorize=True)
    z : float
        Physical z coordinate at the point (scalar)
    z_0 : float
        Reference height for this point (pre-computed)
    vertically_sorted_ds : xr.Dataset
        Dataset containing sorted density profile and dz

    Notes
    -----
    ρ0 is the module-level reference density constant, not a parameter.
    """
    ρ_sorted_profile = vertically_sorted_ds.rho_1d_sorted

    # Calculate displacement and slice
    if z > z_0:
        displacement_slice = slice(z_0, z)
        signed_dz_flat = +vertically_sorted_ds.dz_1d_sorted.sel(z_1d_sorted=displacement_slice)
    else:
        displacement_slice = slice(z, z_0)
        signed_dz_flat = -vertically_sorted_ds.dz_1d_sorted.sel(z_1d_sorted=displacement_slice)

    # Calculate buoyancy difference and integrate
    ρ_sorted_profile_slice = ρ_sorted_profile.sel(z_1d_sorted=displacement_slice)

    b_l = - g * (ρ - ρ_sorted_profile_slice) / ρ0

    return -(b_l * signed_dz_flat).sum("z_1d_sorted")  # APE in Boussinesq units [m² s⁻²]



def vectorized_local_APE_on_the_fly_integral(ds0, vertically_sorted_ds, verbose=True, z_name="z_aac", z0=None):
    """
    Vectorized calculation of local APE using on-the-fly summation method for all grid points

    Parameters
    ----------
    ds0 : xr.Dataset
        Dataset at a single time containing rho field
    vertically_sorted_ds : xr.Dataset
        Dataset containing sorted density profile and dz
    verbose : bool
        Whether to print progress messages
    z_name : str
        Name of vertical coordinate
    z0 : xr.DataArray, optional
        Pre-computed reference height field (same shape as ds0.rho). If None,
        it is calculated from the sorted profile.

    Returns
    -------
    xr.DataArray
        Local APE values with same dimensions as ds0.rho
    """
    if verbose:
        print("Computing local APE using xarray on-the-fly method...")

    if z0 is None:
        z0 = calculate_z0_field_xarray(ds0.rho, vertically_sorted_ds, z_name=z_name)

    z_broadcast = xr.zeros_like(ds0.rho) + ds0[z_name]

    result = xr.apply_ufunc(
        _local_APE_on_the_fly_integral_xarray,
        ds0.rho,
        z_broadcast,
        z0,
        vectorize=True,
        dask="allowed",
        kwargs=dict(vertically_sorted_ds=vertically_sorted_ds),
    )

    if verbose: print("Done!")
    return result
#---

#+++ Local APE calculations using precomputed integral method
def _local_APE_precomputed_integral_numpy(ρ_3d, z_3d, ρ_sorted_array, z_sorted_array, cumulative_rho_dz, cumulative_dz, z_0_3d=None):
    """
    Compute APE for all points using cumulative integrals - fully vectorized, no loops!

    This is the fastest method - uses precomputed cumulative integrals so each point
    calculation becomes just array lookups and arithmetic operations.

    Strategy: Use broadcasting and vectorized operations to compute all points at once.

    Parameters
    ----------
    ρ_3d : np.ndarray
        3D density field
    z_3d : np.ndarray
        3D physical z coordinate field
    ρ_sorted_array : np.ndarray
        1D sorted density profile
    z_sorted_array : np.ndarray
        1D z coordinates of sorted profile
    cumulative_rho_dz : np.ndarray
        Precomputed cumulative integral of ρ * dz, with leading 0
    cumulative_dz : np.ndarray
        Precomputed cumulative integral of dz, with leading 0
    z_0_3d : np.ndarray, optional
        Pre-computed 3D field of reference heights z_0. If provided, the
        z_0 lookup loop is skipped entirely (significant speedup).
    """

    # Flatten inputs
    z_flat = z_3d.ravel()
    ρ_flat = ρ_3d.ravel()
    n_points = len(z_flat)

    # Find z indices for all points at once
    z_indices = np.searchsorted(z_sorted_array, z_flat)

    # Find z_0 indices — use pre-computed values if available, otherwise compute
    if z_0_3d is not None:
        z_0_indices = np.searchsorted(z_sorted_array, z_0_3d.ravel())
    else:
        z_0_indices = np.zeros(n_points, dtype=np.int32)
        unique_densities = np.unique(ρ_flat)

        for ρ_val in unique_densities:
            point_mask = (ρ_flat == ρ_val)
            if not point_mask.any():
                continue

            density_mask = (ρ_sorted_array == ρ_val)
            z_possibilities_idx = np.where(density_mask)[0]

            if len(z_possibilities_idx) == 0:
                # Density not in sorted profile (e.g. filtered ρ̄ not in full ρ sort):
                # fall back to nearest density, matching on-the-fly method behaviour
                z_0_indices[point_mask] = np.argmin(np.abs(ρ_sorted_array - ρ_val))
                continue

            z_possibilities = z_sorted_array[z_possibilities_idx]
            z_points = z_flat[point_mask]

            # Broadcasting: (n_points_with_density, n_possibilities)
            distances = np.abs(z_points[:, np.newaxis] - z_possibilities[np.newaxis, :])
            closest_possibility_idx = np.argmin(distances, axis=1)
            z_0_indices[point_mask] = z_possibilities_idx[closest_possibility_idx]

    # Now compute APE for all points at once (fully vectorized!)
    # Get cumulative integrals at z and z_0 for all points
    rho_integral_at_z = cumulative_rho_dz[z_indices]
    rho_integral_at_z0 = cumulative_rho_dz[z_0_indices]
    dz_integral_at_z = cumulative_dz[z_indices]
    dz_integral_at_z0 = cumulative_dz[z_0_indices]

    # Simpler approach: just compute absolute difference, then apply sign
    rho_integral = np.abs(rho_integral_at_z - rho_integral_at_z0)
    dz_integral = np.abs(dz_integral_at_z - dz_integral_at_z0)

    # Sign convention: positive if z > z_0, negative if z < z_0
    sign = np.where(z_indices > z_0_indices, 1.0, -1.0)
    ape_flat = sign * g * (ρ_flat * dz_integral - rho_integral) / ρ0  # Boussinesq units [m² s⁻²]

    return ape_flat.reshape(ρ_3d.shape)


def _local_APE_precomputed_integral_xarray(ρ, z, z_0, vertically_sorted_ds):
    """
    Compute APE for a single point using cumulative integral method (scalar inputs) according to the
    definition:

    E_a = g ∫_{z_0}^{z} (ρ - ρ_ref) dz

    Thus the output units are: kg m^2 s^-2 / m^3, which is the APE by unit of volume.

    Parameters
    ----------
    ρ : float
        Density at the point
    z : float
        Z coordinate at the point
    z_0 : float
        Reference height for this point (pre-computed)
    vertically_sorted_ds : xr.Dataset
        Dataset containing sorted density profile and dz

    Returns
    -------
    float
        Local APE

    Notes
    -----
    This function is designed to be called by xr.apply_ufunc with vectorize=True.
    """

    # Get cumulative integral of sorted density profile
    cumulative_ρ_sorted_integral = vertically_sorted_ds["rho_1d_sorted_cumulative_integral"]
    ρ_sorted_integral = (cumulative_ρ_sorted_integral.sel(z_1d_sorted=z, method="nearest") -
                         cumulative_ρ_sorted_integral.sel(z_1d_sorted=z_0))

    # Get cumulative integral of dz
    cumulative_dz_sorted_integral = vertically_sorted_ds["dz_1d_sorted_cumulative_integral"]
    dz_integral = (cumulative_dz_sorted_integral.sel(z_1d_sorted=z, method="nearest") -
                   cumulative_dz_sorted_integral.sel(z_1d_sorted=z_0))

    # Calculate local APE in Boussinesq units [m² s⁻²]
    ρ_constant_integral = ρ * dz_integral
    local_ape = g * (ρ_constant_integral - ρ_sorted_integral) / ρ0

    return float(local_ape)

def vectorized_local_APE_precomputed_integral(ds0, vertically_sorted_ds, use_numpy_version=True, verbose=False, z_name="z_aac", z0=None):
    """
    Vectorized calculation of local APE using cumulative integral method for all grid points

    Uses xr.apply_ufunc with vectorize=True to apply the calculation to all points
    without explicit loops.

    Parameters
    ----------
    ds0 : xr.Dataset
        Dataset at a single time containing rho field
    vertically_sorted_ds : xr.Dataset
        Dataset containing cumulative integrals of sorted density and dz
    use_numpy_version : bool
        Whether to use numpy optimized version
    verbose : bool
        Whether to print progress messages
    z_name : str
        Name of vertical coordinate
    z0 : xr.DataArray, optional
        Pre-computed reference height field (same shape as ds0.rho). If None,
        it is calculated from the sorted profile.

    Returns
    -------
    xr.DataArray
        Local APE values with same dimensions as ds0.rho
    """
    if verbose: print("Computing local APE using vectorized cumulative method (xr.apply_ufunc)...")

    if z0 is None:
        if use_numpy_version:
            z0 = calculate_z0_field_numpy(ds0.rho, vertically_sorted_ds, z_name=z_name)
        else:
            z0 = calculate_z0_field_xarray(ds0.rho, vertically_sorted_ds, z_name=z_name)

    z_broadcast = xr.zeros_like(ds0.rho) + ds0[z_name]

    if use_numpy_version:
        ρ_sorted_array = vertically_sorted_ds.rho_1d_sorted.values
        dz_sorted_array = vertically_sorted_ds.dz_1d_sorted.values
        z_sorted_array = vertically_sorted_ds.z_1d_sorted.values

        # Precompute cumulative integrals once (with leading 0)
        cumulative_rho_dz = np.concatenate([[0], np.cumsum(ρ_sorted_array * dz_sorted_array)])
        cumulative_dz = np.concatenate([[0], np.cumsum(dz_sorted_array)])

        # Call numpy function directly with 3D arrays, passing pre-computed z0
        ape_values = _local_APE_precomputed_integral_numpy(
            ds0.rho.values,
            z_broadcast.values,
            ρ_sorted_array,
            z_sorted_array,
            cumulative_rho_dz,
            cumulative_dz,
            z_0_3d=z0.values,
        )

        result = xr.DataArray(
            ape_values,
            dims=ds0.rho.dims,
            coords=ds0.rho.coords,
            name='local_ape'
        )
    else:
        result = xr.apply_ufunc(
            _local_APE_precomputed_integral_xarray,
            ds0.rho,
            z_broadcast,
            z0,
            vectorize=True,
            dask="allowed",
            kwargs={"vertically_sorted_ds": vertically_sorted_ds},
        )

    if verbose: print("Done!")
    return result
#---

#+++ Per-timestep worker (pure numpy, parallelisable)
def _process_single_timestep(
    rho_np,           # 3D np.ndarray — density to compute APE for
    rho_to_sort_np,   # 3D np.ndarray — density to sort; None when pre-sorted arrays are given
    dz_flat_np,       # 3D np.ndarray — dV/LxLy, time-independent
    z_np,             # 3D np.ndarray — z-coordinate broadcast, time-independent
    z_min,            # float
    use_inverse_sort, # bool — True when sorting and rho_np is rho_to_sort_np
    rho_sorted_1d,    # 1D np.ndarray — pre-sorted reference density; None triggers sorting
    dz_sorted_1d,     # 1D np.ndarray — pre-sorted cell heights; None triggers sorting
    z_sorted_1d,      # 1D np.ndarray — z coords of sorted state; None triggers recompute
):
    """
    Compute APE, z_0, and upsilon for one timestep.  All inputs/outputs are
    plain numpy arrays so the function can be run in a thread or process pool.
    """
    shape = rho_np.shape

    if rho_sorted_1d is not None:
        # Use pre-sorted reference state — skip sorting entirely
        rho_1d_sorted = rho_sorted_1d
        dz_1d_sorted  = dz_sorted_1d
        z_1d_sorted   = z_sorted_1d
    else:
        # Sort rho_to_sort_np to build the reference state
        rho_1d        = rho_to_sort_np.ravel()
        dz_flat_1d    = dz_flat_np.ravel()
        sort_indices  = np.argsort(-rho_1d)
        dz_1d_sorted  = dz_flat_1d[sort_indices]
        rho_1d_sorted = rho_1d[sort_indices]
        z_1d_sorted   = np.cumsum(dz_1d_sorted) + z_min + dz_1d_sorted[0] / 2

    # --- cumulative integrals for APE ---
    cumulative_rho_dz = np.concatenate([[0.0], np.cumsum(rho_1d_sorted * dz_1d_sorted)])
    cumulative_dz     = np.concatenate([[0.0], np.cumsum(dz_1d_sorted)])

    # --- z_0 ---
    if use_inverse_sort:
        # O(N): inverse permutation — only valid when rho_np came from rho_to_sort_np
        n = len(sort_indices)
        inverse = np.empty(n, dtype=np.intp)
        inverse[sort_indices] = np.arange(n, dtype=np.intp)
        z0_3d = z_1d_sorted[inverse].reshape(shape)
    else:
        # Vectorised binary search — for filtered density or pre-sorted reference
        rho_bar_flat   = rho_np.ravel()
        neg_sorted     = -rho_1d_sorted
        idx            = np.searchsorted(neg_sorted, -rho_bar_flat, side="left")
        idx            = np.clip(idx, 0, len(rho_1d_sorted) - 1)
        idx_left       = np.maximum(idx - 1, 0)
        dist_left      = np.abs(rho_1d_sorted[idx_left] - rho_bar_flat)
        dist_right     = np.abs(rho_1d_sorted[idx]      - rho_bar_flat)
        best_idx       = np.where(dist_left < dist_right, idx_left, idx)
        z0_3d          = z_1d_sorted[best_idx].reshape(shape)

    # --- APE ---
    ape_3d = _local_APE_precomputed_integral_numpy(
        rho_np, z_np, rho_1d_sorted, z_1d_sorted,
        cumulative_rho_dz, cumulative_dz,
        z_0_3d=z0_3d,
    )

    # --- upsilon: Υ = g (z - z_0) / ρ0 ---
    upsilon_3d = g * (z_np - z0_3d) / ρ0

    return ape_3d, z0_3d, upsilon_3d, rho_1d_sorted, dz_1d_sorted, z_1d_sorted
#---

#+++ Local APE and TPE time series calculations
def local_potential_energies_timeseries(ds, test=False, verbose_level=1, sorting_method="vertically_flattened",
                                        ape_method="precomputed_integral", use_numpy_version=True,
                                        density_name="rho", dV_name="dV", LxLy_name="LxLy",
                                        z_min_name="z_min", Lz_name="Lz", z_name="z_aac",
                                        rho_sorted=None, dz_sorted=None, n_workers=None):
    """
    Calculate local APE and TPE fields for all time steps

    This function computes the 3D local APE field and scalar TPE value
    for each time step in the dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing simulation data with time dimension
    test : bool
        Whether to run tests on the sorting
    verbose_level : int
        Verbosity level (0=quiet, 1=progress, 2=detailed)
    sorting_method : str
        Method for sorting density ("sorting" or "PDF")
    ape_method : str
        Method for computing local APE ("on_the_fly" or "precomputed_integral")
    use_numpy_version : bool
        Whether to use numpy-optimized version (only for on_the_fly method)
    density_name : str
        Name of density field in dataset
    dV_name : str
        Name of volume element field
    LxLy_name : str
        Name of horizontal area field/attribute
    z_min_name : str
        Name of minimum z attribute
    Lz_name : str
        Name of vertical extent attribute
    z_name : str
        Name of vertical coordinate
    rho_sorted : xr.DataArray, optional
        Pre-sorted reference density profile (time, z_1d_sorted), e.g. from a
        previous call's ``result.rho_sorted``.  When provided together with
        ``dz_sorted``, the sorting step is skipped entirely and the supplied
        arrays are used as the reference state directly.
    dz_sorted : xr.DataArray, optional
        Pre-sorted cell heights (time, z_1d_sorted), e.g. from a previous
        call's ``result.dz_sorted``.  Must be provided together with
        ``rho_sorted``.
    n_workers : int or None, optional
        Number of parallel workers for the time loop.  None means use
        os.cpu_count().  Set to 1 to disable parallelism.

    Returns
    -------
    xr.Dataset
        Dataset containing:
        - ape       : 4D DataArray (time, x, y, z) — local APE density [J m⁻³]
        - tpe       : 4D DataArray (time, x, y, z) — local TPE density g·ρ·z
        - rpe       : 2D DataArray (time, z_1d_sorted) — local RPE density in sorted state
        - rho_sorted: 2D DataArray (time, z_1d_sorted) — sorted reference density profile
        - dz_sorted : 2D DataArray (time, z_1d_sorted) — cell heights in sorted state
    """
    if verbose_level > 0: print("Calculating local APE and TPE time series...")

    n_times = len(ds.time)

    # Get grid info
    dV   = ds[dV_name]
    LxLy = ds[LxLy_name]
    z_min = ds.attrs[z_min_name] if isinstance(z_min_name, str) else z_min_name

    # Pre-extract time-independent numpy arrays to avoid repeated xarray overhead
    rho_all_np = ds[density_name].values            # (time, x, y, z)
    dz_flat_np = (dV / LxLy).values                 # (x, y, z)
    z_np       = (xr.zeros_like(ds[density_name].isel(time=0)) + ds[density_name].isel(time=0)[z_name]).values  # (x, y, z)

    # Build per-timestep argument tuples (all numpy, picklable)
    task_args = []
    if rho_sorted is not None and dz_sorted is not None:
        # Pre-sorted path: extract all numpy arrays once; skip sorting in worker
        rho_sorted_all_np = rho_sorted.values                        # (time, N)
        dz_sorted_all_np  = dz_sorted.values                         # (time, N)
        z_sorted_1d_np    = rho_sorted.coords["z_1d_sorted"].values  # (N,)
        for i in range(n_times):
            task_args.append((rho_all_np[i], None, dz_flat_np, z_np, z_min, False,
                              rho_sorted_all_np[i], dz_sorted_all_np[i], z_sorted_1d_np))
    else:
        # Self-sort path: sort ds[density_name] and use O(N) inverse permutation for z_0
        for i in range(n_times):
            task_args.append((rho_all_np[i], rho_all_np[i], dz_flat_np, z_np, z_min, True,
                              None, None, None))

    # --- parallel or serial time loop ---
    if n_workers == 1 or n_times == 1:
        results = []
        for i, a in enumerate(task_args):
            if verbose_level > 0: print(f"  Processing time step {i+1}/{n_times}", end="\r")
            results.append(_process_single_timestep(*a))
    else:
        max_w = n_workers  # None → ThreadPoolExecutor uses os.cpu_count()
        if verbose_level > 0: print(f"  Using ThreadPoolExecutor (n_workers={max_w})...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_w) as pool:
            futures = {pool.submit(_process_single_timestep, *a): i for i, a in enumerate(task_args)}
            results_unordered = {}
            for fut in concurrent.futures.as_completed(futures):
                i = futures[fut]
                results_unordered[i] = fut.result()
                if verbose_level > 0:
                    print(f"  Completed time step {len(results_unordered)}/{n_times}", end="\r")
        results = [results_unordered[i] for i in range(n_times)]

    if verbose_level > 0: print("\nDone (time loop)!")

    # --- Reassemble results into xarray ---
    rho_t0   = ds[density_name].isel(time=0)  # template for coords/dims
    coords0  = rho_t0.coords
    dims0    = rho_t0.dims

    local_ape_list        = []
    local_z0_list         = []
    local_upsilon_list    = []
    local_rho_sorted_list = []
    local_dz_sorted_list  = []

    for i, (ape_3d, z0_3d, upsilon_3d, rho_1d_sorted, dz_1d_sorted, z_1d_sorted) in enumerate(results):
        local_ape_list.append(     xr.DataArray(ape_3d,     dims=dims0, coords=coords0))
        local_z0_list.append(      xr.DataArray(z0_3d,      dims=dims0, coords=coords0))
        local_upsilon_list.append( xr.DataArray(upsilon_3d, dims=dims0, coords=coords0))
        local_rho_sorted_list.append(xr.DataArray(rho_1d_sorted, dims="z_1d_sorted",
                                                   coords=dict(z_1d_sorted=z_1d_sorted)))
        local_dz_sorted_list.append( xr.DataArray(dz_1d_sorted,  dims="z_1d_sorted",
                                                   coords=dict(z_1d_sorted=z_1d_sorted)))

    if verbose_level > 0: print("\nDone!")

    # Concatenate local APE fields along time dimension
    local_ape_4d = xr.concat(local_ape_list, dim="time")
    local_ape_4d["time"] = ds.time

    local_z0_4d = xr.concat(local_z0_list, dim="time")
    local_z0_4d["time"] = ds.time

    local_upsilon_4d = xr.concat(local_upsilon_list, dim="time")
    local_upsilon_4d["time"] = ds.time

    local_rho_sorted_4d = xr.concat(local_rho_sorted_list, dim="time")
    local_rho_sorted_4d["time"] = ds.time

    local_dz_sorted_4d = xr.concat(local_dz_sorted_list, dim="time")
    local_dz_sorted_4d["time"] = ds.time

    tpe = local_TPE(ds[density_name], z_name=z_name)
    rpe = local_TPE(local_rho_sorted_4d, z_name="z_1d_sorted")

    # Combine into a Dataset
    local_potential_energies_ds = xr.Dataset(dict(
        ape = local_ape_4d,
        z0 = local_z0_4d,
        upsilon = local_upsilon_4d,
        tpe = tpe,
        rpe = rpe,
        rho_sorted = local_rho_sorted_4d,
        dz_sorted = local_dz_sorted_4d,
    ))

    return local_potential_energies_ds
#---

#+++ Rate-of-change-of-reference-density correction term (R)
def calculate_drho_star_dt(rho_sorted):
    """
    Compute ∂ρ_*/∂t by differentiating the reference density profile in time.

    Uses xarray's differentiate (centered finite differences) along the time axis.

    Parameters
    ----------
    rho_sorted : xr.DataArray
        2D array (time, z_1d_sorted) of the reference density profile,
        e.g. local_potential_energies_ds.rho_sorted.

    Returns
    -------
    xr.DataArray
        2D array (time, z_1d_sorted) of ∂ρ_*/∂t.
    """
    Δt = rho_sorted.time.diff("time").sel(time=slice(None, None, 2))
    Δρ = rho_sorted.diff("time").sel(time=slice(None, None, 2))
    drho_star_dt = Δρ / Δt

    return drho_star_dt
#---

#+++ Reference-tendency correction R
def calculate_R_reference_tendency(z0, drho_star_dt, dz_sorted, z_name="z_aac"):
    """
    Compute the reference-tendency correction term

        R = -(g/ρ₀) ∫_{z_*(ρ)}^{z} ∂ρ_*(z̃)/∂t dz̃

    using the cumulative-integral method:

        F(z̃, t) = ∫_{z_bottom}^{z̃} (∂ρ_*/∂t)(z̃', t) dz̃'   [cumulated from bottom]
        R(x, y, z, t) = -(g/ρ₀) [F(z, t) - F(z₀(x,y,z,t), t)]

    Pass z0 = full_local_pes.z0  to get the total R,
    pass z0 = filt_local_pes.z0  to get the large-scale R_l.
    The subfilter correction is then R_s = filter(R) - R_l.

    Parameters
    ----------
    z0 : xr.DataArray
        4D reference-height field (time, x, y, z), e.g. full_local_pes.z0
        or filt_local_pes.z0.
    drho_star_dt : xr.DataArray
        2D array (time, z_1d_sorted) — ∂ρ_*/∂t from calculate_drho_star_dt().
    dz_sorted : xr.DataArray
        2D array (time, z_1d_sorted) — cell heights in sorted state,
        e.g. local_potential_energies_ds.dz_sorted.
    z_name : str
        Name of the vertical coordinate in z0.

    Returns
    -------
    xr.DataArray
        4D field of R values (time, x, y, z), same shape and coordinates as z0.
    """
    # Cumulative integral F(z̃, t) = ∫_bottom^{z̃} ∂ρ_*/∂t dz̃
    F = (drho_star_dt * dz_sorted).cumsum("z_1d_sorted")  # (time, z_1d_sorted)

    z_sorted_vals = drho_star_dt.z_1d_sorted.values  # monotonically increasing physical z

    R_list = []
    for time in drho_star_dt.time:
        z0_t = z0.sel(time=time)       # (x, y, z)
        F_t = F.sel(time=time).values  # 1D, length nz_sorted

        # Physical z of each grid cell, broadcast to full (x, y, z) shape
        z_3d = (xr.zeros_like(z0_t) + z0_t[z_name]).values.ravel()
        z0_flat = z0_t.values.ravel()

        # Nearest-index lookup in the sorted z grid
        iz  = np.clip(np.searchsorted(z_sorted_vals, z_3d),   0, len(F_t) - 1)
        iz0 = np.clip(np.searchsorted(z_sorted_vals, z0_flat), 0, len(F_t) - 1)

        R_flat = -(g / ρ0) * (F_t[iz] - F_t[iz0])
        R_list.append(xr.DataArray(
            R_flat.reshape(z0_t.shape),
            dims=z0_t.dims,
            coords=z0_t.coords,
        ))

    R = xr.concat(R_list, dim="time")
    R["time"] = drho_star_dt.time
    return R
#---

#+++ SFS APE tendency
def calculate_sfs_ape_tendency(subfilter_local_ape):
    """
    Compute ∂Eaˢ/∂t as a centred finite difference in time.

    Parameters
    ----------
    subfilter_local_ape : xr.DataArray
        4D subfilter APE field (time, x, y, z),
        e.g. filter(full_local_pes.ape) - filt_local_pes.ape.

    Returns
    -------
    xr.DataArray
        4D tendency field on the staggered (mid-point) time grid.
    """
    Δt = subfilter_local_ape.time.diff("time").sel(time=slice(None, None, 2))
    ΔE = subfilter_local_ape.diff("time").sel(time=slice(None, None, 2))
    return ΔE / Δt
#---

#+++ SFS reference-tendency correction R_s
def calculate_sfs_R_correction(full_rho_sorted, full_z0, filt_z0, full_dz_sorted,
                                filter, filter_dims=["x_caa", "y_aca"], z_name="z_aac"):
    """
    Compute the subfilter reference-tendency correction

        R_s = filter(R) - R_l

    where:
        R   = -(g/ρ₀) ∫_{z_*(ρ)}^{z}  ∂ρ_*/∂t dz̃   (total,      uses full z₀)
        R_l = -(g/ρ₀) ∫_{z_*(ρ̄)}^{z} ∂ρ_*/∂t dz̃   (large-scale, uses filtered z₀)

    Parameters
    ----------
    full_rho_sorted : xr.DataArray
        2D reference density profile (time, z_1d_sorted),
        e.g. full_local_pes.rho_sorted.
    full_z0 : xr.DataArray
        4D reference-height field using the full density (time, x, y, z),
        e.g. full_local_pes.z0.
    filt_z0 : xr.DataArray
        4D reference-height field using the filtered density (time, x, y, z),
        e.g. filt_local_pes.z0.
    full_dz_sorted : xr.DataArray
        2D cell heights in sorted state (time, z_1d_sorted),
        e.g. full_local_pes.dz_sorted.
    filter : gcm_filters.Filter
        Filter object used for the spatial filtering operation.
    filter_dims : list of str
        Spatial dimensions along which to apply the filter.
    z_name : str
        Name of the vertical coordinate in z0.

    Returns
    -------
    xr.DataArray
        4D subfilter correction R_s (time, x, y, z).
    """
    drho_star_dt = calculate_drho_star_dt(full_rho_sorted)
    R_full = calculate_R_reference_tendency(full_z0, drho_star_dt, full_dz_sorted, z_name=z_name)
    R_l    = calculate_R_reference_tendency(filt_z0, drho_star_dt, full_dz_sorted, z_name=z_name)
    return filter.apply(R_full, dims=filter_dims) - R_l
#---

#+++ SFS flux tensor (general)
def calculate_sfs_flux_tensor(a, b, filter, filter_dims=["x_caa", "y_aca"],
                              filtered_a=None, filtered_b=None):
    """
    Calculate the SFS flux tensor filtered(a·b) - filtered(a)·filtered(b)

    This is the general building block for subfilter-scale flux quantities: it
    measures the covariance between a and b at scales smaller than the filter
    width.

    Parameters
    ----------
    a : xr.DataArray
        First (unfiltered) field
    b : xr.DataArray
        Second (unfiltered) field
    filter : gcm_filters.Filter
        Filter object used to apply the spatial filtering operation
    filter_dims : list of str
        Spatial dimensions along which to apply the filter
    filtered_a : xr.DataArray, optional
        Pre-computed filtered(a). If None, it is computed from a.
    filtered_b : xr.DataArray, optional
        Pre-computed filtered(b). If None, it is computed from b.

    Returns
    -------
    xr.DataArray
        SFS flux tensor filtered(a·b) - filtered(a)·filtered(b)
    """
    if filtered_a is None:
        filtered_a = filter.apply(a, dims=filter_dims)

    if filtered_b is None:
        filtered_b = filter.apply(b, dims=filter_dims)

    return filter.apply(a * b, dims=filter_dims) - filtered_a * filtered_b
#---

#+++ Subfilter stress tensor
def calculate_subfilter_tracer_flux(rho, u_i, filter, filter_dims=["x_caa", "y_aca"],
                                    filtered_density=None, filtered_velocity_vector=None):
    """
    Calculate the subfilter stress tensor τᵢ = filtered(ρ uᵢ) - filtered(ρ) filtered(uᵢ)

    This represents the subfilter momentum flux arising from correlations between
    density and velocity fluctuations at scales smaller than the filter width.

    Parameters
    ----------
    rho : xr.DataArray
        Full (unfiltered) density field
    u_i : xr.DataArray
        Full (unfiltered) velocity vector with an "i" dimension indexing the
        three components (shape: i × time × z × y × x), as produced by
        condense_velocities()
    filter : gcm_filters.Filter
        Filter object used to apply the spatial filtering operation
    filter_dims : list of str
        Spatial dimensions along which to apply the filter
    filtered_density : xr.DataArray, optional
        Pre-computed filtered(ρ). If None, it is computed by applying
        filter to rho.
    filtered_velocity_vector : xr.DataArray, optional
        Pre-computed filtered(uᵢ). If None, it is computed by applying
        filter to u_i.

    Returns
    -------
    xr.DataArray
        Subfilter stress τᵢ [kg m⁻² s⁻¹] with the same dimensions as u_i
    """
    tau_i = calculate_sfs_flux_tensor(rho, u_i, filter,
                                      filter_dims=filter_dims,
                                      filtered_a=filtered_density,
                                      filtered_b=filtered_velocity_vector)
    tau_i.name = "τᵢ"
    return tau_i
#---

#+++ KE-APE exchange term
def calculate_ape_to_ke_exchange_term(w, b, filter, filter_dims=["x_caa", "y_aca"],
                                      filtered_w=None, filtered_b=None):
    """
    Calculate the SFS KE->APE exchange term +(filtered(w·b) - filtered(w)·filtered(b))

    This represents the SFS flux of KE to APE: the rate at which small-scale KE is converted to APE.

    Parameters
    ----------
    w : xr.DataArray
        Full (unfiltered) vertical velocity field
    b : xr.DataArray
        Full (unfiltered) buoyancy field
    filter : gcm_filters.Filter
        Filter object used to apply the spatial filtering operation
    filter_dims : list of str
        Spatial dimensions along which to apply the filter
    filtered_w : xr.DataArray, optional
        Pre-computed filtered(w). If None, it is computed from w.
    filtered_b : xr.DataArray, optional
        Pre-computed filtered(b). If None, it is computed from b.

    Returns
    -------
    xr.DataArray
        SFS KE->APE exchange term +(filtered(w·b) - filtered(w)·filtered(b))
    """
    result = calculate_sfs_flux_tensor(w, b, filter,
                                       filter_dims=filter_dims,
                                       filtered_a=filtered_w,
                                       filtered_b=filtered_b)
    result.name = "SFS KE->APE exchange"
    return result
#---

#+++ SFS APE dissipation
def calculate_sfs_ape_dissipation(rho, upsilon, upsilon_l, kappa, filter,
                                  filter_dims=["x_caa", "y_aca"],
                                  filtered_density=None, index_dim="i"):
    """
    Calculate the SFS APE dissipation ε_s = filtered(κ ∇ρ · ∇Υ) - κ ∇ρ̄ · ∇Υˡ

    The SFS APE dissipation quantifies the removal of large-scale APE by
    subfilter-scale diffusive processes:

        ε_s = filtered(κ ∇ρ · ∇Υ) - κ ∇ρ̄ · ∇Υˡ

    where:
        Υ  = g (z - z_*(ρ)) / ρ₀   — displacement potential using full density
        Υˡ = g (z - z_*(ρ̄)) / ρ₀   — displacement potential using filtered density
        κ  — diffusivity field

    Parameters
    ----------
    rho : xr.DataArray
        Full (unfiltered) density field ρ
    upsilon : xr.DataArray
        Buoyancy displacement potential Υ(ρ, z) = g(z - z_*(ρ))/ρ₀, computed
        from the full density sort (full_local_potential_energies.upsilon)
    upsilon_l : xr.DataArray
        Large-scale displacement potential Υˡ(ρ̄, z) = g(z - z_*(ρ̄))/ρ₀,
        computed from the filtered density sort (filt_local_potential_energies.upsilon)
    kappa : xr.DataArray
        Diffusivity field κ (e.g. ds.κ_e from SmagorinskyLilly)
    filter : gcm_filters.Filter
        Filter object used to apply the spatial filtering operation
    filter_dims : list of str
        Spatial dimensions along which to apply the filter
    filtered_density : xr.DataArray, optional
        Pre-computed filtered density ρ̄. If None, it is computed by applying
        filter to rho.
    index_dim : str, optional
        Name of the vector index dimension, default "i"

    Returns
    -------
    xr.DataArray
        SFS APE dissipation ε_s [J m⁻³ s⁻¹] with the same spatial dimensions as rho
    """
    # Term 1: filtered(κ ∇ρ · ∇Υ)
    grad_rho = calculate_gradient(rho)
    grad_upsilon = calculate_gradient(upsilon)
    kappa_grad_dot = kappa * (grad_rho * grad_upsilon).sum(dim=index_dim)
    term1 = filter.apply(kappa_grad_dot, dims=filter_dims)

    # Term 2: κ ∇ρ̄ · ∇Υˡ
    if filtered_density is None:
        filtered_density = filter.apply(rho, dims=filter_dims)
    grad_rho_bar = calculate_gradient(filtered_density)
    grad_upsilon_l = calculate_gradient(upsilon_l)
    term2 = kappa * (grad_rho_bar * grad_upsilon_l).sum(dim=index_dim)

    return term1 - term2
#---

#+++ Cross-scale APE flux
def calculate_cross_scale_ape_flux(rho, u_i, upsilon, filter, filter_dims=["x_caa", "y_aca"],
                                    filtered_density=None, filtered_velocity_vector=None,
                                    index_dim="i"):
    """
    Calculate the cross-scale APE flux Π = -τᵢ · ∇Υ

    The cross-scale APE flux quantifies the transfer of APE across the filter
    scale via the contraction of the subfilter tracer flux τᵢ with the gradient
    of the buoyancy displacement potential Υ = g(z - z₀)/ρ₀:

        Π = -(filtered(ρ uᵢ) - filtered(ρ) filtered(uᵢ)) · ∂Υ/∂xᵢ

    Parameters
    ----------
    rho : xr.DataArray
        Full (unfiltered) density field
    u_i : xr.DataArray
        Full (unfiltered) velocity vector with an "i" index dimension, as
        produced by condense_velocities()
    upsilon : xr.DataArray
        Buoyancy displacement potential Υ = g(z - z₀)/ρ₀, typically taken
        from the filtered potential energies dataset
    filter : gcm_filters.Filter
        Filter object used to apply the spatial filtering operation
    filter_dims : list of str
        Spatial dimensions along which to apply the filter
    filtered_density : xr.DataArray, optional
        Pre-computed filtered(ρ). Passed through to calculate_subfilter_tracer_flux.
    filtered_velocity_vector : xr.DataArray, optional
        Pre-computed filtered(uᵢ). Passed through to calculate_subfilter_tracer_flux.
    index_dim : str, optional
        Name of the vector index dimension, default "i"

    Returns
    -------
    xr.DataArray
        Cross-scale APE flux Π [J m⁻³ s⁻¹] with the same spatial dimensions
        as rho (the i dimension is summed over)
    """
    tau_i = calculate_subfilter_tracer_flux(
        rho, u_i, filter, filter_dims,
        filtered_density=filtered_density,
        filtered_velocity_vector=filtered_velocity_vector,
    )
    grad_upsilon = calculate_gradient(upsilon)
    return -(tau_i * grad_upsilon).sum(dim=index_dim)
#---
