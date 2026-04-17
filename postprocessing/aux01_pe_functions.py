"""
Potential energy calculation functions for Available Potential Energy (APE) analysis

This module contains functions for calculating TPE, RPE, and APE using the sorting method
following Winters et al. (1995).
"""

import numpy as np
import xarray as xr
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

#+++ Calculate relative buoyancy b_r
def calculate_b_r(rho, rho_sorted, z_name="z_aac"):
    """
    Calculate relative buoyancy b_r = -g(ρ - ρ_*(z)) / ρ₀.

    ρ_*(z) is the sorted reference density evaluated at each vertical level,
    obtained by interpolating the sorted profile onto the domain z grid.

    Parameters
    ----------
    rho : xr.DataArray
        Full density field, with vertical coordinate z_name.
    rho_sorted : xr.DataArray
        Sorted reference density profile (time, z_1d_sorted).
    z_name : str
        Name of the vertical coordinate in rho.

    Returns
    -------
    xr.DataArray
        b_r with the same shape as rho [m s⁻²].
    """
    z_target     = rho.coords[z_name].values          # (Nz,) — target z grid
    z_sorted_vals = rho_sorted.coords["z_1d_sorted"].values  # (N,) — sorted z positions
    Nz = len(z_target)

    def _interp(rho_sorted_1d):
        return np.interp(z_target, z_sorted_vals, rho_sorted_1d)

    rho_star = xr.apply_ufunc(
        _interp,
        rho_sorted,
        input_core_dims=[["z_1d_sorted"]],
        output_core_dims=[[z_name]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[rho.dtype],
        dask_gufunc_kwargs={"output_sizes": {z_name: Nz}},
    ).assign_coords({z_name: rho.coords[z_name]})

    b_r = -g * (rho - rho_star) / ρ0
    b_r.name = "b_r"
    return b_r
#---

#+++ Calculate relative buoyancy b_r (simple xarray interpolation)
def calculate_b_r_simple(rho, rho_sorted, z_name="z_aac"):
    """
    Calculate relative buoyancy b_r = -g(ρ - ρ_*(z)) / ρ₀.

    Uses xarray .rename/.interp to map the sorted reference profile onto the
    domain z grid. Simpler than calculate_b_r but requires the sorted profile
    coordinate to be named 'z_1d_sorted'.

    Parameters
    ----------
    rho : xr.DataArray
        Full density field, with vertical coordinate z_name.
    rho_sorted : xr.DataArray
        Sorted reference density profile with coordinate 'z_1d_sorted'.
    z_name : str
        Name of the vertical coordinate in rho.

    Returns
    -------
    xr.DataArray
        b_r with the same shape as rho [m s⁻²].
    """
    rho_star = rho_sorted.rename(z_1d_sorted=z_name).interp({z_name: rho[z_name]})
    b_r = -g * (rho - rho_star) / ρ0
    b_r.name = "b_r"
    return b_r
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
#---

#+++ Sort-only helpers
def _sort_single_timestep(rho_np, dz_flat_np, z_min):
    """Return (rho_1d_sorted, dz_1d_sorted, z_1d_sorted) for one timestep."""
    rho_1d        = rho_np.ravel()
    dz_flat_1d    = dz_flat_np.ravel()
    sort_indices  = np.argsort(-rho_1d)
    dz_1d_sorted  = dz_flat_1d[sort_indices]
    rho_1d_sorted = rho_1d[sort_indices]
    z_1d_sorted   = np.cumsum(dz_1d_sorted) + z_min - dz_1d_sorted[0] / 2
    return rho_1d_sorted, dz_1d_sorted, z_1d_sorted


def sorted_timeseries(ds, field_to_sort="rho", dV_name="dV", LxLy_name="LxLy",
                      z_min_name="z_min", n_workers=None, verbose_level=1,
                      fixed_reference=False):
    """
    Compute the sorted reference-density profile for every timestep.

    Only performs the adiabatic sorting step; no APE or z_0 calculation.
    Use this when you need rho_sorted / dz_sorted to pass to
    local_potential_energies_timeseries() or calculate_energy_transfer(),
    avoiding redundant sorts across multiple callers.

    Parameters
    ----------
    ds : xr.Dataset
        Must contain field_to_sort, dV_name, LxLy_name, and z_min_name attribute.
    field_to_sort : str
        Name of the field to sort.
    dV_name, LxLy_name : str
        Names of cell-volume and horizontal-area fields.
    z_min_name : str
        Attribute name for the minimum z value.
    n_workers : int or None
        Thread-pool workers; None uses os.cpu_count(); 1 disables parallelism.
    verbose_level : int
        0 = quiet, 1 = progress messages.
    fixed_reference : bool
        If True, sort only the first time step (t=0) and broadcast that
        reference profile to all time steps, yielding a time-invariant
        reference state.  If False (default), sort each time step
        independently.

    Returns
    -------
    xr.Dataset with:
        rho_sorted : (time, z_1d_sorted) — sorted reference density
        dz_sorted  : (time, z_1d_sorted) — sorted cell heights
    """
    n_times  = len(ds.time)
    z_min    = ds.attrs[z_min_name] if isinstance(z_min_name, str) else z_min_name
    rho_all  = ds[field_to_sort].values                    # (time, …)
    dz_flat  = (ds[dV_name] / ds[LxLy_name]).values       # (…)

    if fixed_reference:
        if verbose_level > 0:
            print("  Sorting t=0 only (fixed reference profile)...")
        t0_result = _sort_single_timestep(rho_all[0], dz_flat, z_min)
        results = [t0_result] * n_times
    else:
        def _run(i):
            if verbose_level > 0:
                print(f"  Sorting time step {i+1}/{n_times}", end="\r")
            return _sort_single_timestep(rho_all[i], dz_flat, z_min)

        if n_workers == 1 or n_times == 1:
            results = [_run(i) for i in range(n_times)]
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = {pool.submit(_sort_single_timestep, rho_all[i], dz_flat, z_min): i
                           for i in range(n_times)}
                results_unordered = {}
                for fut in concurrent.futures.as_completed(futures):
                    i = futures[fut]
                    results_unordered[i] = fut.result()
                    if verbose_level > 0:
                        print(f"  Sorted time step {len(results_unordered)}/{n_times}", end="\r")
            results = [results_unordered[i] for i in range(n_times)]

    if verbose_level > 0:
        print("\nDone!")

    rho_sorted_list, dz_sorted_list = [], []
    for rho_1d_sorted, dz_1d_sorted, z_1d_sorted in results:
        coord = dict(z_1d_sorted=z_1d_sorted)
        rho_sorted_list.append(xr.DataArray(rho_1d_sorted, dims="z_1d_sorted", coords=coord))
        dz_sorted_list.append(xr.DataArray(dz_1d_sorted, dims="z_1d_sorted", coords=coord))

    rho_sorted_da = xr.concat(rho_sorted_list, dim="time").assign_coords(time=ds.time)
    dz_sorted_da  = xr.concat(dz_sorted_list,  dim="time").assign_coords(time=ds.time)

    return xr.Dataset(dict(rho_sorted=rho_sorted_da, dz_sorted=dz_sorted_da))
#---

#+++ Per-timestep worker (pure numpy, parallelisable)
def _process_single_timestep(rho_np, z_np, rho_sorted_1d, dz_sorted_1d, z_sorted_1d):
    """
    Compute APE, z_0, and upsilon for one timestep.  All inputs/outputs are
    plain numpy arrays so the function can be run in a thread or process pool.
    """
    shape = rho_np.shape

    # --- cumulative integrals for APE ---
    cumulative_rho_dz = np.concatenate([[0.0], np.cumsum(rho_sorted_1d * dz_sorted_1d)])
    cumulative_dz     = np.concatenate([[0.0], np.cumsum(dz_sorted_1d)])

    # --- z_0: vectorised binary search ---
    rho_flat   = rho_np.ravel()
    neg_sorted = -rho_sorted_1d
    idx        = np.searchsorted(neg_sorted, -rho_flat, side="left")
    idx        = np.clip(idx, 0, len(rho_sorted_1d) - 1)
    idx_left   = np.maximum(idx - 1, 0)
    dist_left  = np.abs(rho_sorted_1d[idx_left] - rho_flat)
    dist_right = np.abs(rho_sorted_1d[idx]       - rho_flat)
    best_idx   = np.where(dist_left < dist_right, idx_left, idx)
    z0_3d      = z_sorted_1d[best_idx].reshape(shape)

    # --- APE ---
    ape_3d = _local_APE_precomputed_integral_numpy(
        rho_np, z_np, rho_sorted_1d, z_sorted_1d,
        cumulative_rho_dz, cumulative_dz,
        z_0_3d=z0_3d,
    )

    # --- upsilon: Υ = g (z - z_0) / ρ0 ---
    upsilon_3d = g * (z_np - z0_3d) / ρ0

    return ape_3d, z0_3d, upsilon_3d
#---

#+++ Local APE and TPE time series calculations
def local_potential_energies_timeseries(ds, rho_sorted, dz_sorted, verbose_level=1,
                                        density_name="rho", z_name="z_aac", n_workers=None):
    """
    Calculate local APE and TPE fields for all time steps.

    Requires a pre-sorted reference state (from sorted_timeseries()).

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing simulation data with time dimension.
    rho_sorted : xr.DataArray
        Pre-sorted reference density profile (time, z_1d_sorted).
    dz_sorted : xr.DataArray
        Pre-sorted cell heights (time, z_1d_sorted).
    verbose_level : int
        Verbosity level (0=quiet, 1=progress).
    density_name : str
        Name of density field in dataset.
    z_name : str
        Name of vertical coordinate.
    n_workers : int or None
        Thread-pool workers; None uses os.cpu_count(); 1 disables parallelism.

    Returns
    -------
    xr.Dataset with:
        - ape     : (time, x, y, z) — local APE density [m² s⁻²]
        - z0      : (time, x, y, z) — reference height z_0
        - upsilon : (time, x, y, z) — buoyancy displacement potential Υ
        - tpe     : (time, x, y, z) — local TPE density g·ρ·z / ρ0
        - rpe     : (time, z_1d_sorted) — local RPE density in sorted state
        - rho_sorted : (time, z_1d_sorted) — sorted reference density (passed through)
        - dz_sorted  : (time, z_1d_sorted) — sorted cell heights (passed through)
    """
    if verbose_level > 0: print("Calculating local APE and TPE time series...")

    n_times = len(ds.time)

    # Pre-extract numpy arrays
    rho_all_np        = ds[density_name].values                        # (time, …)
    z_np              = (xr.zeros_like(ds[density_name].isel(time=0))
                         + ds[density_name].isel(time=0)[z_name]).values  # (…)
    rho_sorted_all_np = rho_sorted.values                              # (time, N)
    dz_sorted_all_np  = dz_sorted.values                               # (time, N)
    z_sorted_1d_np    = rho_sorted.coords["z_1d_sorted"].values        # (N,)

    task_args = [(rho_all_np[i], z_np,
                  rho_sorted_all_np[i], dz_sorted_all_np[i], z_sorted_1d_np)
                 for i in range(n_times)]

    # --- parallel or serial time loop ---
    if n_workers == 1 or n_times == 1:
        results = []
        for i, a in enumerate(task_args):
            if verbose_level > 0: print(f"  Processing time step {i+1}/{n_times}", end="\r")
            results.append(_process_single_timestep(*a))
    else:
        max_w = n_workers
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
    rho_t0  = ds[density_name].isel(time=0)
    coords0 = rho_t0.coords
    dims0   = rho_t0.dims

    local_ape_list     = []
    local_z0_list      = []
    local_upsilon_list = []

    for ape_3d, z0_3d, upsilon_3d in results:
        local_ape_list.append(    xr.DataArray(ape_3d,     dims=dims0, coords=coords0))
        local_z0_list.append(     xr.DataArray(z0_3d,      dims=dims0, coords=coords0))
        local_upsilon_list.append(xr.DataArray(upsilon_3d, dims=dims0, coords=coords0))

    if verbose_level > 0: print("\nDone!")

    local_ape_4d     = xr.concat(local_ape_list,     dim="time").assign_coords(time=ds.time)
    local_z0_4d      = xr.concat(local_z0_list,      dim="time").assign_coords(time=ds.time)
    local_upsilon_4d = xr.concat(local_upsilon_list, dim="time").assign_coords(time=ds.time)

    tpe = local_TPE(ds[density_name], z_name=z_name)
    rpe = local_TPE(rho_sorted, z_name="z_1d_sorted")

    # Combine into a Dataset
    local_potential_energies_ds = xr.Dataset(dict(
        ape = local_ape_4d,
        z0 = local_z0_4d,
        upsilon = local_upsilon_4d,
        tpe = tpe,
        rpe = rpe,
        rho_sorted = rho_sorted,
        dz_sorted = dz_sorted,
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
def _compute_R_single_timestep(z0_t, F_t, z_sorted_vals, z_name):
    """Compute R for a single time step (called in parallel by calculate_R_reference_tendency)."""
    z_3d    = (xr.zeros_like(z0_t) + z0_t[z_name]).values.ravel()
    z0_flat = z0_t.values.ravel()
    iz  = np.clip(np.searchsorted(z_sorted_vals, z_3d),   0, len(F_t) - 1)
    iz0 = np.clip(np.searchsorted(z_sorted_vals, z0_flat), 0, len(F_t) - 1)
    R_flat = -(g / ρ0) * (F_t[iz] - F_t[iz0])
    return xr.DataArray(R_flat.reshape(z0_t.shape), dims=z0_t.dims, coords=z0_t.coords)


def calculate_R_reference_tendency(z0, drho_star_dt, dz_sorted, z_name="z_aac", n_workers=None):
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
    n_workers : int or None
        Thread-pool workers; None uses os.cpu_count(); 1 disables parallelism.

    Returns
    -------
    xr.DataArray
        4D field of R values (time, x, y, z), same shape and coordinates as z0.
    """
    # Cumulative integral F(z̃, t) = ∫_bottom^{z̃} ∂ρ_*/∂t dz̃
    F = (drho_star_dt * dz_sorted).cumsum("z_1d_sorted")  # (time, z_1d_sorted)

    z_sorted_vals = drho_star_dt.z_1d_sorted.values  # monotonically increasing physical z

    times = list(drho_star_dt.time)
    task_args = [(z0.sel(time=t), F.sel(time=t).values, z_sorted_vals, z_name) for t in times]

    if n_workers == 1 or len(times) == 1:
        R_list = [_compute_R_single_timestep(*a) for a in task_args]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = [pool.submit(_compute_R_single_timestep, *a) for a in task_args]
            R_list = [f.result() for f in futures]

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
                               filter, filter_dims=["x_caa", "y_aca"], z_name="z_aac",
                               n_workers=None):
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
    R_full = calculate_R_reference_tendency(full_z0, drho_star_dt, full_dz_sorted, z_name=z_name, n_workers=n_workers)
    R_l    = calculate_R_reference_tendency(filt_z0, drho_star_dt, full_dz_sorted, z_name=z_name, n_workers=n_workers)
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
