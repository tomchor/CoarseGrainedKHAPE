"""
Kinetic energy calculation functions

This module contains functions for calculating kinetic energy (KE).
"""

import xarray as xr
from src.aux00_utils import (integrate, calculate_gradient,
                         condense_uw_velocities,
                         make_gaussian_filter, filter_fields)
from src.aux01_pe_functions import (calculate_density_fields_from_buoyancy,
                                sorted_timeseries,
                                local_potential_energies_timeseries,
                                calculate_cross_scale_ape_flux,
                                calculate_b_r,
                                calculate_ape_to_ke_exchange_term)

# Physical constants
ρ0 = 1025  # reference density [kg/m^3]

#+++ SFS stress tensor
def calculate_sfs_stress_tensor(u_i, filter, filter_dims=["x_caa", "y_aca"],
                                filtered_u_i=None, index_dim="i"):
    """
    Calculate the full 3×3 SFS stress tensor τ̄ℓⁱʲ(u, u)

    Following Aluie et al. (2018, JPO) Eq. (5), each component is:

        τ̄ℓⁱʲ = filter(uⁱ uʲ) - ūⁱ ūʲ

    The result is a symmetric tensor with a new dimension "j" (same index values
    as the input "i" dimension). The trace gives twice the SFS KE:

        (1/2) tr(τ̄ℓ) = (1/2)[ filter(|u|²) - |ū|² ] = KE_s  ≥ 0

    The filter is applied to the outer product uⁱ uʲ using xarray broadcasting,
    which evaluates all 9 components in a single pass — the same approach as
    calculate_sfs_flux_tensor() in aux01_pe_functions.py.

    Parameters
    ----------
    u_i : xr.DataArray
        Full (unfiltered) velocity tensor with index dimension (e.g. i=1,2,3).
    filter : gcm_filters.Filter
        Filter object used for the spatial filtering operation.
    filter_dims : list of str
        Spatial dimensions along which to apply the filter.
    filtered_u_i : xr.DataArray, optional
        Pre-computed filtered velocity ūⁱ. If None, it is computed from u_i.
    index_dim : str
        Name of the vector index dimension (default "i").

    Returns
    -------
    xr.DataArray
        Tensor τ̄ℓⁱʲ with dimensions (i, j, ...) where j carries the same
        coordinate values as i. Select a component via e.g.
        tau.sel(i=1, j=2) for τxy.
    """
    if filtered_u_i is None:
        filtered_u_i = filter.apply(u_i, dims=filter_dims)

    # Rename index dimension to "j" on the second operand so xarray broadcasts
    # u_i (i, ...) × u_j (j, ...) → outer product with shape (i, j, ...)
    u_j     = u_i.rename({index_dim: "j"})
    u_j_bar = filtered_u_i.rename({index_dim: "j"})

    # τ̄ℓⁱʲ = filter(uⁱ uʲ) - ūⁱ ūʲ  (all components in one vectorised call)
    tau = filter.apply(u_i * u_j, dims=filter_dims) - filtered_u_i * u_j_bar

    return tau
#---

#+++ Velocity gradient tensor
def calculate_velocity_gradient_tensor(u_i_bar, dimensions=("x_caa", "y_aca", "z_aac"), index_dim="i"):
    """
    Compute the large-scale velocity gradient tensor ∂ūⁱ/∂xʲ

    Each row i is the gradient of the i-th filtered velocity component, giving
    a tensor of shape (i, j, ...) where index j runs over spatial directions.

    Parameters
    ----------
    u_i_bar : xr.DataArray
        Filtered velocity vector with index dimension (e.g. i=1,2,3 for ū, v̄, w̄).
    dimensions : tuple of str
        Ordered spatial coordinate names matching index values 1, 2, 3.
    index_dim : str
        Name of the velocity index dimension (default "i").

    Returns
    -------
    xr.DataArray
        Tensor ∂ūⁱ/∂xʲ with dimensions (i, j, ...).
        Select a component via e.g. grad_u.sel(i=1, j=2) for ∂ū/∂y.
    """
    j_indices = list(u_i_bar[index_dim].values)
    grad_components = []
    for k in j_indices:
        u_k = u_i_bar.sel({index_dim: k}, drop=True)
        # calculate_gradient returns a vector along dimname="j"
        grad_k = calculate_gradient(u_k, dimensions=dimensions, dimname="j", indices=j_indices)
        grad_components.append(grad_k)

    return xr.concat(
        grad_components,
        dim=xr.DataArray(j_indices, dims=index_dim, name=index_dim),
    )
#---

#+++ Large-scale strain rate tensor
def calculate_strain_tensor(u_i_bar, dimensions=("x_caa", "y_aca", "z_aac"), index_dim="i"):
    """
    Compute the large-scale strain rate tensor S̄ℓⁱʲ

    S̄ℓ is the symmetric part of the velocity gradient tensor:

        S̄ℓⁱʲ = (1/2)(∂ūⁱ/∂xʲ + ∂ūʲ/∂xⁱ)

    For incompressible flow tr(S̄ℓ) = ∇·ū = 0.

    Parameters
    ----------
    u_i_bar : xr.DataArray
        Filtered velocity vector with index dimension (e.g. i=1,2,3 for ū, v̄, w̄).
    dimensions : tuple of str
        Ordered spatial coordinate names matching index values 1, 2, 3.
    index_dim : str
        Name of the velocity index dimension (default "i").

    Returns
    -------
    xr.DataArray
        Symmetric tensor S̄ℓⁱʲ with dimensions (i, j, ...).
        Select a component via e.g. S.sel(i=1, j=2) for S̄ˣʸ.
    """
    grad_u = calculate_velocity_gradient_tensor(u_i_bar, dimensions=dimensions,
                                                 index_dim=index_dim)

    # Transpose: swap i ↔ j so that grad_u_T[i,j] = ∂ūʲ/∂xⁱ
    grad_u_T = grad_u.rename({index_dim: "j", "j": index_dim})

    return (grad_u + grad_u_T) / 2
#---

#+++ SFS KE dissipation
def calculate_sfs_ke_dissipation(S, ν, filter, filter_dims=["x_caa", "y_aca"],
                                 index_dims=("i", "j")):
    """
    Compute the SFS KE dissipation ε<ℓ = 2ν τ(S, S)

    The τ operator applied to the strain rate tensor yields a scalar via
    double contraction:

        τ(S, S) = Σᵢⱼ [ filter(Sⁱʲ Sⁱʲ) - filter(Sⁱʲ)² ]

    which is the subfilter variance of the strain field.  Multiplied by
    2ν this gives the rate at which viscosity dissipates subfilter KE.

    The structure mirrors calculate_sfs_ape_dissipation() in
    aux01_pe_functions.py: filter(S²) and filter(S)² are each evaluated
    across all (i,j) components in one call thanks to xarray broadcasting,
    then the index dimensions are summed away.

    Parameters
    ----------
    S : xr.DataArray
        Strain rate tensor Sⁱʲ with dimensions (i, j, ...).
        Should be computed from the *full* (unfiltered) velocity to capture
        all scales; typically from calculate_strain_tensor()
        applied to the unfiltered velocity field.
    ν : xr.DataArray or float
        Kinematic viscosity ν [m² s⁻¹] (scalar or spatially varying field,
        e.g. the eddy viscosity ds.ν from SmagorinskyLilly).
    filter : gcm_filters.Filter
        Filter object used for the spatial filtering operation.
    filter_dims : list of str
        Spatial dimensions along which to apply the filter.
    index_dims : tuple of str
        Names of the two tensor index dimensions to contract over.

    Returns
    -------
    xr.DataArray
        SFS KE dissipation ε<ℓ [m² s⁻³], same spatial dimensions as S
        (the i and j index dimensions are contracted away).
    """
    S̄   = filter.apply(S, dims=filter_dims) # filter(Sⁱʲ)
    tau_S_S = filter.apply(S * S, dims=filter_dims) - S̄ * S̄ # τ(Sⁱʲ, Sⁱʲ) = filter(S²) - filter(S)²
    return 2 * ν * tau_S_S.sum(list(index_dims))
#---

#+++ SFS KE tendency
def calculate_sfs_ke_tendency(sfs_ke_density):
    """
    Compute ∂KE_s/∂t as a centred finite difference in time.

    Parameters
    ----------
    sfs_ke_density : xr.DataArray
        4D subfilter KE field (time, x, y, z),
        e.g. sfs_stress_tensor_trace / 2.

    Returns
    -------
    xr.DataArray
        4D tendency field on the staggered (mid-point) time grid.
    """
    Δt = sfs_ke_density.time.diff("time").sel(time=slice(None, None, 2))
    ΔKE = sfs_ke_density.diff("time").sel(time=slice(None, None, 2))
    return ΔKE / Δt
#---

#+++ Cross-scale KE flux
def calculate_cross_scale_ke_flux(τ, S̄, index_dims=("i", "j")):
    """
    Compute the cross-scale KE flux Πℓ = -ρ₀ S̄ℓ : τ̄ℓ

    Following Aluie et al. (2018, JPO) Eq. (7).  Πℓ > 0 means forward
    (downscale) KE transfer; Πℓ < 0 means inverse (upscale) transfer.

    The double contraction of two symmetric 3×3 tensors is

        A : B = Σᵢⱼ Aⁱʲ Bⁱʲ
              = Aˣˣ Bˣˣ + Aʸʸ Bʸʸ + Aᶻᶻ Bᶻᶻ
                + 2 Aˣʸ Bˣʸ + 2 Aˣᶻ Bˣᶻ + 2 Aʸᶻ Bʸᶻ

    Because both tensors are stored as full (i, j, ...) DataArrays with all
    nine entries, the sum over both index dimensions gives the correct result
    (the off-diagonal factor-of-2 arises automatically from the (i,j) + (j,i)
    entries):

        Πℓ = -ρ₀ * (S * τ).sum(["i", "j"])

    Parameters
    ----------
    τ : xr.DataArray
        SFS stress tensor τⁱʲ with dimensions (i, j, ...).
        Typically from calculate_sfs_stress_tensor().
    S̄ : xr.DataArray
        Large-scale strain rate tensor S̄ℓⁱʲ with dimensions (i, j, ...).
        Typically from calculate_strain_tensor().
    index_dims : tuple of str
        Names of the two tensor index dimensions to sum over.

    Returns
    -------
    xr.DataArray
        Cross-scale KE flux Πℓ [m² s⁻³], same spatial dimensions as S̄ / τ
        (the i and j dimensions are contracted away).
    """
    return -(τ * S̄).sum(index_dims)
#---

#+++ Cross-scale energy transfer pipeline
def calculate_energy_transfer(ds, filter_scales,
                              ds_filt=None, rho_sorted=None, dz_sorted=None, n_workers=18,
                              include_pi_k=True):
    """Calculate cross-scale KE and APE transfer terms at each filter scale.

    Parameters
    ----------
    ds : xr.Dataset
        Full (unfiltered) simulation dataset. Must contain velocity components
        (u, w), buoyancy b, and grid variables dV, LxLy.
    filter_scales : array-like
        Physical length scales at which to compute the transfer terms.
    ds_filt : xr.Dataset, optional
        Pre-computed filtered fields (ūᵢ, b̄) indexed by filter_scale.
        If None, filter_fields() is called internally.
    rho_sorted : xr.DataArray, optional
        Pre-sorted reference density (time, z_1d_sorted), e.g. loaded from a
        ``*_sorted_density.nc`` file.  When provided together with
        ``dz_sorted``, the density-sorting step is skipped entirely.
    dz_sorted : xr.DataArray, optional
        Pre-sorted cell heights (time, z_1d_sorted). Must be supplied together
        with ``rho_sorted``.
    n_workers : int
        Number of threads for APE sorting (ThreadPoolExecutor).
    include_pi_k : bool
        If True (default) also compute the cross-scale KE flux Π_K. The KH budget
        pipeline computes Π_K online and passes include_pi_k=False to skip the
        offline recompute (the sweep keeps the default and still gets Π_K).

    Returns
    -------
    xr.Dataset
        Dataset with Π_A, the SFS APE->KE exchange term and the coarse conversion
        w̄·b̄ᵣ (plus their volume integrals) indexed by filter_scale — and Π_K (with
        ∫Π_K dV) when include_pi_k=True.
    """
    filtered_dimensions = ["x_caa", "z_aac"]
    tensor_dimensions   = ("x_caa", "z_aac")

    if ds_filt is None:
        ds_filt = filter_fields(ds, filter_scales)

    ds = condense_uw_velocities(ds, indices=(1, 3))
    ds_full = ds[["b", "dV", "LxLy", "uᵢ"]].copy()

    ds_full = calculate_density_fields_from_buoyancy(ds_full, buoyancy_name="b", density_name="ρ")

    # Use pre-sorted reference state if provided; otherwise sort the full density field
    if rho_sorted is not None and dz_sorted is not None:
        print("Using pre-sorted reference density (skipping sort).")
    else:
        print("Computing full-field reference state (rho_sorted)...")
        _full_sorted = sorted_timeseries(ds_full, field_to_sort="ρ", n_workers=n_workers)
        rho_sorted = _full_sorted.rho_sorted
        dz_sorted  = _full_sorted.dz_sorted

    # Relative buoyancy b_r is scale-independent — compute once outside the loop
    b_r = calculate_b_r(ds_full.ρ, rho_sorted)
    w_full = ds_full["uᵢ"].sel(i=3)

    dV = ds_full.dV
    transfer_list = []

    for ℓ in filter_scales:
        print(f"\n--- filter_scale = {ℓ:.4f} ---")
        gaussian_filter = make_gaussian_filter(ℓ, ds)

        ds_filt_ℓ = ds_filt.sel(filter_scale=ℓ).drop_vars("filter_scale")
        ds_filt_ℓ["LxLy"] = ds["LxLy"]
        ds_filt_ℓ.attrs.update(ds.attrs)

        # --- KE cross-scale transfer (Π_K) ---
        # Computed online by the simulation; skipped here when include_pi_k=False so the offline
        # pipeline does not redundantly recompute it (the SFS KE budget reads the online Π_K).
        ke_vars = {}
        if include_pi_k:
            # τⁱʲ = filter(uⁱuʲ) - ūⁱūʲ ;  Π_K = -τⁱʲ : S̄ⁱʲ
            sfs_stress_tensor = calculate_sfs_stress_tensor(ds_full["uᵢ"], gaussian_filter,
                                                            filter_dims=filtered_dimensions,
                                                            filtered_u_i=ds_filt_ℓ["ūᵢ"])
            strain_rate_tensor_l = calculate_strain_tensor(ds_filt_ℓ["ūᵢ"], dimensions=tensor_dimensions)
            Π_K = calculate_cross_scale_ke_flux(sfs_stress_tensor, strain_rate_tensor_l)
            ke_vars = {"Π_K": Π_K, "∫Π_K dV": integrate(Π_K, dV)}

        # --- APE->KE conversion terms ---
        # SFS exchange:        filter(w·b_r) - filter(w)·filter(b_r)
        # Coarse conversion: filter(w) · filter(b_r)
        w_bar   = ds_filt_ℓ["ūᵢ"].sel(i=3)
        b_r_bar = gaussian_filter.apply(b_r, dims=filtered_dimensions)
        ape_to_ke_exchange = calculate_ape_to_ke_exchange_term(w_full, b_r,
                                                               gaussian_filter,
                                                               filter_dims=filtered_dimensions,
                                                               filtered_w=w_bar,
                                                               filtered_b=b_r_bar)
        wbar_b_r_bar = (w_bar * b_r_bar).rename("w̄·b̄ᵣ")

        # --- APE cross-scale transfer ---
        # Compute ρ̄ and the large-scale reference state z₀(ρ̄) → Υˡ
        # Pass pre-sorted full-field reference state to avoid re-sorting each iteration
        ds_filt_ℓ = calculate_density_fields_from_buoyancy(ds_filt_ℓ, buoyancy_name="b̄", density_name="ρ̄")
        filt_local_pes = local_potential_energies_timeseries(ds_filt_ℓ, density_name="ρ̄",
                                                             rho_sorted=rho_sorted,
                                                             dz_sorted=dz_sorted,
                                                             n_workers=n_workers)
        # Π_A = -(filter(ρuᵢ) - ρ̄ūᵢ) · ∇Υˡ
        Π_A = calculate_cross_scale_ape_flux(ds_full.ρ, ds_full["uᵢ"], filt_local_pes.upsilon,
                                              gaussian_filter, filter_dims=filtered_dimensions,
                                              filtered_density=ds_filt_ℓ.ρ̄,
                                              filtered_velocity_vector=ds_filt_ℓ["ūᵢ"])

        int_Π_A                = integrate(Π_A, dV)
        int_ape_to_ke_exchange = integrate(ape_to_ke_exchange, dV)
        int_wbar_b_r_bar       = integrate(wbar_b_r_bar, dV)

        transfer_vars = {
            "Π_A":                  Π_A,
            "SFS APE->KE exchange": ape_to_ke_exchange,
            "w̄·b̄ᵣ":                 wbar_b_r_bar,
            "∫Π_A dV":              int_Π_A,
            "∫(SFS APE->KE) dV":    int_ape_to_ke_exchange,
            "∫w̄·b̄ᵣ dV":             int_wbar_b_r_bar,
            **ke_vars,
        }
        transfer_list.append(xr.Dataset(transfer_vars))

    scale_coord = xr.DataArray(filter_scales, dims="filter_scale",
                               name="filter_scale")
    return xr.concat(transfer_list, dim=scale_coord)
#---
