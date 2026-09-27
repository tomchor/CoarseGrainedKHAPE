import os
import re
from pathlib import Path
import numpy as np
import xarray as xr

# $KHAPE_PP_OUTPUT redirects the derived budget files, as $KHAPE_OUTPUT_DIR does the simulation output.
PP_OUTPUT = Path(os.environ.get("KHAPE_PP_OUTPUT") or Path(__file__).resolve().parent.parent / "output")

#+++ Multi-grid output files
# A NetCDFWriter holding outputs on more than one grid disambiguates by suffixing every dimension name
# (`z_aac` -> `z_aac_grid1`) and prefixing the grid metadata groups (`underlying_grid_reconstruction_kwargs`
# -> `grid_1_underlying_grid_reconstruction_kwargs`). The simulation does this when run with
# `--save_sorted`, since the Winters sorted column lives on its own 1x1xN grid. The rest of this
# pipeline is written against the plain names, so the suffix is stripped at load time and everything
# downstream is unaffected. Which number the model grid gets depends on the order the outputs were
# declared, so it is read off a variable known to live on the model grid rather than assumed to be grid1.
GRID_SUFFIX_RE = re.compile(r"_grid\d+$")


def model_grid_suffix(ds, reference_var="b"):
    """Return the `_gridN` suffix carried by the model grid's dimensions, or "" for a single-grid file."""
    if reference_var not in ds.variables:
        return ""
    for dim in ds[reference_var].dims:
        match = GRID_SUFFIX_RE.search(str(dim))
        if match:
            return match.group(0)
    return ""


def open_grid_group(filename, suffix=None):
    """Open the model grid's underlying-grid reconstruction group, whatever it is called in this file.

    `suffix` is the model grid's `_gridN` tag as returned by `model_grid_suffix`. Pass it if already
    known; otherwise it is detected from the file, so callers can just say `open_grid_group(filename)`
    and get the right group for both single- and multi-grid outputs.
    """
    if suffix is None:
        with xr.open_dataset(filename, decode_times=False) as probe:
            suffix = model_grid_suffix(probe)
    group = "underlying_grid_reconstruction_kwargs"
    if suffix:
        group = f"grid_{suffix.removeprefix('_grid')}_{group}"
    return xr.open_dataset(filename, group=group)


def strip_grid_suffix(ds, suffix):
    """Rename the model grid's dimensions and coordinates back to their plain, unsuffixed names.

    Variables on *other* grids keep their own suffixes, so they stay in the dataset, distinguishable
    and harmless to code that selects on the plain names (`"z_aac" in da.dims` does not match
    `z_aac_grid2`). That is what lets `inv06` read the sorted column out of the same file.
    """
    if not suffix:
        return ds
    renames = {n: str(n)[: -len(suffix)] for n in (*ds.dims, *ds.coords, *ds.variables)
               if str(n).endswith(suffix)}
    return ds.rename(renames)
#---

#+++ Integrations and sums
def integrate(da, dV, dims=("x_caa", "y_aca", "z_aac")):
    """Integrate a DataArray over spatial dimensions"""
    return (da * dV).sum(dims)
#---

#+++ Load data
def required_pad_margin(filter_scales):
    """Physical z margin the padding must provide for the widest filter in `filter_scales`.

    The sub-filter decomposition S̃ = Ē_A - L̃ is an identity only where filter(z) = z, which fails
    within one stencil of an array boundary: the truncated, asymmetric stencil no longer reproduces a
    linear coordinate. Padding restores it, provided the physical domain sits at least a full stencil
    half-width — 4σ, matching scipy's truncate=4 — inside the padded array.

    The default Nz//2 padding gives a margin of Lz/2, which is 12.5 for this setup: enough for ℓ = 7
    (4σ = 11.89) and not for anything larger. The sweep spans ℓ up to 20 (4σ = 33.97), so it needs
    roughly 3.7x the domain height in padding.
    """
    σ_max = max(float(ℓ) for ℓ in filter_scales) * _FWHM_TO_SIGMA
    return 4.0 * σ_max


def pad_margin_of(ds_filt):
    """The padding margin the filtering step recorded, so every later step pads identically.

    The sort in 02 and the budgets in 03-05 must see the same padded grid the fields were filtered on,
    so the margin is written once by 01/sweep1 and read back here rather than recomputed. Returns None
    for files written before this was recorded, which restores the old Nz//2 default.
    """
    m = ds_filt.attrs.get("pad_margin")
    return None if m is None else float(m)


def pad_margin_for_run(filtered_filename, required=False):
    """Read the recorded margin off a run's filtered-fields file, or None if it has none.

    Every step after 01 must pad exactly as 01 did -- the sort in 02 and the budgets in 03-05 all have
    to see the same padded grid -- so each reads the margin from the file 01 wrote rather than deriving
    it again. A file without the attribute returns None, which restores the Nz//2 default and keeps
    output written before it existed readable. A missing file does the same unless `required`, in which
    case it raises: the steps that follow 01 would otherwise pad to Nz//2 whatever 01 went on to use.
    """
    import xarray as _xr
    try:
        with _xr.open_dataset(filtered_filename, decode_times=False) as d:
            return pad_margin_of(d)
    except (FileNotFoundError, OSError):
        if required:
            raise FileNotFoundError(f"{filtered_filename} is missing or unreadable. Run the filtering step "
                                    f"(01, or sweep1 for the sweep) first: its recorded margin is what every "
                                    f"later step pads to.") from None
        return None


def check_same_padded_grid(ds, other, other_name):
    """Raise unless `other` (e.g. 02's sorted density) was built on the padded grid `ds` was loaded on.

    The sorted column's heights are the padded grid's own, so a column sorted on a different padding shifts
    every z✶ with nothing downstream to reveal it. 02 copies the loaded dataset's attributes onto its output,
    so the padding each side recorded can be compared directly. Output written before these attributes
    existed carries none and is refused too: it cannot be shown to match.
    """
    for key in ("n_pad_z", "z_extension"):
        mine, theirs = ds.attrs.get(key), other.attrs.get(key)
        if theirs is None or theirs != mine:
            raise ValueError(f"{other_name} was built on a padded grid with {key}={theirs}, but this step loaded "
                             f"{key}={mine}. Rerun 02 (after 01, if the filter scales changed).")


# Admissible ways to extend b and b✶ past a wall (Wenegrat, Chor & Barkan §2): the extended profile must
# stay monotonic, and extending a fluid at rest must add no APE. Both of these qualify, and the paper
# requires b and b✶ be treated the same way -- which padding b and *then* sorting does automatically,
# since for a saturated profile the wall values are the global density extremes and the padded fluid
# sorts to the ends of the column. Anything that changes between the two is a choice artifact, not
# physics, which is what makes running both a test of §4's "does not affect results".
_EXTENSIONS = {
    "edge": dict(mode="edge"),                            # repeat the wall value (§4, the default)
    "odd":  dict(mode="reflect", reflect_type="odd"),     # odd reflection about the wall value
}

# The fields the extension rule applies to. The rule is a choice about the buoyancy (§2), so every other
# field keeps the wall-value extension and an edge-vs-odd comparison changes b alone. Reflecting u too puts
# ±3U in the padding at ℓ=20 and moved ∫Π_K, which never involves b, by ~30% at Nz=192. Recorded on the
# padded dataset as `z_extension_vars`, so output padded before this restriction can be told apart.
EXTENSION_VARS = ("b",)


def _pad_along_z(da, n, kw, z_name="z_aac"):
    """np.pad along z alone, for pad widths that may exceed the axis length (ℓ=20 needs 2784 of 2048)."""
    def _p(a):                       # apply_ufunc puts the core dim last
        return np.pad(a, [(0, 0)] * (a.ndim - 1) + [(n, n)], **kw)
    # Padding needs the whole column: an edge value comes from one end of z and a reflection reaches an
    # arbitrary depth into it, so z cannot be split across chunks. The production files are chunked in z
    # (a 128-cell test file is not, which is why this only shows up at Nz=2048), so rechunk rather than
    # pass allow_rechunk -- this way the single chunk is along z alone, and time and x stay as they were.
    if da.chunks is not None:
        da = da.chunk({z_name: -1})
    return xr.apply_ufunc(
        _p, da,
        input_core_dims=[[z_name]], output_core_dims=[[z_name]],
        exclude_dims={z_name},        # the core dim changes length; apply_ufunc requires this to be declared
        dask="parallelized", output_dtypes=[da.dtype],
        dask_gufunc_kwargs={"output_sizes": {z_name: da.sizes[z_name] + 2 * n}},
    )


def _pad_domain_in_z(ds, min_margin=None, extension="edge"):
    """Extend the z domain past both walls, by `extension` (see `_EXTENSIONS`).

    Adds cells at the bottom and the top. The fields in `EXTENSION_VARS` (the buoyancy) are extended by
    `extension`; every other field repeats its wall value, whatever `extension` is. By default it adds
    Nz//2 each side, doubling the domain height; `min_margin` (a physical z distance, e.g. from
    `required_pad_margin`) widens that when a filter needs more room, and never narrows it. Assumes a
    uniform z grid. Δz_aac is extended with the same constant dz; dV and z-extent attributes are recomputed.
    """
    if extension not in _EXTENSIONS:
        raise ValueError(f"unknown extension {extension!r}; expected one of {sorted(_EXTENSIONS)}")
    pad_kw  = _EXTENSIONS[extension]
    edge_kw = _EXTENSIONS["edge"]

    Nz     = ds.sizes["z_aac"]
    dz     = float(ds.Δz_aac.isel(z_aac=0))
    Nz_pad = Nz // 2
    if min_margin is not None:
        Nz_pad = max(Nz_pad, int(np.ceil(float(min_margin) / dz)))

    z_orig = ds.z_aac.values
    z_bot  = z_orig[0]  - np.arange(Nz_pad, 0, -1) * dz
    z_top  = z_orig[-1] + np.arange(1, Nz_pad + 1) * dz
    z_new  = np.concatenate([z_bot, z_orig, z_top])

    new_vars = {}
    for name, da in ds.data_vars.items():
        if name in {"Δz_aac", "dV"} or "z_aac" not in da.dims:
            new_vars[name] = da
            continue
        # np.pad rather than building slabs by hand: it is the one formulation that covers every mode
        # and, importantly, pad widths larger than the axis itself (ℓ=20 needs 2784 cells of a 2048 grid),
        # which a single mirrored slab cannot express.
        kw = pad_kw if name in EXTENSION_VARS else edge_kw
        new_vars[name] = (_pad_along_z(da, Nz_pad, kw)
                          .assign_coords(z_aac=z_new).transpose(*da.dims))

    new_vars["Δz_aac"] = xr.DataArray(
        np.full(len(z_new), dz), dims=["z_aac"],
        coords={"z_aac": z_new}, attrs=ds["Δz_aac"].attrs,
    )

    other_coords = {k: v for k, v in ds.coords.items() if k != "z_aac"}
    ds_new = xr.Dataset(new_vars, coords={**other_coords, "z_aac": z_new}, attrs=ds.attrs)
    ds_new["dV"] = ds_new.Δx_caa * ds_new.Δy_aca * ds_new.Δz_aac

    # The padding carries no volume, so every `integrate(·, dV)` covers the physical domain alone and
    # no call site has to know the padding exists. It is not merely that padded cells are unphysical:
    # they are edge-valued, so ⟨ρ_*⟩ is exactly constant there and inverting it for z̃_* is degenerate.
    # Π_A = -τ(uᵢ,b) ∂ᵢΥ̃ is then pure noise rather than ~0, and re-randomises under perturbations as
    # small as round-off -- measured at Nz=256, two computations of ⟨ρ_*⟩ agreeing to 2.4e-15 gave Π_A
    # fields that were bit-identical across the interface and fully decorrelated in the padding
    # (rms(diff) ≈ rms(Π_A)), moving ∫Π_A dV by 76% at ℓ=1. Fields keep the padding, since the budget
    # needs filter(z) = z a stencil deep and `drop_padding` in the tests cuts it there.
    # dV stays the true cell volume: `sorted_timeseries` builds the sorted column's slot heights from
    # dV/(Lx·Ly), so zeroing it there collapses those slots to zero height and the z_1d_sorted coordinate
    # repeats. `dV_physical` is the integration weight instead — the same volumes with the padding set to
    # zero — and every `integrate(·, ·)` in the budget uses it.
    physical = (ds_new.z_aac >= z_orig[0] - dz/2) & (ds_new.z_aac <= z_orig[-1] + dz/2)
    ds_new["dV_physical"] = ds_new["dV"].where(physical, 0.0)

    ds_new.attrs["n_pad_z"]        = int(Nz_pad)
    ds_new.attrs["z_extension"]      = extension
    ds_new.attrs["z_extension_vars"] = ",".join(EXTENSION_VARS)
    ds_new.attrs["z_min_physical"] = float(z_orig[0])  - dz / 2
    ds_new.attrs["z_max_physical"] = float(z_orig[-1]) + dz / 2

    ds_new.attrs["z_min"] = float(z_new[0])  - dz / 2
    ds_new.attrs["z_max"] = float(z_new[-1]) + dz / 2
    ds_new.attrs["Lz"]    = ds_new.attrs["z_max"] - ds_new.attrs["z_min"]

    return ds_new


def extension_suffix(extension):
    """Filename tag for a non-default extension, so the two runs can sit side by side."""
    return "" if extension == "edge" else f"_{extension}"


def scale_subset_tag(filter_scales):
    """Filename tag for a sweep run over a chosen subset of scales, so it never replaces the full sweep.

    The full sweep (sweep1's default 30 scales) carries no tag; a run given --filter-scales carries `_l` and
    its scales, e.g. `_l20` or `_l1-7`, ahead of the extension tag. sweep2 and compare_extension rebuild it
    from the same scales to find that run's files.
    """
    return "" if filter_scales is None else "_l" + "-".join(f"{float(s):g}" for s in filter_scales)


def reference_suffix(reference):
    """Filename tag for `--reference true`, so its output never overwrites the default (filtered) run's.

    02's sorted density does not depend on the reference and carries no tag; the outputs of 03-06 and
    sweep2 do, and 03/04 also record the reference as the `ape_reference` attribute that 05 checks.
    """
    return "" if reference == "filtered" else f"_{reference}ref"


def extension_of(ds_filt):
    """The wall extension the filtering step used, so every later step extends identically."""
    return ds_filt.attrs.get("z_extension", "edge")


def extension_for_run(filtered_filename):
    """Read the extension off a run's filtered-fields file; 'edge' for files written before it existed."""
    try:
        with xr.open_dataset(filtered_filename, decode_times=False) as d:
            return extension_of(d)
    except (FileNotFoundError, OSError):
        return "edge"


def load_dataset_and_grid(filename, min_margin=None, extension="edge"):
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

    # A `--save_sorted` run writes the sorted column on its own grid, which makes the writer suffix
    # every dimension name; strip the model grid's suffix so the rest of the pipeline sees the plain
    # names it expects. A no-op on single-grid files.
    suffix = model_grid_suffix(ds)
    grid = open_grid_group(filename, suffix)
    ds = strip_grid_suffix(ds, suffix)

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

    # Pad domain in z: at least Nz//2 cells each side, more when a filter needs it (see _pad_domain_in_z)
    ds = _pad_domain_in_z(ds, min_margin=min_margin, extension=extension)

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


def filter_fields(ds, filter_scales):
    """Filter velocity and buoyancy fields at each length scale in x and z.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with velocity components (u, w) and buoyancy b.
    filter_scales : array-like
        Filter length scales (FWHM) in physical units.

    Returns
    -------
    ds_filt : xr.Dataset
        Dataset with filtered fields ūᵢ and b̄ at each filter_scale,
        plus dV (scale-independent).
    """
    ds = condense_uw_velocities(ds, indices=(1, 3))

    ds_filt_list = []
    for ℓ in filter_scales:
        print(f"  filter_scale = {ℓ:.4f}...")
        gf = make_gaussian_filter(ℓ, ds)
        ds_filt_list.append(xr.Dataset({
            "ūᵢ": gf.apply(ds["uᵢ"], dims=["x_caa", "z_aac"]),
            "b̄":  gf.apply(ds["b"],  dims=["x_caa", "z_aac"]),
        }))

    scale_coord = xr.DataArray(filter_scales, dims="filter_scale",
                               name="filter_scale")
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