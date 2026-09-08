"""
Positivity tests for the energy densities whose sign is fixed by construction.

Two of the pipeline's energies cannot be negative, so a negative value means a broken reference state
or a broken filter rather than an unusual flow:

  * the local APE Eₐ = ∫_{z✶}^{z}[b✶(z̃) - b] dz̃, the work needed to move a parcel from its reference
    height z✶ to where it sits. b✶ increases with height and b✶(z✶) = b, so the integrand keeps one
    sign over the whole path and the integral comes out positive whichever side of z✶ the parcel is
    on. Checked for the full field, the filtered field, and the filtered full field.
  * the SFS KE ½τᵢᵢ = ½(filter(uᵢuᵢ) - ūᵢūᵢ), non-negative by Jensen's inequality: the Gaussian
    filter's weights are positive and sum to one, so filtering a square is at least the square of the
    filtered field.

The SFS APE Eₐˢ = filter(eₐ) - eₐˡ is deliberately absent. The same Jensen argument would need the
filter to act on b alone; this one also acts in z, where eₐ(·, z) varies with height, so the bound
does not carry. Negative Eₐˢ is physical — most of the domain is negative at the wider filter scales.

The total KE ½uᵢuᵢ is absent for the opposite reason: it is a sum of squares, so a test of its sign
would only be testing numpy.

Both reference-profile variants are covered, since CI runs the suite twice (`--ref-suffix`).
Positivity needs the reference profile to be monotone in z, not to be the sort of the current field,
so it holds for either.
"""

import pytest
import numpy as np
import xarray as xr
from pathlib import Path

REPO_ROOT  = Path(__file__).resolve().parent.parent
PP_OUTPUT  = REPO_ROOT / "postprocessing" / "output"
SIM_OUTPUT = REPO_ROOT / "output" / "khi_Nz512_Ri0.10.nc"
STEM       = "khi_Nz512_Ri0.10"

#+++ Tolerances
# Worst allowed excursion below zero, as a fraction of the field's own rms: min(field)/rms(field) > -TOL.
#
# APE: z✶ comes from a nearest-density lookup into the discrete sorted column, so a parcel whose
# reference height falls inside its own cell integrates a sliver of either sign. That puts a third of
# the domain a hair below zero without meaning anything — which is why the test is on magnitude and
# not on how many points are negative. Measured on an Nz=512 run the worst point sits at -8e-06 of
# rms(Eₐ) across every filter scale, so this leaves two decades of headroom and still catches a sign
# error, which would be O(1).
APE_TOL = 1e-3

# SFS KE: ½τᵢᵢ ≥ 0 is exact in exact arithmetic, so only floating-point roundoff is allowed. The
# measured minima on the same run are positive at every filter scale, the smallest at +1e-12 of rms.
KE_TOL = 1e-8
#---

#+++ Helpers
@pytest.fixture(scope="session")
def ref_suffix(request):
    return request.config.getoption("--ref-suffix")


def load(suffix, ref_suffix=""):
    """Open one of the 4D field files. Chunked: these hold every local budget field, so they are far
    too big to pull into memory whole."""
    path = PP_OUTPUT / f"{STEM}_{suffix}{ref_suffix}.nc"
    assert path.exists(), f"Output file not found: {path}"
    return xr.open_dataset(path, decode_timedelta=False, chunks={"time": 1})


def positivity_stats(da):
    """(min, rms, min/rms, fraction of points below zero), computed in a single pass over the data."""
    stats = xr.Dataset(dict(minimum=da.min(), mean_square=(da**2).mean(), frac_neg=(da < 0).mean())).compute()
    minimum, rms, frac_neg = float(stats.minimum), float(np.sqrt(stats.mean_square)), float(stats.frac_neg)
    return minimum, rms, (minimum / rms if rms > 0 else 0.0), frac_neg


def check_positive(da, name, tol):
    """Assert that `da` dips no further below zero than `tol` x its own rms."""
    minimum, rms, rel, frac_neg = positivity_stats(da)
    print(f"  {name:<16}  min={minimum:+.4e}  rms={rms:.4e}  min/rms={rel:+.3e}  frac(<0)={frac_neg:.3e}"
          f"  ({'PASS' if rel > -tol else 'FAIL'}, tolerance={-tol:.0e})")
    assert rel > -tol, (f"{name} goes negative beyond roundoff: min = {minimum:.4e} = {rel:.3e} x rms, "
                        f"tolerance is {-tol:.0e} x rms")
#---

#+++ Local APE (offline pipeline)
APE_FIELDS = [
    "Ea(ρ, z)",   # full field
    "Ea(ρ̄, z)",   # filtered field
    "Ēa(ρ, z)",   # filtered full field
]

@pytest.fixture(scope="module")
def ape_fields(ref_suffix):
    return load("sfs_ape_budget_fields", ref_suffix)


@pytest.mark.parametrize("var", APE_FIELDS)
def test_local_ape_is_positive(ape_fields, l_idx, var):
    l = ape_fields.filter_scale.values[l_idx]
    print(f"\nLocal APE  (l={l:.4f})")
    check_positive(ape_fields[var].sel(filter_scale=l), var, APE_TOL)
#---

#+++ SFS KE (offline pipeline)
@pytest.fixture(scope="module")
def ke_fields(ref_suffix):
    return load("sfs_ke_budget_fields", ref_suffix)


def test_sfs_ke_is_positive(ke_fields, l_idx):
    l = ke_fields.filter_scale.values[l_idx]
    print(f"\nSFS KE  (l={l:.4f})")
    check_positive(ke_fields["KE_of_sfs_flow"].sel(filter_scale=l), "KE_of_sfs_flow", KE_TOL)
#---

#+++ Local APE (online, --save_sorted)
# The simulation's own Eₐ, built from Oceanostics' ThreeDimensionalSort z✶ instead of the offline
# nearest-density lookup: a second implementation of the same integral, and the only check here that
# sees the Julia side. Written only under --save_sorted (which CI uses), so it skips otherwise.
@pytest.fixture(scope="module")
def sim_output():
    if not SIM_OUTPUT.exists():
        pytest.skip(f"Simulation output not found: {SIM_OUTPUT}")
    return xr.open_dataset(SIM_OUTPUT, decode_times=False, chunks={"time": 1})


@pytest.mark.parametrize("var", ["E_a", "∫E_a"])
def test_online_local_ape_is_positive(sim_output, var):
    if var not in sim_output:
        pytest.skip(f"'{var}' not in simulation output — the run did not use --save_sorted")
    print(f"\nOnline local APE")
    check_positive(sim_output[var], var, APE_TOL)
#---
