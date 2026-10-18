"""
Budget closure tests for SFS KE and APE budgets.

For each filter scale, checks that the residual is small relative to the
budget's terms: rms(residual) / mean_v(rms(term_v)) < THRESHOLD.
"""

import pytest
import numpy as np
import xarray as xr
from conftest import PP_OUTPUT, STEM

# Residual must be < THRESHOLD x 100% of the mean budget term. Measured on the assembled online budgets over all
# records of CI's 3D run (Nz=128, Δ = 0.195h, Re = 258): KE 3.4% and 1.6%, APE 4.3% and 4.0% at ℓ=1 and 7. That is
# the resolution, not the third dimension: the 2D code at the same Δx and Re (Nz=256) gives KE 3.1% and 0.24%, APE
# 1.3% and 0.66%, and the old 2D CI at Nz=512 closed to 0.85%/0.19% (KE) and 0.46%/0.40% (APE), 0.48%/0.29% and
# 0.14%/0.18% at Nz=1024. A missing or mis-signed term is O(30-100%), so 6% still catches what the test is for.
THRESHOLD = 0.06


def rms(arr):
    """Root mean square of an array, ignoring NaNs."""
    return np.sqrt(np.nanmean(arr**2))


def relative_residual(ds, residual_var, budget_vars):
    """rms(residual) / mean_v(rms(term_v))

    The denominator is the mean of the non-zero rms over the budget terms (a zero-rms term sets no scale and is
    excluded). It used to be the smallest term, which made the metric depend on whichever term happens to be
    small at a filter scale: at ℓ=7, ∫Rˢ dV is a tenth of the mean, so a 1.6% discretisation difference in the
    offline ε_Aˢ read as 20% of the budget. Against the mean, a fractional error in one term is caught once it
    exceeds THRESHOLD x mean/rms(term), so 1% still catches a 10% error in that smallest term.
    """
    residual   = rms(ds[residual_var].values)
    term_norms = [rms(ds[v].values) for v in budget_vars]
    nonzero    = [s for s in term_norms if s > 0]
    if not nonzero:
        raise ValueError(f"All budget terms have zero RMS — cannot normalise residual.")
    scale = np.mean(nonzero)
    return residual / scale


def print_budget_summary(ds, residual_var, budget_vars, rel):
    """Print a table of rms(term) values and the relative residual."""
    print()
    print(f"  {'term':<35}  rms(term)")
    print(f"  {'-'*35}  {'-'*12}")
    for v in budget_vars:
        print(f"  {v:<35}  {rms(ds[v].values):.4e}")
    print(f"  {residual_var:<35}  {rms(ds[residual_var].values):.4e}")
    print(f"  {'residual / mean(terms)':<35}  {rel:.3%}  ({'PASS' if rel < THRESHOLD else 'FAIL'}, threshold={THRESHOLD:.0%})")


def load(suffix):
    path = PP_OUTPUT / f"{STEM}_{suffix}.nc"
    assert path.exists(), f"Output file not found: {path}"
    return xr.open_dataset(path, decode_timedelta=False)


# ---------------------------------------------------------------------------
# KE budget
# ---------------------------------------------------------------------------
KE_BUDGET_VARS = [
    "∫-∂ₜ SFS KE dV",
    "∫Π_K dV",
    "∫-ε_Kˢ dV",
    "∫(SFS APE->KE) dV",
]

@pytest.fixture(scope="module")
def ke_budget():
    return load("sfs_ke_budget_integrated")


def test_ke_budget_residual(ke_budget, l_idx):
    l = ke_budget.filter_scale.values[l_idx]
    ds_l = ke_budget.sel(filter_scale=l)
    rel = relative_residual(ds_l, "residual_K", KE_BUDGET_VARS)
    print(f"\nKE budget  (l={l:.4f})")
    print_budget_summary(ds_l, "residual_K", KE_BUDGET_VARS, rel)
    assert rel < THRESHOLD, (
        f"KE budget residual too large at l={l:.4f}: "
        f"relative residual = {rel:.3%} > {THRESHOLD:.0%}"
    )


# ---------------------------------------------------------------------------
# APE budget
# ---------------------------------------------------------------------------
APE_BUDGET_VARS = [
    "∫-∂ₜ SFS APE dV",
    "∫Π_A dV",
    "∫-ε_Aˢ dV",
    "∫(SFS KE->APE) dV",
    "∫Rˢ dV",
]

@pytest.fixture(scope="module")
def ape_budget():
    return load("sfs_ape_budget_integrated")


def test_ape_budget_residual(ape_budget, l_idx):
    l = ape_budget.filter_scale.values[l_idx]
    ds_l = ape_budget.sel(filter_scale=l)
    rel = relative_residual(ds_l, "residual_A", APE_BUDGET_VARS)
    print(f"\nAPE budget  (l={l:.4f})")
    print_budget_summary(ds_l, "residual_A", APE_BUDGET_VARS, rel)
    assert rel < THRESHOLD, (
        f"APE budget residual too large at l={l:.4f}: "
        f"relative residual = {rel:.3%} > {THRESHOLD:.0%}"
    )
