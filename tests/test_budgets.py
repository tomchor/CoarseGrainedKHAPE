"""
Budget closure tests for SFS KE and APE budgets.

For each filter scale, checks that the residual is small relative to
the smallest budget term: rms(residual) / min_v(rms(term_v)) < THRESHOLD.
"""

import pytest
import numpy as np
import xarray as xr
from pathlib import Path

PP_OUTPUT = Path(__file__).parent.parent / "postprocessing" / "output"
STEM      = "khi_Nz512_Ri0.10"
# Residual must be < THRESHOLD x 100% of the smallest budget term (this number is large since we test with a
# short, coarse simulation
THRESHOLD = 0.1


def rms(arr):
    """Root mean square of an array, ignoring NaNs."""
    return np.sqrt(np.nanmean(arr**2))


def relative_residual(ds, residual_var, budget_vars):
    """rms(residual) / min_v(rms(term_v))

    The denominator is the smallest RMS among all budget terms.
    This provides a stricter normalisation than dividing by the largest term.
    """
    residual = rms(ds[residual_var].values)
    scale    = min(rms(ds[v].values) for v in budget_vars)
    return residual / scale


def print_budget_summary(ds, residual_var, budget_vars, rel):
    """Print a table of rms(term) values and the relative residual."""
    print()
    print(f"  {'term':<35}  rms(term)")
    print(f"  {'-'*35}  {'-'*12}")
    for v in budget_vars:
        print(f"  {v:<35}  {rms(ds[v].values):.4e}")
    print(f"  {residual_var:<35}  {rms(ds[residual_var].values):.4e}")
    print(f"  {'residual / min(terms)':<35}  {rel:.3%}  ({'PASS' if rel < THRESHOLD else 'FAIL'}, threshold={THRESHOLD:.0%})")


def load(suffix):
    path = PP_OUTPUT / f"{STEM}_{suffix}.nc"
    assert path.exists(), f"Output file not found: {path}"
    return xr.open_dataset(path, decode_timedelta=False)


# ---------------------------------------------------------------------------
# KE budget
# ---------------------------------------------------------------------------
KE_BUDGET_VARS = [
    "∫-∂ₜ SFS KE dV",
    "∫Π_KE dV",
    "∫-εₛ dV",
    "∫(SFS APE->KE) dV",
]

@pytest.fixture(scope="module")
def ke_budget():
    return load("sfs_ke_budget_integrated")


@pytest.mark.parametrize("l_idx", range(len(load("sfs_ke_budget_integrated").filter_length_scale)))
def test_ke_budget_residual(ke_budget, l_idx):
    l = ke_budget.filter_length_scale.values[l_idx]
    ds_l = ke_budget.sel(filter_length_scale=l)
    rel = relative_residual(ds_l, "residual_KE", KE_BUDGET_VARS)
    print(f"\nKE budget  (l={l:.4f})")
    print_budget_summary(ds_l, "residual_KE", KE_BUDGET_VARS, rel)
    assert rel < THRESHOLD, (
        f"KE budget residual too large at l={l:.4f}: "
        f"relative residual = {rel:.3%} > {THRESHOLD:.0%}"
    )


# ---------------------------------------------------------------------------
# APE budget
# ---------------------------------------------------------------------------
APE_BUDGET_VARS = [
    "∫-∂ₜ SFS APE dV",
    "∫Π_APE dV",
    "∫-χₛ dV",
    "∫(SFS KE->APE) dV",
    "∫Rˢ dV",
]

@pytest.fixture(scope="module")
def ape_budget():
    return load("sfs_ape_budget_integrated")


@pytest.mark.parametrize("l_idx", range(len(load("sfs_ape_budget_integrated").filter_length_scale)))
def test_ape_budget_residual(ape_budget, l_idx):
    l = ape_budget.filter_length_scale.values[l_idx]
    ds_l = ape_budget.sel(filter_length_scale=l)
    rel = relative_residual(ds_l, "residual_APE", APE_BUDGET_VARS)
    print(f"\nAPE budget  (l={l:.4f})")
    print_budget_summary(ds_l, "residual_APE", APE_BUDGET_VARS, rel)
    assert rel < THRESHOLD, (
        f"APE budget residual too large at l={l:.4f}: "
        f"relative residual = {rel:.3%} > {THRESHOLD:.0%}"
    )
