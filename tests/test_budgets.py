"""
Budget closure tests for SFS KE and APE budgets.

For each filter scale, checks that the residual is small relative to
the largest term in the budget: max(|residual|) / max(|terms|) < THRESHOLD.
"""

import pytest
import numpy as np
import xarray as xr
from pathlib import Path

PP_OUTPUT = Path(__file__).parent.parent / "postprocessing" / "output"
STEM      = "khi_180x1x512"
THRESHOLD = 0.05  # residual must be < 5% of the largest budget term


def relative_residual(ds, residual_var, budget_vars):
    """max(|residual|) / max over budget terms of max(|term|)"""
    residual = np.nanmax(np.abs(ds[residual_var].values))
    scale    = max(np.nanmax(np.abs(ds[v].values)) for v in budget_vars)
    return residual / scale


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
    return load("sfs_ke_budget")


@pytest.mark.parametrize("l_idx", range(4))   # 4 filter scales set in 01_filter_and_prepare_fields.py
def test_ke_budget_residual(ke_budget, l_idx):
    l = ke_budget.filter_length_scale.values[l_idx]
    ds_l = ke_budget.sel(filter_length_scale=l)
    rel = relative_residual(ds_l, "residual_KE", KE_BUDGET_VARS)
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
    return load("sfs_ape_budget")


@pytest.mark.parametrize("l_idx", range(4))
def test_ape_budget_residual(ape_budget, l_idx):
    l = ape_budget.filter_length_scale.values[l_idx]
    ds_l = ape_budget.sel(filter_length_scale=l)
    rel = relative_residual(ds_l, "residual_APE", APE_BUDGET_VARS)
    assert rel < THRESHOLD, (
        f"APE budget residual too large at l={l:.4f}: "
        f"relative residual = {rel:.3%} > {THRESHOLD:.0%}"
    )
