import os
import sys
import pytest
import xarray as xr
from pathlib import Path

#+++ The run under test
# CI's run (.github/workflows/test.yml), so STEM has to follow its --Nz. The test modules import these from here
# and name the file nowhere else: a stale copy fails quietly, because the tests that read SIM_OUTPUT skip when it
# is missing.
# $KHAPE_PP_OUTPUT and $KHAPE_OUTPUT_DIR redirect the post-processing and simulation output (aux00_utils.py and
# kelvin_helmholtz_instability.jl), so the tests look where those wrote.
REPO_ROOT  = Path(__file__).resolve().parent.parent
PP_OUTPUT  = Path(os.environ.get("KHAPE_PP_OUTPUT") or REPO_ROOT / "postprocessing" / "output")
STEM       = os.environ.get("KHAPE_TEST_STEM", "khi_Nz128_Ri0.10")   # $KHAPE_TEST_STEM points the suite at another run
SIM_OUTPUT = Path(os.environ.get("KHAPE_OUTPUT_DIR") or REPO_ROOT / "output") / f"{STEM}.nc"
#---


# The budgets the tests read are assembled from the simulation's online terms (postprocessing/01_online_budgets.py).
# `--offline-check` additionally runs the offline pipeline (postprocessing/offline/) and compares every term
# against them (tests/test_offline_check.py), and runs the online-vs-offline validation scripts
# (tests/test_online_vs_offline.py). Off by default: it costs about an hour at the CI resolution, and CI
# runs it in its own job.
OFFLINE_PP_OUTPUT = PP_OUTPUT / "offline"   # where offline/run_offline_budgets.sh writes


#+++ The online budgets, 3D terms included
@pytest.fixture(scope="session")
def online_budget():
    """{"ke": ..., "ape": ...}: both SFS budgets as `online_budgets` assembles them from the simulation output, lazily.

    The 3D terms are read from the simulation file itself (01_online_budgets.py writes only the integrals), at the
    filter scales and records of the integrated files, which is what `l_idx` indexes.
    """
    assert SIM_OUTPUT.exists(), f"Simulation output not found: {SIM_OUTPUT}"
    sys.path.insert(0, str(REPO_ROOT / "postprocessing"))
    from src.aux00_utils import load_dataset_and_grid
    from src.aux04_online_budgets import online_budgets
    integrated = xr.open_dataset(PP_OUTPUT / f"{STEM}_sfs_ke_budget_integrated.nc", decode_timedelta=False)
    scales, records = integrated.filter_scale.values, integrated.attrs.get("online_records", "differenced")
    ds = load_dataset_and_grid(str(SIM_OUTPUT), pad=False).chunk({"time": 1})
    ke, ape = online_budgets(ds, filter_scales=scales, records=records)
    return {"ke": ke, "ape": ape}
#---


def pytest_addoption(parser):
    parser.addoption(
        "--offline-check",
        action="store_true",
        default=False,
        help="Run the offline budget pipeline and check that it reproduces the online budgets (slow; CI's offline-check job; "
             "needs a simulation run with --offline_check)",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "offline_check: needs --offline-check (runs the offline pipeline, about an hour at the CI resolution)",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--offline-check"):
        return
    skip = pytest.mark.skip(reason="needs --offline-check")
    for item in items:
        if "offline_check" in item.keywords:
            item.add_marker(skip)


def pytest_generate_tests(metafunc):
    if "l_idx" in metafunc.fixturenames:
        path = PP_OUTPUT / f"{STEM}_sfs_ke_budget_integrated.nc"
        ds = xr.open_dataset(path, decode_timedelta=False)
        metafunc.parametrize("l_idx", range(len(ds.filter_scale)))
