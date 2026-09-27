import os
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
STEM       = "khi_Nz1024_Ri0.10"
SIM_OUTPUT = Path(os.environ.get("KHAPE_OUTPUT_DIR") or REPO_ROOT / "output") / f"{STEM}.nc"
#---


def pytest_addoption(parser):
    parser.addoption(
        "--ref-suffix",
        default="",
        help="Suffix appended to postprocessing output filenames (e.g. '_fixed_ref')",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "ref_suffix: parameterise tests by reference-profile suffix",
    )


def pytest_generate_tests(metafunc):
    if "l_idx" in metafunc.fixturenames:
        ref_suffix = metafunc.config.getoption("--ref-suffix")
        path = PP_OUTPUT / f"{STEM}_sfs_ke_budget_integrated{ref_suffix}.nc"
        ds = xr.open_dataset(path, decode_timedelta=False)
        metafunc.parametrize("l_idx", range(len(ds.filter_scale)))
