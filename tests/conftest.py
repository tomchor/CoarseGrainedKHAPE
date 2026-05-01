import xarray as xr
from pathlib import Path

PP_OUTPUT = Path(__file__).parent.parent / "postprocessing" / "output"
STEM = "khi_Nz512_Ri0.10"


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
