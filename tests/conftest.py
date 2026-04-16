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
