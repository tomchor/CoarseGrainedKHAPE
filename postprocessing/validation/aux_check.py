"""
Tolerance checking shared by the online-vs-offline validation scripts (`inv0*`).

Those scripts exist to be read: they recompute a diagnostic offline, compare it against the
simulation's online version, print relative differences and save comparison figures. This module lets
the same scripts double as a pass/fail check without duplicating any of that comparison code, which is
what `tests/test_online_vs_offline.py` runs in CI.

A script opts in by calling `add_tolerance_arg(parser)`, then `set_tolerance(args.tolerance)`, wrapping
each metric it already prints in `check(...)`, and calling `finalize()` at the end. Without
`--tolerance` the behaviour is unchanged: everything is reported and nothing fails.
"""
_TOLERANCE = None
_FAILURES = []
_CHECKED = 0


def add_tolerance_arg(parser):
    """Add the `--tolerance` option that turns the script into a pass/fail check."""
    parser.add_argument("--tolerance", type=float, default=None,
                        help="If set, exit nonzero when any reported relative difference exceeds this "
                             "value. Without it the script only reports, which is the default.")


def set_tolerance(tolerance):
    global _TOLERANCE
    _TOLERANCE = tolerance


def check(value, message, log=print):
    """Report a relative difference, and record it as a failure if it exceeds the tolerance.

    `message` is the line the script would have printed anyway; a PASS/FAIL marker is appended only
    when a tolerance is in force. NaN counts as a failure — it means the comparison did not produce a
    number, which is never a passing result.
    """
    global _CHECKED
    if _TOLERANCE is None:
        log(message)
        return value

    _CHECKED += 1
    ok = value == value and value <= _TOLERANCE   # `value == value` is False for NaN
    if not ok:
        _FAILURES.append((message.strip(), value))
    log(f"{message}   [{'PASS' if ok else 'FAIL'} vs tol {_TOLERANCE:.1e}]")

    return value


def finalize(log=print):
    """Exit nonzero if any checked value exceeded the tolerance. A no-op without `--tolerance`."""
    if _TOLERANCE is None:
        return

    if _FAILURES:
        lines = "\n".join(f"    {msg}   (value {val:.3e} > tol {_TOLERANCE:.1e})" for msg, val in _FAILURES)
        raise SystemExit(f"\n{len(_FAILURES)} of {_CHECKED} online-vs-offline checks exceeded the "
                         f"tolerance:\n{lines}")

    if _CHECKED == 0:
        # Silence here would read as success while actually checking nothing.
        raise SystemExit("--tolerance was given but no checks ran; the script compared nothing.")

    log(f"All {_CHECKED} online-vs-offline checks within tolerance {_TOLERANCE:.1e}.")
