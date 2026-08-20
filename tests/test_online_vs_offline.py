"""
Check that the simulation's online diagnostics match the offline post-processing ones.

Several diagnostics that the offline pipeline would otherwise recompute in Python are computed online
by the Julia simulation instead (the filtered fields, the cross-scale KE flux Π_K, the SFS KE
dissipation ε_Kˢ, the Winters sorted reference state, and the sub-filter APE dissipation ε_Aˢ), and
the pipeline then reads them straight out of the simulation output. Nothing else in the test suite compares the two implementations: the
budget-closure tests in `test_budgets.py` would only notice an online error large enough to break
closure at the 10% level, and they cannot see the sorted state at all.

Rather than reimplement the comparisons, this runs the `postprocessing/validation/inv0*` scripts that
already do them. Each recomputes its diagnostic offline, compares against the online field, and with
`--tolerance` exits nonzero if any relative difference exceeds it (see `validation/aux_check.py`).
Those scripts also write the comparison figures, which is why they run as subprocesses rather than
being imported: they are top-level scripts, not modules.

Only the time-varying reference is covered. Most of these quantities are reference-independent — a
filtered field, Π_K and ε_Kˢ are built from velocities alone, and the sorted state is a property of the
instantaneous buoyancy field — so running them again under `--fixed-reference` would repeat identical
work on identical inputs. ε_Aˢ is the exception: it does depend on the reference state, and the online
one is always the sort of the current buoyancy, so `05_sfs_ape_budget.py` reads it only for the
time-varying reference and recomputes it offline under `--fixed-reference`. There is nothing to compare
in that variant either way.

`inv03` (the S̄/τ tensor components) is not included: it needs a `--save_tensors` run, which writes six
extra 3D fields per filter scale, and the quantities it checks already enter Π_K, which `inv02` covers.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
VALIDATION = REPO_ROOT / "postprocessing" / "validation"
SIM_OUTPUT = REPO_ROOT / "output" / "khi_Nz512_Ri0.10.nc"

# Tolerances are on rms(online - offline) / rms(online), and were calibrated by measurement rather than
# guessed. They fall into two groups.
#
# The *field* comparisons cannot agree to roundoff. The online and offline paths use the same Gaussian
# kernel but not the same arithmetic: Oceananigans operators on the model grid versus scipy on a
# z-padded domain, differentiating fields that have been written to disk and reloaded. The residual
# scales with how well the filter is resolved — σ = ℓ/2.355 in cells — so the ℓ=1 quantities, and
# anything built from a product of two filtered-and-differentiated fields (Π_K), are the loose ones.
#
# Calibrated on a run at **Re = 262, matching CI's Nz=512 exactly** (Re = Re₀·Nz², so Nz=128 with
# Re₀=1.6e-2 reproduces CI's Reynolds number at a quarter the cost). Calibrating at CI's *resolution*
# but not its Reynolds number is useless: at Nz=64/Re₀=1e-3 the flow is Re≈4, the instability never
# develops, Π_K decays to ~1e-10, and every relative metric becomes noise divided by noise.
#
# Worst measured value per script, at Re=262 (the tolerance is roughly 2x that, as headroom):
#   inv01  1.3e-01   w_ℓ7          (w is small-scale, so a wide filter leaves little signal)
#   inv02  6.7e-01   Π_K ℓ=1 map   (∫Π_K dV agrees to 9.4e-02 — the bulk transfer is right, the map is noisy)
#   inv05  3.2e-01   ∫ε_Kˢ ℓ=1
# These are upper bounds at σ≈2.2 cells for ℓ=1; CI resolves the same filter over ~8.7 cells, so the
# real values there should be markedly smaller and these can be tightened once a green run reports them.
# Even as they stand they catch what this test is for: a sign error, a factor of two, a filter
# mismatch, or a diagnostic silently reading a stale field.
#
# `inv06` is different in kind. It asserts only quantities that are exactly equal by construction — the
# sorted profile is a permutation of the same buoyancy values, and the three sorting methods must agree
# on ∫E_b — so it is held near machine precision (worst measured 6.8e-13). Its genuinely approximate
# comparisons (the offline nearest-density z₀ lookup, and the padded-domain sort, which differs from
# the online ∫E_b by ~3%) are reported but deliberately not asserted: they measure a methodological
# difference this script exists to quantify, not a regression.
#
# `inv07` also belongs to the exact-by-construction group. It compares the local APE computed two ways:
# online, Eₐ from the ThreeDimensionalSort z✶ (ranked slots); offline, the same Holliday–McIntyre
# integral from the nearest-density z₀ lookup. Those disagree only over tied buoyancies — and Eₐ ≈ 0
# there for both (a parcel in uniform fluid sits at its own reference height), so unlike inv06's sorted
# *state* the local *energy* is insensitive to the tie-handling. Measured at Re=262 the field and the
# volume integral both agree to ~1e-11, so it is held at 1e-6 (six orders of headroom over the measured
# value, and still far below the percent level a real physics regression would show).
#
# `inv08` (the sub-filter ε_Aˢ that `05_sfs_ape_budget.py` reads online) is approximate for two
# reasons. First, discretization: the online form pairs its two factors on the face where both
# differences live and interpolates the product to the cell center, while the offline
# `calculate_gradient` takes centered derivatives at the center and multiplies those, which filters out
# exactly the grid-scale correlation that product is made of. Second, a definitional difference in the
# second term: online ε_Aˡ contracts the filtered flux filter(κ∂ᵢb) with ∇Υˡ, while offline it rebuilds
# the flux from the filtered density as κ∇ρ̄. For the constant κ used here those agree in the interior
# (filtering and differencing are both convolutions on a uniform grid, so they commute) and part ways
# only against the walls. Being a difference of two comparable quantities, ε_Aˢ could have amplified
# the first gap the way ε_Kˢ does; measured at Nz=192/Re=262 it does not:
#   7.8e-02 (field, l=1)   1.4e-01 (int eps_As dV, l=1)
#   4.0e-02 (field, l=7)   1.1e-01 (int eps_As dV, l=7)
# so 0.30 is ~2x the worst, and stays under the 0.5 a factor-of-two error would produce.

CASES = [
    pytest.param("inv01_compare_filters.py", 0.25, [], id="filtered_fields"),
    pytest.param("inv02_compare_ke_transfer.py", 1.0, [], id="Pi_K"),
    pytest.param("inv05_compare_dissipation.py", 0.5, [], id="eps_Ks"),
    pytest.param("inv06_compare_sorted_profiles.py", 1e-9, ["--n-workers", "2"], id="sorted_state"),
    pytest.param("inv07_compare_local_ape.py", 1e-6, ["--n-workers", "2"], id="local_ape"),
    pytest.param("inv08_compare_sfs_ape_dissipation.py", 0.30, ["--n-workers", "2"], id="sfs_ape_dissipation"),
]


@pytest.mark.parametrize("script,tolerance,extra", CASES)
def test_online_matches_offline(script, tolerance, extra, pytestconfig):
    # Skip under `--ref-suffix _fixed_ref` rather than relying on the caller not to ask for it: none of
    # these comparisons involve the reference profile, so the fixed-reference run would be a byte-for-byte
    # repeat of the time-varying one.
    ref_suffix = pytestconfig.getoption("--ref-suffix")
    if ref_suffix:
        pytest.skip(f"reference-independent comparison; runs only for the time-varying reference "
                    f"(got --ref-suffix {ref_suffix!r})")

    if not SIM_OUTPUT.exists():
        pytest.skip(f"simulation output not found: {SIM_OUTPUT} (run the simulation first)")

    cmd = [sys.executable, "-u", str(VALIDATION / script),
           "--filename", str(SIM_OUTPUT), "--tolerance", repr(tolerance), *extra]

    # Headless: these scripts save figures, and CI has no display.
    env = {**os.environ, "MPLBACKEND": "Agg"}
    result = subprocess.run(cmd, cwd=VALIDATION, env=env, capture_output=True, text=True)

    # The scripts log to stderr (logging's default), so surface both streams on failure — the
    # per-quantity PASS/FAIL lines are what make a failure diagnosable.
    if result.returncode != 0:
        pytest.fail(f"{script} reported an online-vs-offline mismatch "
                    f"(exit {result.returncode}, tolerance {tolerance:g})\n"
                    f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}")

    print(result.stdout or result.stderr)
