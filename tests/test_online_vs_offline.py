"""
Check that the simulation's online diagnostics match the offline post-processing ones.

Several diagnostics that the offline pipeline would otherwise recompute in Python are computed online
by the Julia simulation instead (the filtered fields, the cross-scale KE flux Π_K, the SFS KE
dissipation ε_Kˢ, the Winters sorted reference state, the cross-scale APE flux Π_A, and the sub-filter APE
dissipation ε_Aˢ), and
the pipeline then reads them straight out of the simulation output. Nothing else in the test suite compares the two implementations: the
budget-closure tests in `test_budgets.py` would only notice an online error large enough to break
closure at the 10% level, and they cannot see the sorted state at all.

Rather than reimplement the comparisons, this runs the `postprocessing/validation/inv0*` scripts that
already do them. Each recomputes its diagnostic offline, compares against the online field, and with
`--tolerance` exits nonzero if any relative difference exceeds it (see `validation/aux_check.py`).
Those scripts also write the comparison figures, which is why they run as subprocesses rather than
being imported: they are top-level scripts, not modules.

These scripts are the *figure-producing* half of the offline cross-check; `tests/test_offline_check.py`
is the other half, comparing every term of the assembled online budgets against the offline pipeline's
files. Both run only under `pytest --offline-check` (CI's offline-check job): each script recomputes its
quantity offline, which is minutes of work at the CI resolution.

`inv03` (the S̄/τ tensor components) is not included: it needs a `--save_tensors` run, which writes six
extra 3D fields per filter scale, and the quantities it checks already enter Π_K, which `inv02` covers.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import SIM_OUTPUT

REPO_ROOT = Path(__file__).resolve().parent.parent
VALIDATION = REPO_ROOT / "postprocessing" / "validation"

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
# `inv08` (the sub-filter ε_Aˢ, which `05_sfs_ape_budget.py` computes offline) is approximate for two
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

# `inv09` (the cross-scale APE flux Π_A, which `03_energy_transfer.py` computes offline) is the APE twin of
# `inv02`, and was expected to be as loose: both are a product of two filtered-and-differentiated
# fields, and inv02's map measures 6.7e-01 against 9.4e-02 on its integral. It is not. Measured at
# Nz=192/Re=262:
#   2.7e-02 (field, l=1)   9.9e-04 (int Pi_A dV, l=1)
#   5.3e-02 (field, l=7)   5.5e-03 (int Pi_A dV, l=7)
# an order tighter than Π_K, because Υˡ is a reference height — a lookup into a sorted profile, smooth
# where the strain that Π_K differentiates is not. So 0.15 is ~3x the worst, and unlike the 1.0 this
# replaces it is tight enough to catch a factor of two.
#
# `inv10` is not an online-vs-offline comparison at all: every term of both budgets is now written by
# the simulation, so it checks that the online budgets *close* on their own, with the same metric
# `test_budgets.py` applies to the assembled files (rms(residual) / mean over terms of rms(term)). It is
# held at the same 0.01 threshold, which is what makes the two directly comparable. Measured at CI's
# Nz=512 on 2026-09-28 (first pair skipped, identical in two runs of the same code): KE 0.79% (ℓ=1) and
# 0.17% (ℓ=7), APE 0.43% and 0.38%; at Nz=1024, all records: KE 0.48% and 0.29%, APE 0.14% and 0.18%.
# The offline pipeline's own budget sits at 2.3% (Nz=512) and 1.2% (Nz=1024) on the same runs, which is
# its ε discretisation; test_offline_check.py holds it at 3%.
CASES = [
    pytest.param("inv01_compare_filters.py", 0.25, [], id="filtered_fields"),
    pytest.param("inv02_compare_ke_transfer.py", 1.0, [], id="Pi_K"),
    pytest.param("inv05_compare_dissipation.py", 0.5, [], id="eps_Ks"),
    pytest.param("inv06_compare_sorted_profiles.py", 1e-9, ["--n-workers", "2"], id="sorted_state"),
    pytest.param("inv07_compare_local_ape.py", 1e-6, ["--n-workers", "2"], id="local_ape"),
    pytest.param("inv08_compare_sfs_ape_dissipation.py", 0.30, ["--n-workers", "2"], id="sfs_ape_dissipation"),
    pytest.param("inv09_compare_ape_transfer.py", 0.15, ["--n-workers", "2"], id="Pi_A"),
    pytest.param("inv10_online_budget_closure.py", 0.01, [], id="online_budget_closure"),
]


# Each script recomputes its quantity offline, minutes of work at the CI resolution, so the whole file runs
# only under --offline-check, in CI's offline-check job beside tests/test_offline_check.py.
pytestmark = pytest.mark.offline_check


@pytest.mark.parametrize("script,tolerance,extra", CASES)
def test_online_matches_offline(script, tolerance, extra):
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
