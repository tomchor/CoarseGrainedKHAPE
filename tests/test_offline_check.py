"""
The offline budget pipeline as a test: does it reproduce the budgets assembled from the online terms?

The production budgets are read off the simulation (`postprocessing/01_online_budgets.py`). The offline
pipeline that used to compute them (`postprocessing/offline/`, steps 01-05: scipy filtering on a z-padded
domain, a numpy sort of the density, an FFT-filtered reference profile, centred gradients) is an independent
implementation of every term, and this test runs it and compares the two, term by term:

  * every 3D field of the KE and APE budgets (the offline pipeline's field files against the online terms in the
    simulation output), as rms(offline - online) / rms(online) over the physical domain (the offline fields carry
    the padding; it is cut here) at every record with t >= T_MIN;
  * every integrated term, as max|offline - online| / rms(online) over the same records;
  * the per-record minimum of S̃ (Eaˢ) at every record, as |min_off - min_on| / rms(online), so the two
    agree on where S̃ dips below zero during the small-amplitude phase and not only on its bulk;
  * the offline budget's own closure, rms(residual) / mean over terms of rms(term) over the same records,
    the metric of tests/test_budgets.py at a looser threshold (CLOSURE_THRESHOLD below says why).

Records before T_MIN are excluded from the field comparisons: there the flow is a small-amplitude wave, S̃
is a ~7% residual of two nearly equal quantities, and relative differences of order 1e-3 are the numerics
of that cancellation rather than of the pipelines.

Tolerances were set from the Nz=1024, Re=524 run of 2026-09-26, at about three times the measured value
(measured: S̃ 1e-5, Rˢ 2e-4, τ 1e-3, ε_Aˢ 6e-3, Π_A 2e-2 for the fields; integrals to five figures). The
differences that remain are discretisation, not construction: online gradients pair their factors on cell
faces where the offline ones are centred, the online ⟨b✶⟩ is block-averaged where the offline one is exact,
and Kˢ interpolates the square where the offline one squares the interpolated velocity. A sign error, a
factor of two or a stale field would exceed every one of these by orders of magnitude. The measured value of
every comparison is printed, so the tolerances can be tightened from a green run.

Runs only under `pytest --offline-check` (CI's offline-check job): the offline pipeline takes about an hour
at the CI resolution. `offline/run_offline_budgets.sh` is invoked here unless its output already exists.
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from conftest import OFFLINE_PP_OUTPUT, PP_OUTPUT, SIM_OUTPUT, STEM

pytestmark = pytest.mark.offline_check

REPO_ROOT = Path(__file__).resolve().parent.parent
POSTPROCESSING = REPO_ROOT / "postprocessing"
FILTER_SCALES = ["1", "7"]   # the CI run's --filter_ls; the offline pipeline is run at the same scales
T_MIN = 10.0                 # records before this are the small-amplitude phase (see the docstring)
MIN_RECORDS = 3              # a run too short to have that many records past T_MIN is compared on all but its first record

#+++ Tolerances: rms(offline - online) / rms(online) per field, max|Δ∫| / rms(∫) per integral
FIELD_TOL = {
    # KE budget
    "KE_of_sfs_flow":       1e-2,   # interpolate-the-square (online) vs square-of-interpolated (offline); unmeasured, provisional
    "∂ₜ SFS KE":            1e-2,   # the same difference, differenced in time
    "Π_K":                  3e-1,   # offline recompute from centred strain (inv02: 6.7e-1 at Nz=128/Re=262, far tighter at Nz=1024)
    "ε_Kˢ":                 5e-1,   # difference of two comparable quantities that amplifies the strain-stencil gap (inv05: 3.2e-1)
    "SFS APE->KE exchange": 3e-3,   # τ(w,b_r): measured 1e-3 at t=2, 1e-4 later
    # APE budget
    "Ea(ρ, z)":             1e-4,   # inv07 measured 1e-11 with the 3D sort; the lookup's tie convention differs, hence the margin
    "Ea(ρ̄, z)":             1e-4,   # L̃
    "Ēa(ρ, z)":             1e-4,   # Ē_A
    "Eaˢ(ρ, z)":            1e-4,   # S̃: measured 1e-5 from t=12 on
    "∂ₜ SFS APE":           1e-3,
    "Π_A":                  5e-2,   # measured 1e-2 to 2e-2
    "ε_Aˢ":                 5e-2,   # 4e-3 to 6e-3 at Nz=1024, 2.1e-2 at CI's Nz=512 (centred vs face-paired gradients, grows with Δz)
    "SFS KE->APE exchange": 3e-3,   # -τ(w,b_r)
    "Rˢ":                   1e-3,   # measured 2e-4
}
INTEGRAL_TOL = {   # max|offline - online| over records / rms(online); measured values in the comments
    "∫-∂ₜ SFS KE dV":    1e-2,   # 3e-3
    "∫Π_K dV":           3e-1,   # offline recompute; inv02 measured 9.4e-2 on the integral at Nz=128/Re=262
    "∫-ε_Kˢ dV":         5e-1,   # offline recompute; inv05 measured 3.2e-1 at Nz=128/Re=262
    "∫(SFS APE->KE) dV": 1e-3,   # 4e-7
    "∫-∂ₜ SFS APE dV":   1e-3,   # 3e-5
    "∫Π_A dV":           1e-2,   # 2e-4 at Nz=1024; 9.3e-4 and 2.6e-3 at ℓ=7 in two Nz=512 runs of the same code
    "∫-ε_Aˢ dV":         1e-1,   # 2.7e-2 at single records (the integral over the run agrees to 0.5%)
    "∫(SFS KE->APE) dV": 1e-3,   # 4e-7
    "∫Rˢ dV":            5e-3,   # 2e-4 at Nz=1024; 9.6e-4 and 1.2e-3 at ℓ=7 in two Nz=512 runs of the same code
}
# The offline budget's own closure, with tests/test_budgets.py's metric (rms(residual) / mean of rms(terms)) at three
# times its 1% threshold. The excess over the online residual is the offline ε_Aˢ alone: its gradients are centred where
# the online ones are face-paired, and that difference correlates with residual_off - residual_on at +1.0000 and is 1.6%
# of ε_Aˢ's own rms at Nz=512, growing with Δz. Measured over t >= T_MIN: 2.34% and 1.97% (APE, ℓ=1 and 7), 2.25% and
# 0.60% (KE) at Nz=512, identical in two runs, against 0.43%, 0.38%, 0.80% and 0.17% online; 0.93%, 0.84%, 1.24% and
# 0.46% at Nz=1024. With the online ε_Aˢ substituted, the offline residual equals the online one to 1e-5, so the two share
# one floor and this threshold cannot be tighter than the online one. It is a sanity check; the tolerances above are the test.
CLOSURE_THRESHOLD = 0.03
SFS_APE_MIN_TOL = 5e-3    # |min_off - min_on| / rms(online) of S̃, record by record, every record (Nz=1024: ~1e-6; Nz=64: 4e-3)
#---


#+++ Run the offline pipeline once per session (or reuse its output)
@pytest.fixture(scope="session")
def offline_files():
    if not SIM_OUTPUT.exists():
        pytest.skip(f"simulation output not found: {SIM_OUTPUT}")
    expected = [OFFLINE_PP_OUTPUT / f"{STEM}_sfs_{k}_budget_{w}.nc" for k in ("ke", "ape") for w in ("fields", "integrated")]
    if not all(f.exists() for f in expected):
        cmd = ["bash", str(POSTPROCESSING / "offline" / "run_offline_budgets.sh"), str(SIM_OUTPUT), "--filter-scales", *FILTER_SCALES]
        env = {**os.environ, "KHAPE_PP_OUTPUT": str(PP_OUTPUT), "MPLBACKEND": "Agg"}
        print(f"\nRunning the offline pipeline: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=POSTPROCESSING, env=env, capture_output=True, text=True)
        if result.returncode != 0:
            pytest.fail(f"offline pipeline failed (exit {result.returncode})\n--- stdout ---\n{result.stdout[-4000:]}\n"
                        f"--- stderr ---\n{result.stderr[-4000:]}")
    missing = [f for f in expected if not f.exists()]
    assert not missing, f"offline pipeline wrote no {missing}"
    return expected


def _open(path):
    return xr.open_dataset(path, decode_times=False, chunks={"time": 1})


@pytest.fixture(scope="session")
def budgets(offline_files, online_budget):
    """(kind -> (online fields, offline fields, online integrated, offline integrated)), time axes aligned."""
    out = {}
    for kind in ("ke", "ape"):
        on_f  = online_budget[kind]   # the online 3D terms, straight from the simulation output
        off_f = _open(OFFLINE_PP_OUTPUT / f"{STEM}_sfs_{kind}_budget_fields.nc")
        on_i  = _open(PP_OUTPUT / f"{STEM}_sfs_{kind}_budget_integrated.nc")
        off_i = _open(OFFLINE_PP_OUTPUT / f"{STEM}_sfs_{kind}_budget_integrated.nc")
        # Both are stated at the upper record of each output pair; align by nearest time and insist they agree.
        off_f = off_f.reindex(time=on_f.time, method="nearest")
        off_i = off_i.reindex(time=on_i.time, method="nearest")
        assert on_f.attrs.get("n_pad_z", 0) == 0, "the online budget is expected on the unpadded grid"
        out[kind] = (on_f, off_f, on_i, off_i)
    return out


def _developed(da):
    """The records the bulk comparisons use: t >= T_MIN, or all but the first when the run is too short for that."""
    late = da.sel(time=slice(T_MIN, None))
    if late.sizes["time"] >= MIN_RECORDS:
        return late
    print(f"\n  note: only {late.sizes['time']} records at t >= {T_MIN:g}; comparing every record after the first instead")
    return da.isel(time=slice(1, None))


def _physical(off_da, on_da):
    """The offline field on the online (unpadded) z grid."""
    z_on = on_da.z_aac.values
    return off_da.sel(z_aac=z_on, method="nearest").assign_coords(z_aac=z_on)


def rms(a):
    return float(np.sqrt(np.nanmean(np.asarray(a, dtype=float) ** 2)))
#---


#+++ Fields
FIELD_CASES = [(kind, var) for kind, names in (("ke", ["KE_of_sfs_flow", "∂ₜ SFS KE", "Π_K", "ε_Kˢ", "SFS APE->KE exchange"]),
                                                ("ape", ["Ea(ρ, z)", "Ea(ρ̄, z)", "Ēa(ρ, z)", "Eaˢ(ρ, z)", "∂ₜ SFS APE", "Π_A",
                                                         "ε_Aˢ", "SFS KE->APE exchange", "Rˢ"])) for var in names]


@pytest.mark.parametrize("kind,var", FIELD_CASES, ids=[f"{k}:{v}" for k, v in FIELD_CASES])
@pytest.mark.parametrize("ell", [float(s) for s in FILTER_SCALES])
def test_field_matches_offline(budgets, kind, var, ell):
    on_f, off_f, _, _ = budgets[kind]
    if var not in on_f:
        pytest.skip(f"'{var}' not in the online budget: the simulation output predates the online L̃ and Ē_A")
    assert var in off_f, f"'{var}' missing from the offline budget file"
    on  = _developed(on_f[var].sel(filter_scale=ell, method="nearest"))
    off = _physical(off_f[var].sel(filter_scale=ell, method="nearest").sel(time=on.time), on)
    diff = rms((off - on).values)
    scale = rms(on.values)
    rel = diff / scale if scale > 0 else float("inf")
    tol = FIELD_TOL[var]
    print(f"\n  {kind.upper():<3} {var:<22} ℓ={ell:g}  rms(off-on)/rms(on) = {rel:.3e}   rms(on) = {scale:.3e}   "
          f"({'PASS' if rel < tol else 'FAIL'}, tol {tol:.0e})")
    assert rel < tol, f"{var} at ℓ={ell:g}: offline and online differ by {rel:.3e} of rms (tolerance {tol:.0e})"
#---


#+++ Integrals
@pytest.mark.parametrize("kind", ["ke", "ape"])
@pytest.mark.parametrize("ell", [float(s) for s in FILTER_SCALES])
def test_integrals_match_offline(budgets, kind, ell):
    _, _, on_i, off_i = budgets[kind]
    print(f"\n  {kind.upper()} integrals, ℓ={ell:g}  (records t >= {T_MIN:g})")
    failures = []
    terms = [v for v in on_i.data_vars if "residual" not in v]
    for var in terms:
        assert var in off_i, f"'{var}' missing from the offline integrated file"
        on_da = _developed(on_i[var].sel(filter_scale=ell, method="nearest"))
        on  = on_da.values
        off = off_i[var].sel(filter_scale=ell, method="nearest").sel(time=on_da.time).values
        scale = rms(on)
        rel = float(np.nanmax(np.abs(off - on))) / scale if scale > 0 else float("inf")
        tol = INTEGRAL_TOL[var]
        ok = rel < tol
        print(f"    {var:<22} max|off-on|/rms(on) = {rel:.3e}   rms(on) = {scale:.3e}   ({'PASS' if ok else 'FAIL'}, tol {tol:.0e})")
        if not ok:
            failures.append((var, rel))
    # The residuals are tiny differences of the terms, so they are not compared to each other; the offline budget
    # is instead held to the closure metric test_budgets.py applies to the online one, at CLOSURE_THRESHOLD. The
    # records matter: over all records the offline KE closure at ℓ=1 is 2.98% at Nz=512, against 2.25% from T_MIN on.
    residual = [v for v in on_i.data_vars if "residual" in v][0]
    times = _developed(on_i[residual].sel(filter_scale=ell, method="nearest")).time
    off_res = off_i[residual].sel(filter_scale=ell, method="nearest").sel(time=times).values
    off_terms = [rms(off_i[v].sel(filter_scale=ell, method="nearest").sel(time=times).values) for v in terms]
    closure = rms(off_res) / np.mean([s for s in off_terms if s > 0])
    print(f"    offline {residual:<14} rms(residual)/mean(rms(terms)) = {closure:.3%}   ({'PASS' if closure < CLOSURE_THRESHOLD else 'FAIL'}, threshold {CLOSURE_THRESHOLD:.0%})")
    assert closure < CLOSURE_THRESHOLD, f"the offline {kind.upper()} budget does not close at ℓ={ell:g}: {closure:.3%}"
    assert not failures, f"{kind.upper()} integrals at ℓ={ell:g} differ: {failures}"
#---


#+++ The early-time minima of S̃ agree
@pytest.mark.parametrize("ell", [float(s) for s in FILTER_SCALES])
def test_sfs_ape_minima_agree(budgets, ell):
    """S̃ dips below zero during the small-amplitude phase in both pipelines; check they agree on how far."""
    on_f, off_f, _, _ = budgets["ape"]
    on  = on_f["Eaˢ(ρ, z)"].sel(filter_scale=ell, method="nearest")
    off = _physical(off_f["Eaˢ(ρ, z)"].sel(filter_scale=ell, method="nearest"), on)
    scale = rms(on.values)
    mins_on  = on.min(dim=[d for d in on.dims if d != "time"]).values
    mins_off = off.min(dim=[d for d in off.dims if d != "time"]).values
    rel = float(np.nanmax(np.abs(mins_off - mins_on))) / scale
    worst = int(np.nanargmin(mins_on))
    print(f"\n  S̃ per-record minima, ℓ={ell:g}: max|min_off - min_on|/rms(on) = {rel:.3e}; "
          f"most negative online record t={float(on.time[worst]):.2f}: min_on={mins_on[worst]:+.3e}, min_off={mins_off[worst]:+.3e}")
    assert rel < SFS_APE_MIN_TOL
#---
