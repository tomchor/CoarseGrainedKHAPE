# How the filtered reference profile ended up the way it is

This is a decision record for the `jw/vertical-filter-ape` work, written so that the next person does not
repeat the measurements. `CLAUDE.md` says what the code *does*; this says **why**, what was tried and
rejected, and what the numbers were. Where something is an inference rather than a measurement, it says so.

The short version: the resolved reservoir is measured against ⟨ρ_*⟩ rather than ρ_*; ⟨ρ_*⟩ is filtered by
FFT at full column resolution, because every scheme for avoiding that cost was solving a problem that did
not exist; and **the reference profile is therefore built in exactly one place — offline — so every term
that depends on it is computed there too.**

That last point is the rule to hold onto, and it splits the budget terms:

| | terms | where |
|---|---|---|
| depend on ⟨b✶⟩ | `Υ̃`, `L̃`, `S̃`, `Π̃_A`, `ε̃ˢ`, `R̃ˢ` | **offline** |
| depend on the sort, but not on ⟨b✶⟩ | `τ(w, b_r)` | **offline**, with the rest |
| depend on neither | `Π_K`, `ε_Kˢ` — velocities alone | either; currently online |

`τ(w, b_r) = filter(w·b_r) − w̄·filter(b_r)` with `b_r = b − b✶(z)` uses the **unfiltered** sorted profile, so
it is untouched by how ⟨b✶⟩ is built and needs no change on that account. But it is **not** interchangeable
between online and offline: the offline pipeline sorts the *z-padded* domain and the simulation sorts only
the true domain, so `b✶` differs. Measured at Nz=128, ℓ=1, restricted to the physical domain and aligned in
time: the online and offline conversion fields have **rms(diff)/rms = 0.68 and correlate at only +0.73**,
while their volume integrals agree to four figures. Agreement in the integral, not pointwise.

Whether the sort or the differing filter implementations dominates that is unresolved; `inv06` runs the
offline sort both padded and unpadded and is the tool for separating them. Until then the safe default is
to compute it offline with everything else rather than mixing a field from one path into a budget built on
the other.

This was not the original design — §2 — and the reason it changed is §6.

---

## 1. Why a filtered reference at all

The pipeline filters in x **and** z. Wenegrat, Chor & Barkan Eq. 2.24–2.26 — measuring both reservoirs
against the unfiltered ρ_* — is the `g_z = δ(z)` horizontal-filter limit, and it is simply wrong for a
kernel with vertical extent: a fluid at rest filters to a profile that still carries APE against ρ_*, so
the resolved reservoir does not vanish at rest and the sub-filter remainder goes negative. Measured on the
synthetic field of `test_jensen.py`, the remainder is negative over 15–48% of the domain, reaching −2.67 of
rms; against ⟨ρ_*⟩ (Eq. 2.3) the same field is non-negative to +2.0e-03.

**Do not revisit this.** `test_jensen.py` and `test_filtered_reference.py` pin it down on a field that
needs no simulation output. The `--reference true` path still exists to reproduce earlier results; nothing
in CI or production uses it.

## 2. Why Π_A and ε_Aˢ *were* computed online, and why they no longer are

`ε_A = κ ∂ᵢb ∂ᵢΥ` is quadratic in gradients, and the offline `calculate_gradient` takes centred
differences, whose `sin(kΔ)/Δ` response kills the 2Δ mode. The online form pairs its two factors on the
face where both differences live and interpolates the product to the centre.

Measured at **Nz=128**: reading the online `ε_Aˢ` took the APE residual from 18.1% to 2.0% of the dominant
term at ℓ=1, and 15.5% to 1.2% at ℓ=7. The offline/online ratio of `ε_Aˢ` drifts 0.97 early to 0.78 once
the interface reaches the grid scale, and the residual matches the discretisation error pointwise in time
(corr +0.995, ratio ~1) — i.e. the offline residual *is* this error, not a physics failure.

**But the case weakens with resolution.** Measured at **Nz=1024** (2026-09-17), the offline and online
`∫ε_Aˢ dV` agree to **0.9927** (ℓ=1) and **0.9909** (ℓ=7), and the residual goes 0.185% → 0.477% (ℓ=1) and
0.287% → 0.320% (ℓ=7) when everything is computed offline. At production resolution the gradient scheme
costs well under a percent.

**That is what made the all-offline pipeline affordable, and §6 is what made it necessary.** Once ⟨ρ_*⟩ is
exact offline and still coarse online, the two are no longer the same construction — so any term built on
the reference profile has to be computed in one place, and that place is where the exact profile lives.
`Π_A`, `ε_Aˢ`, `Υ̃`, `L̃`, `S̃` and `Rˢ` are therefore offline, and the online versions become the
independent cross-check that `inv08`/`inv09`/`inv10` were written to be. The ~0.4% the gradient scheme
costs is the price of that consistency, and it is small because the resolution is high.

`Π_K` and `ε_Kˢ` never touch the reference state at all, so they can continue to be read online without
breaking anything — which also avoids recomputing the parts the simulation does better. `τ(w, b_r)` is a
middle case: independent of ⟨b✶⟩ but not of the sort, and measurably different online versus offline
pointwise (see the table above), so it goes offline with the terms it appears alongside.

Note the setup runs `Re = Re₀·Nz²`, so refining the grid also raises Re and holds the resolution of the
dissipative scale roughly fixed. It was *predicted* on that basis that the discretisation error would not
shrink with resolution. It did shrink, by 25×. Predictions of this kind have not been reliable here.

## 3. The coarse column and `REFERENCE_FILTER_K` — a wrong turn

⟨ρ_*⟩ is a Gaussian convolution of the sorted column. Filtering it at the column's own resolution with
`scipy.ndimage.gaussian_filter1d` costs `O(N · stencil)` with a stencil of `8σ/Δz` slots, and that was
measured as genuinely prohibitive: on the Nz=1024 padded column, **28.9 s per record at ℓ=1, 211 s at ℓ=7,
600 s at ℓ=20**. A 30-scale sweep projects to ~100 h; a real attempt reached 11% in 11 hours.

The response was to block-average the column onto `M = K·Lz/σ` levels ("K levels per σ"), filter there, and
hand that to the lookup. `K = 1000` was chosen as a deliberately safe starting value.

**This was solving the wrong problem.** The cost is a property of *direct convolution*, not of the
operation. The same convolution by FFT is `O(N log N)`: **0.04 / 0.07 / 0.16 s** at ℓ = 1 / 7 / 20 on that
same column — flat in σ — agreeing with the direct result to **2.5e-12 on a field of rms 1025** (2.4e-15
relative). See §6.

### What the coarsening cost, measured

A K ladder at Nz=256 (K ∈ 3000, 1000, 300, 100, 30, 10; K=3000 gives `M = N` and is therefore the exact
answer) showed:

- `Π_A` and `ε_Aˢ` are the K-sensitive terms, as expected — they differentiate `Υ̃`. Terms that *integrate*
  `Υ̃` (`L̃`, `S̃`, `Rˢ`) are ~100× less sensitive.
- **There is no plateau.** Restricted to |z| < 3 where `Π_A` lives, the ℓ=7 deviation from exact is
  0.205 at K=1000, 0.271 at 300, 0.311 at 100, 0.604 at 30, 0.647 at 10. The production value was already
  costing ~20% on the field and 5.5% on the volume integral.
- The ℓ=1 column of that ladder is **not signal** — see §7. Its `Π_A` is 3× smaller and the metric was
  dominated by the padded region.

**Do not look for a good K.** There isn't one; the error grows monotonically as K falls and the only exact
setting is "no coarsening". Both the constant and the `--reference-K` flag added to sweep it are gone.

### Two GPU constraints found along the way

Both were discovered only on the HPC, after a ~27 min compile, and neither reproduces on CPU:

- **`Kernel invocation uses too much parameter memory`** (sm_80, 63.3 KiB against a 32 KiB limit).
  Oceanostics' `GaussianFilterKernel` stores its weights as an `NTuple` *inside the kernel type* and fully
  unrolls the stencil loop — the right design at the widths the x–z filter uses, and fatal at ~8000, where
  the tuple is passed by value into CUDA's parameter space. Fixed in `a46b394` with a repo-local kernel
  holding the weights in a device array (72 bytes as a parameter, whatever the length). **The x–z filter on
  the model grid hits 30.4 KiB at ℓ=7, Nz=4096** — one resolution step from the same failure. The durable
  fix is width dispatch upstream.
- **`Scalar indexing is disallowed`** — `ReferenceTendencyCorrection`'s `compute!` opened its cumulative
  integral with `Ψface[1] = 0`. Fixed in `4f29c99`.

## 4. Lookup quantisation, and the interpolation

Handing the lookup the coarse `(b✶, z✶)` pair quantises `Υ̃ = z̃✶(b̄) − z` to `Lz/M`. `Π̃_A = −τ(uᵢ,b) ∂ᵢΥ̃`
differentiates it with nothing downstream to smooth the steps, and at Nz=1024, ℓ=7 that is a step of
`dz/8.2` — a ~12% staircase in `∂Υ̃/∂z`. It was **visible as striping in the Π_A panel** of the ℓ=7
animation while every term that integrates `Υ̃` stayed smooth.

`d47fe37` filters on the coarse grid but interpolates ⟨b✶⟩ back onto the column before the lookup. Measured
at N=294912, n=35: reading the coarse cells whole leaves **97.1% of adjacent column slots identical** with
jumps of 2.07e-03; interpolating leaves none identical and a largest step of 5.91e-05, 35× smaller. The
lookup is a binary search (`searchsortedfirst`), so N vs M levels costs ~7 more comparisons per cell.

This is still how the **online** path works. Offline, §6 made it moot.

## 5. The reference profile must be monotonic

`ProfileLookup` resolves heights with `searchsortedfirst`/`searchsortedlast`, and on a non-monotonic
profile those "return whatever slot it likes, silently" (Oceanostics' own warning). This is not a
theoretical concern: the sorted column is **99.31% tie runs** at Nz=256, so nearly every point sits on a
flat stretch. Direct convolution of a non-increasing profile with a non-negative kernel comes out exactly
monotonic (zero violations measured); FFT round-off does not. Hence the `np.minimum.accumulate` clamp in
`_fft_gaussian` — it can only remove round-off, never real structure, but it is **not optional**.

## 6. Why the profile is filtered by FFT

`4d052f8`. `fftconvolve` computes a *linear* convolution (it zero-pads internally), and pre-padding the
column with `4σ` copies of each end reproduces `mode='nearest'` exactly. Verified on the real sorted
column at ℓ = 1, 7 and 20 — including ℓ=20, where the padding is 68% of the array and most of the kernel
weight sits on the extension:

| | max\|FFT − direct\| | rms | monotonic |
|---|---|---|---|
| ℓ=7 | 6.3e-14 | 1.33 | both |
| ℓ=20 | 1.1e-13 | 1.18 | both |

and on the real column, 2.5e-12 against rms 1025. The resulting `Π_A` field is **bit-identical** to the
exact direct-convolution result throughout the interface.

So the coarse grid, `K`, the histogram binning and the interpolation back are all gone offline. The profile
is exact, and there is no parameter to choose.

**The online path still coarsens, and that is what forces every reference-dependent term offline.** The
simulation builds ⟨b✶⟩ on a coarse column (§3, §4); offline it is exact. Two different constructions cannot
both feed one budget, so `Π_A`, `ε_Aˢ`, `Υ̃`, `L̃`, `S̃` and `Rˢ` are computed where the exact profile is —
offline — and the online set becomes the cross-check. `Π_K` and `ε_Kˢ` are unaffected: they never touch the
reference state. `τ(w, b_r)` does not depend on ⟨b✶⟩ either, but it does depend on the sort, which differs
between the padded offline domain and the simulation's — so it is computed offline too.

Porting the FFT online would remove that split. It needs a custom `compute!` with CUFFT — buildable
(`ReferenceTendencyCorrection` is the precedent, and Oceananigans already depends on `CUDA.CUFFT`), but it
is GPU code, and see §3 for how that has gone. It is only worth doing if the online diagnostics are meant
to be the scientific product rather than the cross-check.

## 7. The padding was in the integrals

`f8db983`, and the most consequential finding of the lot.

`load_dataset_and_grid` pads the domain with edge-valued cells so the stencil stays inside the array, and
`_pad_domain_in_z` rebuilt `dV` across the whole padded axis — so every `integrate(·, dV)` summed over
manufactured fluid. In that region ⟨ρ_*⟩ is *exactly constant*, so inverting it for `z̃_*` is **degenerate**:
`Π_A` there is noise rather than ~0.

Measured at Nz=256: two computations of ⟨ρ_*⟩ agreeing to 2.4e-15 relative gave `Π_A` fields that were
bit-identical across the interface and **fully decorrelated in the padding** (rms(diff) ≈ rms(Π_A)) —
moving `∫Π_A dV` by **76%** at ℓ=1. The integral was not merely including unphysical fluid; it was not
reproducible.

Zeroing `dV` outside the physical domain removed **32.7% of `∫Π_A dV` at ℓ=1** and 9.7% at ℓ=7.

**This contaminates earlier numbers.** Any offline integral computed before `f8db983` — including the K
ladder's integral column and the residuals quoted above — carries this. The interface-restricted *field*
comparisons do not.

## 8. Things that are measured, and things that are not

Measured and reliable: everything with a number above. The FFT/direct agreement, the K ladder's field
deviations, the padding contribution, the CI timings (277 → 66 min after `f447807`+`65bd51e`; setup 190 →
26 min), `inv10`'s fully-online closure at Nz=1024 (0.03–0.20% of the dominant term across both budgets and
both scales).

**Unresolved:** the `S̃ ≥ 0` failure at ℓ=1 sits at −1.4e-03 of rms at Nz=512 against a 1e-03 tolerance, and
neither ties, clamping, the transport term nor `Υ̃` quantisation explains it. It improves with resolution
(p ≈ 2.2). §7 may bear on it and has not been rechecked since.

**A methodological note, earned the hard way.** Operation counts repeatedly misled here — by 800× (direct
vs FFT), by 17× (the sweep's per-unit-ℓ cost), and in the opposite direction when "infeasible" turned out
to be a few node-hours under dask. Timing claims in this project should be measured, not derived, and a
projection from a single-core laptop measurement to an 18-worker Casper node is worth nothing.
