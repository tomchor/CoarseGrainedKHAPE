# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

KHAPE (Kelvin-Helmholtz Available Potential Energy) computes Available Potential Energy (APE) from Kelvin-Helmholtz instability simulations using the Winters et al. (1995) sorting method. The pipeline is:
1. **Julia simulation** (Oceananigans.jl on GPU) -> NetCDF output
2. **Python post-processing** -> filter fields, sort density, compute energy transfer and SFS budgets, plot

GitHub remote: `git@github.com:tomchor/CoarseGrainedKHAPE.git`

## Running the Code

### Full pipeline (simulation + post-processing + sweep)
```bash
bash submit_all_pbs.sh                        # default Nz=2048, FIXED_REF=0
bash submit_all_pbs.sh NZ=1024 FIXED_REF=1   # custom resolution, fixed reference
```
Jobs are chained via PBS `afterok` dependencies. Always use `submit_*.sh` wrappers, never submit `*.pbs` files directly.

### Simulation only
```bash
bash submit_simulation.sh NZ=2048
```
Account: `UMCP0023`, queue: `casper`, 1x A100, 8 cores, 64 GB RAM.

The Julia simulation accepts CLI args: `--Nz`, `--Ri`, `--stop_time`, `--Re0`, `--Pr`, `--U`, `--h`, `--perturbation_amplitude`, `--filter_ls` (one or more online filter length scales ℓ; default `1 7`), `--save_tensors` (flag; also writes the per-scale strain/stress tensor components for online-vs-offline validation), and `--save_sorted` (flag; also writes the Winters (1995) sorted reference state under all three Oceanostics sorting methods). For local CPU development:
```bash
julia --project -t 8 kelvin_helmholtz_instability.jl
julia --project -t 8 kelvin_helmholtz_instability.jl --Nz 512 --Ri 0.1 --stop_time 70 --Re0 1e-3
```

### Post-processing only
```bash
cd postprocessing
bash submit_budgeting.sh NZ=2048 FIXED_REF=both   # runs both reference profile variants
bash submit_sweep.sh NZ=2048 FIXED_REF=both
```
`FIXED_REF=0` (default) recomputes the reference profile each timestep; `FIXED_REF=1` fixes it to t=0; `FIXED_REF=both` submits both variants sharing a single filter job.

### Local post-processing (no PBS)
```bash
cd postprocessing
bash 00_get_budgets.sh output/khi_Nz512_Ri0.10.nc --filter-scales 1 7
bash 00_get_budgets.sh output/khi_Nz512_Ri0.10.nc --filter-scales 1 7 --fixed-reference
```
Set `N_WORKERS` env var to control parallelism (default 1): `N_WORKERS=4 bash 00_get_budgets.sh ...`

### Running tests
```bash
pytest tests/ -v -s                                  # default (time-varying reference)
pytest tests/ -v -s --ref-suffix _fixed_ref          # fixed reference variant
pytest tests/test_gaussian_filter.py -v -s           # single file (filter unit tests, no pipeline output needed)
pytest "tests/test_budgets.py::test_ke_budget_residual" -v -s   # single test
```
`test_budgets.py` checks SFS KE and APE budget closure (rms(residual)/min(rms(terms)) < 10%) and **requires the post-processing output** in `postprocessing/output/` for `khi_Nz1024_Ri0.10`, the CI run (run `00_get_budgets.sh` first; it is parametrized per filter scale via `conftest.py`). That name is set once, as `STEM` in `tests/conftest.py`, with `SIM_OUTPUT` derived from it; the test modules import both from there, so testing a different run means changing that one line. `test_filter.py` / `test_gaussian_filter.py` are self-contained unit tests of the offline `GaussianFilter` and need no pipeline output.

`test_positivity.py` checks the energies whose sign is fixed by construction: the local APE Eₐ (of the full field, of the filtered field, and filtered) and the SFS KE ½τᵢᵢ, each held to `min(field) > -tol x rms(field)`. The tolerance is 1e-3 for the APE, whose nearest-density z✶ lookup leaves a third of the domain a hair below zero at ~1e-5 of rms, and 1e-8 for the SFS KE, which is non-negative by Jensen and so may only lose roundoff. It reads the `*_budget_fields*.nc` output of steps 04/05, plus the online `E_a` from the simulation output when the run used `--save_sorted` (it skips that check otherwise). It also checks the SFS APE S̃ (`test_sfs_ape_is_positive`), gated on the file's `ape_reference` attribute so it runs for the filtered reference and skips for `--reference true`, where negative values are what the remainder construction genuinely produces. `test_jensen.py` explains why: on a synthetic wave-displaced stratification it checks that the remainder Eₐˢ **is** non-negative pointwise when the filter acts in x alone (eₐ(·, z) is convex in b, so Jensen applies when every kernel weight sits at one height) and that it goes negative once the filter also acts in z, which is what `filtered_dimensions` does. Those synthetic tests need no simulation or post-processing output; they are the justification for the filtered-reference construction, so they stay. The file's last test applies the bound to the simulation's own `E_as_ℓ<ℓ>`, which is now S̃ and so is asserted non-negative — to 1e-3 of rms, as `test_positivity`, since online S̃ carries the nearest-slot z✶ lookup on both profiles plus the M-level granularity of the coarse column ⟨b✶⟩ is filtered on.

### CI
GitHub Actions (`.github/workflows/test.yml`) runs on push to `main` and on PR comments starting with `test` (via `test_trigger.yml`). The CI pipeline: Julia simulation (Nz=1024, run with `--save_sorted`) -> post-processing (both reference variants in parallel) -> pytest -> animation generation. Uses `pip install -r tests/requirements.txt` (not conda). Artifacts uploaded per run: `budget-plots-*` (`figures/*.png`), `validation-plots-time-varying` (`figures/validation/*.png`, the online-vs-offline comparison figures), `animation-online` (the mp4s the simulation job itself draws from the online 2D fields, panels animations included), `animation-*` (the offline-pipeline versions, so the two can be compared side by side), and `postprocessing-output-*`. All are uploaded on `always()`, so they survive a failing test — which is when the comparison plots are most useful.

### Python environment
```bash
conda env create -f environment.yml   # creates env "py313"
conda activate py313
```

## Architecture

### Post-processing pipeline (`postprocessing/`)

Sequential numbered scripts (01-06), each reading the previous step's output. `00_get_budgets.sh` runs them all in sequence:

| Script | Purpose |
|--------|---------|
| `01_filter_fields.py` | Gaussian-filter velocity and buoyancy at multiple length scales |
| `02_sort_density.py` | Sort density to compute reference state (Winters et al. 1995) |
| `03_energy_transfer.py` | APE↔KE exchange (Π_K and Π_A are computed online — see Data flow) |
| `04_sfs_ke_budget.py` | Sub-filter-scale KE budget terms |
| `05_sfs_ape_budget.py` | Sub-filter-scale APE budget terms |
| `06_plot_budgets.py` | Plot budget time series |

`sweep*` scripts are the sweep variant (parameter sweep over filter scales): `sweep1_filter_fields.py` filters, `sweep2_energy_transfer.py` computes transfer, `sweep3_plot_transfer_spectrum.py` plots spectra.

`postprocessing/validation/` holds the online-vs-offline comparison scripts, each of which recomputes a quantity offline and compares it against the simulation's online output: `inv01_compare_filters.py` (filtered fields), `inv02_compare_ke_transfer.py` (Π_K), `inv03_compare_tensor.py` (the S̄/τ tensor components), `inv05_compare_dissipation.py` (SFS KE dissipation ε_Kˢ), `inv06_compare_sorted_profiles.py` (the Winters sorted reference state: the sorted profile b✶(z✶) and the reference height z✶, for all three sorting methods), `inv07_compare_local_ape.py` (the local available potential energy Eₐ, field and integral), `inv08_compare_sfs_ape_dissipation.py` (the sub-filter APE dissipation ε_Aˢ at every filter scale, field and integral), and `inv09_compare_ape_transfer.py` (the cross-scale APE flux Π_A). `inv04_animate_comparison.py` makes the animated online | offline | difference version for a chosen `--field` (Π_K, ε_Ks, or a filtered field). `validation.pbs` runs them all. Each writes exactly **one** figure, to `figures/validation/` (their own subdirectory, so they stay separable from the budget and paper figures in `figures/`); CI uploads that directory as the `validation-plots-time-varying` artifact. The tensor-component comparison (`inv03`) needs a `--save_tensors` run and the sorted-state, local-APE and APE-dissipation comparisons (`inv06`, `inv07`, `inv08`, `inv09`) need a `--save_sorted` run; Π_K and ε_Kˢ are always written.

Standalone visualization scripts (not part of the numbered pipeline):
- `plot1_panels.py` -- 4-panel snapshot of local SFS budget fields
- `plot2_budgets.py` -- 2x2 panel of SFS KE and APE budget time series
- `plot3_plot_transfer_spectrum.py` -- cross-scale transfer spectra
- `anim1_panels.py` -- animated version of plot1 panels (requires ffmpeg)

Shared utilities:
- `aux00_utils.py` -- data loading (`load_dataset_and_grid`), filtering (`filter_fields`, `GaussianFilter`, `DaskParallelFilter`), domain padding (`_pad_domain_in_z`), spatial derivatives (`calculate_gradient`), tensor condensing (`condense_velocities`)
- `aux01_pe_functions.py` -- density sorting (`sorted_timeseries`), potential energy calculations (`local_potential_energies_timeseries`), APE budget terms (SFS flux tensor, cross-scale APE flux, total APE dissipation `calculate_ape_dissipation`, SFS APE dissipation, reference-tendency correction R)
- `aux02_ke_functions.py` -- SFS stress tensor, strain rate tensor, cross-scale KE flux, SFS KE dissipation, full energy transfer pipeline (`calculate_energy_transfer`, with an `include_pi_k` flag to skip Π_K)
- `aux03_plotting.py` -- plotting helpers (`budget_colors`, `run_label`, `plot_sfs_budget`)

All post-processing scripts accept `--filename`, `--filter-scales`, `--n-workers`, `--fixed-reference` via argparse; `03`, `05` and `sweep2` additionally accept `--reference {filtered,true}` (see *Filtered-reference scale decomposition*). Output goes to `postprocessing/output/`.

### Data flow between pipeline steps

The sorted density (`*_sorted_density.nc`) produced by step 02 is reused by steps 03 and 05, avoiding redundant sorts. When `--fixed-reference` is used, output files are suffixed `_fixed_ref`. **The sweep does not share it.** The column's z✶ are the padded grid's own heights, and the sweep pads to 4σ of ℓ=20 while 02 pads to the budget scales, so 02's column belongs to a different grid (at Nz=2048: 1024 cells per side against the sweep's 2784) and reading it would shift every displacement with nothing downstream to reveal it. `sweep2_energy_transfer.py --fixed-reference` therefore builds its own frozen column, by sorting t=0 on the grid it loaded and broadcasting that row over the time axis — one sort, not one per output, and bit-identical to 02's whenever the two paddings do coincide.

The cross-scale KE transfer **Π_K and the SFS KE dissipation ε_Kˢ are computed online** by the Julia simulation (`kelvin_helmholtz_instability.jl`, output as `Π_K_ℓ<ℓ>` / `ε_Ks_ℓ<ℓ>`). To avoid recomputing them offline, `03_energy_transfer.py` runs with `include_pi_k=False` (Π_A + exchange only) and `04_sfs_ke_budget.py` reads Π_K and ε_Kˢ directly from the simulation output (it still computes the SFS-KE density/tendency and the APE↔KE exchange offline — the exchange needs the sorted reference state). The **sub-filter APE dissipation ε_Aˢ is computed online** the same way (output as `ε_As_ℓ<ℓ>`, gated behind `--save_sorted`) and read back by `05_sfs_ape_budget.py` — but **only for the time-varying reference**. Π_K and ε_Kˢ are built from velocities alone and so are reference-independent, which is why `04` can always read them; ε_Aˢ is measured against a reference state, and the online one is always the sort of the *current* buoyancy. Under `--fixed-reference` every other APE term is measured against the t=0 profile, so reading the online ε_Aˢ there would put one term on a different reference state from the rest and the budget would not close (it did not: ~240% residual). That variant falls back to the offline `calculate_sfs_ape_dissipation`, which is also what `inv08` checks the online field against. **The cross-scale APE flux Π_A is online on exactly the same terms** (`Π_A_ℓ<ℓ>`, `--save_sorted`, `AvailablePotentialEnergyCrossScaleFlux`): `03_energy_transfer.py` reads it for the time-varying reference and recomputes it offline under `--fixed-reference`, since Υˡ is a reference-state quantity too. Reading it also skips the sort of the filtered density that Υˡ needs, which is the expensive half of that step. `inv09` checks it. That script still computes the local APE fields, the tendency and the reference-tendency correction Rˢ offline. `--save_sorted` is **on by default** in the submission scripts, precisely because these two reads are what close the budget: at Nz=128 the APE residual goes 18.1% → 2.0% (ℓ=1) and 15.5% → 1.2% (ℓ=7) against the dominant term. They remain optional in the code — pass `SAVE_SORTED=0`, or run the simulation without the flag, and `03`/`05` recompute offline and say so in the log rather than failing — but that is the looser-closure path, not the default one. When the online fields *are* read, the budget filter scales (`--filter-scales`) must be a subset of the simulation's online `filter_ℓs` (set by `--filter_ls`, default `(1, 7)`); the offline-recompute paths are kept under `validation/` for cross-checking (`inv02` for Π_K, `inv05` for ε_Kˢ).

### Filtered-reference scale decomposition (`--reference`)

> **Why it is this way:** `filtered_reference_decisions.md` is the decision record for this work — what was
> tried, what was measured, and what was rejected. Read it before changing how ⟨ρ_*⟩ is built, how the
> sub-filter flux and dissipation are computed, or how the integrals treat the z padding. Several of those
> choices look arbitrary and are not, and one of them (the coarse column parameterised by levels-per-σ) was
> a wrong turn that cost real time to diagnose and undo.

`03_energy_transfer.py`, `05_sfs_ape_budget.py` and `sweep2_energy_transfer.py` take `--reference {filtered,true}`, which selects the reference state the **resolved** reservoir is measured against. Default `filtered`.

The pipeline filters in x **and** z (`filtered_dimensions = ["x_caa", "z_aac"]`). For a kernel with vertical extent the two reservoirs cannot both be measured against the unfiltered ρ_*: a fluid at rest filters to a profile that still carries APE against ρ_*, so the resolved reservoir does not vanish at rest and the remainder Eₐˢ = filter(eₐ) − eₐ(ρ̄) goes negative. That is the whole content of `test_jensen.py`.

Wenegrat, Chor & Barkan resolve it by measuring the resolved reservoir against the rest state *as the filter sees it* — the vertically filtered profile ⟨ρ_*⟩ (their Eq. 2.3):

- `filtered` (default) — `filtered_reference_profile()` builds ⟨ρ_*⟩ by convolving the sorted column with the vertical marginal of the same Gaussian, on the column's own grid so the z_*(ρ̄) lookup keeps its full N = Nx·Ny·Nz height resolution. Then `L̃ = Ẽ_A(ρ̄, z)` against ⟨ρ_*⟩, `S̃ = Ē_A − L̃`, `Υ̃ = z̃_*(ρ̄) − z`, and `b̄_r = b̄ − ⟨b_*⟩(z)` — which against ⟨ρ_*⟩ is exactly filter(b_r), since ρ_*(z) depends on z alone. Both reservoirs are non-negative and both vanish at rest, for any kernel.
- `true` — the unfiltered ρ_* for both reservoirs. This is the `g_z = δ(z)` horizontal-filter limit (their Eq. 2.24–2.26), correct only for a horizontal kernel, and is what the pipeline did before. Kept so the published results stay reproducible and so the two can be compared.

Three consequences worth knowing:

- **The online APE terms are the filtered-reference ones, and the only ones.** `Π_A_ℓ<ℓ>` / `ε_As_ℓ<ℓ>` are built against ⟨b✶⟩; the simulation no longer writes an unfiltered counterpart, so `--reference true` always recomputes offline and logs that it is doing so. **Reading them is what makes the budget close**: the offline `calculate_ape_dissipation` takes centred differences of a quantity quadratic in gradients, and measured against the online field that discretisation error accounts for essentially the whole offline residual (corr +0.995, ratio ~1 pointwise in time; offline/online ε_Aˢ drifts 0.97 → 0.78 as the interface reaches the grid scale). At Nz=128 the residual/min(terms) goes 80.1% → 8.8% (ℓ=1) and 274% → 21.3% (ℓ=7), or 18.1% → 2.0% and 15.5% → 1.2% against the dominant term. `--fixed-reference` still recomputes offline, since the online fields track the sort of the *current* buoyancy; a missing field falls back with the reason logged.
- **`⟨ρ_*⟩` is scale-dependent**, so it is rebuilt inside each filter-scale loop rather than once per run.
- **Cost is essentially unchanged.** The construction is the same lookup and the same cumulative integral against a different profile; the only new work is one 1D convolution of the profile per timestep. The expensive full-field sort in `02` is untouched and still shared. A *frozen* profile is filtered once and broadcast rather than once per output, since `sorted_timeseries(fixed_reference=True)` repeats one row — bit-identical, and 12–45× faster at Nz=128.

- **The z padding is sized to the widest filter.** `S̃ = Ē_A − L̃` is an identity only where `filter(z) = z`, which fails within one stencil of an array boundary. `required_pad_margin` asks for 4σ of margin; `01`/`sweep1` pad with it and record it on the filtered-fields file, and `02`–`05`/`sweep2` read it back so every step sees the same padded grid. The old fixed `Nz//2` gives a margin of Lz/2 = 12.5, which is just enough for ℓ = 7 (4σ = 11.89) and fails from ℓ ≈ 7.7 — the next scale in the sweep's `geomspace(0.02, 20, 30)`, whose top end needs 33.97. A run at ℓ ≤ 7 is unchanged, bit-identical. **This fixes only the stencil-truncation half of the large-ℓ problem:** a filter whose σ is a third of the domain also flattens ⟨b_*⟩ until its inverse z̃_* is poorly conditioned, and padding cannot fix that. The sweep's column filter is also O(N²) in the column length, ~9e12 operations at Nz=512 over its 30 scales, which wider padding makes worse; building ⟨ρ_*⟩ on the model grid instead would cut that by ~10⁵ at the price of a coarser z̃_* lookup.

`test_filtered_reference.py` pins this down on the synthetic field of `test_jensen.py`: S̃ and L̃ are non-negative, L̃ + S̃ reassembles Ē_A to machine precision, and the remainder construction still breaks on the same field. Measured interior min/rms for S̃ is +8.6e-06, +6.9e-05 and +2.0e-03 at ℓ = 2, 4 and 8 cells, against −0.69, −2.31 and −2.67 for the remainder, which is negative over 15–48% of the domain.

**Edge caveat.** S̃ is computed as Ē_A − L̃, an identity that needs filter(z) = z. `GaussianFilter` extends the bounded z axis with its edge value, so within a stencil of a wall the kernel is one-sided and the identity picks up a spurious τ(z, b). Every violation measured on the synthetic field lives in that band and nowhere else, which `test_filtered_reference.py::test_edge_cells_are_the_only_place_the_identity_fails` asserts. The production pipeline does not meet this, because `_pad_domain_in_z` doubles the domain height with edge values at load time, putting the physical domain far from the filter's edge.

**Now computed online, and online computes nothing else.** Under `--save_sorted` the simulation writes the filtered-reference set per scale — `L_ℓ<ℓ>` (L̃), `E_as_ℓ<ℓ>` (S̃), `Υ_ℓ<ℓ>`, `Π_A_ℓ<ℓ>`, `ε_As_ℓ<ℓ>`, `R_s_ℓ<ℓ>` and the S̃ tendency `dEas_dt_ℓ<ℓ>` — under the plain names, the unfiltered set having been removed entirely so that nothing downstream can read the superseded formulation by accident. (`wb_rs_ℓ<ℓ>`, the APE→KE conversion, is reference-independent and unchanged.) It needed no Oceanostics change: `ProfileLookup` accepts an external profile as a pair of `Field`s and refreshes it on every `compute!` (`refresh_profile_source!(p::Field, time) = compute_at!(p, time)`), supplying one skips the O(N log N) sort, and the two-argument constructors `AvailablePotentialEnergy(model, z✶)` / `FilteredAvailablePotentialEnergy(model, z✶ˡ)` / `AvailablePotentialEnergyCrossScaleFlux(model, filter, z✶ˡ)` let the two halves carry different profiles, composed in the simulation script exactly as `R_s = Field(gf(R_full)) - R_l` already is. Three things that cost real time to learn and are easy to undo by accident:

- **Composed expressions must be wrapped in a `KernelFunctionOperation`** with a trivial passthrough kernel, as every Oceanostics diagnostic already does (`subfilter_ape_ccc` and friends). A writer specialises on the type of its whole output NamedTuple, and an unwrapped `Field(gf(Field(a))) - b` is a deep `BinaryOperation` nest; several of those sent LLVM's instruction selector quadratic (`DAGCombiner::CombineToPostIndexedLoadStore` → `hasPredecessorHelper`) and stalled compilation of the writer for **>45 min at Nz=32**. `flatten` in the simulation script collapses each term to one flat type. Symptom: the run sits at 100% CPU writing nothing, because Julia buffers stdout and stderr when redirected — an explicit `flush` is the only way to see where it is.
- **Keep the 2D panels writer's tuple small and keep the `flatten` wrappers on it.** It carries the per-scale set so `plot_kelvin_helmholtz_instability.jl` draws the panels animation from the filtered-reference fields, which is the whole point, but it is the largest output tuple and therefore where the LLVM stall above bites first.
- **Measured at Nz=128/Re=262**, the online construction beats the offline one on every count: at rest L̃ is 1.09e-07 (ℓ=1) and 8.04e-08 (ℓ=7) against 1.56e-07 and 1.13e-07 offline, with the true-reference `E_a` at machine zero; S̃ min/rms is −9.4e-03 and −1.2e-06 against −2.7e-02 and −1.6e-06, and at ℓ=7 the interior is strictly positive. The online path has no z padding, so the filter's edge band sits in real fluid — measured, that costs nothing here (edge band machine-clean at ℓ=1, worst violation in the interior), because the domain's z walls sit deep in the saturated tanh where b is uniform and E_A ≈ 0.

**Open question:** which grid ⟨b_*⟩ is built on. Offline it is the sorted column, whose slots are uniform only because the model grid is; Oceanostics' filter is grid-general, so building ⟨b_*⟩ on the column with a single rescaled σ would narrow that. The alternative — filter `ReferenceProfileAtHeight` on the model grid, then interpolate onto the column heights for the lookup — keeps both the grid-generality and the height resolution, and needs testing. Note also that Oceanostics' own docstring (`FilteredAvailablePotentialEnergyEquation.jl`, on `FilteredAvailablePotentialEnergy`) still asserts the filtered buoyancy must be measured against the same profile as the full one, which is the assumption this construction overturns.

### Julia layer
- `kelvin_helmholtz_instability.jl` -- main simulation (Oceananigans.jl `NonhydrostaticModel`, `Centered(order=4)` advection, adaptive timestep via `TimeStepWizard`)
- `utils.jl` -- `closest_factor_number()` (FFT-friendly grid sizes), `show_gpu_status()`
- `online_diagnostics.jl` -- the only budget term Oceanostics does not provide: `ReferenceTendencyCorrection` (R = ∫_{z✶}^{z} ∂ₜb✶ dz̃). The sub-filter APE→KE conversion τ(w, b_r) used to live here too; it moved upstream in Oceanostics PR #301 as `SubFilterAvailablePotentialToKineticEnergyConversion`, which builds both halves of the split from one kernel and one `b✶(z)` rather than differencing two separately-built diagnostics
- Setup: shear flow u(z)=U·tanh(z/h), stratification b(z)=B₀·tanh(z/h) with B₀=U²·Ri/h; perturbation seeded on w. Defaults U=1, Ri=0.1, h=1
- Domain: Lx=λ_max (the most-unstable KH wavelength ≈14.1h), Ly=λ_max/3, Lz=25h; topology (Periodic, Periodic, Bounded). `y_aspect_ratio=Inf` ⇒ **Ny=1**, so runs are effectively 2D in x–z (v ≡ 0)
- Output: `output/khi_Nz<Nz>_Ri<Ri>.nc` (3D fields, Float64, consecutive-iteration pairs for time derivatives) and `output/khi_Nz<Nz>_Ri<Ri>_2d.nc` (x–z slice, Float32)

#### Online cross-scale diagnostics
The simulation computes, at each scale in `filter_ℓs` (set by `--filter_ls`, default `(1, 7)`), the sub-filter quantities the offline pipeline would otherwise recompute in Python — so they are produced once, on the GPU, and read back later (see Data flow):
- Filtered fields (Oceanostics `GaussianFilter`), the cross-scale KE flux Πₖ = −τⁱʲ S̄ⁱʲ (`KineticEnergyCrossScaleFlux`), and the SFS KE dissipation ε_Kˢ = filter(ε) − ε̄ (where `ε` is the total viscous dissipation `KineticEnergyEquation.DissipationRate` and `ε̄` is `CoarseGrainedKineticEnergyDissipationRate`, the filtered-flow dissipation). For validation only, the resolved strain rate S̄ⁱʲ (`StrainRateTensor`) and sub-filter stress τⁱʲ = filter(uⁱuʲ) − ūⁱūʲ (`subfilter_stress_tensor`) components are also emitted.
- The Gaussian filter is configured to **match the offline filter exactly**: periodic x, edge-extended bounded z, stencil truncated at 4σ (matching scipy `gaussian_filter1d`'s default `truncate=4`; Oceanostics defaults to 2σ). Only i,j ∈ {1,3} are kept (2D x–z).
- `Π_K_ℓ<ℓ>` and `ε_Ks_ℓ<ℓ>` (and their volume integrals) are always written and read back by `04_sfs_ke_budget.py`; the individual S̄/τ components (`S11/S33/S13_ℓ<ℓ>`, `tau11/tau33/tau13_ℓ<ℓ>`) are gated behind `--save_tensors` and consumed only by `postprocessing/validation/`.
- Every online diagnostic used here comes from Oceanostics, with no bespoke diagnostic code left in this repo. `AvailablePotentialEnergyCrossScaleFlux` (Π_A) landed in Oceanostics PR #295 and `SubFilterAvailablePotentialToKineticEnergyConversion` (τ(w, b_r)) in PR #301, so `Manifest.toml` tracks the `tc/subfilter-ape-ke-conversion` branch until the next release. Oceanostics provides `GaussianFilter`, `StrainRateTensor`, `subfilter_stress_tensor`, `KineticEnergyCrossScaleFlux`, `SubFilterKineticEnergyDissipationRate`, the whole `BackgroundPotentialEnergyEquation`/`AvailablePotentialEnergyEquation` reference-state machinery, and `AvailablePotentialEnergyDisplacementPotential`/`AvailablePotentialEnergyDissipationRate` (0.20.0 renamed the first from `BuoyancyDisplacementPotential`). Note 0.19.0 renamed `KineticEnergyBuoyancyProduction` → `PotentialToKineticEnergyConversion`; this repo does not use it.

#### Online sorted reference state (`--save_sorted`)
Oceanostics' `AvailablePotentialEnergyEquation` module sorts the buoyancy field adiabatically into its minimum-PE state, giving the Winters et al. (1995) reference height z✶. `--save_sorted` emits it under all three sorting methods, which describe the same reference state but differ in where they place cells of *equal* buoyancy and on what grid they answer:
- `z✶_3dsort` (`ThreeDimensionalSort`) and `z✶_heaviside` (`HeavisideIntegral`) — 3D fields on the model grid. Tied cells take consecutive slots in the first and share their layer's mid-height (Winters eq. 11) in the second.
- `z✶_1dsort`, `b✶_1dsort` (`VerticalSort`) — the sorted column, on its own N = Nx·Ny·Nz vertical axis (one slab per model cell). All of these go in the **main** output file: one `NetCDFWriter` holds both grids (as the upstream `lock_release.jl` example does). The cost is that a writer holding two grids makes Oceananigans suffix every dimension (`x_caa` → `x_caa_grid1` for the model grid, `_grid2` for the column) and rename the metadata groups (`underlying_grid_reconstruction_kwargs` → `grid_1_...`). The offline pipeline reads the plain names, so `load_dataset_and_grid` strips the model grid's suffix at load time (`model_grid_suffix` / `strip_grid_suffix` / `open_grid_group` in `aux00_utils.py`), leaving 01–06, the sweep, and the `inv0*` scripts unaffected; the column keeps its own suffix and is read by `inv06`/`inv07`. The 2D slice writer gets the model-grid `outputs` plus, under `--save_sorted`, the sub-filter APE fields and `b_r` (all model-grid), so it stays single-grid; those slices are what let `plot_kelvin_helmholtz_instability.jl` draw the SFS-budget panels animation (`animations/khi_Nz<Nz>_Ri<Ri>_panels_l<ℓ>.mp4`, the online counterpart of `anim1_panels.py`) straight from the `_2d.nc` file, with no offline pipeline.
- Only the column is a reference *profile* as written. For the two model-grid methods `reference_buoyancy` is the model's own `b` (already an output), so their profiles come from ordering the (z✶, b) pairs by z✶ — see `profile_from_model_grid` in `inv06`, and the `lock_release.jl` example upstream.
- `b✶` (from `reference_buoyancy`) recomputes itself: its `compute!` delegates to the parent `z✶`, so handing it to an output writer triggers the sort, and the status gate means listing both does not sort twice. Earlier revisions of the branch returned a bare `Field` that `compute!` ignored, which silently wrote the *previous* output's profile; fixed upstream via `SortedBuoyancyState`. `inv06` asserts `b✶` is a permutation of `b`, which guards against a regression.
- `--save_sorted` also emits the **online local available potential energy**: `E_a` (the local field, `AvailablePotentialEnergy(model, z✶_3dsort)`) and its integral `∫E_a`, plus `∫E_b`. `AvailablePotentialEnergy` now computes the Holliday & McIntyre (1981) integral density Eₐ = ∫_{z✶}^{z}[b✶(z̃) − b] dz̃, identical to the offline `ape` field in `local_potential_energies_timeseries` (with b = g(ρ₀−ρ)/ρ₀); `inv07_compare_local_ape.py` checks the two. It shares the `ThreeDimensionalSort` z✶ rather than re-sorting.
- The API was renamed upstream: `sorted_reference_height` → `reference_height`, `sorted_buoyancy` → `reference_buoyancy`, `OneDimensionalSort` → `VerticalSort`, `AbstractSortingMethod` → `AbstractReferenceHeightMethod`.
- This is the online counterpart of the offline sort in `02_sort_density.py`. The offline pipeline sorts the *z-padded* domain (`load_dataset_and_grid` doubles the height with edge values at load time) while the online sort sees only the true domain, so the two do not sort the same field; `inv06_compare_sorted_profiles.py` quantifies that, running the offline sort both padded and unpadded to separate the padding effect from any genuine online-vs-offline difference.

#### Online sub-filter APE dissipation ε_Aˢ (also gated behind `--save_sorted`)
`SubFilterAvailablePotentialEnergyDissipationRate` (Oceanostics PR #293, branch `tc/subfilter-ape`) computes, per online filter scale, the diffusive sink of the sub-filter APE budget:
- **ε_Aˢ = filter(ε_A) − ε_Aˡ**, with **ε_A = κ ∂ᵢb ∂ᵢΥ** the total APE dissipation, **Υ = z✶ − z** the buoyancy displacement potential, **ε_Aˡ = −q̄ᵢ ∂ᵢΥˡ**, **q̄ᵢ = filter(κ ∂ᵢb)** and **Υˡ = z✶(b̄) − z**. Output as `ε_As_ℓ<ℓ>` and `ε_As_ℓ<ℓ>_int`, matching the `Π_K_ℓ<ℓ>` / `ε_Ks_ℓ<ℓ>` naming. `05_sfs_ape_budget.py` reads them back instead of calling `calculate_sfs_ape_dissipation`, exactly as `04` does for the KE pair.
- **Only ε_Aˢ is written.** Υ and the total ε_A are Oceanostics' `BuoyancyDisplacementPotential` and `AvailablePotentialEnergyDissipationRate` (both developed here and merged upstream in PR #276); the sub-filter diagnostic builds them internally, but neither is a budget term, so neither is emitted.
- **What ε_A is.** It is the sink of the local APE equation (the paper's Eqs. 11 and 14, where it enters as −ε_A). With `∂Eₐ/∂b = z✶ − z = Υ`, the diffusive part of `DEₐ/Dt` is `Υκ∇²b = ∇·(κΥ∇b) − κ∇Υ·∇b`. Written out, `ε_A = κ[(∂z✶/∂b)|∇b|² − ∂b/∂z]`: the Winters diapycnal mixing rate less the diffusion the reference state undergoes on its own. The two cancel exactly for a statically stable, horizontally uniform stratification (z✶ = z), so ε_A is **not** sign-definite pointwise the way a `κ|∇b|²` would be. `κ ∂ᵢb` comes from the closure's own `diffusive_flux_*` rather than a diffusivity passed in, which is why the buoyancy has to be a tracer the closure diffuses (`BuoyancyTracer` only, enforced with an error).
- **Units convention.** The offline pipeline is written in density: `Υ = g(z − z✶(ρ))/ρ₀`, Eq. (7) of Wenegrat, Chor & Barkan (2026). A Boussinesq `BuoyancyTracer` model has no g or ρ₀, so the online Υ is the buoyancy form `z✶ − z`, related by `Υ_density = −(g/ρ₀) Υ_buoyancy`. Since `b = g(ρ₀−ρ)/ρ₀` makes `∇ρ = −(ρ₀/g)∇b`, both sign flips cancel in the contraction, so **ε_A — and hence ε_Aˢ — is the same number in either form** and `inv08` compares them directly.
- **Only the time-varying reference reads it.** The online ε_Aˢ is tied to the simulation's own sort of the current buoyancy, so `05_sfs_ape_budget.py` reads it only when the reference profile is recomputed each step, and recomputes ε_Aˢ offline under `--fixed-reference`. See the Data flow section.
- **One shared reference profile.** Both states are measured against the same profile, which is what makes `filter(ε_A) − ε_Aˡ` a decomposition rather than a difference of two unrelated quantities, so `method` has to be a `ProfileLookup`. The simulation hands it the `VerticalSort` column `--save_sorted` already builds, so every filter scale shares one sort. That lookup is also exactly the offline semantics: `local_potential_energies_timeseries` gets z₀(ρ̄) by a nearest-density search of the *filtered* density into the *full* field's sorted profile.
- **The second term differs from the offline one by construction.** Online ε_Aˡ contracts the *filtered flux* `filter(κ∂ᵢb)` with ∇Υˡ; offline term 2 rebuilds the flux from the filtered density as `κ∇ρ̄`. Filtering the flux rather than recomputing it is what makes ε_Aˡ the sink of the filtered-state budget when κ varies in space. For the constant κ used here the two agree in the interior (filtering and differencing are both convolutions on a uniform grid, so they commute) and part ways only against the walls, where the filter's edge extension and the derivative's one-sided stencil disagree.
- `inv08_compare_sfs_ape_dissipation.py` measures what is left of that, plus a conservative-vs-centered discretization gap: the online form pairs its two factors on the face where both differences live and interpolates the *product* to the center, while the offline `calculate_gradient` takes centered derivatives at the center and multiplies *those*. Being a difference of two comparable quantities, ε_Aˢ could have amplified that the way ε_Kˢ does; measured at Nz=192/Re=262 it does not, landing at 7.8e-02 (field) and 1.4e-01 (∫ε_Aˢ dV) for ℓ=1, and 4.0e-02 / 1.1e-01 for ℓ=7.
- The offline expression stays in `calculate_sfs_ape_dissipation` (`aux01_pe_functions.py`) as the reference `inv08` compares against, in the same way the offline Π_K and ε_Kˢ paths live on under `validation/`.

#### Fully online SFS budgets
Every term of both sub-filter budgets is now written by the simulation, so closure can be checked without the offline pipeline at all (`inv10_online_budget_closure.py`, which uses the same metric `test_budgets.py` applies offline):
- **KE**: `residual_K = −∂ₜKˢ + Π_K − ε_Kˢ + τ(w, b_r)`. **APE**: `residual_A = −∂ₜEₐˢ − τ(w, b_r) + Π_A − ε_Aˢ + Rˢ`. The transport `∂ᵢFᵢˢ` is not computed in either, as offline: it integrates to zero over the closed domain.
- **Tendencies** are written both as full fields (`dKs_dt_ℓ<ℓ>`, `dEas_dt_ℓ<ℓ>`) and as tendencies of the integrals (`..._int`, the budget-closure form). They come from Oceananigans' `TimeDerivative`, which is deferred: the writer evaluates it when a record opens (opening the differencing window) and once more on the following iteration, back-filling the record with the forward difference over that single timestep, (aⁿ⁺¹ − aⁿ)/Δt labelled tⁿ. No callback is involved; a `TimeDerivative` advances whenever it is evaluated, idempotently per time. R rides the same machinery: computing `ReferenceTendencyCorrection` advances its internal ∂ₜb✶, and `deferred_output(::ReferenceTendencyState) = true` makes every output built on R deferred too, so the sorted column is only re-sorted at writer actuations rather than every timestep. The last record of a run holds NaN for every deferred output (its window never completes).
- **R is built from the reference profile's own time derivative**, not from `∂ₜeₐ − Υ∂ₜb`. The latter is cheaper but makes `Rˢ` contain `∂ₜEₐˢ` identically, so the APE budget would close in its tendency by construction rather than as a test. `ReferenceTendencyCorrection` differentiates only `b✶`, cumulatively integrates it over the sorted column (whose slots are equal-volume, hence uniformly spaced, so `Ψ̇` is piecewise linear on a uniform grid) and evaluates between `z✶` and `z`.
- **The first output pair is not a budget statement.** Its derivatives span the initialisation transient where `∫Eₐˢ` climbs from zero to its working value. `inv10` drops both records (`--skip`, default 2) and the trailing NaN record.
- **Eₐˢ is not sign-definite.** `filter(eₐ) − eₐˡ ≥ 0` follows from Jensen only when the filter acts on `b` alone; this filter also acts in z, where `eₐ(·, z)` changes with height, so the bound does not carry. Negative values are real, not a bug.
- Measured at Nz=192/Re=262 against the offline pipeline on the same run: KE ℓ=1 online 8.8% vs offline 9.6%, KE ℓ=7 0.53% vs 1.1%, APE ℓ=1 5.5% vs 4.4%, APE ℓ=7 1.9% vs 2.4%. All four are under the 10% threshold; three of four beat the offline residual, and the APE ℓ=1 gap sits in the late, strongly mixing phase (t ≳ 64) where the interface is grid-scale sharp.

### Key dependencies
- **Python**: `numpy`, `xarray`, `scipy`, `matplotlib`, `dask`, `gcm_filters`, `netcdf4`
- **Julia**: `Oceananigans`, `Oceanostics`, `CUDA`, `NCDatasets`, `CairoMakie`

## Physics Reference

- **TPE** = integral of g*rho*z dV  (total potential energy)
- **RPE** = minimum PE achievable by adiabatic rearrangement (from sorted reference state)
- **APE** = TPE - RPE  (available for conversion to KE)
- **Π_K**, **Π_A** -- cross-scale energy transfer (sub-filter to resolved)
- Physical constants: `g=9.81`, `rho_0=1025`

## Code Style

- Do not break a command/statement into multiple lines if it fits within 140 columns.
- Always delimit code sections with `#+++` on the opening line and `#---` on the closing line:
  ```python
  #+++ Section name
  ...code...
  #---
  ```

## Maintenance Rules

- **Always update `README.md` when the job submission scheme changes.** This includes: adding/removing/renaming PBS scripts or wrapper scripts, changing argument names or defaults, adding new pipeline stages, or changing job dependency chains.

## Notes

- Output files are excluded from git (`.nc`, `.mp4`, `.pdf`, `.png`, `.jld2`).
- Simulation output can reach 650 GB; scratch directory is `/glade/derecho/scratch/tomasc/khape/output/`.
- Logs: `logs/<job_name>.log` (PBS), `logs/<job_name>.out` (Python stdout via tee). Job names follow `<stage>_Nz<NZ>_Ri0.10[_fixed_ref]`.
