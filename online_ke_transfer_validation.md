# Computing the cross-scale KE transfer online — and checking it against the offline pipeline

KHAPE already computes the cross-scale kinetic-energy transfer **Π_K** *offline*, as a post-processing
step on the saved Kelvin–Helmholtz fields (`postprocessing/03_energy_transfer.py`). That works, but it
means re-filtering the full 3D velocity at every saved snapshot after the run, and it ties the diagnostic
to whatever cadence the fields happen to be written at.

This post is about computing the **same** quantity *online* — inside the Oceananigans simulation, as a
regular output — and verifying that the online result reproduces the offline one.

## The quantity

Π_K is the cross-scale (sub-filter → resolved) kinetic-energy flux of
[Aluie et al. (2018, JPO)](https://doi.org/10.1175/JPO-D-17-0100.1), Eq. (7):

```
Π_K = -τⁱʲ S̄ⁱʲ ,   τⁱʲ = filter(uⁱuʲ) - ūⁱ ūʲ ,   S̄ⁱʲ = ½(∂ūⁱ/∂xʲ + ∂ūʲ/∂xⁱ)
```

where the overbar is a Gaussian filter of width ℓ (FWHM). `τⁱʲ` is the sub-filter stress tensor and
`S̄ⁱʲ` the strain rate of the *filtered* velocity. `Π_K > 0` is forward (downscale) transfer. The KH runs
are 2D in x–z (`v ≡ 0`), so only the `i,j ∈ {1,3}` components survive, and we omit `ρ₀` (Π_K is per unit
mass, units m² s⁻³).

## The online implementation

Two of the three ingredients already exist in Oceanostics
([PR #244](https://github.com/tomchor/Oceanostics.jl/pull/244)):

- **`GaussianFilter`** — a separable Gaussian filter as a `KernelFunctionOperation`, applied in x
  (periodic) and z (bounded), exactly as the offline filter does.
- **`StrainRateTensor`** — the strain-rate tensor `S̄ⁱʲ` of a velocity field, returned as staggered KFOs.

The missing piece is the **sub-filter stress tensor**, which we build from filtered velocity products
(following the idiom in Oceanostics' own `spatial_filtering` example) as its own function:

```julia
# τⁱʲ = filter(uⁱuʲ) - ūⁱ ūʲ at cell centers, returned as a NamedTuple — its own calculation
function sfs_stress_tensor(filt, u, w, ū, w̄)
    τ₁₁ = Field(filt(Field(uᶜ * uᶜ))) - ūᶜ * ūᶜ
    τ₃₃ = Field(filt(Field(wᶜ * wᶜ))) - w̄ᶜ * w̄ᶜ
    τ₁₃ = Field(filt(Field(uᶜ * wᶜ))) - ūᶜ * w̄ᶜ
    return (; τ₁₁, τ₃₃, τ₁₃)
end
```

The cross-scale flux then computes the strain-rate tensor and the stress tensor **separately**, and only
contracts them at the end:

```julia
ū, v̄, w̄ = Field(filt(u)), Field(filt(v)), Field(filt(w))   # filtered velocities, shared by both tensors

S̄ = StrainRateTensor(grid, ū, v̄, w̄)        # resolved-scale strain rate  S̄ⁱʲ
τ = sfs_stress_tensor(filt, u, w, ū, w̄)     # sub-filter stress           τⁱʲ

Π_K = -(τ.τ₁₁ * S̄.S₁₁ + τ.τ₃₃ * S̄.S₃₃ + 2 * τ.τ₁₃ * S̄₁₃ᶜ)   # contraction at cell centers
```

`τⁱʲ` is a lazy Oceananigans operation, so it composes with `StrainRateTensor` and is recomputed by the
output writer at every snapshot — no callbacks, no precomputation. We output both the field
`Π_K_ℓ{1,7}` and its volume integral `Π_K_ℓ{1,7}_int = ∫Π_K dV`. The full functions `sfs_stress_tensor`
and `cross_scale_ke_flux` are in `kelvin_helmholtz_instability.jl`.

## The check

`postprocessing/inv02_compare_ke_transfer.py` recomputes Π_K *offline* from the saved velocity using the
existing `aux02` machinery (the same code path as `03_energy_transfer.py`) and compares it, point by point
and as a volume integral, against the online output written by the simulation.

The test run is a small CPU case — `Nz = 128`, `Re ≈ 1638`, `Ri = 0.1`, run to `t = 60` so the billow
fully rolls up — filtered at `ℓ = 1` and `ℓ = 7`.

## Results

### Snapshot fields at t = 30

![Online vs offline Π_K maps, Nz=128](figures/khi_Nz128_Ri0.10_ke_transfer_comparison_maps_t30.0.png)

*Online (left) and offline (middle) Π_K, with their difference (right), for ℓ=1 (top) and ℓ=7 (bottom).
Note the difference colorbars are ~100× tighter than the field colorbars.*

The online and offline fields are visually indistinguishable. At **ℓ=7** the difference is ~1% of the
field (colorbar `3e-5` vs field `3e-3`) — a smooth large-scale forward-transfer lobe along the overturning
billow. At **ℓ=1** the difference lives on the sharp billow edges, where the discrete strain operators
differ most; this scale is barely resolved at `Nz=128` (the Gaussian is only ~1 cell wide in x), so it is
the harder case.

### Volume-integrated transfer ∫Π_K dV

![Online vs offline integral, Nz=128](figures/khi_Nz128_Ri0.10_ke_transfer_comparison_integral.png)

*∫Π_K dV vs time: offline (black), the online field integrated in post (red dashed), and the online
`Integral` diagnostic written by the simulation (orange dotted).*

For **ℓ=7** the three curves lie exactly on top of each other through the whole evolution: transfer builds
to a forward (downscale) peak ≈ 0.67 as the billow rolls up near `t ≈ 37`, then decays and turns weakly
negative (backscatter) as the billow breaks down. The two online curves — the field integrated offline and
the online `Integral` output — agree to **machine precision** (2e-15), confirming the online integral
diagnostic.

## The one subtlety: filter truncation

The first attempt was off by ~18%. The culprit was *not* the physics but the filter kernel: Oceanostics'
`GaussianFilter` truncates the stencil at **2σ** by default, while the offline `scipy.ndimage.gaussian_filter1d`
keeps it out to **4σ**. The Gaussian weight at 2σ is still `exp(-2) ≈ 0.135`, so the two kernels are
genuinely different filters. Matching the online stencil radius to scipy's (via the `N` keyword) makes the
two kernels *identical* — same `exp(-Δi²/2σ²)` weights, same normalization — and the disagreement collapses
to the level set by the strain-operator discretization alone.

## Convergence

With the kernels matched, the only remaining difference is discretization: `StrainRateTensor` uses compact
staggered derivatives, while the offline strain uses a wider centered (numpy `gradient`) stencil. This is a
genuine `O(Δ²)`-type difference that vanishes with resolution, as a `Nz = 128 → 256` refinement confirms:

| metric | ℓ=1 (128 → 256) | ℓ=7 (128 → 256) |
|---|---|---|
| field rms error | 39% → 11% | 3.3% → **1.1%** |
| ∫Π_K dV error | 5.6% → 1.2% | 0.1% → **0.04%** |

The error drops ~3–4× per doubling — clean convergence, not a bug. At the production resolution
(`Nz ≥ 2048`) both scales will agree far more tightly, and even `ℓ=1` becomes well resolved.

## Takeaway

Π_K can be computed online during the simulation using `GaussianFilter` + `StrainRateTensor` plus a
hand-built sub-filter stress tensor, and it reproduces the offline post-processing result up to a
discretization difference that converges away with resolution. The single thing to watch is that the online
Gaussian filter is configured to the same truncation (4σ) as the offline one.

### Reproduce

```bash
julia --project -t 8 kelvin_helmholtz_instability.jl --Nz 128 --Re0 0.1 --stop_time 60
cd postprocessing && python inv02_compare_ke_transfer.py --filename output/khi_Nz128_Ri0.10.nc
```
