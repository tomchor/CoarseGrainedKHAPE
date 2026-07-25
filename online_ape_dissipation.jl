# Online (simulation-time) buoyancy displacement potential Υ and available potential energy
# dissipation rate ε_A, for the coarse-grained local APE framework of Wenegrat, Chor & Barkan (2026).
#
# Both are built on the Winters et al. (1995) reference height z✶ that Oceanostics' `reference_height`
# produces, and both follow the same interface its `BackgroundPotentialEnergy` and
# `AvailablePotentialEnergy` use: the keyword form builds its own z✶ from a `method`, and the
# two-argument form takes a z✶ you built yourself so a set of diagnostics can share one sort.
#
#     Υ   = z✶ - z                     [m]         `BuoyancyDisplacementPotential`
#     ε_A = κ ∂ᵢb ∂ᵢΥ                  [m² s⁻³]    `AvailablePotentialEnergyDissipationRate`
#
# Υ is the buoyancy form of the paper's Eq. (7), which is written for density,
#
#     Υ(ρ, z) = g(z - z✶(ρ))/ρ₀ ,
#
# and the two are related by Υ = -(g/ρ₀) Υ_here, since b = g(ρ₀ - ρ)/ρ₀ makes ∇ρ = -(ρ₀/g)∇b. The two
# sign flips cancel in the contraction, so ε_A = κ ∂ᵢρ ∂ᵢΥ = κ ∂ᵢb ∂ᵢΥ_here is the *same* number in
# either form — only Υ itself carries the ρ₀/g rescaling. A Boussinesq model with a `BuoyancyTracer`
# has no g or ρ₀ to rescale by, so the buoyancy form is the one that can be computed here; the
# offline pipeline works in density and writes the paper's form, and
# `postprocessing/validation/inv08_compare_ape_dissipation.py` converts between them.
#
# ε_A is the sink in the local APE equation (the paper's Eqs. 11 and 14, where it enters as -ε_A):
# with Eₐ = ∫_{z✶}^{z}[b✶(z̃) - b] dz̃ and ∂Eₐ/∂b = z✶ - z = Υ, the diffusive part of DEₐ/Dt is
# Υκ∇²b = ∇·(κΥ∇b) - κ∇Υ·∇b, so ε_A = κ∇Υ·∇b is what is left once the flux divergence is set aside.
# Writing it out, ε_A = κ[(∂z✶/∂b)|∇b|² - ∂b/∂z]: the first part is the diapycnal mixing rate and the
# second is the diffusion the reference state undergoes on its own. It vanishes identically for a
# statically stable, horizontally uniform stratification, where z✶ = z and there is no APE to destroy.
#
# Only the *total* dissipation is computed here. Its sub-filter counterpart ε_Aˢ = filter(κ∂ᵢρ∂ᵢΥ) -
# κ∂ᵢρ̄∂ᵢΥˡ needs a second reference state sorted from the filtered buoyancy, and is still computed
# offline by `postprocessing/05_sfs_ape_budget.py`.

using Oceananigans: fields
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.BuoyancyFormulations: Zᶜᶜᶜ
using Oceananigans.Fields: Field
using Oceananigans.Grids: Center
using Oceananigans.Models: model_geopotential_height
using Oceananigans.Operators: Axᶠᶜᶜ, Ayᶜᶠᶜ, Azᶜᶜᶠ, Vᶜᶜᶜ, δxᶠᵃᵃ, δyᵃᶠᵃ, δzᵃᵃᶠ, ℑxᶜᵃᵃ, ℑyᵃᶜᵃ, ℑzᵃᵃᶜ
using Oceananigans.TurbulenceClosures: diffusive_flux_x, diffusive_flux_y, diffusive_flux_z
using Oceanostics
using Oceanostics: CustomKFO, validate_location
using Oceanostics.BackgroundPotentialEnergyEquation: HeavisideIntegral, SortedReferenceHeightField, reference_height
using Oceanostics.PotentialEnergyEquation: BuoyancyTracerModel

import Oceananigans.AbstractOperations: operation_name   # so Oceanostics' `@diagnostic_show` extends it here

#+++ Shared validation
# Both diagnostics read the parcel's own height off the grid `z✶` lives on, so both need that to be the
# model grid. `VerticalSort` answers on the sorted column instead, where the grid's own `Zᶜᶜᶜ` *is* z✶
# (making Υ silently zero) and a horizontal gradient of `b` means nothing.
validate_reference_height_grid(diagnostic, model, z✶) =
    z✶.grid === model.grid ||
        throw(ArgumentError("`$diagnostic` needs a reference height on the model grid, but this one lives on a \
                             $(summary(z✶.grid)). Use `HeavisideIntegral()`, `ThreeDimensionalSort()` or \
                             `ProfileLookup()` rather than `VerticalSort()`."))
#---

#+++ Buoyancy displacement potential Υ
@inline upsilon_ccc(i, j, k, grid, z✶) = @inbounds z✶[i, j, k] - Zᶜᶜᶜ(i, j, k, grid)

const BuoyancyDisplacementPotential = CustomKFO{<:typeof(upsilon_ccc)}

"""
    BuoyancyDisplacementPotential(model; method = HeavisideIntegral(), geopotential_height, location)
    BuoyancyDisplacementPotential(model, z✶)

Return a `KernelFunctionOperation` computing the buoyancy displacement potential

```
    Υ = z✶ - z
```

how far below its actual height a parcel's reference height sits, and so how far it would have to
travel to reach the adiabatically resorted state. It is the derivative of the local available
potential energy with respect to buoyancy, `Υ = ∂Eₐ/∂b`, which is what makes it the natural
conjugate of `b`: contracting it with a buoyancy gradient (`AvailablePotentialEnergyDissipationRate`)
or with a sub-filter buoyancy flux (the cross-scale APE flux `Π_A`) gives an APE transfer rate.

This is the buoyancy form of `Υ(ρ, z) = g(z - z✶(ρ))/ρ₀` as
[Wenegrat, Chor & Barkan (2026)](https://arxiv.org/abs/2605.15879) write it in their Eq. (7); the two
differ by the factor `-g/ρ₀` that converts between buoyancy and density, which cancels wherever `Υ` is
contracted with a buoyancy gradient. The result lives at `(Center, Center, Center)` and is a length
(units `m`).

`z✶` is the reference height computed by
[`reference_height`](@ref Oceanostics.BackgroundPotentialEnergyEquation.reference_height); pass one
explicitly to share a single sort with the other reference-state diagnostics, or pass `method` through
to choose how it is built.

`HeavisideIntegral` is the default here rather than Oceanostics' `ThreeDimensionalSort` because `Υ` is
a map, and every use of it differentiates that map. Only Eq. (11) of Winters et al. makes `z✶` a
function of buoyancy alone, so tied cells share one reference height instead of taking consecutive
slots; with `ThreeDimensionalSort` a run of equal buoyancy spreads `z✶` over the depth it fills, which
is harmless in a volume integral but shows up in `∇Υ` as grid-scale noise.
"""
function BuoyancyDisplacementPotential(model; method = HeavisideIntegral(),
                                       geopotential_height = model_geopotential_height(model),
                                       location = (Center, Center, Center))
    validate_location(location, "BuoyancyDisplacementPotential")
    return BuoyancyDisplacementPotential(model, reference_height(model; method, geopotential_height))
end

function BuoyancyDisplacementPotential(model, z✶::SortedReferenceHeightField)
    validate_reference_height_grid("BuoyancyDisplacementPotential", model, z✶)
    return KernelFunctionOperation{Center, Center, Center}(upsilon_ccc, z✶.grid, z✶)
end
#---

#+++ Available potential energy dissipation rate ε_A
# `ε_A = κ ∂ᵢb ∂ᵢΥ = -qᵢ ∂ᵢΥ`, where `qᵢ = -κ ∂ᵢb` is the tracer's diffusive flux. Taking it from the
# closure's own `diffusive_flux_*` rather than from a `κ` of our own makes this work for any closure,
# and makes the dissipation consistent with the diffusion the model actually applied — the same
# conservative formulation Oceanostics' `TracerVarianceDissipationRate` uses for `χ = 2 ∂ⱼc·Fⱼ`. Each
# product is formed on the face where both factors live and only then interpolated to the cell center,
# so the flux at a no-flux boundary (where the tracer halo is mirrored, making `δb` there exactly zero)
# contributes nothing.
@inline Axᶠᶜᶜ_δΥᶠᶜᶜ_q₁ᶠᶜᶜ(i, j, k, grid, Υ, closure, closure_fields, id, c, args...) =
    - Axᶠᶜᶜ(i, j, k, grid) * δxᶠᵃᵃ(i, j, k, grid, Υ) * diffusive_flux_x(i, j, k, grid, closure, closure_fields, id, c, args...)

@inline Ayᶜᶠᶜ_δΥᶜᶠᶜ_q₂ᶜᶠᶜ(i, j, k, grid, Υ, closure, closure_fields, id, c, args...) =
    - Ayᶜᶠᶜ(i, j, k, grid) * δyᵃᶠᵃ(i, j, k, grid, Υ) * diffusive_flux_y(i, j, k, grid, closure, closure_fields, id, c, args...)

@inline Azᶜᶜᶠ_δΥᶜᶜᶠ_q₃ᶜᶜᶠ(i, j, k, grid, Υ, closure, closure_fields, id, c, args...) =
    - Azᶜᶜᶠ(i, j, k, grid) * δzᵃᵃᶠ(i, j, k, grid, Υ) * diffusive_flux_z(i, j, k, grid, closure, closure_fields, id, c, args...)

@inline ape_dissipation_rate_ccc(i, j, k, grid, args...) =
    (ℑxᶜᵃᵃ(i, j, k, grid, Axᶠᶜᶜ_δΥᶠᶜᶜ_q₁ᶠᶜᶜ, args...) + # F, C, C  → C, C, C
     ℑyᵃᶜᵃ(i, j, k, grid, Ayᶜᶠᶜ_δΥᶜᶠᶜ_q₂ᶜᶠᶜ, args...) + # C, F, C  → C, C, C
     ℑzᵃᵃᶜ(i, j, k, grid, Azᶜᶜᶠ_δΥᶜᶜᶠ_q₃ᶜᶜᶠ, args...)   # C, C, F  → C, C, C
     ) / Vᶜᶜᶜ(i, j, k, grid) # the division by volume, against the `A δΥ` above, is what makes it a derivative

const AvailablePotentialEnergyDissipationRate = CustomKFO{<:typeof(ape_dissipation_rate_ccc)}

"""
    AvailablePotentialEnergyDissipationRate(model; method = HeavisideIntegral(), geopotential_height, location)
    AvailablePotentialEnergyDissipationRate(model, z✶; upsilon = nothing)

Return a `KernelFunctionOperation` computing the rate at which molecular diffusion destroys available
potential energy,

```
    ε_A = κ ∂ᵢb ∂ᵢΥ = κ [(∂z✶/∂b)|∇b|² - ∂b/∂z] ,
```

the sink of the local APE equation of
[Wenegrat, Chor & Barkan (2026)](https://arxiv.org/abs/2605.15879) (their Eqs. 11 and 14, where it
appears as `-ε_A`), with `Υ` the [`BuoyancyDisplacementPotential`](@ref). The first part is the
diapycnal mixing rate of [Winters et al. (1995)](https://doi.org/10.1017/S002211209500125X), the work
done rearranging the reference state; the second is the diffusion that state undergoes on its own,
which carries no APE with it. The two cancel exactly for a statically stable, horizontally uniform
stratification, where `z✶ = z` and there is no available energy to destroy, so `ε_A` measures only the
APE actually lost — it is not the sign-definite `κ|∇b|²`-like quantity that name might suggest.

`κ ∂ᵢb` is taken from the closure's own diffusive flux rather than from a diffusivity supplied here, so
this follows whatever closure the model runs with, and is written in the same conservative form
Oceanostics' [`TracerVarianceDissipationRate`](@ref Oceanostics.TracerVarianceEquation.TracerVarianceDissipationRate)
uses. The result lives at `(Center, Center, Center)`, per unit mass (units `m² s⁻³`).

The buoyancy has to be a tracer the closure diffuses, so this is defined for `BuoyancyTracer` models
only — `SeawaterBuoyancy` would need the diffusive fluxes of temperature and salinity combined through
the equation of state.

`z✶` is the reference height computed by
[`reference_height`](@ref Oceanostics.BackgroundPotentialEnergyEquation.reference_height), and has to
be one that lives on the model grid (`HeavisideIntegral`, `ThreeDimensionalSort` or `ProfileLookup`,
not `VerticalSort`) since `∇b` is taken on that grid. `upsilon` takes a `Field` of `Υ` you already
have, so that writing both out costs one sort and one `Υ` rather than two of each:

```julia
z✶ = reference_height(model, method=HeavisideIntegral())
Υ  = Field(BuoyancyDisplacementPotential(model, z✶))
ε_A = AvailablePotentialEnergyDissipationRate(model, z✶; upsilon=Υ)
```
"""
function AvailablePotentialEnergyDissipationRate(model; method = HeavisideIntegral(),
                                                 geopotential_height = model_geopotential_height(model),
                                                 location = (Center, Center, Center))
    validate_location(location, "AvailablePotentialEnergyDissipationRate")
    return AvailablePotentialEnergyDissipationRate(model, reference_height(model; method, geopotential_height))
end

function AvailablePotentialEnergyDissipationRate(model, z✶::SortedReferenceHeightField; upsilon = nothing)

    model.buoyancy isa BuoyancyTracerModel ||
        throw(ArgumentError("`AvailablePotentialEnergyDissipationRate` needs the buoyancy to be a tracer the closure \
                             diffuses, so that `κ∇b` is the closure's own diffusive flux, but this model's buoyancy is \
                             a $(summary(model.buoyancy)). Only `BuoyancyTracer` is supported for now."))

    validate_reference_height_grid("AvailablePotentialEnergyDissipationRate", model, z✶)

    Υ = isnothing(upsilon) ? Field(BuoyancyDisplacementPotential(model, z✶)) : upsilon
    tracer_index = findfirst(n -> n === :b, propertynames(model.tracers))

    return KernelFunctionOperation{Center, Center, Center}(ape_dissipation_rate_ccc, model.grid,
                                                           Υ,
                                                           model.closure,
                                                           model.closure_fields,
                                                           Val(tracer_index),
                                                           model.tracers.b,
                                                           model.clock,
                                                           fields(model),
                                                           model.buoyancy)
end
#---

#+++ Display
# Same one-line description Oceanostics gives its own diagnostics, so these print like the rest.
Oceanostics.@diagnostic_show BuoyancyDisplacementPotential "BuoyancyDisplacementPotential" "buoyancy displacement potential  Υ = z✶ - z"
Oceanostics.@diagnostic_show AvailablePotentialEnergyDissipationRate "AvailablePotentialEnergyDissipationRate" "available potential energy dissipation rate  ε_A = κ ∂ᵢb ∂ᵢΥ"
#---
