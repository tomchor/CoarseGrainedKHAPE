# Online budget terms that Oceanostics does not (yet) provide, for the coarse-grained KE and APE
# budgets of Wenegrat, Chor & Barkan (2026). Everything else the budgets need is upstream; these two
# are what is left:
#
#   SubFilterAvailablePotentialToKineticEnergyConversion   τ(w, b_r) = filter(w b_r) - w̄ b_rˡ
#   ReferenceTendencyCorrection                            R = ∫_{z✶}^{z} ∂ₜb✶(z̃) dz̃
#
# The conversion is the sub-filter half of the split whose filtered half Oceanostics exports as
# `FilteredAvailablePotentialToKineticEnergyConversion`: together they sum to filter(w b_r), so the two
# budgets exchange exactly what the full field converts.
#
# R is the term that exists because the local APE is measured against a reference state that itself
# evolves: a parcel's APE changes even when neither the parcel nor its buoyancy moves, simply because
# b✶ has shifted underneath it. See the note on `ReferenceTendencyCorrection` for why it is built from
# the reference profile's own time derivative rather than from ∂ₜeₐ.

using Oceananigans: NonhydrostaticModel
using Oceananigans.AbstractOperations: KernelFunctionOperation, @at
using Oceananigans.Architectures: architecture, on_architecture, CPU
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.BuoyancyFormulations: Zᶜᶜᶜ
using Oceananigans.Fields: Field, CenterField, FieldStatus, compute_at!, interior, set_status!
using Oceananigans.Grids: Center, Face, znode
using Oceananigans.Operators: ℑzᵃᵃᶜ, ℑzᵃᵃᶠ
using Oceananigans.OutputWriters: TimeDerivative
using Oceanostics
using Oceanostics: CustomKFO
using Oceanostics.BackgroundPotentialEnergyEquation: reference_height, reference_buoyancy, VerticalSort,
                                                     ProfileLookup, SortedReferenceHeightField,
                                                     reference_buoyancy_at_height, buoyancy_field
using Oceanostics.AvailablePotentialEnergyEquation: BuoyancyDisplacementPotential

import Oceananigans.Fields: compute!
import Oceananigans.AbstractOperations: operation_name

#+++ Sub-filter APE → KE conversion
# τ(w, b_r) = filter(w b_r) - w̄ b_rˡ, with b_r = b - b✶(z) and b_rˡ = b̄ - b✶(z). The reference profile
# is *not* filtered in either term, matching the convention Oceanostics' filtered conversion uses, so
# that the filtered and sub-filter halves sum to filter(w b_r) exactly. `w` lives on the z face, so
# each product is formed there and interpolated to the center, as both upstream conversions do.
@inline b_rᶜᶜᶜ(i, j, k, grid, b, b✶z) = @inbounds b[i, j, k] - b✶z[i, j, k]

@inline wb_rᶜᶜᶠ(i, j, k, grid, w, b, b✶z) = @inbounds w[i, j, k] * ℑzᵃᵃᶠ(i, j, k, grid, b_rᶜᶜᶜ, b, b✶z)

@inline subfilter_conversion_ccc(i, j, k, grid, wb_rˢ) = @inbounds wb_rˢ[i, j, k]

const SubFilterAvailablePotentialToKineticEnergyConversion = CustomKFO{<:typeof(subfilter_conversion_ccc)}

"""
    SubFilterAvailablePotentialToKineticEnergyConversion(model, filter; method = ProfileLookup(), geopotential_height)

Return the sub-filter conversion of available potential energy into kinetic energy,

```
    τ(w, b_r) = filter(w b_r) - w̄ b_rˡ ,   b_r = b - b✶(z) ,   b_rˡ = b̄ - b✶(z)
```

the sub-filter half of the split whose filtered half is Oceanostics'
`FilteredAvailablePotentialToKineticEnergyConversion` `w̄b_rˡ`. The two sum to `filter(w b_r)`, so the
sub-filter APE and KE budgets exchange exactly what the full field converts. It carries the opposite
sign in the two budgets, which is what makes it a reversible exchange rather than a source or a sink.

The reference profile is **not** filtered in either term, following the upstream convention: `b_rˡ` is
`b̄ - b✶(z)`, not `filter(b_r)`. That is what makes the two halves an exact decomposition, and it is
the one place this differs from the offline `calculate_ape_to_ke_exchange_term`, which filters `b_r`.

`method` has to be a `ProfileLookup`, as for every filtered-state diagnostic: the reference profile is
the sorted state of the *full* buoyancy, which the filtered field is measured against.
"""
function SubFilterAvailablePotentialToKineticEnergyConversion(model, filter; method = ProfileLookup(),
                                                              geopotential_height = Oceananigans.Models.model_geopotential_height(model))
    isnothing(method.profile) &&
        throw(ArgumentError("`SubFilterAvailablePotentialToKineticEnergyConversion` needs a `ProfileLookup` holding a \
                             profile, so that the filtered and sub-filter halves share one reference state. Pass \
                             `ProfileLookup(z✶_column)` with a column built by `reference_height(model, method=VerticalSort())`."))

    b   = buoyancy_field(model, model.buoyancy, geopotential_height)
    b̄   = Field(filter(b))
    w   = model.velocities.w
    w̄   = Field(filter(w))
    b✶z = reference_buoyancy_at_height(model.grid, method.profile)

    # filter(w b_r) and w̄ b_rˡ, each formed on the face and interpolated to the center, then differenced
    wb_r  = KernelFunctionOperation{Center, Center, Center}(
                (i, j, k, grid, w, b, b✶z) -> ℑzᵃᵃᶜ(i, j, k, grid, wb_rᶜᶜᶠ, w, b, b✶z), model.grid, w, b, b✶z)
    w̄b_rˡ = KernelFunctionOperation{Center, Center, Center}(
                (i, j, k, grid, w̄, b̄, b✶z) -> ℑzᵃᵃᶜ(i, j, k, grid, wb_rᶜᶜᶠ, w̄, b̄, b✶z), model.grid, w̄, b̄, b✶z)

    wb_rˢ = Field(filter(Field(wb_r))) - Field(w̄b_rˡ)

    return KernelFunctionOperation{Center, Center, Center}(subfilter_conversion_ccc, model.grid, wb_rˢ)
end
#---

#+++ Reference-tendency correction R
"""
    ReferenceTendencyState

Operand of the `Field` [`ReferenceTendencyCorrection`](@ref) returns. It carries the time derivative of
the sorted reference profile, the heights to evaluate the correction between, and the workspace each
`compute!` reuses. Like the sorted reference state itself this cannot be a `KernelFunctionOperation`:
evaluating `Ψ̇` at a height needs a cumulative integral over the whole column, not a stencil.
"""
struct ReferenceTendencyState{D, Z, S, W, FT}
    ∂ₜb✶ :: D            # TimeDerivative of the column's reference buoyancy
    z✶ :: Z              # model-grid reference height the correction is measured from
    source_height :: S   # each cell's own height, flattened
    workspace :: W       # cumulative ∫∂ₜb✶ dz̃ at the slot faces
    z_bottom :: FT
    Δz✶ :: FT            # the column's slot thickness (VerticalSort gives equal-volume slots)
end

const ReferenceTendencyField = Field{<:Any, <:Any, <:Any, <:ReferenceTendencyState}

"Ψ̇(ζ) = ∫_bottom^ζ ∂ₜb✶ dz̃, evaluated by locating ζ's slot in a uniformly spaced column."
@inline function psi_dot(ζ, Ψface, ∂ₜb✶, z_bottom, Δz✶, N)
    k = clamp(floor(Int, (ζ - z_bottom) / Δz✶) + 1, 1, N)
    return @inbounds Ψface[k] + ∂ₜb✶[k] * (ζ - (z_bottom + (k - 1) * Δz✶))
end

function compute!(R::ReferenceTendencyField, time=nothing)
    s = R.operand
    compute_at!(s.z✶, time)

    ∂ₜb✶ = vec(interior(s.∂ₜb✶.result))
    N = length(∂ₜb✶)

    # Ψ̇ at the slot faces: a cumulative integral up the column, closed off at the bottom by zero
    Ψface = s.workspace
    @inbounds Ψface[1] = zero(eltype(Ψface))
    cumsum!(view(Ψface, 2:N+1), ∂ₜb✶ .* s.Δz✶)

    z✶ = vec(interior(s.z✶))
    R_flat = psi_dot.(s.source_height, Ref(Ψface), Ref(∂ₜb✶), s.z_bottom, s.Δz✶, N) .-
             psi_dot.(z✶,              Ref(Ψface), Ref(∂ₜb✶), s.z_bottom, s.Δz✶, N)

    interior(R) .= reshape(R_flat, size(R))
    fill_halo_regions!(R)
    set_status!(R.status, time)

    return R
end

"""
    ReferenceTendencyCorrection(model, ∂ₜb✶, z✶)

Return a `Field` holding the reference-tendency correction

```
    R = ∫_{z✶}^{z} ∂ₜb✶(z̃) dz̃
```

the rate at which a parcel's local available potential energy changes because the reference state
itself is evolving, with the parcel and its buoyancy held fixed. It is the explicit `∂ₜeₐ|_{b,z}`, and
appears in the local APE budget as `+R`.

`∂ₜb✶` is an Oceananigans `TimeDerivative` of the sorted reference profile — the `reference_buoyancy` of
a column built with `VerticalSort` — and `z✶` is the model-grid reference height the correction is
measured from: the full field's for `R`, the filtered field's for `Rˡ`. The sub-filter correction is
then `Rˢ = filter(R) - Rˡ`.

Building it from the profile's own time derivative is deliberate. `R` can equally be written
`∂ₜeₐ - Υ ∂ₜb`, which is cheaper, but that route makes `Rˢ` contain `∂ₜEₐˢ` identically, and the
sub-filter APE budget then closes in its tendency by construction rather than as a test. Differentiating
only the profile keeps `R` independent of the tendency the budget is checking.

The column's slots hold equal volume, so its faces are uniformly spaced and `Ψ̇` is piecewise linear on
a uniform grid, which is what lets the evaluation be a clamped index rather than a search.
"""
function ReferenceTendencyCorrection(model, ∂ₜb✶, z✶::SortedReferenceHeightField)
    grid = model.grid
    FT   = eltype(grid)
    N    = length(interior(∂ₜb✶.result))

    z_bottom = convert(FT, znode(1, 1, 1, on_architecture(CPU(), grid), Center(), Center(), Face()))
    Δz✶ = convert(FT, grid.Lz / N)

    source_height = on_architecture(architecture(grid), zeros(FT, prod(size(grid))))
    reshape(source_height, size(grid)) .= interior(Field(KernelFunctionOperation{Center, Center, Center}(Zᶜᶜᶜ, grid)))

    operand = ReferenceTendencyState(∂ₜb✶, z✶, source_height,
                                     on_architecture(architecture(grid), zeros(FT, N + 1)),
                                     z_bottom, Δz✶)

    return Field{Center, Center, Center}(grid; operand, status = FieldStatus())
end
#---

#+++ Display
Oceanostics.@diagnostic_show SubFilterAvailablePotentialToKineticEnergyConversion "SubFilterAvailablePotentialToKineticEnergyConversion" "sub-filter APE to KE conversion  τ(w, b_r) = filter(w b_r) - w̄b_rˡ"
#---
