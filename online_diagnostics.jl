# The one online budget term Oceanostics does not provide, for the coarse-grained APE budget of
# Wenegrat, Chor & Barkan (2026):
#
#   ReferenceTendencyCorrection   R = ∫_{z✶}^{z} ∂ₜb✶(z̃) dz̃
#
# R exists because the local APE is measured against a reference state that itself evolves: a parcel's
# APE changes even when neither the parcel nor its buoyancy moves, simply because b✶ has shifted
# underneath it. See the note on `ReferenceTendencyCorrection` for why it is built from the reference
# profile's own time derivative rather than from ∂ₜeₐ.
#
# Everything else both budgets need is upstream, the sub-filter APE→KE conversion τ(w, b_r) included
# (Oceanostics `SubFilterAvailablePotentialToKineticEnergyConversion`, PR #301).

using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Architectures: architecture, on_architecture, CPU
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.BuoyancyFormulations: Zᶜᶜᶜ
using Oceananigans.Fields: Field, FieldStatus, compute_at!, interior, set_status!
using Oceananigans.Grids: Center, Face, znode
using Oceanostics.BackgroundPotentialEnergyEquation: SortedReferenceHeightField

import Oceananigans.Fields: compute!
import Oceananigans.OutputWriters: deferred_output

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

# R holds a TimeDerivative, so like a bare TimeDerivative it is only complete on the iteration after
# the writer actuates. `deferred_output` recurses through fields and operations down to this operand,
# so Rˢ = filter(R) - Rˡ and ∫Rˢ dV are deferred too: the writer evaluates them when a record opens
# (opening the ∂ₜb✶ window) and again on the following iteration, writing the completed difference.
deferred_output(::ReferenceTendencyState) = true

"Ψ̇(ζ) = ∫_bottom^ζ ∂ₜb✶ dz̃, evaluated by locating ζ's slot in a uniformly spaced column."
@inline function psi_dot(ζ, Ψface, ∂ₜb✶, z_bottom, Δz✶, N)
    k = clamp(floor(Int, (ζ - z_bottom) / Δz✶) + 1, 1, N)
    return @inbounds Ψface[k] + ∂ₜb✶[k] * (ζ - (z_bottom + (k - 1) * Δz✶))
end

function compute!(R::ReferenceTendencyField, time=nothing)
    s = R.operand
    compute_at!(s.z✶, time)
    compute_at!(s.∂ₜb✶, time)   # advances the TimeDerivative (a no-op when already at `time`)

    ∂ₜb✶ = vec(interior(s.∂ₜb✶))
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
