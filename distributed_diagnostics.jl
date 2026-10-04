# The two pieces of the online diagnostics that need the whole domain at once: the sorted reference column, and the
# x-pass of the Gaussian filter. On one GPU both see the whole domain anyway. On a `Distributed` grid (`--ranks R`,
# x-slabs) every rank holds only its own slab, so both gather the other ranks' slabs first, with a CUDA-aware
# `MPI.Allgather!`.
#
# Neither has a counterpart in Oceanostics (0.21.2) that works on a `Distributed` grid:
#   - `reference_height(model, method=VerticalSort())` builds its 1×1×N column on the model's architecture, which on a
#     `Distributed` grid splits the size-1 horizontal dimensions across ranks and errors; and a column with Periodic x
#     and y keeps a (1, 1, 3) halo, so each of its Fields costs 9× its data (8 GB per Field at Nz=1024).
#   - The staged `GaussianFilter` wraps a periodic direction with the operand's *local* extent, and on a rank the
#     partitioned x is `FullyConnected`, so its x-pass clamps at the slab's edges without a word.
# Both belong upstream once they have run. Every view taken below is contiguous, which on a GPU makes it a `CuArray`
# itself (GPUArrays' `unsafe_contiguous_view`), so the sort and MPI see device memory, not a wrapper.

using MPI
using Oceananigans.AbstractOperations: KernelFunctionOperation, _compute!
using Oceananigans.Architectures: AbstractArchitecture, architecture, on_architecture, CPU
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.DistributedComputations: Distributed, DistributedGrid, child_architecture, reconstruct_global_grid
using Oceananigans.Fields: Field, FieldStatus, CenterField, compute_at!, interior, location, set_status!
using Oceananigans.Grids: AbstractGrid, Center, Face, Flat, Bounded, Periodic, RectilinearGrid, topology, znode
using Oceananigans.Operators: Vᶜᶜᶜ
using Oceananigans.Utils: KernelParameters, launch!, sync_device!
using Oceanostics.SpatialFilters: _GaussianFilter2D, _GaussianFilter3D, AbstractGaussianFilterKernel, GaussianFilterKernel,
                                  _single_dim_kfo, _launch_compute_into!, _compute_staged_filter!, check_filter_staging

import Oceananigans.Fields: compute!

#+++ Ranks
# The model is split into x-slabs only (`Partition(x=R)`): the FFT pressure solver takes no partition in z, and with
# y whole every rank holds complete y–z planes, so the y- and z-passes of the filter and the 2D writer's j=1 slice
# stay local.
x_ranks(grid::AbstractGrid) = x_ranks(architecture(grid))
x_ranks(arch::Distributed) = arch.ranks[1]
x_ranks(::AbstractArchitecture) = 1

function validate_x_slabs(grid)
    arch = architecture(grid)
    arch.ranks[2] == arch.ranks[3] == 1 ||
        throw(ArgumentError("the distributed diagnostics assume x-slabs (Partition(x=R)); this grid is split $(arch.ranks)"))
    return nothing
end
#---

#+++ Gathering the x-slabs
# One pair of buffers serves every gather: `send`, this rank's slab, and `recv`, every rank's, in rank order. Both are
# sized for the largest operand, a field at faces of the bounded z, so a single allocation of about one global field
# serves every filter and the sort: the filters read the first R·n entries of `recv`, the sort the first N.
const GATHER_BUFFERS = IdDict{Any, Any}()

function gather_buffers(grid)
    return get!(GATHER_BUFFERS, grid) do
        arch = child_architecture(architecture(grid))
        nx, ny, nz = size(grid)
        n = nx * ny * (nz + 1)
        (send = on_architecture(arch, zeros(eltype(grid), n)),
         recv = on_architecture(arch, zeros(eltype(grid), n * x_ranks(grid))))
    end
end

"""
    gather_slabs!(ψ, grid, sz)

Evaluate `ψ` (a `Field` or an operation) over the rank's interior, of size `sz` at ψ's location, and gather every
rank's slab into `gather_buffers(grid).recv`: rank `r`'s at `recv[r*n+1 : (r+1)*n]`, with `n = prod(sz)`, each in
column-major order. Returns `recv`. On one rank it only evaluates `ψ`, straight into `recv`.
"""
function gather_slabs!(ψ, grid, sz)
    buffers = gather_buffers(grid)
    arch = architecture(grid)
    R = x_ranks(arch)
    n = prod(sz)

    slab = reshape(view(R == 1 ? buffers.recv : buffers.send, 1:n), sz)
    launch!(arch, grid, KernelParameters(sz, (0, 0, 0)), _compute!, slab, ψ)

    if R > 1
        sync_device!(arch)   # the kernel runs asynchronously, and MPI reads the buffer itself
        FT = eltype(buffers.send)
        MPI.Allgather!(MPI.Buffer(buffers.send, n, MPI.Datatype(FT)), MPI.UBuffer(buffers.recv, n), arch.communicator)
    end

    return buffers.recv
end
#---

#+++ The sorted reference column
"""
    GlobalSortState

Operand of the `b✶` Field [`sorted_column`](@ref) returns. Like Oceanostics' `SortedReferenceState` it hooks a
whole-field computation into `compute!`, here a gather of every rank's buoyancy and a sort of the lot, so the column
is refreshed whenever anything reads it.
"""
struct GlobalSortState{B, G}
    buoyancy :: B   # the model's buoyancy
    grid :: G       # its grid, distributed or not
end

const GlobalSortedBuoyancyField = Field{<:Any, <:Any, <:Any, <:GlobalSortState}

function compute!(b✶::GlobalSortedBuoyancyField, time=nothing)
    s = b✶.operand
    compute_at!(s.buoyancy, time)

    recv = gather_slabs!(s.buoyancy, s.grid, size(s.buoyancy))
    sorted = view(recv, 1:prod(size(b✶)))
    sort!(sorted)                             # values only: nothing here needs to know where a parcel came from
    interior(b✶) .= reshape(sorted, size(b✶))

    set_status!(b✶.status, time)
    return b✶
end

"""
    sorted_column(b)

The Winters et al. (1995) sorted reference state of the buoyancy `b` as a column `(b✶, z✶)`, each a `Field` on a
`(Flat, Flat, Bounded)` grid of N cells, with N the number of cells in the whole domain, on the child architecture
(so every rank holds its own copy). `b✶` is every cell's buoyancy in ascending order, gathered and sorted again on
every `compute!`; `z✶` is the heights of the N equal-volume slots, fixed. The pair is what `VerticalSort` hands
Oceanostics' `ProfileLookup(z✶_column)`, given instead as `ProfileLookup(b✶, z✶)`.

The slot heights use Oceanostics' own arithmetic (`slot_centers!`): cells stacked from the bottom by cumulative
volume, with the horizontal area taken from the total volume so the column fills the domain's depth exactly. A Flat
cross-section carries no halo, so each Field costs N values rather than the 9N of a 1×1 Periodic one.
"""
function sorted_column(b)
    grid = b.grid
    arch = child_architecture(architecture(grid))
    FT   = eltype(grid)
    N    = prod(size(grid)) * x_ranks(grid)

    cpu_grid = on_architecture(CPU(), grid)
    z_bottom = convert(FT, znode(1, 1, 1, cpu_grid, Center(), Center(), Face()))
    column   = RectilinearGrid(arch, FT; size = N, topology = (Flat, Flat, Bounded), z = (z_bottom, z_bottom + grid.Lz))

    b✶ = Field{Center, Center, Center}(column; operand = GlobalSortState(b, grid), status = FieldStatus())
    z✶ = CenterField(column)

    ΔV = convert(FT, Vᶜᶜᶜ(1, 1, 1, cpu_grid))   # every cell holds the same volume
    cell_volume = on_architecture(arch, fill(ΔV, N))
    horizontal_area = convert(FT, sum(cell_volume) / grid.Lz)
    cumulative_volume = similar(cell_volume)
    cumsum!(cumulative_volume, cell_volume)
    interior(z✶) .= reshape(@.(z_bottom + (cumulative_volume - ΔV / 2) / horizontal_area), size(z✶))

    return b✶, z✶
end
#---

#+++ The filter's x-pass across ranks
# A Gaussian-filtered Field on a distributed grid is computed as Oceanostics' staged filter is, pass by pass, except
# that the x-pass reads every rank's slab: the operand is gathered, and the pass wraps periodically on the *global* Nx
# (the policy the filter carries for x is the slab's `EdgeBoundary`, which it ignores). The y- and z-passes are
# Oceanostics' own kernels, unchanged. The taps are summed in Oceanostics' order, so a filtered field is the serial one,
# bit for bit, given the same operand.
compute!(comp::Field{<:Any, <:Any, <:Any, <:_GaussianFilter2D, <:DistributedGrid}, time=nothing) = compute_distributed_filter!(comp, time)
compute!(comp::Field{<:Any, <:Any, <:Any, <:_GaussianFilter3D, <:DistributedGrid}, time=nothing) = compute_distributed_filter!(comp, time)

# Two intermediate Fields per operand size, reused by every filter, where Oceanostics allocates two fresh ones on every
# compute. An operand's extent depends on its location only through faces of the bounded z (x is periodic and y
# whole), so two sizes, and four Fields, cover every location the diagnostics filter.
const STAGED_SCRATCH = Dict{Any, Any}()

staged_scratch(grid, sz) = get!(STAGED_SCRATCH, (objectid(grid), sz)) do
    LZ = sz[3] == size(grid, 3) ? Center : Face
    scratch = (Field{Center, Center, LZ}(grid), Field{Center, Center, LZ}(grid))
    size(first(scratch)) == sz || throw(ArgumentError("no scratch Field has an operand's extent $sz on this grid"))
    scratch
end

# The x weights as a device array: a plain loop over it, where Oceanostics unrolls an `NTuple` passed by value, which
# at Nz=1024 and ℓ=7 is 975 weights of kernel parameter space.
const FILTER_WEIGHTS = Dict{Any, Any}()

filter_weights(kern::GaussianFilterKernel, grid) =
    get!(() -> on_architecture(child_architecture(architecture(grid)), collect(kern.weights)), FILTER_WEIGHTS, kern.weights)

const GLOBAL_X_TOPOLOGY = IdDict{Any, Any}()

global_x_topology(grid) = get!(() -> topology(reconstruct_global_grid(grid), 1), GLOBAL_X_TOPOLOGY, grid)

@inline width_of(::Val{w}) where w = w

@inline function _x_pass_gathered(i, j, k, grid, recv, weights, width, nx, ny, nz, i₀, Nx)
    s = zero(grid); w_sum = zero(grid)
    for m in 1:(2width + 1)
        ig = i₀ + i + m - width - 1
        ig = ig + Nx * (ig < 1) - Nx * (ig > Nx)   # Oceanostics' `wrap_periodic_index`, on the global Nx
        r  = (ig - 1) ÷ nx                         # the rank that holds global column ig ...
        il = ig - r * nx                           # ... and its index there
        @inbounds ω = weights[m]
        @inbounds s += ω * recv[il + nx * ((j - 1) + ny * ((k - 1) + nz * r))]
        w_sum += ω
    end
    return s / w_sum
end

function x_pass_across_ranks!(dest, ψ, kern, width, loc, grid)
    arch = architecture(grid)
    sz = size(dest)   # ψ's whole extent at its location
    nx, ny, nz = sz
    Nx = nx * x_ranks(arch)

    global_x_topology(grid) === Periodic ||
        throw(ArgumentError("the distributed x-pass wraps periodically, but this grid's x is $(global_x_topology(grid))"))
    2width + 1 ≤ 2Nx + 1 ||
        throw(ArgumentError("an x stencil of $(2width + 1) points wraps more than once around Nx = $Nx"))

    recv = gather_slabs!(ψ, grid, sz)
    i₀ = (arch.local_index[1] - 1) * nx   # this rank's offset in x
    kfo = KernelFunctionOperation{loc...}(_x_pass_gathered, grid, recv, filter_weights(kern, grid), width, nx, ny, nz, i₀, Nx)
    _launch_compute_into!(dest, grid, kfo)

    return dest
end

function compute_distributed_filter!(comp, time)
    op    = comp.operand
    grid  = op.grid
    kern1 = op.kernel_function

    # Dims are sorted, so a filter that acts in x makes it the first pass; one that does not is local as it stands.
    kern1 isa AbstractGaussianFilterKernel{1} || return _compute_staged_filter!(comp, time)
    kern1 isa GaussianFilterKernel{1} ||
        throw(ArgumentError("the distributed x-pass needs a uniform x; this filter has a stretched-grid kernel in x"))
    validate_x_slabs(grid)

    loc  = location(op)
    args = op.arguments
    ψ    = args[end]
    compute_at!(ψ, time)   # first: computing ψ may run other filters, which share the scratch below

    temp1, temp2 = staged_scratch(grid, size(grid, loc))
    x_pass_across_ranks!(temp1, ψ, kern1, width_of(args[1]), loc, grid)

    if length(args) == 6   # a 2D filter: the second pass writes the result
        final = _single_dim_kfo(loc, grid, args[3], args[4], args[5], temp1)
    else                   # (x, y, z)
        _launch_compute_into!(temp2, grid, _single_dim_kfo(loc, grid, args[3], args[4], args[5], temp1))
        final = _single_dim_kfo(loc, grid, args[6], args[7], args[8], temp2)
    end

    _launch_compute_into!(comp, grid, final)
    fill_halo_regions!(comp)
    set_status!(comp.status, time)

    return comp
end
#---

#+++ Windowed Fields on a distributed grid
# The NetCDFWriter writes every output through a window onto its interior (`with_halos = false`), a Field whose data
# holds no halo cells. On a one-process grid a window gets no boundary conditions in its windowed directions, so
# filling its halos does nothing; Oceananigans' (0.113.3) distributed `Field` constructor injects the halo
# communication conditions in x regardless, so the first `compute!` of a window, which ends by filling its halos, sends
# cells it does not have and fails out of bounds (`_fill_east_send_buffer!`), for any written output that is computed.
# A window can never exchange halos it does not hold, so this more specific method keeps its windowed conditions as
# they are. (JLD2Writer defaults to `with_halos = true`, which is why distributed JLD2 output does not meet this.)
using Oceananigans.Fields: validate_field_data, validate_boundary_conditions
using Oceananigans.Grids: validate_indices
using Oceananigans.DistributedComputations: communication_buffers

function Oceananigans.Fields.Field(loc::Tuple{<:LX, <:LY, <:LZ}, grid::DistributedGrid, data, bcs,
                                   indices::Tuple{<:UnitRange, <:Any, <:Any}, op, status) where {LX, LY, LZ}
    indices = validate_indices(indices, loc, grid)
    validate_field_data(loc, data, grid, indices)
    validate_boundary_conditions(loc, grid, bcs)
    return Field{LX, LY, LZ}(grid, data, bcs, indices, op, status, communication_buffers(grid, data, bcs))
end
#---

#+++ Guard
"""
    validate_staged_filters(outputs)

On a `Distributed` grid only a filter that is a `Field`'s direct operand gathers across ranks (`compute!` above); one
nested in another operation runs Oceanostics' fused kernel, which clamps x at the slab's edges as the unpatched
staged one does. Refuse to start if any output would evaluate a filter that way.
"""
function validate_staged_filters(outputs)
    for (name, output) in pairs(outputs)
        check_filter_staging(output; warn = false) ||
            throw(ArgumentError("output `$name` evaluates a multi-direction filter on Oceanostics' fused path, which \
                                 does not gather across ranks; materialize the filtered field with `Field(...)` first"))
    end
    return nothing
end
#---
