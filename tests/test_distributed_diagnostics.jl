# Checks distributed_diagnostics.jl on x-slabs against the same computation on the whole grid in one process, bit
# for bit:
#   - the Gaussian filter (3D and 2D) of fields at (C,C,C), (F,C,C) and (C,C,F) and of two operations, with x stencils
#     wider than one rank's slab and as wide as the whole domain, against Oceanostics' staged filter;
#   - the sorted column `sorted_column(b)`, against the same on the whole grid and against Oceanostics' `VerticalSort`;
#   - the reference heights a `ProfileLookup` of that column assigns each cell;
# and, to roundoff, the horizontal average the simulation takes as an area integral (Oceananigans' `Average` over a
# partitioned dimension divides by one rank's cell count).
#
# Run on the CPU, with the MPI that MPI.jl bundles (no GPU or system MPI needed):
#   julia --project tests/test_distributed_diagnostics.jl
# which runs itself under mpiexec with 2 and then 4 ranks and fails if any rank finds a mismatch.

using MPI

if !haskey(ENV, "KHAPE_DISTRIBUTED_TEST")
    project = dirname(@__DIR__)
    for R in (2, 4)
        cmd = `$(MPI.mpiexec()) -n $R $(Base.julia_cmd()) --project=$project --startup-file=no $(@__FILE__)`
        run(addenv(cmd, "KHAPE_DISTRIBUTED_TEST" => "1"))
    end
    exit(0)
end

MPI.Init()

using Random
using Oceananigans
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceanostics: GaussianFilter
using Oceanostics.AvailablePotentialEnergyEquation: reference_height, reference_buoyancy, VerticalSort, ProfileLookup

include(joinpath(@__DIR__, "..", "distributed_diagnostics.jl"))

const comm = MPI.COMM_WORLD
const R    = MPI.Comm_size(comm)
const rank = MPI.Comm_rank(comm)

#+++ Grids and fields, the same global data in both
Nx, Ny, Nz = 24, 8, 16
domain = (x = (0, 3), y = (0, 1), z = (-1, 1), topology = (Periodic, Periodic, Bounded))   # isotropic, Δ = 1/8
serial = RectilinearGrid(CPU(); size = (Nx, Ny, Nz), domain...)
slabs  = RectilinearGrid(Distributed(CPU(); partition = Partition(x = R)); size = (Nx, Ny, Nz), domain...)

nx = Nx ÷ R
i₀ = rank * nx   # this rank's offset in x

rng  = Xoshiro(1)
data = (c = randn(rng, Nx, Ny, Nz), u = randn(rng, Nx, Ny, Nz), w = randn(rng, Nx, Ny, Nz + 1))

function fields_on(grid)
    f = (c = CenterField(grid), u = XFaceField(grid), w = ZFaceField(grid))
    for name in keys(f)
        set!(f[name], data[name])   # the two-argument set! hands each rank its part of the global array
        fill_halo_regions!(f[name])
    end
    return f
end

fs = fields_on(serial)
fd = fields_on(slabs)

operands(f) = (c = f.c, u = f.u, w = f.w, c² = f.c * f.c, ∂xc = ∂x(f.c))
#---

#+++ Checks
results = Pair{String, Bool}[]
check!(name, ok) = (push!(results, name => ok); ok)

local_part(a) = a[i₀+1:i₀+nx, :, :]

# x stencils wider than a slab (radius 15 > Nx/R) and at the one-period cap (2Nx + 1), y at its cap, z edge-extended
filters = (wide  = GaussianFilter(; dims = (1, 2, 3), σ = 0.6, boundary = :edge, N = (31, 9, 11)),
           whole = GaussianFilter(; dims = (1, 2, 3), σ = 2.0, boundary = :edge, N = (2Nx + 1, 2Ny + 1, 2Nz + 1)),
           xz    = GaussianFilter(; dims = (1, 3), σ = 0.6, boundary = :edge, N = (31, 11)))

for (fname, gf) in pairs(filters), (oname, ψs) in pairs(operands(fs))
    ψd = operands(fd)[oname]
    Fs = Field(gf(ψs))
    Fd = Field(gf(ψd))
    check!("filter $fname of $oname", interior(Fd) == local_part(interior(Fs)))
end

# The sorted column: the whole domain's on every rank, identical to one process's and to Oceanostics' VerticalSort
b✶s, z✶s = sorted_column(fs.c)
b✶d, z✶d = sorted_column(fd.c)
compute!(b✶s)
compute!(b✶d)
check!("sorted column b✶", interior(b✶d) == interior(b✶s))
check!("sorted column z✶", interior(z✶d) == interior(z✶s))

z✶_vs = reference_height(fs.c; method = VerticalSort())
compute!(z✶_vs)
check!("b✶ against VerticalSort", vec(interior(b✶s)) == vec(interior(reference_buoyancy(z✶_vs))))
check!("z✶ against VerticalSort", vec(interior(z✶s)) == vec(interior(z✶_vs)))

# The reference height each cell is given by a lookup into that column
z✶ˡs = reference_height(fs.c; method = ProfileLookup(b✶s, z✶s))
z✶ˡd = reference_height(fd.c; method = ProfileLookup(b✶d, z✶d))
check!("ProfileLookup reference height", interior(z✶ˡd) == local_part(interior(z✶ˡs)))

# What the NetCDFWriter computes: windows onto the interior (`with_halos = false`), of a filter, of an operation and of
# an existing Field. On a distributed grid they must compute, without the halo exchange they hold no cells for, to
# what the full fields hold.
interior_indices = (1:nx, 1:Ny, 1:Nz)
for (name, op) in (("a filter", filters.wide(fd.c)), ("an operation", fd.c * fd.c))
    window = Field(op; indices = interior_indices, compute = false)
    compute!(window)
    check!("window of $name", interior(window) == interior(Field(op)))
end
existing = Field(filters.wide(fd.c))
window = Field(existing; indices = interior_indices)
compute!(window)
check!("window of an existing Field", interior(window) == interior(existing))

# The horizontal average as the simulation takes it (an area integral), against `Average` on the whole grid
ε̄s = Field(Average(fs.c, dims = (1, 2)))
ε̄d = Field(Field(Integral(fd.c, dims = (1, 2))) / (3 * 1))
check!("horizontal average", isapprox(interior(ε̄d), interior(ε̄s); rtol = 1e-13))
#---

#+++ Report
all_ok = MPI.Allreduce(all(last, results), &, comm)
if rank == 0
    println("distributed_diagnostics.jl on $R x-slabs:")
    foreach(((name, ok),) -> println("  ", rpad(name, 32), ok ? "ok" : "MISMATCH (rank 0)"), results)
end
for r in 1:R-1   # every other rank reports only what it got wrong
    MPI.Barrier(comm)
    rank == r && foreach(((name, ok),) -> ok || println("  ", rpad(name, 32), "MISMATCH (rank $r)"), results)
end
MPI.Barrier(comm)
rank == 0 && println(all_ok ? "  all ranks agree" : "  FAILED")
exit(all_ok ? 0 : 1)
#---
