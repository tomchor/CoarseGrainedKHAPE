# The vertically filtered reference profile ⟨b✶⟩, built online on a coarse column.
#
# The sub-filter APE terms measure the resolved reservoir against the rest state *as the filter sees it*:
# the sorted column b✶(z✶) convolved with the vertical marginal of the same Gaussian the x–z filter uses
# (Wenegrat, Chor & Barkan, Eq. 2.3). This file holds everything that builds that profile from the
# sorted column (`sorted_column` in distributed_diagnostics.jl); `filtered_reference_profile` is the one entry point
# the simulation calls.
#
# The column is a grid of N cells spanning Lz uniformly, so a filter carrying the *physical* σ convolves in true
# height — true only because the model grid is uniform in z. On a stretched grid the column's slot heights
# vary (slot k is ΔV_k/(Lx·Ly)) while its grid stays uniform, so the convolution would silently run in index
# space; ⟨b✶⟩ would then have to be built on the model grid instead. Same restriction the offline
# `filtered_reference_profile` (postprocessing/src/aux01_pe_functions.py) carries.
#
# ⟨b✶⟩ is a Gaussian convolution of width σ, so it carries no structure below σ and filtering it at the
# column's own resolution is wasted work: the column has one slot per grid cell, which oversamples the
# result by ~400x at ℓ=1 and ~3000x at ℓ=7. Worse, the cost is quadratic -- N slots against a stencil
# that itself grows with N, since σ in slot units is σN/Lz -- so at Nz=512 the two scales together cost
# ~6e9 operations per output. Oceanostics' filter is a KernelAbstractions kernel written for GPUs, and on
# CI's two CPU threads that measured ~3 hours of a 4h23m run.
#
# Instead, block-average the column onto REFERENCE_FILTER_K levels per σ, filter there, and hand that pair
# straight to `ProfileLookup`, which takes a (b✶, z✶) of any length. Both grids are uniform, so a block
# average is an exact area average -- the right resampling, since the column carries structure below the
# coarse spacing (its tie runs among it) that point-sampling would alias.
#
# The filter runs on those M levels, and ⟨b✶⟩ is then interpolated back onto the column's N heights before
# the lookup, so only the *convolution* is coarse: Υ̃ = z̃✶(b̄) - z keeps the column's own resolution. Reading
# the coarse cells whole instead quantises Υ̃ to Lz/M, and Π̃_A = -τ(uᵢ,b) ∂ᵢΥ̃ differentiates it with nothing
# downstream to smooth the steps -- at Nz=1024, ℓ=7 that is δz̃✶ = dz/8.2, a ~12% staircase in ∂Υ̃/∂z, plainly
# visible in the Π_A panel while every term that *integrates* Υ̃ stayed smooth. The interpolation costs one
# O(N) kernel and nothing in the lookup, which is a binary search (`searchsortedfirst`), so O(log N).
#
# The offline profile has no levels-per-σ parameter: it filters the whole column by FFT, so the two
# constructions differ. Measured at Nz=1024 the difference is ≤1e-5 on S̃ and 0.007% on ∫Π_A
# (filtered_reference_decisions.md §3, §9).

using Oceananigans: RectilinearGrid
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Architectures: architecture, on_architecture
using Oceananigans.DistributedComputations: child_architecture
using Oceananigans.Fields: Field
using Oceananigans.Grids: Bounded, Center, Face, Flat, znode

#+++ Coarse column
# Levels per σ on the coarse grid the online ⟨b✶⟩ is filtered on; see `coarse_column`.
const REFERENCE_FILTER_K = 1000

# One coarse cell is the mean of `n` consecutive column slots. The kernel is launched over the coarse
# grid and reads the column field directly; both are 1×1×· so only k varies.
@inline function _block_mean_ccc(i, j, k, coarse_grid, fine, n)
    acc = zero(eltype(fine))
    @inbounds for m in 1:n
        acc += fine[1, 1, (k - 1) * n + m]
    end
    return acc / n
end

# ...and back: coarse cell κ is the mean of column slots (κ-1)n+1 … κn, so it stands at fractional column
# index (κ-0.5)n + 0.5, and column slot k sits at fractional coarse index (k-0.5)/n + 0.5. Interpolating
# there rather than reading the coarse cell whole is what keeps the z̃✶ lookup at the column's own
# resolution: the filter still runs on M levels (that is the expensive part), but Υ̃ = z̃✶(b̄) - z is no
# longer quantised to Lz/M. Mapping in index space rather than in z avoids depending on M*n == N, which
# `fld` does not guarantee. Outside the range this clamps, matching the edge extension either side and
# numpy's interp, which the offline `filtered_reference_profile` ends with.
#
# The result must be non-decreasing, because ProfileLookup rejects any step down. (1 - w)c₀ + w c₁ is not
# monotone in floating point: near the walls ⟨b✶⟩ rises by less than an ulp per slot (at ℓ = 1, from about
# Nz = 4700), and its rounding then steps down. c₀ + w(c₁ - c₀) is monotone in w, and clamping it to [c₀, c₁]
# keeps each segment within its end values, so the joined profile is non-decreasing wherever the coarse one is.
# The two forms differ by at most an ulp. The offline twin clamps too (`np.minimum.accumulate` in _fft_gaussian).
@inline function _interp_from_coarse_ccc(i, j, k, column_grid, coarse, n, M)
    FT = eltype(column_grid)
    t  = (k - FT(0.5)) / n + FT(0.5)
    κ  = clamp(floor(Int, t), 1, M - 1)
    w  = clamp(t - κ, zero(FT), one(FT))
    c₀ = @inbounds coarse[1, 1, κ]
    c₁ = @inbounds coarse[1, 1, κ + 1]
    return clamp(c₀ + w * (c₁ - c₀), min(c₀, c₁), max(c₀, c₁))
end

"""
    coarse_column(grid, N, σ)

Coarse column of M cells spanning the model `grid`'s height, and the block size `n` that maps the N-slot sorted
column onto it, for a Gaussian of standard deviation `σ`. Returns `(coarse, n, M)`.

Like the sorted column it reads (`sorted_column` in distributed_diagnostics.jl), it has a `Flat` cross-section and
lives on the child architecture: every rank of a `Distributed` model builds the same profile from its own copy of
the column, and a grid on the `Distributed` architecture would split the column's single x–y cell across ranks.
"""
function coarse_column(grid, N, σ)
    M_target = ceil(Int, REFERENCE_FILTER_K * grid.Lz / σ)
    n = max(1, fld(N, min(M_target, N)))          # slots per coarse cell
    M = fld(N, n)                                  # drops at most n-1 slots at the top
    z_bottom = znode(1, 1, 1, grid, Center(), Center(), Face())
    coarse = RectilinearGrid(child_architecture(architecture(grid)), eltype(grid);
                             size = M, topology = (Flat, Flat, Bounded), z = (z_bottom, z_bottom + grid.Lz))
    return coarse, n, M
end
#---

#+++ The column filter
# The column filter is Oceanostics' Gaussian in every respect except how the weights are carried, and it
# is spelled out here rather than reused because that difference is fatal at this width. `GaussianFilterKernel`
# stores its weights as an `NTuple` inside the kernel's *type* and fully unrolls the stencil loop, which is
# the right design at the widths the x-z filter uses (tens) and pathological at ~8K: the tuple is passed
# **by value** into CUDA's 32 KiB kernel parameter space, so a 8017-wide stencil is 62.6 KiB and the launch
# fails outright ("Kernel invocation uses too much parameter memory", sm_80), and asking LLVM to unroll 8017
# iterations is its own cost. Weights in a device array and a plain loop: the array costs 72 bytes as a
# kernel parameter no matter how long it is, so the width ceiling goes away and K is an accuracy choice again.
@inline function _gauss_column_ccc(i, j, k, coarse_grid, ψ, w, hw)
    FT = eltype(coarse_grid)
    s = zero(FT); w_sum = zero(FT)
    Nz = size(coarse_grid, 3)
    @inbounds for m = -hw:hw
        kk = min(max(k + m, 1), Nz)    # edge extension, matching Oceanostics' boundary=:edge and scipy's :nearest
        ω  = w[m + hw + 1]
        s     += ω * ψ[1, 1, kk]
        w_sum += ω
    end
    return s / w_sum
end

"""
    coarse_filter(grid, coarse, σ)

The vertical marginal of the x–z Gaussian of standard deviation `σ`, as an operator on fields of the
`coarse` column (stencil ≈ 8K taps, not 8σN/Lz), truncated at 4σ to match scipy's `truncate=4`.
"""
function coarse_filter(grid, coarse, σ)
    Δ  = grid.Lz / size(coarse, 3)
    hw = max(1, floor(Int, 4σ / Δ + 0.5))          # truncate at 4σ, matching scipy
    FT = eltype(grid)
    w  = on_architecture(architecture(coarse), FT[exp(-(m * Δ)^2 / (2σ^2)) for m = -hw:hw])
    return ψ -> KernelFunctionOperation{Center, Center, Center}(_gauss_column_ccc, coarse, ψ, w, hw)
end
#---

#+++ Entry point
"""
    filtered_reference_profile(b✶, grid, σ)

⟨b✶⟩: the sorted column's buoyancy `b✶` (a `Field` on the column grid of N cells, as `sorted_column` returns it)
filtered with the vertical marginal of the Gaussian of standard deviation `σ`, returned as a `Field` on the same
column grid, so it pairs with the column's heights in a `ProfileLookup`. Block-averages the column onto the coarse
grid of `coarse_column`, filters there, and interpolates back; recomputed on every `compute!`, so the lookup tracks
the flow.
"""
function filtered_reference_profile(b✶::Field, grid, σ)
    column_grid = b✶.grid
    N = size(column_grid, 3)
    coarse, n, M = coarse_column(grid, N, σ)
    b✶_coarse = Field(KernelFunctionOperation{Center, Center, Center}(_block_mean_ccc, coarse, b✶, n))
    b✶_filtered_coarse = Field(coarse_filter(grid, coarse, σ)(b✶_coarse))                    # ⟨b✶⟩ on M levels
    return Field(KernelFunctionOperation{Center, Center, Center}(_interp_from_coarse_ccc,
                                         column_grid, b✶_filtered_coarse, n, M))              # ...back onto the N slots
end
#---
