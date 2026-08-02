# Baroclinic adjustment (double-front, doubly-periodic-horizontal channel)
using Oceananigans
using Oceananigans.Units
using Printf
using Random
using ArgParse
using Oceanostics: PotentialEnergyEquation, GaussianFilter, KineticEnergyCrossScaleFlux, SubFilterKineticEnergyDissipationRate, FilteredKineticEnergyDissipationRate

@info "Finished loading packages"
Random.seed!(8675309)

include("utils.jl")

# Πₖ and ε_Kˢ used to be deferred to offline Python post-processing because Oceanostics v0.17.2's
# multi-direction `GaussianFilter(; dims=(1,2), σ)` crashed (heap-corruption SIGILL) on a grid with
# real Ny>1 and periodic y -- see https://github.com/tomchor/Oceanostics.jl/issues/262. Fixed upstream
# in v0.17.3 (PR #263). `KineticEnergyCrossScaleFlux`/`SubFilterKineticEnergyDissipationRate` (the
# latter added in the still-unmerged `tc/sfs-ke` branch, pinned in Project.toml/Manifest.toml) are now
# computed online directly below -- both validated against the previous offline Python implementation
# on a smoke-test run (0.99 spatial correlation, rms agreement within ~1%) before switching over.

#+++ Parse command-line arguments
let s = ArgParseSettings()
    @add_arg_table! s begin
        "--Nx"
            help = "Number of x grid points (default: 48, matching the Oceananigans baroclinic_adjustment example)"
            arg_type = Int
            required = false
            default = 48

        "--Ny"
            help = "Number of y grid points (default: 48)"
            arg_type = Int
            required = false
            default = 48

        "--Nz"
            help = "Number of z grid points (default: 8)"
            arg_type = Int
            required = false
            default = 8

        "--architecture"
            help = "Hardware architecture: 'auto' (default; use GPU() if CUDA.functional() reports a usable \
                    GPU, else fall back to CPU()), 'cpu' (force CPU() regardless of GPU availability), or \
                    'gpu' (force GPU(); errors out immediately if no functional CUDA GPU is detected, rather \
                    than silently falling back to CPU -- use this for a GPU smoke test so a misconfigured \
                    node fails loudly instead of quietly running on CPU)"
            arg_type = String
            required = false
            default = "auto"
            range_tester = (s -> s in ("auto", "cpu", "gpu"))

        "--N2"
            help = "Background stratification N² (default: 1e-5 s⁻²)"
            arg_type = Float64
            required = false
            default = 1e-5

        "--mixed_layer_depth"
            help = "Depth of a surface mixed layer with N²=0, in meters (default: 0.0, i.e. disabled -- the \
                    plain uniformly-stratified profile, matching all prior behavior). The initial buoyancy \
                    profile is b(z) = N²·min(z, -mixed_layer_depth) instead of the plain N²·z -- i.e. \
                    buoyancy is constant (well-mixed) from the surface down to this depth, then stratified \
                    at the background N² below it, continuous at the base of the mixed layer. Pass e.g. \
                    200 for a 200 m mixed layer."
            arg_type = Float64
            required = false
            default = 0.0

        "--M2"
            help = "Horizontal buoyancy gradient M² across each front (default: 1e-7 s⁻²)"
            arg_type = Float64
            required = false
            default = 1e-7

        "--front_width"
            help = "Width of each front, in km (default: 100)"
            arg_type = Float64
            required = false
            default = 100.0

        "--perturbation_amplitude"
            help = "Noise amplitude as a fraction of the front's buoyancy jump Δb (default: 0.01)"
            arg_type = Float64
            required = false
            default = 0.01

        "--latitude"
            help = "Latitude for the beta-plane Coriolis approximation (default: -45)"
            arg_type = Float64
            required = false
            default = -45.0

        "--implicit"
            help = "Run with fully implicit (numerical) dissipation: forces --closure=nothing (no explicit \
                    viscosity/diffusivity at all) and --advection_scheme=WENO(order=--implicit_weno_order), \
                    ignoring (with a warning) any --closure/--advection_scheme/--Pe_cell_h/--Pe_cell_v/ \
                    --nu_h/--nu_v flags that look like they were also explicitly set. Relies entirely on \
                    WENO's own implicit, scale-selective numerical dissipation for stability -- an \
                    implicit-LES configuration. IMPORTANT: the online ε_Kˢ/εˡ SFS dissipation diagnostics \
                    and the offline APE dissipation term both read from an explicit closure's ν/κ, which is \
                    nothing here, so they report ~zero rather than the real (numerical, untracked) \
                    dissipation actually happening -- the post-processing pipeline detects this via this \
                    flag's own recorded global attribute and substitutes a residual-based estimate instead \
                    (see 04_sfs_ke_budget.py/05_sfs_ape_budget.py). Default: false."
            action = :store_true

        "--implicit_weno_order"
            help = "WENO order used when --implicit is active (default: 9). Ignored unless --implicit. \
                    Lower order (5) is a less dissipative/less compact stencil than 9 -- both are valid \
                    implicit-LES configurations, just with different amounts of implicit numerical \
                    dissipation; 9 was the original/default choice, 5 is offered for comparison since it's \
                    also the order used by --advection_scheme=weno's own (non-implicit) WENO path."
            arg_type = Int
            required = false
            default = 9
            range_tester = (o -> o in (5, 9))

        "--closure"
            help = "Turbulence closure: 'constant' (ScalarDiffusivity/anisotropic-Laplacian with fixed \
                    --nu_h/--nu_v, matches the KH setup's explicit-dissipation design, but doesn't converge \
                    as resolution improves), 'scale_aware' (default; anisotropic Laplacian with ν set from a \
                    grid-Péclet-number criterion, ν = (U/Pe_cell)·Δ per direction -- shrinks automatically \
                    as the grid is refined, and is automatically much smaller vertically than horizontally \
                    since Δz ≪ Δx,Δy; see --Pe_cell_h/--Pe_cell_v), or 'smagorinsky' (SmagorinskyLilly LES closure -- NOT \
                    recommended at this resolution: its Richardson-number stability correction (Cb) zeroes \
                    out the eddy viscosity almost everywhere at mesoscale-permitting resolution, since \
                    resolved horizontal straining is tiny compared to the background stratification; \
                    disabling Cb instead makes it wildly too large, since its filter width assumes an \
                    isotropic grid cell -- kept only for reference/future work, see --C_smag/--Cb_smag)"
            arg_type = String
            required = false
            default = "scale_aware"
            range_tester = (s -> s in ("constant", "scale_aware", "smagorinsky"))

        "--nu_h"
            help = "Horizontal viscosity ν_h for the 'constant' closure (default: 1.0 m² s⁻¹). Ignored \
                    unless --closure=constant; see --Pe_cell_h/--Pe_cell_v for the 'scale_aware' closure instead."
            arg_type = Float64
            required = false
            default = 1.0

        "--nu_v"
            help = "Vertical viscosity ν_v for the 'constant' closure (default: 1.0 m² s⁻¹; see --nu_h). \
                    Ignored unless --closure=constant."
            arg_type = Float64
            required = false
            default = 1.0

        "--Pe_cell_h"
            help = "Target horizontal cell Péclet number Pe = UΔx/ν_h for the 'scale_aware' closure \
                    (default: 100.0 -- empirically tuned against horizontal surface-buoyancy smoothness \
                    across a resolution sweep; well above the classical linear-stability threshold of ~2, \
                    which was found to over-damp the actual baroclinic instability). Sets \
                    ν_h = (U/Pe_cell_h)·√(Δx·Δy), where U = M²·Lz/f is the thermal-wind velocity scale \
                    intrinsic to this problem's own parameters. Lower Pe_cell_h -> more horizontal \
                    dissipation. Independent of --Pe_cell_v (see its help for why they're not tied \
                    together). Ignored unless --closure=scale_aware."
            arg_type = Float64
            required = false
            default = 100.0

        "--Pe_cell_v"
            help = "Target vertical cell Péclet number Pe = UΔz/ν_v for the 'scale_aware' closure \
                    (default: 50.0 -- empirically re-tuned by a Pe_cell_v sweep at 96x96x16, see below; NOT \
                    the same as --Pe_cell_h's 100.0). Sets ν_v = (U/Pe_cell_v)·Δz. Kept separate from \
                    --Pe_cell_h because there's no reason the same empirically-tuned Péclet target should \
                    transfer between directions: the two grid spacings differ by ~2 orders of magnitude and \
                    the physics forcing each direction's grid-scale noise (horizontal straining vs. vertical \
                    shear) is different. Centered advection has no implicit dissipation, so nonlinear terms \
                    continuously alias energy toward the grid scale in both directions as soon as any \
                    resolved eddying motion exists -- sharing a single Pe_cell=100 under-damped this in the \
                    vertical once Δz was refined (visibly noisier fields, persistently-elevated SFS budget \
                    terms that didn't shrink with resolution the way a one-off IC transient would). A sweep \
                    over Pe_cell_v ∈ {100,50,20,10,5,2} at 96x96x16 found: below ~20 the flow becomes visibly \
                    over-damped (max|u| dropping from ~2 m/s at Pe_cell_v=100 to ~0.1 m/s at Pe_cell_v=2 --  \
                    baroclinic instability itself gets suppressed, not just grid noise), which spuriously \
                    *improves* the KE-budget residual ratio simply because all terms shrink together, not \
                    because the numerics genuinely improved -- watch for this artifact rather than chasing \
                    the residual ratio down in isolation. Pe_cell_v=50 was the best point that still matched \
                    the baseline's growth strength while measurably improving KE closure (residual ratio \
                    ~30-42% -> ~22-28%). Lower Pe_cell_v -> more vertical dissipation. Ignored unless \
                    --closure=scale_aware."
            arg_type = Float64
            required = false
            default = 50.0

        "--C_smag"
            help = "Smagorinsky constant C for the 'smagorinsky' closure (default: 0.16, Lilly 1966). \
                    νₑ = (C·Δᶠ)²·√(2Σ²)·√(1 - Cb·N²/Σ²). Ignored unless --closure=smagorinsky."
            arg_type = Float64
            required = false
            default = 0.16

        "--Cb_smag"
            help = "Stratification-correction multiplier Cb for the 'smagorinsky' closure (default: 1.0; \
                    set to 0 to disable -- see --closure's help for why neither setting works well at this \
                    resolution). Ignored unless --closure=smagorinsky."
            arg_type = Float64
            required = false
            default = 1.0

        "--Pr"
            help = "Prandtl number: sets κ_h = ν_h/Pr, κ_v = ν_v/Pr for the 'constant'/'scale_aware' \
                    closures, or the turbulent Prandtl number κₑ = νₑ/Pr for 'smagorinsky' (default: 1.0)"
            arg_type = Float64
            required = false
            default = 1.0

        "--bottom_drag"
            help = "Apply a quadratic bottom drag boundary condition on u/v: τ = Cd|U|u, U=(u,v,w) at the \
                    bottom cell, following \
                    https://github.com/whitleyv/IntWaveSlope/blob/main/Simulations/IntWave.jl. Cd is \
                    computed from a Monin-Obukhov log law, not set directly -- see --z0. Default: false \
                    (no bottom drag; free-slip at the bottom, matching all prior behavior)."
            action = :store_true

        "--z0"
            help = "Bottom roughness length, in meters, for the quadratic bottom drag's log-law \
                    coefficient Cd = (κᵥₖ/log(Δz/(2·z0)))², where κᵥₖ=0.4 (von Kármán constant, fixed) and \
                    Δz is this grid's vertical spacing. Ignored unless --bottom_drag. Default: 0.01 (1 cm, \
                    matching the reference implementation's Charnock roughness). Cd is resolution-dependent \
                    by design (a log law evaluated at the first grid point above the bottom, not a fixed \
                    physical constant) -- expect different effective drag at different Nz for the same z0."
            arg_type = Float64
            required = false
            default = 0.01

        "--advection_scheme"
            help = "Advection scheme: 'centered' (default; Centered(order=4), matches KH's setup) or 'weno' \
                    (WENO(order=5), matches the Oceananigans baroclinic_adjustment example; has its own \
                    implicit, scale-selective numerical dissipation on top of whichever explicit closure is \
                    also active -- recommended pairing is --advection_scheme=weno --closure=smagorinsky)"
            arg_type = String
            required = false
            default = "centered"
            range_tester = (s -> s in ("centered", "weno"))

        "--stop_time"
            help = "Simulation stop time in days (default: 20)"
            arg_type = Float64
            required = false
            default = 20.0

        "--output_interval_hours"
            help = "Interval between saved 3D/surface output snapshots, in hours (default: 12.0). Every \
                    offline coarse-graining diagnostic (∂ₜ tendency, Πₖ/Π_A cross-scale flux, εₖˢ/ε_Aˢ \
                    dissipation) is computed only from these saved snapshots -- a coarser interval means a \
                    coarser centered time-difference for tendencies, and nonlinear flux/dissipation \
                    products (uᵢuⱼ, uᵢρ) evaluated at more widely-spaced-in-time snapshots rather than the \
                    continuously-evolving field. Smaller values test whether the budget-closure residual is \
                    partly an aliasing artifact of insufficient temporal sampling."
            arg_type = Float64
            required = false
            default = 12.0

        "--filter_scales_m"
            help = "Two horizontal filter scales (FWHM, in meters) for the online coarse-graining diagnostics \
                    (default: 50000 100000), matching the units used throughout the offline post-processing \
                    pipeline (--filter-scales/FILTER_SCALES_M). Renamed from the old --filter_scales (which \
                    took km) specifically so a stale invocation fails loudly (unrecognized argument) instead \
                    of silently applying scales 1000x too small. No halo-size penalty for large scales -- the \
                    periodic (x,y) filter stencil reads via wrapped interior indexing, not halo cells (see the \
                    grid-construction note)."
            arg_type = Float64
            nargs = 2
            required = false
            default = [50000.0, 100000.0]

        "--progress_interval"
            help = "Print a progress message every this many iterations (default: 100; use a small value \
                    like 1 for short debug/smoke-test runs where 100 iterations may never be reached)"
            arg_type = Int
            required = false
            default = 100
    end
    global parsed_args = parse_args(s, as_symbols=true)
end
params = (; parsed_args...)
#---

#+++ Define simulation parameters
# Domain matches the Oceananigans baroclinic_adjustment literated example exactly (Lx=Ly=1000km, Lz=1km),
# but topology is (Periodic, Periodic, Bounded) instead of (Periodic, Bounded, Bounded): rather than a
# single front against two channel walls, we place *two* opposite-signed fronts so the buoyancy field
# closes periodically in y (see the double-ramp `bᵢ` below). This avoids side-wall boundary layers, which
# would otherwise complicate the coarse-grained KE/APE budget (extra wall dissipation term) and the
# horizontal Gaussian filter (which would need edge-extension at the bounded y walls, as in the original
# KH setup's bounded z). With both horizontal directions periodic, the filter is a pure periodic wrap.
Lx = 1000kilometers
Ly = 1000kilometers
Lz = 1kilometers
params = (; params..., Lx, Ly, Lz)
#---

#+++ Create grid
Nx = closest_factor_number((2, 3, 5), params.Nx)
Ny = closest_factor_number((2, 3, 5), params.Ny)
Nz = params.Nz
params = (; params..., Nx, Ny)

# The horizontal Gaussian filter's periodic (x, y) stencil does NOT need a halo sized to its truncation
# radius: Oceanostics' GaussianFilter reads periodic-direction neighbors via a wrapped *interior* array
# index (`wrap_periodic_index`), never through halo/ghost cells, however wide the stencil is (confirmed
# directly in Oceanostics' SpatialFilters source, and by tomchor). A small, fixed halo -- matching the z
# halo below, and Oceananigans' own advection-scheme default -- is sufficient regardless of filter scale.
# (An earlier version of this file sized Hx/Hy from the filter's 4σ radius, defending against a heap-
# corruption bug that was actually a kernel-launch-sizing issue in Oceanostics <v0.17.3, unrelated to halo
# width -- see the note above on issue #262/PR #263, which this repo is already pinned past.)
_FWHM_to_σ(ℓ) = ℓ / (2 * sqrt(2 * log(2)))

Δx = params.Lx / Nx
Δy = params.Ly / Ny
Δz = params.Lz / Nz

architecture = if params.architecture == "gpu"
    CUDA.functional() || error("--architecture=gpu was requested but CUDA.functional() is false -- no usable " *
                                "GPU detected. Check that a GPU is allocated to this job and the `cuda` module is loaded.")
    GPU()
elseif params.architecture == "cpu"
    CPU()
else # "auto"
    CUDA.functional() ? GPU() : CPU()
end
@info "Using architecture: $architecture"

grid = RectilinearGrid(architecture;
                       size=(Nx, Ny, Nz),
                       x=(0, params.Lx),
                       y=(-params.Ly/2, params.Ly/2),
                       z=(-params.Lz, 0),
                       halo=(3, 3, 3),
                       topology=(Periodic, Periodic, Bounded))
@info "Grid created: Nx=$Nx, Ny=$Ny, Nz=$Nz, halo=(3, 3, 3) (may be auto-inflated below if the advection \
scheme needs more, e.g. --implicit's WENO(order=$(params.implicit_weno_order)))"
#---

#+++ Create model
# Three closure options (--closure):
#
# 'constant' -- anisotropic Laplacian ScalarDiffusivity with fixed --nu_h/--nu_v. The grid is highly
# anisotropic (Δx,Δy ~ 20km vs Δz ~ 125m), so a single isotropic ν/κ gives a vertical diffusive damping
# time orders of magnitude shorter than horizontal at the same value -- e.g. ν=κ=1 m²/s gives a ~264 day
# horizontal damping time at the deformation radius (harmless over a 20-day run) but a ~4.3 hour vertical
# damping time (much faster than the ~1.2 day Eady growth time), smearing out the vertical
# shear/stratification structure baroclinic instability depends on before it can develop. Its downside:
# a fixed ν doesn't converge as the grid is refined, so it isn't scale-selective enough to control
# Centered's lack of implicit dissipation without either over-damping real structure (too large) or
# under-damping grid-scale noise (too small) -- see 'scale_aware' below.
#
# 'scale_aware' (default) -- anisotropic Laplacian ScalarDiffusivity, but with ν set automatically from a
# grid-Péclet-number criterion instead of a fixed value: ν_h = (U/Pe_cell_h)·√(Δx·Δy), ν_v =
# (U/Pe_cell_v)·Δz, where U is a velocity scale intrinsic to this problem (the thermal-wind scale
# U = M²·Lz/f). This makes ν shrink automatically as the grid is refined, and is automatically
# anisotropic (νv ≪ νh follows directly from Δz ≪ Δx,Δy) -- while remaining an explicit, deterministic
# Laplacian closure (not a residual, not high-order/biharmonic), so the SFS dissipation diagnostic stays
# as well-defined as the 'constant' closure's. Pe_cell_h and Pe_cell_v are independent CLI parameters
# (--Pe_cell_h defaults to 100, --Pe_cell_v to 50 -- see --Pe_cell_v's help for the sweep that picked 50):
# a resolution sweep with a single shared Pe_cell=100 (tuned only by eye against horizontal
# surface-buoyancy smoothness) showed visibly noisier fields and
# persistently-elevated (not resolution-decaying) SFS budget terms as Δz was refined at fixed Pe_cell --
# i.e. the horizontally-tuned Pe_cell under-damps continuously-regenerated vertical grid-scale noise
# (Centered advection has no implicit dissipation, so nonlinear terms keep aliasing energy toward the
# grid scale in both directions once any resolved eddying motion exists). Splitting the two lets each
# direction's dissipation be tuned to its own noise-generation/damping balance.
#
# 'smagorinsky' -- SmagorinskyLilly LES closure: a *diagnostic* eddy viscosity νₑ computed from the
# locally resolved strain rate. NOT recommended at this resolution: testing showed its Richardson-number
# stability correction (Cb term) clamps νₑ to exactly zero almost everywhere, since the resolved
# horizontal straining at mesoscale-permitting resolution is orders of magnitude smaller than the
# background stratification N² -- and disabling Cb (--Cb_smag 0) instead makes νₑ wildly too large
# (~1000s of m²/s), since SmagorinskyLilly's filter width assumes an isotropic grid cell, which this grid
# is not. Kept only for reference/future work at finer resolution.
# --implicit forces closure=nothing and advection_scheme=WENO(order=--implicit_weno_order), ignoring (with
# a warning) any --closure/--advection_scheme/--Pe_cell_h/--Pe_cell_v/--nu_h/--nu_v flags that look like
# they were also explicitly set (differ from their own defaults) -- "warn and override" rather than
# erroring, matching this CLI's generally permissive style elsewhere.
if params.implicit
    conflicting = String[]
    params.closure != "scale_aware"       && push!(conflicting, "--closure=$(params.closure)")
    params.advection_scheme != "centered" && push!(conflicting, "--advection_scheme=$(params.advection_scheme)")
    params.Pe_cell_h != 100.0             && push!(conflicting, "--Pe_cell_h=$(params.Pe_cell_h)")
    params.Pe_cell_v != 50.0              && push!(conflicting, "--Pe_cell_v=$(params.Pe_cell_v)")
    params.nu_h != 1.0                    && push!(conflicting, "--nu_h=$(params.nu_h)")
    params.nu_v != 1.0                    && push!(conflicting, "--nu_v=$(params.nu_v)")
    isempty(conflicting) ||
        @warn "--implicit=true forces closure=nothing and advection_scheme=WENO(order=$(params.implicit_weno_order)); ignoring: $(join(conflicting, ", "))"
end

νh_scale_aware = νv_scale_aware = nothing  # populated below only for --closure=scale_aware
closure = if params.implicit
    # NOT `closure = nothing`: confirmed directly (a failed smoke test) that Oceananigans has no
    # `viscous_flux_ux` method for a `Nothing` closure at all -- `nothing` is not a valid "no dissipation"
    # sentinel for NonhydrostaticModel in this version, it's simply undispatchable and errors with a
    # MethodError inside the first tendency calculation. An explicit zero-valued diffusivity dispatches to
    # the same AbstractScalarDiffusivity code path already used by the other closures below (proven to
    # work), just multiplying out to zero everywhere -- giving the same "no explicit dissipation" physics
    # without hitting the missing-method gap.
    (HorizontalScalarDiffusivity(ν=0, κ=0),
     VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization(); ν=0, κ=0))
elseif params.closure == "smagorinsky"
    SmagorinskyLilly(C=params.C_smag, Cb=params.Cb_smag, Pr=params.Pr)
elseif params.closure == "scale_aware"
    Ω_earth = 7.2921159e-5  # rad/s
    f = 2 * Ω_earth * sind(params.latitude)
    U_scale = params.M2 * params.Lz / abs(f)
    global νh_scale_aware = (U_scale / params.Pe_cell_h) * sqrt(Δx * Δy)
    global νv_scale_aware = (U_scale / params.Pe_cell_v) * Δz
    κh, κv = νh_scale_aware / params.Pr, νv_scale_aware / params.Pr
    (HorizontalScalarDiffusivity(ν=νh_scale_aware, κ=κh),
     VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization(); ν=νv_scale_aware, κ=κv))
else
    νh, νv = params.nu_h, params.nu_v
    κh, κv = params.nu_h / params.Pr, params.nu_v / params.Pr
    # (Previously special-cased ν=κ=0 to `closure=nothing` here -- removed: that hits the same missing
    # `viscous_flux_ux` dispatch as --implicit's case above, and was never actually exercised/tested before
    # now. A zero-valued ScalarDiffusivity works fine and gives identical physics.)
    (HorizontalScalarDiffusivity(ν=νh, κ=κh),
     VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization(); ν=νv, κ=κv))
end

# WENO(order=9) needs a wider stencil (halo≥5) than the grid's fixed halo=(3,3,3) -- order=5 needs halo≥3,
# already satisfied -- but this needs no manual handling either way: NonhydrostaticModel auto-inflates and
# rebuilds the grid's halo to match whatever the advection scheme requires (confirmed directly -- it logs
# "Inflating model grid halo size to ..." and transparently recreates the grid), and auto-downgrades the
# order in any direction too short to support the full stencil (e.g. a tiny Nz in a smoke test). No
# silent-corruption risk here, unlike this repo's own filter-halo history -- that was a hand-rolled stencil
# bypassing Oceananigans' own machinery, not this.
advection_scheme = params.implicit ? WENO(order=params.implicit_weno_order) : (params.advection_scheme == "weno" ? WENO(order=5) : Centered(order=4))

# Quadratic bottom drag (--bottom_drag): τ = Cd|U|u, U=(u,v,w) at the bottom cell, following
# https://github.com/whitleyv/IntWaveSlope/blob/main/Simulations/IntWave.jl. Cd is a Monin-Obukhov log-law
# coefficient (see --z0's help for why it's resolution-dependent by design), applied as a
# FluxBoundaryCondition on u and v at the bottom only -- this domain has a flat top and bottom (unlike the
# reference's sloping-bottom/immersed-boundary setup), so no immersed-boundary drag case is needed. Cd is
# passed via `parameters` rather than closed over as a global: a non-const global referenced inside a
# hot-path boundary function (called every timestep, every bottom cell) is a classic Julia performance trap.
κᵥₖ = 0.4  # von Kármán constant
Cd = nothing
boundary_conditions = NamedTuple()
if params.bottom_drag
    global Cd = (κᵥₖ / log(Δz / (2 * params.z0)))^2
    bottom_drag_u(x, y, t, u, v, w, p) = -p.Cd * u * sqrt(u^2 + v^2 + w^2)
    bottom_drag_v(x, y, t, u, v, w, p) = -p.Cd * v * sqrt(u^2 + v^2 + w^2)
    u_drag_bc = FluxBoundaryCondition(bottom_drag_u, field_dependencies=(:u, :v, :w), parameters=(; Cd))
    v_drag_bc = FluxBoundaryCondition(bottom_drag_v, field_dependencies=(:u, :v, :w), parameters=(; Cd))
    global boundary_conditions = (; u=FieldBoundaryConditions(bottom=u_drag_bc),
                                     v=FieldBoundaryConditions(bottom=v_drag_bc))
    @info "Bottom drag enabled: z0=$(params.z0) m, Δz=$Δz m, Cd=$Cd"
end

# Overwrite nu_h/nu_v/closure/advection_scheme/bottom_drag/Cd with the values actually used (rather than
# the raw --closure/--advection_scheme CLI strings, which stay at their defaults whenever --implicit
# overrode them) so they're recorded correctly in the NetCDF global attributes below and in the
# log/summary output. The Python postprocessing pipeline falls back to nu_h/nu_v/Pr for any closure whose
# ν/κ isn't written out as a spatial field (i.e. everything except 'smagorinsky'), checks the `implicit`
# attribute to decide whether to use the online/offline dissipation diagnostics or a residual-based
# estimate instead, and checks bottom_drag/Cd directly to decide whether to assemble the drag-work budget
# terms (see 04_sfs_ke_budget.py/05_sfs_ape_budget.py). `Cd=0.0` (not `nothing`) when disabled, since
# NCDatasets.jl can only write numeric/string attributes, not `nothing`.
if params.implicit
    global params = (; params..., closure="none", advection_scheme="weno$(params.implicit_weno_order)", nu_h=0.0, nu_v=0.0)
elseif params.closure == "scale_aware"
    global params = (; params..., nu_h=νh_scale_aware, nu_v=νv_scale_aware)
end
global params = (; params..., Cd=(Cd === nothing ? 0.0 : Cd))

# NonhydrostaticModel instead of HydrostaticFreeSurfaceModel: no free surface at all (no η, no
# barotropic pressure correction -- see conversation for the NetCDFWriter dimension-inference limitation
# that made patching the free-surface term into the old hydrostatic setup impractical), and w becomes a
# genuine prognostic variable with its own momentum equation and dissipative dynamics rather than
# diagnostic. This matches tomchor's own Eady baroclinic-instability example (Oceanostics PR #260),
# which closes its coarse-grained KE budget cleanly (~11-15% residual/dominant, vs our ~40-60% under the
# old hydrostatic setup) -- see conversation for the ongoing investigation into how much of that gap is
# closure/numerics vs. the buoyancy-production-term convention (w̄b̄ vs w̄b̄ᵣ).
model = NonhydrostaticModel(grid;
                            coriolis = BetaPlane(latitude=params.latitude),
                            buoyancy = BuoyancyTracer(),
                            tracers = :b,
                            advection = advection_scheme,
                            closure = closure,
                            boundary_conditions = boundary_conditions)
u, v, w = model.velocities
b = model.tracers.b
actual_halo = (model.grid.Hx, model.grid.Hy, model.grid.Hz)
if params.closure == "scale_aware"
    @info "Model created (advection=$(params.advection_scheme), closure=scale_aware, νh=$νh_scale_aware, νv=$νv_scale_aware, halo=$actual_halo, bottom_drag=$(params.bottom_drag))"
else
    @info "Model created (advection=$(params.advection_scheme), closure=$(params.closure), halo=$actual_halo, bottom_drag=$(params.bottom_drag))"
end
#---

#+++ Define initial conditions: double front + background stratification + noise
# `ramp(y, Δy)` is the same linear ramp (0 → 1 over a width Δy) used in the Oceananigans example. A single
# ramp centered at y₀ would leave b(-Ly/2) ≠ b(Ly/2), which isn't compatible with a periodic y topology.
# Instead we place two ramps of opposite sign at y₁ = -Ly/4 and y₂ = +Ly/4: the buoyancy is low outside
# the two fronts (wrapping through the periodic boundary) and high in the plateau between them, so the
# profile is periodic by construction and the domain hosts two opposite-signed fronts (a standard
# "double-front"/periodic-strip configuration used to avoid channel side walls).
ramp(y, Δy) = min(max(0, y/Δy + 1/2), 1)

Δy = params.front_width * kilometers
Δb = Δy * params.M2
ϵb = params.perturbation_amplitude * Δb

y₁ = -params.Ly/4
y₂ = +params.Ly/4

double_ramp(y, Δy) = ramp(y - y₁, Δy) - ramp(y - y₂, Δy)

bᵢ(x, y, z) = params.N2 * min(z, -params.mixed_layer_depth) + Δb * double_ramp(y, Δy) + ϵb * randn()
#---

#+++ Set initial conditions: u in thermal-wind balance with the front (v=w=0 -- the front is x-invariant,
# so the balanced flow is purely zonal), plus the buoyancy front itself (bᵢ, defined above).
# ramp is piecewise-linear, so its derivative is a boxcar of height 1/Δy over the ramp's transition width --
# the y-derivative of bᵢ's front term is then Δb * double_ramp_prime(y, Δy), i.e. ∂b/∂y, following exactly
# the same two-ramps structure as bᵢ/double_ramp above (so this is visibly "the derivative of that", not a
# separately-derived expression that could drift out of sync with it).
ramp_prime(y, Δy) = (-Δy/2 < y < Δy/2) ? 1/Δy : zero(y)
double_ramp_prime(y, Δy) = ramp_prime(y - y₁, Δy) - ramp_prime(y - y₂, Δy)

# Reused directly from the model's own coriolis object (not recomputed from params.latitude independently)
# so this is guaranteed consistent with whatever Coriolis parameter the momentum equation itself actually
# uses going forward -- a hand-rederived β could silently drift from Oceananigans' own value (e.g. a
# different assumed planetary radius) without erroring.
f₀ = model.coriolis.f₀
β  = model.coriolis.β
f_cor(y) = f₀ + β * y

# Thermal wind: f ∂u/∂z = -∂b/∂y, integrated from a reference level z_ref where u=0. z_ref = -Lz/2
# (mid-depth) specifically so the initial state carries no barotropic (depth-independent) velocity
# component -- integrating from any other level would leave a spurious depth-uniform u offset, since
# nothing else in this setup drives a real barotropic flow for that offset to correspond to.
z_ref = -params.Lz/2
db_dy(y) = Δb * double_ramp_prime(y, Δy)
u_geo(x, y, z) = -(db_dy(y) / f_cor(y)) * (z - z_ref)

set!(model, u=u_geo, b=bᵢ)
@info "Initial conditions set (u initialized in thermal-wind balance, z_ref=$(z_ref)m)"
#---

#+++ Setup simulation
# Start conservatively (Δt=1minute) rather than jumping straight to max_Δt: the random per-cell noise in
# the initial buoyancy perturbation (ϵb·randn()) gets proportionally sharper as the grid is refined (same
# noise amplitude spread over a smaller Δx means a steeper initial gradient), and at fine resolution this
# can blow up Centered advection within the first handful of iterations -- faster than a wizard that only
# re-checks every IterationInterval(20) steps can react. IterationInterval(5) lets it adapt quickly once
# the flow is actually resolved smoothly (usually within the first ~10-20 iterations).
simulation = Simulation(model, Δt=1minute, stop_time=params.stop_time * days)
# diffusive_cfl matters once the 'scale_aware' closure is in play: ν grows at coarser resolution (it's
# tied to Δ, not fixed), so the explicit horizontal-diffusion stability limit can become the binding
# constraint (tighter than plain advective CFL) -- without this, a run can blow up (NaN) well before the
# advective cfl=0.2 limit would ever ask for a smaller Δt.
conjure_time_step_wizard!(simulation, IterationInterval(5), cfl=0.2, diffusive_cfl=0.2, max_Δt=20minutes)
@info "Simulation object created"
#---

#+++ Add progress messenger
wall_clock = Ref(time_ns())

function print_progress(sim)
    u, v, w = model.velocities
    progress = 100 * (time(sim) / sim.stop_time)
    elapsed = (time_ns() - wall_clock[]) / 1e9

    @printf("[%05.2f%%] i: %d, t: %s, wall time: %s, max|u,v,w|: (%6.3e, %6.3e, %6.3e) m/s, next Δt: %s\n",
            progress, iteration(sim), prettytime(sim), prettytime(elapsed),
            maximum(abs, u), maximum(abs, v), maximum(abs, w), prettytime(sim.Δt))

    wall_clock[] = time_ns()
    return nothing
end
add_callback!(simulation, print_progress, IterationInterval(params.progress_interval))
#---

#+++ Add output writer
# Interpolate velocities to (Center, Center, Center) before writing/filtering: leaving them on their
# native staggered (Face) locations makes the offline Python post-processing (which multiplies fields
# together, e.g. uⁱuʲ for the stress tensor) broadcast across mismatched staggered coordinate dims
# (x_faa, y_afa, z_aaf) instead of a pointwise product -- silently producing a hugely-wrong-shaped,
# nonsensical result. Matches the KH setup's convention (u_center/v_center/w_center).
u_center = @at (Center, Center, Center) u
v_center = @at (Center, Center, Center) v
w_center = @at (Center, Center, Center) w

ζ = Field(∂x(v) - ∂y(u))
ρ₀ = 1025 # kg/m^3
pe = ρ₀ * PotentialEnergyEquation.PotentialEnergy(model)
PE = Integral(pe)

#+++ Gaussian-filtered u, v, w, b at multiple filter scales — HORIZONTAL ONLY (dims=(1,2)).
# Coarse-graining is applied over (x, y) at each depth level, not in z: horizontal scales span the
# mesoscale/submesoscale range this budget targets, while the vertical direction has its own distinct
# structure (stratification, surface/bottom boundary layers) that shouldn't be smoothed over. Both
# horizontal directions are periodic here, so (unlike the KH setup's bounded-z filter) no edge-extension
# boundary handling is needed — it's a pure periodic wrap.
filter_ℓs = Tuple(params.filter_scales_m .* meters)
gaussian_filters = [GaussianFilter(; dims=(1, 2), σ=_FWHM_to_σ(ℓ)) for ℓ in filter_ℓs]  # one reusable filter object per scale

_fields = (u=u_center, v=v_center, w=w_center, b=b)
_filt_pairs = [Symbol("$(n)_ℓ$(round(Int, ℓ/1000))km") => gf(f) for (ℓ, gf) in zip(filter_ℓs, gaussian_filters) for (n, f) in pairs(_fields)]
filtered_fields = (; _filt_pairs...)
#---

@info "Online diagnostics (filtered fields) built"

#+++ Cross-scale KE flux Πₖ and SFS KE dissipation ε_Kˢ, per filter scale
# w is a genuine prognostic variable in this NonhydrostaticModel, with its own momentum equation and
# dissipative dynamics -- unlike the earlier HydrostaticFreeSurfaceModel setup, where w was diagnostic
# and had to be excluded from the KE budget for it to close even in principle. Πₖ is therefore the full
# 3D contraction (dims=(1,2,3)), not horizontal-only: restricting to horizontal-only here would
# reintroduce a different missing term (horizontal-vertical pressure redistribution) in place of the
# free-surface one this model switch removes.
#
# ε_Kˢ (SubFilterKineticEnergyDissipationRate) and εˡ (FilteredKineticEnergyDissipationRate) have no
# `dims` restriction in their public API at all -- both always include w's full contribution via the
# model's actual per-direction viscous fluxes, which is the physically correct behavior now (it was a
# small "phantom" w-diffusion term under the old hydrostatic setup, verified negligible there via a
# smoke test: ~1e-8 relative magnitude, 0.99 spatial correlation with the w-excluded offline formula).
#
# εˡ is included alongside Πₖ/ε_Kˢ for an ongoing investigation into whether the *filtered* (large-scale)
# KE budget (∂ₜK̄ = w̄b̄ᵣ - Πₖ - εˡ, K̄ = ½(ū²+v̄²+w̄²)) is a more robust diagnostic than the SFS budget this
# repo has focused on so far -- unlike ε_Kˢ = filter(ε) - εˡ (a difference of two large, closely-related
# quantities, fragile by construction), εˡ is a single directly-computed term with no cancellation,
# matching how tomchor's own Eady baroclinic-instability example (Oceanostics PR #260) closes its
# coarse-grained KE budget. That example's own setup (NonhydrostaticModel, no free surface, full 3D Πₖ)
# is exactly what this switch adopts; see conversation history for the ongoing investigation into how
# much of the closure gap is numerics vs. the buoyancy-production-term convention (w̄b̄ vs w̄b̄ᵣ).
_ke_pairs = vcat(
    [Symbol("Π_K_ℓ$(round(Int, ℓ/1000))km")  => KineticEnergyCrossScaleFlux(model, gf; dims=(1, 2, 3)) for (ℓ, gf) in zip(filter_ℓs, gaussian_filters)],
    [Symbol("ε_Kˢ_ℓ$(round(Int, ℓ/1000))km") => SubFilterKineticEnergyDissipationRate(model, gf)        for (ℓ, gf) in zip(filter_ℓs, gaussian_filters)],
    [Symbol("ε_l_ℓ$(round(Int, ℓ/1000))km")  => FilteredKineticEnergyDissipationRate(model, gf)         for (ℓ, gf) in zip(filter_ℓs, gaussian_filters)],
)
ke_budget_fields = (; _ke_pairs...)
#---

#+++ Bottom drag work diagnostics: τ̄ (filtered stress) and overline{τ·u_b} (filtered pointwise work), per
# filter scale -- only when --bottom_drag. These are the "primitive" SFS-decomposition ingredients: offline
# assembly (04_sfs_ke_budget.py) combines them into the large-scale term -(τ̄·ū_b) and the SFS term
# -(overline{τ·u_b} - τ̄·ū_b). ū_b/v̄_b need no separate computation here -- they're already the bottom
# z-slice of the filtered u_ℓ/v_ℓ fields built above. τ_x, τ_y, and the raw work τ·u_b are built as full 3D
# expressions from u_center/v_center/w_center for convenience (reusing the same gaussian_filters, which only
# ever touch x,y), but only the bottom z-level (written via the new :bottom output writer below,
# indices=(:,:,1)) is physically meaningful -- horizontal-only filtering and z-slicing commute, so this is
# equivalent to slicing first and filtering the 2D result.
bottom_drag_fields = NamedTuple()
if params.bottom_drag
    τ_x = Cd * sqrt(u_center^2 + v_center^2 + w_center^2) * u_center
    τ_y = Cd * sqrt(u_center^2 + v_center^2 + w_center^2) * v_center
    τu_b_raw = τ_x * u_center + τ_y * v_center  # τ·u_b, pointwise

    _τx_pairs   = [Symbol("τx_b_ℓ$(round(Int, ℓ/1000))km") => gf(τ_x)      for (ℓ, gf) in zip(filter_ℓs, gaussian_filters)]
    _τy_pairs   = [Symbol("τy_b_ℓ$(round(Int, ℓ/1000))km") => gf(τ_y)      for (ℓ, gf) in zip(filter_ℓs, gaussian_filters)]
    _work_pairs = [Symbol("τu_b_ℓ$(round(Int, ℓ/1000))km") => gf(τu_b_raw) for (ℓ, gf) in zip(filter_ℓs, gaussian_filters)]
    global bottom_drag_fields = (; _τx_pairs..., _τy_pairs..., _work_pairs...)
end
#---

outputs = (; ζ, b, pe, PE, u=u_center, v=v_center, w=w_center, filtered_fields..., ke_budget_fields...)

# Smagorinsky's eddy viscosity/diffusivity are spatially/temporally varying fields (unlike the
# 'constant' closure's fixed ν/κ, which are recorded as scalar global attributes below), so they must be
# written out as actual output fields for the offline SFS dissipation diagnostics to use -- matching the
# KH setup's equivalent conditional block for non-constant closures.
if params.closure == "smagorinsky"
    νₑ = viscosity(model)
    κₑ = diffusivity(model, Val(:b))
    outputs = (; outputs..., νₑ, κₑ)
end

using NCDatasets
simulation_name = "bci_Nx$(params.Nx)_Ny$(params.Ny)_Nz$(params.Nz)"
output_filename = "output/$(simulation_name).nc"

output_interval = params.output_interval_hours * hours

# NCDatasets.jl cannot write a raw Bool as a NetCDF attribute at all (errors with "KeyError: key Bool not
# found") -- confirmed directly. Write both `implicit` and `bottom_drag` as Int (0/1) here, in a separate
# variable, so `params` itself keeps genuine Bools for use in conditionals earlier in this file. The
# Python post-processing pipeline reads them back as plain 0/1 (`bool(ds.attrs["implicit"])`,
# `bool(ds.attrs["bottom_drag"])`), same as any other numeric attribute.
netcdf_attributes = (; params..., implicit=Int(params.implicit), bottom_drag=Int(params.bottom_drag))

simulation.output_writers[:fields] =
    NetCDFWriter(model, outputs,
                 schedule = ConsecutiveIterations(TimeInterval(output_interval)),
                 filename = output_filename,
                 array_type = Array{Float64},
                 global_attributes = netcdf_attributes,
                 overwrite_existing = true)

output_filename_2d = "output/$(simulation_name)_surface.nc"
simulation.output_writers[:surface] =
    NetCDFWriter(model, outputs,
                 schedule = TimeInterval(output_interval),
                 filename = output_filename_2d,
                 array_type = Array{Float32},
                 indices = (:, :, grid.Nz),
                 global_attributes = netcdf_attributes,
                 overwrite_existing = true)

if params.bottom_drag
    # ConsecutiveIterations (not plain TimeInterval, unlike :surface) -- matches :fields' own schedule, so
    # this file's time grid aligns with ds_filt's (01_filter_fields.py operates on the :fields output).
    # Confirmed necessary directly: a first attempt with plain TimeInterval produced silent all-NaN results
    # in 04_sfs_ke_budget.py when combining τ̄ (from this file) with ū_b (from ds_filt) on mismatched times.
    output_filename_bottom = "output/$(simulation_name)_bottom.nc"
    simulation.output_writers[:bottom] =
        NetCDFWriter(model, bottom_drag_fields,
                     schedule = ConsecutiveIterations(TimeInterval(output_interval)),
                     filename = output_filename_bottom,
                     array_type = Array{Float32},
                     indices = (:, :, 1),
                     global_attributes = netcdf_attributes,
                     overwrite_existing = true)
end

@info "Output writers configured. Output will be saved to: $(output_filename)"
#---

#+++ Run simulation
@info @sprintf("""
================================================================================
  Baroclinic adjustment (double-front, doubly-periodic-horizontal channel)
================================================================================
  Grid:          Nx=%d, Ny=%d, Nz=%d
  Domain:        Lx=%.1f km, Ly=%.1f km, Lz=%.1f km
  Stop time:     %.1f days
  N²=%.2e s⁻², M²=%.2e s⁻², front width=%.1f km
  Mixed layer:   %.1f m (N²=0 above this depth)
  Latitude:      %.1f
  Advection:     %s
  Closure:       %s%s
  Bottom drag:   %s
================================================================================
""",
    params.Nx, params.Ny, params.Nz,
    params.Lx/1000, params.Ly/1000, params.Lz/1000,
    params.stop_time,
    params.N2, params.M2, params.front_width,
    params.mixed_layer_depth,
    params.latitude,
    params.advection_scheme,
    params.closure,
    params.closure == "scale_aware" ? @sprintf(" (νh=%.4g, νv=%.4g m² s⁻¹, Pe_cell_h=%.4g, Pe_cell_v=%.4g)",
                                                params.nu_h, params.nu_v, params.Pe_cell_h, params.Pe_cell_v) : "",
    params.bottom_drag ? @sprintf("enabled (z0=%.4g m, Cd=%.4g)", params.z0, params.Cd) : "disabled")
@info "Running baroclinic adjustment simulation..."
run!(simulation)
#---
