# Kelvin-Helmholtz instability simulation

# Before anything loads MPI: MPI.jl asks UCX (under the system OpenMPI of a multi-GPU run) to leave SIGSEGV to Julia,
# which stops its threads for GC with it, and not to cache memory types, but from its __init__, after MPItrampoline has
# loaded UCX, which reads its settings as it loads. UCX's handler then caught an idle thread's GC safepoint and aborted
# the run. simulation.pbs exports both for its ranks; setting them here covers any other launch.
get!(ENV, "UCX_ERROR_SIGNALS", "SIGILL,SIGBUS,SIGFPE")
get!(ENV, "UCX_MEMTYPE_CACHE", "no")

using Oceananigans
using CairoMakie
using Printf
using Random
using Logging
using ArgParse
using CUDA: has_cuda_gpu

using Oceananigans.Architectures: on_architecture
using Oceananigans.Grids: topology, znode, minimum_yspacing, xnodes, znodes
import Oceananigans.Utils: actuates_next_iteration

using Oceanostics: PotentialEnergyEquation, KineticEnergyEquation, FlowDiagnostics, GaussianFilter, StrainRateTensor, SubFilterKineticEnergyEquation
using Oceanostics: AvailablePotentialEnergyCrossScaleFlux
using Oceanostics: SubFilterKineticEnergy
using Oceanostics: SubFilterAvailablePotentialToKineticEnergyConversion
using Oceanostics.AvailablePotentialEnergyEquation: reference_height, ThreeDimensionalSort, HeavisideIntegral, ProfileLookup
using Oceanostics.AvailablePotentialEnergyEquation: AvailablePotentialEnergyDissipationRate
using Oceanostics.FilteredAvailablePotentialEnergyEquation: FilteredAvailablePotentialEnergy,
      FilteredAvailablePotentialEnergyDissipationRate, FilteredAvailablePotentialToKineticEnergyConversion,
      FilteredAvailablePotentialEnergyDisplacementPotential
using Oceanostics.AvailablePotentialEnergyEquation: BackgroundPotentialEnergy, AvailablePotentialEnergy, ReferenceBuoyancyAnomaly
using Oceanostics.ProgressMessengers

@info "Finished loading packages"

include("utils.jl")
include("online_diagnostics.jl")   # the one budget term Oceanostics does not provide
include("filtered_reference_profile.jl")   # ⟨b✶⟩, the vertically filtered reference profile, on a coarse column
include("distributed_diagnostics.jl")   # the sorted column and the filter's x-pass, which need the whole domain

#+++ Parse command-line arguments
let s = ArgParseSettings()
    @add_arg_table! s begin
        "--Nz"
            help = "Number of vertical grid points (default: 128 on CPU, 512 on GPU)"
            arg_type = Int
            required = false
            default = has_cuda_gpu() ? 512 : 128

        "--U"
            help = "Velocity profile amplitude U₀ (default: 1.0)"
            arg_type = Float64
            required = false
            default = 1

        "--stop_time"
            help = "Simulation stop time (default: 200.0)"
            arg_type = Float64
            required = false
            default = 200.0

        "--Re0"
            help = "Base Reynolds number: Re = Re0 Nz^(4/3) (default: 0.1)"
            arg_type = Float64
            required = false
            default = 0.1

        "--Ri"
            help = "Base Richardson number (default: 0.1)"
            arg_type = Float64
            required = false
            default = 0.1

        "--Pr"
            help = "Prandtl number (default: 1.0)"
            arg_type = Float64
            required = false
            default = 1

        "--h"
            help = "Buoyancy layer half-width relative to velocity half-width (default: 1.0, i.e. same scale for both)"
            arg_type = Float64
            required = false
            default = 1

        "--perturbation_amplitude"
            help = "Perturbation amplitude (default: 0.05)"
            arg_type = Float64
            required = false
            default = 0.05

        "--output_interval"
            help = "Time between outputs, for both writers (default: 2.0); with --offline_check, each 3D output is a \
                    consecutive-iteration pair"
            arg_type = Float64
            required = false
            default = 2.0

        "--filter_ls"
            help = "Filter length scales ℓ (FWHM) for the online sub-filter budgets (filtered fields and every term of the \
                    SFS KE and APE budgets). postprocessing/01_online_budgets.py reads these scales (default: 1 7)"
            arg_type = Int
            nargs = '+'
            default = [1, 7]

        "--save_tensors"
            help = "Also output the strain-rate (S̄ⁱʲ) and sub-filter stress (τⁱʲ) tensor components at each filter scale \
                    (for online-vs-offline validation). These are full 3D fields, so off by default to keep production \
                    output lean."
            action = :store_true

        "--save_sorted"
            help = "Also output the validation-only view of the Winters et al. (1995) sorted reference state: the reference \
                    height z✶ under the two model-grid sorting methods, the sorted column b✶(z✶) itself, and ∫E_b. The SFS \
                    budgets themselves are always written; this adds two 3D fields and two extra sorts per output, so it is \
                    off by default (for postprocessing/validation/inv06-inv07)."
            action = :store_true

        "--offline_check"
            help = "Also write the record one time step after each 3D output (ConsecutiveIterations pairs), which the offline \
                    pipeline differences for its own tendencies in `pytest --offline-check`. The online tendencies come from \
                    TimeDerivative and need no pair, so it is off by default, at half the 3D output."
            action = :store_true

        "--ranks"
            help = "Number of MPI ranks, one GPU each, the domain is split across in x (default: 1, a single GPU or CPU \
                    and no MPI). Launch with as many MPI processes (`mpiexec -n R`); each rank writes its own files \
                    (<name>_rank<r>.nc), which merge_rank_output.jl stitches into the single-GPU layout."
            arg_type = Int
            default = 1
    end
    global parsed_args = parse_args(s, as_symbols=true)
end
# The control flags go into `params`, and so into both output files' global attributes, as 0/1 Ints (NetCDF has
# no Bool attribute). The post-processing reads them back from there (`output_flag` in postprocessing/src/aux00_utils.py)
# rather than inferring them from a file's contents: `offline_check` says whether the 3D output comes in
# consecutive-iteration pairs, `save_tensors` whether the S̄ⁱʲ/τⁱʲ components are there, `save_sorted` whether the
# sorted-state view is. filter_ls is a vector (the online filter scales, encoded in the output variable names as
# `_ℓ<ℓ>`), so it stays out of `params`.
save_tensors = pop!(parsed_args, :save_tensors)
save_sorted = pop!(parsed_args, :save_sorted)
offline_check = pop!(parsed_args, :offline_check)
filter_ls = pop!(parsed_args, :filter_ls)
params = (; parsed_args..., save_tensors=Int(save_tensors), save_sorted=Int(save_sorted), offline_check=Int(offline_check))
#---

#+++ Define simulation parameters
# Theoretical most unstable wavenumber for the KH instability taken from
# Kaminski and Smyth (2019): https://doi.org/10.1016/j.ocemod.2019.04.005
# which in turn refers to Miles (1961).
# We refer to Michalke (1964)'s resuts: k_max · δ_u = 0.4446 which seems to also match.
let
    k_max = 0.4446 / params.h
    λ_max = 2π / k_max

    Lx = λ_max
    Ly = λ_max / 3
    Lz = 25 * params.h
    Re₀ = params.Re0
    B₀ = params.U^2 * params.Ri / params.h
    global params = (; params..., k_max, λ_max, Lx, Ly, Lz, B₀, Re₀)
end
@info @sprintf("Most unstable KH wavenumber: k_max = %.4f  (λ_max = %.2f, Lx = %.1f)",
               params.k_max, params.λ_max, params.Lx)
#---

#+++ Create grid
x_aspect_ratio = 1   # Δx / Δz ratio
y_aspect_ratio = 1   # Δy / Δz ratio

@info "Cell aspect ratio: Δx/Δz = $(x_aspect_ratio), Δy/Δz = $(y_aspect_ratio)"

# Calculate horizontal resolutions based on aspect ratios
Nx = round(Int, params.Nz * (params.Lx / params.Lz) / x_aspect_ratio)
Ny = round(Int, params.Nz * (params.Ly / params.Lz) / y_aspect_ratio)

# Adjust grid sizes to be factorizable by 2, 3, and 5 (for FFT performance)
Nx = closest_factor_number((2, 3, 5), Nx)
Ny = closest_factor_number((2, 3, 5), Ny)

params = (; params..., Nx, Ny)

# One GPU (or the CPU) and no MPI by default. `--ranks R` splits the domain into R x-slabs, one per MPI rank and GPU
# (distributed_diagnostics.jl says why x only): R must divide Ny, for the FFT pressure solver's transposes, and Nx, for
# the filter's gather of the slabs. Under mpiexec the number of processes must be R as well: with `--ranks 1` every
# process would run the whole simulation and write the same file.
if has_cuda_gpu()
    child_arch = GPU()
else
    @warn "No CUDA GPU detected. Running on CPU."
    child_arch = CPU()
end

launched_processes = parse(Int, get(ENV, "OMPI_COMM_WORLD_SIZE", get(ENV, "PMI_SIZE", "1")))
launched_processes == params.ranks ||
    error("launched as $launched_processes MPI processes, but --ranks is $(params.ranks); they must match")

if params.ranks > 1
    params.Nx % params.ranks == 0 && params.Ny % params.ranks == 0 ||
        error("--ranks $(params.ranks) must divide both Nx = $(params.Nx) and Ny = $(params.Ny)")
    arch = Distributed(child_arch; partition = Partition(x = params.ranks))
    rank = arch.local_rank
else
    arch = child_arch
    rank = 0
end

# Every rank takes part in every collective (reductions, halo exchanges, the gathers), the progress messages included,
# but only rank 0 prints them: the others keep their warnings and errors.
rank == 0 || global_logger(ConsoleLogger(stderr, Logging.Warn))

global_topology = (Periodic, Periodic, Bounded)

grid = RectilinearGrid(arch; size=(params.Nx, params.Ny, params.Nz),
                       x=(-params.Lx/2, params.Lx/2),
                       y=(-params.Ly/2, params.Ly/2),
                       z=(-params.Lz/2, params.Lz/2),
                       topology=global_topology)
#---

#+++ Define Reynolds number, viscosity and diffusivity
let
    Re = params.Re₀ * params.Nz^(4/3)   # Kolmogorov scaling: Δ ∝ η ∝ Re^(-3/4) at fixed Lz
    ν = params.U * params.h / Re
    κ = ν / params.Pr
    global params = merge(params, (; ν, κ, Re))
end
#---

#+++ Create model
model = NonhydrostaticModel(grid;
                            advection = Centered(order=4),
                            closure = ScalarDiffusivity(ν=params.ν, κ=params.κ),
                            buoyancy = BuoyancyTracer(),
                            tracers = :b)
u, v, w = model.velocities
b = model.tracers.b
#---

#+++ Define initial conditions: shear flow with stratification and perturbation
shear_flow(x, z) = params.U * tanh(z / params.h) # Base shear flow
stratification(x, z) = params.B₀ * tanh(z / params.h) # Base stratification

# Small perturbation to trigger instability, at w's nodes: A |n| exp(-z²) sin(k_max x - π), with n a standard normal
# drawn per point. The noise is drawn once for the whole domain from its own seeded generator, and every point takes the
# draw of its global index, so any decomposition, one GPU or R ranks, starts from the same w. (Drawn inside `set!`,
# every rank would repeat one sequence over its own slab, and on one GPU the draws would follow the CPU threads that
# evaluate the function.) The two-argument `set!` hands each rank its part of the global array.
let
    global_grid = RectilinearGrid(CPU(); size=(params.Nx, params.Ny, params.Nz), x=(-params.Lx/2, params.Lx/2),
                                  y=(-params.Ly/2, params.Ly/2), z=(-params.Lz/2, params.Lz/2), topology=global_topology)
    x = reshape(xnodes(global_grid, Center()), :, 1, 1)
    z = reshape(znodes(global_grid, Face()), 1, 1, :)
    noise = abs.(randn(Xoshiro(546), params.Nx, params.Ny, params.Nz + 1))
    set!(model.velocities.w, @. params.perturbation_amplitude * noise * exp(-z^2) * sin(x * params.k_max - π))
end

# Set initial conditions; setting the model also projects the velocity, the w above included, onto a divergence-free one
uᵢ(x, y, z) = shear_flow(x, z)
bᵢ(x, y, z) = stratification(x, z)
set!(model, u=uᵢ, b=bᵢ)
#---

#+++ Setup simulation
#+++ Set initial Δt to 10% of the CFL condition using params.U
Δx = minimum_xspacing(grid)
initial_Δt = 0.1 * Δx / params.U
simulation = Simulation(model, Δt=initial_Δt, stop_time=params.stop_time)
#---

#+++ Add progress messenger
walltime_per_timestep = StepDuration(with_prefix=false)
walltime = Walltime()

Δx = minimum_xspacing(grid)

ε = KineticEnergyEquation.DissipationRate(model)
# The horizontal average as an integral over the area: on a `Distributed` grid Oceananigans' (0.113.3) `Average`
# divides the sum over every rank by one rank's cell count, and so comes out R times too large.
ε̄ = Field(Field(Integral(ε, dims=(1, 2))) / (params.Lx * params.Ly))

#+++ Minimum Kolmogorov scale, after Kaminski & Smyth (2019, JFM 862, 639-658, doi:10.1017/jfm.2018.973)
# L_K is built from the horizontally averaged dissipation ε̄(z) evaluated at the height where it
# peaks, i.e. the smallest Kolmogorov scale anywhere in the (x, y)-averaged profile — not the
# pointwise minimum over the field, which no DNS resolution criterion refers to. The criterion applied here is
# 1.2 L_K ≥ Δx, so the ratio reported below is ≥ 1 while the run is resolved.
#
# Where the factor comes from. Kaminski & Smyth's criterion is 2.5 L_K ≥ Δx, the Smyth-group rule set out in Smyth &
# Moum (2000, Phys. Fluids 12, 1327-1342, doi:10.1063/1.870385, §II), who set the grid spacing to 2.5 times the minimum
# Batchelor scale L_B = L_K Pr^(-1/2) (L_K at the Pr = 1 used here) and validate it against the Nasmyth and
# Panchev-Kesich spectra. They cite Moin & Mahesh (1998, Annu. Rev. Fluid Mech. 30, 539-578,
# doi:10.1146/annurev.fluid.30.1.539): a DNS needs Δ "no greater than a few (3-6) times" L_K, since the smallest resolved
# scale need only be O(η); most of the dissipation occurs at scales well above η (the dissipation spectrum peaks near
# 24η, Pope 2000, Turbulent Flows, §6.5.4). In wavenumber terms 2.5 L_K ≥ Δx is k_max L_K ≥ π/2.5 ≈ 1.26, next to
# Pope's (§9.1.2) k_max η ≥ 1.5 for spectral DNS, beyond which 0.2% of the dissipation remains.
#
# Those are spectral-code rules, exact up to the Nyquist wavenumber π/Δx. Oceananigans differences over one cell, whose
# modified wavenumber 2 sin(kΔ/2)/Δ is within 10% of k only up to kΔ ≈ 1.6, half the Nyquist wavenumber (Moin & Mahesh,
# §2.2: to differentiate a 3η wave to 5%, a Fourier scheme needs Δ = 1.5η, fourth-order central 0.55η, second-order
# 0.26η). Scaling the spectral rule by that effective resolution gives 2.5/2 ≈ 1.2. Measured on Pope's model spectrum
# (§6.5.3, β = 5.2, c_η = 0.4), the second-order viscous operator loses 9.8% of the dissipation at Δx = 2.5 L_K and 2.5%
# at 1.2 L_K (3.5% and 0.8% at the spectral peak), with 37% and 5% of the dissipation at scales shorter than 6 cells,
# where the fourth-order advection starts to lose accuracy. That model spectrum is for 3D turbulence with an inertial
# range, so a convergence test at fixed Re is the real arbiter, and the factor is a floor rather than a guarantee.
#
# Read the ratio over the turbulent phase (the billow's breakdown, t ≈ 60-70 here), not at t = 0: the initial w
# perturbation carries `abs(randn())` per grid point, grid-scale noise whose gradients make ε̄ 3-8× the laminar shear's
# dissipation until viscosity removes it within a time unit, and that sets the minimum over the whole run.
# L_K goes in the output writer as a scalar time series, which is what Kaminski & Smyth's figure 8(d) plots.
ε̄_max = Field(Reduction(maximum!, ε̄, dims=(1, 2, 3)))
L_K = (params.ν^3 / ε̄_max)^(1/4) |> Field

# compute! chains down through ε̄_max to ε̄, so this reads the current state rather than whatever the
# output writer last left there. L_K is a (1, 1, 1) field, so `maximum` just reads its one value off
# the device (avoiding scalar indexing on the GPU).
function kolmogorov_resolution(sim)
    compute!(L_K)
    return @sprintf("1.2L_K/Δx = %.2f", 1.2 * maximum(L_K) / Δx)
end
#---


# The message is built on every rank and only then logged: it reduces over the whole domain (the CFL number, max|w|,
# L_K), which every rank must take part in, and a disabled `@info` (every rank but 0) never evaluates its message.
function progress(simulation)
    message = (PercentageProgress(with_prefix=false, with_units=false)
               + walltime
               + TimeStep()
               + "CFL = " * AdvectiveCFLNumber(with_prefix=false)
               + "Diffusive CFL = " * DiffusiveCFLNumber(with_prefix=false)
               + MaxWVelocity()
               + "step dur = " * walltime_per_timestep
               + kolmogorov_resolution
               )(simulation)
    @info message
end
simulation.callbacks[:progress] = Callback(progress, IterationInterval(20))
#---

#+++ Add TimeStepWizard for adaptive timestepping
N²_max = ∂z(b) |> Field |> maximum
max_Δt = 0.2 / √N²_max # Max timestep is 0.2 times the buoyancy period
conjure_time_step_wizard!(simulation, IterationInterval(1);
                          max_change=1.05,
                          cfl=0.8,
                          diffusive_cfl=0.3,
                          min_Δt=1e-4,
                          max_Δt)
#---
#---

#+++ Add output writer
u_center = @at (Center, Center, Center) u
v_center = @at (Center, Center, Center) v
w_center = @at (Center, Center, Center) w

Ri_field = FlowDiagnostics.GradientRichardsonNumber(model)
S_field  = FlowDiagnostics.StrainRateTensorModulus(model)

ρ₀ = 1025 # kg/ m^3
pe = ρ₀ * PotentialEnergyEquation.PotentialEnergy(model)

PE = Integral(pe)

vorticity = Field(∂z(u) - ∂x(w))

#+++ Gaussian-filtered u, v, w, b at multiple filter scales for subfilter-scale analysis
# ℓ is the FWHM of the Gaussian kernel; σ = ℓ / (2√(2 ln 2)) is the std dev passed to GaussianFilter
filter_ℓs = Tuple(filter_ls)  # from --filter_ls (default (1, 7))
_FWHM_to_σ(ℓ) = ℓ / (2 * sqrt(2 * log(2)))
#---

#+++ Online cross-scale KE transfer Πₖ and SFS KE dissipation ε_Kˢ  (Oceanostics)
# Computed at each filter scale ℓ (coarse-graining framework of Aluie et al. 2018, JPO):
#   Πₖ   = -τⁱʲ S̄ⁱʲ        cross-scale (resolved → subfilter) KE flux      [KineticEnergyCrossScaleFlux]
#   ε_Kˢ = filter(ε) - εˡ   sub-filter-scale viscous dissipation           [SubFilterKineticEnergyDissipationRate]
# where ε is the total viscous dissipation (KineticEnergyEquation.DissipationRate) and εˡ is the dissipation
# of the filtered flow (FilteredKineticEnergyDissipationRate). SubFilterKineticEnergyDissipationRate assembles
# εˢ = filter(ε) - εˡ in one diagnostic (previously done by hand). This equals 2ν Σ[filter(SⁱʲSⁱʲ) - filter(Sⁱʲ)²]
# ≥ 0, exactly what calculate_sfs_ke_dissipation computes offline in postprocessing/src/aux02_ke_functions.py.
# The Gaussian filter reproduces the offline post-processing filter (periodic x and y, edge-extended z, 4σ
# truncation — scipy gaussian_filter1d's default; Oceanostics truncates at 2σ). Both are per unit mass (m² s⁻³).
to_center(ψ) = @at (Center, Center, Center) ψ

# Per-direction Gaussian stencil widths matching scipy's truncate=4 (radius = ⌊4σ/Δ + ½⌋ cells), capped at one
# period along a periodic direction, which is as far as the filter's wrapping reaches (the offline filter caps
# identically). The sizes and topology are the whole domain's: on a `Distributed` grid `size(grid)` is one rank's
# slab, and its x is `FullyConnected` rather than `Periodic`.
_stencil_radius(σ, Δ, N, periodic) = min(max(1, floor(Int, 4σ / Δ + 0.5)), periodic ? N : typemax(Int))
function _filter_N(σ)
    Δs = (minimum_xspacing(grid), minimum_yspacing(grid), minimum_zspacing(grid))
    Ns = (params.Nx, params.Ny, params.Nz)
    return ntuple(d -> 2 * _stencil_radius(σ, Δs[d], Ns[d], global_topology[d] === Periodic) + 1, 3)
end

# One reusable, offline-matched filter per scale, shared by the KE diagnostics here, the written filtered
# fields, and the sub-filter APE terms below.
#
# It returns one Field per operand, built the first time that operand is filtered and handed back every time after:
# Oceanostics' diagnostics each filter the velocities and the buoyancy they need themselves, so ū and v̄ would
# otherwise exist three times per scale, w̄ six times and b̄ four, each a full field. Every diagnostic wraps what the
# filter returns in `Field(...)`, which of a Field is the Field itself, so all of them share one.
function matched_filter(ℓ)
    σ = _FWHM_to_σ(ℓ)
    gf = GaussianFilter(; dims=(1, 2, 3), σ, boundary=:edge, N=_filter_N(σ))
    filtered = IdDict{Any, Any}()
    return ψ -> get!(() -> Field(gf(ψ)), filtered, ψ)
end
filters = Dict(ℓ => matched_filter(ℓ) for ℓ in filter_ℓs)

# Oceanostics wraps every composed sub-filter expression in a KernelFunctionOperation carrying a trivial
# passthrough kernel (see `subfilter_ape_ccc` and friends upstream). That is not cosmetic: a writer is
# specialised on the type of its whole output NamedTuple, and an unwrapped `Field(gf(Field(a))) - b` is a
# deep BinaryOperation nest. Handing a writer several of those sent LLVM's instruction selector quadratic
# (DAGCombiner::CombineToPostIndexedLoadStore -> hasPredecessorHelper) and stalled compilation for >45 min
# at Nz=32. Wrapping collapses each term to one flat KernelFunctionOperation, as the built-in ones already are.
@inline _passthrough_ccc(i, j, k, grid, a) = @inbounds a[i, j, k]
flatten(op) = KernelFunctionOperation{Center, Center, Center}(_passthrough_ccc, grid, op)

# A TimeDerivative only an output writer reads. The writer builds its own derivative from the operand
# (`construct_output(::TimeDerivative, ...)` in Oceananigans), so the two full fields `TimeDerivative(operand)`
# allocates would never be written; this one points them at the operand instead.
function writer_time_derivative(operand::Field)
    LX, LY, LZ = location(operand)
    G, T, O = typeof(operand.grid), eltype(operand), typeof(operand)
    return TimeDerivative{LX, LY, LZ, G, T, O, O, Float64, Float64}(operand, operand, operand, 0.0, 1.2, operand.grid)
end

# The written filtered fields use the same offline-matched filter as every sub-filter term below, so
# `u_ℓ<ℓ>`, `v_ℓ<ℓ>`, `w_ℓ<ℓ>`, `b_ℓ<ℓ>` are the fields those terms are built from (Oceanostics' own defaults
# truncate at 2σ and shrink the stencil at the walls, which is not what the offline pipeline does).
# The 3D writer reads each through a view of its interior, which costs no memory. The 2D writer gets it behind a
# passthrough, so that its j=1 slice reads the filtered field rather than filtering again for the slice.
_fields = (u=u_center, v=v_center, w=w_center, b=b)
_filt_pairs = [Symbol("$(n)_ℓ$(ℓ)") => filters[ℓ](f) for ℓ in filter_ℓs for (n, f) in pairs(_fields)]
filtered_fields    = (; _filt_pairs...)
filtered_fields_2d = map(flatten, filtered_fields)

_ke_pairs = Pair{Symbol, Any}[]
for ℓ in filter_ℓs
    gf = filters[ℓ]

    Πₖ   = SubFilterKineticEnergyEquation.KineticEnergyCrossScaleFlux(model, gf; dims=(1, 2, 3))
    ε_Ks = SubFilterKineticEnergyEquation.SubFilterKineticEnergyDissipationRate(model, gf) # εˢ = filter(ε) - εˡ
    # Kˢ = filter(K) - Kˡ = ½τⁱⁱ, the energy the budget below is of: one Field for its output, its integral and its tendency
    K_s  = Field(SubFilterKineticEnergy(model, gf))
    push!(_ke_pairs, Symbol("Π_K_ℓ$(ℓ)")        => Πₖ,   Symbol("Π_K_ℓ$(ℓ)_int")  => Integral(Πₖ),
                     Symbol("ε_Ks_ℓ$(ℓ)")       => ε_Ks, Symbol("ε_Ks_ℓ$(ℓ)_int") => Integral(ε_Ks),
                     Symbol("K_s_ℓ$(ℓ)")        => K_s,  Symbol("K_s_ℓ$(ℓ)_int")  => Integral(K_s),
                     Symbol("dKs_dt_ℓ$(ℓ)")     => writer_time_derivative(K_s),
                     Symbol("dKs_dt_ℓ$(ℓ)_int") => TimeDerivative(Integral(K_s)))

    # Individual strain (S̄ⁱʲ) and sub-filter stress (τⁱʲ) components at cell centers, for the
    # online-vs-offline validation in postprocessing/validation/. Full 3D fields → gated behind
    # --save_tensors to keep production output lean.
    if save_tensors
        ū = Field(gf(u)); v̄ = Field(gf(v)); w̄ = Field(gf(w))
        S̄ = StrainRateTensor(grid, ū, v̄, w̄; dims=(1, 2, 3))      # strain of the filtered velocity
        τ = SubFilterKineticEnergyEquation.subfilter_stress_tensor(model, gf; dims=(1, 2, 3))   # τⁱʲ = filter(uⁱuʲ) - ūⁱūʲ
        for (ij, Sᵢⱼ, τᵢⱼ) in ((11, S̄.S₁₁, τ.τ₁₁), (22, S̄.S₂₂, τ.τ₂₂), (33, S̄.S₃₃, τ.τ₃₃),
                               (12, S̄.S₁₂, τ.τ₁₂), (13, S̄.S₁₃, τ.τ₁₃), (23, S̄.S₂₃, τ.τ₂₃))
            push!(_ke_pairs, Symbol("S$(ij)_ℓ$(ℓ)") => flatten(to_center(Sᵢⱼ)), Symbol("tau$(ij)_ℓ$(ℓ)") => flatten(to_center(τᵢⱼ)))
        end
    end
end
ke_transfer_fields = (; _ke_pairs...)
#---

#+++ Online Winters et al. (1995) sorted reference state and the sub-filter APE budget  (Oceanostics)
# Sorting the buoyancy field adiabatically into its minimum-PE state assigns every parcel a reference
# height z✶. The simulation does this once per output, on the GPU, with `sorted_column` (distributed_diagnostics.jl):
# the sorted column itself, N cells on a Flat cross-section, which is the reference profile b✶(z✶) every APE term
# below is measured against. It is what Oceanostics' `VerticalSort` builds, but of the whole domain on every rank of a
# distributed run, and without the column halos that cost VerticalSort 9× its data.
# The two model-grid methods describe the same reference state and differ only in where they put cells of
# *equal* buoyancy; they are validation outputs (`inv06`), gated behind --save_sorted (one GPU only):
#   ThreeDimensionalSort  z✶ on the model grid; tied cells take consecutive slots (z✶ spreads over a cell)
#   HeavisideIntegral     z✶ on the model grid; tied cells share their layer's mid-height (Winters eq. 11)
#
# The column lives on its own grid, which a single NetCDFWriter handles alongside the model grid, as the
# lock_release example upstream does. Holding two grids does make the writer disambiguate: every dimension
# gets a suffix (z_aac → z_aac_grid1 for the model grid, _grid2 for the column) and the grid metadata groups
# get a matching prefix. The offline code is written against the plain names, so `load_dataset_and_grid`
# strips the model grid's suffix at load time (`strip_grid_suffix` in postprocessing/src/aux00_utils.py).
#
# The 3D writer's schedule, defined here so the time derivatives inside R below can update on the iterations
# around its outputs, which include every one of the 2D writer's. The online tendencies (TimeDerivative, below)
# difference each output against the iteration before it on their own, so one record per output time is a
# complete budget statement. --offline_check adds the record one time step after each output: the offline
# pipeline (postprocessing/offline/, the CI cross-check) differences that pair for its own tendencies and reads
# nothing else from it, and it doubles the 3D output, so production runs leave it off.
output_schedule = offline_check ? ConsecutiveIterations(TimeInterval(params.output_interval)) : TimeInterval(params.output_interval)

b✶_1dsort, z✶_1dsort = sorted_column(b)   # b✶ self-recomputing (reading it gathers and sorts); z✶ fixed

# Every APE term shares this one sort: `ProfileLookup` takes the column as an external (b✶, z✶) pair,
# refreshes it on every compute! when it is a Field, and skips the O(N log N) sort. The reference profile's
# own time derivative feeds R; R is the output, not ∂ₜb✶, so no writer advances it: a TimeDerivativeCallback
# does, around each output, as does each scale's ∂ₜ⟨b✶⟩ below.
lookup = ProfileLookup(b✶_1dsort, z✶_1dsort)
∂ₜb✶ = TimeDerivativeCallback(b✶_1dsort, schedule=PrecedingIterations(output_schedule))
simulation.callbacks[:∂ₜb✶] = ∂ₜb✶

# R against the full field's reference height; Rˡ below uses the filtered field's, and Rˢ = filter(R) - Rˡ.
z✶_lookup = reference_height(model, method=lookup)
R_full = ReferenceTendencyCorrection(model, ∂ₜb✶.func, z✶_lookup)

# The online local available potential energy Eₐ = ∫_{z✶}^{z}[b✶(z̃) - b] dz̃ (Holliday & McIntyre 1981),
# per unit mass, the same integral the offline `local_potential_energies_timeseries` builds (`inv07` checks
# the two). It reads the same lookup as every sub-filter term, so it adds no sort; ∫Eₐ with ∫E_b (below,
# under --save_sorted) gives the online TPE = BPE + APE split, which ∫pe closes. It is one Field for its output, its
# integral and every scale's Ē_A below, as the total APE dissipation ε_A is for every scale's ε̃ˢ.
E_a = Field(AvailablePotentialEnergy(model, z✶_lookup))
∫E_a = Integral(E_a)
ε_A = Field(AvailablePotentialEnergyDissipationRate(model, z✶_lookup))

# The sub-filter APE terms use the filtered-reference scale decomposition (Wenegrat, Chor & Barkan
# Eqs. 2.3-2.5, 2.19-2.22). Measuring the resolved reservoir against the unfiltered b✶ instead is the
# g_z = δ(z) horizontal-filter limit: with a kernel that has vertical extent a fluid at rest still
# carries APE against b✶, so that reservoir does not vanish at rest and the remainder goes negative.
# This filter acts in x *and* z, so only ⟨b✶⟩ gives a decomposition into two non-negative reservoirs,
# and the unfiltered form is not written at all.
#
# This set *is* the SFS APE budget: `postprocessing/01_online_budgets.py` reads it into the budget files,
# and the offline pipeline (`postprocessing/offline/`) recomputes it only as the CI cross-check
# (`pytest --offline-check`). Nothing here needs a new Oceanostics diagnostic: the two-argument constructors
# let the two halves of each sub-filter quantity carry *different* reference profiles, composed here
# exactly as R_s is.
_ape_pairs    = Pair{Symbol, Any}[]   # both writers: the terms the panels animation draws
_ape_3d_pairs = Pair{Symbol, Any}[]   # 3D writer only: the two halves of S̃, which the tests read
for ℓ in filter_ℓs
    gf = filters[ℓ]
    b✶_flt = filtered_reference_profile(b✶_1dsort, grid, _FWHM_to_σ(ℓ))   # ⟨b✶⟩ on the column's N slots
    lookup_flt = ProfileLookup(b✶_flt, z✶_1dsort)                         # heights unchanged; only b✶ filtered
    z✶ˡ_flt = reference_height(gf(b); method=lookup_flt)                   # z̃✶(b̄), the inverse of ⟨b✶⟩
    Υ̃ = Field(FilteredAvailablePotentialEnergyDisplacementPotential(model, z✶ˡ_flt))   # z̃✶(b̄) - z: one for Π̃_A and ε̃ˡ

    # τˡ(w, b_r) = filter(w b_r) - w̄ b_rˡ, the sub-filter half of the APE↔KE conversion. It is a *term* in
    # both budgets (+1 in the KE residual, -1 in the APE one) and the field `plot_kelvin_helmholtz_instability.jl`
    # discovers the panel scales from, so do not drop it.
    #
    # It is a reversible exchange, not a source or a sink, so **neither half of the separation has a
    # fixed sign**: filter(w b_r) and w̄ b_rˡ are each of either sign, and so is their difference.
    # Unlike ε_Kˢ (a dissipation) or S̃ (non-negative by construction against ⟨b✶⟩), τˡ admits no
    # positivity check — a negative value here is physics, not a symptom.
    #
    # Reference profile. S̃ is measured against ⟨b✶⟩, so the resolved half must be too:
    # τˡ = filter(w(b - b✶)) - w̄(b̄ - ⟨b✶⟩). Upstream's SubFilterAvailablePotentialToKineticEnergyConversion
    # reads one profile, from `method`, for *both* halves, so no single call gives it: `method=lookup` is off
    # by w̄δ and `method=lookup_flt` by filter(wδ), with δ = b✶ - ⟨b✶⟩. Adding the filtered conversion
    # w̄(b̄ - b✶) and subtracting w̄(b̄ - ⟨b✶⟩) moves the resolved half onto ⟨b✶⟩ and leaves the filtered
    # product alone. It changes the field, not the integral: ∫w̄ f(z) dV = 0 for any f(z), since ∫w dx dy
    # vanishes at every height and filtering preserves that.
    wb_rs = flatten(SubFilterAvailablePotentialToKineticEnergyConversion(model, gf; method=lookup) +
                    FilteredAvailablePotentialToKineticEnergyConversion(model, gf; method=lookup) -
                    FilteredAvailablePotentialToKineticEnergyConversion(model, gf; method=lookup_flt))

    # S̃ = Ē_A - L̃: the filtered full-field APE against b✶, less the filtered field's APE against ⟨b✶⟩.
    # Both halves are written on their own (3D writer) so the offline check can compare each; S̃ itself is one Field
    # for its output, its integral and its tendency.
    Ea_flt = gf(E_a)                                                         # Ē_A
    L      = FilteredAvailablePotentialEnergy(model, z✶ˡ_flt)               # L̃ = Ẽ_A(b̄, z)
    E_as   = Field(flatten(Ea_flt - L))                                      # S̃
    Π_A    = AvailablePotentialEnergyCrossScaleFlux(model, gf, z✶ˡ_flt; dims=(1, 2, 3), upsilon=Υ̃)   # Π̃_A = -τ(uᵢ,b)∂ᵢΥ̃
    ε_As   = flatten(gf(ε_A) - FilteredAvailablePotentialEnergyDissipationRate(model, gf, z✶ˡ_flt; upsilon=Υ̃))   # ε̃ˢ
    # R̃ˡ follows the same reference, so its tendency is ∂ₜ⟨b✶⟩ rather than ∂ₜb✶ (Eq. 2.18).
    ∂ₜb✶_flt = TimeDerivativeCallback(b✶_flt, schedule=PrecedingIterations(output_schedule))
    simulation.callbacks[Symbol("∂ₜb✶_flt_ℓ$(ℓ)")] = ∂ₜb✶_flt
    R_l  = ReferenceTendencyCorrection(model, ∂ₜb✶_flt.func, z✶ˡ_flt)
    R_s  = flatten(gf(R_full) - R_l)

    push!(_ape_pairs, Symbol("ε_As_ℓ$(ℓ)")  => ε_As, Symbol("ε_As_ℓ$(ℓ)_int") => Integral(ε_As),
                      Symbol("Π_A_ℓ$(ℓ)")   => Π_A,  Symbol("Π_A_ℓ$(ℓ)_int")  => Integral(Π_A),
                      Symbol("E_as_ℓ$(ℓ)")  => E_as, Symbol("E_as_ℓ$(ℓ)_int") => Integral(E_as),
                      Symbol("wb_rs_ℓ$(ℓ)") => wb_rs, Symbol("wb_rs_ℓ$(ℓ)_int") => Integral(wb_rs),
                      Symbol("R_s_ℓ$(ℓ)")   => R_s,  Symbol("R_s_ℓ$(ℓ)_int")  => Integral(R_s),
                      Symbol("dEas_dt_ℓ$(ℓ)")     => writer_time_derivative(E_as),
                      Symbol("dEas_dt_ℓ$(ℓ)_int") => TimeDerivative(Integral(E_as)))
    push!(_ape_3d_pairs, Symbol("L_ℓ$(ℓ)") => L, Symbol("Ea_flt_ℓ$(ℓ)") => Ea_flt)
end
sfs_ape_fields = (; _ape_pairs...)

# The 3D writer gets the whole APE budget; the 2D writer gets the per-scale terms and b_r (all model-grid, so
# the 2D file stays single-grid), which is what lets `plot_kelvin_helmholtz_instability.jl` draw the SFS-budget
# panels animation straight from the slice file.
#
# Keep the 2D tuple small. A writer is specialised on the type of its whole output NamedTuple, and growing
# it from ~50 to ~76 heterogeneous entries once sent LLVM's instruction selector quadratic
# (CombineToPostIndexedLoadStore -> hasPredecessorHelper) and stalled compilation of this writer for
# >45 min at Nz=32. `flatten` above is what keeps each term one flat type; do not remove it.
budget_fields = (; E_a, ∫E_a, sfs_ape_fields..., _ape_3d_pairs...)
twod_extra    = (; b_r = ReferenceBuoyancyAnomaly(model, z✶_lookup), sfs_ape_fields...)

# Validation-only outputs: the two model-grid sorts, ∫E_b, and the column itself. `inv06` compares the three
# methods against each other and against the offline sort, `inv07` reads the column. None of them enters the
# budget, so --save_sorted changes no budget number; it costs two extra 3D sorts per output. Both model-grid methods
# sort each rank's slab on its own on a `Distributed` grid, so they are refused there.
save_sorted && params.ranks > 1 && error("--save_sorted needs a single GPU: its model-grid sorts would each sort one rank's slab")
sorted_fields = NamedTuple()
if save_sorted
    z✶_3dsort    = reference_height(model, method=ThreeDimensionalSort())
    z✶_heaviside = reference_height(model, method=HeavisideIntegral())
    ∫E_b = Integral(BackgroundPotentialEnergy(model, z✶_3dsort))
    sorted_fields = (; z✶_3dsort, z✶_heaviside, z✶_1dsort, b✶_1dsort, ∫E_b)
end
#---

outputs = (; ω=vorticity, b, pe, PE, u=u_center, v=v_center, w=w_center, filtered_fields..., ke_transfer_fields...,
           ε̄, ε, L_K, Ri=Ri_field, S=S_field)

using NCDatasets
simulation_name = "khi_Nz$(params.Nz)_Ri$(@sprintf("%.2f", params.Ri))"
# Output lands in $KHAPE_OUTPUT_DIR when set, so an HPC run can write to scratch without symlinking the
# repo directory (which does not work: `output/.gitkeep` is tracked, so git reports it deleted).
output_dir = get(ENV, "KHAPE_OUTPUT_DIR", "output")
mkpath(output_dir)
# With --ranks R every rank writes its own slab to its own files, `<name>_rank<r>.nc` (Oceananigans' NetCDFWriter has
# no notion of ranks: given one filename, every rank would write the same file); merge_rank_output.jl stitches them
# into the single-GPU layout afterwards.
rank_suffix = params.ranks > 1 ? "_rank$(rank)" : ""
output_filename = joinpath(output_dir, "$(simulation_name)$(rank_suffix).nc")

if !(model.closure isa ScalarDiffusivity)
    ν = viscosity(model)
    κ = diffusivity(model, Val(:b))
    outputs = (; outputs..., ν, κ)
end

# The 2D writer's tuple: the same outputs, with the filtered fields behind their passthroughs (see `filtered_fields_2d`)
outputs_2d = (; outputs..., filtered_fields_2d...)

# The 3D writer updates the TimeDerivatives among its outputs, and R's callbacks theirs, on PrecedingIterations
# of `output_schedule`, which falls back to every iteration for a schedule it cannot anticipate, as the
# ConsecutiveIterations of --offline_check is upstream: the writer would filter Kˢ and S̃, and the callbacks sort
# the column, on every time step. Its actuations are the parent's plus the iterations right after, whose preceding
# iteration is already an actuation, so only the parent's next actuation needs anticipating. This belongs upstream.
# (The plain TimeInterval of the default is anticipated upstream and needs nothing here.)
actuates_next_iteration(schedule::ConsecutiveIterations, clock, growth) = actuates_next_iteration(schedule.parent, clock, growth)

# The model-grid z✶ fields go in the 3D file only; the 2D writer below slices with `indices` for a
# lightweight x–z animation and has no use for them.
simulation.output_writers[:fields] = NetCDFWriter(model, (; outputs..., budget_fields..., sorted_fields...),
                                                  schedule = output_schedule,
                                                  filename = output_filename,
                                                  array_type = Array{Float64},
                                                  global_attributes = params,
                                                  overwrite_files = true)

output_filename_2d = joinpath(output_dir, "$(simulation_name)_2d$(rank_suffix).nc")
simulation.output_writers[:twod_fields] = NetCDFWriter(model, (; outputs_2d..., twod_extra...),
                                                       schedule = TimeInterval(params.output_interval),
                                                       filename = output_filename_2d,
                                                       array_type = Array{Float32},
                                                       indices = (:, 1, :),
                                                       global_attributes = params,
                                                       overwrite_files = true)

@info "Output will be saved to: $(output_filename)"

params.ranks > 1 && validate_staged_filters((; outputs..., budget_fields..., outputs_2d..., twod_extra...))
#---

#+++ Run simulation
show_gpu_status(label = params.ranks > 1 ? "rank $rank of $(params.ranks)" : "")
@info @sprintf("""
================================================================================
  Kelvin-Helmholtz instability simulation
================================================================================
  Grid:          Nx=%d, Ny=%d, Nz=%d
  Ranks:         %d (x-slabs of %d)
  Domain:        Lx=%.1f, Ly=%.1f, Lz=%.1f
  Stop time:     %.1f
  Richardson:    Ri = %.4f
  Reynolds:      Re = %.1f  (Re₀ = %.2e)
  Prandtl:       Pr = %.1f
  Viscosity:     ν  = %.2e
  Diffusivity:   κ  = %.2e
  KH wavenumber: k_max = %.4f  (λ_max = %.2f)
================================================================================
""",
    params.Nx, params.Ny, params.Nz,
    params.ranks, params.Nx ÷ params.ranks,
    params.Lx, params.Ly, params.Lz,
    params.stop_time,
    params.Ri,
    params.Re, params.Re₀,
    params.Pr,
    params.ν,
    params.κ,
    params.k_max, params.λ_max)
@info "Running Kelvin-Helmholtz instability simulation..."
run!(simulation)
#---

#+++ Plot results
# A distributed run's 2D output is one file per rank until merge_rank_output.jl stitches them together, so
# simulation.pbs draws its animation after the merge.
if params.ranks == 1
    @info "Creating animation..."
    plot_filepath = output_filename_2d
    include("plot_kelvin_helmholtz_instability.jl")
end
#---
