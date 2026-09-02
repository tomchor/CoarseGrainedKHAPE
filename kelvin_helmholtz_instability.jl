# Kelvin-Helmholtz instability simulation
using Oceananigans
using CairoMakie
using Printf
using Random
using ArgParse
using CUDA: has_cuda_gpu
using Oceananigans.Architectures: on_architecture
using Oceanostics: PotentialEnergyEquation, KineticEnergyEquation, FlowDiagnostics, GaussianFilter, StrainRateTensor, SubFilterKineticEnergyEquation
using Oceanostics: SubFilterAvailablePotentialEnergyDissipationRate, AvailablePotentialEnergyCrossScaleFlux
using Oceanostics: SubFilterAvailablePotentialEnergy, SubFilterKineticEnergy
using Oceanostics: SubFilterAvailablePotentialToKineticEnergyConversion
using Oceananigans.OutputWriters: TimeDerivative
using Oceanostics.AvailablePotentialEnergyEquation: reference_height, reference_buoyancy, ThreeDimensionalSort, HeavisideIntegral, VerticalSort, ProfileLookup
using Oceanostics.AvailablePotentialEnergyEquation: BackgroundPotentialEnergy, AvailablePotentialEnergy, ReferenceBuoyancyAnomaly
using Oceanostics.ProgressMessengers

@info "Finished loading packages"
Random.seed!(546)

include("utils.jl")
include("online_diagnostics.jl")   # the one budget term Oceanostics does not provide

#+++ Parse command-line arguments
let s = ArgParseSettings()
    @add_arg_table! s begin
        "--Nz"
            help = "Number of vertical grid points (default: 512 on CPU, 4096 on GPU)"
            arg_type = Int
            required = false
            default = has_cuda_gpu() ? 4096 : 256

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
            help = "Base Reynolds number (default: 5e-4)"
            arg_type = Float64
            required = false
            default = 5e-4

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

        "--filter_ls"
            help = "Filter length scales ℓ (FWHM) for the online sub-filter diagnostics (filtered fields, Πₖ, ε_Kˢ, and Π_A/ε_Aˢ under --save_sorted). The offline budget pipeline's --filter-scales must be a subset of these (default: 1 7)"
            arg_type = Int
            nargs = '+'
            default = [1, 7]

        "--save_tensors"
            help = "Also output the strain-rate (S̄ⁱʲ) and sub-filter stress (τⁱʲ) tensor components at each filter scale (for online-vs-offline validation). These are full 3D fields, so off by default to keep production output lean."
            action = :store_true

        "--save_sorted"
            help = "Also output the Winters et al. (1995) sorted reference state: the reference height z✶ under each of the three Oceanostics sorting methods, the sorted buoyancy profile b✶(z✶), the local APE Eₐ, and the cross-scale APE flux Π_A with the sub-filter APE dissipation ε_Aˢ at each online filter scale. Adds a few 3D fields and a full-domain sort per output, so off by default (for online-vs-offline validation)."
            action = :store_true
    end
    global parsed_args = parse_args(s, as_symbols=true)
end
# Keep the save_tensors control flag out of `params` (it is a Bool, which NetCDF can't store as a
# global attribute, and it is not a physical parameter). Likewise filter_ls is a vector (the online
# filter scales, encoded in the output variable names as `_ℓ<ℓ>`), so keep it out of `params` too.
save_tensors = pop!(parsed_args, :save_tensors)
save_sorted = pop!(parsed_args, :save_sorted)
filter_ls = pop!(parsed_args, :filter_ls)
params = (; parsed_args...)
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
if has_cuda_gpu()
    arch = GPU()
    x_aspect_ratio = 1   # Δx / Δz ratio
    y_aspect_ratio = Inf # Δy / Δz ratio
else
    @warn "No CUDA GPU detected. Running on CPU with a coarse grid and high aspect ratio."

    arch = CPU()
    x_aspect_ratio = 2   # Δx / Δz ratio
    y_aspect_ratio = Inf # Δy / Δz ratio
end

@info "Cell aspect ratio: Δx/Δz = $(x_aspect_ratio), Δy/Δz = $(y_aspect_ratio)"

# Calculate horizontal resolutions based on aspect ratios
Nx = round(Int, params.Nz * (params.Lx / params.Lz) / x_aspect_ratio)
Ny = isinf(y_aspect_ratio) ? 1 : round(Int, params.Nz * (params.Ly / params.Lz) / y_aspect_ratio)

# Adjust grid sizes to be factorizable by 2, 3, and 5 (for FFT performance)
Nx = closest_factor_number((2, 3, 5), Nx)
Ny = closest_factor_number((2, 3, 5), Ny)

params = (; params..., Nx, Ny)

grid = RectilinearGrid(arch; size=(params.Nx, params.Ny, params.Nz),
                       x=(-params.Lx/2, params.Lx/2),
                       y=(-params.Ly/2, params.Ly/2),
                       z=(-params.Lz/2, params.Lz/2),
                       topology=(Periodic, Periodic, Bounded))
#---

#+++ Define Reynolds number, viscosity and diffusivity
let
    if grid.Ny == 1
        Re = params.Re₀ * params.Nz^2
    else
        Re = params.Re₀ * params.Nz^(4/3) # Double check this
    end
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
perturbation(x, z) = params.perturbation_amplitude * abs(randn()) * exp(-z^2) * sin(x * params.k_max - π) # Small perturbation to trigger instability

# Set initial conditions
uᵢ(x, y, z) = shear_flow(x, z)
bᵢ(x, y, z) = stratification(x, z)
wᵢ(x, y, z) = perturbation(x, z)
set!(model, u=uᵢ, b=bᵢ, w=wᵢ)
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
ε̄ = Average(ε, dims=(1, 2)) |> Field

#+++ Minimum Kolmogorov scale, following Kaminski & Smyth (2019, JFM 862, 639-658)
# L_K is built from the horizontally averaged dissipation ε̄(z) evaluated at the height where it
# peaks, i.e. the smallest Kolmogorov scale anywhere in the (x, y)-averaged profile — not the
# pointwise minimum over the field, which no DNS resolution criterion refers to. Their criterion is
# 2.5 L_K ≥ Δx, so the ratio reported below is ≥ 1 while the run is resolved.
# L_K goes in the output writer as a scalar time series, which is what their figure 8(d) plots.
ε̄_max = Field(Reduction(maximum!, ε̄, dims=(1, 2, 3)))
L_K = (params.ν^3 / ε̄_max)^(1/4) |> Field

# compute! chains down through ε̄_max to ε̄, so this reads the current state rather than whatever the
# output writer last left there. L_K is a (1, 1, 1) field, so `maximum` just reads its one value off
# the device (avoiding scalar indexing on the GPU).
function kolmogorov_resolution(sim)
    compute!(L_K)
    return @sprintf("2.5L_K/Δx = %.2f", 2.5 * maximum(L_K) / Δx)
end
#---


progress(simulation) = @info (PercentageProgress(with_prefix=false, with_units=false)
                              + walltime
                              + TimeStep()
                              + "CFL = " * AdvectiveCFLNumber(with_prefix=false)
                              + "Diffusive CFL = " * DiffusiveCFLNumber(with_prefix=false)
                              + MaxWVelocity()
                              + "step dur = " * walltime_per_timestep
                              + kolmogorov_resolution
                              )(simulation)
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

Ri_field = FlowDiagnostics.RichardsonNumber(model)
S_field  = FlowDiagnostics.StrainRateTensorModulus(model)

ρ₀ = 1025 # kg/ m^3
pe = ρ₀ * PotentialEnergyEquation.PotentialEnergy(model)

PE = Integral(pe)

vorticity = Field(∂z(u) - ∂x(w))

#+++ Gaussian-filtered u, v, w, b at multiple filter scales for subfilter-scale analysis
# ℓ is the FWHM of the Gaussian kernel; σ = ℓ / (2√(2 ln 2)) is the std dev passed to GaussianFilter
filter_ℓs = Tuple(filter_ls)  # from --filter_ls (default (1, 7))
_FWHM_to_σ(ℓ) = ℓ / (2 * sqrt(2 * log(2)))
_fields = (u=u_center, v=v_center, w=w_center, b=b)
_filt_pairs = [Symbol("$(n)_ℓ$(ℓ)") => GaussianFilter(f; dims=(1, 3), σ=_FWHM_to_σ(ℓ)) for ℓ in filter_ℓs for (n, f) in pairs(_fields)]
filtered_fields = (; _filt_pairs...)
#---

#+++ Online cross-scale KE transfer Πₖ and SFS KE dissipation ε_Kˢ  (Oceanostics)
# Computed at each filter scale ℓ (coarse-graining framework of Aluie et al. 2018, JPO):
#   Πₖ   = -τⁱʲ S̄ⁱʲ        cross-scale (resolved → subfilter) KE flux      [KineticEnergyCrossScaleFlux]
#   ε_Kˢ = filter(ε) - εˡ   sub-filter-scale viscous dissipation           [SubFilterKineticEnergyDissipationRate]
# where ε is the total viscous dissipation (KineticEnergyEquation.DissipationRate) and εˡ is the dissipation
# of the filtered flow (FilteredKineticEnergyDissipationRate). SubFilterKineticEnergyDissipationRate assembles
# εˢ = filter(ε) - εˡ in one diagnostic (previously done by hand). This equals 2ν Σ[filter(SⁱʲSⁱʲ) - filter(Sⁱʲ)²]
# ≥ 0, exactly what calculate_sfs_ke_dissipation computes offline in postprocessing/src/aux02_ke_functions.py.
# The Gaussian filter reproduces the offline post-processing filter (periodic x, edge-extended z, 4σ truncation
# — scipy gaussian_filter1d's default; Oceanostics truncates at 2σ). 2D x–z runs (v ≡ 0) so dims=(1, 3); both
# are per unit mass (m² s⁻³).
to_center(ψ) = @at (Center, Center, Center) ψ

# Per-direction Gaussian stencil widths matching scipy's truncate=4 (radius = ⌊4σ/Δ + ½⌋ cells).
_filter_N(σ) = (2 * max(1, floor(Int, 4σ / minimum_xspacing(grid) + 0.5)) + 1,
                2 * max(1, floor(Int, 4σ / minimum_zspacing(grid) + 0.5)) + 1)

# One reusable, offline-matched filter per scale, shared by the KE diagnostics here and by the
# sub-filter APE dissipation under --save_sorted below.
function matched_filter(ℓ)
    σ = _FWHM_to_σ(ℓ)
    return GaussianFilter(; dims=(1, 3), σ, boundary=:edge, N=_filter_N(σ))
end

_ke_pairs = Pair{Symbol, Any}[]
for ℓ in filter_ℓs
    gf = matched_filter(ℓ)

    Πₖ   = SubFilterKineticEnergyEquation.KineticEnergyCrossScaleFlux(model, gf; dims=(1, 3))
    ε_Ks = SubFilterKineticEnergyEquation.SubFilterKineticEnergyDissipationRate(model, gf) # εˢ = filter(ε) - εˡ
    K_s  = SubFilterKineticEnergy(model, gf)   # Kˢ = filter(K) - Kˡ = ½τⁱⁱ, the energy the budget below is of
    push!(_ke_pairs, Symbol("Π_K_ℓ$(ℓ)")        => Πₖ,   Symbol("Π_K_ℓ$(ℓ)_int")  => Integral(Πₖ),
                     Symbol("ε_Ks_ℓ$(ℓ)")       => ε_Ks, Symbol("ε_Ks_ℓ$(ℓ)_int") => Integral(ε_Ks),
                     Symbol("K_s_ℓ$(ℓ)")        => K_s,  Symbol("K_s_ℓ$(ℓ)_int")  => Integral(K_s),
                     Symbol("dKs_dt_ℓ$(ℓ)")     => TimeDerivative(K_s, model),
                     Symbol("dKs_dt_ℓ$(ℓ)_int") => TimeDerivative(Integral(K_s), model))

    # Individual strain (S̄ⁱʲ) and sub-filter stress (τⁱʲ) components at cell centers, for the
    # online-vs-offline validation in postprocessing/validation/. Full 3D fields → gated behind
    # --save_tensors to keep production output lean.
    if save_tensors
        ū = Field(gf(u)); w̄ = Field(gf(w))
        S̄ = StrainRateTensor(grid, ū, v, w̄; dims=(1, 3))      # strain of the filtered velocity
        τ = SubFilterKineticEnergyEquation.subfilter_stress_tensor(model, gf; dims=(1, 3))   # τⁱʲ = filter(uⁱuʲ) - ūⁱūʲ
        push!(_ke_pairs,
              Symbol("S11_ℓ$(ℓ)")   => to_center(S̄.S₁₁), Symbol("S33_ℓ$(ℓ)")   => to_center(S̄.S₃₃), Symbol("S13_ℓ$(ℓ)")   => to_center(S̄.S₁₃),
              Symbol("tau11_ℓ$(ℓ)") => to_center(τ.τ₁₁), Symbol("tau33_ℓ$(ℓ)") => to_center(τ.τ₃₃), Symbol("tau13_ℓ$(ℓ)") => to_center(τ.τ₁₃))
    end
end
ke_transfer_fields = (; _ke_pairs...)
#---

#+++ Online Winters et al. (1995) sorted reference state  (Oceanostics)
# Sorting the buoyancy field adiabatically into its minimum-PE state assigns every parcel a reference
# height z✶. Offline this is done in Python by 02_sort_density.py (an argsort of the whole field per
# timestep, held in host RAM and written out at 2× the raw field size); done here it is one GPU sort
# per output. The three methods describe the same reference state and agree on every volume integral,
# but differ in where they put cells of *equal* buoyancy and on what grid they answer:
#   ThreeDimensionalSort  z✶ on the model grid; tied cells take consecutive slots (z✶ spreads over a cell)
#   HeavisideIntegral     z✶ on the model grid; tied cells share their layer's mid-height (Winters eq. 11)
#   VerticalSort          the sorted column itself, on a 1×1×N grid → the reference profile b✶(z✶)
# All three are emitted so postprocessing/validation/inv06_compare_sorted_profiles.py can compare them
# against each other and against the offline sort. Note the offline pipeline sorts the *z-padded* domain
# (load_dataset_and_grid doubles the height with edge values), so the two do not sort the same field
# near the top and bottom boundaries — quantifying that is part of what inv06 checks.
#
# Only the column is a reference *profile* as written. For the two model-grid methods `reference_buoyancy`
# is the model's own `b`, which is already an output, so their profiles are recovered by pairing z✶ with b
# and ordering by z✶ — the same thing the lock_release example in the Oceanostics PR does.
sorted_fields = NamedTuple()
twod_extra = NamedTuple()   # panel fields the 2D writer adds under --save_sorted
if save_sorted
    z✶_3dsort    = reference_height(model, method=ThreeDimensionalSort())
    z✶_heaviside = reference_height(model, method=HeavisideIntegral())


    # The column lives on its own 1×1×N grid (N = Nx·Ny·Nz), which a single NetCDFWriter handles
    # alongside the model grid, as the lock_release example upstream does. Holding two grids does make
    # the writer disambiguate: every dimension gets a suffix (z_aac → z_aac_grid1 for the model grid,
    # _grid2 for the column) and the grid metadata groups get a matching prefix. The offline pipeline
    # is written against the plain names, so `load_dataset_and_grid` strips the model grid's suffix at
    # load time (`strip_grid_suffix` in postprocessing/src/aux00_utils.py) and everything downstream
    # is unaffected; the column's variables keep their own suffix and are read by inv06.
    z✶_1dsort = reference_height(model, method=VerticalSort())
    b✶_1dsort = reference_buoyancy(z✶_1dsort)   # self-recomputing; writing it triggers the sort

    # Online local available potential energy (Oceanostics PR #274). AvailablePotentialEnergy now
    # computes the Holliday & McIntyre (1981) local APE density Eₐ = ∫_{z✶}^{z}[b✶(z̃) - b] dz̃, the same
    # positive-definite integral the offline pipeline builds in local_potential_energies_timeseries
    # (its `ape` field): with b = g(ρ₀-ρ)/ρ₀ the two are identical, per unit mass (m² s⁻²), no ρ₀/sign
    # conversion. Reuse the ThreeDimensionalSort z✶ above so the sort is shared, not repeated. Eₐ (the
    # local field) is validated against the offline `ape` by inv07; ∫Eₐ and ∫E_b give the online
    # TPE = BPE + APE split, which ∫pe (already written) closes. E_b's local field is the trivial -bz✶,
    # so only its integral is emitted.
    E_a = AvailablePotentialEnergy(model, z✶_3dsort)
    ∫E_a = Integral(E_a)
    ∫E_b = Integral(BackgroundPotentialEnergy(model, z✶_3dsort))

    # Sub-filter APE dissipation ε_Aˢ = filter(ε_A) - ε_Aˡ, the diffusive sink of the sub-filter APE
    # budget, at each online filter scale. Its two halves are built internally against one shared
    # reference profile — hence the `ProfileLookup`, handed the VerticalSort column above so every
    # scale shares that one sort.
    # The cross-scale APE flux Π_A = -τᵢ(b, uᵢ) ∂ᵢΥˡ rides along: it is measured against the same
    # filtered reference state ε_Aˢ uses, so it shares the filter and the column and adds no sort. Both
    # are 2D x–z here (v ≡ 0), hence dims=(1, 3), matching the online Π_K.
    # The reference profile's own time derivative, shared by every R below. A TimeDerivative advances
    # whenever it is evaluated, and R is evaluated only when the writer fetches it, so ∂ₜb✶ follows the
    # writer's schedule with no callback: the R outputs are deferred (see online_diagnostics.jl), so the
    # writer evaluates them when a record opens and once more on the following iteration, and the
    # difference written spans that single timestep, like the other tendencies.
    lookup = ProfileLookup(z✶_1dsort)
    ∂ₜb✶ = TimeDerivative(reference_buoyancy(z✶_1dsort), model)

    # R against the full field's reference height; Rˡ below uses the filtered field's, and Rˢ = filter(R) - Rˡ.
    z✶_lookup = reference_height(model, method=lookup)
    R_full = ReferenceTendencyCorrection(model, ∂ₜb✶, z✶_lookup)

    _ape_pairs = Pair{Symbol, Any}[]
    for ℓ in filter_ℓs
        gf = matched_filter(ℓ)
        ε_As = SubFilterAvailablePotentialEnergyDissipationRate(model, gf; method=lookup)
        Π_A  = AvailablePotentialEnergyCrossScaleFlux(model, gf; dims=(1, 3), method=lookup)
        E_as = SubFilterAvailablePotentialEnergy(model, gf; method=lookup)
        wb_rs = SubFilterAvailablePotentialToKineticEnergyConversion(model, gf; method=lookup)

        # Rˢ = filter(R) - Rˡ, both measured against the same shared profile
        z✶ˡ = reference_height(Field(gf(b)); method=lookup)
        R_l = ReferenceTendencyCorrection(model, ∂ₜb✶, z✶ˡ)
        R_s = Field(gf(R_full)) - R_l

        push!(_ape_pairs, Symbol("ε_As_ℓ$(ℓ)")        => ε_As, Symbol("ε_As_ℓ$(ℓ)_int") => Integral(ε_As),
                          Symbol("Π_A_ℓ$(ℓ)")         => Π_A,  Symbol("Π_A_ℓ$(ℓ)_int")  => Integral(Π_A),
                          Symbol("E_as_ℓ$(ℓ)")        => E_as, Symbol("E_as_ℓ$(ℓ)_int") => Integral(E_as),
                          Symbol("wb_rs_ℓ$(ℓ)")       => wb_rs, Symbol("wb_rs_ℓ$(ℓ)_int") => Integral(wb_rs),
                          Symbol("R_s_ℓ$(ℓ)")         => R_s,  Symbol("R_s_ℓ$(ℓ)_int")  => Integral(R_s),
                          Symbol("dEas_dt_ℓ$(ℓ)")     => TimeDerivative(E_as, model),
                          Symbol("dEas_dt_ℓ$(ℓ)_int") => TimeDerivative(Integral(E_as), model))
    end
    sfs_ape_fields = (; _ape_pairs...)

    sorted_fields = (; z✶_3dsort, z✶_heaviside, z✶_1dsort, b✶_1dsort, E_a, ∫E_a, ∫E_b, sfs_ape_fields...)

    # The 2D writer also gets the sub-filter APE fields (and b_r, sharing the lookup z✶ above), so the
    # panels animation can be drawn straight from the slice file by plot_kelvin_helmholtz_instability.jl.
    # All are model-grid, so the 2D file stays single-grid.
    twod_extra = (; b_r = ReferenceBuoyancyAnomaly(model, z✶_lookup), sfs_ape_fields...)
end
#---

outputs = (; ω=vorticity, b, pe, PE, u=u_center, v=v_center, w=w_center, filtered_fields..., ke_transfer_fields..., ε̄, ε, L_K, Ri=Ri_field, S=S_field)

using NCDatasets
simulation_name = "khi_Nz$(params.Nz)_Ri$(@sprintf("%.2f", params.Ri))"
output_filename = "output/$(simulation_name).nc"

if !(model.closure isa ScalarDiffusivity)
    ν = viscosity(model)
    κ = diffusivity(model, Val(:b))
    outputs = (; outputs..., ν, κ)
end

# The model-grid z✶ fields go in the 3D file only; the 2D writer below slices with `indices` for a
# lightweight x–z animation and has no use for them.
simulation.output_writers[:fields] = NetCDFWriter(model, (; outputs..., sorted_fields...),
                                                  schedule = ConsecutiveIterations(TimeInterval(2)),
                                                  filename = output_filename,
                                                  array_type = Array{Float64},
                                                  global_attributes = params,
                                                  overwrite_existing = true)

output_filename_2d = "output/$(simulation_name)_2d.nc"
simulation.output_writers[:twod_fields] = NetCDFWriter(model, (; outputs..., twod_extra...),
                                                       schedule = TimeInterval(2),
                                                       filename = output_filename_2d,
                                                       array_type = Array{Float32},
                                                       indices = (:, 1, :),
                                                       global_attributes = params,
                                                       overwrite_existing = true)

@info "Output will be saved to: $(output_filename).nc"
#---

#+++ Run simulation
show_gpu_status()
@info @sprintf("""
================================================================================
  Kelvin-Helmholtz instability simulation
================================================================================
  Grid:          Nx=%d, Ny=%d, Nz=%d
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
@info "Creating animation..."
plot_filepath = output_filename_2d
include("plot_kelvin_helmholtz_instability.jl")
#---
