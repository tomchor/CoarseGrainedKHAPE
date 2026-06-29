# Kelvin-Helmholtz instability simulation
using Oceananigans
using CairoMakie
using Printf
using ArgParse
using CUDA: has_cuda_gpu
using Oceananigans.Architectures: on_architecture
using Oceanostics: PotentialEnergyEquation, KineticEnergyEquation, FlowDiagnostics, GaussianFilter, StrainRateTensor, subfilter_stress_tensor, KineticEnergyCrossScaleFlux
using Oceanostics.ProgressMessengers
@info "Finished loading packages"

include("utils.jl")

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

        "--save_tensors"
            help = "Also output the strain-rate (S̄ⁱʲ) and sub-filter stress (τⁱʲ) tensor components at each filter scale (for online-vs-offline validation). These are full 3D fields, so off by default to keep production output lean."
            action = :store_true
    end
    global parsed_args = parse_args(s, as_symbols=true)
end
# Keep the save_tensors control flag out of `params` (it is a Bool, which NetCDF can't store as a
# global attribute, and it is not a physical parameter).
save_tensors = pop!(parsed_args, :save_tensors)
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
η = (params.ν^3 / ε̄) ^ (1/4)


progress(simulation) = @info (PercentageProgress(with_prefix=false, with_units=false)
                              + walltime
                              + TimeStep()
                              + "CFL = " * AdvectiveCFLNumber(with_prefix=false)
                              + "Diffusive CFL = " * DiffusiveCFLNumber(with_prefix=false)
                              + MaxWVelocity()
                              + "step dur = " * walltime_per_timestep
                              + (sim -> @sprintf("Kolmogorov length/Δx = %.2f", minimum(η) / Δx))
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
filter_ℓs = (1, 7)
_FWHM_to_σ(ℓ) = ℓ / (2 * sqrt(2 * log(2)))
_fields = (u=u_center, v=v_center, w=w_center, b=b)
_filt_pairs = [Symbol("$(n)_ℓ$(ℓ)") => GaussianFilter(f; dims=(1, 3), σ=_FWHM_to_σ(ℓ)) for ℓ in filter_ℓs for (n, f) in pairs(_fields)]
filtered_fields = (; _filt_pairs...)
#---

#+++ Online cross-scale KE transfer Πₖ = -τⁱʲ S̄ⁱʲ  (Oceanostics KineticEnergyCrossScaleFlux)
# Cross-scale (subfilter → resolved) kinetic-energy flux of Aluie et al. (2018, JPO), computed by
# Oceanostics' KineticEnergyCrossScaleFlux at each filter scale ℓ. The Gaussian filter is configured
# to reproduce the offline post-processing filter (postprocessing/src/aux00_utils.py): periodic x,
# edge-extended bounded z, and a stencil truncated at 4σ (scipy gaussian_filter1d's default;
# Oceanostics would otherwise truncate at 2σ). The runs are 2D in x–z (v ≡ 0) so dims=(1, 3); Πₖ is
# per unit mass (m² s⁻³, no ρ₀) and Πₖ > 0 is forward (downscale) transfer.
to_center(ψ) = @at (Center, Center, Center) ψ

# Per-direction Gaussian stencil widths matching scipy's truncate=4 (radius = ⌊4σ/Δ + ½⌋ cells).
_filter_N(σ) = (2 * max(1, floor(Int, 4σ / minimum_xspacing(grid) + 0.5)) + 1,
                2 * max(1, floor(Int, 4σ / minimum_zspacing(grid) + 0.5)) + 1)

_ke_pairs = Pair{Symbol, Any}[]
for ℓ in filter_ℓs
    σ = _FWHM_to_σ(ℓ);  N = _filter_N(σ)
    Πₖ = KineticEnergyCrossScaleFlux(model; σ, dims=(1, 3), boundary=:edge, N)
    push!(_ke_pairs, Symbol("Π_K_ℓ$(ℓ)") => Πₖ, Symbol("Π_K_ℓ$(ℓ)_int") => Integral(Πₖ))

    # Individual strain (S̄ⁱʲ) and sub-filter stress (τⁱʲ) components at cell centers, for the
    # online-vs-offline validation in postprocessing/validation/. Full 3D fields → gated behind
    # --save_tensors to keep production output lean.
    if save_tensors
        filt = ψ -> GaussianFilter(ψ; dims=(1, 3), σ, boundary=:edge, N)
        ū = Field(filt(u)); w̄ = Field(filt(w))
        S̄ = StrainRateTensor(grid, ū, v, w̄; dims=(1, 3))          # strain of the filtered velocity
        τ = subfilter_stress_tensor(model; σ, dims=(1, 3), boundary=:edge, N)  # τⁱʲ = filter(uⁱuʲ) - ūⁱūʲ
        push!(_ke_pairs,
              Symbol("S11_ℓ$(ℓ)")   => to_center(S̄.S₁₁), Symbol("S33_ℓ$(ℓ)")   => to_center(S̄.S₃₃), Symbol("S13_ℓ$(ℓ)")   => to_center(S̄.S₁₃),
              Symbol("tau11_ℓ$(ℓ)") => to_center(τ.τ₁₁), Symbol("tau33_ℓ$(ℓ)") => to_center(τ.τ₃₃), Symbol("tau13_ℓ$(ℓ)") => to_center(τ.τ₁₃))
    end
end
ke_transfer_fields = (; _ke_pairs...)
#---

outputs = (; ω=vorticity, b, pe, PE, u=u_center, v=v_center, w=w_center, filtered_fields..., ke_transfer_fields..., ε̄, ε, Ri=Ri_field, S=S_field)

using NCDatasets
simulation_name = "khi_Nz$(params.Nz)_Ri$(@sprintf("%.2f", params.Ri))"
output_filename = "output/$(simulation_name).nc"

if !(model.closure isa ScalarDiffusivity)
    ν = viscosity(model)
    κ = diffusivity(model, Val(:b))
    outputs = (; outputs..., ν, κ)
end

simulation.output_writers[:fields] =
    NetCDFWriter(model, outputs,
                 schedule = ConsecutiveIterations(TimeInterval(2)), # Consecutive iterations every 4 periods to calculate time derivatives
                 filename = output_filename,
                 array_type = Array{Float64},
                 global_attributes = params,
                 overwrite_existing = true)

output_filename_2d = "output/$(simulation_name)_2d.nc"
simulation.output_writers[:twod_fields] =
NetCDFWriter(model, outputs,
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
