# Kelvin-Helmholtz instability simulation

using Oceananigans
using CairoMakie
using Printf
using ArgParse
using CUDA: has_cuda_gpu
using Oceananigans.Architectures: on_architecture
using Oceanostics: PotentialEnergyEquation, KineticEnergyEquation
using Oceanostics.ProgressMessengers

include("utils.jl")

#+++ Parse command-line arguments
let s = ArgParseSettings()
    @add_arg_table! s begin
        "--Nz"
            help = "Number of vertical grid points (default: 512 on CPU, 4096 on GPU)"
            arg_type = Int
            required = false
            default = has_cuda_gpu() ? 4096 : 512
    end
    global parsed_args = parse_args(s)
end
#---

Nz = parsed_args["Nz"]

#+++ Define simulation parameters
params = (
    Lx = 10,
    Ly = 5,
    Lz = 14,
    Ri = 0.1,
    h = 1/4,
    perturbation_amplitude = 0.01,
    stop_time = 200.0,
)
#---

#+++ Create grid
if has_cuda_gpu()
    arch = GPU()
    x_aspect_ratio = 1   # Δx / Δz ratio
    y_aspect_ratio = Inf # Δy / Δz ratio
    ν = 5e-5
    κ = 5e-5
else
    @warn "No CUDA GPU detected. Running on CPU with a coarse grid and high aspect ratio."

    arch = CPU()
    x_aspect_ratio = 2   # Δx / Δz ratio
    y_aspect_ratio = Inf # Δy / Δz ratio
    ν = 2e-3
    κ = 2e-3
end

@info "Cell aspect ratio: Δx/Δz = $(x_aspect_ratio), Δy/Δz = $(y_aspect_ratio)"

# Calculate horizontal resolutions based on aspect ratios
Nx = round(Int, Nz * (params.Lx / params.Lz) / x_aspect_ratio)
Ny = isinf(y_aspect_ratio) ? 1 : round(Int, Nz * (params.Ly / params.Lz) / y_aspect_ratio)

# Adjust grid sizes to be factorizable by 2, 3, and 5 (for FFT performance)
Nx = closest_factor_number((2, 3, 5), Nx)
Ny = closest_factor_number((2, 3, 5), Ny)

params = (; params..., Nx, Ny, Nz, ν, κ)

grid = RectilinearGrid(arch; size=(params.Nx, params.Ny, params.Nz),
                       x=(-params.Lx/2, params.Lx/2),
                       y=(-params.Ly/2, params.Ly/2),
                       z=(-params.Lz/2, params.Lz/2),
                       topology=(Periodic, Periodic, Bounded))
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
shear_flow(x, z) = tanh(z) # Base shear flow
stratification(x, z) = params.h * params.Ri * tanh(z / params.h) # Base stratification
perturbation(x, z) = params.perturbation_amplitude * sin(2π * x / 10) * exp(-z^2 / 2) # Small perturbation to trigger instability

# Set initial conditions
uᵢ(x, y, z) = shear_flow(x, z)
bᵢ(x, y, z) = stratification(x, z)
wᵢ(x, y, z) = params.perturbation_amplitude * cos(2π * x / 10 + π/2) * exp(-z^2 / 2)

set!(model, u=uᵢ, b=bᵢ, w=wᵢ)
#---

#+++ Setup simulation
simulation = Simulation(model, Δt=0.01, stop_time=params.stop_time)

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
                              + MaxUVelocity()
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

ρ₀ = 1025 # kg/ m^3
pe = ρ₀ * PotentialEnergyEquation.PotentialEnergy(model)

PE = Integral(pe)

vorticity = Field(∂z(u) - ∂x(w))

outputs = (; ω=vorticity, b, pe, PE, u=u_center, v=v_center, w=w_center, ε̄)

using NCDatasets
output_filename = "output/khi_$(params.Nx)x$(params.Ny)x$(params.Nz)"
if !(model.closure isa ScalarDiffusivity)
    ν = viscosity(model)
    κ = diffusivity(model, Val(:b))
    outputs = (; outputs..., ν, κ)
end

simulation.output_writers[:fields] =
    NetCDFWriter(model, outputs,
                 schedule = ConsecutiveIterations(TimeInterval(4)), # Consecutive iterations every 4 periods to calculate time derivatives
                 filename = output_filename,
                 array_type = Array{Float64},
                 global_attributes = params,
                 overwrite_existing = true)

output_filename_2d = "output/khi_$(params.Nx)x$(params.Ny)x$(params.Nz)_2d.nc"
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

# Run simulation
show_gpu_status()
@info "Running Kelvin-Helmholtz instability simulation..."
run!(simulation)

#+++ Load and plot results
@info "Creating animation..."

filepath = simulation.output_writers[:twod_fields].filepath

ω_timeseries = FieldTimeSeries(filepath, "ω", architecture=CPU())
b_timeseries = FieldTimeSeries(filepath, "b", architecture=CPU())

# The architecture of the FieldTimeSeries isn't working as expected, so we need to load the data on the CPU manually:
ω_timeseries = on_architecture(CPU(), ω_timeseries)
b_timeseries = on_architecture(CPU(), b_timeseries)

times = ω_timeseries.times

n = Observable(1)

ωₙ = @lift view(ω_timeseries[$n], :, 1, :)
bₙ = @lift view(b_timeseries[$n], :, 1, :)

fig = Figure(size=(900, 500))

title = @lift @sprintf("Kelvin-Helmholtz Instability: t = %.1f", times[$n])
fig[1, :] = Label(fig, title, fontsize=24, tellwidth=false)

kwargs = (xlabel="x", ylabel="z", limits=((-5, 5), (-5, 5)), aspect=1)

ax_ω = Axis(fig[2, 1]; title="Vorticity", kwargs...)
ax_b = Axis(fig[2, 3]; title="Buoyancy", kwargs...)

hm_ω = heatmap!(ax_ω, ωₙ; colormap=:balance, colorrange=(-1, 1))
Colorbar(fig[2, 2], hm_ω)

hm_b = heatmap!(ax_b, bₙ; colormap=:balance, colorrange=(-0.05, 0.05))
Colorbar(fig[2, 4], hm_b)

frames = 1:length(times)

animation_filename = "animations/$(basename(output_filename)).mp4"
record(fig, animation_filename, frames, framerate=12) do i
    @info "Plotting frame $i of $(frames[end])..."
    n[] = i
end

@info "Animation saved as $(animation_filename)"
#---
