# Kelvin-Helmholtz instability simulation
using Oceananigans
using CairoMakie
using Printf
using ArgParse
using CUDA: has_cuda_gpu
using Oceananigans.Architectures: on_architecture
using Oceanostics: PotentialEnergyEquation, KineticEnergyEquation
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
            default = has_cuda_gpu() ? 4096 : 512

        "--Ri"
            help = "Richardson number (default: 0.1)"
            arg_type = Float64
            required = false
            default = 0.1

        "--stop-time"
            help = "Simulation stop time (default: 200.0)"
            arg_type = Float64
            required = false
            default = 200.0
    end
    global parsed_args = parse_args(s)
end
#---

Nz = parsed_args["Nz"]
Ri = parsed_args["Ri"]
stop_time = parsed_args["stop-time"]

#+++ Define simulation parameters
params = (
    Lx = 10,
    Ly = 5,
    Lz = 14,
    Ri = Ri,
    h = 1/4,
    perturbation_amplitude = 0.01,
    stop_time = stop_time,
    Re₀ = 1e-3, # Reynolds number (ν = 1/Re)
    Pr = 1,     # Prandtl number (κ = ν/Pr)
)
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
    params = (; params..., Re = 500) # Reduce Re for coarser CPU run
end

@info "Cell aspect ratio: Δx/Δz = $(x_aspect_ratio), Δy/Δz = $(y_aspect_ratio)"

# Calculate horizontal resolutions based on aspect ratios
Nx = round(Int, Nz * (params.Lx / params.Lz) / x_aspect_ratio)
Ny = isinf(y_aspect_ratio) ? 1 : round(Int, Nz * (params.Ly / params.Lz) / y_aspect_ratio)

# Adjust grid sizes to be factorizable by 2, 3, and 5 (for FFT performance)
Nx = closest_factor_number((2, 3, 5), Nx)
Ny = closest_factor_number((2, 3, 5), Ny)

params = (; params..., Nx, Ny, Nz)

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
    ν = 1 / Re
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
simulation_name = "khi_Nz$(params.Nz)_Ri$(@sprintf("%.2f", params.Ri))"
output_filename = "output/$(simulation_name).nc"

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

params_str = @sprintf("Re = %d,  Ri = %.2f,  Pr = %d", params.Re, params.Ri, params.Pr)
title = @lift @sprintf("Kelvin-Helmholtz Instability  (%s)\nt = %.1f", params_str, times[$n])
fig[1, 1:4] = Label(fig, title, fontsize=20, tellwidth=false, justification=:center)

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
