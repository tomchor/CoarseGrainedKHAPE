# Plot Kelvin-Helmholtz instability simulation results.
#
# Run standalone:
#   julia --project plot_kelvin_helmholtz_instability.jl <2d_output_filepath>
#
# Or called automatically at the end of kelvin_helmholtz_instability.jl.

using CairoMakie
using Printf
using Oceananigans
using Oceananigans.Architectures: on_architecture
using NCDatasets

#+++ Get filepath
if !@isdefined(plot_filepath)
    length(ARGS) > 0 || error("Usage: julia --project plot_kelvin_helmholtz_instability.jl <2d_output_filepath>")
    plot_filepath = ARGS[1]
end
@info "Plotting from: $plot_filepath"
#---

#+++ Read simulation parameters from file global attributes
Re, Ri, Pr = NCDataset(plot_filepath, "r") do ds
    Float64(ds.attrib["Re"]), Float64(ds.attrib["Ri"]), Float64(ds.attrib["Pr"])
end
#---

#+++ Load timeseries
@info "Loading timeseries..."
ω_timeseries = FieldTimeSeries(plot_filepath, "ω", architecture=CPU())
b_timeseries = FieldTimeSeries(plot_filepath, "b", architecture=CPU())
S_timeseries = FieldTimeSeries(plot_filepath, "S", architecture=CPU())

# The architecture of the FieldTimeSeries isn't working as expected, so load on CPU manually:
ω_timeseries = on_architecture(CPU(), ω_timeseries)
b_timeseries = on_architecture(CPU(), b_timeseries)
S_timeseries = on_architecture(CPU(), S_timeseries)

times = ω_timeseries.times
#---

#+++ Build figure
n = Observable(1)

ωₙ = @lift view(ω_timeseries[$n], :, 1, :)
bₙ = @lift view(b_timeseries[$n], :, 1, :)
Sₙ = @lift view(S_timeseries[$n], :, 1, :)

fig = Figure(size=(1200, 500))

params_str = @sprintf("Re = %d,  Ri = %.2f,  Pr = %d", Re, Ri, Pr)
title = @lift @sprintf("Kelvin-Helmholtz Instability  (%s)\nt = %.1f", params_str, times[$n])
fig[1, 1:6] = Label(fig, title, fontsize=20, tellwidth=false, justification=:center)

kwargs = (xlabel="x", ylabel="z", aspect=1)

ax_ω = Axis(fig[2, 1]; title="Vorticity",       kwargs...)
ax_b = Axis(fig[2, 3]; title="Buoyancy",         kwargs...)
ax_S = Axis(fig[2, 5]; title="Strain rate (S)",  kwargs...)

hm_ω = heatmap!(ax_ω, ωₙ; colormap=:balance, colorrange=(-1, 1))
Colorbar(fig[2, 2], hm_ω)

hm_b = heatmap!(ax_b, bₙ; colormap=:balance, colorrange=(-0.08, 0.08))
Colorbar(fig[2, 4], hm_b)

hm_S = heatmap!(ax_S, Sₙ; colormap=:thermal)
Colorbar(fig[2, 6], hm_S)
#---

#+++ Record animation
frames = 1:length(times)
animation_filename = "animations/" * replace(basename(plot_filepath), ".nc" => ".mp4")
record(fig, animation_filename, frames, framerate=12) do i
    @info "Plotting frame $i of $(frames[end])..."
    n[] = i
end
@info "Animation saved as $(animation_filename)"
#---
