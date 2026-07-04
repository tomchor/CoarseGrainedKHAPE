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
filter_widths = (1, 7)
#---

#+++ Load timeseries
@info "Loading timeseries..."
ω_timeseries = FieldTimeSeries(plot_filepath, "ω", architecture=CPU()) |> x -> on_architecture(CPU(), x)
b_timeseries = FieldTimeSeries(plot_filepath, "b", architecture=CPU()) |> x -> on_architecture(CPU(), x)
S_timeseries = FieldTimeSeries(plot_filepath, "S", architecture=CPU()) |> x -> on_architecture(CPU(), x)
bℓ1_timeseries = FieldTimeSeries(plot_filepath, "b_ℓ1", architecture=CPU()) |> x -> on_architecture(CPU(), x)
bℓ7_timeseries = FieldTimeSeries(plot_filepath, "b_ℓ7", architecture=CPU()) |> x -> on_architecture(CPU(), x)

times = ω_timeseries.times
#---

#+++ Build figure
n = Observable(1)

ωₙ   = @lift view(ω_timeseries[$n], :, 1, :)
bₙ   = @lift view(b_timeseries[$n], :, 1, :)
Sₙ   = @lift view(S_timeseries[$n], :, 1, :)
bℓ1ₙ = @lift view(bℓ1_timeseries[$n], :, 1, :)
bℓ7ₙ = @lift view(bℓ7_timeseries[$n], :, 1, :)

fig = Figure(size=(1200, 900))

params_str = @sprintf("Re = %d,  Ri = %.2f,  Pr = %d", Re, Ri, Pr)
title = @lift @sprintf("Kelvin-Helmholtz Instability  (%s)\nt = %.1f", params_str, times[$n])
fig[1, 1:6] = Label(fig, title, fontsize=20, tellwidth=false, justification=:center)

kwargs = (xlabel="x", ylabel="z", aspect=1)

ax_ω = Axis(fig[2, 1]; title="Vorticity", kwargs...)
ax_b = Axis(fig[2, 3]; title="Buoyancy", kwargs...)
ax_S = Axis(fig[2, 5]; title="Strain rate (S)", kwargs...)
b_crange = (-0.1, 0.1)

hm_ω = heatmap!(ax_ω, ωₙ; colormap=:balance, colorrange=(-1, 1))
Colorbar(fig[2, 2], hm_ω)

hm_b = heatmap!(ax_b, bₙ; colormap=:balance, colorrange=b_crange)
Colorbar(fig[2, 4], hm_b)

hm_S = heatmap!(ax_S, Sₙ; colormap=:thermal)
Colorbar(fig[2, 6], hm_S)

ax_bf1 = Axis(fig[3, 1]; title="Filtered b (ℓ = 1)", kwargs...)
hm_bf1 = heatmap!(ax_bf1, bℓ1ₙ; colormap=:balance, colorrange=b_crange)
Colorbar(fig[3, 2], hm_bf1)

ax_bf7 = Axis(fig[3, 3]; title="Filtered b (ℓ = 7)", kwargs...)
hm_bf7 = heatmap!(ax_bf7, bℓ7ₙ; colormap=:balance, colorrange=b_crange)
Colorbar(fig[3, 4], hm_bf7)
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
