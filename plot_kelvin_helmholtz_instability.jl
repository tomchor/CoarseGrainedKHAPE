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

#+++ SFS budget panels animation (per filter scale), from the online 2D fields
# Mirrors postprocessing/anim1_panels.py: 3×4 snapshot panels over the two integrated budgets, but
# drawn entirely from the simulation's own outputs — no offline pipeline. Runs only when the run was
# made with --save_sorted, which is what puts the sub-filter APE fields and b_r in the 2D file.
ds_nc = NCDataset(plot_filepath, "r")
panel_ℓs = sort([parse(Int, split(name, "_ℓ")[end]) for name in keys(ds_nc) if startswith(name, "wb_rs_ℓ") && !endswith(name, "_int")])

if isempty(panel_ℓs)
    @info "No sub-filter APE fields in $plot_filepath (run without --save_sorted); skipping the panels animation."
else
    zlim = 3.8
    x  = ds_nc["x_caa"][:]
    z  = ds_nc["z_aac"][:]
    kz = searchsortedfirst(z, -zlim):searchsortedlast(z, zlim)
    t  = ds_nc["time"][:]
    nt = length(t) - 1   # the final record holds NaN for every deferred output (tendencies, Rˢ)

    slab(name, n)  = ds_nc[name][:, 1, kz, n]
    series(name)   = Float64.(ds_nc[name][1:nt])
    "Symmetric color limit from the 99th percentile of |field| over ~10 sampled frames."
    function clim(name)
        samples = round.(Int, range(1, nt, length=min(10, nt)))
        v = sort(abs.(filter(isfinite, vcat([vec(slab(name, n)) for n in samples]...))))
        v = isempty(v) ? 1.0 : v[max(1, round(Int, 0.99 * length(v)))]
        return v > 0 ? v : 1.0
    end

    for ℓ in panel_ℓs
        panel_names = ["ω"          "w"          "b"          "b_r";
                       "K_s_ℓ$ℓ"    "Π_K_ℓ$ℓ"    "ε_Ks_ℓ$ℓ"   "wb_rs_ℓ$ℓ";
                       "E_as_ℓ$ℓ"   "Π_A_ℓ$ℓ"    "ε_As_ℓ$ℓ"   "R_s_ℓ$ℓ"]
        panel_titles = ["Vorticity ω"  "Vertical velocity w"  "Buoyancy b"       "Relative buoyancy b_r";
                        "SFS KE"       "Π_K"                  "ε_Kˢ"             "APE→KE exchange τ(w,b_r)";
                        "SFS APE"      "Π_A"                  "ε_Aˢ"             "Rˢ"]

        n = Observable(1)
        fig = Figure(size=(1600, 1250))
        title = @lift @sprintf("%s,  ℓ = %d,  t = %.1f", params_str, ℓ, t[$n])
        fig[0, 1:4] = Label(fig, title, fontsize=20, tellwidth=false)

        for row in 1:3, col in 1:4
            name = panel_names[row, col]
            ax = Axis(fig[row, col]; title=panel_titles[row, col], aspect=DataAspect(),
                      ylabel=(col == 1 ? "z" : ""), xlabel=(row == 3 ? "x" : ""),
                      xticklabelsvisible=(row == 3), yticklabelsvisible=(col == 1))
            v = clim(name)
            heatmap!(ax, x, z[kz], @lift(slab(name, $n)); colormap=:balance, colorrange=(-v, v))
        end

        wb  = series("wb_rs_ℓ$(ℓ)_int")
        ke_terms  = [("-∂ₜKˢ",  -series("dKs_dt_ℓ$(ℓ)_int"),  :royalblue), ("Π_K",  series("Π_K_ℓ$(ℓ)_int"),  :orange),
                     ("-ε_Kˢ",  -series("ε_Ks_ℓ$(ℓ)_int"),    :crimson),   ("τ(w,b_r)", wb,                   :seagreen)]
        ape_terms = [("-∂ₜEₐˢ", -series("dEas_dt_ℓ$(ℓ)_int"), :royalblue), ("Π_A",  series("Π_A_ℓ$(ℓ)_int"), :orange),
                     ("-ε_Aˢ",  -series("ε_As_ℓ$(ℓ)_int"),    :crimson),   ("-τ(w,b_r)", -wb,                 :seagreen),
                     ("Rˢ",     series("R_s_ℓ$(ℓ)_int"),      :purple)]

        for (rowidx, terms, label) in ((4, ke_terms, "SFS KE budget"), (5, ape_terms, "SFS APE budget"))
            ax = Axis(fig[rowidx, 1:4]; ylabel=label, xlabel=(rowidx == 5 ? "t" : ""), xticklabelsvisible=(rowidx == 5))
            for (lbl, y, color) in terms
                lines!(ax, t[1:nt], y; label=lbl, color, linewidth=1.5)
            end
            lines!(ax, t[1:nt], sum(y for (_, y, _) in terms); label="residual", color=:gray, linestyle=:dash)
            vlines!(ax, @lift([t[$n]]); color=:black, linestyle=:dot)
            axislegend(ax, position=:rt, labelsize=9)
        end
        rowsize!(fig.layout, 4, Relative(0.12)); rowsize!(fig.layout, 5, Relative(0.12))

        panels_filename = "animations/" * replace(basename(plot_filepath), "_2d.nc" => "") * @sprintf("_panels_l%.4f.mp4", ℓ)
        record(fig, panels_filename, 1:nt, framerate=12) do i
            i % 10 == 1 && @info "Panels ℓ=$ℓ: frame $i of $nt..."
            n[] = i
        end
        @info "Panels animation saved as $(panels_filename)"
    end
end
close(ds_nc)
#---
