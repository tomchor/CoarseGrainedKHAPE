# Exploratory: volume render and isosurfaces of one snapshot of a 3D run.
#
#   julia --project postprocessing/X13_3d_volume.jl <3d_output_filepath> [--field b] [--mode both] ...
#
# GLMakie rather than CairoMakie: a volume render is GPU raymarching, which Cairo cannot do at all, and
# Makie's 3D `contour` wants the same backend. It needs OpenGL, so this runs on a workstation or an HPC
# node with VirtualGL/EGL -- not in a plain batch job. X12_3d_snapshot.py is the no-display counterpart.
#
# The companion to X12: that one cuts planes, this one looks through the volume. For the KH run the
# isosurface is the more useful of the two -- Q or enstrophy picks out the streamwise vortices that
# braid the billow, which is the whole reason the simulation went 3D.

using NCDatasets
using GLMakie
using Printf
using Statistics

GLMakie.activate!(visible = false)   # render offscreen; `save` still works

#+++ Arguments
# Deliberately hand-rolled rather than ArgParse: this is called by hand, and keeping it dependency-light
# means it also runs under `julia --project` with nothing precompiled beyond Makie and NCDatasets.
length(ARGS) > 0 || error("Usage: julia --project postprocessing/X13_3d_volume.jl <3d_output_filepath> [--field F] [--mode M] ...")
filepath = ARGS[1]
opts = Dict{String,String}()
for i in 2:2:length(ARGS)-1
    startswith(ARGS[i], "--") || error("expected a --flag at argument $i, got $(ARGS[i])")
    opts[ARGS[i][3:end]] = ARGS[i+1]
end
field   = get(opts, "field", "b")
mode    = get(opts, "mode", "both")            # volume | isosurface | both
tsel    = haskey(opts, "time") ? parse(Float64, opts["time"]) : nothing
zlim    = parse(Float64, get(opts, "zlim", "4.0"))
pct     = parse(Float64, get(opts, "clim-percentile", "99.0"))
nlevels = parse(Int,     get(opts, "levels", "4"))
alpha   = parse(Float64, get(opts, "alpha", "0.35"))
azim    = parse(Float64, get(opts, "azimuth", "1.15"))    # in units of π, as Makie's Axis3 takes it
elev    = parse(Float64, get(opts, "elevation", "0.12"))
mode in ("volume", "isosurface", "both") || error("--mode must be volume, isosurface or both; got $mode")

repo_root = dirname(@__DIR__)
figures   = joinpath(@__DIR__, "extra_figures")
mkpath(figures)
isabspath(filepath) || (filepath = joinpath(repo_root, filepath))
stem = replace(basename(filepath), ".nc" => "")
@info "Reading $filepath"
#---

#+++ Read the snapshot
# NCDatasets indexes in the file's own order, which is the reverse of Python's: (x, y, z, time).
ds = NCDataset(filepath, "r")
x, y, z = ds["x_caa"][:], ds["y_aca"][:], ds["z_aac"][:]
times   = ds["time"][:]
n = isnothing(tsel) ? length(times) : argmin(abs.(times .- tsel))
t = times[n]

read3d(name) = Float64.(ds[name][:, :, :, n])

#+++ Derived fields
# The file's ω is the spanwise component alone. u, v and w are all written at cell centres, so the
# gradient tensor is a centred difference on one grid, with no staggered interpolation. Periodic in x
# and y, one-sided at the z walls -- matching how the rest of the pipeline treats the bounded axis.
function ∂(a, dim, coord)
    d = diff(coord)
    all(≈(d[1]), d) || error("X13 assumes a uniform grid along dimension $dim")
    h = d[1]
    if dim == 3   # z is Bounded: one-sided at the walls
        g = similar(a)
        g[:, :, 2:end-1] = (a[:, :, 3:end] .- a[:, :, 1:end-2]) ./ (2h)
        g[:, :, 1]       = (a[:, :, 2]   .- a[:, :, 1])   ./ h
        g[:, :, end]     = (a[:, :, end] .- a[:, :, end-1]) ./ h
        return g
    end           # x and y are Periodic
    return (circshift(a, ntuple(i -> i == dim ? -1 : 0, 3)) .- circshift(a, ntuple(i -> i == dim ? 1 : 0, 3))) ./ (2h)
end

function velocity_gradient()
    u, v, w = read3d("u"), read3d("v"), read3d("w")
    coords = (x, y, z)
    return [∂(a, j, coords[j]) for a in (u, v, w), j in 1:3]   # A[i, j] = ∂uⁱ/∂xʲ
end

function derived(name)
    if name == "speed"
        u, v, w = read3d("u"), read3d("v"), read3d("w")
        return sqrt.(u.^2 .+ v.^2 .+ w.^2)
    end
    A = velocity_gradient()
    if name == "enstrophy"
        ωx = A[3, 2] .- A[2, 3]; ωy = A[1, 3] .- A[3, 1]; ωz = A[2, 1] .- A[1, 2]
        return ωx.^2 .+ ωy.^2 .+ ωz.^2
    elseif name == "Q"
        # Q = ½(|Ω|² − |S|²), Hunt et al. (1988): positive where rotation beats strain.
        S2 = sum((0.5 .* (A[i, j] .+ A[j, i])).^2 for i in 1:3, j in 1:3)
        Ω2 = sum((0.5 .* (A[i, j] .- A[j, i])).^2 for i in 1:3, j in 1:3)
        return 0.5 .* (Ω2 .- S2)
    end
    error("unknown derived field $name; try Q, enstrophy or speed")
end
#---

data = field in ("Q", "enstrophy", "speed") ? derived(field) : read3d(field)

# Lz = 25h is mostly quiescent: crop before rendering, or the billow is a sliver in an empty box.
kz = findall(zi -> abs(zi) <= zlim, z)
z, data = z[kz], data[:, :, kz]
Re, Ri = Float64(ds.attrib["Re"]), Float64(ds.attrib["Ri"])
close(ds)
@info @sprintf("%s at t = %.1f, |z| < %.1f: %s", field, t, zlim, size(data))
#---

#+++ Colour range
signed = minimum(data) < 0 < maximum(data)
absq(p) = quantile(abs.(vec(data)), p / 100)
colorrange = signed ? (-absq(pct), absq(pct)) :
                      (quantile(vec(data), 1 - pct / 100), quantile(vec(data), pct / 100))
colormap = signed ? :balance : :magma
@info @sprintf("colour range [%.3g, %.3g] (%s)", colorrange[1], colorrange[2], signed ? "signed" : "positive")

# Isosurface levels. For a signed field take a symmetric pair per level so both rotation senses show;
# for a positive one (enstrophy, speed) take the upper tail, where the structures are.
levels = if signed
    f = collect(range(0.3, 0.85, max(nlevels ÷ 2, 1)))
    sort(vcat(-colorrange[2] .* f, colorrange[2] .* f))
else
    [quantile(vec(data), q) for q in range(0.90, 0.995, nlevels)]
end
@info "isosurface levels: " * join((@sprintf("%.3g", l) for l in levels), ", ")
#---

#+++ Figure
Lx, Ly, Lz = x[end] - x[1], y[end] - y[1], z[end] - z[1]
panels = mode == "both" ? ("volume", "isosurface") : (mode,)
fig = Figure(size = (720 * length(panels), 640))

for (col, kind) in enumerate(panels)
    ax = Axis3(fig[1, col]; aspect = (Lx, Ly, Lz), xlabel = "x", ylabel = "y", zlabel = "z",
               title = kind == "volume" ? "volume (MIP)" : "isosurfaces", azimuth = azim * π, elevation = elev * π)
    if kind == "volume"
        # Maximum-intensity projection: no transfer function to tune, and it shows where the extremes
        # are. :absorption looks better but needs an opacity curve matched to the field's range.
        volume!(ax, x[1] .. x[end], y[1] .. y[end], z[1] .. z[end], data;
                algorithm = :mip, colormap, colorrange)
    else
        contour!(ax, x[1] .. x[end], y[1] .. y[end], z[1] .. z[end], data;
                 levels, colormap, colorrange, alpha, transparency = true)
    end
end

Colorbar(fig[1, length(panels) + 1]; colormap, colorrange, label = field, height = Relative(0.6))
Label(fig[0, :], @sprintf("%s   t = %.1f      Re = %d,  Ri = %.2f", field, t, round(Int, Re), Ri), fontsize = 16)

outfile = joinpath(figures, @sprintf("%s_3dvol_%s_t%.1f.png", stem, field, t))
save(outfile, fig; px_per_unit = 2)
@info "Figure saved to: $outfile"
#---
