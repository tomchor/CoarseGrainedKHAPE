# Exploratory: volume render and isosurfaces of one snapshot of a 3D run.
#
#   julia --project postprocessing/X13_3d_volume.jl <3d_output_filepath> [--field b] [--mode both] ...
#
# One --field lays the panels out by mode (volume and isosurface of it); a comma-separated list lays them
# out by field, one panel each, in the single --mode given. --scale appends the _ℓ<ℓ> every budget term
# carries, so the six-term budget figure is one readable line:
#
#   ... --field Q,wb_rs,Π_K,Π_A,ε_Ks,ε_As --scale 1 --mode isosurface --time 108
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
# --field takes one name or a comma-separated list. One name lays the panels out by --mode (volume and/or
# isosurface of that field); several lay them out by field, one panel each, in the single --mode given.
fields  = String.(split(get(opts, "field", "b"), ","))
scale   = get(opts, "scale", "")               # appended as _ℓ<scale> to any name that needs it (see resolve)
mode    = get(opts, "mode", "both")            # volume | isosurface | both
cols    = parse(Int, get(opts, "cols", "3"))
tsel    = haskey(opts, "time") ? parse(Float64, opts["time"]) : nothing
zlim    = parse(Float64, get(opts, "zlim", "4.0"))
pct     = parse(Float64, get(opts, "clim-percentile", "99.0"))
nlevels = parse(Int,     get(opts, "levels", "4"))
alpha   = parse(Float64, get(opts, "alpha", "0.35"))
# Camera, in units of π, as Makie's Axis3 takes it. Azimuth π looks straight down +x, so the streamwise
# direction is fully into the page; 1.5π looks along +y and gives x the full width but puts y edge-on and
# collides its labels. 1.3π is where x still spans nearly the whole frame and y keeps enough depth to read.
azim    = parse(Float64, get(opts, "azimuth", "1.30"))
elev    = parse(Float64, get(opts, "elevation", "0.12"))
mode in ("volume", "isosurface", "both") || error("--mode must be volume, isosurface or both; got $mode")
length(fields) == 1 || mode != "both" ||
    error("--mode both lays panels out by mode, so it takes one --field; got $(length(fields)). Pick --mode isosurface or volume.")

repo_root = dirname(@__DIR__)
figures   = joinpath(@__DIR__, "extra_figures")
mkpath(figures)
isabspath(filepath) || (filepath = joinpath(repo_root, filepath))
stem = replace(basename(filepath), ".nc" => "")
@info "Reading $filepath"
#---

#+++ Read the snapshot
# A multi-GPU run's merged 3D file is HDF5 virtual datasets over the rank files, which netCDF-C reports with a
# storage code NCDatasets does not map, so a read throws a KeyError until the codes are registered. The canonical
# definition is `allow_virtual_storage!` in merge_rank_output.jl, which says "any Julia reader of a virtual file
# calls this first"; copied rather than included, since that file pulls in Oceananigans and HDF5_jll for what is
# one line. Harmless on an ordinary file, and a no-op if NCDatasets ever maps the codes itself.
allow_virtual_storage!() = foreach(code -> get!(NCDatasets.NCSymbols, code, :contiguous), 2:4)
allow_virtual_storage!()

# NCDatasets indexes in the file's own order, which is the reverse of Python's: (x, y, z, time).
ds = NCDataset(filepath, "r")
x, y, z = ds["x_caa"][:], ds["y_aca"][:], ds["z_aac"][:]
times   = ds["time"][:]
n = isnothing(tsel) ? length(times) : argmin(abs.(times .- tsel))
t = times[n]

# Lz = 25h is mostly quiescent, so everything is read and differentiated on the cropped slab alone: at Nz=1024 a
# single field is 0.91 GB and the gradient tensor is nine of them, so cropping after the fact would peak around
# 15-25 GB to render a third of it. One cell of margin each side keeps the centred z difference exact over the
# slab proper -- `kz` below trims it off again -- and collapses to the true walls when the crop spans the domain.
kz      = findall(zi -> abs(zi) <= zlim, z)
kz_read = max(1, first(kz) - 1):min(length(z), last(kz) + 1)
kz      = (first(kz) - first(kz_read) + 1):(last(kz) - first(kz_read) + 1)   # the slab proper, within what is read
z_read  = z[kz_read]

read3d(name) = Float64.(ds[name][:, :, kz_read, n])

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
    coords = (x, y, z_read)
    return [∂(a, j, coords[j]) for a in (u, v, w), j in 1:3]   # A[i, j] = ∂uⁱ/∂xʲ
end

const DERIVED_NAMES = ("Q", "enstrophy", "speed")

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

# The budget fields are all per-scale, written as <name>_ℓ<ℓ>. --scale lets them be named bare, so the
# six-panel budget figure is a readable command line rather than six copies of the suffix.
function resolve(name)
    name in DERIVED_NAMES && return name
    haskey(ds, name) && return name
    suffixed = isempty(scale) ? name : "$(name)_ℓ$(scale)"
    haskey(ds, suffixed) && return suffixed
    error("neither $name nor $suffixed is in $(basename(filepath)), and it is not one of $(join(DERIVED_NAMES, ", "))")
end

# Pretty panel titles. Anything not listed falls back to the variable's own name.
const TITLES = Dict("Q" => "Q  (vortex criterion)", "enstrophy" => "|ω|²", "speed" => "|u|",
                    "wb_rs" => "τ(w, b_r)  conversion", "Π_K" => "Π_K  KE flux", "Π_A" => "Π_A  APE flux",
                    "ε_Ks" => "ε_Kˢ  KE dissipation", "ε_As" => "ε_Aˢ  APE dissipation", "b" => "b")
pretty(bare, resolved) = get(TITLES, bare, resolved)

"""Read or derive one field, cropped, with its colour scale and isosurface levels."""
function prepare(bare)
    name = resolve(bare)
    a = name in DERIVED_NAMES ? derived(name) : read3d(name)
    a = a[:, :, kz]   # drop the one-cell margin the z derivative needed

    signed = minimum(a) < 0 < maximum(a)
    absq(p) = quantile(abs.(vec(a)), p / 100)
    crange = signed ? (-absq(pct), absq(pct)) : (quantile(vec(a), 1 - pct / 100), quantile(vec(a), pct / 100))
    cmap   = signed ? :balance : :magma

    # For a signed field take a symmetric pair per level so both senses show; for a positive one the upper
    # tail, where the structures are. Levels are kept inside crange: a level past its end renders saturated
    # and the outermost shell becomes indistinguishable from the next one in.
    levels = if signed
        f = collect(range(0.3, 0.85, max(nlevels ÷ 2, 1)))
        sort(vcat(-crange[2] .* f, crange[2] .* f))
    else
        clamp.([quantile(vec(a), q) for q in range(0.90, 0.995, nlevels)], crange[1], crange[2])
    end
    @info @sprintf("  %-12s -> %-12s  range [%.3g, %.3g] (%s), levels %s", bare, name, crange[1], crange[2],
                   signed ? "signed" : "positive", join((@sprintf("%.3g", l) for l in levels), ", "))
    return (; name, bare, data = a, colorrange = crange, colormap = cmap, levels)
end

@info @sprintf("t = %.1f, |z| < %.1f, %d x %d x %d", t, zlim, length(x), length(y), length(kz))
prepared = [prepare(f) for f in fields]
z = z_read[kz]
Re, Ri = Float64(ds.attrib["Re"]), Float64(ds.attrib["Ri"])
close(ds)
#---

#+++ Figure
Lx, Ly, Lz = x[end] - x[1], y[end] - y[1], z[end] - z[1]
xr, yr, zr = x[1] .. x[end], y[1] .. y[end], z[1] .. z[end]

# One field: panels are the modes. Several: panels are the fields, in the one mode given.
panels = length(prepared) == 1 && mode == "both" ?
         [(kind, only(prepared), kind == "volume" ? "volume (MIP)" : "isosurfaces") for kind in ("volume", "isosurface")] :
         [(mode, p, pretty(p.bare, p.name)) for p in prepared]

# Each panel carries its own colorbar: the budget terms differ by orders of magnitude (Π_K ~ 1e-3 against
# Q ~ 1e-1 on the test run), so one shared scale would flatten all but the largest.
ncols = min(cols, length(panels))
nrows = cld(length(panels), ncols)
# Wide and shallow per panel: after the z crop the domain is about 14:4.7:8, and viewed near side-on a
# square panel is mostly empty above and below the box. The colorbar column adds its own width.
fig = Figure(size = (720 * ncols, 520 * nrows))

for (i, (kind, p, title)) in enumerate(panels)
    row, col = fldmod1(i, ncols)
    gl = fig[row, col] = GridLayout()
    ax = Axis3(gl[1, 1]; aspect = (Lx, Ly, Lz), xlabel = "x", ylabel = "y", zlabel = "z",
               title, titlesize = 15, azimuth = azim * π, elevation = elev * π)
    if kind == "volume"
        # Maximum-intensity projection: no transfer function to tune, and it shows where the extremes
        # are. :absorption looks better but needs an opacity curve matched to the field's range.
        volume!(ax, xr, yr, zr, p.data; algorithm = :mip, colormap = p.colormap, colorrange = p.colorrange)
    else
        contour!(ax, xr, yr, zr, p.data; levels = p.levels, colormap = p.colormap,
                 colorrange = p.colorrange, alpha, transparency = true)
    end
    Colorbar(gl[1, 2]; colormap = p.colormap, colorrange = p.colorrange, height = Relative(0.6),
             ticklabelsize = 10, width = 12)
    colgap!(gl, 4)
end

Label(fig[0, :], @sprintf("t = %.1f      Re = %d,  Ri = %.2f%s", t, round(Int, Re), Ri,
                          isempty(scale) ? "" : @sprintf("      ℓ = %s", scale)), fontsize = 17)

# One field keeps its name in the filename; several would make it unreadable, so they become "budget6".
tag = length(fields) == 1 ? only(fields) : "$(length(fields))panel"
outfile = joinpath(figures, @sprintf("%s_3dvol_%s%s_t%.1f.png", stem, tag, isempty(scale) ? "" : "_l$scale", t))
save(outfile, fig; px_per_unit = 2)
@info "Figure saved to: $outfile"
#---
