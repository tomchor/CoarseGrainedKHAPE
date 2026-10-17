# Exploratory: volume render and isosurfaces of one snapshot of a 3D run.
#
#   julia --project postprocessing/plot7_3d_volume.jl <3d_output_filepath> [--field b] [--mode both] ...
#
# One --field lays the panels out by mode (volume and isosurface of it); a comma-separated list lays them
# out by field, one panel each, in the single --mode given. --scale appends the _ℓ<ℓ> every budget term
# carries, so the six-term budget figure is one readable line:
#
#   ... --field Q,wb_rs,Π_K,Π_A,ε_Ks,ε_As --scale 1 --mode isosurface --time 108
#
# Several times render in one process (`--times 100,120` or `--from 10 --to 200 [--every 4]`), one figure
# each, exactly as a single --time would draw it. Each panel is scaled by its own 99th percentile at that
# time unless --ranges says otherwise, which is what an animation needs: a scale fixed over the run, so a
# surface means the same value in every frame and growth and decay stay visible. Two passes:
#
#   ... --from 10 --measure ranges.toml                    # each panel's scale at every time, no rendering
#   ... --from 10 --ranges ranges.toml --frame-dir frames --movie anim.mp4
#
# GLMakie rather than CairoMakie: a volume render is GPU raymarching, which Cairo cannot do at all, and
# Makie's 3D `contour` wants the same backend. It needs an OpenGL context but no GPU: on Casper, Xvfb
# plus Mesa's software rasteriser gives one on a plain CPU node, which is what 04_render3d.pbs sets up.
#
# The isosurface is the more useful of the two modes for this flow -- Q or enstrophy picks out the
# streamwise vortices that braid the billow, which is the whole reason the simulation went 3D.

using NCDatasets
using GLMakie
using Printf
using Statistics
using TOML

GLMakie.activate!(visible = false)   # render offscreen; `save` still works

# Evenly spaced level fractions of a panel's colour range, used by --levels and by the isosurface overlay.
# Defined up here because the argument block below calls it. `range(lo, hi, 1)` throws when the endpoints
# differ, so a single pair takes the upper fraction alone -- the strongest surface, which is what one pair
# should show.
fractions(lo, hi, n) = n <= 1 ? [hi] : collect(range(lo, hi, n))


#+++ Arguments
# Deliberately hand-rolled rather than ArgParse: this is called by hand, and keeping it dependency-light
# means it also runs under `julia --project` with nothing precompiled beyond Makie and NCDatasets.
length(ARGS) > 0 || error("Usage: julia --project postprocessing/plot7_3d_volume.jl <3d_output_filepath> [--field F] [--mode M] ...")
filepath = ARGS[1]
opts = Dict{String,String}()
for i in 2:2:length(ARGS)-1
    startswith(ARGS[i], "--") || error("expected a --flag at argument $i, got $(ARGS[i])")
    opts[ARGS[i][3:end]] = ARGS[i+1]
end
# A field may be prefixed "-" to negate it, for a term that enters two budgets with opposite sign, and a
# lone "." leaves a cell empty, so a row with fewer terms than the grid is wide keeps its alignment.
# --field takes one name or a comma-separated list. One name lays the panels out by --mode (volume and/or
# isosurface of that field); several lay them out by field, one panel each, in the single --mode given.
fields  = String.(split(get(opts, "field", "b"), ","))
scale   = get(opts, "scale", "")               # appended as _ℓ<scale> to any name that needs it (see resolve)
mode    = get(opts, "mode", "both")            # volume | isosurface | both
cols    = parse(Int, get(opts, "cols", "2"))
tsel    = haskey(opts, "time") ? parse(Float64, opts["time"]) : nothing
# Several times: an explicit list, or a window [from, to] taken every record or every `every` in time. Each
# requested time goes to the nearest record, as --time does, and a record asked for twice renders once.
tlist   = haskey(opts, "times") ? [parse(Float64, s) for s in split(opts["times"], ",")] : nothing
t_from  = haskey(opts, "from") ? parse(Float64, opts["from"]) : nothing
t_to    = parse(Float64, get(opts, "to", "Inf"))
t_every = haskey(opts, "every") ? parse(Float64, opts["every"]) : nothing
count(!isnothing, (tsel, tlist, t_from)) <= 1 || error("--time, --times and --from are alternatives; give one")
!(haskey(opts, "to") || haskey(opts, "every")) || !isnothing(t_from) || error("--to and --every need --from")
isnothing(t_every) || t_every > 0 || error("--every must be positive; got $t_every")
# The fixed scale for an animation. --measure writes, for every panel (and the overlay), its --clim-percentile
# of |field| and its min and max at each requested time, and draws nothing. --ranges reads such a file and
# fixes each panel's range at --range-stat of that series (max, or pNN for a percentile over time) within
# --range-window FROM:TO, and fixes whether it is drawn signed: a field that changes sign at any of those
# times keeps its symmetric pair of surfaces throughout, rather than gaining or losing them mid-animation.
measure_file = get(opts, "measure", nothing)
ranges_file  = get(opts, "ranges", nothing)
isnothing(measure_file) || isnothing(ranges_file) || error("--measure and --ranges are the two passes; give one")
range_stat   = get(opts, "range-stat", "max")
range_stat == "max" || occursin(r"^p\d+(\.\d+)?$", range_stat) || error("--range-stat must be max or pNN; got $range_stat")
# Either end may be left open: "90:" is from t = 90 to the last measured time.
window_ends = split(get(opts, "range-window", ":"), ":")
length(window_ends) == 2 || error("--range-window takes FROM:TO, either end optional; got $(opts["range-window"])")
range_window = (isempty(window_ends[1]) ? -Inf : parse(Float64, window_ends[1]),
                isempty(window_ends[2]) ? Inf : parse(Float64, window_ends[2]))
# Frames for a movie: frame_0001.png, ... in --frame-dir, numbered in time order, instead of the usual named
# figures, so an animation can never overwrite a standalone figure. --movie joins them with Makie's own
# ffmpeg (FFMPEG_jll), at --fps frames per second.
frame_dir = get(opts, "frame-dir", nothing)
movie     = get(opts, "movie", nothing)
fps       = parse(Int, get(opts, "fps", "10"))
isnothing(movie) || !isnothing(frame_dir) || error("--movie joins the frames --frame-dir writes; give both")
zlim   = parse(Float64, get(opts, "zlim", "4.0"))
pct     = parse(Float64, get(opts, "clim-percentile", "99.0"))
# Where the isosurfaces sit, as fractions of each panel's own colour range. By default --levels n spreads
# n/2 of them evenly over 0.18-0.85; --fractions places them explicitly instead, e.g. "0.1,0.5,0.85". The
# two are alternatives, not a pair.
if haskey(opts, "levels") && haskey(opts, "fractions")
    error("--levels and --fractions both set; they are alternative ways to place the same surfaces")
end
nlevels = parse(Int, get(opts, "levels", "6"))
level_fracs = haskey(opts, "fractions") ?
              [parse(Float64, f) for f in split(opts["fractions"], ",")] :
              fractions(0.18, 0.85, nlevels ÷ 2)
all(0 .< level_fracs .<= 1) || error("--fractions must lie in (0, 1]; got $level_fracs")
alpha   = parse(Float64, get(opts, "alpha", "0.35"))
# The diverging map every panel is drawn on. :balance (cmocean) by default; the perceptually uniform
# Scientific Colour Maps (:vik, :broc, :berlin, :cork, ...) and the ColorBrewer names also work.
cmap_name = Symbol(get(opts, "colormap", "balance"))
# Camera, in units of π, as Makie's Axis3 takes it. Azimuth π looks straight down +x, so the streamwise
# direction is fully into the page; 1.5π looks along +y and gives x the full width but puts y edge-on and
# collides its labels. 1.3π is where x still spans nearly the whole frame and y keeps enough depth to read.
azim    = parse(Float64, get(opts, "azimuth", "1.30"))
elev    = parse(Float64, get(opts, "elevation", "0.12"))
# A second field drawn as isosurfaces over every panel, in grey, as a shared reference: b traces the
# deformed interface the budget terms live on, which is otherwise invisible in a Q or Π panel.
overlay   = get(opts, "overlay", "")
ov_alpha  = parse(Float64, get(opts, "overlay-alpha", "0.45"))
# One flat colour, not a colormap, and deliberately not grey: the panels use :balance (blue-white-red), whose
# middle *is* grey, so a grey overlay reads as washed-out data rather than as context. A saturated hue off that
# axis stays legibly separate. Any Makie colour name works.
ov_color  = Symbol(get(opts, "overlay-color", "seagreen"))
ov_clip   = parse(Float64, get(opts, "overlay-clip", "0.45"))
# Which velocity Q, enstrophy and speed are built from: the filtered field at --scale, or the full one.
# Filtered by default, so they sit on the same scale as the budget terms beside them. Note this makes them
# consistent in *velocity*, not in decomposition: Q(ū) is a resolved-scale quantity where Π_K, ε_Kˢ, ε_Aˢ
# and τ are sub-filter, and Q, being quadratic in gradients, has no natural resolved/sub-filter split.
# Rendered pixels per logical unit. The isosurfaces are a raymarched/transparent mesh, so the output is a
# raster however it is saved -- a PDF would just embed the same bitmap. Raising this is the only way to
# sharpen the edges; it does not touch the marching-cubes faceting, which is set by the data grid.
px_per_unit = parse(Float64, get(opts, "px-per-unit", "2"))
# A 3D contour is not a mesh: Makie raymarches the box and draws a step wherever the value lies within
# ±isorange of a level, lighting it with a normal finite-differenced at the step size. --samples is the
# number of steps per ray (800 by default: Makie's 200 is coarser than the data at Nz=1024, and 800 cost
# ~2 min per six-panel figure there against ~90 s). --isorange is the shell half-width in the normalised
# (-1, 1) units every panel is drawn in: 0.02 by default, and "auto" for Makie's own, a tenth of the
# smallest gap between levels (~0.034 at the default six). The visible surface is where a ray first meets
# level ± isorange, so a wider shell draws each surface that much further out: on the Nz=128 run, Q's
# −0.515 blobs show at ~0.034, vanish at 0.015 and spread into sheets at 0.06. More samples only sharpen
# the edges.
samples  = parse(Int, get(opts, "samples", "800"))
isorange_arg = get(opts, "isorange", "0.02")
isorange = isorange_arg == "auto" ? Makie.automatic : parse(Float64, isorange_arg)
derived_from = get(opts, "derived-from", "filtered")
derived_from in ("filtered", "full") || error("--derived-from must be filtered or full; got $derived_from")
# How the overlay is drawn. `wall` paints it on the bounding planes behind the data, `iso` draws it as
# isosurfaces in the box.
#
# `wall` is the default because `iso` does not work here, for a reason no amount of tuning fixes: two sets
# of interpenetrating isosurfaces always fight. Transparency cannot separate them, because GLMakie ignores
# alpha on a volume contour (--overlay-alpha 0.05 and 1.0 render byte-identically), so the overlay is a
# solid sheet across whatever the panel is about. Clipping it to a curtain helps a little and still looks
# cluttered. On the wall the same field is context, not a competing object, and occludes nothing.
ov_style  = get(opts, "overlay-style", "wall")
# Lines on a wall want more of them than surfaces in the volume want of themselves, so the default differs.
ov_levels = parse(Int, get(opts, "overlay-levels", ov_style == "wall" ? "10" : "1"))
# A deliberately low-contrast ramp for the walls: no black, no white. A full-range map (:bone, :grays)
# bottoms out at black, and a big black panel behind the data reads as a rendering fault rather than as
# background. Named maps still work via --overlay-colormap.
ov_cmap   = haskey(opts, "overlay-colormap") ? Symbol(opts["overlay-colormap"]) : cgrad([:gray88, :gray52])
ov_style in ("wall", "iso") || error("--overlay-style must be wall or iso; got $ov_style")
# A thumbnail (JFM's, for one): one field, no text at all -- no title, colourbar, axis labels or ticks --
# at an aspect ratio of exactly 1.2:1. --thumbnail W sets the width in figure units and the height is W/1.2,
# so W must be a multiple of 6 for a whole number of units; the pixels are those times --px-per-unit (the
# default 1200 at 2 gives 2400 x 2000). The box frame and the overlay's wall contours stay: they are what
# makes it read as a volume. The camera, levels and overlay are the usual flags.
thumb_w = haskey(opts, "thumbnail") ? parse(Int, opts["thumbnail"]) : nothing
# The thumbnail's background (any Makie colour; a dark one, e.g. black or "#0b1a2e", makes saturated surfaces
# stand out at the size a journal shows it) and whether the box's edges are drawn (--frame 0 leaves only the data).
thumb_bg    = parse(Makie.Colors.Colorant, get(opts, "background", "white"))
thumb_frame = get(opts, "frame", "1") == "1"
if !isnothing(thumb_w)
    thumb_w > 0 && thumb_w % 6 == 0 || error("--thumbnail W must be a positive multiple of 6, so W/1.2 is whole; got $thumb_w")
    length(fields) == 1 && only(fields) != "." || error("--thumbnail draws one field; got --field $(join(fields, ","))")
    mode != "both" || error("--thumbnail draws one panel; give --mode isosurface or volume")
    isnothing(frame_dir) && isnothing(measure_file) || error("--thumbnail makes stills; it takes no --frame-dir or --measure")
end
mode in ("volume", "isosurface", "both") || error("--mode must be volume, isosurface or both; got $mode")
length(fields) == 1 || mode != "both" ||
    error("--mode both lays panels out by mode, so it takes one --field; got $(length(fields)). Pick --mode isosurface or volume.")

repo_root = dirname(@__DIR__)
figures   = joinpath(repo_root, "figures")
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
# NaN rather than `missing` for masked values. A file that declares a _FillValue otherwise reads as
# Union{Missing, Float64}: Makie's contour! has no method for that (the wall overlay throws), and read3d's
# Float64.() throws on any value the fill value masks, such as the NaN last record of a tendency. Every
# variable of a 03_subset_3d.py subset declares one (xarray's default for floats); the simulation's own
# files declare none, so for them nothing changes. A masked value stays NaN, which `prepare` filters out.
ds = NCDataset(filepath, "r"; maskingvalue = NaN)
x, y, z = ds["x_caa"][:], ds["y_aca"][:], ds["z_aac"][:]
times   = ds["time"][:]
nearest(τ) = argmin(abs.(times .- τ))
records = if !isnothing(tlist)
    unique(nearest.(tlist))
elseif !isnothing(t_from)
    in_window = findall(τ -> t_from <= τ <= t_to, times)
    isempty(in_window) && error(@sprintf("no record in [%g, %g]; the run spans t = %g to %g", t_from, t_to, times[1], times[end]))
    isnothing(t_every) ? in_window :
        unique(nearest.(range(times[first(in_window)], times[last(in_window)] + 1e-9 * t_every; step = t_every)))
else
    [isnothing(tsel) ? length(times) : nearest(tsel)]
end
# `read3d` and `prepare` read the current record from these two globals, which the loop at the bottom sets.
n = first(records)
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
    all(≈(d[1]), d) || error("plot7 assumes a uniform grid along dimension $dim")
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

# The velocity the derived fields are built from. `--derived-from filtered` takes the online filtered field
# at --scale, which the simulation writes for every scale in its filter_ℓs; `full` takes the raw one. A
# filtered field that is not in the run falls back to the raw one and says so, rather than failing.
function velocity(component)
    derived_from == "full" && return read3d(component)
    filtered = isempty(scale) ? component : "$(component)_ℓ$scale"
    haskey(ds, filtered) && return read3d(filtered)
    @warn "$filtered is not in this run; building the derived fields from the unfiltered $component instead"
    return read3d(component)
end

function velocity_gradient()
    u, v, w = velocity("u"), velocity("v"), velocity("w")
    coords = (x, y, z_read)
    return [∂(a, j, coords[j]) for a in (u, v, w), j in 1:3]   # A[i, j] = ∂uⁱ/∂xʲ
end

const DERIVED_NAMES = ("Q", "enstrophy", "speed", "b_r")

# b_r = b - b✶(z), the buoyancy relative to the Winters sorted reference state. Only the 2D writer emits
# it, and that is one x-z slice, so it is rebuilt here. On a uniform grid the sort is exact and needs no
# iteration: order every cell by buoyancy, and since each model level holds Nx·Ny cells of equal volume,
# the k-th group of Nx·Ny sorted values is the fluid that comes to rest at level k. The sort has to see the
# *whole* column, not the z crop -- a reference state built from a slab is a different reference state --
# so this is the one derived field that reads beyond `kz_read`.
function reference_buoyancy_anomaly()
    b_full = Float64.(ds["b"][:, :, :, n])
    nx, ny, nz = size(b_full)
    b_star = vec(mean(reshape(sort(vec(b_full)), nx * ny, nz), dims = 1))   # ascending: lightest on top
    return b_full[:, :, kz_read] .- reshape(b_star[kz_read], 1, 1, :)
end

function derived(name)
    name == "b_r" && return reference_buoyancy_anomaly()
    if name == "speed"
        u, v, w = velocity("u"), velocity("v"), velocity("w")
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

# Panel titles as LaTeX, rendered by Makie's MathTeXEngine. Rich text put a subscript and a superscript
# side by side rather than stacked, so ε_Kˢ came out with the s trailing the K; LaTeX sets them over each
# other as the notation intends, and matches how plot6 writes the same terms in matplotlib mathtext.
# Stored as math-mode bodies without the $...$, so a panel letter and a minus sign can be prefixed before
# the string is wrapped. A name that is not listed falls back to plain text, since an arbitrary variable
# name (Π_K_ℓ1) is not valid LaTeX.
const TITLES = Dict(
    "Q"         => raw"Q \;\; \mathrm{vortex\ criterion}",
    "enstrophy" => raw"|\omega|^2",
    "speed"     => raw"|\mathbf{u}|",
    "b"         => raw"b",
    "b_r"       => raw"b_r \;\; \mathrm{buoyancy\ anomaly}",
    "wb_rs"     => raw"\tau(w, b_r) \;\; \mathrm{conversion}",
    "Π_K"       => raw"\Pi_K \;\; \mathrm{KE\ flux}",
    "Π_A"       => raw"\Pi_A \;\; \mathrm{APE\ flux}",
    "ε_Ks"      => raw"\varepsilon_K^s \;\; \mathrm{KE\ dissipation}",
    "ε_As"      => raw"\varepsilon_A^s \;\; \mathrm{APE\ dissipation}",
    "K_s"       => raw"K^s \;\; \mathrm{subfilter\ KE}",
    "E_as"      => raw"E_a^s \;\; \mathrm{subfilter\ APE}",
    "R_s"       => raw"R^s \;\; \mathrm{reference\ tendency}",
    "dKs_dt"    => raw"\partial_t K^s \;\; \mathrm{KE\ tendency}",
    "dEas_dt"   => raw"\partial_t E_a^s \;\; \mathrm{APE\ tendency}",
)

"""The first title line: the term, with its panel letter and, for a negated field, a minus sign."""
function panel_title(p, letter)
    body = get(TITLES, p.bare, nothing)
    isnothing(body) && return (isnothing(letter) ? "" : "($letter)  ") * (p.neg ? "−" : "") * p.name
    prefix = isnothing(letter) ? "" : "(\\mathrm{$letter})" * raw"\;\;"
    return Makie.LaTeXStrings.latexstring(prefix * (p.neg ? "-" : "") * body)
end

# Every number in the title in one fixed-width form: three significant figures in scientific notation, the
# integral always signed, and U+2212 for the minus, which is as wide as + in the figure font where a hyphen is
# not. The line is centred, so with %g its width changed with every digit count and sign (0.359, 0.0203,
# -0.0701) and the whole line slid sideways from frame to frame of an animation; measured, this holds it to
# the same width for any value.
fixed_width(v; signed = true) = replace(signed ? @sprintf("%+.2e", v) : @sprintf("%.2e", v), "-" => "−")

"""The second title line: the panel's own range, and its volume integral where the file carries one."""
function scale_line(p)
    s = "range = ±" * fixed_width(p.scale; signed = false)
    p.fixed && (s *= " (fixed)")
    isnothing(p.integral) || (s *= "    ∫dV = " * fixed_width(p.integral))
    return s
end


"""One field at the current record, read or derived and cropped, with its finite values."""
function field_at_record(spec)
    neg  = startswith(spec, "-")
    bare = neg ? spec[2:end] : spec
    name = resolve(bare)
    a = name in DERIVED_NAMES ? derived(name) : read3d(name)
    a = a[:, :, kz]   # drop the one-cell margin the z derivative needed
    neg && (a = -a)
    # The tendencies come from Oceananigans' deferred TimeDerivative, whose last record of a run is NaN
    # (its differencing window never closes), so the quantile is taken over the finite values only.
    finite = filter(isfinite, vec(a))
    return (; neg, bare, name, a, finite)
end

"""What --measure records of one field at the current record: the percentile --ranges fixes, and the extremes
that decide whether it is drawn signed. All NaN when no value is finite (a tendency's last record)."""
function field_stats(spec)
    (; finite) = field_at_record(spec)
    isempty(finite) && return (NaN, NaN, NaN)
    return (quantile(abs.(finite), pct / 100), minimum(finite), maximum(finite))
end

"""Each panel's fixed (range, signed) from a --measure file: --range-stat of its series within --range-window,
and signed if it changed sign at any of those times; with the measured times that window actually spans, which
the colourbar label states."""
function fixed_ranges(file)
    d = TOML.parsefile(file)
    meta = d["meta"]
    # A scale measured on a different crop, percentile, filter scale or velocity is a different number, so it
    # is refused rather than applied. The file name only warns: a 03_subset_3d.py subset of a run measures the same.
    for (key, here) in (("zlim", zlim), ("clim_percentile", pct), ("scale", scale), ("derived_from", derived_from))
        meta[key] == here || error("$file was measured with $key = $(meta[key]), but this render has $here")
    end
    meta["file"] == basename(filepath) || @warn "$file was measured on $(meta["file"]), not $(basename(filepath))"
    ts = Float64.(meta["times"])
    sel = findall(τ -> range_window[1] <= τ <= range_window[2], ts)
    isempty(sel) && error("no measured time in --range-window $(range_window); $file holds t = $(ts[1]) to $(ts[end])")
    q = range_stat == "max" ? nothing : parse(Float64, range_stat[2:end]) / 100
    fields = Dict(map(collect(d["fields"])) do (name, s)
        p, mn, mx = (Float64.(s[k])[sel] for k in ("p", "min", "max"))
        ok = isfinite.(p)
        any(ok) || error("$name is non-finite at every measured time in the window")
        hi = isnothing(q) ? maximum(p[ok]) : quantile(p[ok], q)
        name => (; hi, signed = any(mn[ok] .< 0 .< mx[ok]))
    end)
    return (; fields, span = (ts[first(sel)], ts[last(sel)]))
end
fixed = isnothing(ranges_file) ? nothing : fixed_ranges(ranges_file)

"""Read or derive one field, cropped, with its colour scale and isosurface levels."""
function prepare(spec)
    (; neg, bare, name, a, finite) = field_at_record(spec)
    isempty(finite) && error("every value of $name at t = $t is non-finite; the last record of a run is " *
                             "NaN for the tendencies and anything built on them -- pick an earlier --time")
    signed = minimum(finite) < 0 < maximum(finite)   # still decides the *levels*: a symmetric pair, or one-sided
    absq(p) = quantile(abs.(finite), p / 100)
    # Every panel gets the same symmetric :balance scale, white at zero, including the positive-definite
    # ones. For those the lower half goes unused, which is the point twice over: the colourbars become the
    # same object, so magnitudes compare across panels without rescaling by eye, and the empty half states
    # that the quantity never changes sign -- true of ε_Kˢ by construction and of nothing else here.
    # The alternative, the upper half of the ramp over (0, hi), renders identically (v maps to 0.5 + v/2hi
    # either way) and differs only in showing no empty half, so it says less for the same picture.
    # Each panel is normalised by its own 99th percentile and drawn on one (-1, 1) scale, so the figure
    # carries a single colourbar instead of one per panel, and a surface at a given colour means the same
    # fraction of the local range wherever it appears. The physical scale moves into the title, where it
    # reads as a number instead of having to be decoded off a bar of five-digit ticks.
    hi     = absq(pct)
    # Under --ranges both the scale and the signedness come from the run, not from this record.
    if !isnothing(fixed)
        haskey(fixed.fields, name) || error("$name is not in $ranges_file; measure it with the same --field and --overlay")
        hi, signed = fixed.fields[name].hi, fixed.fields[name].signed
    end
    crange = (-1.0, 1.0)
    cmap   = cgrad(cmap_name)

    # For a signed field take a symmetric pair per level so both senses show; for a positive one the upper
    # tail, where the structures are. Levels are kept inside crange: a level past its end renders saturated
    # and the outermost shell becomes indistinguishable from the next one in.
    # One set of fractions for every panel, so a surface means the same thing wherever it appears: this
    # much of the panel's own range. A positive field gets exactly the magnitudes the signed panels put
    # their red surfaces at, and simply has no blue counterparts -- so `--levels n` is n surfaces when the
    # field changes sign and n/2 when it does not. Earlier revisions placed positive levels by quantile of
    # the field's own distribution, which put them where the data was but at magnitudes unrelated to any
    # other panel's, so nothing could be read across the figure.
    f = level_fracs
    levels = signed ? sort(vcat(-f, f)) : f
    a = a ./ hi
    # The budget terms carry their own volume integral, which is the number the closure is stated in and
    # the one thing a self-scaled panel cannot show: without it the panels are six shapes with no sense of
    # which dominates. Note it is over the *whole* domain while the panel is cropped in z, so it is the
    # budget's number rather than a sum over what is drawn. Derived fields have none and get no annotation.
    int_name = name * "_int"
    integral = haskey(ds, int_name) ? Float64(ds[int_name][n]) : nothing
    neg && !isnothing(integral) && (integral = -integral)
    # ±hi, not crange: crange is the normalised (-1, 1) every panel is drawn on, which says nothing.
    @info @sprintf("  %-12s -> %-12s  range [%.3g, %.3g] (%s%s)%s  levels %s", bare, name, -hi, hi,
                   signed ? "signed" : "positive", isnothing(fixed) ? "" : ", fixed",
                   isnothing(integral) ? "" : @sprintf("  ∫dV = %.3g", integral),
                   join((@sprintf("%.3g", l) for l in levels), ", "))
    return (; name, bare, neg, data = a, colorrange = crange, colormap = cmap, levels, integral, scale = hi,
            fixed = !isnothing(fixed))
end

any(!=("."), fields) || error("--field is all blanks")
z = z_read[kz]
Re, Ri = Float64(ds.attrib["Re"]), Float64(ds.attrib["Ri"])
#---

#+++ Figure
# Panel letters, for referring to a panel in text. They prefix the title rather than sitting inside the
# box: an annotation inside an Axis3 would need 3D data coordinates and would move with the camera, where
# the title is already parked in the cell's top protrusion. Blank cells take no letter, so the sequence
# counts drawn panels; a single-panel figure gets none, having nothing to distinguish.
const PANEL_LETTERS = 'a':'z'

"""Draw the figure of the current record from its prepared panels and overlay, and save it to `outfile`."""
function render(prepared, ov, outfile)
    Lx, Ly, Lz = x[end] - x[1], y[end] - y[1], z[end] - z[1]
    xr, yr, zr = x[1] .. x[end], y[1] .. y[end], z[1] .. z[end]
    # Makie's Axis3 autoscales its limits to the data extent plus 5% on every side, which is what makes the
    # box read as a box rather than as the surface of the field. Those limits are reproduced explicitly here,
    # unchanged, so the padding is a known number rather than an implementation detail -- `aspect` stays on the
    # data extents, as it was, so the panel looks exactly as it did.
    pad_x, pad_y, pad_z = 0.05Lx, 0.05Ly, 0.05Lz
    x_wall, y_wall = x[end] + pad_x, y[end] + pad_y
    box_limits = (x[1] - pad_x, x_wall, y[1] - pad_y, y_wall, z[1] - pad_z, z[end] + pad_z)
    # A wall contour drawn over the data's own coordinates covers only the data footprint, leaving a bare strip
    # along each edge of the wall where the box is wider -- most visible as a gap at the far end of the back
    # wall. Extending the coordinates to the box and repeating the edge slice into the padding runs every
    # contour flat out to the corners.
    x_span, y_span = vcat(x[1] - pad_x, x, x_wall), vcat(y[1] - pad_y, y, y_wall)
    z_span = vcat(z[1] - pad_z, z, z[end] + pad_z)
    edge_pad(a) = (b = vcat(a[1:1, :], a, a[end:end, :]); hcat(b[:, 1:1], b, b[:, end:end]))

    """The panel's own field, and the overlay on its walls or in its volume."""
    function draw_panel!(ax, kind, p, ov)
        if kind == "volume"
            # Maximum-intensity projection: no transfer function to tune, and it shows where the extremes
            # are. :absorption looks better but needs an opacity curve matched to the field's range.
            volume!(ax, xr, yr, zr, p.data; algorithm = :mip, colormap = p.colormap, colorrange = p.colorrange)
        else
            c = contour!(ax, xr, yr, zr, p.data; levels = p.levels, colormap = p.colormap,
                         colorrange = p.colorrange, alpha, transparency = true, isorange)
            # contour! takes no `samples`; it lives on the volume plot the recipe draws with.
            c.plots[1].samples[] = samples
        end
        # The overlay goes on last so Makie sorts it in front; grey keeps it off the panel's own colour scale,
        # and its levels are a symmetric pair about the field's middle, which for b is the interface either side.
        if ov !== nothing && ov_style == "wall"
            # Two bounding planes, textured with the overlay: the far spanwise wall and the floor. `surface!`
            # with one coordinate held constant is the only way to put a 2D field on an arbitrary plane in a
            # 3D axis -- Makie's `heatmap!` is xy-only. They sit at the box edges, so nothing in the volume is
            # hidden, and a pale sequential map keeps them visibly background against the panel's own data.
            # Line contours, not shading: a filled wall is a second data layer competing with the panel's own,
            # and no colormap makes it recede enough. Lines read as annotation. They go on the two *vertical*
            # walls only -- b at fixed z is nearly uniform, so a floor is one flat colour carrying nothing.
            # `transformation` is how Makie puts a 2D recipe on a plane of a 3D axis.
            wall_lv = collect(range(ov.colorrange[1], ov.colorrange[2], ov_levels + 2))[2:end-1]
            wall_kw = (; levels = wall_lv, color = :gray40, linewidth = 1.0)
            contour!(ax, x_span, z_span, edge_pad(ov.data[:, end, :]); transformation = (:xz, y_wall), wall_kw...)
            contour!(ax, y_span, z_span, edge_pad(ov.data[end, :, :]); transformation = (:yz, x_wall), wall_kw...)
        elseif ov !== nothing
            # One surface means the field's own middle, which for b is the interface itself. More than one
            # takes a symmetric pair about it, which for a rolled-up billow quickly becomes opaque.
            ov_lv = ov_levels <= 1 ? [0.0] :
                    sort(vcat(-ov.colorrange[2] .* fractions(0.25, 0.7, ov_levels ÷ 2),
                               ov.colorrange[2] .* fractions(0.25, 0.7, ov_levels ÷ 2)))
            jmax = clamp(round(Int, ov_clip * length(y)), 2, length(y))
            contour!(ax, xr, y[1] .. y[jmax], zr, ov.data[:, 1:jmax, :]; levels = ov_lv,
                     colormap = [ov_color, ov_color], colorrange = ov.colorrange,
                     alpha = ov_alpha, transparency = true)
        end
    end

    if !isnothing(thumb_w)
        # Nothing but the box and what is in it: no protrusions, so the box fills the canvas as far as its
        # aspect allows, and hidedecorations! takes the labels, ticks and grid. The padding is a small margin
        # all round; at none the box's outer edges fall on the canvas edge and are cut in half.
        fig = Figure(size = (thumb_w, thumb_w * 5 ÷ 6), figure_padding = thumb_w ÷ 60, backgroundcolor = thumb_bg)
        ax = Axis3(fig[1, 1]; aspect = (Lx, Ly, Lz), azimuth = azim * π, elevation = elev * π,
                   limits = box_limits, viewmode = :fitzoom, protrusions = 0)
        hidedecorations!(ax)
        thumb_frame || hidespines!(ax)
        draw_panel!(ax, mode, only(prepared), ov)
        save(outfile, fig; px_per_unit = px_per_unit)
        h, w = size(Makie.FileIO.load(outfile))
        w * 5 == h * 6 || error("$outfile is $w x $h, not 1.2:1")
        @info "Thumbnail ($w x $h px) saved to: $outfile"
        return
    end

    # One field: panels are the modes. Several: panels are the fields, in the one mode given.
    panels = length(prepared) == 1 && mode == "both" ?
             [(kind, only(prepared), kind == "volume" ? "volume (MIP)" : "isosurfaces") for kind in ("volume", "isosurface")] :
             [(mode, p, nothing) for p in prepared]

    # Each panel carries its own colorbar: the budget terms differ by orders of magnitude (Π_K ~ 1e-3 against
    # Q ~ 1e-1 on the test run), so one shared scale would flatten all but the largest.
    ncols = min(cols, length(panels))
    nrows = cld(length(panels), ncols)
    # Wide and shallow per panel: after the z crop the domain is about 14:4.7:8, and viewed near side-on a
    # square panel is mostly empty above and below the box. The colorbar column adds its own width.
    # figure_padding and the inter-panel gaps below are the binding constraint on how large the boxes are
    # drawn, not the cell size: with Makie's defaults the box fills about half its cell. `viewmode = :fitzoom`
    # on each axis then zooms the box to the space that frees up -- on its own, at default padding, it does
    # nothing visible.
    fig = Figure(size = (640 * ncols + 120, 520 * nrows), figure_padding = 4)

    # cumsum gives each panel its position among the drawn ones.
    panel_letter = cumsum(!isnothing(q[2]) for q in panels)
    n_drawn = last(panel_letter)

    for (i, (kind, p, title)) in enumerate(panels)
        row, col = fldmod1(i, ncols)
        isnothing(p) && continue   # a "." in --field: the cell stays empty so the row keeps its alignment
        letter = n_drawn > 1 ? PANEL_LETTERS[panel_letter[i]] : nothing
        # `title` is set for the --mode both panels (a plain mode name); otherwise it is built from the field.
        title_tex = isnothing(title) ? panel_title(p, letter) :
                    (isnothing(letter) ? title : "($letter)  " * title)
        # The title is two Labels in a GridLayout hung in the cell's top protrusion. Every simpler arrangement
        # fails: Axis3's own `title` takes no line break and rejects rich text; two Labels placed directly in
        # the protrusion overlap rather than stack; Axis3's title under a protrusion Label overlaps too; and a
        # LaTeX \substack spanning both lines does not parse in MathTeXEngine. A GridLayout *in the
        # protrusion* stacks them and still leaves the axis the whole cell -- unlike one in the cell itself,
        # which shrinks the axis into a corner.
        gl = GridLayout(fig[row, col, Top()])
        Label(gl[1, 1], title_tex;      fontsize = 19, padding = (0, 0, 4, 0))
        Label(gl[2, 1], scale_line(p);  fontsize = 15, padding = (0, 0, 2, 0))
        rowgap!(gl, 0)
        ax = Axis3(fig[row, col]; aspect = (Lx, Ly, Lz), xlabel = "x", ylabel = "y", zlabel = "z",
                   azimuth = azim * π, elevation = elev * π, limits = box_limits, viewmode = :fitzoom)
        draw_panel!(ax, kind, p, ov)
        # Every panel shares one grid and one camera, so repeating the axis labels and tick labels six times is
        # ink that says nothing. They stay on the first panel and come off the rest; the box and grid stay
        # everywhere, so each panel still reads as the same volume.
        i == 1 || hidedecorations!(ax; grid = false)
    end

    # One colourbar for the figure: every panel is on the same normalised scale, so six of them said the same
    # thing six times. Its ticks are fractions of each panel's own range; the ranges themselves are in the titles.
    # Under --ranges the label also says which times the fixed ranges come from: a frame outside them may exceed
    # its panels' ranges, which no surface can show, since every one sits at or inside 0.85 of the range. Here
    # rather than in the figure title, because it qualifies exactly what this label defines.
    cbar_label = "fraction of each panel's range" *
                 (isnothing(fixed) ? "" : @sprintf("\n(ranges: %s over t = %g–%g)", range_stat, fixed.span...))
    Colorbar(fig[1:nrows, ncols + 1]; colormap = cgrad(cmap_name), colorrange = (-1.0, 1.0),
             ticks = ([-1, -0.5, 0, 0.5, 1], ["-1", "-0.5", "0", "0.5", "1"]),
             label = cbar_label, height = Relative(0.5), width = 14,
             labelsize = 20, ticklabelsize = 18)

    colgap!(fig.layout, 0)
    rowgap!(fig.layout, 0)
    # The time padded to five figures with U+2007, the figure space, which is a digit wide: "t = 90.0" and
    # "t = 100.0" are then one width, and the centred title does not shift as an animation passes t = 100.
    t_str = lpad(@sprintf("%.1f", t), 5, '\u2007')
    Label(fig[0, :], @sprintf("t = %s      Re = %d,  Ri = %.2f%s", t_str, round(Int, Re), Ri,
                              isempty(scale) ? "" : @sprintf("      ℓ = %s", scale)),
          fontsize = 17, padding = (0, 0, 0, 10))

    save(outfile, fig; px_per_unit = px_per_unit)
    @info "Figure saved to: $outfile"
end
#---

#+++ Render, or measure, every requested record
# One field keeps its name in the filename; several would make it unreadable, so they become "6panel".
tag = length(fields) == 1 ? only(fields) : "$(length(fields))panel"
figure_name(t) = joinpath(figures, @sprintf("%s_3dvol_%s%s%s_t%.1f.png", stem, tag, isempty(scale) ? "" : "_l$scale",
                                            isnothing(thumb_w) ? "" : "_thumbnail", t))
# Stale frames from an earlier, longer run would otherwise be joined into this one's movie.
if !isnothing(frame_dir)
    mkpath(frame_dir)
    foreach(f -> rm(joinpath(frame_dir, f)), filter(f -> occursin(r"^frame_\d{4}\.png$", f), readdir(frame_dir)))
end
measured = Dict{String,Any}()   # name => (p, min, max) series, for --measure
specs = unique([[f for f in fields if f != "."]; isempty(overlay) ? String[] : [overlay]])
failed = 0
for (frame, record) in enumerate(records)
    global n = record
    global t = times[n]
    @info @sprintf("t = %.1f, |z| < %.1f, %d x %d x %d", t, zlim, length(x), length(y), length(kz))
    if !isnothing(measure_file)
        # A field measured once whether it is drawn plain or negated: the scale is of |field|, and the sign
        # flips min and max, which is undone here so the file holds the field itself.
        for spec in specs
            name = resolve(startswith(spec, "-") ? spec[2:end] : spec)
            haskey(measured, name) && length(measured[name][1]) == frame && continue
            p, mn, mx = field_stats(name)
            s = get!(measured, name, (Float64[], Float64[], Float64[]))
            push!(s[1], p); push!(s[2], mn); push!(s[3], mx)
            @info @sprintf("  %-12s  p%g |·| = %.3g   min %.3g   max %.3g", name, pct, p, mn, mx)
        end
        continue
    end
    # A record that fails is reported and the rest continue; the process exits non-zero at the end, so a
    # batch job is marked failed rather than leaving the previous figure at that time looking like a result.
    try
        # Timed in two parts, so a job's log shows whether reading and deriving the fields or drawing them dominates.
        t_start = time()
        prepared = [f == "." ? nothing : prepare(f) for f in fields]
        ov = isempty(overlay) ? nothing : prepare(overlay)
        t_drawn = time()
        render(prepared, ov, isnothing(frame_dir) ? figure_name(t) : joinpath(frame_dir, @sprintf("frame_%04d.png", frame)))
        @info @sprintf("t = %.1f took %.1f s: %.1f s preparing the fields, %.1f s rendering", t, time() - t_start,
                       t_drawn - t_start, time() - t_drawn)
    catch err
        global failed = failed + 1
        @error "t = $t failed" exception = (err, catch_backtrace())
    end
end
close(ds)

if !isnothing(measure_file)
    meta = Dict("file" => basename(filepath), "zlim" => zlim, "clim_percentile" => pct, "scale" => scale,
                "derived_from" => derived_from, "times" => times[records])
    fields_out = Dict(name => Dict("p" => s[1], "min" => s[2], "max" => s[3]) for (name, s) in measured)
    mkpath(dirname(abspath(measure_file)))
    open(io -> TOML.print(io, Dict("meta" => meta, "fields" => fields_out)), measure_file, "w")
    @info "Measured $(length(records)) records of $(length(measured)) fields into $measure_file"
end

failed == 0 || (@error "$failed of $(length(records)) records failed; see above"; exit(1))

if !isnothing(movie)
    # yuv420p, which every player reads, needs even dimensions; the pad adds at most a pixel to each.
    mkpath(dirname(abspath(movie)))
    run(`$(Makie.FFMPEG_jll.ffmpeg()) -y -loglevel error -framerate $fps -i $(joinpath(frame_dir, "frame_%04d.png"))
         -c:v libx264 -pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" $movie`)
    @info "Movie of $(length(records)) frames at $fps fps saved to: $movie"
end
#---
