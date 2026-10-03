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
# Makie's 3D `contour` wants the same backend. It needs an OpenGL context but no GPU: on Casper, Xvfb
# plus Mesa's software rasteriser gives one on a plain CPU node, which is what render3d.pbs sets up.
#
# The isosurface is the more useful of the two modes for this flow -- Q or enstrophy picks out the
# streamwise vortices that braid the billow, which is the whole reason the simulation went 3D.

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
nlevels = parse(Int,     get(opts, "levels", "6"))
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

# Panel titles as Makie rich text, so the sub- and superscripts that name these terms are typeset rather
# than spelled out in lookalike unicode: ε with a subscript K and a superscript s, not "ε_Kˢ". Anything
# unlisted falls back to the variable's own name.
const TITLES = Dict(
    "Q"         => rich("Q", "  vortex criterion"),
    "enstrophy" => rich("|ω|", superscript("2")),
    "speed"     => rich("|u|"),
    "b"         => rich("b"),
    "b_r"       => rich("b", subscript("r"), "  buoyancy anomaly"),
    "wb_rs"     => rich("τ(w, b", subscript("r"), ")  conversion"),
    "Π_K"       => rich("Π", subscript("K"), "  KE flux"),
    "Π_A"       => rich("Π", subscript("A"), "  APE flux"),
    "ε_Ks"      => rich("ε", subscript("K"), superscript("s"), "  KE dissipation"),
    "ε_As"      => rich("ε", subscript("A"), superscript("s"), "  APE dissipation"),
    "K_s"       => rich("K", superscript("s"), "  sub-filter KE"),
    "E_as"      => rich("E", subscript("a"), superscript("s"), "  sub-filter APE"),
    "R_s"       => rich("R", superscript("s"), "  reference tendency"),
)
pretty(bare, resolved) = get(TITLES, bare, rich(resolved))

"""The second title line: the panel's own range, and its volume integral where the file carries one."""
function scale_line(p)
    s = @sprintf("range = ±%.3g", p.scale)
    isnothing(p.integral) || (s *= @sprintf("    ∫dV = %.3g", p.integral))
    return s
end

# Level fractions of the colour range. `range(lo, hi, 1)` throws when the endpoints differ, so a single
# pair takes the upper fraction alone -- the strongest surface, which is what one pair should show.
fractions(lo, hi, n) = n <= 1 ? [hi] : collect(range(lo, hi, n))

"""Read or derive one field, cropped, with its colour scale and isosurface levels."""
function prepare(bare)
    name = resolve(bare)
    a = name in DERIVED_NAMES ? derived(name) : read3d(name)
    a = a[:, :, kz]   # drop the one-cell margin the z derivative needed

    signed = minimum(a) < 0 < maximum(a)   # still decides the *levels*: a symmetric pair, or one-sided
    absq(p) = quantile(abs.(vec(a)), p / 100)
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
    f = fractions(0.18, 0.85, nlevels ÷ 2)
    levels = signed ? sort(vcat(-f, f)) : f
    a = a ./ hi
    # The budget terms carry their own volume integral, which is the number the closure is stated in and
    # the one thing a self-scaled panel cannot show: without it the panels are six shapes with no sense of
    # which dominates. Note it is over the *whole* domain while the panel is cropped in z, so it is the
    # budget's number rather than a sum over what is drawn. Derived fields have none and get no annotation.
    int_name = name * "_int"
    integral = haskey(ds, int_name) ? Float64(ds[int_name][n]) : nothing
    @info @sprintf("  %-12s -> %-12s  range [%.3g, %.3g] (%s)%s", bare, name, crange[1], crange[2],
                   signed ? "signed" : "positive",
                   isnothing(integral) ? "" : @sprintf("  ∫dV = %.3g", integral))
    return (; name, bare, data = a, colorrange = crange, colormap = cmap, levels, integral, scale = hi)
end

@info @sprintf("t = %.1f, |z| < %.1f, %d x %d x %d", t, zlim, length(x), length(y), length(kz))
prepared = [prepare(f) for f in fields]
ov = isempty(overlay) ? nothing : prepare(overlay)
z = z_read[kz]
Re, Ri = Float64(ds.attrib["Re"]), Float64(ds.attrib["Ri"])
close(ds)
#---

#+++ Figure
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

# One field: panels are the modes. Several: panels are the fields, in the one mode given.
panels = length(prepared) == 1 && mode == "both" ?
         [(kind, only(prepared), rich(kind == "volume" ? "volume (MIP)" : "isosurfaces")) for kind in ("volume", "isosurface")] :
         [(mode, p, rich(pretty(p.bare, p.name), "\n", rich(scale_line(p), fontsize = 13))) for p in prepared]

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

for (i, (kind, p, title)) in enumerate(panels)
    row, col = fldmod1(i, ncols)
    # The title is a Label above the axis, not Axis3's own `title`: that attribute takes a plain string or
    # a LaTeXString but rejects rich text, and no form of it accepts a line break, so neither the typeset
    # subscripts nor the second line of scale and integral would survive it.
    # `fig[row, col, Top()]` hangs the label in the cell's top protrusion rather than taking a grid row,
    # so the axis keeps the whole cell. A nested GridLayout with a label row shrinks the axis to a corner.
    Label(fig[row, col, Top()], title; fontsize = 15, font = :bold, padding = (0, 0, 6, 0))
    ax = Axis3(fig[row, col]; aspect = (Lx, Ly, Lz), xlabel = "x", ylabel = "y", zlabel = "z",
               azimuth = azim * π, elevation = elev * π, limits = box_limits, viewmode = :fitzoom)
    if kind == "volume"
        # Maximum-intensity projection: no transfer function to tune, and it shows where the extremes
        # are. :absorption looks better but needs an opacity curve matched to the field's range.
        volume!(ax, xr, yr, zr, p.data; algorithm = :mip, colormap = p.colormap, colorrange = p.colorrange)
    else
        contour!(ax, xr, yr, zr, p.data; levels = p.levels, colormap = p.colormap,
                 colorrange = p.colorrange, alpha, transparency = true)
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
    # Every panel shares one grid and one camera, so repeating the axis labels and tick labels six times is
    # ink that says nothing. They stay on the first panel and come off the rest; the box and grid stay
    # everywhere, so each panel still reads as the same volume.
    i == 1 || hidedecorations!(ax; grid = false)
end

# One colourbar for the figure: every panel is on the same normalised scale, so six of them said the same
# thing six times. Its ticks are fractions of each panel's own range; the ranges themselves are in the titles.
Colorbar(fig[1:nrows, ncols + 1]; colormap = cgrad(cmap_name), colorrange = (-1.0, 1.0),
         ticks = ([-1, -0.5, 0, 0.5, 1], ["-1", "-0.5", "0", "0.5", "1"]),
         label = "fraction of each panel's range", height = Relative(0.5), width = 14)

colgap!(fig.layout, 0)
rowgap!(fig.layout, 0)
Label(fig[0, :], @sprintf("t = %.1f      Re = %d,  Ri = %.2f%s", t, round(Int, Re), Ri,
                          isempty(scale) ? "" : @sprintf("      ℓ = %s", scale)),
      fontsize = 17, padding = (0, 0, 0, 10))

# One field keeps its name in the filename; several would make it unreadable, so they become "budget6".
tag = length(fields) == 1 ? only(fields) : "$(length(fields))panel"
outfile = joinpath(figures, @sprintf("%s_3dvol_%s%s_t%.1f.png", stem, tag, isempty(scale) ? "" : "_l$scale", t))
save(outfile, fig; px_per_unit = 2)
@info "Figure saved to: $outfile"
#---
