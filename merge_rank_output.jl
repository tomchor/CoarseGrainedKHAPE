# Stitches the per-rank NetCDF files of a distributed run (`--ranks R`, x-slabs) into the two files a single-GPU run
# writes, so that nothing downstream (the Python pipeline, `plot_kelvin_helmholtz_instability.jl`) knows the difference:
#
#   julia --project merge_rank_output.jl <dir>/khi_Nz1024_Ri0.10 [--delete]
#
# merges <stem>_rank<r>.nc into <stem>.nc and <stem>_2d_rank<r>.nc into <stem>_2d.nc, r = 0, …, R-1, with R read from
# the files' own `ranks` attribute. `--delete` removes the rank files once both merges have succeeded.
#
# What a rank file holds: its own x-slab of every field, on the dimensions a single-GPU file has (`x_caa`, `x_faa`,
# …), with real coordinates; every variable without an x dimension (time, the `_int` volume integrals, ε̄, L_K, …)
# whole and identical on every rank, since the reductions are all-reduced; and the grid of its own slab in the grid
# metadata groups. So the merge
#   - concatenates every variable with an x dimension in rank order, record by record (a slab at a time in memory),
#     and checks the x coordinates come out increasing;
#   - copies every other variable from rank 0, after checking that all ranks agree bit for bit;
#   - rewrites each variable's `indices` attribute, which records the slab's x range, to the whole domain's;
#   - writes the grid metadata groups for the whole domain with Oceananigans' own writer, on the CPU (a file that
#     says `GPU()` cannot be read back where CUDA is not loaded, e.g. by the merge job's own animation).

using NCDatasets
using Oceananigans
using Oceananigans: RectilinearGrid, CPU, Periodic, Bounded

const NCExt = Base.get_extension(Oceananigans, :OceananigansNCDatasetsExt)

is_x_dim(name) = startswith(name, "x_")

#+++ Rank files
function rank_files(stem)
    first = stem * "_rank0.nc"
    isfile(first) || error("no rank files to merge: $first does not exist")
    R = NCDataset(ds -> Int(ds.attrib["ranks"]), first)
    files = [stem * "_rank$(r).nc" for r in 0:R-1]
    absent = filter(!isfile, files)
    isempty(absent) || error("$(length(absent)) of the $R rank files of $stem are missing: $(join(absent, ", "))")
    return files
end
#---

#+++ Attributes
# `indices` is written as e.g. "(1:72, 1:1, 1:1024)": the first range is the slab's, so it becomes the whole x
function global_indices(indices::AbstractString, dims, global_size)
    x = findfirst(is_x_dim, dims)
    isnothing(x) && return indices
    merged = replace(indices, r"^\(\s*\d+:\d+" => "(1:$(global_size[dims[x]])")
    merged == indices && error("unexpected `indices` attribute $(repr(indices)) on a variable with dimensions $dims")
    return merged
end
global_indices(indices, dims, global_size) = indices
#---

#+++ Merge
function merge_ranks(stem)
    files = rank_files(stem)
    merged_file = stem * ".nc"
    @info "Merging $(length(files)) rank files into $merged_file"

    sources = [NCDataset(f, "r") for f in files]
    ds₀ = first(sources)
    try
        NCDataset(merged_file, "c") do out
            for (name, value) in ds₀.attrib
                out.attrib[name] = value
            end

            # Dimensions: x is the sum of the slabs, everything else the same on every rank
            unlimited_dims = NCDatasets.unlimited(ds₀.dim)
            global_size = Dict{String, Int}()
            for (name, n) in ds₀.dim
                sizes = [ds.dim[name] for ds in sources]
                if is_x_dim(name)
                    global_size[name] = sum(sizes)
                else
                    all(==(n), sizes) || error("dimension $name differs across ranks: $sizes")
                    global_size[name] = n
                end
                defDim(out, name, name in unlimited_dims ? Inf : global_size[name])
            end

            for name in keys(ds₀)
                v = ds₀[name]
                attrib = Dict{String, Any}(k => a for (k, a) in v.attrib)
                fillvalue = pop!(attrib, "_FillValue", nothing)
                haskey(attrib, "indices") && (attrib["indices"] = global_indices(attrib["indices"], dimnames(v), global_size))
                defVar(out, name, eltype(v.var), dimnames(v); attrib, fillvalue)
            end

            for name in keys(ds₀)
                dims = dimnames(ds₀[name])
                x = findfirst(is_x_dim, dims)
                if isnothing(x)
                    copy_whole!(out, sources, name)
                else
                    concatenate_in_x!(out, sources, name, x, findfirst(in(unlimited_dims), dims))
                end
            end

            check_increasing(out)
            write_global_grid!(out, ds₀)
        end
    finally
        foreach(close, sources)
    end

    return files
end

# Identical on every rank (the integrals are all-reduced): checked bit for bit, NaN equal to NaN, then copied. The
# ranges are explicit: `time` is written first, while the record dimension is still empty, where `:` writes nothing.
function copy_whole!(out, sources, name)
    data = Array(first(sources)[name].var)
    for (r, ds) in enumerate(sources)
        isequal(Array(ds[name].var), data) || error("$name differs between rank 0 and rank $(r - 1)")
    end
    ndims(data) == 0 ? (out[name].var[] = data[]) : (out[name].var[map(n -> 1:n, size(data))...] = data)
    return nothing
end

# Each rank's slab into its place along x, one record at a time
function concatenate_in_x!(out, sources, name, x, t)
    n = ndims(first(sources)[name].var)
    records = isnothing(t) ? (nothing,) : 1:size(first(sources)[name].var, t)
    for record in records
        offset = 0
        for ds in sources
            slab = ds[name].var
            nx = size(slab, x)
            source = ntuple(d -> d == t ? record : Colon(), n)
            target = ntuple(d -> d == x ? (offset+1:offset+nx) : d == t ? record : Colon(), n)
            out[name].var[target...] = slab[source...]
            offset += nx
        end
    end
    return nothing
end

function check_increasing(out)
    for name in keys(out.dim)
        if is_x_dim(name) && haskey(out, name)
            x = Array(out[name].var)
            all(>(0), diff(x)) || error("the merged $name is not increasing: are the rank files from the same run?")
        end
    end
    return nothing
end

# The grid metadata groups a single-GPU run writes, for the whole domain. The simulation's grid is fixed by its
# global attributes: (Nx, Ny, Nz) cells on [-Lx/2, Lx/2] × [-Ly/2, Ly/2] × [-Lz/2, Lz/2], (Periodic, Periodic, Bounded),
# the default halo. Checked against the merged coordinates before it is written.
function write_global_grid!(out, ds₀)
    N = Int.((ds₀.attrib["Nx"], ds₀.attrib["Ny"], ds₀.attrib["Nz"]))
    L = Float64.((ds₀.attrib["Lx"], ds₀.attrib["Ly"], ds₀.attrib["Lz"]))
    grid = RectilinearGrid(CPU(); size = N, x = (-L[1]/2, L[1]/2), y = (-L[2]/2, L[2]/2), z = (-L[3]/2, L[3]/2),
                           topology = (Periodic, Periodic, Bounded))

    if haskey(out, "x_caa")
        x = Array(out["x_caa"].var)
        length(x) == N[1] && isapprox(x, collect(xnodes(grid, Center())); rtol = 1e-12, atol = 1e-12) ||
            error("the merged x_caa does not match the grid the global attributes describe")
    end

    NCExt.write_grid_reconstruction_data!(out, grid, nothing)
    return nothing
end
#---

#+++ Main
if abspath(PROGRAM_FILE) == @__FILE__
    stem = only(filter(!startswith("--"), ARGS))
    delete = "--delete" in ARGS
    merged = vcat(merge_ranks(stem), merge_ranks(stem * "_2d"))
    if delete
        foreach(rm, merged)
        @info "Deleted the $(length(merged)) rank files"
    end
end
#---
