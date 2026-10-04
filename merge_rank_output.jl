# Stitches the per-rank NetCDF files of a distributed run (`--ranks R`, x-slabs) into the two files a single-GPU run
# writes, so that nothing downstream (the Python pipeline, `plot_kelvin_helmholtz_instability.jl`) knows the difference:
#
#   julia --project merge_rank_output.jl <dir>/khi_Nz1024_Ri0.10
#
# merges <stem>_rank<r>.nc into <stem>.nc and <stem>_2d_rank<r>.nc into <stem>_2d.nc, r = 0, …, R-1, with R read from
# the files' own `ranks` attribute.
#
# The 3D file is merged virtually: its fields are HDF5 virtual datasets that read each rank's slab from the rank file
# where it lies (see *Virtual datasets*), so the merge takes seconds where copying the ~3.8 TB of an Nz=1024 run took
# 7 h, and the rank files must stay beside it. The 2D file is written out (9 GB at Nz=1024): NCDatasets needs
# `allow_virtual_storage!` to read a virtual file, and the animation reads that one in Julia. A standalone copy of the
# 3D file, which the rank files are no longer needed for, is `nccopy -k nc4 <stem>.nc <copy>.nc` (hours at Nz=1024).
#
# What a rank file holds: its own x-slab of every field, on the dimensions a single-GPU file has (`x_caa`, `x_faa`,
# …), with real coordinates; every variable without an x dimension (time, the `_int` volume integrals, ε̄, L_K, …)
# whole and identical on every rank, since the reductions are all-reduced; and the grid of its own slab in the grid
# metadata groups. So the merge
#   - maps or concatenates every variable with an x dimension in rank order (the x coordinates are written out, and
#     checked to come out increasing);
#   - copies every other variable from rank 0, after checking that all ranks agree bit for bit;
#   - rewrites each variable's `indices` attribute, which records the slab's x range, to the whole domain's;
#   - writes the grid metadata groups for the whole domain with Oceananigans' own writer, on the CPU (a file that
#     says `GPU()` cannot be read back where CUDA is not loaded, e.g. by the animation on a CPU node).

using NCDatasets
using HDF5_jll: libhdf5, libhdf5_hl
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
"""
    merge_ranks(stem; virtual)

Merge the rank files of `stem` into `stem.nc`. With `virtual = true` every time-dependent field with an x dimension
is a virtual dataset over the rank files (see *Virtual datasets*), and everything else is written out; with
`virtual = false`, everything.
"""
function merge_ranks(stem; virtual)
    files = rank_files(stem)
    merged_file = stem * ".nc"
    @info "Merging $(length(files)) rank files into $merged_file" * (virtual ? ", virtually" : "")

    sources = [NCDataset(f, "r") for f in files]
    ds₀ = first(sources)
    virtual_fields = []
    try
        NCDataset(merged_file, "c") do out
            for (name, value) in ds₀.attrib
                out.attrib[name] = value
            end
            virtual && (out.attrib["virtual_rank_files"] = join(basename.(files), " "))

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
                t = findfirst(in(unlimited_dims), dims)
                if isnothing(x)
                    copy_whole!(out, sources, name)
                elseif virtual && !isnothing(t)
                    push!(virtual_fields, (name, collect(dims), x, [size(ds[name].var) for ds in sources]))
                else
                    concatenate_in_x!(out, sources, name, x, t)
                end
            end

            check_increasing(out)
            write_global_grid!(out, ds₀)
        end
    finally
        foreach(close, sources)
    end

    if virtual
        make_virtual!(merged_file, files, virtual_fields)
        check_virtual(merged_file, files, virtual_fields)
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

# Each rank's slab into its place along x, one record at a time, read into one buffer per rank
function concatenate_in_x!(out, sources, name, x, t)
    slabs = [ds[name].var for ds in sources]
    merged = out[name].var
    n = ndims(first(slabs))
    records = isnothing(t) ? (nothing,) : 1:size(first(slabs), t)
    buffers = [Array{eltype(slab)}(undef, (size(slab, d) for d in 1:n if d != t)...) for slab in slabs]
    for record in records
        source = ntuple(d -> d == t ? record : Colon(), n)
        offset = 0
        for (slab, buffer) in zip(slabs, buffers)
            nx = size(slab, x)
            NCDatasets.load!(slab, buffer, source...)
            merged[ntuple(d -> d == x ? (offset+1:offset+nx) : d == t ? record : Colon(), n)...] = buffer
            offset += nx
        end
    end
    return nothing
end

#+++ Virtual datasets
# netCDF-C cannot write an HDF5 virtual dataset, but reads one as an ordinary variable, so the virtual merge writes the
# merged file with NCDatasets, every time-dependent field with an x dimension defined and left empty,
# and then replaces each of those, through the HDF5 library netCDF-C itself is built on, with a virtual dataset that
# maps every rank file's slab onto its x-range: same name, type, attributes and dimensions. The rank files are named by
# their basenames, which HDF5 resolves in the merged file's own directory, so the directory can move as a whole; a rank
# file that is missing reads as fill values, without an error, which is what `virtual_rank_files` lets readers check.
const H5P_DEFAULT = Int64(0)
const H5F_ACC_RDWR = Cuint(1)
const H5S_SELECT_SET = Cint(0)
const H5T_VLEN = Cint(9)
const H5D_FILL_VALUE_UNDEFINED = Cint(0)

h5(result, call) = result < 0 ? error("HDF5: $call failed") : result

function h5_dataset_create_class()
    h5(ccall((:H5open, libhdf5), Cint, ()), "H5open")
    return unsafe_load(cglobal((:H5P_CLS_DATASET_CREATE_ID_g, libhdf5), Int64))
end

# A dataspace of fixed extent `size` (C order), wholly selected
simple_space(size) = h5(ccall((:H5Screate_simple, libhdf5), Int64, (Cint, Ptr{UInt64}, Ptr{UInt64}), length(size), size, size),
                        "H5Screate_simple")

push_attribute_name(location, name, info, names) = (push!(names::Vector{String}, unsafe_string(name)); Cint(0))

function attribute_names(object)
    names = String[]
    callback = @cfunction(push_attribute_name, Cint, (Int64, Cstring, Ptr{Cvoid}, Any))
    h5(ccall((:H5Aiterate2, libhdf5), Cint, (Int64, Cint, Cint, Ptr{UInt64}, Ptr{Cvoid}, Any),
             object, 0, 0, C_NULL, callback, names), "H5Aiterate2")
    return names
end

# Byte for byte, in the attribute's own type and shape (variable-length strings included)
function copy_attribute!(from, to, name)
    attribute = h5(ccall((:H5Aopen, libhdf5), Int64, (Int64, Cstring, Int64), from, name, H5P_DEFAULT), "H5Aopen")
    type = h5(ccall((:H5Aget_type, libhdf5), Int64, (Int64,), attribute), "H5Aget_type")
    space = h5(ccall((:H5Aget_space, libhdf5), Int64, (Int64,), attribute), "H5Aget_space")
    points = h5(ccall((:H5Sget_simple_extent_npoints, libhdf5), Int64, (Int64,), space), "H5Sget_simple_extent_npoints")
    buffer = zeros(UInt8, max(points, 1) * ccall((:H5Tget_size, libhdf5), Csize_t, (Int64,), type))
    h5(ccall((:H5Aread, libhdf5), Cint, (Int64, Int64, Ptr{UInt8}), attribute, type, buffer), "H5Aread")
    duplicate = h5(ccall((:H5Acreate2, libhdf5), Int64, (Int64, Cstring, Int64, Int64, Int64, Int64),
                         to, name, type, space, H5P_DEFAULT, H5P_DEFAULT), "H5Acreate2")
    h5(ccall((:H5Awrite, libhdf5), Cint, (Int64, Int64, Ptr{UInt8}), duplicate, type, buffer), "H5Awrite")
    variable_length = ccall((:H5Tdetect_class, libhdf5), Cint, (Int64, Cint), type, H5T_VLEN) > 0 ||
                      ccall((:H5Tis_variable_str, libhdf5), Cint, (Int64,), type) > 0
    variable_length && ccall((:H5Treclaim, libhdf5), Cint, (Int64, Int64, Int64, Ptr{UInt8}), type, space, H5P_DEFAULT, buffer)
    foreach(id -> ccall((:H5Aclose, libhdf5), Cint, (Int64,), id), (duplicate, attribute))
    ccall((:H5Sclose, libhdf5), Cint, (Int64,), space)
    ccall((:H5Tclose, libhdf5), Cint, (Int64,), type)
    return nothing
end

"""
    make_virtual!(merged_file, files, fields)

Replace each of `fields`, `(name, dims, x, slab_sizes)` with `dims` and `slab_sizes` in Julia's order (x the slab
axis, each rank's extent in `slab_sizes`), with a virtual dataset over the rank `files`, in place.
"""
function make_virtual!(merged_file, files, fields)
    dataset_create = h5_dataset_create_class()
    file = h5(ccall((:H5Fopen, libhdf5), Int64, (Cstring, Cuint, Int64), merged_file, H5F_ACC_RDWR, H5P_DEFAULT), "H5Fopen")
    for (name, dims, x, slab_sizes) in fields
        empty = h5(ccall((:H5Dopen2, libhdf5), Int64, (Int64, Cstring, Int64), file, name, H5P_DEFAULT), "H5Dopen2 $name")
        type = h5(ccall((:H5Dget_type, libhdf5), Int64, (Int64,), empty), "H5Dget_type")
        empty_create = h5(ccall((:H5Dget_create_plist, libhdf5), Int64, (Int64,), empty), "H5Dget_create_plist")

        # HDF5 orders dimensions the C way, the reverse of Julia's
        n = length(dims)
        global_size = collect(UInt64, reverse(ntuple(d -> d == x ? sum(s[x] for s in slab_sizes) : slab_sizes[1][d], n)))
        c_x = n - x + 1
        space = simple_space(global_size)
        create = h5(ccall((:H5Pcreate, libhdf5), Int64, (Int64,), dataset_create), "H5Pcreate")

        fill_status = Ref{Cint}(H5D_FILL_VALUE_UNDEFINED)
        h5(ccall((:H5Pfill_value_defined, libhdf5), Cint, (Int64, Ref{Cint}), empty_create, fill_status), "H5Pfill_value_defined")
        if fill_status[] != H5D_FILL_VALUE_UNDEFINED
            fill = zeros(UInt8, ccall((:H5Tget_size, libhdf5), Csize_t, (Int64,), type))
            h5(ccall((:H5Pget_fill_value, libhdf5), Cint, (Int64, Int64, Ptr{UInt8}), empty_create, type, fill), "H5Pget_fill_value")
            h5(ccall((:H5Pset_fill_value, libhdf5), Cint, (Int64, Int64, Ptr{UInt8}), create, type, fill), "H5Pset_fill_value")
        end

        offset = UInt64(0)
        for (path, slab_size) in zip(files, slab_sizes)
            source_size = collect(UInt64, reverse(slab_size))
            source = simple_space(source_size)
            start = zeros(UInt64, n)
            start[c_x] = offset
            h5(ccall((:H5Sselect_hyperslab, libhdf5), Cint, (Int64, Cint, Ptr{UInt64}, Ptr{UInt64}, Ptr{UInt64}, Ptr{UInt64}),
                     space, H5S_SELECT_SET, start, C_NULL, source_size, C_NULL), "H5Sselect_hyperslab")
            h5(ccall((:H5Pset_virtual, libhdf5), Cint, (Int64, Int64, Cstring, Cstring, Int64),
                     create, space, basename(path), "/" * name, source), "H5Pset_virtual $name")
            ccall((:H5Sclose, libhdf5), Cint, (Int64,), source)
            offset += source_size[c_x]
        end
        ccall((:H5Sselect_all, libhdf5), Cint, (Int64,), space)

        provisional = "__virtual__" * name
        virtual = h5(ccall((:H5Dcreate2, libhdf5), Int64, (Int64, Cstring, Int64, Int64, Int64, Int64, Int64),
                           file, provisional, type, space, H5P_DEFAULT, create, H5P_DEFAULT), "H5Dcreate2 $name")
        for attribute in attribute_names(empty)
            attribute == "DIMENSION_LIST" || copy_attribute!(empty, virtual, attribute)
        end
        for (i, dim) in enumerate(reverse(dims))   # each dimension's scale moves to the virtual dataset
            scale = h5(ccall((:H5Dopen2, libhdf5), Int64, (Int64, Cstring, Int64), file, dim, H5P_DEFAULT), "H5Dopen2 $dim")
            h5(ccall((:H5DSdetach_scale, libhdf5_hl), Cint, (Int64, Int64, Cuint), empty, scale, i - 1), "H5DSdetach_scale")
            h5(ccall((:H5DSattach_scale, libhdf5_hl), Cint, (Int64, Int64, Cuint), virtual, scale, i - 1), "H5DSattach_scale")
            ccall((:H5Dclose, libhdf5), Cint, (Int64,), scale)
        end

        foreach(id -> ccall((:H5Dclose, libhdf5), Cint, (Int64,), id), (virtual, empty))
        foreach(id -> ccall((:H5Pclose, libhdf5), Cint, (Int64,), id), (create, empty_create))
        ccall((:H5Sclose, libhdf5), Cint, (Int64,), space)
        ccall((:H5Tclose, libhdf5), Cint, (Int64,), type)
        h5(ccall((:H5Ldelete, libhdf5), Cint, (Int64, Cstring, Int64), file, name, H5P_DEFAULT), "H5Ldelete $name")
        h5(ccall((:H5Lmove, libhdf5), Cint, (Int64, Cstring, Int64, Cstring, Int64, Int64),
                 file, provisional, file, name, H5P_DEFAULT, H5P_DEFAULT), "H5Lmove $name")
    end
    h5(ccall((:H5Fclose, libhdf5), Cint, (Int64,), file), "H5Fclose")
    return nothing
end

# NCDatasets (0.14) maps only netCDF-C's storage codes for chunked (0) and contiguous (1), and reads every variable
# through its chunking, so a virtual one, which netCDF-C reports as virtual (4) or, built on HDF5 2.x, as unknown (3),
# fails with a KeyError. Treating the other codes as contiguous is all a read needs; any Julia reader of a virtual file
# calls this first.
allow_virtual_storage!() = foreach(code -> get!(NCDatasets.NCSymbols, code, :contiguous), 2:4)

# Each virtual field, read back through netCDF-C, against the rank files: the middle level of the first and last
# records, a plane that crosses every slab
function check_virtual(merged_file, files, fields)
    allow_virtual_storage!()
    NCDataset(merged_file, "r") do merged
        sources = [NCDataset(f, "r") for f in files]
        try
            for (name, dims, x, slab_sizes) in fields
                t = findfirst(==("time"), dims)
                z = findfirst(startswith("z_"), dims)
                records = size(merged[name].var, t)
                for record in unique((1, records)), level in (isnothing(z) ? (nothing,) : (cld(size(merged[name].var, z), 2),))
                    index = ntuple(d -> d == t ? record : d == z ? level : Colon(), length(dims))
                    slabs = [Array(ds[name].var[index...]) for ds in sources]
                    expected = cat(slabs...; dims = x - count(d -> d < x && (d == t || d == z), 1:length(dims)))
                    isequal(Array(merged[name].var[index...]), expected) ||
                        error("the virtual $name does not read back as its rank files at $index")
                end
            end
        finally
            foreach(close, sources)
        end
    end
    return nothing
end
#---

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
    length(ARGS) == 1 && !startswith(only(ARGS), "--") ||
        error("usage: julia --project merge_rank_output.jl <dir>/<stem> (the 3D file is merged virtually; for a standalone " *
              "copy, `nccopy -k nc4 <stem>.nc <copy>.nc`)")
    stem = only(ARGS)
    merge_ranks(stem; virtual = true)
    merge_ranks(stem * "_2d"; virtual = false)
end
#---
