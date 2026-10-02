#!/usr/bin/env bash
# One-time setup for multi-GPU runs (NGPUS > 1). Run it once per machine, on Casper (a login node will do):
#   bash setup_mpi_env.sh
#
# Oceananigans' `Distributed(GPU())` hands GPU arrays straight to MPI, so a multi-GPU run needs a CUDA-aware MPI, which
# the one MPI.jl bundles is not; Casper's OpenMPI (5.0.8, built with CUDA and UCX) is. MPI.jl is not pointed at that
# library directly, though. Every JLL built against MPI follows MPI.jl's choice of ABI, and with the system OpenMPI
# chosen, HDF5_jll (under NCDatasets) loads OpenMPI_jll, a second OpenMPI (5.0.11), which cannot share a process with
# the system's ("undefined symbol: opal_single_threaded"). So MPI.jl and those JLLs all go through MPItrampoline
# (MPItrampoline_jll), and an MPIwrapper built here against the system OpenMPI receives its calls at run time
# (MPITRAMPOLINE_LIB). GPU buffers pass through untouched. Under $KHAPE_MPI_DEPOT (khape_defaults.sh) it makes
#   - mpiwrapper/, MPIwrapper at the revision MPItrampoline_jll was built with, compiled against the system OpenMPI;
#   - $KHAPE_MPI_ENV, a Julia environment whose LocalPreferences.toml holds MPIPreferences' choice of MPItrampoline_jll.
#     simulation.pbs stacks it on the load path of a multi-GPU job only. The project does not list MPIPreferences, so
#     its own (tracked) LocalPreferences.toml is not involved, and one-GPU jobs and CI keep the bundled MPI;
#   - compiled/, MPI.jl, the MPI JLLs and everything above them compiled for MPItrampoline. simulation.pbs puts this
#     depot first on a multi-GPU job's depot path: those cache files are not keyed on the MPI choice, so sharing one
#     depot would have the two builds overwrite each other at every switch.
set -eo pipefail
cd "$(dirname "$0")"
REPO=$(pwd)
source khape_defaults.sh

# The OpenMPI simulation.pbs loads (it needs a compiler module first), and cmake for MPIwrapper
module --force purge
module load ncarenv/25.10
module load intel/2025.2.1
module load openmpi/5.0.8
module load cmake
: "${NCAR_ROOT_OPENMPI:?the openmpi module did not set NCAR_ROOT_OPENMPI}"

if [ -n "${JULIA:-}" ]; then :
elif command -v juliaup >/dev/null 2>&1; then juliaup add 1.13; JULIA="julia +1.13"
elif [ -x "$WORK/julia-1.13/bin/julia" ]; then JULIA="$WORK/julia-1.13/bin/julia"
else echo "error: no Julia 1.13 found (see simulation.pbs)" >&2; exit 1
fi
JULIA_BIN=$($JULIA --startup-file=no -e 'print(joinpath(Sys.BINDIR, "julia"))')

# The version of a package in the project's Manifest
manifest_version() {
    awk -v pkg="$1" '$0 == "[[deps." pkg "]]" {f = 1; next} /^\[\[/ {f = 0} f && /^version = / {gsub(/"/, "", $3); print $3}' Manifest.toml
}
MPIPREFERENCES_VERSION=$(manifest_version MPIPreferences)
MPITRAMPOLINE_VERSION=$(manifest_version MPItrampoline_jll)
if [ -z "$MPIPREFERENCES_VERSION" ] || [ -z "$MPITRAMPOLINE_VERSION" ]; then
    echo "error: no MPIPreferences or MPItrampoline_jll in Manifest.toml" >&2; exit 1
fi

#+++ MPIwrapper, at the revision MPItrampoline_jll's README names, so that the two speak the same version of the interface
README=$(JULIA_DEPOT_PATH="$WORK/.julia" $JULIA_BIN --project --startup-file=no -e '
    jll = Base.PkgId(Base.UUID("f1f71cc9-e9ae-5b93-9b94-4fe0e1ad3748"), "MPItrampoline_jll")   # an indirect dependency
    print(joinpath(dirname(dirname(Base.locate_package(jll))), "README.md"))')
MPIWRAPPER_REVISION=$(grep -A3 "eschnett/MPIwrapper" "$README" | grep -o 'revision: `[0-9a-f]*' | grep -o '[0-9a-f]*$' | head -1)
if [ -z "$MPIWRAPPER_REVISION" ]; then
    echo "error: could not read the MPIwrapper revision from MPItrampoline_jll's README ($README)" >&2; exit 1
fi
echo "MPItrampoline_jll $MPITRAMPOLINE_VERSION: building MPIwrapper $MPIWRAPPER_REVISION against $NCAR_ROOT_OPENMPI"

MPIWRAPPER="$KHAPE_MPI_DEPOT/mpiwrapper"
SOURCE="$KHAPE_MPI_DEPOT/mpiwrapper-src"
rm -rf "$SOURCE" "$MPIWRAPPER"
mkdir -p "$KHAPE_MPI_DEPOT"
git clone --quiet https://github.com/eschnett/MPIwrapper "$SOURCE"
git -C "$SOURCE" checkout --quiet "$MPIWRAPPER_REVISION"
cmake -S "$SOURCE" -B "$SOURCE/build" -DMPIEXEC_EXECUTABLE="$(command -v mpiexec)" -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DCMAKE_INSTALL_PREFIX="$MPIWRAPPER" > "$SOURCE/cmake.log"
cmake --build "$SOURCE/build" -j 4 > "$SOURCE/build.log"
cmake --install "$SOURCE/build" > "$SOURCE/install.log"
export MPITRAMPOLINE_LIB=$(ls "$MPIWRAPPER"/lib*/libmpiwrapper.so)
export MPITRAMPOLINE_MPIEXEC="$MPIWRAPPER/bin/mpiwrapperexec"
#---

#+++ The environment, and the depot its packages are compiled into
export JULIA_DEPOT_PATH="$KHAPE_MPI_DEPOT:$WORK/.julia"
export JULIA_CPU_TARGET="generic"   # as simulation.pbs, so what this compiles is reused there
mkdir -p "$KHAPE_MPI_ENV"
$JULIA_BIN --project="$KHAPE_MPI_ENV" --startup-file=no -e "
    using Pkg
    Pkg.add(name = \"MPIPreferences\", version = \"$MPIPREFERENCES_VERSION\")
    using MPIPreferences
    MPIPreferences.use_jll_binary(\"MPItrampoline_jll\"; force = true)"
#---

#+++ Checks, from the project, as simulation.pbs runs it
# MPI.jl must reach the system OpenMPI through the trampoline, and the NetCDF stack (HDF5_jll, which follows the same
# choice of MPI) must load beside it; this also compiles everything into the depot above.
export JULIA_LOAD_PATH="@:$KHAPE_MPI_ENV:@stdlib"
$JULIA_BIN --project --startup-file=no -e 'using Pkg; Pkg.instantiate()'
$JULIA_BIN --project --startup-file=no -e '
    using MPI
    MPI.MPIPreferences.binary == "MPItrampoline_jll" || error("MPI.jl uses $(MPI.MPIPreferences.binary), not MPItrampoline_jll")
    MPI.Init()
    library = MPI.Get_library_version()
    occursin("Open MPI", library) || error("MPItrampoline did not reach the system Open MPI; it reports:\n$library")
    using CUDA, Oceananigans, Oceanostics, NCDatasets, CairoMakie, ArgParse
    println("MPI.jl reaches ", first(split(library, ",")), " through MPItrampoline, and the NetCDF stack loads beside it")'

echo "MPI environment ready: $KHAPE_MPI_ENV, with MPIwrapper in $MPIWRAPPER (depot $KHAPE_MPI_DEPOT)"
#---
