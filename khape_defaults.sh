# Defaults for the submit wrappers: the project to charge and the python of the py313 environment. Every
# submit_*.sh sources this file, so this is the one place to change either; a value already set in the
# environment (export KHAPE_ACCOUNT=..., KHAPE_PYTHON=...) wins. The .pbs files do not read this file: the
# wrapper hands them the values (qsub -A and -v). See the README, Environment.
KHAPE_ACCOUNT=${KHAPE_ACCOUNT:-UMCP0061}
KHAPE_PYTHON=${KHAPE_PYTHON:-$HOME/miniconda3/envs/py313/bin/python}

# Optional output redirection: $KHAPE_OUTPUT_DIR (simulation output) and $KHAPE_PP_OUTPUT (post-processing output),
# when set, are handed to every job (each wrapper appends $KHAPE_REDIRECT to its qsub -v), so a run can write beside
# the production output instead of over it, and a single stage can be rerun on a redirected run. The PBS scripts,
# kelvin_helmholtz_instability.jl and src/aux00_utils.py all read them; give absolute paths.
KHAPE_REDIRECT=""
[ -n "${KHAPE_OUTPUT_DIR:-}" ] && KHAPE_REDIRECT="$KHAPE_REDIRECT,KHAPE_OUTPUT_DIR=$KHAPE_OUTPUT_DIR"
[ -n "${KHAPE_PP_OUTPUT:-}" ]  && KHAPE_REDIRECT="$KHAPE_REDIRECT,KHAPE_PP_OUTPUT=$KHAPE_PP_OUTPUT"

# For the wrappers that submit Python jobs: fail at submission, not in the queue, when the interpreter is not there.
check_khape_python() {
    [ -x "$KHAPE_PYTHON" ] && return 0
    echo "error: KHAPE_PYTHON=$KHAPE_PYTHON is not an executable; set it to your py313 environment's python (see the README, Environment)" >&2
    exit 2
}

# Multi-GPU runs (NGPUS > 1) run under Casper's CUDA-aware OpenMPI, which MPI.jl reaches through MPItrampoline and an
# MPIwrapper built against it, chosen by an environment of its own stacked on the project's load path, with the packages
# compiled for it in a depot of their own (setup_mpi_env.sh makes all three, once). One-GPU runs and CI never see them.
KHAPE_MPI_DEPOT=${KHAPE_MPI_DEPOT:-$WORK/.julia-mpi}
KHAPE_MPI_ENV=${KHAPE_MPI_ENV:-$KHAPE_MPI_DEPOT/environments/khape-mpi}

# The default number of GPUs for a resolution: one, and no MPI, up to Nz=512, the largest 3D run one A100-80GB holds;
# one whole Casper A100 node, four GPUs, up to Nz=1024. Beyond that NGPUS has to be given (see the README, Multi-GPU
# runs: every rank keeps its own copy of the sorted column, which grows 8x per doubling of Nz).
default_ngpus() {
    if   [ "$1" -le 512 ];  then echo 1
    elif [ "$1" -le 1024 ]; then echo 4
    else echo "error: no default NGPUS for NZ=$1; give NGPUS= (a multiple of 4)" >&2; return 2
    fi
}

# submit_simulation_job NAME NZ NGPUS VARS
# Submit simulation.pbs as job NAME, on NGPUS GPUs, with VARS (a comma-separated KEY=VALUE list) in its environment,
# and print the id of the job everything that reads the run must wait for.
#   NGPUS=1   the one-GPU request in simulation.pbs; the job draws its own figures. Prints its id.
#   NGPUS>1   whole Casper A100-80GB nodes (NGPUS a multiple of 4), one MPI rank per GPU, each writing its own files,
#             then merge.pbs, a CPU job, to stitch them into the one-GPU layout and draw the figures. Prints the
#             merge job's id.
# The job learns its rank count as RANKS, not NGPUS: PBS and NCAR's set_gpu_rank use NGPUS for the per-node count.
submit_simulation_job() {
    local name=$1 nz=$2 ngpus=$3 vars=$4
    if [ "$ngpus" = "1" ]; then
        qsub -N "$name" -A "$KHAPE_ACCOUNT" -o "logs/${name}.log" -e "logs/${name}.log" -v "$vars,RANKS=1" simulation.pbs
        return
    fi
    if [ $((ngpus % 4)) -ne 0 ]; then
        echo "error: NGPUS=$ngpus; a multi-GPU run takes whole 4-GPU nodes, so NGPUS must be a multiple of 4" >&2; return 2
    fi
    if [ ! -f "$KHAPE_MPI_ENV/LocalPreferences.toml" ] || ! ls "$KHAPE_MPI_DEPOT"/mpiwrapper/lib*/libmpiwrapper.so >/dev/null 2>&1; then
        echo "error: no MPI environment in $KHAPE_MPI_DEPOT; run setup_mpi_env.sh once (see the README, Multi-GPU runs)" >&2; return 2
    fi
    local sim_job merge_job
    sim_job=$(qsub -N "$name" -A "$KHAPE_ACCOUNT" -o "logs/${name}.log" -e "logs/${name}.log" \
                   -l select=$((ngpus / 4)):ncpus=32:mpiprocs=4:ompthreads=8:ngpus=4:gpu_type=a100_80gb:mem=200GB \
                   -l walltime=08:00:00 \
                   -v "$vars,RANKS=$ngpus,KHAPE_MPI_DEPOT=$KHAPE_MPI_DEPOT,KHAPE_MPI_ENV=$KHAPE_MPI_ENV" simulation.pbs) || return
    merge_job=$(qsub -N "merge_${name}" -A "$KHAPE_ACCOUNT" -o "logs/merge_${name}.log" -e "logs/merge_${name}.log" \
                     -W depend=afterok:$sim_job -v "$vars" merge.pbs) || return
    echo "Submitted the simulation on $ngpus GPUs ($sim_job) and its merge ($merge_job)" >&2
    echo "$merge_job"
}
