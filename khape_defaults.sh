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
# and print its id, which everything that reads the run waits for. The job draws its own figures.
#   NGPUS=1   the one-GPU request in simulation.pbs.
#   NGPUS>1   whole Casper A100-80GB nodes (NGPUS a multiple of 4), one MPI rank per GPU, each writing its own files,
#             which the job stitches into the one-GPU layout after the run.
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
    qsub -N "$name" -A "$KHAPE_ACCOUNT" -o "logs/${name}.log" -e "logs/${name}.log" \
         -l select=$((ngpus / 4)):ncpus=32:mpiprocs=4:ompthreads=8:ngpus=4:gpu_type=a100_80gb:mem=200GB \
         -l walltime=08:00:00 \
         -v "$vars,RANKS=$ngpus,KHAPE_MPI_DEPOT=$KHAPE_MPI_DEPOT,KHAPE_MPI_ENV=$KHAPE_MPI_ENV" simulation.pbs
}

# The default number of jobs the sweep transfer (postprocessing/sweep2_energy_transfer.py) splits a run's records across.
# Its memory grows with records x cells: the 3D Nz=512 run took 689 GiB of a 730 GiB node in one job, and at Nz=1024 one
# record's sorted column of the padded domain alone is 421 M slots, 6.3 GiB with its slot heights. Beyond Nz=1024
# SWEEP_PARTS has to be given.
default_sweep_parts() {
    if   [ "$1" -le 512 ];  then echo 1
    elif [ "$1" -le 1024 ]; then echo 6
    else echo "error: no default SWEEP_PARTS for NZ=$1; give SWEEP_PARTS=" >&2; return 2
    fi
}

# submit_sweep_transfer NAME PARTS DEPEND VARS
# Submit postprocessing/sweep_transfer.pbs (from postprocessing/) as job NAME, with VARS (a comma-separated KEY=VALUE
# list) in its environment, after DEPEND (a PBS dependency such as afterok:123, or empty), and print the id of the job
# after which the transfer file is whole and sweep3 has run.
#   PARTS=1   one job: sweep2, then sweep3. Prints its id.
#   PARTS>1   PARTS jobs at once, each over its share of the records (PART=K/PARTS), then a small one after all of them
#             that joins their files and runs sweep3 (MERGE_PARTS=PARTS). Prints that one's id.
submit_sweep_transfer() {
    local name=$1 parts=$2 depend=$3 vars=$4
    local after=(); [ -n "$depend" ] && after=(-W depend=$depend)
    if [ "$parts" = "1" ]; then
        qsub -N "$name" -A "$KHAPE_ACCOUNT" -o "logs/${name}.log" -e "logs/${name}.log" -v "$vars" "${after[@]}" sweep_transfer.pbs
        return
    fi
    local k part job jobs=""
    for k in $(seq 1 "$parts"); do
        part="${name}_part${k}of${parts}"
        job=$(qsub -N "$part" -A "$KHAPE_ACCOUNT" -o "logs/${part}.log" -e "logs/${part}.log" -v "$vars,PART=$k/$parts" \
                   "${after[@]}" sweep_transfer.pbs) || return
        jobs="$jobs:$job"
    done
    job=$(qsub -N "${name}_merge" -A "$KHAPE_ACCOUNT" -o "logs/${name}_merge.log" -e "logs/${name}_merge.log" \
               -l select=1:ncpus=2:mem=32GB:ngpus=0 -l walltime=02:00:00 -v "$vars,MERGE_PARTS=$parts" \
               -W depend=afterok$jobs sweep_transfer.pbs) || return
    echo "Submitted the sweep transfer in $parts parts (${jobs#:}) and the job that joins them ($job)" >&2
    echo "$job"
}
