#!/bin/bash
# Submit the simulation and post-processing as chained PBS jobs (afterok dependencies);
# each stage only runs if the previous one succeeds. Optional validation and plotting stages.
#
#   simulation → budgeting → sweep_filter → sweep_transfer                      (always)
#   + validation  (online-vs-offline figures + animations; parallel after sim)  (VALIDATE=1)
#   + plots       (plot2 transfer spectrum, plot3 budgets, plot4 panels)        (PLOTS=1)
#
# The SFS KE and APE budgets are computed by the simulation itself; `budgeting` assembles and plots them
# (postprocessing/01_online_budgets.py, 02_plot_budgets.py). The sweep over filter scales stays offline.
#
# Usage: bash submit_all_pbs.sh [NZ=512] [NGPUS=] [VALIDATE=0] [PLOTS=0] [SAVE_SORTED=1] [FIXED_REF=0] [SIMULATION=1]
#   NZ         vertical resolution
#   NGPUS      GPUs to run the simulation on (default: 1 up to NZ=512, 4 up to NZ=1024). More than one splits the
#              domain across MPI ranks, whose files the simulation job merges after the run; see submit_simulation.sh
#              and the README, Multi-GPU runs
#   SIMULATION run the simulation (1), or start the chain at budgeting on the run already in ${KHAPE_OUTPUT_DIR:-output} (0)
#   VALIDATE   also run the online-vs-offline validation (adds --save_tensors for the tensor comparison and
#              forces --save_sorted, which inv06-inv07 read): 0 or 1
#   SAVE_SORTED  also write the validation-only sorted-state fields (default 1 on one GPU, 0 on several, where it
#              is refused): 0 or 1. The budgets do not depend on it.
#   PLOTS      also run the final plots after sweep_transfer: 0 or 1
#   FIXED_REF  the sweep transfer's fixed-in-time reference profile (the budgets have no such variant): 0 or 1
#
# To run post-processing alone:
#   bash postprocessing/submit_budgeting.sh [NZ=512]
# or the whole chain from budgeting on, on a run that already exists:
#   bash submit_all_pbs.sh SIMULATION=0 [PLOTS=1]

NZ=512; NGPUS=""; FIXED_REF=0; VALIDATE=0; PLOTS=0; SIMULATION=1
# An unknown KEY=VALUE is refused rather than ignored, so a misspelled flag cannot silently fall back to its default.
for arg in "$@"; do case $arg in
  NZ=*)          NZ="${arg#*=}";;
  NGPUS=*)       NGPUS="${arg#*=}";;
  FIXED_REF=*)   FIXED_REF="${arg#*=}";;
  VALIDATE=*)    VALIDATE="${arg#*=}";;
  SAVE_SORTED=*) SAVE_SORTED="${arg#*=}";;
  PLOTS=*)       PLOTS="${arg#*=}";;
  SIMULATION=*)  SIMULATION="${arg#*=}";;
  *) echo "unknown argument: $arg (expected NZ=, NGPUS=, FIXED_REF=, VALIDATE=, PLOTS=, SAVE_SORTED= or SIMULATION=)" >&2; exit 2;;
esac; done
# The allocation to charge and the Python to run: defaults in khape_defaults.sh, overridden from the environment.
source "$(dirname "$0")/khape_defaults.sh"
check_khape_python
[ "$FIXED_REF" = "1" ] && REF_SUFFIX="_fixed_ref" || REF_SUFFIX=""
# --save_sorted writes the validation-only sorted-state fields (the two model-grid z✶ methods, the column,
# ∫E_b) that inv06-inv07 read. Every budget term is written regardless, so SAVE_SORTED=0 gives smaller
# output and changes no budget number; VALIDATE=1 turns it back on.
# On several GPUs it is refused (its model-grid sorts would sort each rank's slab), and so is VALIDATE=1, which needs it.
NGPUS=${NGPUS:-$(default_ngpus $NZ)} || exit 2
if [ "$NGPUS" != "1" ]; then
    if [ "$VALIDATE" = "1" ] || [ "${SAVE_SORTED:-0}" = "1" ]; then
        echo "error: VALIDATE=1 and SAVE_SORTED=1 need a single GPU (NGPUS=1): the sorted-state view sorts each rank's slab" >&2; exit 2
    fi
    SAVE_SORTED=0
fi
SAVE_SORTED=${SAVE_SORTED:-1}
if [ "$VALIDATE" = "1" ]; then SAVE_SORTED=1; fi          # inv06-inv07 read the sorted state
[ "$VALIDATE" = "1" ] && SAVE_TENSORS=1 || SAVE_TENSORS=0   # inv03 reads the per-scale tensors

# AFTER_SIM is what the jobs that read the simulation output wait for: the simulation job, or nothing with SIMULATION=0.
AFTER_SIM=()
if [ "$SIMULATION" = "1" ]; then
    SIM_JOB=$(submit_simulation_job kelvin_helmholtz_${NZ} $NZ $NGPUS \
              "NZ=$NZ,SAVE_TENSORS=$SAVE_TENSORS,SAVE_SORTED=$SAVE_SORTED,KHAPE_PYTHON=$KHAPE_PYTHON$KHAPE_REDIRECT") || exit 2
    echo "Submitted simulation (Nz=$NZ, ngpus=$NGPUS, save_tensors=$SAVE_TENSORS, save_sorted=$SAVE_SORTED): $SIM_JOB"
    AFTER_SIM=(-W depend=afterok:$SIM_JOB)
else
    RUN="${KHAPE_OUTPUT_DIR:-output}/khi_Nz${NZ}_Ri0.10.nc"
    [ -f "$RUN" ] || { echo "SIMULATION=0, but there is no run to start from: $RUN does not exist" >&2; exit 2; }
    SIM_JOB="nothing (SIMULATION=0: $RUN)"
fi

# Optional validation — parallel branch, runs after the simulation succeeds
if [ "$VALIDATE" = "1" ]; then
    cd postprocessing/validation
    mkdir -p logs
    VAL_NAME="validation_Nz${NZ}_Ri0.10"
    VAL_JOB=$(qsub -N "$VAL_NAME" \
                   -A "$KHAPE_ACCOUNT" \
                   -o "logs/${VAL_NAME}.log" \
                   -e "logs/${VAL_NAME}.log" \
                   -v NZ=$NZ,KHAPE_PYTHON=$KHAPE_PYTHON$KHAPE_REDIRECT \
                   "${AFTER_SIM[@]}" \
                   validation.pbs)
    echo "Submitted validation (depends on $SIM_JOB): $VAL_JOB"
    cd ../..
fi

cd postprocessing

PP_NAME="budgeting_Nz${NZ}_Ri0.10"
PP_JOB=$(qsub -N "$PP_NAME" \
              -A "$KHAPE_ACCOUNT" \
              -o "logs/${PP_NAME}.log" \
              -e "logs/${PP_NAME}.log" \
              -v NZ=$NZ,KHAPE_PYTHON=$KHAPE_PYTHON$KHAPE_REDIRECT \
              "${AFTER_SIM[@]}" \
              budgeting.pbs)
echo "Submitted budgeting (depends on $SIM_JOB): $PP_JOB"

SF_NAME="sweep_filter_Nz${NZ}_Ri0.10"
SF_JOB=$(qsub -N "$SF_NAME" \
              -A "$KHAPE_ACCOUNT" \
              -o "logs/${SF_NAME}.log" \
              -e "logs/${SF_NAME}.log" \
              -v NZ=$NZ,KHAPE_PYTHON=$KHAPE_PYTHON$KHAPE_REDIRECT \
              -W depend=afterok:$PP_JOB \
              sweep_filter.pbs)
echo "Submitted sweep filter (depends on $PP_JOB): $SF_JOB"

ST_NAME="sweep_transfer_Nz${NZ}_Ri0.10${REF_SUFFIX}"
ST_JOB=$(qsub -N "$ST_NAME" \
              -A "$KHAPE_ACCOUNT" \
              -o "logs/${ST_NAME}.log" \
              -e "logs/${ST_NAME}.log" \
              -v NZ=$NZ,FIXED_REF=$FIXED_REF,KHAPE_PYTHON=$KHAPE_PYTHON$KHAPE_REDIRECT \
              -W depend=afterok:$SF_JOB \
              sweep_transfer.pbs)
echo "Submitted sweep transfer (depends on $SF_JOB): $ST_JOB"

# Optional final plots — after sweep_transfer (which is downstream of budgeting, so both are done)
if [ "$PLOTS" = "1" ]; then
    PLOTS_NAME="plots_Nz${NZ}_Ri0.10"
    PLOTS_JOB=$(qsub -N "$PLOTS_NAME" \
                     -A "$KHAPE_ACCOUNT" \
                     -o "logs/${PLOTS_NAME}.log" \
                     -e "logs/${PLOTS_NAME}.log" \
                     -v NZ=$NZ,KHAPE_PYTHON=$KHAPE_PYTHON$KHAPE_REDIRECT \
                     -W depend=afterok:$ST_JOB \
                     plots.pbs)
    echo "Submitted final plots (depends on $ST_JOB): $PLOTS_JOB"
fi
cd ..
