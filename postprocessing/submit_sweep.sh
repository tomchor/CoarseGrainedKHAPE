#!/usr/bin/env bash
# Submit sweep jobs: shared filter step (inv1) once, then per-FIXED_REF transfer steps (inv2+inv3).
# Usage: bash submit_sweep.sh [NZ=2048] [FIXED_REF=0|1|both]
#   FIXED_REF=both  submits transfer jobs for both 0 and 1 (filter runs only once)
NZ=2048; FIXED_REF=0; N_TIME_SKIP=2
for arg in "$@"; do case $arg in NZ=*) NZ="${arg#*=}";; FIXED_REF=*) FIXED_REF="${arg#*=}";; N_TIME_SKIP=*) N_TIME_SKIP="${arg#*=}";; esac; done

FILTER_NAME="sweep_filter_Nz${NZ}_Ri0.10"
FILTER_JOB=$(qsub -N "$FILTER_NAME" \
                  -o "logs/${FILTER_NAME}.log" \
                  -e "logs/${FILTER_NAME}.log" \
                  -v NZ=$NZ,N_TIME_SKIP=$N_TIME_SKIP \
                  sweep_filter.pbs)
echo "Submitted filter job (Nz=$NZ): $FILTER_JOB"

submit_transfer() {
    local fr=$1
    [ "$fr" = "1" ] && REF_SUFFIX="_fixed_ref" || REF_SUFFIX=""
    local NAME="sweep_transfer_Nz${NZ}_Ri0.10${REF_SUFFIX}"
    local JOB=$(qsub -N "$NAME" \
                     -o "logs/${NAME}.log" \
                     -e "logs/${NAME}.log" \
                     -v NZ=$NZ,FIXED_REF=$fr \
                     -W depend=afterok:$FILTER_JOB \
                     sweep_transfer.pbs)
    echo "Submitted transfer job FIXED_REF=$fr (depends on $FILTER_JOB): $JOB"
}

if [ "$FIXED_REF" = "both" ]; then
    submit_transfer 0
    submit_transfer 1
else
    submit_transfer "$FIXED_REF"
fi
