#!/usr/bin/env bash
# Submit budgeting jobs: shared filter step (01) once, then per-FIXED_REF steps (02-06).
# Usage: bash submit_budgeting.sh [NZ=2048] [FIXED_REF=0|1|both]
#   FIXED_REF=both  submits budget jobs for both 0 and 1 (filter runs only once)
NZ=2048; FIXED_REF=0
# An unknown KEY=VALUE is refused rather than ignored, so a misspelled flag cannot silently fall back to its default.
for arg in "$@"; do case $arg in
  NZ=*)        NZ="${arg#*=}";;
  FIXED_REF=*) FIXED_REF="${arg#*=}";;
  *) echo "unknown argument: $arg (expected NZ= or FIXED_REF=)" >&2; exit 2;;
esac; done
# The allocation to charge and the Python to run belong to whoever submits, so they come from the environment.
: "${KHAPE_ACCOUNT:?set KHAPE_ACCOUNT to the project code to charge; see the README, Environment}"
: "${KHAPE_PYTHON:?set KHAPE_PYTHON to the python of your py313 environment; see the README, Environment}"

FILTER_NAME="budgeting_filter_Nz${NZ}_Ri0.10"
FILTER_JOB=$(qsub -N "$FILTER_NAME" \
                  -A "$KHAPE_ACCOUNT" \
                  -o "logs/${FILTER_NAME}.log" \
                  -e "logs/${FILTER_NAME}.log" \
                  -v NZ=$NZ,KHAPE_PYTHON=$KHAPE_PYTHON \
                  budgeting_filter.pbs)
echo "Submitted filter job (Nz=$NZ): $FILTER_JOB"

submit_budget() {
    local fr=$1
    [ "$fr" = "1" ] && REF_SUFFIX="_fixed_ref" || REF_SUFFIX=""
    local NAME="budgeting_Nz${NZ}_Ri0.10${REF_SUFFIX}"
    local JOB=$(qsub -N "$NAME" \
                     -A "$KHAPE_ACCOUNT" \
                     -o "logs/${NAME}.log" \
                     -e "logs/${NAME}.log" \
                     -v NZ=$NZ,FIXED_REF=$fr,KHAPE_PYTHON=$KHAPE_PYTHON \
                     -W depend=afterok:$FILTER_JOB \
                     budgeting.pbs)
    echo "Submitted budget job FIXED_REF=$fr (depends on $FILTER_JOB): $JOB"
}

if [ "$FIXED_REF" = "both" ]; then
    submit_budget 0
    submit_budget 1
else
    submit_budget "$FIXED_REF"
fi
