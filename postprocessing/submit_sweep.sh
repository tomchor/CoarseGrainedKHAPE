#!/usr/bin/env bash
# Submit sweep jobs: shared filter step (sweep1) once, then per-FIXED_REF transfer steps (sweep2+sweep3).
# Usage: bash submit_sweep.sh [NZ=2048] [FIXED_REF=0|1|both] [N_TIME_SKIP=2] [EXTENSION=edge|odd]
#   FIXED_REF=both  submits transfer jobs for both 0 and 1 (filter runs only once)
#   EXTENSION=odd   runs the whole sweep with odd reflection of b past the walls, into _sweep_odd files
NZ=2048; FIXED_REF=0; N_TIME_SKIP=2; EXTENSION=edge
# An unknown KEY=VALUE is refused rather than ignored: ignoring EXTENSION=odd used to rerun the edge sweep
# over the production files.
for arg in "$@"; do case $arg in
  NZ=*)          NZ="${arg#*=}";;
  FIXED_REF=*)   FIXED_REF="${arg#*=}";;
  N_TIME_SKIP=*) N_TIME_SKIP="${arg#*=}";;
  EXTENSION=*)   EXTENSION="${arg#*=}";;
  *) echo "unknown argument: $arg (expected NZ=, FIXED_REF=, N_TIME_SKIP= or EXTENSION=)" >&2; exit 2;;
esac; done
# The allocation to charge and the Python to run belong to whoever submits, so they come from the environment.
: "${KHAPE_ACCOUNT:?set KHAPE_ACCOUNT to the project code to charge; see the README, Environment}"
: "${KHAPE_PYTHON:?set KHAPE_PYTHON to the python of your py313 environment; see the README, Environment}"
case $EXTENSION in edge|odd) ;; *) echo "EXTENSION must be edge or odd, not '$EXTENSION'" >&2; exit 2;; esac
[ "$EXTENSION" = "edge" ] && EXT_TAG="" || EXT_TAG="_${EXTENSION}"

FILTER_NAME="sweep_filter_Nz${NZ}_Ri0.10${EXT_TAG}"
FILTER_JOB=$(qsub -N "$FILTER_NAME" \
                  -A "$KHAPE_ACCOUNT" \
                  -o "logs/${FILTER_NAME}.log" \
                  -e "logs/${FILTER_NAME}.log" \
                  -v NZ=$NZ,N_TIME_SKIP=$N_TIME_SKIP,EXTENSION=$EXTENSION,KHAPE_PYTHON=$KHAPE_PYTHON \
                  sweep_filter.pbs)
echo "Submitted filter job (Nz=$NZ, extension=$EXTENSION): $FILTER_JOB"

submit_transfer() {
    local fr=$1
    [ "$fr" = "1" ] && REF_SUFFIX="_fixed_ref" || REF_SUFFIX=""
    local NAME="sweep_transfer_Nz${NZ}_Ri0.10${REF_SUFFIX}${EXT_TAG}"
    local JOB=$(qsub -N "$NAME" \
                     -A "$KHAPE_ACCOUNT" \
                     -o "logs/${NAME}.log" \
                     -e "logs/${NAME}.log" \
                     -v NZ=$NZ,FIXED_REF=$fr,EXTENSION=$EXTENSION,KHAPE_PYTHON=$KHAPE_PYTHON \
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
