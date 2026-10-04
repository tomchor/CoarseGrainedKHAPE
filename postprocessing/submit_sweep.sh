#!/usr/bin/env bash
# Submit sweep jobs: shared filter step (sweep1) once, then per-FIXED_REF transfer steps (sweep2+sweep3).
# Usage: bash submit_sweep.sh [NZ=512] [FIXED_REF=0|1|both] [N_TIME_SKIP=2] [EXTENSION=edge|odd] [SWEEP_PARTS=]
#                              [FILTER=1] [PLOTS=0]
#   FIXED_REF=both  submits transfer jobs for both 0 and 1 (filter runs only once)
#   EXTENSION=odd   runs the whole sweep with odd reflection of b past the walls, into _sweep_odd files
#   SWEEP_PARTS     jobs each transfer splits the records across, joined by a small job that then runs sweep3 (default:
#                   1 up to NZ=512, 6 up to NZ=1024; see default_sweep_parts in khape_defaults.sh)
#   FILTER=0        skip sweep1: the transfer starts at once, on the filtered file an earlier sweep1 wrote
#   PLOTS=1         also run every plot*.py (plots.pbs) once the transfer is done
NZ=512; FIXED_REF=0; N_TIME_SKIP=2; EXTENSION=edge; SWEEP_PARTS=""; FILTER=1; PLOTS=0
# An unknown KEY=VALUE is refused rather than ignored: ignoring EXTENSION=odd used to rerun the edge sweep
# over the production files.
for arg in "$@"; do case $arg in
  NZ=*)          NZ="${arg#*=}";;
  FIXED_REF=*)   FIXED_REF="${arg#*=}";;
  N_TIME_SKIP=*) N_TIME_SKIP="${arg#*=}";;
  EXTENSION=*)   EXTENSION="${arg#*=}";;
  SWEEP_PARTS=*) SWEEP_PARTS="${arg#*=}";;
  FILTER=*)      FILTER="${arg#*=}";;
  PLOTS=*)       PLOTS="${arg#*=}";;
  *) echo "unknown argument: $arg (expected NZ=, FIXED_REF=, N_TIME_SKIP=, EXTENSION=, SWEEP_PARTS=, FILTER= or PLOTS=)" >&2; exit 2;;
esac; done
# The allocation to charge and the Python to run: defaults in khape_defaults.sh, overridden from the environment.
source "$(dirname "$0")/../khape_defaults.sh"
check_khape_python
case $EXTENSION in edge|odd) ;; *) echo "EXTENSION must be edge or odd, not '$EXTENSION'" >&2; exit 2;; esac
[ "$EXTENSION" = "edge" ] && EXT_TAG="" || EXT_TAG="_${EXTENSION}"
SWEEP_PARTS=${SWEEP_PARTS:-$(default_sweep_parts $NZ)} || exit 2

AFTER_FILTER=""
if [ "$FILTER" = "1" ]; then
    FILTER_NAME="sweep_filter_Nz${NZ}_Ri0.10${EXT_TAG}"
    FILTER_JOB=$(qsub -N "$FILTER_NAME" \
                      -A "$KHAPE_ACCOUNT" \
                      -o "logs/${FILTER_NAME}.log" \
                      -e "logs/${FILTER_NAME}.log" \
                      -v NZ=$NZ,N_TIME_SKIP=$N_TIME_SKIP,EXTENSION=$EXTENSION,KHAPE_PYTHON=$KHAPE_PYTHON$KHAPE_REDIRECT \
                      sweep_filter.pbs)
    echo "Submitted filter job (Nz=$NZ, extension=$EXTENSION): $FILTER_JOB"
    AFTER_FILTER="afterok:$FILTER_JOB"
else
    FILTERED="${KHAPE_PP_OUTPUT:-output}/khi_Nz${NZ}_Ri0.10_filtered_velocities_sweep${EXT_TAG}.nc"
    [ -f "$FILTERED" ] || { echo "FILTER=0, but there is no filtered file to start from: $FILTERED does not exist" >&2; exit 2; }
    echo "FILTER=0: the transfer reads the filtered fields already in $FILTERED"
fi

# TRANSFER_JOBS collects the job each transfer finishes with, for the plots to wait on.
TRANSFER_JOBS=""
submit_transfer() {
    local fr=$1
    [ "$fr" = "1" ] && REF_SUFFIX="_fixed_ref" || REF_SUFFIX=""
    local NAME="sweep_transfer_Nz${NZ}_Ri0.10${REF_SUFFIX}${EXT_TAG}"
    local JOB
    JOB=$(submit_sweep_transfer "$NAME" "$SWEEP_PARTS" "$AFTER_FILTER" \
          "NZ=$NZ,FIXED_REF=$fr,EXTENSION=$EXTENSION,KHAPE_PYTHON=$KHAPE_PYTHON$KHAPE_REDIRECT") || exit 2
    echo "Submitted transfer FIXED_REF=$fr in $SWEEP_PARTS part(s)${FILTER_JOB:+ (depends on $FILTER_JOB)}: $JOB"
    TRANSFER_JOBS="$TRANSFER_JOBS:$JOB"
}

if [ "$FIXED_REF" = "both" ]; then
    submit_transfer 0
    submit_transfer 1
else
    submit_transfer "$FIXED_REF"
fi

if [ "$PLOTS" = "1" ]; then
    PLOTS_NAME="plots_Nz${NZ}_Ri0.10"
    PLOTS_JOB=$(qsub -N "$PLOTS_NAME" \
                     -A "$KHAPE_ACCOUNT" \
                     -o "logs/${PLOTS_NAME}.log" \
                     -e "logs/${PLOTS_NAME}.log" \
                     -v NZ=$NZ,KHAPE_PYTHON=$KHAPE_PYTHON$KHAPE_REDIRECT \
                     -W depend=afterok$TRANSFER_JOBS \
                     plots.pbs)
    echo "Submitted final plots (depends on ${TRANSFER_JOBS#:}): $PLOTS_JOB"
fi
