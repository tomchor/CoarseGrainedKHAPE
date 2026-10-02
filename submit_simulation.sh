#!/usr/bin/env bash
# Usage: bash submit_simulation.sh [NZ=512] [NGPUS=] [SAVE_TENSORS=0] [SAVE_SORTED=1] [OFFLINE_CHECK=0]
#   NZ            vertical resolution
#   NGPUS         GPUs to run on (default: 1 up to NZ=512, 4 up to NZ=1024; see default_ngpus in khape_defaults.sh).
#                 One GPU runs as always; more split the domain into x-slabs, one MPI rank per GPU (whole 4-GPU
#                 Casper nodes, so a multiple of 4), and chain a merge job that stitches the rank files into the
#                 one-GPU layout and draws the figures (see the README, Multi-GPU runs)
#   SAVE_TENSORS  also write the per-scale strain/stress tensor components (0 or 1, for online-vs-offline validation)
#   SAVE_SORTED   also write the validation-only view of the Winters (1995) sorted reference state (0 or 1;
#                 default 1 on one GPU -- the validation scripts and the online panels animation use them; the budget
#                 does not -- and 0, the only choice, on several, where its model-grid sorts would sort each rank's slab)
#   OFFLINE_CHECK also write each 3D output as a consecutive-iteration pair, which only the offline check
#                 (`pytest --offline-check`) reads, at twice the 3D output (0 or 1)
NZ=512
NGPUS=""
SAVE_TENSORS=0
SAVE_SORTED=""
OFFLINE_CHECK=0
# An unknown KEY=VALUE is refused rather than ignored, so a misspelled flag cannot silently fall back to its default.
for arg in "$@"; do case $arg in
  NZ=*)           NZ="${arg#*=}";;
  NGPUS=*)        NGPUS="${arg#*=}";;
  SAVE_TENSORS=*) SAVE_TENSORS="${arg#*=}";;
  SAVE_SORTED=*)  SAVE_SORTED="${arg#*=}";;
  OFFLINE_CHECK=*) OFFLINE_CHECK="${arg#*=}";;
  *) echo "unknown argument: $arg (expected NZ=, NGPUS=, SAVE_TENSORS=, SAVE_SORTED= or OFFLINE_CHECK=)" >&2; exit 2;;
esac; done
# The allocation to charge, and the Python for the figures simulation.pbs draws after the run: defaults in
# khape_defaults.sh, overridden from the environment.
source "$(dirname "$0")/khape_defaults.sh"
check_khape_python
NGPUS=${NGPUS:-$(default_ngpus $NZ)} || exit 2
if [ "$NGPUS" = "1" ]; then
    SAVE_SORTED=${SAVE_SORTED:-1}
elif [ "${SAVE_SORTED:-0}" = "1" ]; then
    echo "error: SAVE_SORTED=1 needs a single GPU (NGPUS=1): its model-grid sorts would each sort one rank's slab" >&2; exit 2
else
    SAVE_SORTED=0
fi
NAME="kelvin_helmholtz_${NZ}"
JOB=$(submit_simulation_job "$NAME" $NZ $NGPUS \
      "NZ=$NZ,SAVE_TENSORS=$SAVE_TENSORS,SAVE_SORTED=$SAVE_SORTED,OFFLINE_CHECK=$OFFLINE_CHECK,KHAPE_PYTHON=$KHAPE_PYTHON$KHAPE_REDIRECT") || exit 2
echo "Submitted simulation (Nz=$NZ, ngpus=$NGPUS, save_tensors=$SAVE_TENSORS, save_sorted=$SAVE_SORTED, offline_check=$OFFLINE_CHECK): $JOB"
