#!/usr/bin/env bash
# Usage: bash submit_simulation.sh [NZ=512] [SAVE_TENSORS=0] [SAVE_SORTED=1] [OFFLINE_CHECK=0]
#   NZ            vertical resolution
#   SAVE_TENSORS  also write the per-scale strain/stress tensor components (0 or 1, for online-vs-offline validation)
#   SAVE_SORTED   also write the validation-only view of the Winters (1995) sorted reference state (0 or 1;
#                 default 1 -- the validation scripts and the online panels animation use them; the budget does not)
#   OFFLINE_CHECK also write each 3D output as a consecutive-iteration pair, which only the offline check
#                 (`pytest --offline-check`) reads, at twice the 3D output (0 or 1)
NZ=512
SAVE_TENSORS=0
SAVE_SORTED=1
OFFLINE_CHECK=0
# An unknown KEY=VALUE is refused rather than ignored, so a misspelled flag cannot silently fall back to its default.
for arg in "$@"; do case $arg in
  NZ=*)           NZ="${arg#*=}";;
  SAVE_TENSORS=*) SAVE_TENSORS="${arg#*=}";;
  SAVE_SORTED=*)  SAVE_SORTED="${arg#*=}";;
  OFFLINE_CHECK=*) OFFLINE_CHECK="${arg#*=}";;
  *) echo "unknown argument: $arg (expected NZ=, SAVE_TENSORS=, SAVE_SORTED= or OFFLINE_CHECK=)" >&2; exit 2;;
esac; done
# The allocation to charge: default in khape_defaults.sh, overridden from the environment.
source "$(dirname "$0")/khape_defaults.sh"
NAME="kelvin_helmholtz_${NZ}"
qsub -N "$NAME" \
     -A "$KHAPE_ACCOUNT" \
     -o "logs/${NAME}.log" \
     -e "logs/${NAME}.log" \
     -v NZ=$NZ,SAVE_TENSORS=$SAVE_TENSORS,SAVE_SORTED=$SAVE_SORTED,OFFLINE_CHECK=$OFFLINE_CHECK \
     simulation.pbs
echo "Submitted simulation (Nz=$NZ, save_tensors=$SAVE_TENSORS, save_sorted=$SAVE_SORTED, offline_check=$OFFLINE_CHECK): $NAME"
