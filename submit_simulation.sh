#!/usr/bin/env bash
# Usage: bash submit_simulation.sh [NZ=1024] [SAVE_TENSORS=0] [SAVE_SORTED=0]
#   NZ            vertical resolution
#   SAVE_TENSORS  also write the per-scale strain/stress tensor components (0 or 1, for online-vs-offline validation)
#   SAVE_SORTED   also write the Winters (1995) sorted reference state (0 or 1, for online-vs-offline validation)
NZ=1024
SAVE_TENSORS=0
SAVE_SORTED=0
for arg in "$@"; do case $arg in NZ=*) NZ="${arg#*=}";; SAVE_TENSORS=*) SAVE_TENSORS="${arg#*=}";; SAVE_SORTED=*) SAVE_SORTED="${arg#*=}";; esac; done
NAME="kelvin_helmholtz_${NZ}"
qsub -N "$NAME" \
     -o "logs/${NAME}.log" \
     -e "logs/${NAME}.log" \
     -v NZ=$NZ,SAVE_TENSORS=$SAVE_TENSORS,SAVE_SORTED=$SAVE_SORTED \
     simulation.pbs
echo "Submitted simulation (Nz=$NZ, save_tensors=$SAVE_TENSORS, save_sorted=$SAVE_SORTED): $NAME"
