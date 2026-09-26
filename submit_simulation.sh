#!/usr/bin/env bash
# Usage: bash submit_simulation.sh [NZ=1024] [SAVE_TENSORS=0] [SAVE_SORTED=1]
#   NZ            vertical resolution
#   SAVE_TENSORS  also write the per-scale strain/stress tensor components (0 or 1, for online-vs-offline validation)
#   SAVE_SORTED   also write the Winters (1995) sorted reference state and the online APE budget terms (0 or 1;
#                 default 1 -- the validation scripts and the online panels animation use them; the budget does not)
NZ=1024
SAVE_TENSORS=0
SAVE_SORTED=1
for arg in "$@"; do case $arg in NZ=*) NZ="${arg#*=}";; SAVE_TENSORS=*) SAVE_TENSORS="${arg#*=}";; SAVE_SORTED=*) SAVE_SORTED="${arg#*=}";; esac; done
NAME="kelvin_helmholtz_${NZ}"
qsub -N "$NAME" \
     -o "logs/${NAME}.log" \
     -e "logs/${NAME}.log" \
     -v NZ=$NZ,SAVE_TENSORS=$SAVE_TENSORS,SAVE_SORTED=$SAVE_SORTED \
     simulation.pbs
echo "Submitted simulation (Nz=$NZ, save_tensors=$SAVE_TENSORS, save_sorted=$SAVE_SORTED): $NAME"
