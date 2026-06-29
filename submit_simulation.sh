#!/usr/bin/env bash
# Usage: bash submit_simulation.sh [NZ=1024] [SAVE_TENSORS=0]
#   NZ            vertical resolution
#   SAVE_TENSORS  also write the per-scale strain/stress tensor components (0 or 1, for online-vs-offline validation)
NZ=1024
SAVE_TENSORS=0
for arg in "$@"; do case $arg in NZ=*) NZ="${arg#*=}";; SAVE_TENSORS=*) SAVE_TENSORS="${arg#*=}";; esac; done
NAME="kelvin_helmholtz_${NZ}"
qsub -N "$NAME" \
     -o "logs/${NAME}.log" \
     -e "logs/${NAME}.log" \
     -v NZ=$NZ,SAVE_TENSORS=$SAVE_TENSORS \
     simulation.pbs
echo "Submitted simulation (Nz=$NZ, save_tensors=$SAVE_TENSORS): $NAME"
