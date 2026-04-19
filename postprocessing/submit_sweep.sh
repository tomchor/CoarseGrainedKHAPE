#!/usr/bin/env bash
# Usage: bash submit_sweep.sh [NZ]
NZ=${1:-4096}
NAME="sweep_Nz${NZ}_Ri0.10"
qsub -N "$NAME" \
     -o "logs/${NAME}.log" \
     -e "logs/${NAME}.log" \
     -v NZ=$NZ \
     sweep.pbs
echo "Submitted sweep (Nz=$NZ): $NAME"
