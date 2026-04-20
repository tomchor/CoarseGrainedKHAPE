#!/usr/bin/env bash
# Usage: bash submit_simulation.sh [NZ=1024]
NZ=1024
for arg in "$@"; do case $arg in NZ=*) NZ="${arg#*=}";; esac; done
NAME="kelvin_helmholtz_${NZ}"
qsub -N "$NAME" \
     -o "logs/${NAME}.log" \
     -e "logs/${NAME}.log" \
     -v NZ=$NZ \
     simulation.pbs
echo "Submitted simulation (Nz=$NZ): $NAME"
