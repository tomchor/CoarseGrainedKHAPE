#!/usr/bin/env bash
# Usage: bash submit_simulation.sh [NZ]
NZ=${1:-1024}
NAME="kelvin_helmholtz_${NZ}"
qsub -N "$NAME" \
     -o "logs/${NAME}.log" \
     -e "logs/${NAME}.log" \
     -v NZ=$NZ \
     simulation.pbs
echo "Submitted simulation (Nz=$NZ): $NAME"
