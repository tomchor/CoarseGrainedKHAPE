#!/usr/bin/env bash
# Usage: bash submit_sweep.sh [NZ=4096] [FIXED_REF=0]
NZ=4096; FIXED_REF=0
for arg in "$@"; do case $arg in NZ=*) NZ="${arg#*=}";; FIXED_REF=*) FIXED_REF="${arg#*=}";; esac; done
[ "$FIXED_REF" = "1" ] && REF_SUFFIX="_fixed_ref" || REF_SUFFIX=""
NAME="sweep_Nz${NZ}_Ri0.10${REF_SUFFIX}"
qsub -N "$NAME" \
     -o "logs/${NAME}.log" \
     -e "logs/${NAME}.log" \
     -v NZ=$NZ,FIXED_REF=$FIXED_REF \
     sweep.pbs
echo "Submitted sweep (Nz=$NZ, FIXED_REF=$FIXED_REF): $NAME"
