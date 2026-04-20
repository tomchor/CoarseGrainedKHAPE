#!/usr/bin/env bash
# Usage: bash submit_budgeting.sh [NZ=2048] [FIXED_REF=0]
NZ=2048; FIXED_REF=0
for arg in "$@"; do case $arg in NZ=*) NZ="${arg#*=}";; FIXED_REF=*) FIXED_REF="${arg#*=}";; esac; done
[ "$FIXED_REF" = "1" ] && REF_SUFFIX="_fixed_ref" || REF_SUFFIX=""
NAME="budgeting_Nz${NZ}_Ri0.10${REF_SUFFIX}"
qsub -N "$NAME" \
     -o "logs/${NAME}.log" \
     -e "logs/${NAME}.log" \
     -v NZ=$NZ,FIXED_REF=$FIXED_REF \
     budgeting.pbs
echo "Submitted budgeting (Nz=$NZ, FIXED_REF=$FIXED_REF): $NAME"
