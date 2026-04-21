#!/bin/bash
# Submit simulation and post-processing as a chained PBS job pair.
# Post-processing only runs if simulation succeeds (afterok dependency).
#
# Usage: bash submit_all_pbs.sh [NZ=2048] [FIXED_REF=0]
#   NZ         vertical resolution
#   FIXED_REF  use fixed-in-time reference profile: 0 or 1
#
# To run post-processing alone:
#   bash postprocessing/submit_budgeting.sh [NZ=2048] [FIXED_REF=0|1|both]

NZ=2048; FIXED_REF=0
for arg in "$@"; do case $arg in NZ=*) NZ="${arg#*=}";; FIXED_REF=*) FIXED_REF="${arg#*=}";; esac; done
[ "$FIXED_REF" = "1" ] && REF_SUFFIX="_fixed_ref" || REF_SUFFIX=""

SIM_JOB=$(qsub -N kelvin_helmholtz_${NZ} \
               -o logs/kelvin_helmholtz_${NZ}.log \
               -e logs/kelvin_helmholtz_${NZ}.log \
               -v NZ=$NZ simulation.pbs)
echo "Submitted simulation (Nz=$NZ): $SIM_JOB"

cd postprocessing

BF_NAME="budgeting_filter_Nz${NZ}_Ri0.10"
BF_JOB=$(qsub -N "$BF_NAME" \
              -o "logs/${BF_NAME}.log" \
              -e "logs/${BF_NAME}.log" \
              -v NZ=$NZ \
              -W depend=afterok:$SIM_JOB \
              budgeting_filter.pbs)
echo "Submitted budgeting filter (depends on $SIM_JOB): $BF_JOB"

PP_NAME="budgeting_Nz${NZ}_Ri0.10${REF_SUFFIX}"
PP_JOB=$(qsub -N "$PP_NAME" \
              -o "logs/${PP_NAME}.log" \
              -e "logs/${PP_NAME}.log" \
              -v NZ=$NZ,FIXED_REF=$FIXED_REF \
              -W depend=afterok:$BF_JOB \
              budgeting.pbs)
echo "Submitted budgeting (depends on $BF_JOB): $PP_JOB"

SF_NAME="sweep_filter_Nz${NZ}_Ri0.10"
SF_JOB=$(qsub -N "$SF_NAME" \
              -o "logs/${SF_NAME}.log" \
              -e "logs/${SF_NAME}.log" \
              -v NZ=$NZ \
              -W depend=afterok:$PP_JOB \
              sweep_filter.pbs)
echo "Submitted sweep filter (depends on $PP_JOB): $SF_JOB"

ST_NAME="sweep_transfer_Nz${NZ}_Ri0.10${REF_SUFFIX}"
ST_JOB=$(qsub -N "$ST_NAME" \
              -o "logs/${ST_NAME}.log" \
              -e "logs/${ST_NAME}.log" \
              -v NZ=$NZ,FIXED_REF=$FIXED_REF \
              -W depend=afterok:$SF_JOB \
              sweep_transfer.pbs)
echo "Submitted sweep transfer (depends on $SF_JOB): $ST_JOB"

cd ..
