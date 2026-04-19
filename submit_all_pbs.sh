#!/bin/bash
# Submit simulation and post-processing as a chained PBS job pair.
# Post-processing only runs if simulation succeeds (afterok dependency).
#
# Usage: bash submit_all_pbs.sh [NZ] [FIXED_REF]
#   NZ         vertical resolution (default: 2048)
#   FIXED_REF  use fixed-in-time reference profile: 0 or 1 (default: 0)
#
# To run post-processing alone:
#   bash postprocessing/submit_budgeting.sh [NZ] [FIXED_REF]

NZ=${1:-2048}
FIXED_REF=${2:-0}
[ "$FIXED_REF" = "1" ] && REF_SUFFIX="_fixed_ref" || REF_SUFFIX=""

SIM_JOB=$(qsub -N kelvin_helmholtz_${NZ} \
               -o logs/kelvin_helmholtz_${NZ}.log \
               -e logs/kelvin_helmholtz_${NZ}.log \
               -v NZ=$NZ simulation.pbs)
echo "Submitted simulation (Nz=$NZ): $SIM_JOB"

cd postprocessing
PP_NAME="budgeting_Nz${NZ}_Ri0.10${REF_SUFFIX}"
PP_JOB=$(qsub -N "$PP_NAME" \
              -o "logs/${PP_NAME}.log" \
              -e "logs/${PP_NAME}.log" \
              -v NZ=$NZ,FIXED_REF=$FIXED_REF \
              -W depend=afterok:$SIM_JOB budgeting.pbs)
echo "Submitted post-processing (depends on $SIM_JOB): $PP_JOB"

SWEEP_NAME="sweep_Nz${NZ}_Ri0.10"
SWEEP_JOB=$(qsub -N "$SWEEP_NAME" \
                 -o "logs/${SWEEP_NAME}.log" \
                 -e "logs/${SWEEP_NAME}.log" \
                 -v NZ=$NZ \
                 -W depend=afterok:$PP_JOB sweep.pbs)
echo "Submitted sweep (depends on $PP_JOB): $SWEEP_JOB"
