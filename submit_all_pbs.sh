#!/bin/bash
# Submit simulation and post-processing as a chained PBS job pair.
# Post-processing only runs if simulation succeeds (afterok dependency).
#
# Usage: bash submit_all_pbs.sh [Nz]
#   Nz  vertical resolution (default: 2084)
#
# To run post-processing alone:
#   qsub -v NZ=<Nz> postprocessing/submit_budgeting_pbs.sh

NZ=${1:-2084}

SIM_JOB=$(qsub -N kelvin_helmholtz_${NZ} \
               -o logs/kelvin_helmholtz_${NZ}.log \
               -e logs/kelvin_helmholtz_${NZ}.log \
               -v NZ=$NZ submit_simulation_pbs.sh)
echo "Submitted simulation (Nz=$NZ): $SIM_JOB"


cd postprocessing
PP_JOB=$(qsub -N budgeting_Nz${NZ}_Ri0.10 \
              -o logs/budgeting_Nz${NZ}_Ri0.10.log \
              -e logs/budgeting_Nz${NZ}_Ri0.10.log \
              -v NZ=$NZ \
              -W depend=afterok:$SIM_JOB submit_budgeting_pbs.sh)
echo "Submitted post-processing (depends on $SIM_JOB): $PP_JOB"

SWEEP_JOB=$(qsub -N sweep_Nz${NZ}_Ri0.10 \
                 -o logs/sweep_Nz${NZ}_Ri0.10.log \
                 -e logs/sweep_Nz${NZ}_Ri0.10.log \
                 -v NZ=$NZ \
                 -W depend=afterok:$PP_JOB submit_sweep_pbs.sh)
echo "Submitted sweep (depends on $PP_JOB): $SWEEP_JOB"
