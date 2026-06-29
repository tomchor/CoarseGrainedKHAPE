#!/bin/bash
# Submit the simulation and post-processing as chained PBS jobs (afterok dependencies);
# each stage only runs if the previous one succeeds. Optional validation and plotting stages.
#
#   simulation → budgeting_filter → budgeting → sweep_filter → sweep_transfer   (always)
#   + validation  (online-vs-offline figures + animations; parallel after sim)  (VALIDATE=1)
#   + plots       (plot2 transfer spectrum, plot3 budgets, plot4 panels)        (PLOTS=1)
#
# Usage: bash submit_all_pbs.sh [NZ=2048] [FIXED_REF=0] [VALIDATE=0] [PLOTS=0]
#   NZ         vertical resolution
#   FIXED_REF  use fixed-in-time reference profile: 0 or 1
#   VALIDATE   also run the online-vs-offline validation (runs the simulation with --save_tensors
#              so the strain/stress tensor comparison works): 0 or 1
#   PLOTS      also run the final plots after sweep_transfer: 0 or 1
#
# To run post-processing alone:
#   bash postprocessing/submit_budgeting.sh [NZ=2048] [FIXED_REF=0|1|both]

NZ=2048; FIXED_REF=0; VALIDATE=0; PLOTS=0
for arg in "$@"; do case $arg in
  NZ=*)        NZ="${arg#*=}";;
  FIXED_REF=*) FIXED_REF="${arg#*=}";;
  VALIDATE=*)  VALIDATE="${arg#*=}";;
  PLOTS=*)     PLOTS="${arg#*=}";;
esac; done
[ "$FIXED_REF" = "1" ] && REF_SUFFIX="_fixed_ref" || REF_SUFFIX=""
[ "$VALIDATE" = "1" ] && SAVE_TENSORS=1 || SAVE_TENSORS=0   # validation needs the per-scale tensors

SIM_JOB=$(qsub -N kelvin_helmholtz_${NZ} \
               -o logs/kelvin_helmholtz_${NZ}.log \
               -e logs/kelvin_helmholtz_${NZ}.log \
               -v NZ=$NZ,SAVE_TENSORS=$SAVE_TENSORS simulation.pbs)
echo "Submitted simulation (Nz=$NZ, save_tensors=$SAVE_TENSORS): $SIM_JOB"

# Optional validation — parallel branch, runs after the simulation succeeds
if [ "$VALIDATE" = "1" ]; then
    cd postprocessing/validation
    mkdir -p logs
    VAL_NAME="validation_Nz${NZ}_Ri0.10"
    VAL_JOB=$(qsub -N "$VAL_NAME" \
                   -o "logs/${VAL_NAME}.log" \
                   -e "logs/${VAL_NAME}.log" \
                   -v NZ=$NZ \
                   -W depend=afterok:$SIM_JOB \
                   validation.pbs)
    echo "Submitted validation (depends on $SIM_JOB): $VAL_JOB"
    cd ../..
fi

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

# Optional final plots — after sweep_transfer (which is downstream of budgeting, so both are done)
if [ "$PLOTS" = "1" ]; then
    PLOTS_NAME="plots_Nz${NZ}_Ri0.10"
    PLOTS_JOB=$(qsub -N "$PLOTS_NAME" \
                     -o "logs/${PLOTS_NAME}.log" \
                     -e "logs/${PLOTS_NAME}.log" \
                     -v NZ=$NZ \
                     -W depend=afterok:$ST_JOB \
                     plots.pbs)
    echo "Submitted final plots (depends on $ST_JOB): $PLOTS_JOB"
fi
cd ..
