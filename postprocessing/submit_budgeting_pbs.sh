#!/bin/bash -l
#PBS -A UMCP0028
#PBS -N budgeting_Nz512_Ri0.10
#PBS -o logs/budgeting_Nz512_Ri0.10.log
#PBS -e logs/budgeting_Nz512_Ri0.10.log
#PBS -l walltime=23:59:00
#PBS -q casper
#PBS -M tchor@umd.edu
#PBS -m ae
#PBS -r n
#PBS -l select=1:ncpus=18:mem=730GB:ngpus=0
#PBS -l job_priority=premium

NZ=${NZ:-512}
SIM=Nz${NZ}_Ri0.10
PYTHON=/glade/u/home/tomasc/miniconda3/envs/py313/bin/python
FIXED_REF=${FIXED_REF:-0}
[ "$FIXED_REF" = "1" ] && FIXED_REF_FLAG="--fixed-reference" || FIXED_REF_FLAG=""

# Clear the environment from any previously loaded modules
module li
module --force purge
module load ncarenv/25.10
module li

#/glade/u/apps/ch/opt/usr/bin/dumpenv # Dumps environment (for debugging with CISL support)

echo $CUDA_VISIBLE_DEVICES

export JULIA_DEPOT_PATH="$WORK/.julia"
export JULIA_CPU_TARGET="generic"
juliaup default 1.12

time $PYTHON -u 01_filter_and_prepare_fields.py --filename output/khi_${SIM}.nc $FIXED_REF_FLAG 2>&1 | tee logs/budgeting_Nz${NZ}_Ri0.10.out
time $PYTHON -u 02_energy_transfer.py --filename output/khi_${SIM}.nc --n-workers 18 $FIXED_REF_FLAG 2>&1 | tee -a logs/budgeting_Nz${NZ}_Ri0.10.out
time $PYTHON -u 03_sfs_ke_budget.py --filename output/khi_${SIM}.nc $FIXED_REF_FLAG 2>&1 | tee -a logs/budgeting_Nz${NZ}_Ri0.10.out
time $PYTHON -u 04_sfs_ape_budget.py --filename output/khi_${SIM}.nc --n-workers 18 $FIXED_REF_FLAG 2>&1 | tee -a logs/budgeting_Nz${NZ}_Ri0.10.out
time $PYTHON -u 05_plot_budgets.py --filename output/khi_${SIM}.nc $FIXED_REF_FLAG 2>&1 | tee -a logs/budgeting_Nz${NZ}_Ri0.10.out

qstat -f $PBS_JOBID >> logs/budgeting_Nz${NZ}_Ri0.10.out