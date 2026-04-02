#!/bin/bash -l
#PBS -A UMCP0028
#PBS -N postprocessing_Nz4096_Ri0.10
#PBS -o logs/postprocessing_Nz4096_Ri0.10.log
#PBS -e logs/postprocessing_Nz4096_Ri0.10.log
#PBS -l walltime=23:59:00
#PBS -q casper
#PBS -M tchor@umd.edu
#PBS -m ae
#PBS -r n
#PBS -l select=1:ncpus=18:mem=1400GB:ngpus=0
#PBS -l job_priority=premium

SIM=Nz4096_Ri0.10
PYTHON=/glade/u/home/tomasc/miniconda3/envs/py313/bin/python

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

time $PYTHON -u postprocessing/01_filter_fields.py --filename output/khi_${SIM}.nc 2>&1 | tee logs/01_filter_fields_${SIM}.out
qstat -f $PBS_JOBID >> logs/01_filter_fields_${SIM}.log
qstat -f $PBS_JOBID >> logs/01_filter_fields_${SIM}.out

time $PYTHON -u postprocessing/02_energy_transfer.py --filename output/khi_${SIM}.nc --n-workers 18 2>&1 | tee logs/02_energy_transfer_${SIM}.out
qstat -f $PBS_JOBID >> logs/02_energy_transfer_${SIM}.log
qstat -f $PBS_JOBID >> logs/02_energy_transfer_${SIM}.out

# time $PYTHON -u postprocessing/03_sfs_ke_budget.py --filename output/khi_${SIM}.nc 2>&1 | tee logs/03_sfs_ke_budget_${SIM}.out
# qstat -f $PBS_JOBID >> logs/03_sfs_ke_budget_${SIM}.log
# qstat -f $PBS_JOBID >> logs/03_sfs_ke_budget_${SIM}.out

# time $PYTHON -u postprocessing/04_sfs_ape_budget.py --filename output/khi_${SIM}.nc --n-workers 18 2>&1 | tee logs/04_sfs_ape_budget_${SIM}.out
# qstat -f $PBS_JOBID >> logs/04_sfs_ape_budget_${SIM}.log
# qstat -f $PBS_JOBID >> logs/04_sfs_ape_budget_${SIM}.out
