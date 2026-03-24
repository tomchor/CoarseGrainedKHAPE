#!/bin/bash -l
#PBS -A UMCP0028
#PBS -N postprocessing_2916x1x4096
#PBS -o logs/postprocessing_2916x1x4096.log
#PBS -e logs/postprocessing_2916x1x4096.log
#PBS -l walltime=23:59:00
#PBS -q casper
#PBS -M tchor@umd.edu
#PBS -m ae
#PBS -r n
#PBS -l select=1:ncpus=18:mem=1400GB:ngpus=0
#PBS -l job_priority=premium

# Clear the environment from any previously loaded modules
module li
module --force purge
module load ncarenv/25.10
module li

#/glade/u/apps/ch/opt/usr/bin/dumpenv # Dumps environment (for debugging with CISL support)

echo $CUDA_VISIBLE_DEVICES

# Disable HDF5 file locking — Lustre (GPFS) advisory locks are unreliable and
# cause EAGAIN failures when multiple dask worker processes write to the same file.
export HDF5_USE_FILE_LOCKING=FALSE

export JULIA_DEPOT_PATH="$WORK/.julia"
export JULIA_CPU_TARGET="generic"
juliaup default 1.12

time /glade/u/home/tomasc/miniconda3/envs/py313/bin/python -u \
    postprocessing/01_filter_fields.py \
    --filename output/khi_2916x1x4096.nc \
    --n-workers 18 \
    --threads-per-worker 3 \
    2>&1 | tee logs/01_filter_fields_2916x1x4096.out
qstat -f $PBS_JOBID >> logs/01_filter_fields_2916x1x4096.log
qstat -f $PBS_JOBID >> logs/01_filter_fields_2916x1x4096.out

time /glade/u/home/tomasc/miniconda3/envs/py313/bin/python -u \
    postprocessing/02_energy_transfer.py \
    --filename output/khi_2916x1x4096.nc \
    --n-workers 18 \
    2>&1 | tee logs/02_energy_transfer_2916x1x4096.out
qstat -f $PBS_JOBID >> logs/02_energy_transfer_2916x1x4096.log
qstat -f $PBS_JOBID >> logs/02_energy_transfer_2916x1x4096.out

# time /glade/u/home/tomasc/miniconda3/envs/py313/bin/python -u \
#     postprocessing/03_sfs_ke_budget.py \
#     --filename output/khi_2916x1x4096.nc \
#     2>&1 | tee logs/03_sfs_ke_budget_2916x1x4096.out
# qstat -f $PBS_JOBID >> logs/03_sfs_ke_budget_2916x1x4096.log
# qstat -f $PBS_JOBID >> logs/03_sfs_ke_budget_2916x1x4096.out

# time /glade/u/home/tomasc/miniconda3/envs/py313/bin/python -u \
#     postprocessing/04_sfs_ape_budget.py \
#     --filename output/khi_2916x1x4096.nc \
#     --n-workers 18 \
#     2>&1 | tee logs/04_sfs_ape_budget_2916x1x4096.out
# qstat -f $PBS_JOBID >> logs/04_sfs_ape_budget_2916x1x4096.log
# qstat -f $PBS_JOBID >> logs/04_sfs_ape_budget_2916x1x4096.out
