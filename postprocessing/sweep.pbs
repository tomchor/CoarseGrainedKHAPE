#!/bin/bash -l
#PBS -A UMCP0028
#PBS -N sweep_Nz4096_Ri0.10
#PBS -o logs/sweep_Nz4096_Ri0.10.log
#PBS -e logs/sweep_Nz4096_Ri0.10.log
#PBS -l walltime=23:59:00
#PBS -q casper
#PBS -M tchor@umd.edu
#PBS -m ae
#PBS -r n
#PBS -l select=1:ncpus=18:mem=730GB:ngpus=0
#PBS -l job_priority=premium

NZ=${NZ:-4096}
SIM=khi_Nz${NZ}_Ri0.10
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

time $PYTHON -u inv1_filter_fields_sweep.py --filename output/${SIM}.nc 2>&1
qstat -f $PBS_JOBID >> logs/inv1_filter_fields_sweep_${SIM}.log
qstat -f $PBS_JOBID >> logs/inv1_filter_fields_sweep_${SIM}.out

time $PYTHON -u inv2_energy_transfer_sweep.py --filename output/${SIM}.nc --n-workers 18 2>&1
qstat -f $PBS_JOBID >> logs/inv2_energy_transfer_sweep_${SIM}.log
qstat -f $PBS_JOBID >> logs/inv2_energy_transfer_sweep_${SIM}.out

time $PYTHON -u inv3_plot_transfer_spectrum.py --filename output/${SIM}.nc 2>&1
qstat -f $PBS_JOBID >> logs/inv3_plot_transfer_spectrum_${SIM}.log
qstat -f $PBS_JOBID >> logs/inv3_plot_transfer_spectrum_${SIM}.out
