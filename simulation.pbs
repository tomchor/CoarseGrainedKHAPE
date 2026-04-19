#!/bin/bash -l
#PBS -A UMCP0028
#PBS -N kelvin_helmholtz_1024
#PBS -o logs/kelvin_helmholtz_1024.log
#PBS -e logs/kelvin_helmholtz_1024.log
#PBS -l walltime=23:59:00
#PBS -q casper
#PBS -M tchor@umd.edu
#PBS -m ae
#PBS -r n
#PBS -l select=1:ncpus=8:ngpus=1:gpu_type=a100:mem=64GB
#PBS -l job_priority=regular

# Clear the environment from any previously loaded modules
module li
module --force purge
module load ncarenv/25.10
module li

#/glade/u/apps/ch/opt/usr/bin/dumpenv # Dumps environment (for debugging with CISL support)

echo $CUDA_VISIBLE_DEVICES

NZ=${NZ:-1024}

export JULIA_DEPOT_PATH="$WORK/.julia"
export JULIA_CPU_TARGET="generic"
juliaup default 1.12
time julia --project -t 8 kelvin_helmholtz_instability.jl --Nz $NZ 2>&1 | tee logs/kelvin_helmholtz_${NZ}.out

qstat -f $PBS_JOBID >> logs/kelvin_helmholtz_${NZ}.log
qstat -f $PBS_JOBID >> logs/kelvin_helmholtz_${NZ}.out
