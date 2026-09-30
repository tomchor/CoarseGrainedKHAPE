#!/usr/bin/env bash
# Submit the budgeting job: assemble the SFS budgets from the simulation's online terms and plot them.
# Usage: bash submit_budgeting.sh [NZ=2048]
NZ=2048
# An unknown KEY=VALUE is refused rather than ignored, so a misspelled flag cannot silently fall back to its default.
for arg in "$@"; do case $arg in
  NZ=*) NZ="${arg#*=}";;
  *) echo "unknown argument: $arg (expected NZ=)" >&2; exit 2;;
esac; done
# The allocation to charge and the Python to run: defaults in khape_defaults.sh, overridden from the environment.
source "$(dirname "$0")/../khape_defaults.sh"
check_khape_python

NAME="budgeting_Nz${NZ}_Ri0.10"
JOB=$(qsub -N "$NAME" \
           -A "$KHAPE_ACCOUNT" \
           -o "logs/${NAME}.log" \
           -e "logs/${NAME}.log" \
           -v NZ=$NZ,KHAPE_PYTHON=$KHAPE_PYTHON \
           budgeting.pbs)
echo "Submitted budget job (Nz=$NZ): $JOB"
