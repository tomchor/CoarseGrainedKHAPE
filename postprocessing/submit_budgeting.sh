#!/usr/bin/env bash
# Submit the budgeting job: assemble the SFS budgets from the simulation's online terms and plot them.
# Usage: bash submit_budgeting.sh [NZ=512]
NZ=512
# An unknown KEY=VALUE is refused rather than ignored, so a misspelled flag cannot silently fall back to its default.
for arg in "$@"; do case $arg in
  NZ=*) NZ="${arg#*=}";;
  *) echo "unknown argument: $arg (expected NZ=)" >&2; exit 2;;
esac; done
# The allocation to charge and the Python to run belong to whoever submits, so they come from the environment.
: "${KHAPE_ACCOUNT:?set KHAPE_ACCOUNT to the project code to charge; see the README, Environment}"
: "${KHAPE_PYTHON:?set KHAPE_PYTHON to the python of your py313 environment; see the README, Environment}"

NAME="budgeting_Nz${NZ}_Ri0.10"
JOB=$(qsub -N "$NAME" \
           -A "$KHAPE_ACCOUNT" \
           -o "logs/${NAME}.log" \
           -e "logs/${NAME}.log" \
           -v NZ=$NZ,KHAPE_PYTHON=$KHAPE_PYTHON \
           budgeting.pbs)
echo "Submitted budget job (Nz=$NZ): $JOB"
