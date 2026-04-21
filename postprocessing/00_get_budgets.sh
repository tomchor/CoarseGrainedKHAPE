#!/usr/bin/env bash
set -euo pipefail

FILENAME="${1:-output/khi_Nz512_Ri0.10.nc}"
shift 1 2>/dev/null || true

# Separate --fixed-reference from the remaining args (e.g. --filter-scales).
# --fixed-reference is passed to scripts 01 and 04; the rest go to script 01 only.
FIXED_REF_FLAG=""
REMAINING_ARGS=()
for arg in "$@"; do
    if [ "$arg" = "--fixed-reference" ]; then
        FIXED_REF_FLAG="--fixed-reference"
    else
        REMAINING_ARGS+=("$arg")
    fi
done

python 01_filter_fields.py    --filename "$FILENAME" "${REMAINING_ARGS[@]+"${REMAINING_ARGS[@]}"}"
python 02_sort_density.py     --filename "$FILENAME" $FIXED_REF_FLAG --n-workers "${N_WORKERS:-1}"
python 03_energy_transfer.py  --filename "$FILENAME" $FIXED_REF_FLAG --n-workers "${N_WORKERS:-1}"
python 04_sfs_ke_budget.py    --filename "$FILENAME" $FIXED_REF_FLAG
python 05_sfs_ape_budget.py   --filename "$FILENAME" $FIXED_REF_FLAG --n-workers "${N_WORKERS:-1}"
python 06_plot_budgets.py     --filename "$FILENAME" $FIXED_REF_FLAG
