#!/usr/bin/env bash
set -euo pipefail

FILENAME="${1:-output/khi_Nz512_Ri0.10.nc}"
shift 1 2>/dev/null || true

# Separate --fixed-reference and --reference <x> from the remaining args (e.g. --filter-scales).
# --fixed-reference goes to 02-06; --reference selects the scale decomposition and goes to 03, 04 and 05,
# the steps that measure a resolved reservoir against a reference profile, and to 06, which has to find
# their output (`--reference true` tags it _trueref). The rest go to script 01 only.
FIXED_REF_FLAG=""
REFERENCE_FLAG=()
REMAINING_ARGS=()
expect_reference=0
for arg in "$@"; do
    if [ "$expect_reference" = "1" ]; then
        REFERENCE_FLAG=(--reference "$arg")
        expect_reference=0
    elif [ "$arg" = "--fixed-reference" ]; then
        FIXED_REF_FLAG="--fixed-reference"
    elif [ "$arg" = "--reference" ]; then
        expect_reference=1
    elif [[ "$arg" == --reference=* ]]; then      # the --reference=true form argparse also accepts
        REFERENCE_FLAG=(--reference "${arg#--reference=}")
    else
        REMAINING_ARGS+=("$arg")
    fi
done
if [ "$expect_reference" = "1" ]; then
    echo "error: --reference needs a value (filtered or true)" >&2
    exit 2
fi

python 01_filter_fields.py    --filename "$FILENAME" "${REMAINING_ARGS[@]+"${REMAINING_ARGS[@]}"}"
python 02_sort_density.py     --filename "$FILENAME" $FIXED_REF_FLAG --n-workers "${N_WORKERS:-1}"
python 03_energy_transfer.py  --filename "$FILENAME" $FIXED_REF_FLAG "${REFERENCE_FLAG[@]+"${REFERENCE_FLAG[@]}"}" --n-workers "${N_WORKERS:-1}"
python 04_sfs_ke_budget.py    --filename "$FILENAME" $FIXED_REF_FLAG "${REFERENCE_FLAG[@]+"${REFERENCE_FLAG[@]}"}"
python 05_sfs_ape_budget.py   --filename "$FILENAME" $FIXED_REF_FLAG "${REFERENCE_FLAG[@]+"${REFERENCE_FLAG[@]}"}" --n-workers "${N_WORKERS:-1}"
python 06_plot_budgets.py     --filename "$FILENAME" $FIXED_REF_FLAG "${REFERENCE_FLAG[@]+"${REFERENCE_FLAG[@]}"}"
