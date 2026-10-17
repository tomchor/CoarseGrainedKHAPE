#!/usr/bin/env bash
# Run the OFFLINE budget pipeline (01 filter -> 02 sort -> 03 transfer -> 04 KE -> 05 APE) on a simulation
# output. This is not the production path any more: 01_online_budgets.py assembles the budgets from the terms
# the simulation computes online, and this pipeline exists to check them (`pytest tests/ --offline-check`,
# which runs it and compares every term). Its output goes to an `offline/` subdirectory of the post-processing
# output directory, so it never overwrites the assembled budgets.
#
# Usage: bash offline/run_offline_budgets.sh [output/khi_Nz128_Ri0.10.nc] [--filter-scales 1 7]
#   N_WORKERS  threads for the sort and the APE lookups (default 1)
set -euo pipefail
cd "$(dirname "$0")/.."           # postprocessing/, where the scripts import `src.*` from
FILENAME="${1:-output/khi_Nz128_Ri0.10.nc}"
shift 1 2>/dev/null || true
export KHAPE_PP_OUTPUT="${KHAPE_PP_OUTPUT:-$(pwd)/output}/offline"
mkdir -p "$KHAPE_PP_OUTPUT"
echo "offline budget output -> $KHAPE_PP_OUTPUT"
python offline/01_filter_fields.py    --filename "$FILENAME" "$@"
python offline/02_sort_density.py     --filename "$FILENAME" --n-workers "${N_WORKERS:-1}"
python offline/03_energy_transfer.py  --filename "$FILENAME" --n-workers "${N_WORKERS:-1}"
python offline/04_sfs_ke_budget.py    --filename "$FILENAME" --recompute-online-terms
python offline/05_sfs_ape_budget.py   --filename "$FILENAME" --n-workers "${N_WORKERS:-1}"
