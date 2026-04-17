#!/usr/bin/env bash
set -euo pipefail

FILENAME="${1:-output/khi_Nz512_Ri0.10.nc}"
shift 1 2>/dev/null || true  # remaining args (e.g. --filter-scales 0.2 0.4) forwarded to script 01

python 01_filter_and_prepare_fields.py --filename "$FILENAME" "$@"
python 02_energy_transfer.py           --filename "$FILENAME" --n-workers "${N_WORKERS:-1}"
python 03_sfs_ke_budget.py             --filename "$FILENAME"
python 04_sfs_ape_budget.py            --filename "$FILENAME" --n-workers "${N_WORKERS:-1}"
python 05_plot_budgets.py              --filename "$FILENAME"
