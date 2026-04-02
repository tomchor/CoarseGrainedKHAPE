#!/usr/bin/env bash
set -euo pipefail

FILENAME="${1:-output/khi_45x1x128.nc}"

python 01_filter_and_prepare_fields.py --filename "$FILENAME"
python 02_energy_transfer.py           --filename "$FILENAME"
python 03_sfs_ke_budget.py             --filename "$FILENAME"
python 04_sfs_ape_budget.py            --filename "$FILENAME"
python 05_plot_budgets.py              --filename "$FILENAME"
