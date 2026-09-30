#!/usr/bin/env bash
# Build and plot the SFS KE and APE budgets for one run, from the terms the simulation computes online.
#
#   01_online_budgets.py   assembles <stem>_sfs_{ke,ape}_budget_{fields,integrated}.nc from the online terms
#   02_plot_budgets.py     plots the integrated budgets, one figure per filter scale
#
# Usage: bash 00_get_budgets.sh [output/khi_Nz128_Ri0.10.nc] [--filter-scales 1 7]
# Every remaining argument goes to 01 (e.g. --filter-scales; default: every scale the simulation wrote).
#
# The offline pipeline that used to compute these terms lives in offline/ and runs only as the CI cross-check:
#   bash offline/run_offline_budgets.sh output/khi_Nz128_Ri0.10.nc --filter-scales 1 7
#   pytest tests/ --offline-check
set -euo pipefail
cd "$(dirname "$0")"
FILENAME="${1:-output/khi_Nz128_Ri0.10.nc}"   # the CI run, which tests/conftest.py names as STEM
shift 1 2>/dev/null || true
python 01_online_budgets.py --filename "$FILENAME" "$@"
python 02_plot_budgets.py   --filename "$FILENAME"
