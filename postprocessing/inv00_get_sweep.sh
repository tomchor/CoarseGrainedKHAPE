#!/usr/bin/env bash
set -euo pipefail

FILENAME="${1:-output/khi_Nz128_Ri0.10.nc}"

python sweep1_filter_fields.py        --filename "$FILENAME"
python sweep2_energy_transfer.py      --filename "$FILENAME"
python sweep3_plot_transfer_spectrum.py --filename "$FILENAME"
