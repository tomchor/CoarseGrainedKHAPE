#!/usr/bin/env bash
# Submit the wall-extension test: the sweep at one filter scale with b extended past the walls by the other
# admissible rule, to compare against the production (edge) sweep with compare_extension.py afterwards.
# Usage: bash submit_extension_test.sh [NZ=2048] [SCALE=20] [EXTENSION=odd] [N_TIME_SKIP=2]
#   SCALE        filter scale to test; extension_test.pbs snaps it to the nearest production sweep scale
#   N_TIME_SKIP  must match the production sweep, or the two are averaged over different times
NZ=2048; SCALE=20; EXTENSION=odd; N_TIME_SKIP=2
for arg in "$@"; do case $arg in
  NZ=*)          NZ="${arg#*=}";;
  SCALE=*)       SCALE="${arg#*=}";;
  EXTENSION=*)   EXTENSION="${arg#*=}";;
  N_TIME_SKIP=*) N_TIME_SKIP="${arg#*=}";;
  *) echo "unknown argument: $arg (expected NZ=, SCALE=, EXTENSION= or N_TIME_SKIP=)" >&2; exit 2;;
esac; done
if [ "$EXTENSION" = "edge" ]; then
    echo "EXTENSION=edge has nothing to test: the edge half of the comparison is the production sweep" >&2
    exit 2
fi

NAME="extension_test_Nz${NZ}_l${SCALE}_${EXTENSION}"
JOB=$(qsub -N "$NAME" \
           -o "logs/${NAME}.log" \
           -e "logs/${NAME}.log" \
           -v NZ=$NZ,SCALE=$SCALE,EXTENSION=$EXTENSION,N_TIME_SKIP=$N_TIME_SKIP \
           extension_test.pbs)
echo "Submitted extension test (Nz=$NZ, l=$SCALE, extension=$EXTENSION): $JOB"
echo "Then run the compare_extension.py line at the end of logs/${NAME}.log; it carries the snapped scale."
