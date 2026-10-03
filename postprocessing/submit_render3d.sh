#!/usr/bin/env bash
# Submit the 3D isosurface rendering job: X13_3d_volume.jl at one or more output times.
# Usage: bash submit_render3d.sh TIMES=100,120,140 [NZ=1024] [FILE=...] [FIELDS=...] [SCALE=1] [LEVELS=6] [ZLIM=4] [OVERLAY=b]
#   TIMES    output times to render, comma- or space-separated (quote if spaces). One figure per time
#   NZ       vertical resolution, used to name the job and find the run
#   FILE     the 3D file to read; default $KHAPE_OUTPUT_DIR/khi_Nz<NZ>_Ri0.10.nc. Give it to render a run
#            that lives somewhere else, e.g. a colleague's scratch
#   FIELDS   comma-separated panels; default is the six-term budget set
#   SCALE    filter scale ℓ whose per-scale terms to draw (the _ℓ<ℓ> suffix the bare names pick up)
#   LEVELS   isosurfaces per panel; 2 is much faster to a first look, 6 shows the nested structure
#   ZLIM     crop to |z| < ZLIM
#   OVERLAY  reference field drawn as contours on the bounding walls ('' for none)
#
# No GPU is requested: the job makes its own OpenGL context with Xvfb and Mesa's software rasteriser.
# See render3d.pbs.
TIMES=""
NZ=1024
FILE=""
FIELDS="Q,wb_rs,Π_K,Π_A,ε_Ks,ε_As"
SCALE=1
LEVELS=6
ZLIM=4
OVERLAY=b
# An unknown KEY=VALUE is refused rather than ignored, so a misspelled flag cannot silently fall back to its default.
for arg in "$@"; do case $arg in
  TIMES=*)   TIMES="${arg#*=}";;
  NZ=*)      NZ="${arg#*=}";;
  FILE=*)    FILE="${arg#*=}";;
  FIELDS=*)  FIELDS="${arg#*=}";;
  SCALE=*)   SCALE="${arg#*=}";;
  LEVELS=*)  LEVELS="${arg#*=}";;
  ZLIM=*)    ZLIM="${arg#*=}";;
  OVERLAY=*) OVERLAY="${arg#*=}";;
  *) echo "unknown argument: $arg (expected TIMES=, NZ=, FILE=, FIELDS=, SCALE=, LEVELS=, ZLIM= or OVERLAY=)" >&2; exit 2;;
esac; done
[ -n "$TIMES" ] || { echo "error: TIMES is required, e.g. TIMES=100,120,140" >&2; exit 2; }
# The allocation to charge: default in khape_defaults.sh, overridden from the environment. No Python here --
# this job is Julia only -- but KHAPE_OUTPUT_DIR still has to reach it, which $KHAPE_REDIRECT carries.
source "$(dirname "$0")/../khape_defaults.sh"

# `qsub -v` splits its own list on commas, so the times travel colon-separated. Spaces are accepted on this
# side for convenience and normalised here.
TIMES_COLON=$(echo "$TIMES"   | tr ', ' '::' | tr -s ':')
FIELDS_COLON=$(echo "$FIELDS" | tr ','   ':')   # same reason: the panel list is comma-separated too
NAME="render3d_Nz${NZ}_Ri0.10"
JOB=$(qsub -N "$NAME" \
           -A "$KHAPE_ACCOUNT" \
           -o "logs/${NAME}.log" \
           -e "logs/${NAME}.log" \
           -v NZ=$NZ,TIMES=$TIMES_COLON,FIELDS=$FIELDS_COLON,SCALE=$SCALE,LEVELS=$LEVELS,ZLIM=$ZLIM,OVERLAY=$OVERLAY,FILE=$FILE$KHAPE_REDIRECT \
           render3d.pbs)
echo "Submitted 3D render (Nz=$NZ, times=$(echo "$TIMES_COLON" | tr ':' ' '), levels=$LEVELS): $JOB"
