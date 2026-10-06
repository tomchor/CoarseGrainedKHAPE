#!/usr/bin/env bash
# Submit the 3D isosurface rendering job: X13_3d_volume.jl at one or more output times.
# Usage: bash submit_render3d.sh TIMES=100,120,140 [NZ=1024] [FILE=...] [FIELDS=...] [SCALE=1] [LEVELS=6] [ZLIM=4] [OVERLAY=b]
#    or: bash submit_render3d.sh FROM=60 TO=160 EVERY=10 [...]
#   TIMES    output times to render, comma- or space-separated (quote if spaces). One figure per time
#   FROM/TO/EVERY  a strided range instead of a list: FROM to TO inclusive in steps of EVERY, which has no
#            default (the wrapper does not open the run to find its output interval). Exclusive with TIMES
#   NZ       vertical resolution, used to name the job and find the run
#   FILE     the 3D file to read; default $KHAPE_OUTPUT_DIR/khi_Nz<NZ>_Ri0.10.nc. Give it to render a run
#            that lives somewhere else, e.g. a colleague's scratch
#   COLS     panels per row (X13 default 3). A two-row budget figure of eight fields wants COLS=4
#   FIELDS   comma-separated panels; default is the six-term budget set. A "-" prefix negates a field
#            (a term entering two budgets with opposite sign) and a lone "." leaves a cell empty
#   SCALE    filter scale ℓ whose per-scale terms to draw (the _ℓ<ℓ> suffix the bare names pick up)
#   LEVELS   isosurfaces per panel; 2 is much faster to a first look, 6 shows the nested structure
#   FRACTIONS  place the surfaces explicitly instead, as fractions of each panel's range, e.g.
#            0.1,0.5,0.85. Alternative to LEVELS, not a companion -- setting both is an error
#   ZLIM     crop to |z| < ZLIM
#   DERIVED  velocity the derived fields (Q, enstrophy, speed) are built from: 'filtered' (default, the
#            online field at SCALE) or 'full' (the raw one)
#   OVERLAY  reference field drawn as contours on the bounding walls ('' for none)
#
# No GPU is requested: the job makes its own OpenGL context with Xvfb and Mesa's software rasteriser.
# See render3d.pbs.
TIMES=""
FROM=""
TO=""
EVERY=""
NZ=1024
FILE=""
FIELDS="Q,wb_rs,Π_K,Π_A,ε_Ks,ε_As"
COLS=3
SCALE=1
LEVELS=6
FRACTIONS=""
ZLIM=4
DERIVED=filtered
OVERLAY=b
# An unknown KEY=VALUE is refused rather than ignored, so a misspelled flag cannot silently fall back to its default.
for arg in "$@"; do case $arg in
  TIMES=*)   TIMES="${arg#*=}";;
  FROM=*)    FROM="${arg#*=}";;
  TO=*)      TO="${arg#*=}";;
  EVERY=*)   EVERY="${arg#*=}";;
  NZ=*)      NZ="${arg#*=}";;
  FILE=*)    FILE="${arg#*=}";;
  FIELDS=*)  FIELDS="${arg#*=}";;
  COLS=*)    COLS="${arg#*=}";;
  SCALE=*)   SCALE="${arg#*=}";;
  LEVELS=*)  LEVELS="${arg#*=}";;
  FRACTIONS=*) FRACTIONS="${arg#*=}";;
  ZLIM=*)    ZLIM="${arg#*=}";;
  DERIVED=*) DERIVED="${arg#*=}";;
  OVERLAY=*) OVERLAY="${arg#*=}";;
  *) echo "unknown argument: $arg (expected TIMES=, FROM=, TO=, EVERY=, NZ=, FILE=, FIELDS=, COLS=, SCALE=, LEVELS=, FRACTIONS=, ZLIM=, DERIVED= or OVERLAY=)" >&2; exit 2;;
esac; done
# A list or a range, not both: silently preferring one would hide a typo in the other.
if [ -n "$TIMES" ] && { [ -n "$FROM" ] || [ -n "$TO" ]; }; then
    echo "error: give TIMES= or FROM=/TO=, not both" >&2; exit 2
fi
if [ -z "$TIMES" ]; then
    { [ -n "$FROM" ] && [ -n "$TO" ] && [ -n "$EVERY" ]; } ||
        { echo "error: give TIMES=100,120,140, or all of FROM=60 TO=160 EVERY=10" >&2; exit 2; }
    # awk, not seq: the times are floats and seq's output format depends on the locale.
    TIMES=$(awk -v a="$FROM" -v b="$TO" -v s="$EVERY" 'BEGIN {
        if (s <= 0) { print "BADSTEP"; exit }
        if (b < a)  { print "EMPTY"; exit }
        n = 0
        for (t = a; t <= b + 1e-9; t += s) { printf "%s%g", (n++ ? "," : ""), t }
        printf "\n"
    }')
    [ "$TIMES" != "BADSTEP" ] || { echo "error: EVERY=$EVERY must be positive" >&2; exit 2; }
    [ "$TIMES" != "EMPTY" ]   || { echo "error: TO=$TO is before FROM=$FROM" >&2; exit 2; }
fi
# The allocation to charge: default in khape_defaults.sh, overridden from the environment. No Python here --
# this job is Julia only -- but KHAPE_OUTPUT_DIR still has to reach it, which $KHAPE_REDIRECT carries.
source "$(dirname "$0")/../khape_defaults.sh"

# `qsub -v` splits its own list on commas, so the times travel colon-separated. Spaces are accepted on this
# side for convenience and normalised here.
TIMES_COLON=$(echo "$TIMES"   | tr ', ' '::' | tr -s ':')
FIELDS_COLON=$(echo "$FIELDS" | tr ','   ':')   # same reason: the panel list is comma-separated too
FRACS_COLON=$(echo "$FRACTIONS" | tr ',' ':')   # and the fraction list
NAME="render3d_Nz${NZ}_Ri0.10"
JOB=$(qsub -N "$NAME" \
           -A "$KHAPE_ACCOUNT" \
           -o "logs/${NAME}.log" \
           -e "logs/${NAME}.log" \
           -v NZ=$NZ,TIMES=$TIMES_COLON,FIELDS=$FIELDS_COLON,COLS=$COLS,SCALE=$SCALE,LEVELS=$LEVELS,FRACTIONS=$FRACS_COLON,ZLIM=$ZLIM,DERIVED=$DERIVED,OVERLAY=$OVERLAY,FILE=$FILE$KHAPE_REDIRECT \
           render3d.pbs)
echo "Submitted 3D render (Nz=$NZ, times=$(echo "$TIMES_COLON" | tr ':' ' '), levels=$LEVELS): $JOB"
