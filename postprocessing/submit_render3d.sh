#!/usr/bin/env bash
# Submit the 3D isosurface rendering job: plot7_3d_volume.jl at one or more output times.
# Usage: bash submit_render3d.sh TIMES=100,120,140 [NZ=1024] [FILE=...] [FIELDS=...] [SCALE=1] [LEVELS=6] [SAMPLES=800] [ISORANGE=0.02] [ZLIM=4] [OVERLAY=b]
#    or: bash submit_render3d.sh FROM=60 TO=160 EVERY=10 [...]
#    or: bash submit_render3d.sh PREVIEW=1 [...]   (a quick test of an animation)
#    or: bash submit_render3d.sh ANIMATE=1 [FROM=10] [TO=...] [EVERY=...] [RANGE_STAT=max] [FPS=10] [RANGES=...] [...]
#   TIMES    output times to render, comma- or space-separated (quote if spaces). One figure per time
#   FROM/TO/EVERY  a strided range instead of a list: FROM to TO inclusive in steps of EVERY, which has no
#            default (the wrapper does not open the run to find its output interval). Exclusive with TIMES
#   NZ       vertical resolution, used to name the job and find the run
#   FILE     the 3D file to read; default $KHAPE_OUTPUT_DIR/khi_Nz<NZ>_Ri0.10.nc. Give it to render a run
#            that lives somewhere else, e.g. a colleague's scratch
#   COLS     panels per row (default 2: the six budget terms in three rows). An eight-field figure in two rows wants COLS=4
#   FIELDS   comma-separated panels; default is the six-term budget set. A "-" prefix negates a field
#            (a term entering two budgets with opposite sign) and a lone "." leaves a cell empty
#   SCALE    filter scale ℓ whose per-scale terms to draw (the _ℓ<ℓ> suffix the bare names pick up)
#   LEVELS   isosurfaces per panel; 2 is much faster to a first look, 6 shows the nested structure
#   FRACTIONS  place the surfaces explicitly instead, as fractions of each panel's range, e.g.
#            0.1,0.5,0.85. Alternative to LEVELS, not a companion -- setting both is an error
#   SAMPLES  ray-march steps per ray for the isosurfaces (default 800; Makie's own is 200). 800 sharpens the edges
#            of small surfaces and costs little: at Nz=1024 a six-panel figure takes ~2 min, against ~90 s at 200. The figure name does
#            not record it, so a render at another SAMPLES replaces the last one at that time
#   ISORANGE shell half-width of each isosurface, as a fraction of the panel range; default 0.02, and
#            auto for Makie's own, a tenth of the smallest gap between levels (~0.034 at LEVELS=6). Not a
#            smoothing knob: the visible surface sits at level ± ISORANGE, so a wider shell draws it further
#            out and weak features near a level appear or vanish with it. Not in the figure name either
#   ZLIM     crop to |z| < ZLIM
#   DERIVED  velocity the derived fields (Q, enstrophy, speed) are built from: 'filtered' (default, the
#            online field at SCALE) or 'full' (the raw one)
#   OVERLAY  reference field drawn as contours on the bounding walls ('' for none)
#   ANIMATE  1 for an animation instead of figures: every record from FROM (default 10) to TO (default the
#            last), or every EVERY in time, with each panel on one scale fixed over those records, so a surface
#            means the same value in every frame. The job measures every panel's scale first, then renders the
#            frames into animations/frames/<name>/ and joins them into animations/<name>.mp4. TIMES is refused
#   RANGE_STAT  how a panel's fixed scale comes from its scale at each record: max (default), or pNN for a
#            percentile over time (p90 lets the peak frames saturate a little for more contrast elsewhere)
#   RANGE_WINDOW  FROM:TO, either end optional (e.g. 90:): take each panel's fixed scale from these times
#            only, while still animating the whole window. A later phase can then set the scale, at the cost
#            of saturating the frames outside it, where a term exceeds that scale. Default: every frame
#   FPS      frames per second of the movie (default 10)
#   PREVIEW  1 for a quick test of an animation: ANIMATE=1 with one frame every 20 time units (EVERY=20) at
#            half the resolution (PX_PER_UNIT=1), under names of its own (<name>_preview.mp4 and its own
#            ranges file and frames), so it never replaces the full animation. Levels and samples are as
#            usual; any option given explicitly overrides the preview's
#   PX_PER_UNIT  pixels per figure unit (default 2; the preview's 1 has a quarter of the pixels)
#   RANGES   a ranges file from an earlier animation job (animations/<name>_ranges.toml) to reuse instead of
#            measuring again, e.g. to re-render at another FPS or RANGE_STAT
#
# No GPU is requested: the job makes its own OpenGL context with Xvfb and Mesa's software rasteriser.
# See 04_render3d.pbs.
TIMES=""
ANIMATE=0
PREVIEW=0
PX_PER_UNIT=""
RANGE_STAT=max
RANGE_WINDOW=""
FPS=10
RANGES=""
FROM=""
TO=""
EVERY=""
NZ=1024
FILE=""
FIELDS="Q,wb_rs,Π_K,Π_A,ε_Ks,ε_As"
COLS=2
SCALE=1
LEVELS=6
FRACTIONS=""
SAMPLES=800
ISORANGE=0.02
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
  SAMPLES=*) SAMPLES="${arg#*=}";;
  ISORANGE=*) ISORANGE="${arg#*=}";;
  ZLIM=*)    ZLIM="${arg#*=}";;
  DERIVED=*) DERIVED="${arg#*=}";;
  OVERLAY=*) OVERLAY="${arg#*=}";;
  ANIMATE=*) ANIMATE="${arg#*=}";;
  PREVIEW=*) PREVIEW="${arg#*=}";;
  PX_PER_UNIT=*) PX_PER_UNIT="${arg#*=}";;
  RANGE_STAT=*) RANGE_STAT="${arg#*=}";;
  RANGE_WINDOW=*) RANGE_WINDOW="${arg#*=}";;
  FPS=*)     FPS="${arg#*=}";;
  RANGES=*)  RANGES="${arg#*=}";;
  *) echo "unknown argument: $arg (expected TIMES=, FROM=, TO=, EVERY=, NZ=, FILE=, FIELDS=, COLS=, SCALE=, LEVELS=, FRACTIONS=, SAMPLES=, ISORANGE=, ZLIM=, DERIVED=, OVERLAY=, ANIMATE=, RANGE_STAT=, RANGE_WINDOW=, FPS=, RANGES=, PREVIEW= or PX_PER_UNIT=)" >&2; exit 2;;
esac; done
# LEVELS is a count, FRACTIONS a list. Passing one to the other fails deep inside the render, where the
# time is logged as a warning and skipped -- and since the figure's name encodes neither, the previous
# figure is left sitting there looking like a result.
case "$LEVELS" in (*[!0-9]*|"")
    echo "error: LEVELS=$LEVELS must be a whole number of surfaces; to place them explicitly use FRACTIONS=0.4,0.85" >&2
    exit 2;;
esac
case "$SAMPLES" in (*[!0-9]*|"")
    echo "error: SAMPLES=$SAMPLES must be a whole number of steps per ray, e.g. 800" >&2
    exit 2;;
esac
case "$ISORANGE" in (auto) ;; (*[!0-9.]*|"")
    echo "error: ISORANGE=$ISORANGE must be a positive number, e.g. 0.02, or auto for Makie's own" >&2
    exit 2;;
esac
case "$FRACTIONS" in (*[!0-9.,]*)
    echo "error: FRACTIONS=$FRACTIONS must be a comma-separated list of numbers, e.g. 0.4,0.85" >&2
    exit 2;;
esac
case "$PREVIEW" in (0|1) ;; (*) echo "error: PREVIEW=$PREVIEW must be 0 or 1" >&2; exit 2;; esac
# A preview is an animation with fewer, smaller frames; what it changes is only a default, so an EVERY or a
# PX_PER_UNIT given explicitly still wins.
if [ "$PREVIEW" = 1 ]; then
    ANIMATE=1
    EVERY=${EVERY:-20}
    PX_PER_UNIT=${PX_PER_UNIT:-1}
fi
PX_PER_UNIT=${PX_PER_UNIT:-2}
case "$PX_PER_UNIT" in (*[!0-9.]*|"") echo "error: PX_PER_UNIT=$PX_PER_UNIT must be a positive number" >&2; exit 2;; esac
case "$ANIMATE" in (0|1) ;; (*) echo "error: ANIMATE=$ANIMATE must be 0 or 1" >&2; exit 2;; esac
if [ "$ANIMATE" = 1 ]; then
    [ -z "$TIMES" ] || { echo "error: an animation takes FROM/TO/EVERY, not TIMES" >&2; exit 2; }
    FROM=${FROM:-10}
    case "$RANGE_STAT" in (max|p[0-9]*) ;; (*) echo "error: RANGE_STAT=$RANGE_STAT must be max or pNN, e.g. p90" >&2; exit 2;; esac
    # Two numbers, either of them optional, around one colon.
    [ -z "$RANGE_WINDOW" ] || [[ "$RANGE_WINDOW" =~ ^[0-9.]*:[0-9.]*$ ]] ||
        { echo "error: RANGE_WINDOW=$RANGE_WINDOW must be FROM:TO, either end optional, e.g. 90:200 or 90:" >&2; exit 2; }
    case "$FPS" in (*[!0-9]*|""|0) echo "error: FPS=$FPS must be a positive whole number" >&2; exit 2;; esac
    if [ -n "$RANGES" ]; then
        # The job resolves a relative path from postprocessing/, where this is run, so make it absolute here.
        [ -f "$RANGES" ] || { echo "error: RANGES=$RANGES does not exist" >&2; exit 2; }
        RANGES=$(cd "$(dirname "$RANGES")" && pwd)/$(basename "$RANGES")
    fi
fi
# A list or a range, not both: silently preferring one would hide a typo in the other.
if [ -n "$TIMES" ] && { [ -n "$FROM" ] || [ -n "$TO" ]; }; then
    echo "error: give TIMES= or FROM=/TO=, not both" >&2; exit 2
fi
if [ -z "$TIMES" ] && [ "$ANIMATE" = 0 ]; then
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
           -v NZ=$NZ,TIMES=$TIMES_COLON,FIELDS=$FIELDS_COLON,COLS=$COLS,SCALE=$SCALE,LEVELS=$LEVELS,FRACTIONS=$FRACS_COLON,SAMPLES=$SAMPLES,ISORANGE=$ISORANGE,ZLIM=$ZLIM,DERIVED=$DERIVED,OVERLAY=$OVERLAY,FILE=$FILE,ANIMATE=$ANIMATE,FROM=$FROM,TO=$TO,EVERY=$EVERY,RANGE_STAT=$RANGE_STAT,RANGE_WINDOW=$RANGE_WINDOW,FPS=$FPS,RANGES=$RANGES,PREVIEW=$PREVIEW,PX_PER_UNIT=$PX_PER_UNIT$KHAPE_REDIRECT \
           04_render3d.pbs)
if [ "$ANIMATE" = 1 ]; then
    echo "Submitted 3D $([ "$PREVIEW" = 1 ] && echo "animation preview" || echo animation) (Nz=$NZ, t=$FROM to ${TO:-end} every ${EVERY:-record}, range_stat=$RANGE_STAT, range_window=${RANGE_WINDOW:-all}, fps=$FPS, px_per_unit=$PX_PER_UNIT): $JOB"
else
    echo "Submitted 3D render (Nz=$NZ, times=$(echo "$TIMES_COLON" | tr ':' ' '), levels=$LEVELS, samples=$SAMPLES, isorange=$ISORANGE): $JOB"
fi
