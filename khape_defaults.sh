# Defaults for the submit wrappers: the project to charge and the python of the py313 environment. Every
# submit_*.sh sources this file, so this is the one place to change either; a value already set in the
# environment (export KHAPE_ACCOUNT=..., KHAPE_PYTHON=...) wins. The .pbs files do not read this file: the
# wrapper hands them the values (qsub -A and -v). See the README, Environment.
KHAPE_ACCOUNT=${KHAPE_ACCOUNT:-UMCP0061}
KHAPE_PYTHON=${KHAPE_PYTHON:-$HOME/miniconda3/envs/py313/bin/python}

# Optional output redirection: $KHAPE_OUTPUT_DIR (simulation output) and $KHAPE_PP_OUTPUT (post-processing output),
# when set, are handed to every job (each wrapper appends $KHAPE_REDIRECT to its qsub -v), so a run can write beside
# the production output instead of over it, and a single stage can be rerun on a redirected run. The PBS scripts,
# kelvin_helmholtz_instability.jl and src/aux00_utils.py all read them; give absolute paths.
KHAPE_REDIRECT=""
[ -n "${KHAPE_OUTPUT_DIR:-}" ] && KHAPE_REDIRECT="$KHAPE_REDIRECT,KHAPE_OUTPUT_DIR=$KHAPE_OUTPUT_DIR"
[ -n "${KHAPE_PP_OUTPUT:-}" ]  && KHAPE_REDIRECT="$KHAPE_REDIRECT,KHAPE_PP_OUTPUT=$KHAPE_PP_OUTPUT"

# For the wrappers that submit Python jobs: fail at submission, not in the queue, when the interpreter is not there.
check_khape_python() {
    [ -x "$KHAPE_PYTHON" ] && return 0
    echo "error: KHAPE_PYTHON=$KHAPE_PYTHON is not an executable; set it to your py313 environment's python (see the README, Environment)" >&2
    exit 2
}
