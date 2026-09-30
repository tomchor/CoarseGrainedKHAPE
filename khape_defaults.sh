# Defaults for the submit wrappers: the project to charge and the python of the py313 environment. Every
# submit_*.sh sources this file, so this is the one place to change either; a value already set in the
# environment (export KHAPE_ACCOUNT=..., KHAPE_PYTHON=...) wins. The .pbs files do not read this file: the
# wrapper hands them the values (qsub -A and -v). See the README, Environment.
KHAPE_ACCOUNT=${KHAPE_ACCOUNT:-UMCP0061}
KHAPE_PYTHON=${KHAPE_PYTHON:-$HOME/miniconda3/envs/py313/bin/python}

# For the wrappers that submit Python jobs: fail at submission, not in the queue, when the interpreter is not there.
check_khape_python() {
    [ -x "$KHAPE_PYTHON" ] && return 0
    echo "error: KHAPE_PYTHON=$KHAPE_PYTHON is not an executable; set it to your py313 environment's python (see the README, Environment)" >&2
    exit 2
}
