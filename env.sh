#!/bin/bash
# Source this to run the lmcos_small fork standalone:  source env.sh
# Prepending src to PYTHONPATH makes `import cts` / `import analysis` resolve to
# THIS forked copy, overriding the editable-installed `cts` in the venv (verified:
# a PYTHONPATH entry shadows the editable meta-path finder). Both top-level
# packages (`cts`, `analysis`) live under src/, so one entry covers both.
HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH="${HERE}/src${PYTHONPATH:+:$PYTHONPATH}"
[ -f /home/hl4291/venv/bin/activate ] && source /home/hl4291/venv/bin/activate
echo "lmcos_small env: cts -> ${HERE}/src/cts ; venv active"
