#!/bin/bash
# Source this to run the lmcos_tiny fork standalone:  source env.sh
# Prepending lmcos_tiny/src to PYTHONPATH makes `import cts` resolve to THIS
# forked copy, overriding the editable-installed `cts` in the venv (verified:
# a PYTHONPATH entry shadows the editable meta-path finder).
HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH="${HERE}/src${PYTHONPATH:+:$PYTHONPATH}"
[ -f /home/hl4291/venv/bin/activate ] && source /home/hl4291/venv/bin/activate
echo "lmcos_tiny env: cts -> ${HERE}/src/cts ; venv active"
