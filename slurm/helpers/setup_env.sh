#!/bin/bash
# Shared environment setup script for lmcos_small SLURM jobs.
# Usage: source slurm/helpers/setup_env.sh [gpu]

set +u
module --force purge >/dev/null 2>&1 || true
module load anaconda3/2023.3

if [ "${1:-}" = "gpu" ]; then
    module load cudatoolkit/12.8
fi

VENV_DIR=/home/hl4291/venv
source "$VENV_DIR/bin/activate"
set -u

export PYTHONPATH="/home/hl4291/chess_analysis/src"
export PYTHONUNBUFFERED=1

export TINY=/home/hl4291/chess_analysis
# Active config = whichever run you're driving. Override by exporting CONFIG
# before sourcing (e.g. CONFIG=$TINY/config_minply10_maxply80.yaml); defaults to
# the canonical run.
export CONFIG="${CONFIG:-$TINY/config_allply.yaml}"
export RENDER="python $TINY/render_stage.py $CONFIG"
export WORK=$($RENDER --get globals.config_dir)
export MCP=$($RENDER --get globals.mc_packed_dir)
export MAT=$($RENDER --get globals.materialized_dir)
export LOGS=$($RENDER --get globals.log_dir)
export ENC=$($RENDER --get encoder.output_checkpoint)

mkdir -p "$TINY/slurm/logs" "$WORK" "$MAT" "$LOGS"
