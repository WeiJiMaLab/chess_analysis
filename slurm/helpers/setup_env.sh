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

export PYTHONPATH="/home/hl4291/chess_analysis/src:/home/hl4291/chess_analysis/src/analysis"
export PYTHONUNBUFFERED=1

# Validate ELO is set
: "${ELO:?ELO required (1800|2000|2200)}"

export TINY=/home/hl4291/chess_analysis
export CONFIG=$TINY/configs/core.yaml
export RENDER="python $TINY/configs/render_stage.py $CONFIG --set globals.sf_elo=${ELO}"
export WORK=$($RENDER --get globals.config_dir)
export MCP=$($RENDER --get globals.mc_packed_dir)
export MAT=$($RENDER --get globals.materialized_dir)
export LOGS=$($RENDER --get globals.log_dir)
export ENC=$($RENDER --get encoder.output_checkpoint)

mkdir -p "$TINY/slurm/logs" "$WORK" "$MAT" "$LOGS"
