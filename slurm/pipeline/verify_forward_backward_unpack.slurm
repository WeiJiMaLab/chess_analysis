#!/bin/bash
# One-off GPU verification (not a pipeline stage) — runs
# _verify_forward_backward_unpack.py: real Lc0 tree generation (cuda backend)
# vs. the retroactive backward-unpack replay, on N real root FENs. See that
# script's own docstring and history.md's frozen-value-fix Progress Log for
# context. Separated from src/cts/tests/ deliberately — needs a GPU, takes
# tens of minutes, not a routine pytest.
#   sbatch slurm/pipeline/verify_forward_backward_unpack.slurm [NUM_FENS]
#SBATCH --job-name=verify-fwd-bwd
#SBATCH --output=/home/hl4291/chess_analysis/slurm/logs/%x_%j.out
#SBATCH --error=/home/hl4291/chess_analysis/slurm/logs/%x_%j.err
#SBATCH --qos=gpu-short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1

set -euo pipefail
TINY=/home/hl4291/chess_analysis
source "$TINY/slurm/helpers/setup_env.sh" gpu

NUM_FENS="${1:-20}"

echo "verify_forward_backward_unpack host=$(hostname) num_fens=$NUM_FENS"; nvidia-smi -L || true; date
python3 "$TINY/slurm/pipeline/_verify_forward_backward_unpack.py" \
  --num-fens "$NUM_FENS" --backend cuda
STATUS=$?
echo "VERIFY DONE exit=$STATUS"; date
exit $STATUS
