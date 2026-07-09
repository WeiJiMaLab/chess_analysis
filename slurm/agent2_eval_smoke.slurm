#!/bin/bash
# plan.md Phase 2, Agent 2 -- smoke test for analysis.evaluate_our_trees_continued (mode=pg)
# against the REAL production checkpoint, small --max-episodes, before spending real compute
# on the full grid. Submitted to SLURM rather than the login-node shell: an earlier attempt on
# the login node was silently killed (0-byte log, no traceback) with no error -- the SAME
# resource-governor-kill failure mode sig_z_eval.slurm's comments already documented for a
# different script; this repo's convention is to always route real-data runs through SLURM.
#
#   sbatch slurm/agent2_eval_smoke.slurm
#SBATCH --job-name=agent2-eval-smoke
#SBATCH --output=/home/hl4291/chess_analysis/slurm/logs/%x_%j.out
#SBATCH --error=/home/hl4291/chess_analysis/slurm/logs/%x_%j.err
#SBATCH --qos=test
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00

set -euo pipefail
TINY=/home/hl4291/chess_analysis
export CONFIG="$TINY/config_minply15_maxply75.yaml"
source "$TINY/slurm/helpers/setup_env.sh"
export OMP_NUM_THREADS=4

OUT_DIR=/scratch/gpfs/GRIFFITHS/hl4291/lmcos/minply15_maxply75/diagnosis/agent2_eval_smoke
mkdir -p "$OUT_DIR"

echo "agent2_eval_smoke host=$(hostname)"; date
python -u -m analysis.evaluate_our_trees_continued \
  --mode pg \
  --packed-root "$MCP" \
  --cache "$MAT/validation_cache.pt" \
  --config "$CONFIG" \
  --checkpoints /scratch/gpfs/GRIFFITHS/hl4291/lmcos/minply15_maxply75/packed/mchalt_controller.pt \
  --epochs 20 \
  --regimes 0.01:0.0 \
  --max-episodes 200 \
  --out-dir "$OUT_DIR"

echo "AGENT2 EVAL SMOKE DONE"; date
