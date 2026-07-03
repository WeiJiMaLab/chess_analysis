#!/bin/bash
# Analysis pipeline step 2 — engine-level (no-normative) RT analysis on the SF-1
# trees from step 1b. Submit as a dependency of 1b (afterok).
#   sbatch --dependency=afterok:<gen_trees_arrayjob> slurm/pipeline/engine_analysis.slurm
#
# CPU partition by design: NO engine inference here. The Stockfish evals are
# already baked into the .pt trees (value heads + oracle traces); this job only
# reads them (torch.load on CPU), derives the SF-1 signals (Gain / MQ / greedy
# action gap / GSS / greedy frac-good / OSS), joins to human RT in DuckDB, and
# plots. Per-tree values are cached to parquet so re-plots need no trees.
#   N_TREES   (default 100000)  N_WORKERS (default = cpus-per-task)  REFRESH=1 forces recompute
# Trees read from human_analysis.trees_default (befs1cp_md36); figures written
# under human_analysis.figures_dir/engine_<unit> (UNIT env: pwin|cp; default pwin).
# The cp twin (engine_analysis_cp.slurm) sets UNIT=cp and reuses this body.
#SBATCH --job-name=engine-pwin
#SBATCH --output=/home/hl4291/chess_analysis/slurm/logs/%x_%j.out
#SBATCH --error=/home/hl4291/chess_analysis/slurm/logs/%x_%j.err
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=02:00:00

set -euo pipefail
TINY=/home/hl4291/chess_analysis
cd "${TINY}"

source "${TINY}/slurm/helpers/setup_env.sh"

N_TREES="${N_TREES:-100000}"
N_WORKERS="${N_WORKERS:-${SLURM_CPUS_PER_TASK:-16}}"
# REFRESH forces a clean recompute of the per-tree parquet cache (default off).
REFRESH_FLAG="${REFRESH:+--refresh}"
UNIT="${UNIT:-pwin}"

echo "engine analysis: unit=${UNIT} n_trees=${N_TREES} n_workers=${N_WORKERS} on $(hostname) at $(date)"
python3 src/analysis/engine.py \
    --mode all \
    --unit "${UNIT}" \
    --n-trees "${N_TREES}" \
    --n-workers "${N_WORKERS}" \
    ${REFRESH_FLAG}
echo "Finished at $(date)"
