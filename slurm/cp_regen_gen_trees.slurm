#!/bin/bash
# plan.md Agent 3 ("CP done for real") — smoke-scale tree generation under the
# `cp_regen` config variant (config_minply15_maxply75.yaml, `variants.cp_regen`):
# value_feature=tanh_cp_value (tanh(cp_order/300) instead of the WDL "value"
# column, computed at GENERATION time so PUCT's own selection can respond to it,
# unlike CP's shape-frozen relabel-and-replay probe) PLUS prune_mode=relative/
# prune_epsilon=0.1 (T3-validated pruning knobs) baked in from the start.
#
# Deliberately a NEW, standalone script rather than reusing gen_trees.slurm:
# (a) VARIANT=cp_regen must be exported for every task so render_stage/
#     cts._config's apply_variant forks output paths + applies the variant's
#     overrides (see config's `variants.cp_regen` block for the exact fields);
#     (b) the shard COUNT here is deliberately smoke-scale (order of magnitude
#     similar to CP's ~16K-raw-tree sample), not gen_trees.slurm's full
#     0-159/400K production array — this is NOT a partial/straggler re-run of
#     the same shape, it is a genuinely smaller corpus by design (plan.md:
#     "smoke-scale first ... NOT the full ~400K production corpus").
#
# Reuses the base run's fens_sample.txt (config `share: [fens_sample]` — same
# root FEN pool as the WDL-baseline corpus, sliced to the first
# SHARD_SIZE * (task_count) FENs), so shape comparisons against the baseline
# aren't confounded by a different root population.
#
#   sbatch --array=0-6 slurm/cp_regen_gen_trees.slurm   # 7 * 2500 = 17,500 raw trees
#SBATCH --job-name=cp-regen-gen-trees
#SBATCH --output=/home/hl4291/chess_analysis/slurm/logs/%x_%A_%a.out
#SBATCH --error=/home/hl4291/chess_analysis/slurm/logs/%x_%A_%a.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00

set -euo pipefail
export VARIANT=cp_regen
TINY=/home/hl4291/chess_analysis
source "$TINY/slurm/helpers/setup_env.sh"

SHARD_SIZE=2500
start=$(( SLURM_ARRAY_TASK_ID * SHARD_SIZE ))
end=$(( start + SHARD_SIZE ))
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# Sourced entirely from config_minply15_maxply75.yaml's `variants.cp_regen` block
# via VARIANT=cp_regen above (value_feature, tanh_cp_temperature, prune_mode,
# prune_epsilon, output_dir all already resolved there) -- only the shard slice
# needs to be passed explicitly here.
OUTDIR=$($RENDER --get treegen.output_dir)
echo "CP-REGEN GEN-TREES task=${SLURM_ARRAY_TASK_ID} start=${start} end=${end} out=${OUTDIR} host=$(hostname)"; date
python -m cts.data.build_tree --config "$CONFIG" --stage treegen \
    --override start_index=${start} --override end_index=${end}
echo "CP-REGEN GEN-TREES DONE start=${start}"; date
