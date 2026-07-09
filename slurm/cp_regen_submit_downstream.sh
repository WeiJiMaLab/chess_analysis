#!/bin/bash
# plan.md Agent 3 (cp_regen) — chain argmax_filter -> pack_trees -> train_encoder ->
# pack_root -> pack_root_merge -> train_readout_pg -> eval for the `cp_regen` variant,
# all with VARIANT=cp_regen exported so every step's paths fork under
# ${run_dir}/variants/cp_regen (see config_minply15_maxply75.yaml's `variants.cp_regen`
# block). Reuses the EXISTING, unmodified pipeline scripts (slurm/pipeline/*.slurm) --
# only the submission or gen_trees itself (already run manually at smoke scale via
# slurm/cp_regen_gen_trees.slurm) diverges from submit_all.sh's normal path, because
# gen_trees needs a smoke-scale array, not pipeline.yaml's full 0-159/400K one.
#
# pack_root's array is deliberately smaller than the base run's 0-39 (this corpus is
# ~1/20th the size after the argmax>2 filter) -- purely a resource-consideration choice,
# no code/config change needed (pack_root.slurm derives worker count from
# SLURM_ARRAY_TASK_COUNT automatically).
#
# Usage: GEN_TREES_JOBID=<jobid> bash slurm/cp_regen_submit_downstream.sh
set -euo pipefail
TINY=/home/hl4291/chess_analysis
cd "$TINY"

GEN_DEP="${GEN_TREES_JOBID:?set GEN_TREES_JOBID to the (completed or afterok-pending) gen_trees array jobid}"
PACK_ROOT_ARRAY="${PACK_ROOT_ARRAY:-0-7}"

sb() { sbatch --parsable --export=ALL,VARIANT=cp_regen "$@"; }

AF=$(sb --dependency=afterok:"$GEN_DEP" slurm/pipeline/argmax_filter.slurm)
echo "argmax_filter: $AF (afterok:$GEN_DEP)"

PT=$(sb --dependency=afterok:"$AF" slurm/pipeline/pack_trees.slurm)
echo "pack_trees: $PT (afterok:$AF)"

TE=$(sb --dependency=afterok:"$PT" slurm/pipeline/train_encoder.slurm)
echo "train_encoder: $TE (afterok:$PT)"

PR=$(sb --array="$PACK_ROOT_ARRAY" --dependency=afterok:"$TE" slurm/pipeline/pack_root.slurm)
echo "pack_root: $PR (array=$PACK_ROOT_ARRAY, afterok:$TE)"

PRM=$(sb --dependency=afterok:"$PR" slurm/pipeline/pack_root_merge.slurm)
echo "pack_root_merge: $PRM (afterok:$PR)"

TR=$(sb --dependency=afterok:"$PRM" slurm/pipeline/train_readout_pg.slurm)
echo "train_readout_pg: $TR (afterok:$PRM)"

EV=$(sb --dependency=afterok:"$TR" slurm/pipeline/eval.slurm)
echo "eval: $EV (afterok:$TR)"

echo "CHAIN: gen_trees($GEN_DEP) -> argmax_filter($AF) -> pack_trees($PT) -> train_encoder($TE) -> pack_root($PR) -> pack_root_merge($PRM) -> train_readout_pg($TR) -> eval($EV)"
