#!/bin/bash
# Submit the entire lmcos_small pipeline (packing -> pretrain -> materialize -> readout)
# for all three Stockfish Elo ratings (1800, 2000, 2200) with chained dependencies.
set -euo pipefail

TINY=/home/hl4291/chess_analysis
cd "$TINY"

RUNGS=(1350 2000)
NFILTER=10    # CPU filter array size (few large shards: per-shard work is tiny, array start-up dominates)
NWORKERS=40   # GPU materialize array size (max 40)

READOUT_IDS=()
for ELO in "${RUNGS[@]}"; do
  echo "=== Submitting ELO ${ELO} ==="

  # 1b. filter_trees array (CPU, 1/NFILTER shard per task)
  FILTER_ID=$(sbatch --parsable --export=ALL,ELO=${ELO} --array=0-$((NFILTER-1)) slurm/pipeline/1b_filter_trees.slurm)
  echo "  1b_filter_trees: ${FILTER_ID} (array 0-$((NFILTER-1)))"

  # 1c. pack_trees (CPU): merge shards -> clean_trees.txt, then split/gnn_pack/mc_pack
  PACK_ID=$(sbatch --parsable --export=ALL,ELO=${ELO} --dependency=afterok:${FILTER_ID} slurm/pipeline/1c_pack_trees.slurm)
  echo "  1c_pack_trees: ${PACK_ID}"

  # 2a. train_encoder (GPU)
  ENC_ID=$(sbatch --parsable --export=ALL,ELO=${ELO} --dependency=afterok:${PACK_ID} slurm/pipeline/2a_train_encoder.slurm)
  echo "  2a_train_encoder: ${ENC_ID}"

  # 2b. pack_root (materialize worker array, GPU)
  REPS_ID=$(sbatch --parsable --export=ALL,ELO=${ELO},NWORKERS=${NWORKERS} --array=0-$((NWORKERS-1)) --dependency=afterok:${ENC_ID} slurm/pipeline/2b_pack_root.slurm)
  echo "  2b_pack_root: ${REPS_ID} (array 0-$((NWORKERS-1)))"

  # 2c. merge_root (CPU)
  MERGE_ID=$(sbatch --parsable --export=ALL,ELO=${ELO},NWORKERS=${NWORKERS} --dependency=afterok:${REPS_ID} slurm/pipeline/2c_merge_root.slurm)
  echo "  2c_merge_root: ${MERGE_ID}"

  # 3. train_readout by POLICY GRADIENT (direct E[regret]; CPU). MSE variant is
  #    slurm/pipeline/3_train_readout.slurm if a baseline is wanted.
  READOUT_ID=$(sbatch --parsable --export=ALL,ELO=${ELO} --dependency=afterok:${MERGE_ID} slurm/pipeline/3_train_readout_pg.slurm)
  echo "  3_train_readout_pg: ${READOUT_ID}"
  READOUT_IDS+=("${READOUT_ID}")
done

# 4. eval (CPU): 4-model eval per rung + cross-Elo ladder, afterok on ALL readouts
echo "=== Submitting eval (after all rungs) ==="
RUNGS_CSV=$(IFS=,; echo "${RUNGS[*]}")
READOUT_DEP=$(IFS=:; echo "${READOUT_IDS[*]}")
EVAL_ID=$(sbatch --parsable --export=ALL,RUNGS=${RUNGS_CSV} --dependency=afterok:${READOUT_DEP} slurm/pipeline/4_eval.slurm)
echo "  4_eval: ${EVAL_ID} (afterok:${READOUT_DEP}, rungs=${RUNGS_CSV})"
