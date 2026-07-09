#!/bin/bash
# plan.md Phase 2, Agent 2, sub-line A -- materialize z_t for EACH of the 3 continued epoch
# checkpoints written by slurm/agent2_e2e_continue_train.slurm (each epoch has its own encoder,
# so each needs its own materialized validation cache -- same requirement Z3 already established
# for the original 1-epoch Z2 checkpoint), then run the paired-significance eval
# (analysis.evaluate_our_trees_continued, mode=e2e) at 3 REGIME-confirmed cost regimes per epoch.
#
# Depends on slurm/agent2_e2e_continue_train.slurm having produced
# agent2_e2e_continued/e2e_controller_continued_epoch{001,002,003}_encoder.pt. Submit with:
#   sbatch --dependency=afterok:<agent2_e2e_continue_train_jobid> \
#       slurm/agent2_e2e_continue_materialize_and_eval.slurm
#
# Writes to a SEPARATE, sub-line-specific figures dir (outputs/figures/.../normative_e2e_continued/)
# -- per direct instruction, NOT shared with sub-line B's normative_pg_continued/ dir (filename
# collision risk) and NOT the original z3_e2e_comparison/ dir SIG-Z already owns.
#
#SBATCH --job-name=agent2-e2e-materialize-eval
#SBATCH --output=/home/hl4291/chess_analysis/slurm/logs/%x_%j.out
#SBATCH --error=/home/hl4291/chess_analysis/slurm/logs/%x_%j.err
#SBATCH --qos=gpu-short
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:30:00
#SBATCH --gres=gpu:1

set -euo pipefail
TINY=/home/hl4291/chess_analysis
export CONFIG="$TINY/config_minply15_maxply75.yaml"
source "$TINY/slurm/helpers/setup_env.sh" gpu
export OMP_NUM_THREADS=4

E2E_DIR=/scratch/gpfs/GRIFFITHS/hl4291/lmcos/minply15_maxply75/diagnosis/agent2_e2e_continued
OUT_DIR=/home/hl4291/chess_analysis/outputs/figures/minply15_maxply75/normative_e2e_continued
mkdir -p "$OUT_DIR"

echo "agent2_e2e_materialize_eval host=$(hostname)"; nvidia-smi -L || true; date

CACHES=""
EPOCHS=""
for N in 001 002 003; do
    ENC="$E2E_DIR/e2e_controller_continued_epoch${N}_encoder.pt"
    if [ ! -f "$ENC" ]; then
        echo "WARNING: $ENC missing, skipping epoch $N (continue-train job may have run fewer epochs)"
        continue
    fi
    WORK="$E2E_DIR/materialized_validation_work_epoch${N}"
    CACHE="$E2E_DIR/materialized_validation_cache_epoch${N}.pt"

    echo "=== materialize epoch $N (encoder=$ENC) ==="
    python -u -m cts.data.preprocess_mc.materialize --config "$CONFIG" --stage materialize \
        --set command=materialize --set device=cuda \
        --set num_workers=1 --set worker_index=0 \
        --set output_dir="$WORK" \
        --set packed_data="$MCP/validation_manifest.json" \
        --set encoder_checkpoint="$ENC" \
        --set loader_workers=0 --set resume=false

    echo "=== merge epoch $N ==="
    python -u -m cts.data.preprocess_mc.materialize --config "$CONFIG" --stage materialize \
        --set command=merge --set num_workers=1 \
        --set output_dir="$WORK" \
        --set final_cache="$CACHE" \
        --set manifest_path="$MCP/validation_manifest.json" \
        --set encoder_checkpoint="$ENC"

    CACHES="${CACHES:+$CACHES,}$CACHE"
    # continued-run epoch N is OVERALL epoch (1 [original Z2] + N)
    EPOCHS="${EPOCHS:+$EPOCHS,}$((1 + 10#$N))"
done

echo "caches=$CACHES epochs=$EPOCHS"
test -n "$CACHES" || { echo "ERROR: no epoch checkpoints found, nothing to evaluate" >&2; exit 1; }

echo "=== paired-significance eval across 3 REGIME-confirmed regimes ==="
python -u -m analysis.evaluate_our_trees_continued \
    --mode e2e \
    --packed-root "$MCP" \
    --original-cache "$MAT/validation_cache.pt" \
    --checkpoints "$CACHES" \
    --epochs "$EPOCHS" \
    --regimes "0.01:0.0,0.01:0.001,0.001:0.0005" \
    --d-embed 32 \
    --out-dir "$OUT_DIR"

echo "AGENT2 E2E MATERIALIZE+EVAL DONE -> $OUT_DIR"; date
