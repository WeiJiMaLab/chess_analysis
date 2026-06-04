# R-A1 — Analysis 1: human FEN trees + oracle vs log(RT)

**Ref:** `R-A1` · [Index](README.md) · Prerequisite: [(R-A0)](analysis-0-oracle-baseline.md)

## Summary (during active implementation)

| | |
|---|---|
| **Description** | Lc0 oracle trees on human `processed_moves_nonzero` FENs (ply 15–75, opp clock ≥ 60s); join `oracle_stop_step` with `log(RT)`; plots vs board features and GNN+MC predictions. |
| **Rationale** | Tier-B validation: same positions, normative stopping vs human effort ([(R-THEORY)](human-theory-stopping.md)). |
| **Expectation** | r(oracle_stop_step, log RT) > 0 on matched FENs; 10K only after 1K smoke timing gate passes. |
| **Open questions / notes** | Budget **96** expansions (no clock-as-budget). ysagiv weights only. Join on full FEN (castling/ep). If r weak → [(R-A3)](analysis-3-weaker-engine.md). |

## Procedure

| Step | Status |
|------|--------|
| Export 1K human FENs → `human_fens_1k.txt` (644 valid) | ✅ done |
| Verify 6-field FEN compatibility with `build_tree.py` | ✅ done |
| SLURM: `VENV_DIR`, `resume: true`, `human_trees_1k_smoke.yaml` | ✅ done |
| 1K tree smoke — **644/644** `.pt` in `human_trees_1k/` | ✅ done |
| Verify tree schema (`oracle_root_q_trace`, `oracle_best_move_index`, …) | ⬜ incomplete |
| Scale to 10K (`human_trees_10k.yaml` or shard orchestrator) | ⬜ incomplete |
| `compute_budgeted_oracle()` on all trees | ⬜ incomplete |
| Join FEN → `personal.db` for log(RT), clock, branching | ⬜ incomplete |
| `human_oracle_comparison.py` + `test_human_oracle_comparison.py` | ⬜ incomplete |
| Plot A: oracle_stop_step vs log(RT) | ⬜ incomplete |
| Plot B: oracle_stop_step vs board features (vs human r) | ⬜ incomplete |
| Plot C: GNN+MC predicted_stop vs oracle_stop_step | ⬜ incomplete |

## Notes

### Decisions (resolved)

- Position selection: uniform random from `processed_moves_nonzero`
- Engine: lc0 + ysagiv `tree_encoder_child_wdl_async_k1_subtree_weighted.pt` — do not retrain
- Output (10K target): `/scratch/gpfs/GRIFFITHS/hl4291/tmp/human_trees/`

### Config (1K smoke)

- FEN: `/scratch/gpfs/GRIFFITHS/hl4291/tmp/human_fens_1k.txt`
- Trees: `/scratch/gpfs/GRIFFITHS/hl4291/tmp/human_trees_1k/`
- YAML: `lmcos/slurm/configs/1_preprocess_data/human_trees_1k_smoke.yaml`

### Commands

```bash
cd /home/hl4291/chess_analysis/lmcos
export VENV_DIR=/home/hl4291/venv
sbatch slurm/1_preprocess_data/generate_dataset_shard.slurm \
  slurm/configs/1_preprocess_data/human_trees_1k_smoke.yaml

# 10K scale (after smoke timing OK):
# ./slurm/1_preprocess_data/submit_generate_dataset_shards.py \
#   --config slurm/configs/1_preprocess_data/human_trees_10k.yaml \
#   --shard-size 1000 --submit
```

### Tests (`lmcos/tests/test_human_oracle_comparison.py`)

- `root_position_spec` matches input FEN
- `oracle_root_q_trace` shape `[96, n_legal_moves]`
- `oracle_stop_step` ∈ [0, 96]; no duplicate FENs; join has no NaN pairs

### Guardrails

- Do **not** scale past 10K until 1K completes in ~15 min on A100 (batching sanity).
- Do **not** start A3 before A1 gives a directional answer on matched positions.
