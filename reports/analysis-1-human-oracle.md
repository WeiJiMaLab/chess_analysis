# R-A1 — Analysis 1: human FEN trees + oracle vs log(RT)

**Ref:** `R-A1` · [Index](README.md) · Prerequisite: [(R-A0)](analysis-0-oracle-baseline.md)

## Summary (during active implementation)

| | |
|---|---|
| **Description** | Lc0 oracle trees on human `processed_moves_nonzero` FENs (ply 15–75, opp clock ≥ 60s); join `oracle_stop_step` with `log(RT)`; plots vs board features and GNN+MC predictions. |
| **Rationale** | Tier-B validation: same positions, normative stopping vs human effort ([(R-THEORY)](human-theory-stopping.md)). |
| **Expectation** | r(oracle_stop_step, log RT) > 0 on matched FENs. Smoke on 1K confirmed direction; 10K needed for statistical power. |
| **Open questions / notes** | Budget **96** expansions (no clock-as-budget). ysagiv weights only. If r weak → [(R-A3)](analysis-3-weaker-engine.md). |

## Procedure

| Step | Status |
|------|--------|
| Export 10K human FENs → `human_fens_10k.txt` + manifest | ✅ done (seed=43) |
| Verify 6-field FEN compatibility with `build_tree.py` | ✅ done |
| Submit 10K as 20 parallel shards (500 FENs × 1h GPU each) | ✅ submitted (jobs 9266775–9266794; 3 running, 17 queued under QOS limit) |
| Verify tree schema on smoke trees | ✅ done — see smoke findings below |
| `compute_budgeted_oracle()` on all 10K trees | ⬜ after jobs complete (via pack helpers) |
| Join FEN → `personal.db` for log(RT), clock, branching | ⬜ after jobs complete |
| `human_oracle_comparison.py` on 10K trees | ⬜ after jobs complete |
| Plot A: oracle_stop_step vs log(RT) | ⬜ after jobs complete |
| Plot B: oracle_stop_step vs board features (vs human r) | ⬜ after jobs complete |
| Plot C: GNN+MC predicted_stop vs oracle_stop_step | ⬜ after jobs complete |

## oracle_stop_step on human FEN trees

Same controller pack path as A0a and GNN training:

```
trajectory = build_compact_trajectory_from_payload(tree)
oracle_stop_step = budgeted_oracle_from_trajectory(trajectory, budget, config).optimal_stop_step
```

Features (`branching`, `gain_depth`, …) come from shared `analysis/board_tree_features.py`.

## Smoke findings (n=497 clean trees, seed=42)

The original 1K smoke job hit the 1-hour SLURM wall and produced 497 trees (FENs 0–496).
The `human_trees_1k/` directory also contained 147 stale trees from a prior run (different positions entirely); those are now deleted. `human_oracle_comparison.py` guards against this at load time by checking `root_position_spec == manifest full_fen`.

| Metric | Value |
|---|---|
| r(oracle_stop_step, log RT) | **+0.091** |
| r(branching, log RT) / oracle | +0.208 / +0.166 |
| r(material, log RT) / oracle | +0.094 / +0.089 |
| r(gain_depth, log RT) / oracle | +0.020 / +0.463 |
| r(toptwo, log RT) / oracle | −0.128 / −0.030 |

Direction check: oracle_stop_step r > 0 ✓ (statistically significant at n=497; threshold |r| > 0.088).

Note: gain_depth shows very strong oracle correlation (+0.463) but near-zero human correlation (+0.020). This is consistent with A0a: gain_depth/VOC is a value-landscape feature that strongly predicts when the oracle halts, but humans don't natively compute VOC — their RT is driven more by structural complexity (branching, material). At 10K, this split should sharpen.

## A0a impact assessment

**A1 uses the same pack-path oracle as training.** `human_oracle_comparison.py` calls `build_compact_trajectory_from_payload` + `budgeted_oracle_from_trajectory`. Board/tree features are shared via `board_tree_features.py`.

## Notes

### Data paths (10K)

- FENs: `/scratch/gpfs/GRIFFITHS/hl4291/tmp/human_fens_10k.txt`
- Manifest: `/scratch/gpfs/GRIFFITHS/hl4291/tmp/human_fens_10k_manifest.parquet`
- Trees: `/scratch/gpfs/GRIFFITHS/hl4291/tmp/human_trees_10k/`
- Shard YAMLs: `lmcos/slurm/configs/1_preprocess_data/human_trees_10k_shards/`

### Rerun command (after 10K trees complete)

```bash
cd /home/hl4291/chess_analysis/lmcos && export VENV_DIR=/home/hl4291/venv
python analysis/human_oracle_comparison.py \
    --trees-dir /scratch/gpfs/GRIFFITHS/hl4291/tmp/human_trees_10k \
    --manifest /scratch/gpfs/GRIFFITHS/hl4291/tmp/human_fens_10k_manifest.parquet
```

### Resubmit remaining shards (if queue clears)

```bash
# Resubmit shards 3-19 as they complete:
export VENV_DIR=/home/hl4291/venv
python slurm/1_preprocess_data/submit_generate_dataset_shards.py \
    --config slurm/configs/1_preprocess_data/human_trees_10k.yaml \
    --shard-size 500 --project-dir /home/hl4291/chess_analysis/lmcos \
    --time 01:00:00 --submit
```

### Guardrails

- Do **not** start A3 before A1 gives a directional answer on matched positions.
- FEN validation is enforced at load time in `human_oracle_comparison.py` — stale trees in the output dir are skipped automatically.
