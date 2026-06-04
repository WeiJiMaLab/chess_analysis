# R-A2 — Analysis 2: minimal meta-controller / skip GNN pretrain

**Ref:** `R-A2` · [Index](README.md) · Prerequisite: [(R-A0)](analysis-0-oracle-baseline.md)

## Summary (during active implementation)

| | |
|---|---|
| **Description** | Build complexity **up**: **A2.1** tree-statistic summaries → MC (no GNN); **A2.2** tiny GNN from scratch if summaries insufficient. Parallel to A1. |
| **Rationale** | GNN pretrain ~1+ day blocks iteration; A0b showed 86.4% sign acc without full encoder — find minimum viable architecture. |
| **Expectation** | Acceptable **GNN loss** (or tree-stat baseline) within ≤2 h GPU; joint `unfreeze_encoder` path viable for 16-dim GNN. |
| **Open questions / notes** | Need ysagiv pretrained GNN loss baseline for “acceptable.” Job **9215254** started tiny child-WDL pretrain (see notebook). Config D YAML still missing. |

## Procedure

| Step | Status |
|------|--------|
| **A2.1** Tree-statistic features → MC (no GNN head) | ⬜ incomplete |
| **A2.2** Config D YAML (`subtree_weighting_root_scratch_D.yaml`, d_embed=16, unfreeze) | ⬜ incomplete |
| Verify `controller_train.py` online GNN path when `unfreeze_encoder: true` | ⬜ incomplete |
| Train Config D on 10–20 ysagiv shards (~5–10%) | ⏳ child-WDL pretrain submitted |
| Compare GNN loss vs pretrained 128-dim baseline | ⬜ incomplete |
| Ablation series Config C, B if D insufficient | ⬜ incomplete |
| `test_minimal_model.py` (shapes, param count, no NaN, reproducibility) | ⬜ incomplete |

## Notes

### Success metric

- Primary: **GNN loss** on ablation shards (not stop-step accuracy alone)
- Behavioral anchor: A0b minimal MLP **86.4%** val sign acc vs GNN+MC **90.1%**

### Config D (planned)

- `d_embed=16`, `d_message=16`, `n_heads=1`, `hidden_dim=32`, `hidden_layers=1`
- `unfreeze_encoder: true` — online embeddings, no pre-materialized `z_root` cache
- Data: `/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/controller_packed_combined_nomaint_no_xaba/train/` (subsample 10 shards)

### Code paths

- `lmcos/src/train/controller_train.py` — joint training
- `lmcos/src/train/gnn_pretrain.py` — fallback two-stage if materialization required
- `lmcos/slurm/configs/4_supervised_controller/`

### Guardrails

- Do **not** run large-scale GNN pretrain before A2 establishes minimum viable architecture on existing shards.
