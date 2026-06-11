# R-LMCOS-STAGE4 — Stage-4 controller ablation harness

**Ref:** `R-LMCOS-STAGE4` · [Index](README.md)

## Summary (after implementation)

| | |
|---|---|
| **Description** | 2026-05-29 harness: compare three fitted-Q configs on full ysagiv materialized caches; metrics YAML + PNG per run; `submit_configs.sh` glob. |
| **Rationale** | Repeatable encoder/input ablations without hand-edited job lists. |
| **Expectation** | Flat `slurm/outputs/4_supervised_controller/` artifacts; step-based val MSE + greedy regret. |
| **Finding** | `ControllerTrainMetricsLogger` replaces JSONL + external plots; configs `legacy_root_budget`, `subtree_weighting_root_budget`, `subtree_weighting_root` documented. |

## Procedure

| Step | Status |
|------|--------|
| Repo layout refactor (`cts.*`, `slurm/<stage>/`) | ✅ done |
| Metrics YAML + `plot-metrics` / `plot-metrics-compare` | ✅ done |
| `submit_configs.sh` + comparison dependency job | ✅ done |
| Document `VENV_DIR` on Della | ✅ done |

## Notes

### Submit

```bash
cd /home/hl4291/chess_analysis/lmcos
export VENV_DIR=/home/hl4291/venv
./slurm/4_supervised_controller/submit_configs.sh
```

Full tables and rename arc: [(R-ARCH-LMCOS)](archive-lmcos-notebook-legacy.md) § 2026-05-29.
