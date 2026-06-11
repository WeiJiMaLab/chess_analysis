# R-CLEANUP-0604 — Repository cleanup log

**Ref:** `R-CLEANUP-0604` · [Index](README.md)

## Summary (after implementation)

| | |
|---|---|
| **Description** | Permanent deletion of stale scratch dirs and orphan smoke scripts during chess_analysis restructuring. |
| **Rationale** | Recover disk; avoid confusion with current `personal.db` / pipeline paths. |
| **Expectation** | Paths logged for possible restore from backup. |
| **Finding** | Scratch under `/scratch/gpfs/GRIFFITHS/hl4291/tmp/` and listed `human_analytics/` test orphans removed. |

## Notes

Full path lists: [(R-ARCH-HUMAN)](archive-human-analytics-notebook-legacy.md) § 2026-06-04 Cleanup Log.

### Scratch (sample)

- `pos_with_engine_eval/`, `pos_with_engine_eval_100k/`, `pos_with_engine_eval_1m/`
- `pipeline_smoke*`, `voc_mq_eval/`, `load_moves/`

### Orphan scripts (sample)

- `slurm/scripts/tests/compare_legacy_new_pipeline_smoke.py`
- `tests/test_pipeline_compare_smoke.py`
