# R-U3 — Model comparison: controller vs hand-written stopping baselines

**Ref:** `R-U3` · [Index](README.md) · Plan: [unify.md §U3](../unify.md) · Thread: LMCOS (parallel to U1/U2)

Situates the GNN+MC budgeted controller against simple/lesioned stop rules, on the **DP-oracle
regret** objective. Runs entirely on **existing** packed-episode diagnostics — no new trees, no
GPU — so it advances in parallel with U1 (tree gen) and U2 (Lc0 gain).

## Summary (after first run)

| | |
|---|---|
| **Description** | `analyze_budgeted_controller_run.py` (`core` mode) on the
`subtree_weighting_root_budget` controller's per-episode diagnostics (**30,630 episodes**),
comparing the controller's predicted stop vs the `_baseline_sweep` hand-written rules vs the DP
oracle. |
| **Rationale** | Quantify the controller's lift; sanity-check the baseline suite (incl. the new
gain-depth rule) on real data. |
| **Finding** | **Controller dominates.** avg regret **+0.026** (best baseline +0.148 → **5.6×
lower**; always-stop +0.52; never-stop +1.37). Exact oracle stop step **63%** of episodes.
**Gain-depth-only is the best simple baseline** (+0.148). |

## Results (30,630 episodes; avg regret vs DP oracle, lower = better)

| Stop rule | avg regret ↓ | exact-stop acc | avg expansions |
|---|---|---|---|
| **GNN+MC controller** | **+0.0264** | **0.630** | 4.9 |
| `halt_when_gain_le_0.05` (gain-depth) | +0.1482 | 0.100 | 1.6 |
| `always_halt_0` (always stop) | +0.5215 | 0.231 | 0.0 |
| `always_continue_to_end` (never stop) | +1.3668 | 0.099 | 30.1 |

Notes:
- **Gain-depth-only** (halt when marginal halt-reward gain ≤ ε) is the strongest hand-written rule
  but still ~5.6× the controller's regret — it halts very early (~1.6 expansions), capturing the
  common "stop almost immediately" cases but missing when deliberation pays off.
- `always_halt_0` gets 23% exact-stop because the oracle itself halts at step 0 on ~23% of
  episodes (many positions need no search) — yet its blanket regret is 20× the controller's.
- Full report + 15 diagnostic plots: `/scratch/gpfs/GRIFFITHS/hl4291/tmp/u3_baselines/`
  (`report.md`, `summary.json`, `regret_by_budget_bucket.png`, …).

## Procedure

| Step | Status |
|------|--------|
| Add gain-depth-only baseline + `tests/test_budgeted_baselines.py` (12 tests) | ✅ done |
| Drop geometric baseline (per hl4291); brainstorm semi-smart rules | ✅ (unify §U3) |
| Run controller-vs-baseline comparison on existing diagnostics | ✅ done (this report) |
| Add semi-smart baselines (argmax-stability, value-plateau, fixed-fraction) + tests | ⬜ next |
| Model-variant comparison: run on other controller diagnostics (ysagiv configs) | ⬜ next |
| MLP-over-tree-stats baseline (`tree_stats_baseline.py`) vs GNN | ⬜ |
| Re-run on the **U1.3** controller (refit on human FENs) | ⬜ after U1.3 |

## Run command

```bash
cd lmcos && source /home/hl4291/venv/bin/activate
PYTHONPATH=$PWD python -m cts.analysis.analyze_budgeted_controller_run --config <yaml>
# yaml: diagnostics_path, log_path (training .out), output_dir, report_mode: core
```

## Notes
- **Independence:** uses existing `*_diagnostics.jsonl` (oracle labels + trajectory + controller
  predictions) — no dependence on U1/U2. Re-run on any controller's diagnostics to compare models.
- The many ysagiv `fittedq_*_diagnostics.jsonl` are a ready-made **model-variant** comparison set
  (different configs/lesions) — same command, different inputs.
