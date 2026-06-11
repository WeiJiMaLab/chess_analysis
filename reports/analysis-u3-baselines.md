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
| `halt_at_frac_0.25` (fixed-fraction) | +0.0693 | 0.193 | 8.0 |
| `halt_at_frac_0.1` | +0.0969 | 0.285 | 3.2 |
| `halt_when_gain_le_0.05` (gain-depth) | +0.1482 | 0.100 | 1.6 |
| `halt_at_frac_0.5` | +0.1932 | 0.104 | 16.1 |
| `value_plateau_w3_le_0.01` | +0.2412 | 0.099 | 3.3 |
| `always_halt_0` (always stop) | +0.5215 | 0.231 | 0.0 |
| `always_continue_to_end` (never stop) | +1.3668 | 0.099 | 30.1 |

Findings:
- **Controller dominates** — ~2.6× lower regret than the best baseline, 20–52× the trivial anchors;
  exact oracle stop step **63%** of episodes.
- **Fixed-fraction-of-budget is the best *hand-written* rule** (`halt_at_frac_0.25`, +0.069), beating
  every value-based rule. A position-blind "spend ~10–25% of the budget then stop" anchor predicts the
  oracle's stop better than the noisy per-step value signal — consistent with the budgeted oracle's
  convex time cost making the stop largely a *budget-structure* decision.
- **Value-plateau under-performs gain-depth** (+0.241 vs +0.148). The windowed smoothing delays the
  stop (avg 3.3 vs 1.6 expansions), but the oracle usually wants to stop *early*, so noise-robustness
  hurts for this objective — the eager single-step rule wins.
- `always_halt_0` gets 23% exact-stop because the oracle itself halts at step 0 on ~23% of episodes
  (many positions need no search) — yet its blanket regret is 20× the controller's.
- Full report + 15 diagnostic plots: `/scratch/gpfs/GRIFFITHS/hl4291/tmp/u3_baselines/`.

## Is regret the right metric? Budget vs value in the oracle's stop

Prompted by fixed-fraction-of-budget being the best hand-written rule, we decomposed the oracle's
stop decision on the same 30,630 episodes (recovered cost config: `time_lambda=18.537`, maint off).
Figures: `human_analytics/figures/u3_baselines/{controller_vs_baselines,oracle_stop_cost_bimodal}.png`.

1. **Stops are bimodal, not uniformly budget-dominated.** At the oracle's stop step, the per-step
   cost is **<0.01 in ~49%** of episodes (value-driven — median *gross value of continuing = 0.000*,
   nothing left to gain) but **>0.1 in ~28%** (cost-forced — budget so depleted the convex time cost
   swamps any value). So budget has outsized sway on a sizable minority; the core is value-driven.
2. **The oracle stops early.** Median stop = **3.8% of the starting budget** (IQR 0.9–14%);
   `corr(stop, budget)=+0.41`. The answer is almost always "stop soon," so a small fixed fraction
   (ρ=0.1–0.25) approximates it *without value info* — which is why a position-blind rule competes.
3. **Regret is a weak discriminator among *good* rules.** The per-episode return spread is large
   (~1.4–1.75 → never-stop regret +1.37), but good rules cluster (controller +0.026, fixed-fraction
   +0.069, gain-depth +0.148 — a few % of the range). **Exact-stop separates them far better**
   (controller 0.63 vs fixed-fraction 0.19 vs gain-depth 0.10).

**Takeaways.** (a) `time_lambda` (cost weight) and the budget distribution are **levers**: ~28% of
stops are budget-forced, masking the value signal — lowering the cost weight / using longer budgets
would sharpen value-driven stopping and *reduce* fixed-fraction's edge. (b) **Report exact-stop (or
a value-normalized regret) alongside regret** — regret alone undersells the value-based controller.

## Procedure

| Step | Status |
|------|--------|
| Add gain-depth-only baseline + `tests/test_budgeted_baselines.py` (12 tests) | ✅ done |
| Drop geometric baseline (per hl4291); brainstorm semi-smart rules | ✅ (unify §U3) |
| Run controller-vs-baseline comparison on existing diagnostics | ✅ done (this report) |
| Add semi-smart baselines (value-plateau, fixed-fraction) + tests | ✅ done (16 tests) |
| Add argmax-stability baseline (needs oracle_best_move trace) | ⬜ later |
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
