# R-A0 — Analysis 0: oracle direction + minimal MC

**Ref:** `R-A0` · [Index](README.md) · Next: [(R-A1)](analysis-1-human-oracle.md), [(R-A2)](analysis-2-minimal-model.md)

## Summary

| | |
|---|---|
| **Description** | **A0a:** `oracle_stop_step` vs board features on all lmcos `filtered_shard` trees (n=39,668). **A0b:** 4-feature MLP vs GNN+MC halt/continue sign accuracy. |
| **Rationale** | De-risk A1 (human FEN trees) and A2 (skip GNN pretrain) before cluster spend. |
| **Expectation** | ≥ 3/4 feature directions match human RT proxies; val sign acc ≥ 80% without GNN. |
| **Finding** | **A0a:** 4/4 RT directions match (branching/material ≈ 0, effectively noise). **A0b:** val sign **54.7%**, exact stop **5.4%**, r(pred,oracle) **+0.29** — prior 86.4% invalidated. |

## Procedure

| Step | Status |
|------|--------|
| A0a `oracle_stop_step_features.py` — corrected halt_rewards + feature definitions | ✅ done |
| A0b `minimal_mc_baseline.py` (2K train / 500 val) — correct re-run | ✅ done |
| `converged_expansions_analysis.py` (renamed from min_expansions_analysis) | ✅ done |
| Oracle invariant tests (43 tests, verified on 1K real trees) | ✅ done |

---

## A0a: oracle_stop_step vs board features

### Definitions

**oracle_stop_step** (`BudgetedOraclePolicy.optimal_stop_step` / packed `oracle_stop_step`):
- From `build_compact_trajectory` → `budgeted_oracle_from_trajectory` (same path as `pack.py`).
- `trajectory["halt_rewards"][i]` = teacher fixed Q for the MCTS-recommended action at controller step `i`.
- **Predicted** stop at eval uses `predicted_stop_from_advantages()` in `oracle.py` (controller eval rule).
- **Oracle** stop is `policy.optimal_stop_step` from the DP — not the greedy first-zero-advantage rule on predicted values.

**converged_expansions** (`converged_expansions_analysis.py`):
- 1 + last step where `oracle_best_move_index` deviated from its final recommendation.
- First step of permanent argmax stability. Excludes step 0 (all-zero Q-trace artifact).
- `oracle_stop_step ≤ converged_expansions` always. Equal iff no MCTS oscillation after first correct recommendation (~43% of trees at full budget).

### Feature definitions and regime

All features use **Lc0** (96-node oracle MCTS, WDL output from the neural network).
Human RT correlation targets (_HUMAN_R) are from a **Stockfish depth=5** analysis of human lichess 10+0 games (human_analytics, n=1M). The comparison is deliberately cross-regime: alignment means positional factors that drive Lc0 oracle search depth also drive human thinking time.

| Feature | Lc0 definition | Stockfish analog | Regime |
|---|---|---|---|
| `toptwo_equiv` | `Q_final[1st best] − Q_final[2nd best]` (gap in deep Lc0 Q-values) | `e_win_best − e_win_second_best` at depth_deep=5 | Lc0 deep (96 nodes) vs SF deep (d=5) |
| `gain_depth_equiv` | `Q_final[best_idx[-1]] − Q_final[best_idx[1]]` | VOC = `V_deep(a_deep) − V_deep(a_shallow)` | Lc0 deep (96 nodes) vs SF deep (d=5); **both** evaluate the recommended actions at deep, not shallow |
| `branching` | `n_possible_moves` from FEN | same | Engine-independent (board geometry) |
| `material` | `n_self_pieces_exc_pawns` from FEN | same | Engine-independent (board geometry) |

Note: `toptwo_equiv` uses the **deep** evaluation (not myopic). This matches the Stockfish definition which also uses `depth_deep` for both moves. Using step-1 Q-values (1 explored action, all others = 0) would be ill-defined and would not match the Stockfish reference.

### Results (n=39,668 trees, budget=T=96, DEFAULT cost config)

Previous run used glob `filtered_shard_0000?` and missed shards 00010–00019 (half the data). Fixed to `filtered_shard_*`; all 20 shards, 39,668 trees in ~133s.

| Feature | r(oracle_stop_step) | r(converged_expansions) | r(human RT) | OSS dir | CE dir |
|---|---|---|---|---|---|
| branching | +0.014 | −0.065 | +0.195 | ✓ (tiny) | ✗ |
| material | +0.011 | −0.059 | +0.039 | ✓ (tiny) | ✗ |
| gain_depth | **+0.233** | +0.052 | +0.096 | ✓ | ✓ |
| toptwo | **−0.290** | −0.590 | −0.064 | ✓ | ✓ |

`r(oracle_stop_step, converged_expansions)` = **0.525**  
oracle_stop_step: mean=23.6, std=23.1, median=15  
converged_expansions: mean=50.9, std=35.4, median=58

**oracle_stop_step: 4/4 correct directions.** **converged_expansions: 2/4** (branching and material wrong direction). Results stable across the doubled sample.

**Interpretation of branching and material (r ≈ 0.01 for oracle_stop_step):**

The near-zero oracle correlations for branching and material are structurally expected. oracle_stop_step is driven by VALUE convergence — when does the Q-value gap between the oracle's recommended action and the alternative become large enough to justify stopping given the search cost? This is a property of the *value landscape*, not of board structural complexity.

- **Branching** drives human RT because humans explicitly enumerate and evaluate candidate moves. More possible moves → more cognitive load → longer RT. But Lc0 handles branching natively through its prior (PUCT selection), so a high branching factor doesn't make it harder for the oracle to converge on a value decision — what matters is whether any action is clearly dominant (captured by toptwo).
- **Material** similarly drives human RT through tactical complexity (pieces create threats to track), but Lc0 evaluates positions holistically via the value network — raw piece count doesn't directly determine how quickly the MCTS Q-values converge.

Toptwo and gain_depth (VOC) are the oracle's natural stopping drivers because they directly capture *value uncertainty*: how clearly dominant is the best action (toptwo), and how much value is gained from searching deeper (gain_depth/VOC)? These are the quantities the DP oracle explicitly optimizes over.

### Budget sensitivity

oracle_stop_step varies with starting budget (cost rises convexly as budget depletes):
- Tighter budgets → earlier oracle_stop_step (cost makes waiting expensive)
- Full budget (96 nodes) → oracle_stop_step approaches the zero-cost case (first step where MCTS recommends final_best)

### Figures

- `lmcos/analysis/figures/oracle_stop_step_vs_human_rt.png` — feature correlations bar
- `lmcos/analysis/figures/oracle_stop_step_correlation_matrix.png` — correlation matrix
- `lmcos/analysis/figures/oracle_stop_vs_converged_expansions.png` — scatter + feature bar (all n=39,668)

---

## A0b: minimal MLP vs GNN+MC

| Model | Val sign acc | Val exact stop | r(pred, oracle) |
|---|---|---|---|
| GNN+MC | 90.1% | — | — |
| Minimal MLP | **54.6%** | **5.2%** | **+0.291** |

**Script:** `lmcos/analysis/minimal_mc_baseline.py`. Run 2026-06-05, 2K train / 500 val trees, budget=43.

**Training dynamics:** MSE loss falls monotonically (0.143 → 0.053 over 20 epochs) but sign accuracy oscillates wildly (0.54 → 0.41 → 0.68 → 0.52). This indicates the model fits the regression target but the zero-crossings of predicted advantages are unstable — the 4 features do not define a clean decision boundary for halt vs. continue.

**Interpretation:** The near-chance sign accuracy confirms the 4 raw scalar features (best_q, wdl_var, t_norm, budget_rem_norm) carry very little information about the oracle's halt/continue decision. The weak r=+0.29 on stop steps suggests a faint timing prior (probably t_norm), but the GNN's encoding of the full Q-value landscape is essential for discriminative performance.

### Invalid runs (do not cite)

| Run | Date | Outcome | Why invalid |
|---|---|---|---|
| A0b minimal MLP (2K/500 val, budget 43) | 2026-06-03 | val sign acc ≈ 86.4% | Used evolving `q[s, best_idx[s]]` for halt_rewards, not `Q_final[best_idx[s]]` |

---

## Code paths (canonical)

| Layer | Module | Role |
|---|---|---|
| Trajectory | `pack.build_compact_trajectory` | Precomputed `halt_rewards`, `tree_sizes` |
| Oracle DP | `oracle.compute_budgeted_oracle` | `target_advantages`, `optimal_stop_step` |
| Analysis glue | `pack.budgeted_oracle_from_trajectory` | Trajectory → policy (same as packing) |
| Features | `analysis/board_tree_features.py` | FEN + Lc0 tree features for A0/A1/CE |
| Predicted stop | `oracle.predicted_stop_from_advantages` | Controller eval rule on model outputs |

---

## Tests

**72 tests, all passing** (`tests/test_oracle_stop_step_features.py`, `tests/test_oracle_stop_vs_min_expansions.py`, `tests/test_converged_expansions.py`, `tests/test_minimal_mc_baseline.py`).

Key invariants verified on real trees (n=1,000):
- `halt_rewards[i] == oracle_final_root_q_values[best_idx[i]]` — no exceptions
- R_halt ≤ R_continue everywhere (zero cost) — no exceptions
- R_halt == R_continue iff `best_idx[i] == final_best` (tie-free trees) — no exceptions
- `oracle_stop_step < converged_expansions` iff MCTS reversal exists between them — no exceptions
- Quality gap > cost at all continue steps — no exceptions
- `oracle_stop_step ≤ converged_expansions` always — no exceptions
- Controller and oracle use the same criterion (advantage ≤ 0) — confirmed

---

## Implementation note: what was corrected

The original A0a analysis had two bugs, both now fixed:

1. **halt_rewards used evolving Q-estimates** (`oracle_root_q_trace[s, best_idx[s]]`) instead of the teacher's fixed Q-values (`oracle_root_q_trace[-1][best_idx[s]]`). This made oracle_stop_step sensitive to the rate of Q-value growth rather than to action identity, producing spurious correlations.

2. **gain_depth used a mixed-regime formula**: numerator from `oracle_root_q_trace[-1].max()` (final Q), denominator from `oracle_root_q_trace[first_nz, :].max()` (evolving Q at first nonzero step). The correct VOC analog uses the teacher's fixed Q for both the deep and shallow recommendation.

Both bugs have been corrected. The original figures are superseded by the corrected results above.
