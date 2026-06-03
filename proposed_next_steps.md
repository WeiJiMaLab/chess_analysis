# Proposed Next Steps

Ordered by feasibility and dependency. Key constraints: pipeline speed (DIRE),
minimal models first, human-machine comparison strength.

**The ordering logic:**
- Analysis 0 answers: is the oracle worth comparing against humans at all?
- Analysis 1 answers: do oracle and humans agree on the same positions?
- Analysis 2 answers: can we make the model fast enough to iterate?
- Analysis 3 answers: does engine strength explain any mismatch from Analysis 1?

---

## Analysis 0 — Directional test (today, no GPU, no new data)

**Goal:** Before committing to expensive tree generation, use existing data to answer
the directional question: does the oracle's stopping behavior track the same position
features as human RT?

This is HIGH PRIORITY. Even though the data sources differ (filtered_shard positions
≠ human game positions), a directional match would provide strong motivation for
Analysis 1. A directional mismatch would raise serious questions about whether
Analysis 1 is worth doing at all.

### Step 0a: Oracle stop step on existing filtered_shard trees

39K trees exist at `/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/generated_trees_combined/`.
Each has `oracle_root_q_trace` and `oracle_best_move_index`. Run
`compute_budgeted_oracle()` from `lmcos/src/data/preprocess_mc/oracle.py` on each
tree to get the DP-optimal `oracle_stop_step`. Extract board features from the root FEN.

Plot `oracle_stop_step` vs {branching, material, gain_depth_equiv, toptwo_equiv}.
Compare the sign of each correlation with the human RT correlations already measured.

**Expected output:** A correlation table analogous to the human one, showing whether
the oracle is sensitive to the same features in the same directions.

**Feasibility:** High. All code exists. No GPU. Estimate: 2–4 hours to implement and run.

**Script to write:** `lmcos/analysis/oracle_stop_step_features.py`

### Step 0b: Logistic regression baseline on packed episode data

**Can be done today. No GPU. No new data.**

Extract per-snapshot scalar statistics from the packed episode shards:
`n_nodes_so_far`, `root_branching`, `best_q_so_far`, `wdl_var_root`,
`remaining_budget`, `move_ply`.

Train a logistic regression: `f(tree_stats, budget) → sign(advantage)` (halt vs continue).

**Why this matters:** If a simple LR over raw scalar statistics reaches ≥80% sign
accuracy (matching the current GNN + MC controller), the GNN is not adding meaningful
representational power. This would make Analysis 2 (skip GNN pretraining) the
obvious path forward.

**Feasibility:** Very high. Estimate: 2–3 hours.
**Script to write:** `lmcos/analysis/lr_advantage_baseline.py`

---

## Analysis 1 — Fair comparison (this week, GPU needed)

**Goal:** Generate oracle trees on the *same* human board positions, compute
`oracle_stop_step`, and compare directly with `log(RT)` for those positions.

**Prerequisite:** Analysis 0 gives at least a directional signal that the oracle and
human RT agree on some features. If Analysis 0 shows complete mismatch, revisit
whether to proceed.

**Decisions resolved:**
- **Budget:** Fixed at 96 expansions (matching training data format). Incorporating
  human clock time as the budget risks giving the oracle implicit knowledge of the
  human behavioral variable we are trying to predict — this would poison the comparison.
- **Lc0 weights:** Use ysagiv's existing weights
  (`tree_encoder_child_wdl_async_k1_subtree_weighted.pt`). Do not retrain.
- **Position selection:** Uniform random from `processed_moves_nonzero`
  (ply 15–75, opponent clock ≥ 60s). Stratify by branching factor only if the
  uniform sample is degenerate.

### Step 1: Generate trees for human FENs

Extract FENs from `processed_moves_nonzero`. Run the lmcos tree-building pipeline
(Lc0 iterative expansion to 96 nodes) on those FENs, using the same `build_tree.py`
machinery as for `sampled_root_fens_2023.txt`.

**Scale:**
- Start with **1K** as the very first smoke test — must complete in minutes, not hours.
  If 1K takes more than ~15 min on the A100, the pipeline needs batching work before scaling.
- If 1K is fast: scale to **10K** for the main comparison.
- Do not go to 50K+ until 10K produces sensible results.

**Feasibility:** Medium. Needs SLURM + A100. Pipeline exists but was run by ysagiv —
may need minor wiring to accept FENs from `personal.db` rather than `sampled_root_fens_2023.txt`.

### Step 2: Compute oracle_stop_step

For each generated tree, call `compute_budgeted_oracle()` with default
`BudgetedOracleConfig`. Returns `policy.optimal_stop_step` — the exact training target.

**Feasibility:** Easy. No GPU. Seconds per tree.

### Step 3: The key plots

**Plot A:** `oracle_stop_step` vs `log(RT)` for the same positions.
A positive correlation supports the quasi-optimality hypothesis. Zero or negative
correlation means the oracle cost function does not match human deliberation cost.

**Plot B:** `oracle_stop_step` vs board features (branching, material, gain_depth, toptwo).
Compare signs with the human RT feature correlations from Analysis 0 and the
existing human dataset.

**Plot C:** Model predicted stop step (GNN + MC controller) vs `oracle_stop_step`.
Should be near-diagonal — deviations identify positions where the model fails.

---

## Analysis 2 — Minimal model (this week, parallel with Analysis 1)

**Goal:** Reduce total training time from day(s) to hours. The GNN pretraining is
the known bottleneck (took ~1+ days). Analysis 2 is HIGH PRIORITY because fast
iteration is critical regardless of what Analysis 1 finds.

**Decisions resolved:**
- Focus on **GNN loss** (not stop-step accuracy) when evaluating smaller architectures.
  Stop-step accuracy is a downstream metric of the MC controller; GNN loss is what
  we can directly optimize and measure.
- 75% stop-step accuracy is acceptable if it enables 2-hour iteration cycles.

### Step 1 (HIGHEST PRIORITY): Can we skip GNN pretraining entirely?

Train a smaller GNN from **random initialization** — no child-WDL pretraining.
This is the burning question. If a randomly initialized smaller GNN reaches
acceptable GNN loss within a few hours, pretraining is unnecessary and the whole
pipeline collapses to a much simpler training loop.

**Ablation series** (run on 10–20% of existing training shards, no new data):

| Config | d_embed | d_message | n_heads | hidden_dim | hidden_layers | Pretraining? |
|--------|---------|-----------|---------|------------|---------------|-------------|
| A (baseline) | 128 | 128 | 4 | 256 | 3 | Yes (current) |
| B | 64 | 64 | 2 | 128 | 2 | No (from scratch) |
| C | 32 | 32 | 1 | 64 | 1 | No (from scratch) |
| D | 16 | 16 | 1 | 32 | 1 | No (from scratch) |

**Success criterion:** Config C or D reaches acceptable GNN loss within ≤2 hours
total training time on a single GPU.

**Feasibility:** High. No new data. Config YAML changes only. SLURM job needed.

### Step 2: Logistic regression baseline

(See Analysis 0b above — this can be done immediately and answers a subset of the
same question. Results inform whether Step 1 is even necessary.)

---

## Analysis 3 — Weaker engine (conditional, on hold)

**Trigger:** Only pursue if Analysis 1 shows that `oracle_stop_step` does NOT
correlate with `log(RT)`, and specifically if the disagreement is in the direction
consistent with the engine being "too strong" (trivially solves positions humans
find hard).

**Engine choice resolved:** Use Stockfish with `UCI_LimitStrength` + `UCI_Elo=2000`
(matching the approximate ELO of the human players). Stockfish is preferred over a
smaller Lc0 network because:
1. It is cheaper and faster to run (CPU, no GPU needed)
2. Analysis 3 requires regenerating all trees from scratch anyway — there is no
   consistency requirement with the existing Lc0 frozen weights (those weights were
   only needed for Analysis 1 because we are using the existing model checkpoint)

**Note on format:** Stockfish provides centipawns, not native WDL. The oracle's
`halt_rewards` would need to be converted (centipawns → win probability via logistic
regression, consistent with Russek et al.'s approach).

---

## Ordering Summary

| # | Analysis | Prerequisite | GPU? | Time estimate | Status |
|---|----------|-------------|------|--------------|--------|
| **0a** | Oracle stop step on existing trees → compare with human RT features | Nothing | No | 2–4 h | Do now |
| **0b** | LR baseline on packed episodes | Nothing | No | 2–3 h | Do now |
| **1** | Generate 1K→10K human FEN trees, compute oracle, compare with RT | A0 directional signal | Yes (A100) | 1–4 h compute | This week |
| **2** | Skip GNN pretraining? Ablation on smaller architectures | Nothing (existing data) | Yes (1 GPU) | ~1 day | This week, parallel |
| **3** | Weaker engine (Stockfish ELO=2000) trees | A1 shows mismatch | CPU | Same as A1 | On hold |

---

## What I Would Not Do

- Do not scale Analysis 1 beyond 10K positions before the 1K smoke test confirms the
  pipeline completes in minutes and the oracle correlates with RT in at least one direction.
- Do not run GNN pretraining before testing whether random initialization (Analysis 2)
  gives good enough GNN loss on a small subsample.
- Do not start Analysis 3 before Analysis 1 gives a directional answer.
- Do not retrain GNN from scratch before the LR baseline (Analysis 0b) establishes
  whether the GNN is even adding value over raw tree statistics.

---

## Alignment with Slides

The slide deck (`human_analytics/presentations/lmcos-overview/slides.md`) reflects
this plan in Section IV:
- **Analysis 1 slide** → fair comparison, 3-step plan
- **Analysis 2 slide** → minimal model (LR baseline + skip pretraining question)
- **Analysis 3 slide** → weaker engine (conditional)

Analysis 0 is not yet in the slides — it is the immediate prerequisite for deciding
whether Analyses 1–3 are worth presenting at all, and should be completed before
the next deck update.
