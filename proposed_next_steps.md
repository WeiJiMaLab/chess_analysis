# Proposed Next Steps

Ordered by scientific importance. Key constraints: pipeline speed, human-machine
comparison strength, and minimal models first.

---

## Analysis 1: Fair Comparison — lmcos vs Human RT

**Goal:** Generate lmcos oracle trees on *human* board positions (from `personal.db`),
compute DP-optimal stopping steps, compare with actual human RT for the same positions.
This is the core comparison the whole project is building toward.

### Step 1: Generate trees for human FENs

Take FENs from `processed_moves_nonzero` (filtered to ply 15–75, opponent clock ≥ 60s)
and run the lmcos tree-building pipeline (Lc0 iterative expansion) on them, exactly
as was done for `sampled_root_fens_2023.txt`.

**Feasibility:** High — but expensive. The tree building pipeline already exists
(`generate_dataset_shard.slurm`, `build_tree.py`). Needs SLURM time on a GPU node.

**Cost estimate:**
- Lc0 on A100, 96 expansions: ~5–15 seconds per tree (rough estimate; depends on
  batching and position complexity)
- 10K positions: ~15–40 hours single-threaded; with a SLURM array of 10 jobs: ~2–4 hours
- 50K positions: ~10–20 hours with 10 SLURM jobs

**Recommendation:** Start with 10K positions to prove the pipeline works end-to-end,
then scale.

**Open questions before implementing:**
1. **How many FENs?** Suggest 10K as the smoke test. What's the minimum that would
   make the human-vs-oracle correlation meaningful?
2. **Which budget?** The oracle's `starting_budget` (in Lc0 node expansions) must
   be set for each tree. Options:
   - **Fixed budget (e.g. 96):** Simple, matches existing training data format.
     Oracle_stop_step then purely reflects position difficulty, independent of
     human clock. This is the clean option.
   - **Budget ∝ clock time:** Convert `player_clock_time` (seconds) to expansions
     via some calibration (Lc0 X nodes/second × T seconds). Conceptually appealing
     but introduces an arbitrary conversion factor.
   - **Recommendation:** Fixed budget first; if this fails to correlate with RT,
     then try budget-as-clock as a follow-up.
3. **Which Lc0 weights?** Use ysagiv's existing weights
   (`tree_encoder_child_wdl_async_k1_subtree_weighted.pt`) to match the training
   distribution. Do not retrain.
4. **Position selection:** Should we select positions uniformly at random from
   `processed_moves_nonzero`, or stratify by feature (e.g. ensure a spread of
   branching factors)?

### Step 2: Compute DP-optimal stopping step

For each generated tree, call `compute_budgeted_oracle()` from
`lmcos/src/data/preprocess_mc/oracle.py` using the default `BudgetedOracleConfig`.
This gives `oracle_stop_step` — the exact same quantity the controller is trained to predict.

**Feasibility:** Easy. Code exists, no GPU needed, runs in seconds per tree.

**Note on inputs:** With `maintenance_scale=0.0` (default), `tree_sizes` do not matter.
`halt_rewards[t] = oracle_root_q_trace[t, oracle_best_move_index[t]]` (best-move WDL
at each expansion step).

### Step 3: The key plots

**Plot A:** `oracle_stop_step` vs `log(RT)` for the same positions.
- If positive correlation: oracle and humans agree on where to spend time. Supports
  quasi-optimality or at least the same sensitivity to position difficulty.
- If no correlation: the oracle's cost function (calibrated for Lc0 compute) is
  not the right normative model for human deliberation. This is itself a finding.

**Plot B:** `predicted_stop_step` (from trained GNN + MC controller) vs `oracle_stop_step`.
- Should be close to a diagonal (the model should predict what the oracle says).
- Deviations identify positions where the model fails — potentially interesting.

**Plot C:** `oracle_stop_step` vs board features (branching, material, gain_depth, toptwo).
- Compare directions with the human RT correlations already established.
- If same direction: oracle and humans are sensitive to the same notion of complexity.

**Feasibility:** Easy once trees are generated.

---

## Analysis 2: Minimal End-to-End Trainer

**Goal:** Find the smallest architecture that trains end-to-end in hours rather than days.
Do this before/independently of Analysis 1 since it unblocks iteration speed.

### Step 1: GNN architecture ablation (on existing data)

Do NOT regenerate data. Subsample the existing 191 training shards (use 10–20 shards
= ~30K–60K episodes) and train with progressively smaller GNN architectures.

**Current architecture:**
```
d_embed=128, d_message=128, n_heads=4, d_att=32, hidden_dim=256, hidden_layers=3
batch_size=18000, epochs=3
```
MC controller training: **52 seconds** for 3 epochs on 17M snapshots. Already fast.
GNN pretraining: **unknown wall time** (ysagiv ran this; need to check).

**Proposed ablation series (smoke tests):**
| Config | d_embed | d_message | n_heads | hidden_dim | hidden_layers | Expected speedup |
|--------|---------|-----------|---------|------------|---------------|-----------------|
| A (current) | 128 | 128 | 4 | 256 | 3 | 1× |
| B | 64 | 64 | 2 | 128 | 2 | ~4–8× |
| C | 32 | 32 | 1 | 64 | 1 | ~16–32× |
| D | 16 | 16 | 1 | 32 | 1 | ~64× |

Target: Config C or D reaching **≥75% exact stop step accuracy** (current: 80%) in
**≤2 hours** total (data loading + GNN pretrain + MC train).

**Open questions:**
1. What is the current GNN pretraining wall time? (Need this to calibrate the target.)
2. Is 75% stop step accuracy "good enough" — does lower accuracy meaningfully hurt
   the human-vs-oracle comparison in Analysis 1?
3. Can we skip GNN pretraining entirely and initialize from scratch with a smaller model?

**Feasibility:** High. Requires code changes to the config files and a SLURM job.
No new data needed.

### Step 2 (or 1b): Logistic regression baseline

**Can be done today. No GPU. No new data.**

Extract per-snapshot tree statistics from the packed episode shards:
- `n_nodes_so_far` (tree size at each snapshot)
- `root_branching` (number of root children)
- `best_q_so_far` (max Q over evaluated root children)
- `wdl_var_root` (variance in WDL across evaluated children)
- `remaining_budget`
- `move_ply`

Train a logistic regression: `f(tree_stats, budget) → sign(advantage)` (halt vs continue).

**Why this matters:** If a simple linear model over raw tree statistics achieves
≥80% sign accuracy, then the GNN is not adding meaningful representational power
over what's already visible in the scalar statistics. This would be strong evidence
that a minimal model is viable.

**Feasibility:** Very high. Can be implemented and run in 2–3 hours.
Script: `lmcos/analysis/lr_advantage_baseline.py` (to be written).

---

## Analysis 3: Weaker/Noisier Engine

**Do in parallel with Analysis 2, but only pursue Analysis 1 first.**
**Lower priority; only pursue if Analysis 1 shows that the oracle does NOT correlate
with human RT.**

**Motivation:** The lmcos engine (Lc0 t1-256x10, ~3000+ ELO) is far stronger than
the human players in the dataset (~2000 ELO). A position that's "obvious" to Lc0
may be genuinely hard for a human. If we match engine strength to human ELO, the
oracle's value-of-computation curves may better align with human deliberation.

### Step 1: Tree generation with weaker engine

Options, in order of implementation difficulty:
1. **Lc0 smaller network:** Use a smaller Lc0 network (e.g. t1-80x10 or smaller if available).
   Matches the existing pipeline exactly, just swap the weights file.
2. **Stockfish at lower depth:** Use Stockfish depth=3 instead of Lc0. Faster, noisier,
   more human-like evaluation errors.
3. **Stockfish at matched ELO:** Stockfish can be set to a specific ELO (via `UCI_LimitStrength`
   and `UCI_Elo` options). Set to 2000 to match the player pool.

**Open questions:**
1. Which network size of Lc0 is available on the cluster (ysagiv's weights directory)?
2. Is the goal to match player ELO exactly, or just to introduce noise?
3. If we switch to Stockfish, does the tree format change? (The oracle code expects
   WDL outputs; Stockfish provides centipawns, not native WDL.)

**Feasibility:** Medium. Tree generation script needs a minor modification to accept a
different engine config. Data volumes are the same as Analysis 1.

---

## Feasibility Summary and Recommended Order

| Analysis | Prerequisite | GPU/SLURM? | Est. time | Priority |
|----------|-------------|-----------|-----------|----------|
| 2-Step2: LR baseline | Nothing | No | 2–3 hours | **Do today** |
| 1-Step2/3: Oracle stop step on existing trees + compare with human RT | Existing filtered_shard trees | No | 2–3 hours | **Do today** |
| 2-Step1: GNN ablation | Existing data, config change | Yes (~2 hrs) | 1 day | This week |
| 1-Step1: Generate trees for 10K human FENs | Lc0 + SLURM | Yes | 2–4 hours compute | This week |
| 1-Step3: Full comparison plots | Trees + oracle + model | No | 1–2 hours | After Step1 |
| 3-Step1: Weaker engine trees | Decision on engine | Yes | 2–4 hours compute | If A1 fails |

---

## Other Opportunities Not in the Original List

**Quickest non-trivial test (no new compute, 1 hour):**
Compute oracle_stop_step on the existing 39K filtered_shard trees using
`compute_budgeted_oracle()` with default config. Plot oracle_stop_step vs board
features (branching, material, toptwo). Compare direction with human RT correlations.
This gives the directional answer without any new Lc0 runs and is the logical next
step before committing to the expensive tree generation in Analysis 1.

**Pure human descriptive paper (fallback if human-machine comparison fails):**
The 1M-position behavioral dataset with gain_depth, MQ, toptwo, and correlation
matrix is already a complete, publishable descriptive analysis. If the oracle does not
correlate with human RT, this is still a strong standalone contribution that characterizes
how strong players allocate deliberation time across board features.

---

## Clarifying Questions

Before implementing Analysis 1, I need answers to:

1. **Fixed vs clock-proportional budget?** My recommendation is fixed budget (96
   expansions) for the initial comparison. Do you agree?
2. **How many FENs for the smoke test?** I suggest 10K. Is that sufficient, or should
   we go to 50K immediately?
3. **What counts as "good enough" for Analysis 2?** Is 75% stop step accuracy acceptable
   if it means training in 2 hours instead of days?
4. **GNN pretraining time?** Do you know how long the GNN child-WDL pretraining took?
   This is the key number for estimating savings from a smaller architecture.
5. **For Analysis 3 — which Lc0 network?** Or do you prefer to swap to Stockfish entirely?

---

## What I Would Not Do

- Do not scale Analysis 1 to the full 88M positions before the 10K smoke test confirms
  the oracle correlates with human RT.
- Do not retrain the GNN from scratch before the logistic regression baseline establishes
  whether the GNN is even needed.
- Do not spend compute on Analysis 3 before Analysis 1 gives a directional answer.
