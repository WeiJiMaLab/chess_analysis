# Proposed Next Steps

**Ordering:** A0 gates whether A1–A3 are worth doing. A2 runs in parallel regardless.
A3 is conditional on A1 finding a mismatch.

**Template per analysis:**
> brief description · rationale · data · expected output · procedure (atomic tasks with times) · total estimate · **tests** · notes/warnings/open questions

---

## Analysis 0a — Directional oracle check on existing trees

**Brief:** Compute DP-optimal `oracle_stop_step` on existing lmcos tree data and
correlate with the same board features that predict human RT. No new data generation.

**Rationale / goal:**
Before committing to expensive tree generation (Analysis 1), check whether the oracle's
stopping behaviour tracks position complexity in the same direction as human RT.
A directional match (same signs) gives strong motivation for A1. A mismatch raises
the question of whether A1 is worth doing at all, and may indicate the oracle cost
function is not the right normative model for human deliberation.

**Data needed:**
- 39K filtered_shard tree files at
  `/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/generated_trees_combined/`
  — each file contains `oracle_root_q_trace`, `oracle_best_move_index`, `root_position_spec`
- `lmcos/src/data/preprocess_mc/oracle.py` — `compute_budgeted_oracle()`, `BudgetedOracleConfig`

**Expected output:**
- Script: `lmcos/analysis/oracle_stop_step_features.py`
- Figure 1: bar chart comparing r(oracle_stop_step, feature) vs r(log RT, feature) for branching, material, gain_depth_equiv, toptwo_equiv — also reporting r² for each
- Figure 2: extended correlation matrix that adds `oracle_stop_step` as a new row/column alongside the existing human features (branching, material, VOC, MQ, toptwo, log RT)
- Go/no-go signal for Analysis 1

**Procedure:**
1. [15 min] Load a sample of 5K filtered_shard trees (already fast from prior experiments)
2. [10 min] For each tree: construct `halt_rewards[t] = oracle_root_q_trace[t, oracle_best_move_index[t]]`
   (best-move WDL at each expansion step). `tree_sizes` can be `[1]*T` since `maintenance_scale=0`.
3. [10 min] Call `compute_budgeted_oracle(halt_rewards, [1]*T, starting_budget=60, config=BudgetedOracleConfig())`
   → `policy.optimal_stop_step`
4. [10 min] Extract position features from root FEN: `n_possible_moves`, `n_self_pieces_exc_pawns`,
   `move_ply` (via `chess.Board()`), plus `toptwo_equiv` and `gain_depth_equiv` from `oracle_root_q_trace`
5. [20 min] Compute Pearson r for each feature vs `oracle_stop_step`
6. [30 min] Plot comparison table alongside the human RT correlations (from `correlation_matrix.png`)

**Total estimate:** ~1.5–2 hours

**Tests to write** (`lmcos/tests/test_oracle_stop_step_features.py`):
- `oracle_stop_step` ∈ [0, budget] for every tree — catch out-of-range values
- `halt_rewards` are all in [0, 1] (WDL range) — catch feature indexing bugs
- Starting position FEN (known) → `n_possible_moves = 20`, `n_self_pieces_exc_pawns = 8`
- Determinism: same tree loaded twice gives identical `oracle_stop_step`
- Trivial case: constant `halt_rewards` → `oracle_stop_step = 0` (no gain from continuing)
- Monotone case: strictly increasing `halt_rewards`, very low cost → `oracle_stop_step` near `budget`
- Reverse case: time cost > max possible halt reward improvement → `oracle_stop_step = 0` always (always halt immediately — cost dominates)
- Correctness check: for episodes present in both the filtered_shard trees AND the packed shard `oracle_stop_steps`, compare our computed value against the ground-truth label from the packed shard — they should agree within ±1 step (accounting for the WDL vs centipawn approximation)
- Correlation signs reproducible: two different random seeds give the same sign pattern for each feature

**Implementation notes / warnings / open questions:**
- Use budget values drawn from the existing `DEFAULT_BUDGET_BUCKETS` (scramble: 1–3, medium-small: 4–10, medium-large: 11–25, large: 26–60, very-large: 61–120) rather than arbitrary values. The cost function is calibrated to these ranges — testing outside them may give invalid oracle estimates. For A0a, use one representative value per bucket (e.g. 2, 7, 18, 43, 90) rather than arbitrary round numbers.
- **Counterfactual:** If r(oracle_stop_step, branching) is negative here but positive for
  humans, it does not conclusively mean A1 will fail — the data sources differ (lmcos trees
  from diverse time controls, ELO 1800–2600 vs human 10+0 ≥2000). A directional mismatch
  in A0a is suggestive but not definitive.
- **Warning:** `oracle_root_q_trace` values are WDL (0–1), while the actual oracle during
  packing used centipawn-derived values (from the encoded `value` feature). The WDL-based
  halt rewards may give slightly different oracle_stop_step values than the packed shard
  `oracle_stop_steps`. This is an approximation — document it clearly.
- Note: these trees are lmcos training positions (sampled from 2023 Lichess, diverse time
  controls) and are NOT the same positions as the human behavioral dataset. The comparison
  is directional only.

---

## Analysis 0b — Minimal MC baseline (tree-stats → MLP → halt/continue)

**Brief:** Replace the GNN encoder with raw tree scalar statistics and test whether
a small MLP (matching the MC architecture) can predict halt/continue as well as the
full GNN + MC pipeline. This is not a generic "linear model" test — it specifically
mirrors the MC's architecture (same MLP head, same training target) but replaces
the z_root embedding with a hand-crafted feature vector from the tree.

**Rationale / goal:**
If this minimal baseline achieves ≥80% sign accuracy (current GNN+MC: 90%), the GNN
encoder is not adding meaningful representational capacity over what is directly visible
in raw tree statistics. This makes "skip GNN pretraining" (Analysis 2) the obvious path
and suggests a much simpler model may be sufficient. Also run timing tests: how fast is
the minimal baseline vs the full GNN+MC at inference? If it's 10× faster and nearly as
accurate, the speedup may justify the accuracy loss.

**Data needed:**
- Packed validation episode shards at
  `/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/controller_packed_combined_nomaint_no_xaba/validation/`
  (11 shards, ~35K episodes)
- Fields needed per snapshot: `target_advantages`, `step_node_cutoffs` (per-snapshot
  node count), `oracle_stop_steps`, `starting_budgets`, `node_features` (root node)

**Expected output:**
- Script: `lmcos/analysis/lr_advantage_baseline.py`
- Metric: sign accuracy of minimal baseline vs GNN+MC (90%) on held-out shards
- Timing: inference speed of minimal baseline vs GNN+MC (ms/snapshot)
- Figure: feature importance / weight magnitude plot
- Decision: if baseline ≥ 80% → GNN is likely redundant; also report timing speedup

**Procedure:**
1. [20 min] Load 8 validation shards; extract per-snapshot scalar features:
   - `n_nodes` from `step_node_cutoffs`
   - `root_branching` from `child_ptr` at root index
   - `best_q` = root `wdl_win` (from `node_features[root_index, 1]`)
   - `wdl_var` = root `wdl_var` (from `node_features[root_index, 4]`)
   - `remaining_budget` = `starting_budget - snapshot_index`
2. [10 min] Construct `(X, y)`: X = feature vector, y = sign(target_advantages)
3. [30 min] Train a small feedforward MLP (same depth/width as the MC head, e.g. 2 layers × 64 units)
   on 7 shards — mirroring the MC architecture but replacing z_root with the raw feature vector
4. [5 min] Evaluate sign accuracy on 1 held-out shard
5. [10 min] Time inference: measure ms/snapshot for minimal baseline vs full GNN+MC
6. [10 min] Compare both accuracy and speed against GNN+MC baseline
7. [20 min] Plot feature weight magnitudes

**Total estimate:** ~1.5–2 hours

**Tests to write** (`lmcos/tests/test_lr_advantage_baseline.py`):
- Feature matrix shape = `(n_snapshots, n_features)` with no NaNs — catch indexing bugs
- Labels are binary `{−1, +1}` only — catch sign/encoding errors
- MLP output shape = `(n_snapshots, 1)` — same as MC head
- Training accuracy > 60% on training shards — if MLP can't beat chance, features are wrong
- No data leakage: held-out shard indices never appear in training set
- Comparison validity: minimal MLP and full GNN+MC evaluated on identical held-out snapshot sets
- Timing test: minimal MLP inference < 5 ms/snapshot (should be much faster than GNN+MC)
- Reproducibility: two runs with same seed → sign accuracy within 0.1%

**Implementation notes / warnings / open questions:**
- **step_node_cutoffs** is the key field — it gives the node count in the tree at each
  snapshot. Double-check this is cumulative from step 0 (not incremental).
- If the MLP fails to reach 80%: the signal is genuinely nonlinear or requires richer tree structure than the scalar statistics capture. Next step would be a slightly deeper or wider MLP before concluding the GNN is necessary — stick to simple feedforward architectures for now.
- **Warning:** `move_ply` is not directly in the packed shard. Options: (a) skip it,
  (b) reconstruct from the source tree path's filename (contains a tree index that
  might encode ply), (c) use `root_wdl_var` as a proxy for position complexity instead.
- **Background:** The current model achieves 90% sign accuracy and 80% exact stop-step
  accuracy. Sign accuracy is the right metric for A0b — exact stop-step is a harder
  target and less relevant for the LR baseline.

---

## Analysis 1 — Fair comparison: oracle on human board positions

**Brief:** Generate Lc0 oracle trees on board positions from the human behavioral
dataset (`personal.db`), compute `oracle_stop_step`, and compare directly with
`log(RT)` for the same positions.

**Rationale / goal:**
This is the core scientific comparison. A0a and A0b are prerequisites that de-risk
this analysis. If A0a shows a directional match and A0b shows the LR baseline is
insufficient, A1 generates the ground-truth data for the comparison.

**Data needed:**
- FENs from `processed_moves_nonzero` in `personal.db` (ply 15–75, opp_clock ≥ 60s)
- Lc0 binary at `/home/hl4291/chess_analysis/../stockfish` (already on cluster)
  and ysagiv's weights: `tree_encoder_child_wdl_async_k1_subtree_weighted.pt`
- `lmcos/src/data/build_tree.py` — the tree generation pipeline
- `lmcos/src/data/preprocess_mc/oracle.py` — oracle computation

**Decisions resolved:**
- Budget: **fixed at 96 expansions** (matching training format). Incorporating clock
  time as budget would give the oracle implicit knowledge of the human behavioral
  variable we are trying to predict — this would poison the comparison.
- Lc0 weights: **use ysagiv's existing weights** — do not retrain.
- Position selection: **uniform random** from `processed_moves_nonzero`.

**Expected output:**
- Generated trees: `/scratch/gpfs/GRIFFITHS/hl4291/tmp/human_trees/` (1K then 10K .pt files)
- Script for feature+oracle extraction: `lmcos/analysis/human_oracle_comparison.py`
- Three plots: (A) oracle_stop_step vs log(RT), (B) oracle_stop_step vs board features,
  (C) predicted_stop_step (GNN+MC) vs oracle_stop_step

**Procedure:**

*Step 1: Extract human FENs*
1. [10 min] SQL: sample 1K FENs from `processed_moves_nonzero` with `move_ply BETWEEN 15 AND 75`
   and `opponent_clock_time >= 60`, joined with `moves` for `move_uci`, `player_clock_time`,
   `move_time`. Save as text file in same format as `sampled_root_fens_2023.txt`.
2. [10 min] Verify the FEN format is compatible with `build_tree.py` (the script expects
   6-field full FENs — `processed_moves_nonzero.fen` is 4-field; need `halfmove` and `fullmove`).
   Fix if needed.

*Step 2: Generate 1K trees (smoke test)*
3. [5 min] Set up SLURM job using `generate_dataset_shard.slurm` template, pointing at
   the human FEN file and ysagiv's Lc0 weights. Target budget: 96 expansions.
4. [5–30 min compute] Run on A100. **Must complete within ~15 min for 1K positions** — if
   slower, there is a batching problem that needs fixing before scaling to 10K.
5. [15 min] Verify output trees have same format as filtered_shard trees (contains
   `oracle_root_q_trace`, `oracle_best_move_index`, `root_position_spec`).

*Step 3: Scale to 10K*
6. [10 min setup + ~1–2 h compute] If Step 2 is fast: SLURM array with 10 jobs, 1K each.

*Step 4: Compute oracle_stop_step and join with behavioral data*
7. [30 min] Run `compute_budgeted_oracle()` on all generated trees → `oracle_stop_step` per position.
8. [20 min] Join on FEN (or gid + move_ply) with `personal.db` to recover `log(RT)`,
   `player_clock_time`, `n_possible_moves`, `move_ply`.

*Step 5: Key plots*
9. [45 min] Plot A: `oracle_stop_step` vs `log(RT)` — scatter + binned trend.
10. [30 min] Plot B: `oracle_stop_step` vs board features — compare r-values with
    human RT r-values side by side.
11. [30 min] Plot C: load trained GNN+MC checkpoint, run inference on generated trees
    → `predicted_stop_step` vs `oracle_stop_step`.

**Total estimate:** ~4–6 hours (dominated by SLURM wait time)

**Tests to write** (`lmcos/tests/test_human_oracle_comparison.py`):
- Generated tree `root_position_spec` matches the input FEN exactly — catch pipeline routing bugs
- `oracle_root_q_trace` shape = [96, n_legal_moves] for every tree — catch budget or branching mismatches
- `oracle_stop_step` ∈ [0, 96] for all generated trees
- FEN join recovers correct RT: spot-check 10 positions where gid+move_ply are known
- All input FENs are valid chess positions — `chess.Board(fen).is_valid()` = True for all
- No duplicate FENs in the sample (dedup assertion)
- `log(RT)` and `oracle_stop_step` are defined for the same set of positions — no join mismatches producing NaN pairs

**Implementation notes / warnings / open questions:**
- **FEN format mismatch:** `processed_moves_nonzero.fen` is a 4-field FEN (no halfmove/fullmove).
  `build_tree.py` may expect a 6-field full FEN. Check the pipeline input format before
  submitting the SLURM job — this could be the main wiring issue.
- **Open question: how does build_tree.py accept FEN input?** The existing pipeline reads
  from a text file of FENs (one per line). Confirm this is how `generate_dataset_shard.slurm`
  is configured.
- **Warning:** The generated trees will not have human behavioral metadata (RT, clock).
  The join step (Step 8) requires matching on FEN string. FEN strings are position-unique
  only if castling rights and en passant are included — confirm the join key.
- **Counterfactual:** If oracle_stop_step does NOT correlate with log(RT): proceed to
  Analysis 3 (weaker engine). Also update the slides to present the 4-case interpretation.
- **Background:** The existing training trees were generated from sampled_root_fens_2023.txt
  which came from Lichess 2023 games with broad ELO/time-control filters. The human positions
  here come from a narrower slice (10+0, ≥2000 ELO). Distribution shift may affect results.

---

## Analysis 2 — Minimal model (parallel, HIGH PRIORITY)

**Brief:** Find the smallest architecture + training regime that runs end-to-end in
hours rather than days. The burning question is whether GNN pretraining can be skipped.

**Rationale / goal:**
GNN pretraining took ~1+ days (ysagiv's run). This is the bottleneck for iteration
speed. If a smaller GNN trained from scratch achieves acceptable GNN loss, pretraining
is unnecessary and the pipeline becomes fast enough to iterate rapidly. This is the
most important analysis for unblocking future work regardless of A1 outcomes.

**Data needed:**
- Existing packed training shards at
  `/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/controller_packed_combined_nomaint_no_xaba/train/`
  (191 shards) — use 10–20 shards (5–10%) as the ablation dataset
- Current config YAMLs at `lmcos/slurm/configs/4_supervised_controller/`
- `lmcos/src/train/gnn_pretrain.py` and `lmcos/src/train/controller_train.py`

**Decisions resolved:**
- Success metric: **GNN loss** (not stop-step accuracy). Stop-step accuracy is a
  downstream metric of the MC controller; GNN loss directly measures whether the
  encoder is learning useful representations.
- Target: acceptable GNN loss within ≤2 hours total training on a single GPU.

**Expected output:**
- Ablation table: GNN loss vs architecture size (4 configs × pretrained vs scratch)
- Script: modified training config YAMLs + `lmcos/analysis/run_ablation.sh`
- Decision: minimum config that achieves acceptable GNN loss from scratch

**Procedure:**

*Step 1 (burning question): train from scratch with smallest config*
1. [15 min] Create new config YAML `subtree_weighting_root_scratch_D.yaml`:
   d_embed=16, d_message=16, n_heads=1, hidden_dim=32, hidden_layers=1, **unfreeze_encoder=true**
   (train GNN jointly with MC, no pre-materialization step).
   Subsample 10 training shards.
2. [5 min] Key code change: when `unfreeze_encoder=true` and no pretrained checkpoint,
   re-compute GNN embeddings online each step (don't use pre-materialized z_root cache).
   Verify this code path exists in `controller_train.py` — if not, it needs to be added.
3. [~30 min–2 h compute] Run training. Track GNN loss curve. If loss converges in <2 h: success.
4. [20 min] Compare GNN loss to pretrained baseline on the same 10 shards.

*Step 2: ablation series if Config D insufficient*
5. [30 min each] Repeat with Config C, B in order until acceptable loss.

*Step 3: LR baseline comparison (see Analysis 0b)*
6. Use LR baseline results to anchor what "acceptable" GNN loss means behaviorally.

**Total estimate:** ~4–8 hours (including SLURM wait)

**Tests to write** (`lmcos/tests/test_minimal_model.py`):
- Config D forward pass output shape = `(batch, 1)` — same interface as Config A
- Parameter count: Config D has <1/64 of Config A's parameters — verify architecture is actually smaller, not a config bug
- GNN loss decreases over first 100 training steps — confirm learning is happening and not diverging
- `unfreeze_encoder=true` actually modifies GNN parameters: assert param tensors differ before vs after one gradient step
- MC sign accuracy > 60% after 1 epoch on 10 shards — better than random chance
- Training reproducible: two runs with same seed → GNN loss within 1%
- No NaN losses at any training step — catch numerical instability from small embedding dimensions

**Implementation notes / warnings / open questions:**
- **Critical code question:** Does `controller_train.py` support joint GNN+MC training
  from scratch, or does it always expect pre-materialized z_root embeddings? If it
  requires pre-materialized embeddings, the flow is:
  (1) train tiny GNN from scratch on child-WDL prediction (gnn_pretrain.py, maybe 30 min),
  (2) materialize z_root with tiny GNN, (3) train MC on materialized embeddings.
  Option (1) may be viable even with the current two-stage pipeline if the tiny GNN
  pretrains fast.
- **Open question:** what GNN loss does the pretrained 128-dim model achieve? Need this
  baseline number to know what "acceptable" means for a smaller model.
- **Warning:** The MC controller (52 seconds) uses pre-materialized embeddings. If we
  move to joint training, each training step requires a GNN forward pass — much slower
  per step. With a 16-dim GNN this may still be fast; with a 128-dim GNN it would not.
- **Background:** The current two-stage design (pretrain GNN → materialize → train MC)
  was chosen to make MC training I/O-light and GPU-bound on the MLP. This design
  assumed the GNN was slow. With a 16-dim GNN, joint training may be faster than
  the two-stage pipeline overall.

---

## Analysis 3 — Weaker engine (conditional, on hold)

**Brief:** Regenerate oracle trees using Stockfish at ELO=2000 (matching the human
player pool), recompute oracle_stop_step, and check whether alignment with human RT
improves.

**Trigger:** Only pursue if Analysis 1 shows `oracle_stop_step` does NOT correlate
with `log(RT)`, and specifically in the direction consistent with the engine being
"too strong" (Lc0 trivially resolves positions humans find hard).

**Rationale / goal:**
Lc0 at ~3000+ ELO "sees through" most positions immediately. The oracle's cost
function is calibrated for Lc0's search — not for the deliberation effort of a 2000-ELO
player. Reducing engine strength may make the oracle's "worth thinking about" signal
more aligned with human cognitive difficulty.

**Data needed:**
- Stockfish binary (already at `/home/hl4291/stockfish/src/stockfish` — SF14)
- Same 10K human FENs from Analysis 1 Step 1
- `build_tree.py` — needs modification to accept Stockfish as the engine

**Decision resolved:** Use **Stockfish with `UCI_LimitStrength=true, UCI_Elo=2000`**
(not a smaller Lc0). Rationale:
- Analysis 3 requires regenerating all trees from scratch anyway — no consistency
  requirement with existing Lc0 frozen weights
- Stockfish is cheaper (CPU), faster, and avoids the need to find a smaller Lc0 network
- **Format note:** Stockfish outputs centipawns, not WDL. `halt_rewards` must be
  converted — use logistic regression: `pwin = 1 / (1 + exp(-cp / 400))` where 400
  is the standard WDL calibration constant (consistent with Russek et al.)

**Expected output:**
- New trees: `/scratch/gpfs/GRIFFITHS/hl4291/tmp/human_trees_sf2000/`
- Metric: r(oracle_stop_step_SF2000, log RT) vs r(oracle_stop_step_Lc0, log RT)
- Figure: side-by-side correlation matrices for Lc0 oracle vs SF2000 oracle

**Procedure:**
1. [1 h] Modify `build_tree.py` to accept Stockfish as an engine option, with
   configurable ELO. Add centipawn → WDL conversion for halt_rewards.
2. [30 min setup + 1–2 h compute] Run tree generation on the same 10K human FENs
   using SF at ELO=2000.
3. [30 min] Compute oracle_stop_step on new trees.
4. [30 min] Compare correlations with human RT and with Analysis 1 results.

**Total estimate:** ~4–6 hours (including code changes)

**Tests to write** (`lmcos/tests/test_sf2000_oracle.py`):
- Centipawn → WDL conversion is monotone: `pwin(−1000) < pwin(0) < pwin(+1000)`
- `pwin(cp=0) ≈ 0.5` within ±0.01 — drawn position equals probability
- `pwin(cp=+1000) > 0.95` — decisive winning advantage
- `oracle_stop_step` ∈ [0, 96] for all SF2000 trees
- SF2000 trees have the same key structure as Lc0 trees — format compatibility check before running oracle
- Stockfish at ELO=2000 fails on a known puzzle where SF14 full-strength succeeds — confirm limiter is active
- `r(oracle_stop_step_SF2000, oracle_stop_step_Lc0)` is in (0.1, 0.9) — they should partially agree but not perfectly

**Implementation notes / warnings / open questions:**
- **Centipawn → WDL calibration:** The constant 400 gives approximately correct WDL
  for Stockfish. For SF14 specifically, the calibration may differ — check the SF source
  for the exact `NormalizeToPawnValue` constant.
- **Open question:** should we also retrain the GNN + MC with SF2000 trees if alignment
  improves? This would produce a fully human-aligned normative agent — potentially the
  paper's main contribution. This would require significant compute.
- **Counterfactual:** If SF2000 oracle also fails to align with human RT, the mismatch
  is fundamental (human deliberation is not captured by any engine's VOC) rather than
  an engine-strength artifact. This strengthens the case for two separate papers.

---

## Summary table

| Analysis | Prerequisite | Compute | Time | Status |
|----------|-------------|---------|------|--------|
| **0a** Oracle on existing trees | Nothing | CPU only | 1.5–2 h | **Do now** |
| **0b** LR baseline | Nothing | CPU only | 1.5 h | **Do now** |
| **1** Human trees + oracle + comparison | A0 ≥ directional signal | A100 SLURM | 4–6 h | After A0 |
| **2** Minimal model / skip pretraining | Nothing (existing data) | 1× GPU | 4–8 h | Parallel now |
| **3** SF2000 weaker engine | A1 mismatch confirmed | CPU | 4–6 h | On hold |

## What I would not do

- Scale A1 beyond 10K positions before the 1K smoke test confirms build_tree.py accepts
  human FENs and completes in minutes (not hours).
- Run GNN pretraining at any scale before A0b LR baseline and A2 scratch training
  establish whether pretraining is even necessary.
- Start A3 before A1 gives a directional answer.
- Build a new end-to-end training pipeline before A2 establishes the minimum viable
  architecture on existing data.
