# Proposed Next Steps

**Ordering:** Analysis 0 is **done** (see `human_analytics/LAB_NOTEBOOK.md` § Analysis 0). **Analysis 1** and **Analysis 2** are both cleared to start; run A2 in parallel with A1. **Analysis 3** stays conditional on A1.
**Template per analysis:**
> **Description** (brief description, setup, decisions resolved, data needed)  
> **Rationale** (scientific/engineering rationale and goals)  
> **Expectation** (hypotheses and expected behavioral/model outcomes)  
> **Output** (deliverables: code, data, plots, tables)  
> **Procedure** (atomic steps with time breakdown)  
> **Testing** (tests to write, test scripts, verification rules)  
> **Time Estimate** (total estimated time)  
> **Additional Info / Questions for Implementation** (implementation notes, warnings, open questions, counterfactuals)

---

# Thread 2: LMCOS

This thread contains the codebase for generating trees and training the meta-control architecture, focusing on simplifying and improving the model.

## Analysis 0 — COMPLETE (2026-06-03)

Results and implementation details: **`human_analytics/LAB_NOTEBOOK.md`** (§ Analysis 0 results, § Analysis readiness).

| Sub-analysis | Script | Outcome |
|---|---|---|
| **0a** Directional oracle check | `lmcos/analysis/oracle_stop_step_features.py` | **Go for A1** — 3/4 feature directions match human RT (toptwo differs by design) |
| **0b** Minimal MC baseline | `lmcos/analysis/minimal_mc_baseline.py` | **Go for A2** — val sign accuracy **86.4%** (threshold 80%; GNN+MC 90.1%) |

Tests: `lmcos/tests/test_oracle_stop_step_features.py`, `test_minimal_mc_baseline.py` (25/25 pass).

---

# Thread 1: Human Data Analytics

This thread focuses on empirical evaluation of human behavioral data and testing cognitive models (like entropy stopping) against it.

## Analysis 1 — Fair comparison: oracle on human board positions

**Readiness (2026-06-04):** A0a passed. FEN export + `build_tree` smoke test are the first actions; `human_oracle_comparison.py` not written yet.

**Description**  
Generate Lc0 oracle trees on board positions from the human behavioral dataset (`personal.db`), compute `oracle_stop_step`, and compare directly with `log(RT)` for the same positions.
*   **Data needed:**
    *   FENs from `processed_moves_nonzero` in `personal.db` (ply 15–75, opp_clock ≥ 60s)
    *   Lc0 binary at `/home/hl4291/chess_analysis/../stockfish` (already on cluster) and ysagiv's weights: `tree_encoder_child_wdl_async_k1_subtree_weighted.pt`
    *   `lmcos/src/data/build_tree.py` — the tree generation pipeline
    *   `lmcos/src/data/preprocess_mc/oracle.py` — oracle computation
*   **Decisions resolved:**
    *   Budget: **fixed at 96 expansions** (matching training format). Incorporating clock time as budget would give the oracle implicit knowledge of the human behavioral variable we are trying to predict — this would poison the comparison.
    *   Lc0 weights: **use ysagiv's existing weights** — do not retrain.
    *   Position selection: **uniform random** from `processed_moves_nonzero`.

**Rationale**  
This is the core scientific comparison. A0a and A0b are prerequisites that de-risk this analysis. If A0a shows a directional match and A0b shows the LR baseline is insufficient, A1 generates the ground-truth data for the comparison.

**Expectation**  
We expect that `oracle_stop_step` will positively correlate with human `log(RT)` for the same positions, confirming that normative MCTS computation allocation directionally aligns with human cognitive effort.

**Output**  
- Generated trees: `/scratch/gpfs/GRIFFITHS/hl4291/tmp/human_trees/` (1K then 10K .pt files)
- Script for feature+oracle extraction: `lmcos/analysis/human_oracle_comparison.py`
- Three plots: (A) oracle_stop_step vs log(RT), (B) oracle_stop_step vs board features, (C) predicted_stop_step (GNN+MC) vs oracle_stop_step

**Procedure**  
*Step 1: Extract human FENs*
1. [10 min] SQL: sample 1K FENs from `processed_moves_nonzero` with `move_ply BETWEEN 15 AND 75` and `opponent_clock_time >= 60`, joined with `moves` for `move_uci`, `player_clock_time`, `move_time`. Save as text file in same format as `sampled_root_fens_2023.txt`. **[Status: Complete — exported 644 valid FENs to `/scratch/gpfs/GRIFFITHS/hl4291/tmp/human_fens_1k.txt`]**
2. [10 min] Verify the FEN format is compatible with `build_tree.py` (the script expects 6-field full FENs — `processed_moves_nonzero.fen` is 4-field; need `halfmove` and `fullmove`). Fix if needed. **[Status: Complete — formatting verified and loads correctly]**

*Step 2: Generate 1K trees (smoke test)*
3. [5 min] Set up SLURM job using `generate_dataset_shard.slurm` template, pointing at the human FEN file and ysagiv's Lc0 weights. Target budget: 96 expansions. **[Status: Wrapper fixed to support VENV_DIR; config human_trees_1k_smoke.yaml updated to resume: true]**
4. [5–30 min compute] Run on A100. **Must complete within ~15 min for 1K positions** — if slower, there is a batching problem that needs fixing before scaling to 10K. **[Status: Running on Della as Job ID 9215248 since 14:19 PM, June 4]**
5. [15 min] Verify output trees have same format as filtered_shard trees (contains `oracle_root_q_trace`, `oracle_best_move_index`, `root_position_spec`).

*Step 3: Scale to 10K*
6. [10 min setup + ~1–2 h compute] If Step 2 is fast: Scale using a SLURM job array or the `submit_generate_dataset_shards.py` wrapper script. 
   - *Job Array Method:* Add `#SBATCH --array=0-9` and map `$SLURM_ARRAY_TASK_ID` to range offsets via `--start-index` and `--end-index` (e.g. `START=$((SLURM_ARRAY_TASK_ID * 1000))` and `END=$((START + 1000))`) to process 10 shards in parallel.
   - *Orchestrator Method:* Run `./slurm/1_preprocess_data/submit_generate_dataset.sh --config slurm/configs/1_preprocess_data/human_trees_10k.yaml --shard-size 1000 --submit` to submit 10 parallel shard jobs.

*Step 4: Compute oracle_stop_step and join with behavioral data*
7. [30 min] Run `compute_budgeted_oracle()` on all generated trees → `oracle_stop_step` per position.
8. [20 min] Join on FEN (or gid + move_ply) with `personal.db` to recover `log(RT)`, `player_clock_time`, `n_possible_moves`, `move_ply`.

*Step 5: Key plots*
9. [45 min] Plot A: `oracle_stop_step` vs `log(RT)` — scatter + binned trend.
10. [30 min] Plot B: `oracle_stop_step` vs board features — compare r-values with human RT r-values side by side.
11. [30 min] Plot C: load trained GNN+MC checkpoint, run inference on generated trees → `predicted_stop_step` vs `oracle_stop_step`.

**Testing**  
*   **Tests to write** (`lmcos/tests/test_human_oracle_comparison.py`):
    *   Generated tree `root_position_spec` matches the input FEN exactly — catch pipeline routing bugs
    *   `oracle_root_q_trace` shape = [96, n_legal_moves] for every tree — catch budget or branching mismatches
    *   `oracle_stop_step` ∈ [0, 96] for all generated trees
    *   FEN join recovers correct RT: spot-check 10 positions where gid+move_ply are known
    *   All input FENs are valid chess positions — `chess.Board(fen).is_valid()` = True for all
    *   No duplicate FENs in the sample (dedup assertion)
    *   `log(RT)` and `oracle_stop_step` are defined for the same set of positions — no join mismatches producing NaN pairs

**Time Estimate**  
~4–6 hours (dominated by SLURM wait time)

**Additional Info / Questions for Implementation**  
*   **FEN format mismatch:** `processed_moves_nonzero.fen` is a 4-field FEN (no halfmove/fullmove). `build_tree.py` may expect a 6-field full FEN. Check the pipeline input format before submitting the SLURM job — this could be the main wiring issue.
*   **Open question: how does build_tree.py accept FEN input?** The existing pipeline reads from a text file of FENs (one per line). Confirm this is how `generate_dataset_shard.slurm` is configured.
*   **Warning:** The generated trees will not have human behavioral metadata (RT, clock). The join step (Step 8) requires matching on FEN string. FEN strings are position-unique only if castling rights and en passant are included — confirm the join key.
*   **Counterfactual:** If oracle_stop_step does NOT correlate with log(RT): proceed to Analysis 3 (weaker engine). Also update the slides to present the 4-case interpretation.
**Status / Results (2026-06-04):**
*   **Job ID:** `9215248` (submitted to Della `gputest` partition, currently `RUNNING`).
*   **Progress:** Resumed from the 390 generated trees in `/scratch/gpfs/GRIFFITHS/hl4291/tmp/human_trees_1k/` by setting `resume: true`.
*   **Discussion:** This smoke test verifies the pipeline's FEN formatting compatibility (6-field FENs successfully routed and read by Lc0 search). Once completed, `human_oracle_comparison.py` will run to verify that the oracle outputs are consistent before scaling up to 10K.

---

## Analysis 2 — Tree-Statistic Summary (parallel, HIGH PRIORITY)

**Readiness (2026-06-04):** A0b passed, but strategy has pivoted. Old tiny-GNN approach abandoned and job cancelled.

**Description**  
Replacing the GNN head with a coarser / simpler tree-statistic summary to obviate the training of the GNN entirely. Instead of training a GNN (even a tiny one), we extract summary statistics directly from the tree structure (e.g. node counts, depth distributions, value aggregates) and feed these to the MC controller.
*   **Data needed:**
    *   Existing packed training shards at `/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/controller_packed_combined_nomaint_no_xaba/train/` (191 shards) — use 10–20 shards (5–10%) as the ablation dataset
    *   Current config YAMLs at `lmcos/slurm/configs/4_supervised_controller/`
    *   `lmcos/src/train/gnn_pretrain.py` and `lmcos/src/train/controller_train.py`
*   **Decisions resolved:**
    *   Success metric: **GNN loss** (not stop-step accuracy). Stop-step accuracy is a downstream metric of the MC controller; GNN loss directly measures whether the encoder is learning useful representations.
    *   Target: acceptable GNN loss within ≤2 hours total training on a single GPU.

**Rationale**  
GNN pretraining took ~1+ days (ysagiv's run). This is the bottleneck for iteration speed. If a smaller GNN trained from scratch achieves acceptable GNN loss, pretraining is unnecessary and the pipeline becomes fast enough to iterate rapidly. This is the most important analysis for unblocking future work regardless of A1 outcomes.

**Expectation**  
A smaller GNN (e.g. 16-dim embedding, 1 head, 1 layer) trained jointly/online from scratch will converge to an acceptable loss comparable to the larger pretrained model, allowing us to bypass the slow 2-stage pretraining pipeline.

**Output**  
- Ablation table: GNN loss vs architecture size (4 configs × pretrained vs scratch)
- Script: modified training config YAMLs + `lmcos/analysis/run_ablation.sh`
- Decision: minimum config that achieves acceptable GNN loss from scratch

**Procedure**  
*Step 1 (burning question): train from scratch with smallest config*
1. [15 min] Create new config YAML `subtree_weighting_root_scratch_D.yaml`: d_embed=16, d_message=16, n_heads=1, hidden_dim=32, hidden_layers=1, **unfreeze_encoder=true** (train GNN jointly with MC, no pre-materialization step). Subsample 10 training shards.
2. [5 min] Key code change: when `unfreeze_encoder=true` and no pretrained checkpoint, re-compute GNN embeddings online each step (don't use pre-materialized z_root cache). Verify this code path exists in `controller_train.py` — if not, it needs to be added.
3. [~30 min–2 h compute] Run training. Track GNN loss curve. If loss converges in <2 h: success.
4. [20 min] Compare GNN loss to pretrained baseline on the same 10 shards.

*Step 2: ablation series if Config D insufficient*
5. [30 min each] Repeat with Config C, B in order until acceptable loss.

*Step 3: LR baseline comparison (see Analysis 0b)*
6. Use LR baseline results to anchor what "acceptable" GNN loss means behaviorally.

**Testing**  
*   **Tests to write** (`lmcos/tests/test_minimal_model.py`):
    *   Config D forward pass output shape = `(batch, 1)` — same interface as Config A
    *   Parameter count: Config D has <1/64 of Config A's parameters — verify architecture is actually smaller, not a config bug
    *   GNN loss decreases over first 100 training steps — confirm learning is happening and not diverging
    *   `unfreeze_encoder=true` actually modifies GNN parameters: assert param tensors differ before vs after one gradient step
    *   MC sign accuracy > 60% after 1 epoch on 10 shards — better than random chance
    *   Training reproducible: two runs with same seed → GNN loss within 1%
    *   No NaN losses at any training step — catch numerical instability from small embedding dimensions

**Time Estimate**  
~4–8 hours (including SLURM wait)

**Additional Info / Questions for Implementation**  
*   **Critical code question:** Does `controller_train.py` support joint GNN+MC training from scratch, or does it always expect pre-materialized z_root embeddings? If it requires pre-materialized embeddings, the flow is: (1) train tiny GNN from scratch on child-WDL prediction (gnn_pretrain.py, maybe 30 min), (2) materialize z_root with tiny GNN, (3) train MC on materialized embeddings. Option (1) may be viable even with the current two-stage pipeline if the tiny GNN pretrains fast.
*   **Open question:** what GNN loss does the pretrained 128-dim model achieve? Need this baseline number to know what "acceptable" means for a smaller model.
*   **Warning:** The MC controller (52 seconds) uses pre-materialized embeddings. If we move to joint training, each training step requires a GNN forward pass — much slower per step. With a 16-dim GNN this may still be fast; with a 128-dim GNN it would not.
*   **Background:** The current two-stage design (pretrain GNN → materialize → train MC) was chosen to make MC training I/O-light and GPU-bound on the MLP. This design assumed the GNN was slow. With a 16-dim GNN, joint training may be faster than the two-stage pipeline overall.

**Status / Results (2026-06-04):**
*   **Job ID:** `9215254` (submitted to Della `gputest` partition, currently `RUNNING`).
*   **Progress:** Training has started successfully:
    `[child-wdl-pretrain] epoch=1/5 phase=train batch=26/63 seen_examples=832 seen_edges=3072064 total_loss=1.065431 target_entropy=0.609526 loss_gap=0.455905`
*   **Discussion:** This is testing whether a tiny GNN (embedding dimension 16, 1 layer) trained from scratch can achieve an acceptable loss gap quickly (within 2 hours). If successful, GNN pretraining can be skipped in future iterations, dramatically speeding up the modeling cycle.

---

## Analysis 3 — Weaker engine (conditional, on hold)

**Description**  
Regenerate oracle trees using Stockfish at ELO=2000 (matching the human player pool), recompute oracle_stop_step, and check whether alignment with human RT improves.
*   **Data needed:**
    *   Stockfish binary (already at `/home/hl4291/stockfish/src/stockfish` — SF14)
    *   Same 10K human FENs from Analysis 1 Step 1
    *   `build_tree.py` — needs modification to accept Stockfish as the engine
*   **Decisions resolved:**
    *   Use **Stockfish with `UCI_LimitStrength=true, UCI_Elo=2000`** (not a smaller Lc0).
    *   Stockfish is cheaper (CPU), faster, and avoids the need to find a smaller Lc0 network.
    *   **Format note:** Stockfish outputs centipawns, not WDL. `halt_rewards` must be converted — use logistic regression: `pwin = 1 / (1 + exp(-cp / 400))` where 400 is the standard WDL calibration constant (consistent with Russek et al.).

**Rationale**  
Lc0 at ~3000+ ELO "sees through" most positions immediately. The oracle's cost function is calibrated for Lc0's search — not for the deliberation effort of a 2000-ELO player. Reducing engine strength may make the oracle's "worth thinking about" signal more aligned with human cognitive difficulty.

**Expectation**  
We expect that an engine matching human strength (SF ELO 2000) will yield an `oracle_stop_step` that correlates more strongly with human RT than the full-strength Lc0 oracle, particularly by matching human difficulty on tactical puzzles/blunders.

**Output**  
- New trees: `/scratch/gpfs/GRIFFITHS/hl4291/tmp/human_trees_sf2000/`
- Metric: r(oracle_stop_step_SF2000, log RT) vs r(oracle_stop_step_Lc0, log RT)
- Figure: side-by-side correlation matrices for Lc0 oracle vs SF2000 oracle

**Procedure**  
1. [1 h] Modify `build_tree.py` to accept Stockfish as an engine option, with configurable ELO. Add centipawn → WDL conversion for halt_rewards.
2. [30 min setup + 1–2 h compute] Run tree generation on the same 10K human FENs using SF at ELO=2000.
3. [30 min] Compute oracle_stop_step on new trees.
4. [30 min] Compare correlations with human RT and with Analysis 1 results.

**Testing**  
*   **Tests to write** (`lmcos/tests/test_sf2000_oracle.py`):
    *   Centipawn → WDL conversion is monotone: `pwin(−1000) < pwin(0) < pwin(+1000)`
    *   `pwin(cp=0) ≈ 0.5` within ±0.01 — drawn position equals probability
    *   `pwin(cp=+1000) > 0.95` — decisive winning advantage
    *   `oracle_stop_step` ∈ [0, 96] for all SF2000 trees
    *   SF2000 trees have the same key structure as Lc0 trees — format compatibility check before running oracle
    *   Stockfish at ELO=2000 fails on a known puzzle where SF14 full-strength succeeds — confirm limiter is active
    *   `r(oracle_stop_step_SF2000, oracle_stop_step_Lc0)` is in (0.1, 0.9) — they should partially agree but not perfectly

**Time Estimate**  
~4–6 hours (including code changes)

**Additional Info / Questions for Implementation**  
*   **Centipawn → WDL calibration:** The constant 400 gives approximately correct WDL for Stockfish. For SF14 specifically, the calibration may differ — check the SF source for the exact `NormalizeToPawnValue` constant.
*   **Open question:** should we also retrain the GNN + MC with SF2000 trees if alignment improves? This would produce a fully human-aligned normative agent — potentially the paper's main contribution. This would require significant compute.
*   **Counterfactual:** If SF2000 oracle also fails to align with human RT, the mismatch is fundamental (human deliberation is not captured by any engine's VOC) rather than an engine-strength artifact. This strengthens the case for two separate papers.

**Status / Results (2026-06-04):**
*   **Status:** **On hold** (conditional on A1 results).
*   **Discussion:** If A1 (high-strength Lc0 oracle) shows poor correlation with human RT (possibly due to Lc0 seeing through the positions instantly), we will activate A3. This will query Stockfish ELO 2000 on the same human positions and convert CP to WDL win probabilities using the logistic mapping to check if a human-matched strength engine yields better cognitive alignment.

---

## Analysis 4 — Information-theoretic stopping (entropy VoI)

**Description**  
Test an information-theoretic stopping rule (Rule B) based on policy entropy reduction over root choices using Stockfish multi-depth search traces generated on positions from the human behavioral dataset (`personal.db`).
*   **Data needed:**
    *   Human positions (FENs, log RT, branching factor) sampled from `processed_moves_nonzero` in `personal.db` (ply 15–75, opp_clock ≥ 60s).
    *   Stockfish binary at `/home/hl4291/stockfish/src/stockfish` (SF14).
*   **Formulation details:**
    *   For a sampled position, run Stockfish multi-depth analysis for depths $d \in [1, 8]$ with `multipv` equal to the number of legal moves (or a large enough K to cover the candidate set).
    *   Convert centipawn scores ($cp$) to win probabilities ($Q$) using the logistic function: $Q_{d, k} = 1 / (1 + \exp(-cp / 400))$.
    *   We define two candidate approaches to compute the policy $P_k(d)$ and its Shannon entropy $H(d)$:
        1. **Approach A (Subset Softmax):** Softmax is computed only over the top-K moves returned by the engine: $\text{Eval}_d = \{k : Q_{d, k} \text{ evaluated}\}$.
        2. **Approach B (Neutral Imputation):** Softmax is computed over all $N$ legal moves, with any unevaluated/low-ranked moves imputed to a neutral win probability of $0.5$.

**Rationale**  
To evaluate whether a normative stopping rule aligns with human deliberation, we must test it on the exact positions from the human behavioral dataset. Stockfish is the engine used for the human behavioral analysis. By running Stockfish at multiple depths (iterative deepening), we obtain search snapshots at each depth step. We stop when the marginal reduction in policy entropy (information gain) drops below a threshold $\theta$.

**Expectation**  
We expect that the entropy-based stopping depth $d^*$ will have a strong positive correlation with both the branching factor ($r(d^*, \text{branching}) > 0$) and the human response time ($r(d^*, \text{log RT}) > 0$), explaining the branching-RT correlation mechanistically. We expect Approach B (neutral imputation) to be more stable than Approach A.

**Output**  
- Script: `human_analytics/entropy_voi_analysis.py` (running inline Stockfish queries on human positions).
- Plots: `human_analytics/figures/entropy_voi_vs_human_rt.png` (Pearson correlation comparing $d^*$ vs branching, material, gain_depth, and toptwo against human RT) and `human_analytics/figures/entropy_voi_trajectories.png` (example trajectories of $H(d)$ and $IG(d)$ over depth steps).
- A table comparing Pearson $r(d^*, \text{branching})$ and $r(d^*, \text{log RT})$ across different thresholds $\theta \in [0.0, 0.001, 0.005, 0.01, 0.05, 0.1]$ and $\beta = 1.0$.

**Procedure**  
*Phase 1: Data Generation (Multi-Depth Snapshot Extraction & Caching)*
1. [25 min] In `human_analytics/entropy_voi_analysis.py`, write `query_stockfish_multidepth(board, max_depth)` using `chess.engine.SimpleEngine`:
   - Run Stockfish analysis up to `max_depth` (default: 8).
   - For each depth $d \in [1, \text{max\_depth}]$, query the evaluation of all legal moves (using `multipv=len(legal_moves)`).
   - Convert centipawn scores to win probabilities: $Q_{d, k} = 1 / (1 + \exp(-cp / 400))$.
   - Return a Q-value matrix of shape `(max_depth, n_legal_moves)`.
2. [15 min] Set up data sampling to load 1,000 moves from `processed_moves_nonzero` in `personal.db` (joined with `moves` table to get UCI, RT, etc.).
3. [30 min] Execute the data generation script locally using Python's multiprocessing pool to query Stockfish in parallel. Cache the generated dataset (Q-value matrices paired with FEN, RT, and branching factor) to a file (e.g., `/scratch/gpfs/GRIFFITHS/hl4291/tmp/sf_multidepth_traces_1k.pkl`).

*Phase 2: Modeling & Optimization (Entropy Calculation, Sweeps, and Plotting)*
4. [20 min] Write the offline analysis functions to load the cached traces:
   - For **Approach B (Neutral Imputation)**: For any move not returned or having a score of 0, impute $Q_{d, k} = 0.5$.
   - Compute policy $P_k(d) = \text{softmax}(\beta \cdot Q_d)$.
   - Compute Shannon entropy $H(d) = -\sum_k P_k(d) \log P_k(d)$.
   - The stopping depth $d^*$ is the first $d \ge 1$ where $H(d-1) - H(d) < \theta$.
5. [15 min] Run sweeps over $\theta \in [0.0, 0.001, 0.005, 0.01, 0.05, 0.1]$ and $\beta = 1.0$. Calculate Pearson correlations with branching factor and human log(RT) and print a markdown summary table.
6. [15 min] Generate `human_analytics/figures/entropy_voi_trajectories.png` showing $H(d)$ and $IG(d)$ over depth steps for 3 selected positions.
7. [15 min] Generate `human_analytics/figures/entropy_voi_vs_human_rt.png` showing the side-by-side Pearson correlations of the best entropy stopping depth $d^*$ vs human log(RT) across the key board features.

**Status / Results (2026-06-04):**
*   **Status:** **10K Scale-Up Completed**.
*   **Progress:** The CPU-only SLURM script `entropy_voi_10k.slurm` was executed successfully, generating traces for 6,494 available positions. The offline modeling sweep was run on the full dataset and plots/tables were generated in `human_analytics/figures/`.
*   **Timing Analysis (10,000 Moves):** The data generation of 6,494 valid multi-depth traces completed in 238.6 seconds (0.037s/position) using 16 CPU workers (~4 minutes total). Scaling A4 without GPU resources is highly efficient.
*   **Discussion (Alignment with Expectation):** With the larger 6,500-move sample, the initial strong correlations disappeared. Using $\beta = 1.0$ and $\theta = 0.001$, the correlation with the `branching factor` remains somewhat positive ($r \approx 0.101$), but the correlation with `log(RT)` dropped to near-zero ($r \approx 0.014$). At $\theta = 0.005$, the correlation with `log(RT)` is slightly negative ($r \approx -0.005$). This is an important finding: it suggests that while entropy reduction might correlate with the objective branching factor, this specific normative stopping rule does not strongly explain human deliberation time (RT). We may need to perform a more exhaustive sweep over the temperature $\beta$, or this could strengthen the case that standard engine evaluations do not align well with human cognitive effort (supporting Analysis 3).

**Testing**  
*   **Tests to write** (`human_analytics/tests/test_entropy_voi.py`):
    *   Centipawn to WDL conversion is monotone and returns $0.5$ for $cp=0$.
    *   Entropy is always non-negative: $H(d) \ge 0$.
    *   Uniform distribution entropy is exactly $\log N$.
    *   The stopping depth $d^*$ is within $[0, \text{max\_depth}]$.
    *   Correct handling of checkmate/drawn position evaluations.

**Time Estimate**  
~3 hours

**Additional Info / Questions for Implementation**  
*   For the first smoke test we don't need the temperature grid sweep for now -- focus on just $\beta = 1.0$ for now.
*   By running Stockfish depth-based evaluations directly on human positions, we sidestep the need for GPU tree generation (which would take hours on SLURM). Stockfish is fast enough to run locally on a CPU multi-processing pool in minutes.

---

## Summary table

| Analysis | Prerequisite | Compute | Time | Status |
|----------|-------------|---------|------|--------|
| **0a** Oracle on existing trees | — | CPU | ~2 h | **Done** — go A1 |
| **0b** Minimal MC baseline | — | CPU | ~2 h | **Done** — go A2 |
| **1** Human trees + oracle + comparison | A0a ✓ | A100 SLURM | 4–6 h | **Ready** — smoke test first |
| **2** Minimal model / skip pretraining | A0b ✓ | 1× GPU | 4–8 h | **Ready** — need Config D YAML |
| **3** SF2000 weaker engine | A1 mismatch | CPU | 4–6 h | On hold |
| **4** Entropy VoI stopping | — | CPU | ~3 h | **Ready** |


## What I would not do

- Scale A1 beyond 10K positions before the 1K smoke test confirms `build_tree.py` accepts
  human FENs and completes in minutes (not hours).
- Run large-scale GNN pretraining before A2 scratch/ablation establishes whether it is necessary (A0b already suggests it is not for sign accuracy).
- Start A3 before A1 gives a directional answer on matched human positions.
- Build a new end-to-end training pipeline before A2 establishes the minimum viable
  architecture on existing data.
