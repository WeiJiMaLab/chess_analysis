# Human Analytics Lab Notebook

Tracks analyses, decisions, and open questions for the human chess decision-making project.  
Repository root: `/home/hl4291/chess_analysis/`  
Database: `/scratch/gpfs/GRIFFITHS/hl4291/personal.db` (DuckDB)

**Cross-repo analyses (lmcos):** Scripts under `lmcos/analysis/`, tests under `lmcos/tests/`, figures at `lmcos/analysis/figures/` (generated locally; not always committed). Pipeline/engine details: `lmcos/LAB_NOTEBOOK.md`. Planned work queue: `proposed_next_steps.md`.

---

## Dataset

**Source:** Lichess games (Oct–Dec 2023), preprocessed via `human_analytics/slurm/`.

**Filters applied at preprocessing:**
- Time control: 10+0 (600s, no increment)
- Min ELO: 2000 (both players)
- Excluded: negative move_time games, berserk games, games where a player granted extra time

**Tables in `personal.db`:**

| Table | Rows | Notes |
|---|---|---|
| `games` | 1,971,698 | Selected games |
| `moves` | 145,142,731 | Raw moves with UCI, clock, board |
| `processed_moves` | 145,142,731 | + FEN, piece counts, ply_tertiles |
| `processed_moves_nonzero` | 135,482,903 | `move_time > 0` only |

`processed_moves` adds: `n_pieces_on_board_{inc,exc}_pawns`, `n_self_pieces_exc_pawns`, `n_opp_pieces_exc_pawns`, `fen` (4-field), `ply_tertiles` (ntile(3) by move_ply).

---

## Prior Analyses (before 2026-06-02)

### Move-time dashboards (`movetime_analysis.py`, `slurm/analysis.sh`)

Six bivariate analyses of `move_time` vs. a predictor, using the `Analyzer` class (SQL-native ntile binning, poster-style plots):

| Analysis | X variable | Filter | Key finding |
|---|---|---|---|
| `clock` | `player_clock_time` | `< 600s` | RT increases strongly with clock (more time → longer think) |
| `clock_opp` | `opponent_clock_time` | `< 600s` | Weaker effect; players are partially sensitive to opponent's clock |
| `npossiblemoves` | `n_possible_moves` | `< 50` | RT increases with branching factor |
| `pieces_exc` | `n_pieces_on_board_exc_pawns` | not null | RT increases in earlier (richer) positions |
| `self_pieces_exc` | `n_self_pieces_exc_pawns` | not null | Similar to pieces_exc |
| `ply` | `move_ply` | `<= 150` | Non-monotone arc: fast openings, slower middlegame, faster endgames |

All plots in `figures/` (2×2 dashboards + quantile heatmaps where applicable).

### Move-time summary (`move_time_summary.py`)

Side-by-side histograms of `move_time` and `ln(move_time)` distributions.  
`ln(move_time)` is approximately log-normal, consistent with Weber's Law interpretation.

### Ply–instant move arc (`ply_premove.py`)

P(move_time = 0) as a function of move ply. Premoves (instant moves) are most common in the opening (book moves played instantly) and late endgame.

---

## 2026-06-02: Structural cleanup + VOC / MQ

### Structural changes

- **Tests moved** from `slurm/scripts/tests/` → `human_analytics/tests/` (updated imports, `pytest.ini` updated with `testpaths`).
- **`utils/helpers.py`**: removed unused `import dask.dataframe as dd`.
- **`move_time_summary.py`**: fixed `CREATE OR REPLACE TABLE` → `CREATE OR REPLACE TEMPORARY TABLE` for histogram bin tables (they were being persisted in `personal.db` as erroneous artifacts `_move_time_hist_bins`, `_ln_move_time_hist_bins`). Both artifact tables dropped from DB.
- **All figure PNGs removed and regenerated** via `bash human_analytics/slurm/analysis.sh` (ran cleanly, ~6 min).

### New code

#### `engine_analysis.py`

Two functions using python-chess + Stockfish or lc0:

```
move_quality(board, move, engine, depth=15) → float | None
    MQ = e_win_taken − e_win_best  (≤ 0; 0 = optimal move played)

voc(board, engine, depth_deep=15, depth_shallow=1) → float | None
    VOC = V_deep(a_deep) − V_deep(a_shallow)  (≥ 0)
    a_shallow = argmax_a V_shallow(a)   (best move at depth_shallow)
    a_deep    = argmax_a V_deep(a)      (best move at depth_deep)
```

Win probability convention: `(wins + 0.5 * draws) / total` from engine WDL, side-to-move perspective.

**Why multipv=2?**  
Avoids a separate `root_moves` fallback call for "near-optimal" moves.  
- For MQ: if the move taken is 1st or 2nd best, one multipv=2 call gives both `e_win_best` and `e_win_taken` in a single pass. For blunders (rank > 2), we fall back to `root_moves=[move]`.
- For VOC: after finding `a_shallow` at `depth_shallow`, the `depth_deep` multipv=2 call covers cases where `a_shallow` is rank 1 or 2, saving an extra call. If it's not in the top 2, we fall back to `root_moves=[a_shallow]`.
- Tradeoff: multipv=2 is ~10–20% slower per call than multipv=1, but avoids a full extra search call in most cases.

**MQ sign convention:**  
MQ ≤ 0 always. This is the negation of the standard "centipawn loss" concept, expressed in win-probability units. MQ = 0 means the optimal move was played. Can be flipped to `error = −MQ ≥ 0` for visualisation if preferred.

#### `slurm/scripts/build_pos_with_engine_eval.py`

Unified engine evaluation pipeline (replaces `script_engine_eval.py` and the earlier `voc_mq_eval.py`). Applies Russek et al. filters; computes all six quantities per position.

Subcommands:
- `eval` — SLURM shard (`--shard-id K --total-shards N`) or local smoke test (`--n-total 10000 --n-workers 10`)
- `merge` — combines output parquets → `pos_with_engine_eval` table in personal.db

#### `voc_mq_analysis.py`

Pure analysis/plotting script. Reads `pos_with_engine_eval` from personal.db (no engine calls). Produces figures in `figures/mq_voc/`:
- `voc_hist.png` — distribution of VOC
- `mq_hist.png` — distribution of MQ
- `toptwo_hist.png` — distribution of top-2 gap
- `rt_vs_voc.png` — log(RT) vs VOC with trend
- `mq_vs_clock.png` — MQ vs player clock with trend

#### `timing_comparison.py`

Times both engines at depths [1, 5, 10] on a small middlegame sample and extrapolates to 10K / 100K positions.

#### `human_analytics/tests/test_engine_analysis.py`

9 unit tests for MQ and VOC contracts (sign, non-positivity, blunder detection, forced-move VOC=0, tactical VOC>0). All pass against Stockfish SF14 after the PositionEval refactor.

---

## VOC / MQ: Design decisions and open questions

### Comparison with Russek et al. (2022)

**Reference:** "Time Spent Thinking in Online Chess Reflects the Value of Computation" (Russek, Acerbi, Ma, Shadmehr, Bhui — *Cognitive Science*, 2025 online).  
519M moves, 12.49M games; Stockfish, iterative depth 1–16.

| | Russek et al. | Our implementation |
|---|---|---|
| VOC (main) | ΔUC = V_deep(a_deep) − V_deep(a_shallow); deep=15, shallow=1 | Same formula; uses multipv=2 + root_moves fallback |
| Candidate set | All legal moves (engine search); E[ΔUC] uses top-5 from depth-1 | All legal moves (same); E[ΔUC] not yet implemented |
| Win prob | Stockfish centipawns → logistic regression to pwin | Engine WDL directly (lc0 native; Stockfish WDL) |
| VOC=0 handling | Included; tested separately as regression subset | Included |
| Exclusion criteria | Ply 15–75; opponent ≥60s remaining; 11 Lichess time controls | Ply > 10 (preliminary; should align to 15–75) |
| Dataset size | 519M moves | 135M moves (nonzero RT subset) |

**Key difference we are NOT computing yet:** E[ΔUC] = expected VOC, which requires evaluating the top-k (k=5) depth-1 moves and weighting by their probability of being chosen. This is the analysis for Figures 4–5 of the paper.

**VOC=0 cluster:**  
Russek et al. DO NOT exclude VOC=0. They arise naturally when depth-15 and depth-1 agree on the best move. In their ply 15–75 middlegame sample, the cluster is present but not dominant because Stockfish's depth-1 evaluation is weak (material count only), so depth-15 frequently finds a better move. With lc0, VOC≈0 for most positions because the neural network prior at depth=1 is already highly accurate — this is a property of lc0's architecture, not a dataset artifact.

### Determinism

Both Stockfish (with Clear Hash) and lc0 (CUDA backend) are **fully deterministic** at fixed depth. Running the same position twice gives identical VOC/MQ values. The "n_repeats" option in `voc_mq_analysis.py` was added to test this; confirmed std=0 across repeats for both engines.

### Engine choice: lc0 vs Stockfish for VOC

- **lc0**: VOC≈0 for most positions at depth ≤ 5 because the NN policy is already near-optimal at depth=1. To get meaningful VOC with lc0, need `depth_deep` ≥ 50 (or node-based limits).
- **Stockfish**: VOC is non-trivial at depth_deep=5 because depth-1 is weak (no learned prior). **Recommended for initial VOC/MQ exploration** until we establish what depth lc0 needs.
- **lc0 on A100**: ~10.8s/position at depth=5, ~72s/position at depth=15.
- **Stockfish (1 core)**: ~0.5–2s/position at depth=5 (TBD from timing comparison).

### Ply filter

Currently using `move_ply > 10`. Should align to Russek et al.'s ply 15–75 to:
1. Exclude opening (VOC trivially 0, move times very short)
2. Exclude endgame (simpler positions, small candidate sets)

---

## Timing benchmarks (2026-06-02)

5 middlegame positions (ply 15–75), MQ + VOC per position (3 engine calls total: 1 MQ + 2 VOC steps), 1 CPU thread.

| Engine | Depth | s/pos | 10K (10× par) | 100K (10× par) |
|---|---|---|---|---|
| Stockfish | 1 | 0.01 | 7s | 1.2m |
| Stockfish | 5 | 0.01 | 10s | **1.7m** |
| Stockfish | 10 | 0.10 | 1.6m | 16.5m |
| lc0 | 1 | 0.18 | 3.0m | 29.9m |
| lc0 | 5 | 0.19 | 3.2m | 31.6m |
| lc0 | 10 | 0.41 | 6.9m | 1.1h |

Stockfish depth=5 is **19× faster than lc0 depth=5** (0.01 vs 0.19s/pos). lc0's GPU batch overhead dominates at shallow depths. Stockfish at depth=5 with 10-way parallelism can annotate 100K positions in under 2 minutes.

lc0 depth=15 (from earlier test): ~72s/pos → 100K positions ≈ 20 hours at 10× par. Not feasible for exploration.

**Recommendation:** Use Stockfish depth=5 for the initial large-scale run (fast, meaningful VOC). Revisit lc0 depth≥50 later if we want NN-native evaluations.

## Histogram summary: Stockfish depth=5, n=100 middlegame positions (2026-06-02)

Positions sampled from ply 15–75, one eval per position.

**VOC (ΔUC):**
- mean=0.107, std=0.210, median=0.000, p90=0.427
- **41% of positions have non-zero VOC** — Stockfish depth=1 and depth=5 frequently disagree on the best move in middlegame positions (consistent with Russek et al.)
- Distribution is heavily right-skewed: most positions have VOC=0 (depth-1 already optimal), but a long tail of tactical positions where deep search finds substantially better moves

**MQ:**
- mean=−0.116, std=0.229, median=0.000, p90=0.000, min=−1.000
- **61% of moves are near-optimal (MQ ≥ −0.005)** — most moves in high-ELO games are good
- Long left tail: occasional blunders pull the mean down; min=−1.0 corresponds to a move that hands a won position to the opponent
- Heavy spike at 0 (played move = best move); non-trivial tail of errors

**Correlation log(RT) vs VOC:** r=0.197 (n=100) — in the expected direction. Higher VOC → more thinking time. Small-sample estimate, but directionally consistent with Russek et al.
Figures in `figures/voc_mq_analysis/`.

## pos_with_engine_eval: what is computed

For each filtered move (ply 15–75, opponent_clock ≥ 60s) the following are computed by `evaluate_position()` in **2–4 engine calls**:

| Column | Formula | Range | Interpretation |
|---|---|---|---|
| `e_win_best` | V_deep(a_deep) | [0, 1] | Win prob of objectively best move (depth_deep=5) |
| `e_win_second_best` | V_deep(rank-2 move) | [0, 1] | Win prob of second-best move |
| `e_win_taken` | V_deep(move_taken) | [0, 1] | Win prob of the move actually played |
| `voc` | e_win_best − V_deep(a_shallow) | [0, 1] | Gain from deep search vs committing to depth-1 best |
| `mq` | e_win_taken − e_win_best | [−1, 0] | How suboptimal was the played move (0 = optimal) |
| `toptwo` | e_win_best − e_win_second_best | [0, 1] | How decisively the best move beats 2nd-best |

All win probabilities are from the **side to move's perspective** before the move is made.
Engine call budget: (1) depth_shallow=1 multipv=1 → a_shallow; (2) depth_deep=5 multipv=2 → a_deep, e_win_best, e_win_second_best; (3–4) optional root_moves fallbacks for e_win_taken and V_deep(a_shallow) if not in top-2.

## build_pos_with_engine_eval smoke test — 10K positions, 10 workers (2026-06-02)

Filters: ply 15–75, opponent_clock ≥ 60s. Stockfish depth=5.

**Timing:**
- Evaluation: **17.7s** for 10K positions → **565 pos/s** with 10 workers (0.0018s/pos)
- Wall clock: 51s total (sampling the JOIN costs ~30s; pre-materialised shard mode avoids this in SLURM)
- Extrapolation (evaluation only): 100K → ~3 min, 1M → ~30 min at 10 workers

**Timing:** 10K positions in **16.7s** → **597 pos/s** (10 workers, Stockfish depth=5/1)

**Output statistics (n=10,000, all 6 quantities):**

| Column | Mean | Std | p50 |
|---|---|---|---|
| `e_win_best` | +0.593 | 0.398 | +0.595 |
| `e_win_taken` | +0.477 | 0.409 | +0.490 |
| `voc` | +0.098 | 0.212 | +0.000 |
| `mq` | −0.117 | 0.235 | +0.000 |
| `toptwo` | +0.122 | 0.217 | +0.011 |

- Pearson r(log RT, VOC) = **+0.086** (n=10K)
- Loaded into `pos_with_engine_eval` table in personal.db
- Figures in `figures/mq_voc/`

## 2026-06-02: Plot standardization + full 100K analysis

### Plot conventions
- All figures flat in `figures/` (no subfolders); named `x_vs_y.png` or `x_histogram.png`
- Histograms: 50 bars, dashed lines at mean (black) and median (gray), title = `n = X moves`
- Bivariate plots: Analyzer 2×2 dashboard (raw trend | quantile bins | ×2 by ply tertile)
- "log" not "ln" in all axis labels
- No scatter in background of bivariate plots

### Structural cleanup
- Removed redundant `script_engine_eval.py` and `script_merge_evals.py` (replaced by `build_pos_with_engine_eval.py`)
- `build_pos_with_engine_eval.py` merge now adds `ply_tertiles` (ntile(3) by move_ply) to `pos_with_engine_eval` for Analyzer compatibility

### MQ sign fix
MQ was occasionally positive due to aspiration window differences between Stockfish's multipv=2 search and the root_moves fallback search. Clamped to `min(0, e_win_taken - e_win_best)` in `PositionEval.mq` property.

### 100K evaluation results (Stockfish depth=5/1, ply 15–75, opponent_clock ≥ 60s)

| Column | Mean | Std | p50 |
|---|---|---|---|
| `e_win_best` | +0.593 | 0.397 | +0.585 |
| `e_win_second_best` | +0.470 | 0.401 | +0.487 |
| `e_win_taken` | +0.480 | 0.408 | +0.490 |
| `VOC` | +0.100 | 0.216 | +0.000 |
| `MQ` | −0.116 | 0.227 | +0.000 |
| `toptwo` | +0.123 | 0.214 | +0.013 |

- VOC non-trivial (>0.005): **33%** of positions
- MQ non-trivial: **38%** of moves
- r(log RT, VOC) = **+0.097** (n=100K) — in the expected direction
- r(MQ, clock) = **−0.098** (n=100K) — see below

### MQ vs clock: finding and interpretation

**Expected:** more clock → more time available → better move → MQ closer to 0 (positive r).

**Observed:** r(MQ, clock) = −0.098, and this is **negative within every ply tertile**:

| Tertile | Ply range | r(MQ, clock) |
|---|---|---|
| 1 | 15–31 | −0.024 |
| 2 | 32–49 | −0.037 |
| 3 | 50–75 | −0.007 |

**Interpretation:** Players with more clock remaining at a given ply have been playing quickly up to that point — moving through low-VOC (book/simple) positions without spending much time. Their intuitive choices in those positions deviate from depth-5's specific tactical preferences, giving more negative MQ. Players with less clock have been burning time on high-VOC (complex) positions, and their deliberate moves align better with depth-5. This is consistent with VOC theory: the MQ-clock relationship is mediated by the complexity (VOC) of the positions being played. At depth=15 (Russek et al.), this effect may differ since the engine's deeper evaluation more closely tracks human strategic understanding.

### VOC vs Russek et al. effect sizes

| | Our result (depth=5, n=100K) | Russek et al. (depth=15, n=519M) |
|---|---|---|
| r(log RT, VOC) | +0.097 | Positive (similar magnitude, Figure 2) |
| Engine | Stockfish | Stockfish |
| Depth | 5 | 15 |
| VOC non-zero | 33% | Higher (depth=15 finds more tactics) |

Our shallower depth (5 vs 15) produces a noisier VOC estimate — ~67% of positions have VOC=0 vs fewer at depth=15. The effect direction is consistent. Increasing depth to 15 would sharpen VOC discrimination and likely increase r.

### Figures produced
All in `figures/` (flat):
`clock_vs_movetime`, `clock_opp_vs_movetime`, `npossiblemoves_vs_movetime`, `pieces_exc_pawns_vs_movetime`, `self_pieces_exc_pawns_vs_movetime`, `ply_vs_movetime`, `ply_vs_pinstant`, `movetime_histogram`, `log_movetime_histogram`, `voc_histogram`, `mq_histogram`, `toptwo_histogram`, `voc_vs_movetime`, `mq_vs_clock`.

## Theoretical framework: what are we really trying to show? (2026-06-03)

### The core question

$$\text{opt\_num\_think\_steps}(s) \leftrightarrow \text{actual\_num\_think\_steps}(s)$$

where:
- **opt_num_think_steps** — the normatively optimal amount of computation for position *s*, defined under a reward–computation tradeoff: the depth (or budget) *d\** where the marginal gain from one more step equals the marginal cost of that step
- **actual_num_think_steps** — the observed think time (RT) in seconds, as a proxy for the number of internal evaluation steps

Everything else in this project is a **proxy for the left side**. The empirical hypothesis is that rational agents allocate computation in proportion to how much it normatively helps.

---

### Hierarchy of proxies for opt_num_think_steps

**Crude proxies** (compare two fixed depth points, ignoring cost):

| Proxy | Description | Limitation |
|---|---|---|
| `gain_depth` = V_5(a_5) − V_5(a_1) | Total gain from depth=5 vs depth=1 | Arbitrary depth comparison; no marginal structure |
| `gain_budget` = V_96(a_96) − V_96(a_1) | Same with node-budget limits | Same; budget limit still arbitrary |

**Better proxy** (captures the marginal value curve):

> `marginal_gain(d)` = V(d+1) − V(d) — how much does *one more* computation step improve the outcome?
>
> opt_num_think_steps is where `marginal_gain(d) = cost`. The proxy needs to approximate this slope, not just the total difference between two endpoints.

Computing `marginal_gain` at multiple depths (d=1,2,3,5,10,15) gives the **benefit curve**, and the optimal stopping point is where it crosses the cost line. Our gain_depth measures the area under this curve between depth 1 and 5, not the slope at the operating point.

---

### Why the current formulation misses the core

**gain_depth/gain_budget capture only search-depth uncertainty**: "does going deeper on the best move help?" They miss:

1. **Candidate-set uncertainty** — the noise from not knowing *which* move to evaluate deeply. A position with 10 near-equal candidates has high uncertainty even if depth-1 and depth-5 agree on the winner.

2. **Position-dependent noise** — the reliability σ(position) of the shallow estimator varies. For a tactical position, V_1(a) is a noisy estimate of V*(a); for a quiet endgame it is not. β = 1/σ² is not a free parameter but should emerge from the position's actual evaluation noise.

3. **Satisficing stopping rule** — agents don't compute to certainty; they stop when confidence is *sufficient*. Even at depth=5, uncertainty remains. The stopping rule is:
   $$T = \min\{d : P(a_d = a^* \mid s) \geq \theta\}$$
   where θ is the agent's satisficing threshold ("80% sure"), not a fixed depth. This means the observed RT is the *time to threshold crossing*, not a fixed budget.

Together, this is a **Drift-Diffusion / sequential stopping** process: evidence for each candidate accumulates as a noisy drift; the decision fires when one candidate's evidence crosses a threshold. High-uncertainty positions take longer to reach threshold.

---

### What E[ΔUC] adds (Russek et al. Figures 4–5)

$$E[\Delta\text{UC}] = V_\text{deep}(a_\text{deep}) - \sum_k P(a_k \mid \text{shallow}) \cdot V_\text{deep}(a_k)$$

where $P(a_k \mid \text{shallow}) \propto \exp(\beta \cdot V_\text{shallow}(a_k))$ over the top-*K* candidates.

This does incorporate candidate-set noise through β. But β should not be a free tuning parameter — it should equal 1/σ²(position), the empirically estimated noise in the agent's shallow evaluator. The softmax also does not collapse at deep depth — even V_deep is uncertain; the agent simply stops when σ_d is small enough relative to θ.

---

### min_expansions — the most natural proxy so far

$$d^*(s) = \min\bigl\{d : a_d(s) = a_\infty(s)\bigr\}$$

where $a_d$ = argmax V_d (best move at depth $d$), $a_\infty$ = best move at "truth" depth (e.g. depth=15).

**min_expansions** is the minimum number of search steps needed before you would make the *same decision* as you would with much longer thought. It is the **first hitting time** of the eventually-correct action under iterative deepening.

**Why this is the cleanest proxy yet:**
- Avoids continuous value comparisons — asks only about decision identity (which move), not numerical difference
- Directly in units of compute steps; maps naturally onto opt_num_think_steps
- min_expansions = 1 → trivially easy (no computation needed); min_expansions = 5 → genuinely needs depth to resolve
- Sidesteps the V_shallow noise problem — you don't need to trust numerical estimates, only move identity

**Two variants:**
- *First convergence:* min{d : a_d = a_∞} — when do you first arrive at the right answer?
- *Stable convergence:* min{d : ∀d' ≥ d, a_d' = a_∞} — when do you commit and stay? (preferred for RT comparison — a fleeting correct answer at depth 3 overturned at depth 4 should not count as resolved)

**Relationship to existing proxies:**
- gain_depth > 0 → min_expansions > 1 (necessary but not sufficient; value gap and decision flip are related but not identical)
- entropy_topk ↑ → min_expansions ↑ (more candidate ambiguity → longer to converge)
- min_expansions is a direct operationalisation of opt_num_think_steps under the assumption that you should think until you'd make the same decision as with unlimited compute

**Implementation:** Stockfish's iterative deepening naturally visits every depth on the way to depth D. Python-chess `engine.analysis()` (streaming) returns the PV at each depth step, so min_expansions can be extracted from a single engine call.

### Next analysis targets

- [ ] **marginal_gain curve**: evaluate positions at depths {1, 2, 3, 5, 10, 15}; compute V(d+1)−V(d) at each step; fit where the curve flattens
- [ ] **entropy_topk**: run multipv=K at depth=1; compute entropy of softmax(β·V_shallow) over top-K moves; compare with RT
- [ ] **E[ΔUC]** with top-5 candidates: implement Russek et al. Figures 4–5 variant
- [ ] **Estimate β from data**: regress human move choices against softmax(β·V_shallow) to calibrate decision temperature
- [ ] **Compare** opt_num_think_steps (normative, from marginal curve) vs actual RT: does the marginal-gain crossing point predict RT better than gain_depth?

---

### Connection to the lmcos architecture

The **DP Oracle** in lmcos computes opt_num_think_steps exactly (under known cost C):
$$V^*(s) = \max\bigl(V(s),\; \mathbb{E}[V^*(\text{child})] - C\bigr)$$

The human data provides actual_num_think_steps. The fundamental validation is:
$$r\bigl(\text{DP-Oracle opt depth},\; \text{human RT}\bigr) > r\bigl(\text{gain\_depth},\; \text{human RT}\bigr)$$

The current analyses (gain_depth, gain_budget, entropy_topk) are stepping stones toward computing the left side of this correlation properly.

---

## Implementation plan: lmcos ↔ human_analytics comparison (2026-06-03)

**Status (2026-06-04):** Superseded for the directional go/no-go by **Analysis 0a** (`lmcos/analysis/oracle_stop_step_features.py`), which recomputes `oracle_stop_step` from `filtered_shard` trees (WDL halt rewards) rather than packed-shard labels. **Analysis 0b** (`lmcos/analysis/minimal_mc_baseline.py`) answers whether the GNN is necessary. Same-position human↔oracle comparison remains **Analysis 1** (not started). See [Analysis 0 results](#analysis-0-results-2026-06-03) and [Analysis readiness](#analysis-readiness-2026-06-04).

### The key bridge (target end state)

```
Same position features (computed from root FEN)
     ↓                    ↓                    ↓
log(RT)          oracle_stop_step      model_predicted_stop
[human]          [normative, exact]    [lmcos controller output]
```

A0a tested the middle column on **lmcos training trees** (not human positions). A1 will align all three on **human FENs**.

---

### What exists in the lmcos packed data

**Per-node scalar features** (from `lmcos/src/core/schema.py`):
- `value`, `wdl_win`, `wdl_draw`, `wdl_loss`, `wdl_var`

**Per-node metadata:**
- `fen` — board position at that node (root node has the root FEN)
- `depth` — plies from root (root = 0, children = 1, …)

**Per-episode controller cache** (from `lmcos/src/train/controller_train.py`):
- `oracle_stop_steps` — DP Oracle optimal halt step (**this is opt_think_steps**)
- `z_root` — pre-computed encoder embedding of the root node (for fast controller inference)

**Trainable outputs:**
- `P(halt | z_root)` — the trained halt controller prediction
- `predicted_stop_step` = first expansion step where `P(halt) > 0.5`

---

### Position features computable from root FEN

The root node's FEN allows extraction of the **same features used in human_analytics** using the existing regex-based functions:

| Feature | Source | Human_analytics analog |
|---|---|---|
| `n_possible_moves` | `chess.Board(root_fen).legal_moves` count | branching factor |
| `n_self_pieces_exc_pawns` | FEN regex `[RNBQK]` for white / `[rnbqk]` for black | own material |
| `move_ply` | FEN halfmove field or game metadata | game stage |

### Tree features computable from the tree structure

| Feature | Computation | Human_analytics analog |
|---|---|---|
| `gain_depth_equiv` | `root_wdl_win_at_max_depth - root_wdl_win_at_depth_1` | gain_depth |
| `toptwo_equiv` | `max(children_wdl) - second_max(children_wdl)` over root children | toptwo |
| `child_wdl_var` | `var(wdl_win over root children)` | candidate-set uncertainty / entropy_topk |
| `tree_size` | total nodes in episode | quasi-cost of search |

---

### Implementation plan (3 steps)

**Step 1: Extract features from lmcos validation episodes**

Write `lmcos/analysis/human_comparison_features.py`:

```python
def extract_episode_features(packed_episode_path: str) -> pd.DataFrame:
    """
    Load packed lmcos episodes and extract per-episode features.
    Returns DataFrame with columns:
        root_fen, oracle_stop_step,
        n_possible_moves, n_self_pieces_exc_pawns, move_ply,
        gain_depth_equiv, toptwo_equiv, child_wdl_var, tree_size
    """
```

- Load from the materialized cache directory (same path used by `ControllerEpisodeDataset`)
- For each episode: get root node FEN + oracle_stop_step + tree structure
- Compute position features using `chess.Board(root_fen)` (same as human_analytics)
- Compute tree features from the node scalar_features dict

**Step 2: Get model predictions**

```python
def add_model_predictions(df: pd.DataFrame, checkpoint_path: str) -> pd.DataFrame:
    """
    Load trained halt controller; run on z_root embeddings from cache;
    add predicted_stop_step column.
    """
```

- Load the controller checkpoint from `lmcos/slurm/outputs/4_supervised_controller/`
- Run on `z_root` tensors (already materialized — no re-encoding needed)
- `predicted_stop_step` = argmax of P(halt) over expansion steps, or first step > 0.5

**Step 3: Make the comparison plots**

Use the `_bin_trend` / `_plot_with_trend` utilities already in `human_analytics/utils/` (or inline equivalents) to produce:

For each position feature in {n_possible_moves, n_self_pieces_exc_pawns, gain_depth_equiv, toptwo_equiv}:

```
Figure: 3-panel comparison
  Left:   feature → log(RT)            [human data, from pos_with_engine_eval joined with processed_moves]
  Middle: feature → oracle_stop_step   [lmcos normative]
  Right:  feature → predicted_stop_step [lmcos controller]
```

Expected qualitative agreement:
- `n_possible_moves` → **positive** in all three (more branching → more compute)
- `gain_depth_equiv` → **positive** in all three (more value in deeper search → more compute)
- `toptwo_equiv` → **negative** in all three (decisive position → stop early)
- `child_wdl_var` → **positive** (high candidate uncertainty → more compute)

**If the signs match across all three columns, the validation is complete.**

---

### Data paths to locate

Before coding, need to confirm paths on cluster:
- [ ] Materialized controller cache: `$SCRATCH/hl4291/...` — check with `ls /scratch/gpfs/GRIFFITHS/hl4291/`
- [ ] Best trained checkpoint: `lmcos/slurm/outputs/4_supervised_controller/` (already in repo)
- [ ] Packed episode shards: same scratch path used in `lmcos/slurm/3_preprocess_root/`

---

### What we explicitly do NOT need

- Running Lc0 on human dataset positions (no new inference needed)
- Matching position FENs between human dataset and lmcos dataset (separate datasets are fine — we just need qualitative directional agreement, not position-level matching)
- depth=15 or deeper engine evaluation (the lmcos tree search already goes as deep as the training pipeline generated)
- A perfect normative theory of VOC (oracle_stop_step IS the normative answer under the lmcos reward function)

---

## Synthesis: Two stories and how they relate (2026-06-03)

Before continuing analysis, it is worth stepping back to be precise about what we are claiming and what the findings could mean. There are two distinct narratives running in parallel, and conflating them produces confusion.

---

### Story 1: The human behavioral story (descriptive)

**What we have established:**

We have 1M positions from high-ELO (≥2000) rapid chess (10+0), filtered to ply 15–75, opponent clock ≥60s. For each position, we measure log(RT) (how long the player thought) and four board features:

| Feature | r with log(RT) | Interpretation |
|---|---|---|
| Branching factor | +0.20 | Wider candidate set → longer thinking |
| Own material | +0.04 | More pieces → more interactions to reason about |
| gain_depth (ΔUC@5) | +0.10 | Positions where deeper search demonstrably helps → longer thinking |
| toptwo | −0.06 | Decisive positions (one clearly best move) → less thinking |

**What this does NOT tell us:**

- Whether humans are thinking optimally. This is purely descriptive: longer thinking *correlates with* these features. Correlation ≠ optimality.
- Whether these features are causes or proxies. Branching is correlated with position complexity (own material, r=+0.47), game stage (ply, r=−0.34). Each feature is a noisy proxy for some underlying concept of "hard position."
- Whether the results generalize. The filters (10+0, ≥2000 ELO, ply 15–75, opponent clock ≥60s) select a specific slice of chess. Bullet players, beginners, or endgames might show different or reversed patterns.

---

### Story 2: The normative story (prescriptive)

**What the DP oracle gives us:**

The lmcos oracle solves:
$$V^*(s) = \max\bigl(V_\text{halt}(s),\ V_\text{continue}(s) - C\bigr)$$
and returns `oracle_stop_step` = the expansion count at which it is DP-optimal to halt, given the cost function (time_lambda=18.537, etc.).

This is a *normative* answer to "when should you stop?" under the assumption that:
1. The value of computation is the improvement in best-move WDL from further search
2. The cost of computation is captured by the time cost function (convex in remaining budget)
3. The agent is Lc0 running MCTS, not a human

**The normative question:** Do board features (branching, material, gain_depth, toptwo) predict oracle_stop_step in the same direction as they predict human RT?

- If **yes, same direction**: both the oracle and humans think harder in the same positions. This is *consistent* with human behavior being near-optimal under the same cost structure.
- If **different directions for some features**: the oracle and humans are responding to different aspects of position complexity. This could mean humans are irrational, OR the cost function is miscalibrated for human cognition, OR the features don't mean the same thing to a neural-network MCTS engine as to a human.

---

### The key interpretive possibilities

**Case A: Oracle and humans agree on all features.**
- Best case for the "humans are quasi-optimal" narrative.
- Caveat: agreement could be spurious if both are driven by the same low-level confound (e.g. tactical positions are both hard for MCTS and hard for humans for unrelated reasons).

**Case B: Oracle and humans agree on some features (e.g. toptwo) but disagree on others (e.g. branching).**
- This is the most likely case and the most informative.
- Branching disagreement is structurally expected: Lc0 with MCTS is naturally sensitive to breadth (more branches = more exploration needed), while humans may be more sensitive to depth (tactical forcing lines) independently of breadth.
- toptwo agreement would suggest both human and oracle stopping is sensitive to "decisiveness" — a clean theoretical alignment.

**Case C: Oracle and humans systematically disagree.**
- Either humans are fundamentally irrational about when to think, OR
- The oracle's cost function (calibrated for Lc0 compute, not human RT) is simply not the right normative model for human deliberation, OR
- The selection filters create sampling artifacts that break the comparison (see below).

---

### The confound: selection on game characteristics

This is a critical open question. Our human dataset is:
1. **10+0 time control** — these are rapid players with a specific time management strategy. They know they have 10 minutes and pace accordingly. This may induce regularities that wouldn't hold for bullet (3+0) or classical (90+30) games.
2. **Both players ≥2000 ELO** — strong players. Their error rates are low (62% near-optimal MQ), their book knowledge is deep (opening positions may be systematically faster), and their time management may be more strategic.
3. **Ply 15–75** — excludes opening book and late endgame. The features we study (branching, material) vary mostly in this range.
4. **Opponent clock ≥60s** — this excludes time-scramble situations. In time scrambles, players behave differently regardless of board complexity.

The key question: **Are the features we measure (branching, gain_depth, etc.) correlated with log(RT) because of their causal relationship to computation demand, or because of how these filters select positions?**

For example: are complex middlegame positions (high branching) more likely to appear at move 25 (mid-game) when clock is still plentiful → players feel they can afford to think longer? This would produce the r=+0.20 correlation even if branching per se had no causal role.

The only clean way to resolve this is to control for ply and clock simultaneously — which the Analyzer's ply-tertile stratification partially does. But the r's are not obviously different across tertiles for branching, suggesting the effect is not purely a ply confound.

---

### The lmcos training data vs human data

There is a further disanalogy: the lmcos oracle is computed on trees generated by **Lc0 searching from sampled root FENs** (sampled_fens_2023_daily_seed42_n100.parquet — 100 FENs per day, broad coverage). The human data is from **specific moves played in Lichess games**, which are biased toward positions that arose in actual games (opening theory, popular middlegame structures). These may be different distributions of positions, making the comparison noisier.

For a clean comparison: take the same FEN positions that appear in the human dataset AND have lmcos oracle solutions, and compare oracle_stop_step vs human RT for the same positions. This is the gold standard but requires overlap between the two datasets.

---

### Research agenda: what to do before more blind analysis

**Clarify the claim you want to make:**

Option A (weakest but most defensible): "Board features that predict human RT also predict oracle_stop_step in the same direction, suggesting that both are sensitive to the same underlying notion of position complexity."
- This requires: oracle_stop_step computed for lmcos validation trees, correlations with board features.
- Does NOT require: same positions as human data.

Option B (stronger): "For the same board positions, humans think longer precisely when the oracle says to think longer."
- This requires: overlap between human dataset and lmcos oracle solutions (same FENs).
- Hard: the datasets are constructed differently.

Option C (aspirational): "The lmcos controller, trained on the oracle, produces halt decisions that correlate with human RT."
- This requires: running the trained controller on positions from the human dataset.
- Feasible: feed human dataset FENs through Lc0 → controller → P(halt), compare with RT.

**Questions to answer before implementing:**

1. Is oracle_stop_step from the lmcos oracle expected to agree with human RT features at all, given that the cost function is calibrated for Lc0 (not humans)?
2. Should we condition on starting_budget when computing oracle_stop_step? (Different budgets give different stopping steps — which budget is most comparable to human 10+0 time control?)
3. Are the lmcos training positions a representative sample of chess positions or a biased sample (e.g. sampled more from complex tactical positions)?

---

## Analysis 0 results (2026-06-03)

**Status: complete.** Removed from `proposed_next_steps.md` (2026-06-04). Go/no-go: **proceed with Analysis 1** (directional oracle–human agreement); **proceed with Analysis 2** (GNN likely redundant for halt/continue).

### Implementation (as built)

| Piece | Path | Notes |
|---|---|---|
| A0a script | `lmcos/analysis/oracle_stop_step_features.py` | Default `--n-trees 5000`, primary budget **43** (medium-large bucket); also sweeps bucket reps `[2, 7, 18, 43, 90]` |
| A0b script | `lmcos/analysis/minimal_mc_baseline.py` | Default `--n-trees 2000` train / `--n-val-trees 500`; budget **43** |
| Supplementary | `lmcos/analysis/min_expansions_analysis.py` | Explains sign flip vs DP oracle (not part of A0 gate) |
| Data | `.../generated_trees_combined/filtered_shard*/` | **Not** packed controller shards; same `.pt` tree format as tree generation |
| Oracle | `src/data/preprocess_mc/oracle.py` → `compute_budgeted_oracle()` | `halt_rewards[t] = oracle_root_q_trace[t, oracle_best_move_index[t]]` (WDL); `tree_sizes = [1]*T` |
| Human r baseline | `_HUMAN_R` in A0a script | From `pos_with_engine_eval` + board features (branching +0.195, material +0.039, gain_depth +0.096, toptwo −0.064) |
| Tests | `test_oracle_stop_step_features.py`, `test_minimal_mc_baseline.py` | **25/25 passing** (`pytest` from `lmcos/`) |

**Approximation (documented in A0a):** halt rewards use **WDL** from `oracle_root_q_trace`; packing used centipawn-derived values. Packed `oracle_stop_steps` agreement is tested in unit tests (±1 step tolerance).

### Analysis 0a: oracle_stop_step vs board features (n=5K lmcos trees, budget=43)

| Feature | Oracle r | Human r | Direction match |
|---|---|---|---|
| branching (`n_possible_moves`) | +0.165 | +0.195 | ✓ |
| material (`n_self_pieces_exc_pawns`) | +0.154 | +0.039 | ✓ |
| gain_depth (`gain_depth_equiv`) | **+0.797** | +0.096 | ✓ |
| toptwo (`toptwo_equiv`) | +0.436 | −0.064 | ✗ |

`oracle_stop_step`: mean=11.7, std=10.3, range=[0, 43].

**Interpretation:**
- **3/4 directional matches** on lmcos trees vs human RT proxies → worth generating human-position trees (A1)
- **gain_depth:** oracle strongly continues where Q still improves; humans weakly in the same direction
- **toptwo mismatch:** oracle keeps searching to refine Q when two moves are close; humans treat large toptwo as “decided” → shorter RT (satisficing)
- Budget 2 too small to discriminate; budgets 7–90 give stable branching r (~+0.13–0.17)

**Figures:** `lmcos/analysis/figures/oracle_stop_step_vs_human_rt.png`, `oracle_stop_step_correlation_matrix.png`

**Not the DP oracle:** `min_expansions` (stable `oracle_best_move_index`) inverts signs vs `oracle_stop_step`; see `min_expansions_analysis.py`.

### Analysis 0b: minimal MC baseline (n=2K train / 500 val trees)

Per-snapshot features: `best_q`, `wdl_var` (evaluated root children), `t_norm`, `budget_rem_norm`.  
MLP: 2×64, same depth as MC head; targets = `sign(compute_budgeted_oracle().target_advantages)`.

| Model | Sign accuracy |
|---|---|
| GNN+MC (baseline, from training logs) | **90.1%** |
| Minimal MLC (train) | 87.4% |
| Minimal MLC (val) | **86.4%** |

Val **above 80% go/no-go**; gap 3.7 pp. Inference ~0.054 ms/snapshot (synthetic batch timing in script).

**Figures:** `minimal_mc_sign_accuracy.png`, `minimal_mc_weights.png` (optional `minimal_mc_timing.png` mentioned in docstring; main() currently saves the two above)

**Note vs original plan:** A0b uses **filtered_shard trees** + on-the-fly oracle targets, not packed validation `target_advantages` shards. Same supervision definition, faster to wire.

---

## Analysis readiness (2026-06-04)

### Analysis 1 — Human FEN trees + oracle vs log(RT)

| Criterion | Status |
|---|---|
| A0a go/no-go (≥3/4 feature directions) | **Pass** (3/4; toptwo known mismatch) |
| Human FEN source | **Ready** — `processed_moves_nonzero` in `personal.db` (ply 15–75, opp_clock ≥ 60s) |
| FEN format | **Likely OK** — 4-field FEN loads in `python-chess` (`Board(fen).is_valid()`); `build_tree.py` reads newline FENs; confirm castling/ep in join key before 10K scale |
| Tree generation | **Not run** — need SLURM config pointing at human FEN file, ysagiv weights, budget=96 |
| Extraction / plots | **Not implemented** — `human_oracle_comparison.py` and tests still to write |
| Smoke test gate | **Required** — 1K trees must finish in ~15 min on A100 before 10K |

**Verdict:** **Ready to start** Step 1 (FEN export + format check) and Step 2 (1K smoke SLURM). Do not scale to 10K until smoke passes.

### Analysis 2 — Minimal model / skip GNN pretraining

| Criterion | Status |
|---|---|
| A0b go/no-go (val sign acc ≥ 80%) | **Pass** (86.4%) |
| Joint GNN+MC training | **Supported** — `controller_train.py` has live-encoder path when `unfreeze_encoder: true` (skips materialized cache) |
| Config D YAML | **Not created** — no `subtree_weighting_root_scratch_D.yaml` yet |
| Ablation data | **Ready** — ysagiv packed train shards (subsample 10–20 for first run) |
| Success metric | GNN loss on subsample vs pretrained baseline (needs baseline number from ysagiv run) |

**Verdict:** **Ready to start** once Config D YAML + 10-shard manifest paths are set; can run **in parallel** with A1 (different GPU jobs).

### Analysis 3 — Weaker engine

**On hold** until A1 reports `r(oracle_stop_step, log RT)` on matched human positions.

---

## The epistemology of stopping (2026-06-03)

### The "chasing tails" phenomenon

The min_expansions analysis revealed a systematic misallocation: both humans and
the oracle (oracle_stop_step) spend *more* time in positions where the correct action
became apparent *earlier* (high gain_depth, large min_expansions is small). This is
the opposite of efficient computation.

**The mechanism:** In dominant positions, each expansion further confirms the dominant
move. Q keeps improving. The Q-refinement stopping signal — "is the search still
teaching me something?" — keeps firing. The agent chases the tail of an asymptotically
converging Q-estimate, spending many steps to learn that A is worth 0.92 rather than 0.85,
even though they would have played A either way.

In ambiguous positions, Q plateaus early (no move stands out). The refinement signal
weakens. The agent stops — but these are exactly the positions that warrant more compute.

### The three epistemic cases

At any point during search, an agent faces uncertainty across three situations:

1. **Converging to the right answer.** Q of the current best keeps rising; the best
   move *is* the globally optimal move. The agent should stop.

2. **Converging to the wrong answer.** Q of the current best keeps rising, but there
   is a globally superior move in the unexplored region. The agent should keep going.
   But the observable signal (Q-improvement) is identical to case 1.

3. **Not converging.** Q of the current best has plateaued. The agent might eventually
   converge with more search, or might not. The signal (low Q-improvement) is consistent
   with "position is genuinely ambiguous" and also with "best move exists but search isn't
   finding it."

### The fundamental asymmetry

The Q-refinement signal is observable. The *reason* for it is not. Cases 1 and 2 produce
identical signals but require opposite actions. Cases 3 and "nothing better exists" also
produce identical signals but require opposite actions.

The correct stopping criterion — *would stopping now cause me to make a wrong decision?* —
requires knowing what further search would reveal. But knowing that requires completing the
search. **This is circular**: you need to search to know whether to stop searching.

There is no symmetric information available. The agent can observe:
- Whether Q is currently improving (local signal)
- How many children have been evaluated (partial observability of search coverage)

The agent cannot observe:
- Whether the unexplored region contains a superior move (requires completing the search)
- Whether continued Q-improvement reflects refinement of the true best move or refinement
  of a false leader that will eventually be overturned

### The Q-refinement rule as a rational but biased heuristic

Using Q-improvement as the stopping signal is the only locally observable proxy. It is
rational in the Bayesian sense — "keep going while learning." But it is systematically
biased because it correlates with positions where the search has already effectively
terminated (dominant move found and just being confirmed), not with positions where
search is genuinely needed (ambiguous, no clear winner).

**Prediction:** the worst blunders (large negative MQ) should cluster in low gain_depth
positions — where both oracle and human stopped early because Q plateaued, but the true
best move was not yet found. This is testable from `pos_with_engine_eval`.

### Connection to VOC and E[ΔUC]

This framing clarifies why Russek et al.'s E[ΔUC] (expected VOC) is the right measure.
E[ΔUC] asks: "given my current distribution over which move I might play (determined by
the shallow Q-values and a softmax temperature β), how much value do I *expect* to gain
from deeper search?" This is exactly the estimate of P(deeper search changes my action)
× E[gain from the change. It tries to directly estimate what the Q-refinement rule
cannot observe: the probability that continued search would alter the decision.

---

## Open items / next steps
- [ ] Align ply filter to 15–75 (matching Russek et al.)
- [ ] Add `opponent_clock_time >= 60s` filter to match Russek et al.
- [ ] Run full MQ vs clock analysis with Stockfish, depth=5 or 10, 10K+ positions
- [ ] Run full log(RT) vs VOC with Stockfish, depth=5 or 10
- [ ] Implement E[ΔUC] (expected VOC over top-5 depth-1 candidates)
- [ ] Loop VOC and MQ into `build_selected_moves_with_engine` so the full dataset can be annotated
- [ ] Investigate whether using pwin from Stockfish WDL (vs centipawns → logistic regression) affects results
- [ ] Determine required depth for lc0 to produce meaningful non-zero VOC

---

## 2026-06-04 Cleanup Log

The following files and directories were permanently deleted during a repository cleanup and restructuring process. They are logged here so their paths can be tracked in case they need to be recreated or restored from backup.

### Scratch Data Directories (`/scratch/gpfs/GRIFFITHS/hl4291/tmp/`)
- `pipeline_smoke_dec_edge_a/`
- `pipeline_smoke_oct_dec31/`
- `pipeline_smoke/`
- `pipeline_smoke_full_compare/`
- `pipeline_smoke_boundary/`
- `chess_analysis_legacy_3953ad/`
- `legacy_3953ad_work/`
- `pos_with_engine_eval/`
- `pos_with_engine_eval_100k/`
- `pos_with_engine_eval_1m/`
- `load_moves/`
- `load_moves_20260422_173541_1318563/`
- `ld_moves_shard/`
- `voc_mq_eval/`

### Orphan Scripts and Smoke Tests (`human_analytics/`)
- `slurm/scripts/tests/compare_legacy_new_pipeline_smoke.py`
- `slurm/scripts/tests/boundary_gid_diagnose.py`
- `tests/test_pipeline_compare_smoke.py`
- `tests/test_preprocess_shard_pipeline.py`
