# Human Analytics Lab Notebook

Tracks analyses, decisions, and open questions for the human chess decision-making project.  
Repository root: `/home/hl4291/chess_analysis/`  
Database: `/scratch/gpfs/GRIFFITHS/hl4291/personal.db` (DuckDB)

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
