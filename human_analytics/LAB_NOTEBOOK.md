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
