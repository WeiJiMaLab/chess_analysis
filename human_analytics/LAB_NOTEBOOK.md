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

#### `voc_mq_analysis.py` (renamed from `voc_mq_exploration.py`)

Exploratory script: samples middlegame positions (`move_ply > 10`) from `processed_moves_nonzero`, evaluates MQ and VOC, plots:
- `mq_vs_clock.png` — MQ vs player clock time
- `rt_vs_voc.png` — log(RT) vs VOC
- `voc_distribution.png` — histogram of VOC
- `voc_variance_dist.png` — histogram of per-position VOC std (for n_repeats > 1)
- `voc_mean_vs_std.png` — scatter of VOC mean vs std

#### `timing_comparison.py`

Times both engines at depths [1, 5, 10] on a small middlegame sample and extrapolates to 10K / 100K positions.

#### `human_analytics/tests/test_engine_analysis.py`

9 unit tests for MQ and VOC contracts (sign, non-positivity, blunder detection, forced-move VOC=0, tactical VOC>0). All pass against Stockfish SF14.

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
