# Human response time — what board features predict it

**Ref:** `R-MOVETIME-BOARD` · [Index](reference.md)

## What board features predict how long humans think?

Humans don't spend equal time on every move. Working model-free (no engine), we ask which
features *of the position itself* predict how long a human thinks — on 135M non-zero-time
moves from 1.97M Lichess games (60+0, Elo ≥ 2000).

> **Result:** Human response time is driven most by the **width of the decision** (the number of
> legal moves) — more than by material, clock, or game stage.

### How is response time distributed?

Across the full move set, log(response time) is approximately **normal** — i.e. response time is
**log-normal**, not exponential or uniform — so players scale thinking *multiplicatively*
with difficulty.

![log response time — histogram + normal QQ](../figures/movetime_logmt_qq.png)

> **Result:** Response time is log-normal (Weber's law) — the multiplicative scaling that
> justifies working in **log(RT)** throughout.

### Which board features move with response time?

Each board feature gets the canonical response-time dashboard (global + by ply tertile), binned the
same way throughout — **K=10 tie-safe quantile bins** with per-bin SEM (tie-safe = a repeated
integer is kept in one bin via quantile cut-points, so discrete counts don't split across bins;
low-cardinality counts collapse to ≤10 integer points):

![player clock vs response time](../figures/clock_vs_movetime.png)
![legal moves vs response time](../figures/legal_moves_vs_movetime.png)
![own non-pawn material vs response time](../figures/own_material_vs_movetime.png)
![game stage (ply) vs response time](../figures/ply_vs_movetime.png)

| Feature (from position) | r with log(RT) | Reading |
|---|---|---|
| legal moves | **+0.20** | more candidate moves → more to weigh |
| Gain (ΔUC, depth 5) | +0.10 | deeper search demonstrably finds a better move |
| own material | +0.04 | more pieces → more interactions |
| action gap (toptwo) | −0.06 | one move clearly best → less to weigh |

### How do the features relate to each other?

A rank (Spearman) correlation matrix over the board features and log(RT) — no engine:

![Spearman correlation — board features](../figures/board_feature_corr.png)

The legal-move count is the strongest single board-feature tie to RT (ρ ≈ +0.26). Material, clock,
and ply move together (material/clock fall as games progress), so they are largely redundant
with the game-stage axis — yet the legal-move count's RT coupling is *not* reducible to that complex.

> **Result:** The **width** of the decision (the number of legal moves) is the strongest
> board-feature predictor of human response time — stronger than realized value-of-search, and not
> reducible to the ply/material/clock complex.

**The raw legal-move count is the *fundamental* width axis.** A natural worry is that the raw count
is a crude stand-in for a smarter "effective width." It isn't: lc0's policy-prior entropy H(π) — a
plausibility-weighted effective width — predicts RT *worse* than the raw count (ρ +0.24 vs +0.33)
and is subsumed by it (partial ρ(H(π), RT | legal moves) ≈ +0.05). So no policy-weighted refinement
beats the raw legal-move count — it is not a proxy *for* a better width measure, it *is* the operative
one. See the policy-entropy test in [the engine report](engine.md).

## Methods

- **Dataset:** Lichess 10+0 (60+0), Elo ≥ 2000, no berserk; 1.97M games → 135M
  non-zero-`move_time` moves in `processed_moves_nonzero` (4-field FEN, board counts, ply
  tertiles). Dataset details: the human-data reference in the index.
- **Per-feature dashboards:** `/human/movetime_analysis.py` (`clock`,
  `legal_moves`, `own_material`, `ply`) — the canonical `Analyzer` 1×2 (global + a-priori ply
  tertiles), K=10 tie-safe quantile bins with per-bin SEM throughout. Distribution: `move_time_summary.py`.
- **Board-feature matrix:** `movetime_analysis.py boardcorr` — Spearman ρ over Ply,
  Legal moves, Own material, Player clock, log(RT) on a 1M-row reservoir sample. Figures land
  in repo-root `figures/`.

> **Decision:** Work in **log(RT)** (response time is log-normal) and use **Spearman** for the
> board matrix — the structural features are skewed/bounded, so rank correlation is the
> honest screen.

### How is the data built? (pipeline & timing)

Board features are **first-class columns of `processed_moves`, computed in preprocess** (not in
the analysis step), so `board.py` just reads them — one `FEATURES` registry drives the figures,
the per-feature partials, and the correlation matrix alike (only the RT histogram and `ply` are
special-cased). Two kinds of feature:

- **SQL, no engine** — weighted material (`P/N/B/R/Q = 1/3/3/5/9`, incl pawns; verified 0/2014
  mismatches vs python-chess), `material_imbalance`, `in_check` (from the parser's
  `player_in_check`), `prev_move_was_capture` (per-game piece-count lag). Computed inside
  `process_moves`.
- **python-chess (legal-move enumeration)** — `n_captures_avail`, `n_checks_avail`. Featurized
  over **distinct** FENs (lossless — a pure function of the FEN), **massively parallel across a
  Slurm array**, then joined back to every move-instance.

Preprocess params (paths, date window, Elo/clock filters, DuckDB threads/mem) live in the config
`preprocess:` section; `personal_db` is unified with `human_analysis.selected_db_default`.

| SLURM step | what | rough wall time |
|---|---|---:|
| `process_moves` | rebuild `processed_moves[_nonzero]` with the SQL features; dump ~84M distinct FENs → parquet | **~5–10 min** |
| `featurize_positions` `[0–31]` | captures/checks over 84M distinct FENs, 32 tasks × 16 cores = **512 cores**, round-robin over parquet row groups → shards | **~2–3 min** |
| `merge_position_features` | join captures/checks shards into `processed_moves[_nonzero]` (single DB writer) | **~3–5 min** |
| `board_analysis` | in-DB game-level sample (60k games ≈ 1.5M instances); histograms/lowess/scatter, both-unit partials + bootstrap CIs, correlation matrix, ∩-shape test | **~20–30 min** |

Outputs are split by type under `figures/<run>/board/{pdf,png,csv}/`. The featurize array replaces
a ~10-min single-node pass; the whole feature build + board analysis is **≈ 35–45 min** end-to-end.

> **Decision:** Featurize **distinct FENs once** and join back to all instances — computationally
> lossless (features depend only on the FEN) and **not** a statistical dedup: the analysis stays
> per-instance and still reports both per-instance and per-FEN. Whole-corpus featurization is
> unnecessary — `in_check` is a parser column and `prev_move_was_capture` is pure SQL.

*(Downstream, off the same `personal.db`: engine tree-signal analysis `engine_analysis_{pwin,cp}`
≈ **1–1.5 h** each; the normative model tail `pack_trees → train_encoder → pack_root[] →
train_readout_pg → eval` ≈ **3–4 h**.)*
