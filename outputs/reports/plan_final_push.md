# Plan — final 2-day push (P0–P3)

**Format: engineering/plan doc** (not a scientific report). **Run: `minply15_maxply75`** — every
analysis, figure, and number below is on the ply-windowed run (`config_minply15_maxply75.yaml`,
plies 15–75), not `allply`. Figures land under `figures/minply15_maxply75/{board,engine,normative}`.

**Goal tier.** Lock **Minimum** (people-findings + normative model + open reconciliation) as the
floor; reach **Medium** (model *qualitatively* replicates the human sign pattern) via P3. **Hard**
(quantitative fit) is explicitly out of scope.

**Standing rules.** Bootstrap CIs (percentile) everywhere; measured claims only (verified or
labeled hypothesis); engine-derived metrics stay **side-burnered** unless they pass the P1 trust
test; Spearman *and* Pearson reported (no global swap).

---

## Status ledger — what is already bankable (before P0)

| # | Claim | Evidence | Status |
|---|---|---|---|
| 1 | Response time is log-normal | board run | **bankable** |
| 2 | Legal moves is the strongest RT predictor (ρ≈+0.26), not reducible to ply/clock | board run | **bankable** |
| 3 | Engine value signals (gss/oss/voc/frac_good) collapse onto the move count (ΔR²≤0.0003) | 3-subagent diagnostic | **bankable (negative result)** |
| 4 | action_gap is the one surviving engine signal (monotone −, ΔR²≈0.014, stable in strata) | diagnostic | **provisional** — pwin units, pending P1/P2 |
| 5 | frac_good net of legal moves flips **+** (ambiguity), mirroring action_gap **−** | confound analysis, CI [+0.034,+0.050] | **provisional** — the P1 trust criterion |
| 6 | pwin saturation contaminates the signals (39% root evals on the walls; voc==0 lifted 6.8× in decided positions) | normality + cp-preview subagents | **bankable (methods)** |
| 7 | Normative per-node-cost stop reproduces **+size / −satisfaction** | prior treesearch result | **bankable** |
| 8 | PG tree-stats readout beats GNN-z (0.104 vs 0.154, non-overlapping CIs) | R-LMCOS-TINY | **bankable** |

---

## P0 — Board-correlate battery (engine-free) — *first priority*

**Why.** More orthogonal board regressors like `n_legal_moves` — features of the position itself,
with **no engine as middleman** — as model-checking targets. No SF-quality confound possible.

**New regressors** (FEN- or game-sequence-derived only), each with a **pre-registered
hypothesis** — no feature runs without a stated prediction and a designated meaning for each
outcome. Frame: RT tracks the size and composition of a selectively-built consideration set,
weighted by the value of computation.

| feature | source | hypothesis (pre-registered) | mechanism / what outcomes mean |
|---|---|---|---|
| `in_check` | FEN | raw **−**; partial vs legal moves: **≈0 if size account complete, + if stakes add** | check collapses the legal-reply set (size → faster) vs "risky → think more" (stakes). Mediation split dissociates them — best test in the set. |
| `n_captures_avail` | FEN | **+ net of legal moves** | composition test: captures are salient/forcing candidates; for fixed width, a capture-rich set has more moves worth evaluating → inclusion is salience-weighted, not uniform. ("more captures = winning → fast" is decidedness — carried by `material_imbalance`, not this.) |
| `n_checks_avail` | FEN | **+ net of legal moves** | same composition logic, stronger (checks most forcing; available checks signal tactics → high VOC). No competing story. |
| `self_material` (non-pawn) | FEN | **+ net of legal moves** (null = informative: width-dominated) | depth test: more own pieces → more interactions per candidate line → deeper lookahead per considered move, beyond root width. |
| `material_imbalance` (non-pawn, mover POV) | FEN | **∩-shape, peak ≈ 0** (RT declines in \|imbalance\|) | **decidedness/VOC without the engine**: far ahead/behind ⇒ decided ⇒ computation worthless ⇒ fast; balanced ⇒ hinges on the move ⇒ think. The engine-free version of the Gain prediction. Star of the battery. |
| `prev_move_was_capture` / `prev_move_was_check` | DB move sequence | **−** | opponent's forcing move constrains the reply (recapture/escape is the default candidate) — `in_check`'s milder cousin. |

*Deferred (not in this battery):* `n_threatened` / `n_attacked` — the threatened-vs-attacked
distinction (en-prise rules, defense, attacker values) is too easy to misinterpret; parked for a
later iteration.

**Method.** python-chess over the analysis FENs (windowed [15,75]); join like the engine signals;
same 1×3 dashboards; extended Spearman+Pearson matrix; partials vs `n_legal_moves` (each new
feature must show it is not just mobility in disguise); binned shape plot (not just a scalar) for
`material_imbalance`'s ∩ test.

**Theory map** (what the battery tests, feature → question):
salience-weighted inclusion (`n_captures_avail`, `n_checks_avail`) · engine-free decidedness/VOC
(`material_imbalance`) · size-vs-stakes dissociation (`in_check`, `prev_move_*`) · width-vs-depth
(`self_material`).

**Intermediate deliverables (user checkpoints):**
- **P0.a** — feature table computed + histogram panel of the new features *(sanity check: distributions look right before any RT analysis)*
- **P0.b** — extended correlation matrix (board features × RT, Spearman + Pearson, bootstrap CIs) + partials vs legal moves
- **P0.c** — dashboards for the 2–3 strongest new features; report section drafted into `minply15_maxply75.md`

**Est.** ~half day. **Risk:** low — all cheap, deterministic computation.

### Computation (P0 features)

*All facts below verified against the live DB (2026-07-02), not assumed.*

**Data source.** DuckDB `/scratch/gpfs/GRIFFITHS/hl4291/lmcos/personal.db` (open read-only). Tables:
`processed_moves` / `processed_moves_nonzero` (`gid, move_ply, fen, n_possible_moves, move_time,
n_self_pieces_exc_pawns, n_opp_pieces_exc_pawns, …`) and `moves` (`gid, move_ply, move_uci,
player_in_check, n_pieces, board_position, …`); also `games`, `pos_with_engine_eval`. Join key
**(gid, move_ply)** — verified: every windowed `processed_moves_nonzero` row has a matching
`moves` row (0 misses). The `fen` column is 4-field (no move clocks) but **does** carry the
en-passant square when set (73/50,000 sampled rows); `chess.Board(fen)` parses it directly
(python-chess 1.11.2 importable with `PYTHONPATH=src`), so legal-move / capture enumeration is
ep-correct. `n_possible_moves` (the legal-moves partial-out variable) is already a column.

| feature | already in DB? | recipe | cost |
|---|---|---|---|
| `in_check` | **yes** — `moves.player_in_check` (verified ≡ `board.is_check()` on 1,961 sampled windowed rows, 0 mismatches; base rate 5.3%) | join `moves` on `(gid, move_ply)` | free (SQL) |
| `n_captures_avail` | no | `sum(board.is_capture(m) for m in board.legal_moves)` on `fen` | python-chess pass |
| `n_checks_avail` | no | `sum(board.gives_check(m) for m in board.legal_moves)` | same pass (`gives_check` is the dominant cost) |
| `self_material` (non-pawn) | raw count **yes** (`n_self_pieces_exc_pawns`) | **raw piece COUNT** (N/B/R/Q, unweighted, king excluded): `sum(len(board.pieces(pt, board.turn)))`. *User decision:* unweighted for now — keeps the metric fully under our control and exactly cross-checkable against the DB column; value-weighting (3/3/5/9) deferred, "build up from there". | same pass |
| `material_imbalance` (non-pawn, mover POV) | raw diff derivable in SQL (`n_self − n_opp`) | `self_material − opp_material`, **raw counts** (see above), side-to-move POV; cross-checkable against the DB columns in SQL | same pass |
| ~~`n_threatened` / `n_attacked`~~ | — | **deferred** (see hypothesis table note) | — |
| `prev_move_was_capture` | ingredients **yes** (`moves.n_pieces`) | pure SQL: `n_pieces < lag(n_pieces) OVER (PARTITION BY gid ORDER BY move_ply)` — the piece count drops **iff** the previous move captured (en passant included; promotions don't change the count). Verified against `board_prev.is_capture(prev_uci)` on 5,000 windowed plies: the only disagreements were en-passant captures, which the count-drop recipe classifies correctly and the naive prev-FEN recipe misses. | free (SQL) |
| `prev_move_was_check` | — | **degenerate — drop.** By the rules of chess the side to move is in check *iff* the opponent's last move delivered check, so this feature is bit-identical to `in_check`. Keep `in_check` only. | — |

**Cost & execution.** Window [15,75] on `processed_moves_nonzero`: **89,218,280 rows /
84,034,834 distinct FENs**. Measured featurization rate (all five FEN-derived features in one
pass, incl. the `gives_check` loop, on 2,000 sampled windowed FENs): **~4,700 FENs/s
single-core** → ~5.0 core-hours for all distinct FENs. Not a login-node job at full scale: run a
short SLURM CPU array (e.g. 32 workers partitioned by `hash(fen)` ≈ 10 min wall; ≤1 h → `test`
QOS per the della routing rule). If P0 instead runs on the tree-joined FEN subset (~405K roots,
as the engine-signal analyses did), the pass is ~90 s single-core and can run locally.
`in_check` and `prev_move_was_capture` need no featurization pass at all (SQL only).

**Open decisions (flagged, need user call):**
1. `prev_move_was_check` is logically identical to `in_check` (see table) — proposal: drop it and
   report the hypothesis under `in_check` / `prev_move_was_capture`.
2. Material features: weighted (3/3/5/9) chosen here over raw piece count — if raw is preferred
   it is free (columns already exist), but the two should not both enter the matrix un-noted.
3. Scope of the featurization pass: **all 84M distinct windowed FENs** (~5 core-hours, SLURM) vs
   **only the analysis-join subset** (minutes, local). The correlation battery only needs the
   joined set; the full pass is only worth it if we want the features materialized as a reusable
   DB-side table.

---

## P1 — Engine trust test (existing trees; no regen)

**Why.** Pre-registered criterion for whether *any* engine metric graduates off the side burner.

**The trust criterion (stated in advance).** `action_gap` and `frac_good` are antithetical by
construction, so net of legal moves their RT trends must show **opposite signs** with
non-overlapping bootstrap CIs. Current evidence is consistent (+0.04 vs −0.05 partials) but in
saturated pwin units → *provisional, not confirmed*.

**New metric — `frac_acceptable`** (the user's absolute-threshold variant): fraction of root moves
above an **absolute** value bar (pwin ≥ 0.5; sensitivity at 0.4/0.6), instead of ε-from-best.
Satisficing against an aspiration level, decoupled from action_gap by construction. Computable
**today** from cached tree child values (no regen).

**Tasks.**
1. Recompute the action_gap / frac_good partial-mirror on the **windowed [15,75]** join, both units of analysis (per-instance + per-FEN), bootstrap CIs.
2. Compute `frac_acceptable` from cached trees; distribution check (point masses?); RT relationship net of legal moves.
3. Verdict table: which engine metrics pass; everything else explicitly side-burnered in the report.

**Intermediate deliverables:**
- **P1.a** — trust-criterion figure: action_gap vs frac_good RT partials side by side, CIs *(the go/no-go picture)*
- **P1.b** — `frac_acceptable` histogram + RT dashboard + partials
- **P1.c** — one-paragraph verdict: which metrics we trust and why, drafted for the report

**Est.** ~2–3 h (cached data). **Risk:** low-medium — the mirror may fail; that outcome is still a deliverable (engine section stays honest-negative).

---

## P2 — cp-BeFS tree regen (gated on P0+P1 landing by end of day 1)

**Why.** (a) cp is **Stockfish's native unit** — search/eval run on integer `Value`
(types.h; `cp = v·100/PawnValueEg`, uci.cpp:309); WDL is a display-layer overlay (uci.cpp:320).
Building the tree in cp respects the engine's own ordering. (b) Greedy **best-first has no
`c_puct`** — no exploration/value scale coupling — and cp resolves the frontier ties that pwin
saturation creates (46% of root children on the ±1 rail). (c) Storing **both** cp/mate and WDL
per node makes every readout (either unit) share **one topology**.

**Scope guard.** Analysis trees only — the encoder/readout training stack is untouched.

**Tasks.**
1. `parsers.py`: capture `score cp|mate` alongside WDL (additive; `TREE_ENCODER_FEATURE_NAMES` untouched — no checkpoint breaks).
2. Tree builder: pure greedy BeFS selection (priority = native Value), replacing PUCT for this tree set; new tree key (e.g. `befs_cp_md36`).
3. Regen on the [15,75] FEN sample (`gen_trees` array; ~1 h wall, ~55 G — disk verified, 1.4 T free); recompute vals/rootmoves caches with signals in **both units**.
4. Re-run the P1 trust test in cp; compare pwin-vs-cp point-mass fractions (predicted: action_gap zeros 19%→~2%, per the cp preview).

**Intermediate deliverables:**
- **P2.a** — parser + BeFS diff for review **before** launching any regen
- **P2.b** — 1-shard pilot (~2.5 k trees): cp/pwin stored correctly, tree shapes sane *(go/no-go before the full array)*
- **P2.c** — full regen + recomputed signal caches; pwin-vs-cp distribution comparison figure
- **P2.d** — P1 trust test rerun in cp; updated verdict

**Est.** ~half day incl. validation. **Risk:** medium — new search rule means new numbers; if the pilot looks odd we stop and ship P1's pwin verdict. **Gate:** only starts if P0+P1 are done and reviewed.

---

## P3 — Model-vs-human sign table (the Medium deliverable)

**Why.** Medium = the normative model **qualitatively replicates the human sign pattern**. Anchor
on board-level facts (trustworthy); engine metrics enter only if they passed P1/P2.

**Method.** Treat the model's stop step (oracle gss/oss; PG controller's halt step where
available) as the model's "response time"; run the **same** correlate analysis on it as on human RT;
place the sign columns side by side.

| effect | humans (RT) | model (stop step) |
|---|---|---|
| size (`n_legal_moves`) | **+** (bankable) | **+** (shown; recompute on this run) |
| satisfaction (mq) | **−** (bankable) | **−** (shown; recompute on this run) |
| decisiveness (action_gap) | **−** (provisional → P1/P2) | to compute (cheap) |
| P0 board correlates | from P0 | same features, model positions |

**Intermediate deliverables:**
- **P3.a** — sign table computed (model column) on minply15_maxply75
- **P3.b** — report section: "Does the model allocate thinking like people?" — measured language, each row tagged bankable/provisional
- **P3.c** — final assembly in `minply15_maxply75.md`: Minimum framing wrapping the Medium result; explicit open-questions section (quantitative fit, cp-native modeling if P2 skipped, noise-injection line formally closed)

**Est.** ~half day. **Risk:** low for the two anchored rows; decisiveness row may stay "provisional" — acceptable.

---

## Sequence & checkpoints

```
Day 1 AM   P0 (board battery)          → checkpoints P0.a–c
Day 1 PM   P1 (trust test, cached)     → checkpoints P1.a–c
           ── GATE: review P0+P1 ──
Day 2 AM   P2 (cp-BeFS regen)  [if gated in]   → P2.a diff review → pilot → full
Day 2 PM   P3 (sign table + report assembly)   → P3.a–c
```

**Dropped from scope** (explicit): noise-injection / rational-inattention policy-resolution
(subsumed by the normative model; falsified variants noted in future work); global Spearman→Pearson
swap (both reported); Hard tier.

**Also running (independent of this plan):** allply pipeline tail — train_readout_pg 10592943 →
eval 10592944; results fold into P3 if landed.
