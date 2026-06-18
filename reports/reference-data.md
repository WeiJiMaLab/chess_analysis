# Reference — data: human dataset & lc0 tree generation

**Ref:** `R-DATA` · [Index](README.md)

Provenance for the two data sources behind the analyses: the human Lichess move dataset (DuckDB)
and the lc0 search-tree dataset. Reference format (facts + decisions), not a scientific-template
report. The full tree-gen engineering blow-by-blow (timing profiles, the rejected pooling/batched
evaluators, parity contracts) lives in git history and the legacy archive; only the load-bearing
facts and decisions are kept here.

## Human Lichess dataset

| | |
|---|---|
| **What** | Lichess **10+0** games (Oct–Dec 2023), preprocessed via `human_analytics/slurm/`, in DuckDB `personal.db`. |
| **Filters** | Time control 10+0 (600s, no increment); both players Elo **≥ 2000**; excluded: negative `move_time`, berserk, extra-time grants. |
| **Scale** | **1.97M** games; **135M** non-zero-RT moves. |

| Table (`/scratch/gpfs/GRIFFITHS/hl4291/personal.db`) | Rows | Notes |
|---|---|---|
| `games` | 1,971,698 | selected games |
| `moves` | 145,142,731 | raw moves (incl. `move_uci`) |
| `processed_moves` | 145,142,731 | + 4-field `fen`, piece counts, `ply_tertiles` |
| `processed_moves_nonzero` | 135,482,903 | `move_time > 0` (the analysis table) |

`processed_moves` adds `n_pieces_on_board_{inc,exc}_pawns`, `n_self_pieces_exc_pawns`,
`n_opp_pieces_exc_pawns`, `fen` (4-field), `ply_tertiles`.

## lc0 search-tree dataset

### Production generation
- **Faithful lc0-UCI `build_tree`**, `max_depth=10`, value = **raw value-head** (`win − loss`),
  budget **96** expansions, with the repetition-scan guard → **~11 s/tree** (A100; realized depth ~6).
  `edge_wdl_targets` included (the GNN child-WDL target).
- **Canonical set:** ysagiv `human_trees` — lc0 search on **2023 human-game root FENs** (4-field).
  *Use this set* (an earlier 150K `lc0_trees` lexicographic slice of the full FEN universe was
  non-representative and has been deleted — see the move-time model report / `branching.md`).
- Trees are `.pt` payloads (`format=cts_raw_pretrain_example_v5`): `root_position_spec`,
  `oracle_root_moves`, `oracle_final_root_q_values`, `oracle_best_move_index`, `oracle_root_q_trace`,
  `node_features`/`feature_names` (`value,wdl_*,prior`), `parent_index`, `is_expanded`, `depth`.
  Exactly **96 expansions** per tree; total nodes ≈ 96 × branching (each expansion attaches the
  node's legal children as unexpanded leaves).

### Engineering decisions (what was tried; what shipped)
- **Cost:** ~16.9 s/tree (A100, budget 96) → **~11 s** with the repetition guard (skip the
  impossible threefold-repetition scan, which was 23% of tree-gen time; replaced by
  `outcome(claim_draw=False)` + explicit halfmove-clock check, provably equivalent while shallow).
- **CPU lane unblocked:** the lc0 binary is CUDA-linked; prepend the venv `nvidia/*/lib` dirs to
  `LD_LIBRARY_PATH` to run it on pure-CPU nodes (BLAS backend, libs load but unused).
- **Rejected (faithfulness-gated):** eval **pooling** (slower in `valuehead` mode, not bit-identical)
  and the in-process **batched evaluator** (NO-GO — only ~2× on the raw path, below the ≥5× gate;
  the bottleneck is Python 112-plane encoding/bookkeeping, not the GPU). Throughput is bought via
  parallelism + the QoS fix (`gpu-short`), not new code.
- **Stockfish swap:** held as a contingency only — far faster on CPU but α-β, so it produces no
  value-network WDL and cannot make the child-WDL target.
- **JAX / `mctx`** (batched MCTS): deferred to v2 — our trees are ragged/dynamically grown, so a
  fixed-size padded-array port is a substantial parity risk; not justified unless CPU-side
  bookkeeping becomes the bottleneck.

### Paths
- DB: `/scratch/gpfs/GRIFFITHS/hl4291/personal.db`
- Canonical trees: `/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/human_trees`
- Per-tree values cache (parquet): `/scratch/gpfs/GRIFFITHS/hl4291/tree_values_cache`

*Merges the former `R-HUMAN-DATA` (human dataset) and `R-TREEGEN` (tree-generation engineering;
its full R-U1 / R-U1-SPEED / R-BATCHGEN detail remains in git history and the legacy archive).*
