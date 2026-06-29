# Reports index & data reference

Stable analysis and reference docs for **chess_analysis** (CMC). The [lab notebook](../labnotebook.md) is the chronological log (**Description | Rationale | Status / finding | Reference**).

Reports come in two formats:

- **Scientific-inquiry reports** follow the **scientific template**: *Overview → Results (plot-heavy,
  figures first) → Methods → Appendix (Logs)*. Figures are embedded from the repo-root `figures/`
  (human analytics), with `figures/archive/` for historical snapshots.
- **Reference / engineering / archive reports** keep their original format (summary
  table · `| Step | Status |` procedure · notes).

> **Didactic mini-paper convention (target).** A scientific report is the **source of truth; the Slidev
> decks in `presentations/` are built off it**, so it should read like a Reader's-Digest walkthrough anyone
> can follow: **plain language** (jargon/equations in Methods), the body as **one investigation
> top-to-bottom**, and **every section header a question** (not a finding, and not numbered). Each
> section/subsection closes with a typed callout that carries the takeaway:
> - `> **Decision:** …` — a modeling/methodological choice.
> - `> **Result:** …` — an empirical finding (e.g. "think time is log-normal").
> - `> **Clarification:** …` — resolves a likely misreading (e.g. a confound, not a sign bug).
>
> See `monkey_4iar`'s R-GAZE / R-MODELCMP / R-RECOVERY for the worked pattern.

> **Cross-references live here, not in the reports.** Individual reports avoid citing each other; this
> index is the single place that maps how they relate. Reports are primarily about **scientific questions**.

## Reports

| Inquiry | Report | Format | Thread | Status |
|---------|--------|--------|--------|--------|
| Human move time — what board features predict it (distribution, per-feature dashboards, board correlations) | [(R-MOVETIME-BOARD)](board.md) | Scientific | Human | ✅ done |
| Human move time — does a normative engine model match it? (Gain / MQ / GSS / action gap, oracle-stop tiers, SF-2000 correlations) | [(R-MOVETIME-MODEL)](engine.md) | Scientific | Human | ✅ done (SF-2000); residualized MQ open |
| Legal moves & resource-rational deliberation (why decision width drives RT; mechanisms + predictions) | [(R-BRANCH)](branching.md) | Scientific (draft) | Human | 📝 proposal; P1–P5 open |
| Meta-controller — why is tree value ≈0 for stopping, and the minimal forward plan (folds the former R-MC-READOUT/COST/SIGNAL/DELIB proposals) | [(R-MC-PLAN)](../mc_minimal_plan.md) | Plan | LMCOS | 📝 active; P0–P3 |
| Halt-policy comparison — do learned readouts beat blind stopping on the SF-2000 budgeted oracle? (regret-vs-compute, PG vs MSE surrogate, GNN-z vs tree-stats) | [(R-LMCOS-TINY)](lmcos_tiny.md) | Scientific | LMCOS | ✅ done (elo2000); seeds/rungs open |
| When ought one think? — calibrating the budgeted-oracle **step\*** against human RT (step\*=0 degeneracy, cost shape/scale, weak-engine rung) | [(R-HALT-CALIB)](halt_calibration.md) | Scientific | LMCOS | 📝 active; step\*↔RT reframe, interim elo2000 |
| Meta-rational account of chess deliberation time — RT as optimal compute under a cost (reward−cost meta-MDP; ~1–2 params); fit `c` to RT vs the descriptive ceiling | [(R-METAREASON)](metareasoning.md) | Scientific | LMCOS | 📝 active; 63k; fit-to-RT frontier |
| Data reference — human Lichess dataset + lc0 tree generation | [(R-DATA)](#data-reference-r-data) | Reference | Data | ✅ stable |

Older lmcos work (Apr–May 2026; GNN-pretrain, meta-controller, tree-gen engineering) lives in the
lab notebook's [§ Legacy section](../labnotebook.md#legacy) (the former archive, merged in).

## How the threads relate

The repo centres on the **human move-time inquiry**, all sitting on [(R-DATA)](#data-reference-r-data)
(the human Lichess dataset + the lc0 search-tree dataset):

- [(R-MOVETIME-BOARD)](board.md) — what board features predict think time (legal moves dominate).
- [(R-MOVETIME-MODEL)](engine.md) — whether engine-search quantities (Gain / MQ / GSS / action gap,
  now on Stockfish Elo-2000 trees) and the normative oracle track human RT.
- [(R-BRANCH)](branching.md) — a resource-rational account of the legal-moves effect, with tree mechanisms
  and predictions (draft).

The **LMCOS model-training thread** (GNN encoder pretraining, the meta-controller and its budgeted
baselines) is no longer tracked as standalone reports — its record lives in the lab notebook's
[§ Legacy section](../labnotebook.md#legacy), the `lmcos/` code, and `lmcos/slurm/README.md`.

**Cite:** `[(R-MOVETIME-MODEL)](engine.md)` from this folder; `[(R-MOVETIME-MODEL)](reports/engine.md)` from the notebook.

## Data reference (R-DATA)

Provenance for the two data sources behind the analyses: the human Lichess move dataset (DuckDB)
and the lc0 search-tree dataset. Reference format (facts + decisions), not a scientific-template
report. The full tree-gen engineering blow-by-blow (timing profiles, the rejected pooling/batched
evaluators, parity contracts) lives in git history and the legacy archive; only the load-bearing
facts and decisions are kept here.

### Human Lichess dataset

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

### lc0 search-tree dataset

#### Production generation
- **Faithful lc0-UCI `build_tree`**, `max_depth=10`, value = **raw value-head** (`win − loss`),
  budget **96** expansions, with the repetition-scan guard → **~11 s/tree** (A100; realized depth ~6).
  `edge_wdl_targets` included (the GNN child-WDL target).
- **Canonical set:** ysagiv `human_trees` — lc0 search on **2023 human-game root FENs** (4-field).
  *Use this set* (an earlier 150K `lc0_trees` lexicographic slice of the full FEN universe was
  non-representative and has been deleted — see the move-time model report / `branching.md`).
- Trees are `.pt` payloads (`format=cts_raw_pretrain_example_v5`): `root_position_spec`,
  `oracle_root_moves`, `oracle_final_root_q_values`, `oracle_best_move_index`, `oracle_root_q_trace`,
  `node_features`/`feature_names` (`value,wdl_*,prior`), `parent_index`, `is_expanded`, `depth`.
  Exactly **96 expansions** per tree; total nodes ≈ 96 × the per-node legal-move fan-out (each
  expansion attaches the node's legal children as unexpanded leaves).

#### Engineering decisions (what was tried; what shipped)
- **Cost:** ~16.9 s/tree (A100, budget 96) → **~11 s** with the repetition guard (skip the
  impossible threefold-repetition scan, which was 23% of tree-gen time; replaced by
  `outcome(claim_draw=False)` + explicit halfmove-clock check, provably equivalent while shallow).
- **CPU lane unblocked:** the lc0 binary is CUDA-linked; prepend the venv `nvidia/*/lib` dirs to
  `LD_LIBRARY_PATH` to run it on pure-CPU nodes (BLAS backend, libs load but unused).
- **Rejected (faithfulness-gated):** eval **pooling** (slower in `valuehead` mode, not bit-identical)
  and the in-process **batched evaluator** (NO-GO — only ~2× on the raw path, below the ≥5× gate;
  the bottleneck is Python 112-plane encoding/bookkeeping, not the GPU). Throughput is bought via
  parallelism + the QoS fix (`gpu-short`), not new code.
- **Stockfish swap:** `StockfishDirectEvalProvider` (Elo-limitable) is implemented and ~1,155× faster
  on CPU. It sets `UCI_ShowWDL` and returns a WDL per node, so it **does** produce child-WDL targets
  (the `edge_wdl_targets` backup loop is provider-agnostic). Differences vs lc0: WDL from Stockfish's
  internal eval→WDL model (not a trained value head); **uniform priors** (no policy head). See
  `mc_minimal_plan.md` P2/P3. *(Supersedes the earlier "α-β cannot make child-WDL" note.)*
- **JAX / `mctx`** (batched MCTS): deferred to v2 — our trees are ragged/dynamically grown, so a
  fixed-size padded-array port is a substantial parity risk; not justified unless CPU-side
  bookkeeping becomes the bottleneck.

#### Paths
- DB: `/scratch/gpfs/GRIFFITHS/hl4291/personal.db`
- Canonical trees: `/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/human_trees`
- Per-tree values cache (parquet): `/scratch/gpfs/GRIFFITHS/hl4291/tree_values_cache`

*Merges the former `R-HUMAN-DATA` (human dataset) and `R-TREEGEN` (tree-generation engineering;
its full R-U1 / R-U1-SPEED / R-BATCHGEN detail remains in git history and the legacy archive).*
