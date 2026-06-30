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
| Halt-policy comparison — do learned readouts beat blind stopping on the SF-2000 budgeted oracle? (regret-vs-compute, PG vs MSE surrogate, GNN-z vs tree-stats) | [(R-LMCOS-TINY)](lmcos_tiny.md) | Scientific | LMCOS | ✅ done (elo2000); seeds/rungs open |
| When ought one think? — calibrating the budgeted-oracle **step\*** against human RT (step\*=0 degeneracy, cost shape/scale, weak-engine rung) | [(R-HALT-CALIB)](halt_calibration.md) | Scientific | LMCOS | 📝 active; step\*↔RT reframe, interim elo2000 |
| **Tree search & deliberation** — the single linear story (mermaid map): when/do people search like an engine? → VOC fails (every signal a legal-moves proxy) → RT = satisficed decision difficulty (size − satisfaction + sharpness) → a meta-rational reward−cost model + the fit-`c`-to-RT program | [(R-TREESEARCH)](treesearch.md) | Scientific | Deliberation | 📝 active; 63k; the single deliberation report (folds in the former R-VOC) |
| **Value-pruning** — changing the **cost profile** (the only live lever once trees are filtered for reward-to-planning) via ε-pruning of implausible leaves; staged plan + feasibility (750K @ n=1) | [(R-PRUNING)](pruning.md) | Plan | Deliberation | ❌ negative — pruned node-count washes out at md36; the win (if any) is the satisficing stop, not the count |
| **Construal** — effort as the **complexity of the simplified game** you build (attended pieces), not tree size; freeze-not-remove with holistic leaf eval; RT ∝ construal size \|S\*\| | [(R-CONSTRUAL)](construal.md) | Plan | Deliberation | ❌ subgame ruled out — proxy |S*|↔RT adds ≤+0.027 over legal; collapses to s_star; do not build |
| **Value of information** — does RT track *belief dynamics over which move is best* (argmax-stabilization `t_stab`, info-gain `ΔH`) rather than value? Cheapest probe; trajectory signals shown decoupled from width+sharpness | [(R-VOI)](value_of_information.md) | Plan | Deliberation | 📝 proposal; trace-only, no regen; `t_stab`/`ΔH` the live candidates |
| **Planning engagement** — is width-dominance a *mixture artifact*? Condition on non-circular deliberation (P-CAP = prior-ply-not-a-capture); do VOC signals re-emerge on deliberated moves? | [(R-PLANNING-ENGAGEMENT)](planning_engagement.md) | Plan | Deliberation | 📝 proposal; DuckDB-only; pilot says likely a clean negative that *defends* the floor |
| **Search shape** — is human search *wide/shallow/optimistic* (depth-limited), and is value in cp or win-prob? Depth contrast (md4/8 vs 36) is the live test | [(R-SEARCH-SHAPE)](search_shape.md) | Plan | Deliberation | 📝 proposal; needs regen (~650K trees); rank/c_puct/depth arms zero-code |
| Data reference — human Lichess dataset + lc0 tree generation | [(R-DATA)](#data-reference-r-data) | Reference | Data | ✅ stable |

Older lmcos work (Apr–May 2026; GNN-pretrain, meta-controller, tree-gen engineering) lives in the
lab notebook's [§ Legacy section](../labnotebook.md#legacy) (the former archive, merged in).

## How the threads relate

The human move-time inquiry now reads as **one arc** — *board → tree-search/deliberation* — all sitting on
[(R-DATA)](#data-reference-r-data) (the human Lichess dataset + the search-tree dataset):

1. **board** — [(R-MOVETIME-BOARD)](board.md): what board features predict think time (legal moves dominate).
2. **tree search & deliberation** — [(R-TREESEARCH)](treesearch.md): the single linear story — does
   value-of-computation / engine-search explain *when* people think? No (every signal a legal-moves proxy) ⇒ RT
   = satisficed decision difficulty ⇒ a meta-rational reward−cost model + the fit-`c`-to-RT program. The
   engine-side analysis [(R-MOVETIME-MODEL)](engine.md) (Gain / MQ / GSS / action gap, oracle tiers) and the
   resource-rational analytics of the legal-moves effect [(R-BRANCH)](branching.md) are the engine- and
   analytics-side of this thread and fold in here; [(R-HALT-CALIB)](halt_calibration.md) calibrates the
   budgeted-oracle step\*.

> **Planned consolidation:** physically merge **engine → VOC** and **branching → construal** (they are the
> engine- and analytics-side of those threads); for now they are grouped here and cross-linked. The
> **LMCOS model-training thread** (GNN pretraining, meta-controller, the fitted-RL baselines) is not tracked as
> standalone reports — record in the lab notebook's [§ Legacy section](../labnotebook.md#legacy), `lmcos/`, and
> `lmcos/slurm/README.md`.

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
  `mc_pipeline.md` P2/P3. *(Supersedes the earlier "α-β cannot make child-WDL" note.)*
- **JAX / `mctx`** (batched MCTS): deferred to v2 — our trees are ragged/dynamically grown, so a
  fixed-size padded-array port is a substantial parity risk; not justified unless CPU-side
  bookkeeping becomes the bottleneck.

#### Paths
- DB: `/scratch/gpfs/GRIFFITHS/hl4291/personal.db`
- Canonical trees: `/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/human_trees`
- Per-tree values cache (parquet): `/scratch/gpfs/GRIFFITHS/hl4291/tree_values_cache`

*Merges the former `R-HUMAN-DATA` (human dataset) and `R-TREEGEN` (tree-generation engineering;
its full R-U1 / R-U1-SPEED / R-BATCHGEN detail remains in git history and the legacy archive).*
