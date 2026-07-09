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
> - `> **Result:** …` — an empirical finding (e.g. "response time is log-normal").
> - `> **Clarification:** …` — resolves a likely misreading (e.g. a confound, not a sign bug).
>
> See `monkey_4iar`'s R-GAZE / R-MODELCMP / R-RECOVERY for the worked pattern.

> **Cross-references live here, not in the reports.** Individual reports avoid citing each other; this
> index is the single place that maps how they relate. Reports are primarily about **scientific questions**.

## Plot standards (board.py / engine.py)

Binding house style for every RT-vs-covariate figure produced by `analysis/board.py` and
`analysis/engine.py` — the two modules share one visual language because their figures are
interleaved in the same reports/decks. A change to one must be mirrored in the other.

**Layout.** Every bivariate dashboard is a fixed **1×3**: `[marginal histogram] [overall trend]
[trend by ply tertile]`. This applies to board.py's `bivariate_analysis` (already 1×3) *and*
engine.py's per-signal tree dashboards (`save_tree_dashboard`, currently 1×2 — must gain the
histogram panel to match).

**Axis labels.**
- Histogram x-axis: the bare variable name (e.g. "Legal Moves") — it's a marginal distribution,
  not a binned trend, so no "(bin)" suffix.
- Trend-panel x-axis: the **capitalized** variable name, with a "(bin)" suffix **only** when x is
  actually quantile-binned (`bin_mode="ntile"`, no zero-inflation/tie-safe/edge-mass — see
  `Analyzer._x_axis_label`).
- **Integer-valued covariates are never quantile-binned** — they use `bin_mode="integer"` (one
  point per legal value, sparse tail merged into a single point at the display clip). No "(bin)"
  suffix; the axis just reads e.g. "Legal Moves".
- **Zero (or other) point-mass isolation:** when quantile-binning a continuous covariate that has
  a large mass at one value (canonical case: Gain==0 in engine.py, ~55% of rows), split that mass
  into its own dedicated point (`zero_inflated=True` / `edge_mass=...`) rather than letting `ntile`
  smear it across bins. That point must (a) be folded into the axis label so the reader knows it's
  not an interior bin (e.g. "Gain (bin; 0 isolated)"), and (b) render as an **✕ marker**, not a
  circle, on every trend panel it appears in — visually flagging it as a qualitatively different
  kind of point (a hurdle mass, not a quantile).
- **Material Imbalance tracks BOTH the signed and absolute value, as two separate features.**
  `material_imbalance` is the natural SIGNED quantity (self − opponent, weighted, mover POV — matches
  the raw DB column), and `abs_material_imbalance` is its explicit absolute-value twin — each gets
  its own bivariate dashboard. Weighting is assumed everywhere now, so neither label spells it out:
  just `"Material Imbalance"` and `"Material Imbalance (Absolute)"` (and `self_material`'s is just
  `"Self Material"`) — no "(Weighted)" suffix. The signed feature's sparse tails at BOTH ends are
  merged (`integer_floor_cut` mirrors `integer_tail_cut` — see `Analyzer`), not display-excluded like
  a naturally-nonnegative covariate's sparse floor (`self_material` below 14) — dropping "mover is
  way behind" rows would bias a signed variable.
- **Supplementary confound-removed dashboards (`supp_*`).** When a wrinkle in a covariate's curve is
  explained by a compositional confound (e.g. legal-moves' 9–10 dip being an in-check artifact —
  see `board.md`), prefer re-running the SAME `bivariate_analysis` on a filtered sub-population and
  letting the reader see the wrinkle vanish, over a table-and-prose argument. `run_plot`'s
  `supplements` dict in board.py is the registry for these — add an entry rather than writing a new
  one-off table when a future wrinkle has a clean confound-removal story.

**n annotation.** Top-left, above the plot, reading `n = {count:,}`. This must be the actual row
count considered by *that specific* analysis (recomputed per call — never a cached/hardcoded
number), so it stays honest under smoke mode, clipping, or filtering. Font size is
`LEGEND_FONTSIZE` (`helpers.py`) — the **same** size as every legend in the figure, exactly, not
just "close to." `_annotate_n` (`utils/plots.py`) and every `ax.legend(...)` call across
board.py/engine.py/`Analyzer` share this one constant — don't hardcode a legend fontsize elsewhere.

**No titles, anywhere.** No `fig.suptitle`, no per-axes `ax.set_title`. The n= annotation plus
axis labels (and legend, where present) carry all the figure's text — a title is redundant with
them and this house style omits it uniformly (feature histograms, correlation matrices, RT
distribution, bivariate dashboards, engine dashboards — all of it).

**Color.** The base series color (`MAIN_COLOR`) is a **blue-leaning indigo** (`#6378f1` — the
presentations' indigo accent, `#6366f1` in `style.css`/`VOC` in `make_rt_figures.py`, with its hue
nudged ~35% toward Tailwind blue-500 so it reads a bit more blue) — histogram bars, trend lines,
boolean bar charts, all of it. `PHASE_COLORS` (the ply-tertile palette) is a light/medium/dark
3-tone ramp at the **same hue** (`#a7b2f1` / `#3e57ea` / `#0c2197`, early→late), so the base series
and the ply-tertile segmentation read as one consistent color family rather than two different
blues.

**Smoke tests.** A fast-iteration mode samples **~50,000 rows** (not the full 80–135M) from the
relevant table. Output artifacts are distinguished with a `smoke_` filename prefix (established
convention). The n= annotation always reflects the *actual* sampled/observed row count post-filter
— never the nominal sample size requested.

**RT-distribution summary figure (`move_time_summary`) specifics.**
- The normal-fit panel is a **P-P plot** (probability–probability), not a QQ plot in raw ln(RT)
  units: both axes run **0 to 1** — x = the theoretical quantile probability, y = the empirical
  quantile mapped through the fitted `Normal(mean, std)` CDF. Under perfect normality every point
  sits on `y = x` regardless of RT's own scale, so this reads the same way across runs/configs.
- The ply-vs-RT panel (whole game, unwindowed) draws an `axvspan` + below-axes `legend()` marking
  the analysis window on a `constrained_layout` multi-panel figure — a reproduced matplotlib
  layout-engine bug corrupts that axis' tick formatter into a 2-entry `FixedFormatter` keyed off
  the span's own bounds (ticks overlap, render blank past the first two) unless the locator AND
  formatter are explicitly reset (`mticker.MaxNLocator` / `mticker.ScalarFormatter`) after the
  `axvspan`+`legend` calls. Any new panel combining `axvspan` + a below-axes legend on a
  `constrained_layout` figure needs the same explicit reset.

**Numbers cited in reports (pgfvals).** Every headline statistic board.py actually computes (RT
mean/median, per-feature Pearson r vs. log RT, …) is registered under a stable key via
`analysis.utils.pgfvals.pgf_set(key, value, fmt)` and dumped to one canonical file,
`outputs/reports/board_stats.tex`, as `\pgfkeyssetvalue{key}{value}` lines (`\pgfkeysvalueof{...}`
to use one in LaTeX). This only happens on a real (non-`--smoke`) run — a smoke sample must never
overwrite the canonical numbers. When a report cites one of these numbers, name its key alongside
the value (e.g. "mean 6.5 s (key: `board/rt/mean_s`)") so a reader can trace it back to the exact
line that produced it, instead of the prose and the code silently drifting apart.

## Reports

| Inquiry | Report | Format | Thread | Status |
|---------|--------|--------|--------|--------|
| Human response time — what board features predict it (distribution, per-feature dashboards, board correlations) | [(R-MOVETIME-BOARD)](board.md) | Scientific | Human | ✅ done |
| **Tree search & deliberation** — the single linear story: does VOC/engine-search explain *when* people think? No → RT = satisficed decision difficulty (size − satisfaction + sharpness) → **the reclaimed result: a resource-rational per-operation-cost stop *reproduces* the decomposition** (legal-moves is the explanandum, not a floor). Folds in the former engine / branching / halt-calibration / VOC threads. | [(R-TREESEARCH)](treesearch.md) | Scientific | Deliberation | 📝 active; `normative_curves` reproduces +size/−satisfaction/+sharpness |
| **Value-pruning** — the cost-profile lever via ε-pruning; staged plan + the (negative) regen result; pruning is a *later refinement* of the resource-rational fit, not the first step | [(R-PRUNING)](pruning.md) | Plan | Deliberation | ❌ pruned node-count washes out at md36; success reframed to *reproduce the curves* |
| ~~Normative controller assessment~~ — **retracted, see R-TREEGEN-POISONED**. | ~~R-EVALUATE~~ | — | Deliberation | 🗑️ deleted 2026-07-06 |
| ~~Budget-regime collapse~~ — **retracted, see R-TREEGEN-POISONED**. | ~~R-BUDGET-COLLAPSE~~ | — | Deliberation | 🗑️ deleted 2026-07-06 |
| **Tree-gen is poisoned** — the entire `befs1cp_md36` BeFS tree corpus (and everything packed/trained/evaluated from it: encoder, controller checkpoints, R-EVALUATE, R-BUDGET-COLLAPSE) is invalid. Root cause: unvisited root-move edges default to `q_value=0.0` instead of a first-play-urgency fallback, so on a zero-centered value scale an untested move silently beats any move that was explored and found merely mediocre — the "best move" `oracle_best_move_index`/`oracle_final_root_q_values` (and hence `halt_reward`/`R(t)`) trace for 63% of the 250K-tree corpus is an artifact of elimination, not validation (96.4% within the `gss=0` bucket alone). Compounded by greedy BeFS having no exploration term: across sampled trees, exactly 1 of 20-38 root children ever gets expanded past one ply. Raw trees, packed/materialized/encoder/controller data, and all downstream figures/reports deleted 2026-07-06 (~148GB) rather than left to mislead a future read. `board.md` is unaffected (no tree-corpus dependency); `treesearch.md`'s core §5-9 result and `pruning.md` predate the switch to BeFS and used the older PUCT-based trees, also unaffected. | (no report yet — this row documents the retraction until one is written) | Scientific | Deliberation | ❌ tree-gen fix (FPU fallback + re-enable exploration) required before any engine-driven analysis on this corpus can be trusted again |
| Data reference — human Lichess dataset + SF/lc0 tree generation | [(R-DATA)](#data-reference-r-data) | Reference | Data | ✅ stable |

Older lmcos work (Apr–May 2026; GNN-pretrain, meta-controller, tree-gen engineering) lives in the
lab notebook's [§ Legacy section](../labnotebook.md#legacy) (the former archive, merged in).

## How the threads relate

The human response-time inquiry now reads as **one arc** — *board → tree-search/deliberation* — all sitting on
[(R-DATA)](#data-reference-r-data) (the human Lichess dataset + the search-tree dataset):

1. **board** — [(R-MOVETIME-BOARD)](board.md): what board features predict response time (legal moves dominate).
2. **tree search & deliberation** — [(R-TREESEARCH)](treesearch.md): the single linear story — does
   value-of-computation / engine-search explain *when* people think? No (every value signal is faint and
   collapses to the move count) ⇒ RT = satisficed decision difficulty ⇒ **the reclaimed result: a
   resource-rational per-operation-cost stop *reproduces* `size − satisfaction + sharpness` (legal-moves is the
   explanandum, not a floor to beat).** The engine value signals (Gain / MQ / GSS / action gap, in
   `/human/engine.py`, `figures/engine/`), the resource-rational width analytics, and the step\*
   calibration are **all folded into this report** (the former standalone engine / branching / halt-calibration
   reports were removed).
3. **value-pruning** — [(R-PRUNING)](pruning.md): pruning as a *later refinement* of the resource-rational fit.

> The **LMCOS model-training thread** (GNN pretraining, meta-controller, PG halt-policy zoo) is not tracked as a
> standalone report — it lives in `lmcos/`, `lmcos/slurm/README.md`, and the lab notebook's
> [§ Legacy section](../labnotebook.md#legacy).

## Data reference (R-DATA)

Provenance for the two data sources behind the analyses: the human Lichess move dataset (DuckDB)
and the lc0 search-tree dataset. Reference format (facts + decisions), not a scientific-template
report. The full tree-gen engineering blow-by-blow (timing profiles, the rejected pooling/batched
evaluators, parity contracts) lives in git history and the legacy archive; only the load-bearing
facts and decisions are kept here.

### Human Lichess dataset

| | |
|---|---|
| **What** | Lichess **10+0** games (Oct–Dec 2023), preprocessed via `slurm/analysis/`, in DuckDB `personal.db`. |
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
  non-representative and has been deleted — see the lab notebook).
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
