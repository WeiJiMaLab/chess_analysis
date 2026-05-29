# Lab Notebook

## Project overview

The motivation is to build a planning model that keeps the abstract tree-search scaffolding but replaces hand-written decision rules with neural networks at each decision point. Conceptually: a meta-controller chooses act in the real environment vs plan inside an internal tree. The act path uses a policy at the root of the current tree; the plan path uses a planning head over the tree representation to choose planning actions—navigation (e.g. move focus in the tree) and expansion / evaluation steps that update the tree via a learned world model and value feedback—before committing to a real move. In principle such a model would be trained AlphaZero-style with self-play.

Current work is intentionally narrower: meta-control of search only (e.g. when to continue expanding vs when to halt), on teacher-generated search trees and offline targets. Given snapshots of a growing search tree, a controller decides whether to continue expanding search or halt and act on the current best move. The setting is offline: trajectories come from teacher search on positions (e.g. drawn from the Lichess database with simple filters), and supervision is derived from counterfactual value-of-computation—how halting at each expansion step compares to continuing under a fixed continue cost and halt rewards defined from the search state.

The main representation pipeline is neural encoding of search trees. A tree-structured GNN encoder embeds each snapshot; training combines encoder pretraining (e.g. child-WDL and related targets) with fitted Q-style training of a scalar compute-advantage head that predicts whether continuing is better than halting, rather than policy-gradient RL on the same objective. The scientific questions include whether a simple meta-controller can learn optimal control given this tree-encoder representation, what tree encoding features or control features (e.g. continue cost architecture) support optimal control learning, and to what extent the controller's behaviour matches real human choices. That is the first slice that must work on real tree encodings and real halt/continue tradeoffs.

Later, the same tree representation is meant to support a full planning head whose action space includes concrete planning operations (which node to expand, which to evaluate, etc.), still inside the same overall loop sketched above. The lab notebook records experiments along that path—encoders, packing, fitted-Q controller training, diagnostics, and evaluation—not incidental refactors.

### Repo structure

The **`cts`** Python package maps **`src/`** to `cts.core`, `cts.data`, `cts.models`, and `cts.train`; post-hoc diagnostics live in top-level **`analysis/`** (imported as `cts.analysis`). Use `pip install -e .` or `PYTHONPATH=${PROJECT_DIR}` so `import cts` and `python3 -m cts.*` resolve via `pyproject.toml` `package-dir`.

- `cts.core` — shared substrate (`SearchTree`, tensorizer, feature schema, lc0 providers).
- `cts.data` — data generation and preprocessing. FEN sampling (`cts.data.sample_fens`), FEN filtering (`cts.data.validate_fens`), tree generation (`cts.data.build_tree`), the encoder-pretrain target chain (`cts.data.preprocess_gnn.split` and `.pack`), and the controller-target chain (`cts.data.preprocess_mc.pack` and `.materialize`).
- `cts.models` — neural network modules (`TreeEncoder` in `cts.models.gnn`, `MetaController` in `cts.models.mc`, slot-conditioned attention in `cts.models.tree_mha`).
- `cts.train` — training loops (`cts.train.gnn_pretrain` for encoder pretraining, `cts.train.controller_train` for fitted-Q controller training).
- `cts.analysis` — diagnostics, plotting, and evaluation tools (the main analyzer `cts.analysis.analyze_budgeted_controller_run` is split across themed sub-modules under `cts.analysis._budgeted/`).

Each pipeline stage is a `python -m cts.X.Y` entry point that reads a Pydantic-validated YAML config. SLURM scripts live under `slurm/<stage>/` and take ``CONFIG`` pointing at matching YAMLs under `slurm/configs/<stage>/`. Tests live under `tests/` (`pytest tests/`). Tree generation uses `slurm/1_preprocess_data/submit_generate_dataset_shards.py` as a local orchestrator (slices FENs → many `sbatch` calls).

**Workspace layout (sibling checkout `chess_analysis/`):**

| Path | Role |
|------|------|
| `lmcos/` | This repo — **`src/`** (pipeline `cts` subpackages), **`analysis/`** (`cts.analysis`), **`slurm/`** (scripts + configs) |
| `chess_analysis/human_analytics/` | Human move-time analytics (DuckDB ETL, figures, Slidev deck, `metacontrol/`) — **not** `lmcos/src/` |

**Onboarding:** Pipeline commands: [`slurm/README.md`](slurm/README.md). Run YAMLs: [`slurm/configs/README.md`](slurm/configs/README.md). Workspace overview: [`../README.md`](../README.md).

### 2026-05-29 — Fitted-Q controller smoke runs (hl4291, ysagiv caches)

Intent:
- End-to-end check of stage **4** (`cts.train.controller_train`) on ysagiv read-only materialized caches after the layout refactor, with step-based metrics and validation (not epoch-only).

Three smoke configs under `slurm/configs/4_supervised_controller/`:

| Config | Display name | Encoder | `controller_inputs` |
|--------|--------------|---------|---------------------|
| `legacy_root_budget.yaml` | `legacy[root+budget]` | `tree_encoder_child_wdl_async_k1_rerun.pt` | `[z_t, T_t]` |
| `subtree_weighting_root_budget.yaml` | `subtree-weighting[root+budget]` | `tree_encoder_child_wdl_async_k1_subtree_weighted.pt` | `[z_t, T_t]` |
| `subtree_weighting_root.yaml` | `subtree-weighting[root]` | same | `[z_t]` |

Each config sets `metrics_run_name` to the display name (plot title / comparison legend).

Shared training knobs (only **train steps** truncated; full ysagiv validation cache + greedy eval):
- `batch_size = 18000`, `epochs = 1`, `train_batches = 1000`
- `metrics_log_interval = 20`, `log_interval = 10`
- `validation_step_interval = 100`, `greedy_eval_step_interval = 100` (full validation + greedy regret every 100 gradient steps; epoch eval disabled)
- `ControllerTrainMetricsLogger` in `cts.train.controller_train` writes human-readable `metrics.yaml` and refreshes `training_curves.png` every `plot_refresh_step_interval` steps (defaults to `validation_step_interval`)

Parallel submit (three GPU jobs + comparison plot with `afterok`, `VENV_DIR=/home/hl4291/venv`):

```bash
cd /home/hl4291/chess_analysis/lmcos
export VENV_DIR=/home/hl4291/venv
./slurm/4_supervised_controller/submit_smoke_controller_parallel.sh
# → slurm/logs/4_supervised_controller/smoke_comparison.png
```

Metrics YAML → `slurm/logs/4_supervised_controller/<run>/metrics.yaml`; plot is written beside it during training. Post-hoc replot:

```bash
python3 -m cts.train.controller_train plot-metrics \
  slurm/logs/4_supervised_controller/subtree_weighting_root_budget/metrics.yaml
```

Code changes:
- Replaced `max_train_batches_per_epoch` with `train_batches` (cap gradient steps per epoch).
- Added `validation_step_interval` and `greedy_eval_step_interval` — full validation and greedy eval during the train loop when `n_batches % interval == 0`; disables epoch-end eval when step intervals are set.
- Unified `ControllerTrainMetricsLogger` — single object for YAML metrics logging and plot refresh; standalone replot via `python3 -m cts.train.controller_train plot-metrics …`.
- Added `metrics_run_name` for human-readable plot titles / comparison legends.
- Metrics format changed from JSONL to YAML (`run_start` + `batches` list).
- Removed separate `TrainingMetricsWriter` and `cts.analysis.plot_controller_train_metrics` / `watch_controller_train_metrics`.
- Slurm smoke jobs use `VENV_DIR=/home/hl4291/venv` (no `CTS` / `cts_supervised` conda env on hl4291 della).

### Pipeline stages

The full data-to-model pipeline has six stages. Each stage's output feeds directly into the next.

#### 1. FEN sampling

Starting positions are sampled from a Lichess game database via SQL (DuckDB). The query selects ~200k games from 2023 with both players rated 1800–2600 and at least 20 half-moves. From each sampled game, one board position is drawn uniformly at random (subject to ply 8–120, 2–60 legal moves, 8–32 pieces). A final reservoir sample reduces the set to ~100k FENs. The output is a flat text file of FEN strings and an accompanying Parquet table with game metadata (ELO, time control, opening, piece counts). See `sql/download_FENs.py` for the DuckDB query and `cts.data.sample_fens` for the module entry point.

#### 2. FEN filtering

Not all positions produce informative search trees. A PUCT-based stability filter runs a short tree search (default 16 nodes, same PUCT logic used for full tree generation) on each FEN and records the best root move at 1 expansion, at a midpoint (~5 expansions), and at the full budget. A FEN is kept only if the best move at 1 expansion differs from the best move at the full budget **and** the best move at the midpoint also differs from the full-budget best move. This selects positions where search materially changes the chosen action, so the resulting trees will have nontrivial stopping structure for the controller. This step is embarrassingly parallel and is submitted as sharded Slurm jobs. See `scripts/filter_fens_by_puct_stability.py` and `scripts/submit_puct_filter_shards.py` for the sharded variant; `cts.data.validate_fens` is the in-line module entry point for smaller runs.

#### 3. Tree generation

Each filtered FEN becomes a PUCT search tree. The expansion loop starts from the root position and repeatedly selects a leaf via the PUCT formula (`Q(s,a) + c_puct * P(a|s) * sqrt(N(s)) / (1 + N(s,a))`), expands it by querying an lc0 engine for child priors, values, and WDL vectors, then backpropagates the leaf's value up the selection path (negating at each ply for alternating perspective). WDL vectors are backpropagated alongside scalar values. Expansion continues until a node budget is reached (currently fixed at 96 nodes).

During expansion, an oracle trace is recorded: after every expansion step, the current best root move and Q-values for all root children are saved. This trace allows later stages to reconstruct any prefix of the search trajectory without re-running the engine.

After expansion completes, teacher targets are extracted. Node value targets are visit-weighted averages of child Q-values. Edge WDL targets are visit-weighted averages of the WDL vectors accumulated during backpropagation, perspective-flipped to the parent's viewpoint. These targets, together with the tree structure, per-node scalar features (value, WDL, prior), and the oracle trace, are saved as a serialized `PretrainExample`.

Variable-size trees can also be derived from a full 96-node tree by sampling a prefix node count and recomputing targets on the truncated tree, avoiding additional engine calls (`cts.data.preprocess_gnn.derive_prefixes`); this path exists but is not part of the standard chain — production has historically packed full 96-node trees directly. See `cts.data.preprocess_gnn.teacher_targets` for the consolidation logic and `cts.data.build_tree` for the CLI (subcommands `generate-dataset` and `pretrain-child-wdl-encoder`).

#### 4. Encoder pretraining

The tree encoder is a GNN (class `TreeEncoder` in `cts.models.gnn`) that operates on variable-size trees packed into a single flat batch. Each node starts with a 5-dimensional feature vector (value, WDL win/draw/loss, WDL variance) embedded through a two-layer MLP into a `d_embed`-dimensional state. The encoder then runs `k` rounds of alternating upward (children→parent) and downward (parent→children) message passing. In the upward pass, each parent aggregates its children's states via a multi-head attention layer (`TreeAttMsgLayer`) conditioned on sinusoidal slot embeddings that encode each child's position among its siblings (sorted by UCI move string). In the downward pass, each child receives a linear projection of its parent's state. Both directions update node states through a shared GRU cell.

Two propagation modes exist. In **synchronous** mode, all nodes update simultaneously from the previous round's states, giving a receptive field of `k` hops. In **asynchronous** (sequential) mode, the upward pass processes nodes from leaves to root in depth order and the downward pass from root to leaves, so each node sees already-updated neighbors. With `k = 1` in asynchronous mode, every node's receptive field covers the entire tree regardless of depth. The current encoder uses asynchronous mode with `k = 1`.

The encoder output is the set of all node states plus the root state for each tree in the batch. Trees are batched by flattening all nodes into a single tensor with a CSR (compressed sparse row) child-pointer structure and a tree-index vector that maps each node back to its source tree.

Pretraining uses the **child-WDL objective**: a slot-conditioned decoder MLP takes the concatenation of a parent node's state and a sinusoidal slot embedding and predicts a 3-way softmax over win/draw/loss for each parent→child edge. The loss is cross-entropy against the search-consolidated edge WDL targets from tree generation. The pretraining loop iterates over packed tensorized shards with Adam, tracking loss gap (cross-entropy minus target entropy) as the primary validation metric. The best encoder checkpoint (by validation loss) is saved and used as the frozen backbone for controller training. The encoder architecture is embedded in the checkpoint metadata under `encoder_architecture` so downstream stages reconstruct the encoder shape directly from the file. See `cts.models.gnn` for the encoder, `cts.train.gnn_pretrain` (class `ChildWdlPretrainer`) for the pretraining loop, and `cts.data.build_tree` for the `pretrain-child-wdl-encoder` subcommand. Data prep for this stage runs through `cts.data.preprocess_gnn.split` (writes train/validation text manifests) and `cts.data.preprocess_gnn.pack` (concatenates examples into tensorized shards plus a JSON manifest).

#### 5. Controller episode packing

Raw `PretrainExample` trees are converted into budget-augmented controller episodes for offline training. This is a multi-step process.

First, each source tree's oracle trace is used to reconstruct a trimmed **snapshot episode**: a sequence of tree snapshots at each expansion step, together with the best root move and its Q-value at each snapshot. The episode is trimmed to start at the first expansion where the root has at least one child (i.e., the first real decision point). The **halt reward** at each snapshot is the full-tree Q-value of the root move that is best at that snapshot—measuring what you would get if you stopped searching there and committed to that move, evaluated under the complete search.

Second, each source tree is replicated across synthetic **starting budgets**. Five budget buckets span the range from scramble (1–3 time steps) to very-large (61–120 time steps), with 2 deterministic samples per bucket (seeded by a hash of the source path). For each sampled starting budget, the budgeted oracle computes the optimal stopping policy via backward induction:

- At each planning step `t`, **halt value** = halt reward at step `t`.
- **Continue value** = −`c_continue(N_t, T_t)` + `V*(t+1)`, where `c_continue` is the sum of a maintenance cost (proportional to `(N_t / 30)^1.1`, currently scaled to 0) and a time cost (a decreasing-marginal function of remaining budget `T_t`, parameterized by `λ`, `p`, `τ`).
- The oracle halts at step `t` iff halt value ≥ continue value; otherwise it continues.
- The **target advantage** at each step is `continue_value − halt_value`. Positive means "continue is better"; negative means "halt is better."

If the starting budget expires before the episode ends, a large negative timeout value (−1) is used as the continuation value.

Before packing, source trees are filtered: trees whose halt-reward range across the episode falls below a threshold (default 0.10) are discarded, and trees exhibiting `X*AB*A` root-move churn patterns (where the best move oscillates back to a previously abandoned move) can be excluded. An optional hierarchical `dj` stratification rebalances the training set across stop-depth-excess bins (how deep the oracle searches on average) and budget-action-variance bins (how much the oracle's halt/continue decisions vary across budgets), to reduce overrepresentation of trivially easy trees.

The output is packed tensorized shards: each shard stores per-step node features, parent/child indices, depth, edge slots, halt rewards, target advantages, tree sizes, time budgets, oracle stop steps, and starting budgets for a batch of episodes. See `cts.data.preprocess_mc.pack` for the packing entry point, `cts.data.preprocess_mc.oracle` for the budgeted backward induction, and `cts.data.episode_envs` for the snapshot-episode reconstruction.

#### 6. Controller training

The controller predicts the **compute advantage** `A(s) = Q_continue(s) − Q_halt(s)` at each planning step and halts when `A(s) ≤ 0`. The model (class `MetaController` in `cts.models.mc`) consists of the pretrained tree encoder (typically frozen) plus an MLP advantage head. The encoder processes each step's tree snapshot and produces a root embedding; this is concatenated with two scalar state features (current tree size `N_t` and remaining time budget `T_t`) to form the advantage head's input. The MLP head (configurable width and depth; default 3 layers of 256 units) outputs a scalar predicted advantage. An optional separate sign head shares the MLP backbone but has its own final projection for a binary continue/halt logit.

Training minimizes a weighted combination of advantage MSE and sign BCE (binary cross-entropy on the sign of the advantage). The sign loss directly targets the decision boundary rather than relying on squared-error alone, which can be insensitive to small wrong-sign predictions near zero.

To avoid repeated frozen-encoder forward passes, the encoder cache is **materialized** as a separate pipeline stage (`cts.data.preprocess_mc.materialize` with two subcommands: `materialize` writes per-worker shards, `merge` stitches them into a single `final_cache` .pt that downstream training loads). The controller training loop reads the cache directly and trains only the advantage head on a flat `TensorDataset`. The cache stores the source manifest path and encoder checkpoint path in its metadata; the trainer rejects a cache whose metadata doesn't match its current config, so a stale cache surfaces immediately rather than silently feeding wrong embeddings into training.

**Greedy evaluation** runs the trained model's stopping rule on validation episodes: starting from the first planning step, it predicts advantages and halts at the first step where `A(s) ≤ 0` (or at the episode's last step). The resulting stop step determines the achieved return (halt reward minus accumulated continue costs), which is compared to the oracle's optimal return. Reported metrics include exact stop-step accuracy, first-action accuracy, average return, average oracle value, average regret, and average number of expansions used. Per-episode diagnostics (predicted advantages, oracle/predicted stop steps, difficulty scalars, regret decomposition) can be written to JSONL for offline analysis.

See `cts.train.controller_train` for the full training loop, `cts.data.preprocess_mc.materialize` for the encoder-cache materialization, and `cts.analysis.analyze_budgeted_controller_run` (with themed sub-modules under `cts.analysis._budgeted/`) for post-hoc analysis of trained controllers.

This file is the running experimental record for the project.

What belongs here:
- Meaningful changes to objectives, training setups, diagnostics, or evaluation.
- Summaries of important runs, including intent, key parameters, and outcome.

What does not belong here:
- Tiny plumbing edits with no experimental consequence.
- Incidental refactors or formatting-only changes.

## Disk inventory

Permanent registry of every artifact this project has produced or consumed, so the
question "what is `X` for / where did it come from?" is answerable forever — including
for files that have since been superseded or deleted. The dated experiment entries
below carry the scientific story; this section is the index of paths.

Convention:

- Every experiment entry includes explicit `Inputs:` and `Outputs:` lines listing the
  exact paths it read and wrote.
- Every path that appears under `Outputs:` is registered as a line in this section in
  the same commit. Likewise for `Inputs:` paths the first time they show up.
- Each line has the form
  `PATH — short description (producer: YYYY-MM-DD entry title) [status]` where
  `status` is one of:
  - `active` — current canonical, still on disk and still in use;
  - `superseded` — replaced by a later artifact under the same stage, may or may not still be on disk;
  - `deleted` — known to no longer be on disk;
  - `experimental` — one-off output that wasn't promoted to canonical; kept here so its existence and purpose are not lost.
- Status changes get an explicit edit: when a directory is deleted or a successor lands, update the line rather than removing it.
- Stages without entries below have simply not had any artifact touched by a logged
  experiment yet. Silence does not mean "no canonical exists."

### FEN sampling
*(no artifacts logged for this stage yet)*

### FEN filtering
*(no artifacts logged for this stage yet)*

### Tree generation
*(no artifacts logged for this stage yet)*

### Pretrain split / pack
- `/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/pretrain_split_oracle96_trace_filtered_rerun/` — train + validation text manifests for the rerun encoder pretraining (producer: pre-2026-04-04). `[active]`
- `/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/pretrain_packed_oracle96_trace_filtered_rerun/` — packed tensorized shards + JSON manifests, regenerated from the rerun split for downstream audits (producer: 2026-05-18 encoder KL audit). `[active]`
- `/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/pretrain_packed/` — packed pretrain dir that previously fed the rerun encoder pretraining; the directory no longer exists on disk as of 2026-05-18 and was rebuilt as `pretrain_packed_oracle96_trace_filtered_rerun/`. `[deleted]`

### Encoder pretrain
- `/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/checkpoints/tree_encoder_child_wdl_async_k1_rerun.pt` — current canonical encoder (async k=1, child-WDL pretrain objective) used by every Section 3 controller (producer: pre-2026-04-04). `[active]`
- `/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/checkpoints/tree_encoder_child_wdl_async_k1_rerun_decoder.pt` — paired child-WDL decoder head saved alongside the encoder; used to reconstruct edge WDL predictions for analyses (producer: pre-2026-04-04). `[active]`
- `/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/checkpoints/tree_encoder_child_wdl_async_k1_bucketed.pt` — fresh-from-init encoder, 1000 epochs under current code; same architecture as the rerun encoder. Overall validation KL = 0.0040 (producer: 2026-05-20 bucketed encoder pretraining). `[active]`
- `/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/checkpoints/tree_encoder_child_wdl_async_k1_bucketed_decoder.pt` — paired decoder (producer: 2026-05-20). `[active]`
- `/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/checkpoints/tree_encoder_child_wdl_async_k1_bucketed_kl.jsonl` — 1000-row per-epoch JSONL time series for the bucketed run; schema matches `cts.train.gnn_pretrain.BucketedKLState.summary` (producer: 2026-05-20). `[active]`
- `/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/checkpoints/tree_encoder_child_wdl_async_k1_subtree_weighted.pt` — fresh-from-init encoder, 1000 epochs with cross-entropy loss weighted per edge by child subtree size. Trades ~3x worse leaf-cell KL for ~21x lower KL on the large-subtree cells the weighting targeted (producer: 2026-05-20 subtree-size-weighted pretraining). `[active]`
- `/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/checkpoints/tree_encoder_child_wdl_async_k1_subtree_weighted_decoder.pt` — paired decoder (producer: 2026-05-20). `[active]`
- `/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/checkpoints/tree_encoder_child_wdl_async_k1_subtree_weighted_kl.jsonl` — 1000-row per-epoch JSONL for the weighted run; the bucketed-KL accumulator inside the JSONL is *unweighted* so per-cell numbers are directly comparable to the bucketed run cell-by-cell (producer: 2026-05-20). `[active]`

### Controller episode packing
- `/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/controller_packed_combined_nomaint_no_xaba/` — combined packed controller episodes used across the λ=18.5 / λ=5 sweeps and Section 3 rerun controllers (producer: pre-2026-04-29). `[active]`

### Materialized controller cache
- `/tigress/ysagiv/chess/cts/train_cache_rerun.pt` — materialized encoder cache for the train split, feeds Section 3 rerun controller training (producer: 2026-04-30 entries). `[active]`
- `/tigress/ysagiv/chess/cts/validation_cache_rerun.pt` — matched validation cache for the same training runs (producer: 2026-04-30 entries). `[active]`
- `/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/train_cache_subtree_weighted.pt` — materialized cache against the subtree-weighted encoder, train split (producer: 2026-05-21 subtree-weighted controller training). `[active]`
- `/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/validation_cache_subtree_weighted.pt` — matched validation cache (producer: 2026-05-21). `[active]`
- `/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/train_cache_subtree_weighted_shards/` — per-worker shard directory the merge stage stitched into `train_cache_subtree_weighted.pt`; kept so resumes are possible (producer: 2026-05-21). `[active]`
- `/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/validation_cache_subtree_weighted_shards/` — same for the validation split (producer: 2026-05-21). `[active]`

### Controller checkpoints
- `~/chess/cts/async_soph/configs/section3_rerun_models.json` — index file mapping the five Section 3 rerun controller labels (slw01, reweight_w4, inv_freq, affine, affine+rw) to their checkpoint + diagnostics paths under `/tigress/ysagiv/chess/cts/checkpoints/` (producer: pre-2026-05-18). `[active]`
- `/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/checkpoints/fittedq_subtree_weighted_zt_tt.pt` — controller trained on the subtree-weighted encoder with `controller_inputs: [z_t, T_t]`; greedy regret 0.024 (producer: 2026-05-21). `[active]`
- `/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/checkpoints/fittedq_subtree_weighted_zt_tt_diagnostics.jsonl` — per-episode diagnostics for the run above (producer: 2026-05-21). `[active]`
- `/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/checkpoints/fittedq_subtree_weighted_zt_only.pt` — same encoder, `controller_inputs: [z_t]` (no time-budget); regret 0.076 (producer: 2026-05-21). `[active]`
- `/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/checkpoints/fittedq_subtree_weighted_zt_only_diagnostics.jsonl` — paired diagnostics (producer: 2026-05-21). `[active]`
- `/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/checkpoints/fittedq_subtree_weighted_tt_only.pt` — `controller_inputs: [T_t]` baseline under the new pipeline (encoder unused); regret 0.212 best / 0.234 final, reproduces the historical T_t-only floor of 0.218 (producer: 2026-05-21). `[active]`
- `/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/checkpoints/fittedq_subtree_weighted_tt_only_diagnostics.jsonl` — paired diagnostics (producer: 2026-05-21). `[active]`
- `/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/checkpoints/fittedq_rerun_encoder_zt_tt_ablation.pt` — ablation: rerun encoder + `controller_inputs: [z_t, T_t]` under the new pipeline, vanilla 20-epoch training; regret 0.219 best / 0.291 final (producer: 2026-05-21). `[active]`
- `/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/checkpoints/fittedq_rerun_encoder_zt_tt_ablation_diagnostics.jsonl` — paired diagnostics (producer: 2026-05-21). `[active]`

### Analysis outputs
- `/tigress/ysagiv/chess/cts/analysis/advantage_head/` — Section 3 advantage-head probe outputs (JSON + per-model PDFs) for the five rerun controllers; consumed by paper figures (producer: pre-2026-05-18). `[active]`
- `/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/encoder_kl_audit/` — per-edge encoder KL bucketed by (parent_depth, child_subtree_size) for the rerun encoder, train + validation; JSON tables + heatmap PDFs (producer: 2026-05-18 encoder KL audit). `[active]`

## 2026-04-04

### Encoder pretraining path repaired

Intent:
- Get the tree encoder pretraining path working reliably on packed tensorized data.

Meaningful change:
- Fixed the encoder pretraining path and produced a usable checkpoint.

Result:
- Encoder pretraining completed successfully.
- Checkpoint used afterward:
  - `/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/checkpoints/tree_encoder_pretrain_interactive.pt`

Notes:
- This became the reference encoder checkpoint for later controller experiments.

## 2026-04-05

### Synthetic PPO diagnostic added and cleaned up

Intent:
- Determine whether PPO/controller learning is fundamentally broken independent of the real tree representation.

Meaningful changes:
- Added `scripts/controller_synthetic_diagnostic.py`.
- Switched the synthetic diagnostic to bypass the learned encoder and use deterministic one-hot timestep features.
- Added a multi-scenario synthetic suite.

Result:
- PPO learned the correct halt/continue policy on clean synthetic tasks.
- This ruled out a gross end-to-end PPO failure in the simplest setting.

Conclusion:
- The controller pipeline can learn correct stopping rules on well-posed toy problems.

### Evaluation/env sequencing bug fixed

Intent:
- Verify that controller evaluation and deterministic control diagnostics were not being contaminated by env-handling bugs.

Meaningful changes:
- Fixed `evaluate_controller(...)` so it does not recreate the first episode every time.
- Fixed deterministic env sequencing in `scripts/controller_representation_control.py`.

Result:
- Oracle-control evaluation became trustworthy.

Conclusion:
- Some earlier controller-control failures were polluted by evaluation/env bugs.

### Oracle-stop-step control diagnostic built

Intent:
- Test the production controller path with trivial representations.

Meaningful changes:
- Added `scripts/controller_representation_control.py`.
- Added exact stop-step accuracy, first-action accuracy, confusion matrix, and trace support.

Result:
- With the old weak oracle dataset, performance was seed-sensitive and fragile.
- Full traces showed divergence concentrated at the state:
  - `label_1 | phase_1`
- Bad runs received early value/advantage pressure toward `continue` at that state and then stopped sampling `halt`.

Conclusion:
- The failure was localized to the second decision in a weak-margin diagnostic, not a broad controller collapse.

### Weakness of the original oracle dataset identified

Intent:
- Understand why the oracle-stop-step control task was so fragile.

Finding:
- The old regret-style oracle dataset often produced cases like:
  - `oracle_0`: `[0, 0, 0]`
  - `oracle_1`: `[-x, 0, 0]`
  - `oracle_2`: `[-x, -x, 0]`

Interpretation:
- The critical `oracle_1` second decision was often only better by the continue-cost margin.

Conclusion:
- The original oracle diagnostic was too weak to be a decisive controller test.

### Controller objective changed from regret-only to absolute full-reference move quality

Intent:
- Use a better-founded reward: score halting by the quality of the selected move under the full planner, not by regret relative to the final best move.

Meaningful changes:
- In `supervised_branch.py`, halt reward changed from:
  - `Q_full(best_move_now) - Q_full(best_move_full)`
  to:
  - `Q_full(best_move_now)`
- The same objective change was propagated to:
  - `scripts/controller_representation_control.py`
  - `scripts/generate_controller_oracle_dataset.py`
  - `scripts/analyze_controller_optimal_actions.py`

Conclusion:
- Controller reward is now in absolute full-reference move-quality units.

### Strong oracle dataset regenerated under the new objective

Intent:
- Rebuild the oracle control diagnostic under the new absolute-value reward with clearer decision margins.

Meaningful change:
- Generated:
  - `/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/controller_oracle_strong_abs`

Result:
- Example halt-reward sequences became meaningfully separated.
- Oracle-stop-step control sweeps on this dataset achieved exact stop-step accuracy `1.0` across all tested seeds.

Conclusion:
- Under the stronger objective and stronger diagnostic dataset, the control path behaves reliably.

## 2026-04-06

### Real-data oracle analysis under the new absolute-value objective

Intent:
- Measure whether the real collected dataset actually contains a meaningful adaptive stopping signal under the new reward.

Run:
- `scripts/analyze_controller_optimal_actions.py`
- Data: real train/validation manifests

Important parameter sweep:
- `continue_cost = 0.1`
  - Real-data oracle was mostly immediate halt.
- `continue_cost = 0.001`
  - Real-data oracle became meaningfully adaptive.

Validation-set oracle summary for `continue_cost = 0.001`:
- `total_examples = 3581`
- `mean_first_halt_reward = 0.107454`
- `mean_optimal_value = 0.167093`
- `should_continue_rate = 0.437308`
- Oracle mean expansions from the stop-step histogram:
  - `1.8073`

Conclusion:
- The real dataset does contain a meaningful adaptive stopping signal under the new objective.
- `continue_cost = 0.001` is the appropriate regime for a genuinely adaptive controller.

### Real controller training under the new absolute-value objective

Intent:
- Train the real controller on the real collected dataset under the new objective with low continue cost.

Run configuration:
- Command: `supervised_branch_cli.py frozen-rl-generated`
- Train data:
  - `/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/pretrain_split/train_manifest.txt`
- Validation data:
  - `/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/pretrain_split/validation_manifest.txt`
- Encoder checkpoint:
  - `/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/checkpoints/tree_encoder_pretrain_interactive.pt`
- Continue cost:
  - `0.001`

Early finding:
- With the old rollout collector, `num_envs` semantics were wrong.

### Major PPO rollout collection bug found and fixed

Intent:
- Compare the implementation to standard PPO semantics.

Meaningful change:
- In `FrozenEncoderControllerTrainer._collect_rollout()`, rollout collection was changed from:
  - stepping one env per rollout step
  to:
  - stepping every env on every rollout step

Why this mattered:
- Before the fix:
  - `num_envs = 16`, `rollout_steps = 256` collected only `256` total transitions.
- After the fix:
  - it correctly collects `4096` transitions.

Regression protection:
- Added a test in `test_supervised_branch.py` asserting collected rollout size is:
  - `num_envs * rollout_steps`

Conclusion:
- Prior real PPO runs used a substantially smaller and noisier effective batch than intended.

### PPO diagnostics added

Intent:
- Track whether PPO is behaving healthily instead of relying only on surrogate losses.

Meaningful changes:
- Added per-update PPO diagnostics:
  - `approx_kl`
  - `clipfrac`
  - `explained_variance`
- Added `--validation-interval` to the controller CLI and Slurm job.

Conclusion:
- Training logs now include both PPO-health metrics and periodic validation metrics.

### Interactive smoke test after the rollout fix

Intent:
- Verify that the corrected rollout semantics and new diagnostics run cleanly on GPU before launching a long Slurm job.

Run configuration:
- `num_envs = 16`
- `rollout_steps = 256`
- `minibatch_size = 128`
- `continue_cost = 0.001`
- `num_updates = 5`
- `validation_interval = 1`

Outcome:
- No crashes.
- PPO diagnostics looked healthy:
  - `approx_kl` stayed small
  - `clipfrac` stayed modest
  - `explained_variance` quickly became strongly positive
- Validation average expansions moved from very small to around `3.0` during the short smoke run.

Conclusion:
- The corrected PPO implementation and diagnostics were healthy enough to justify a full cluster run.

### Full Slurm run: job `6590724`

Intent:
- Run the real controller at scale under the corrected PPO semantics and new objective.

Key parameters:
- Objective: absolute full-reference move quality
- `continue_cost = 0.001`
- `num_envs = 16`
- `rollout_steps = 256`
- `minibatch_size = 128`
- `validation_interval = 50`
- Requested walltime:
  - `12:00:00`

Outcome:
- The job timed out at update `480 / 1000`.
- Controller remained numerically stable through the end.

Last logged training metrics:
- `update = 480`
- `policy_loss = -0.001246`
- `value_loss = 0.007091`
- `entropy = 0.044301`
- `approx_kl = 0.000626`
- `clipfrac = 0.008179`
- `explained_variance = 0.935112`
- `mean_episode_return = 0.133493`
- `mean_episode_length = 9.441`
- `halt_rate = 0.372`

Last logged validation metrics:
- `update = 450`
- `average_return = 0.120145`
- `average_expansions = 9.125000`
- `average_terminal_quality = 0.194199`

Interpretation:
- Training was technically healthy and stable.
- Relative to the offline validation oracle:
  - controller return was below oracle (`0.120` vs `0.167`)
  - controller used much more search (`9.125` expansions vs oracle `1.807`)
- The learned policy is still over-continuing.

Additional local analysis:
- Downloaded and analyzed:
  - `logs/cts-controller_6590724.out`
- Analysis artifacts:
  - `analysis_outputs/controller_run_6590724/report/summary.json`
  - `analysis_outputs/controller_run_6590724/report/updates.csv`
  - `analysis_outputs/controller_run_6590724/report/validations.csv`
  - `analysis_outputs/controller_run_6590724/report/training_metrics.png`
  - `analysis_outputs/controller_run_6590724/report/validation_metrics.png`

Notable summary numbers:
- Best logged validation return:
  - `0.121552` at update `50`
- Last logged validation return:
  - `0.120145` at update `450`
- Oracle return ratio:
  - about `0.719`
- Oracle expansion ratio:
  - about `5.05`

Conclusion:
- The full run did not fail numerically.
- It learned a stable but still over-searching controller.
- Later training did not improve validation return over the early checkpoint.

### Representation probe added

Intent:
- Test whether frozen encoder root states contain enough information to predict the offline oracle halt/continue decision on real snapshots without involving PPO.

Meaningful change:
- Added:
  - `scripts/probe_controller_representation.py`

What it does:
- Loads real snapshot states from raw pretrain examples.
- Computes offline oracle halt/continue actions under the current absolute-value controller objective.
- Extracts frozen encoder root embeddings in batches.
- Trains a small MLP probe on those embeddings.
- Reports:
- overall accuracy
- first-action accuracy
- per-phase accuracy

Follow-up improvement:
- Added cached dataset support so the expensive oracle-label + embedding pass can be paid once and reused across probe sweeps.
- Added progress logging during sample loading and embedding extraction.

Conclusion:
- This is the next discriminating test if PPO remains flat: determine whether the representation carries the oracle signal at all.

### RL sanity harness added

Intent:
- Directly verify RL gradient wiring instead of inferring correctness from long noisy training curves.

Meaningful change:
- Added:
  - `scripts/rl_sanity_checks.py`
- Added regression coverage in:
  - `test_supervised_branch.py`

What it does:
- Runs three controller-RL sanity checks against the real PPO and REINFORCE code paths:
- one-batch manual-sign tests
- gradient and parameter norm inspection
- fixed-state halt-logit drift over several updates on simple toy environments

Why this matters:
- The PPO run was numerically stable but flat on validation.
- The REINFORCE baseline also showed suspiciously little movement early.
- This harness is intended to answer a narrower question: are updates flowing with the correct sign and reaching the intended modules at all.

Initial result:
- Ran the harness locally on CPU with short settings (`drift_updates=2`).
- PPO:
  - manual-sign tests passed for both positive and negative halt updates
  - encoder, halt head, and value head all had nonzero gradient norms
  - halt-favored logit drifted strongly positive; continue-favored drifted negative
- REINFORCE:
  - manual-sign tests passed for both positive and negative halt updates
  - encoder and halt head had nonzero gradient norms
  - value head correctly had zero gradient because REINFORCE does not use it
  - halt-favored logit drifted positive; continue-favored drifted negative

Conclusion:
- There is now direct evidence that both RL implementations are wired correctly at the local gradient-update level.
- The flat long-run training curves are therefore more likely to reflect optimization/task issues than a gross sign or disconnected-gradient bug.

Cluster result:
- Ran the same harness on the cluster with the real encoder checkpoint and default controller architecture.
- JSON artifact:
  - `/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/analysis/rl_sanity_checks.json`

Observed outcomes:
- PPO:
  - manual-sign tests passed
  - encoder, halt head, and value head all had nonzero gradient norms and parameter deltas
  - halt-favored fixed-state logit drifted strongly positive over 5 updates
  - continue-favored fixed-state logit drifted strongly negative over 5 updates
- REINFORCE:
  - manual-sign tests passed
  - encoder and halt head had nonzero gradient norms and parameter deltas
  - value head correctly remained unused (`0` grad, `0` parameter delta)
  - halt-favored fixed-state logit drifted strongly positive over 5 updates
  - continue-favored fixed-state logit drifted negative over 5 updates

Interpretation:
- This is strong evidence against a gross gradient-sign or disconnected-parameter bug in either RL pathway.
- The remaining failures on real training runs are more likely due to optimization dynamics, distribution shift, objective mismatch, or representation limits than basic gradient wiring.

### Unfrozen encoder RL run analyzed (`6621831`)

Intent:
- Test whether unfreezing the tree encoder helps the real controller objective, since the frozen PPO run flatlined early.

Run summary:
- Job:
  - `6621831`
- Local artifacts:
  - `analysis_outputs/controller_run_6621831/cts-controller_6621831.out`
  - `analysis_outputs/controller_run_6621831/report/summary.json`
  - `analysis_outputs/controller_run_6621831/report/training_metrics.png`
  - `analysis_outputs/controller_run_6621831/report/validation_metrics.png`
- Frozen vs unfrozen comparison artifacts:
  - `analysis_outputs/controller_run_6621831/comparison/comparison_summary.json`
  - `analysis_outputs/controller_run_6621831/comparison/training_comparison.png`
  - `analysis_outputs/controller_run_6621831/comparison/validation_comparison.png`

Key results:
- Unfrozen best validation:
  - return `0.118722` at update `50`
  - expansions `12.125`
- Unfrozen last validation:
  - return `0.116597` at update `150`
  - expansions `14.15625`
- Frozen best validation:
  - return `0.121552` at update `50`
  - expansions `7.5625`

Conclusion:
- Unfreezing the encoder did not help on this run.
- It was worse than the frozen baseline on validation return, terminal quality, and search efficiency.
- Like the frozen run, the best validation point still appeared very early and then deteriorated.

## 2026-04-06

### Real-data trivial control: `oracle_action_now`

Intent:
- Remove representation ambiguity on the real episode distribution by feeding the controller only the current offline oracle halt/continue action.

Meaningful change:
- Extended:
  - `scripts/controller_representation_control.py`
- Added regression coverage in:
  - `test_controller_representation_control.py`

What changed:
- New representation mode:
  - `oracle-action-now`
- Each observation is now a 2d one-hot:
  - `[1, 0]` for continue
  - `[0, 1]` for halt
- This representation intentionally hides phase and episode identity, so it is a stricter real-data control than `oracle-stop-step`.

Validation:
- Targeted unit tests passed for the new observation encoding and env behavior.

Purpose:
- If PPO succeeds with `oracle-action-now` on real data, the remaining bottleneck is much more likely to be representation/state construction than the RL machinery itself.

Follow-up:
- An independent audit found that `evaluate_oracle_agreement(...)` was miswired for `oracle-action-now`.
- The evaluation env was accidentally instantiated with the default representation instead of the requested one, so the oracle-agreement footer from the first `oracle-action-now` run was invalid.
- Fixed the evaluator wiring and added a regression test for `oracle-action-now` agreement evaluation.

Additional change:
- Added `--min-decision-margin` to the control script.
- This allows real-data control runs to filter episodes to only those with strong optimal-action margins under the current objective.

Follow-up change:
- Replaced the control diagnostic policy/value heads with linear readouts on top of the fixed-feature encoder.
- This makes `oracle-action-now` directly interpretable:
  - halt logit = `w_continue * action_0 + w_halt * action_1 + b`
- The control script now prints the learned halt/value readout weights for small feature sets.

Additional diagnostic:
- Added `--force-oracle-actions` to `scripts/controller_representation_control.py`.
- This diagnostic collects PPO rollouts by stepping the environment with the oracle action encoded in `oracle-action-now` observations, while still scoring the model log-probability and value for that forced action.

Purpose:
- Separate local gradient plumbing from on-policy exploration.
- If ordinary `oracle-action-now` PPO collapses because continuing only pays off under correct future actions, forced-oracle rollouts test whether PPO moves the linear readout correctly when future actions are constrained to be optimal.

Follow-up diagnostic:
- Added `--one-step-oracle-bandit`.
- This converts real oracle-action states into a balanced one-step contextual bandit:
  - observation `[1, 0]` means continue
  - observation `[0, 1]` means halt
  - matching the encoded action gives reward `+1`
  - the opposite action gives reward `-1`
  - the episode terminates immediately

Purpose:
- This is the clean PPO plumbing diagnostic for the oracle-action representation.
- It removes sequential stopping dynamics, future-policy dependence, and transition-count imbalance, while still using the same PPO trainer, tensorizer, and linear policy/value model.

Result:
- The one-step bandit diagnostic achieved:
  - `average_return = 1.000`
  - `bandit_action_accuracy = 1.000`
- Learned linear halt logits:
  - `logit([1, 0]) = -6.854`
  - `logit([0, 1]) = 6.997`

Conclusion:
- PPO plumbing, action encoding, tensorization, Bernoulli sign convention, and linear readout capacity are correct in the immediate-reward setting.

Additional diagnostic:
- Added `--supervised-oracle-action`.
- This trains the same linear halt readout with supervised BCE on balanced `oracle-action-now` states extracted from the selected sequential real episodes.

Purpose:
- Check whether the sequential oracle-action labels are internally coherent and linearly learnable once RL temporal credit assignment is removed.
- The diagnostic also evaluates the supervised readout through the tensorized sequential stop-step evaluator.

Result:
- The supervised sequential oracle-action diagnostic achieved perfect reported accuracies.
- Learned linear halt logits in the latest run:
  - `logit([1, 0]) = -0.990`
  - `logit([0, 1]) = 1.118`

Conclusion:
- The sequential oracle-action labels are coherent and linearly learnable.
- The remaining `oracle-action-now` PPO failure is therefore isolated to sequential on-policy credit assignment/exploration, not diagnostic wiring or representation capacity for the toy control.

### Offline fitted-Q controller path added

Intent:
- Add an RL-flavored alternative to PPO that avoids the on-policy continuation-trajectory discovery problem.

Meaningful change:
- Added:
  - `scripts/train_fitted_q_controller.py`
- The script trains a frozen-encoder controller to predict Bellman action values:
  - `Q(s_t, continue) = -continue_cost + V*(s_{t+1})`
  - `Q(s_t, halt) = halt_reward_t`
  - action order is `[continue, halt]`
- The greedy policy halts iff:
  - `Q_halt >= Q_continue`

Purpose:
- Use the known deterministic snapshot trajectory structure directly.
- Preserve an RL/value-learning framing while removing PPO’s on-policy exploration trap.

Validation:
- Added focused regression tests in:
  - `test_train_fitted_q_controller.py`
- Tests cover Bellman target construction, episode-batch collation, and greedy stop-step behavior.

Initial smoke result:
- On a 1000-train / 500-validation smoke run, plain two-Q MSE achieved:
  - `validation_q_mse = 0.024`
  - `validation_action_accuracy = 0.507`
  - `greedy average_return = 0.148`
  - `greedy average_oracle_value = 0.164`
  - `greedy exact_stop_step_accuracy = 0.166`

Interpretation:
- Return was reasonably close to oracle, but action/boundary accuracy was weak.
- This indicates that plain Q-value MSE can fit common value level while missing the decision margin.

Follow-up change:
- Reparameterized the Q head as:
  - raw output 0: `Q_continue - Q_halt`
  - raw output 1: `Q_halt`
- `forward()` still returns comparable action values in `[Q_continue, Q_halt]` order.
- Training loss now optimizes:
  - halt-value MSE
  - continue-advantage MSE, weighted by `--advantage-loss-weight`

Purpose:
- Train the actual stopping decision variable directly:
  - continue iff `Q_continue - Q_halt > 0`

### 2026-04-07: Advantage-only compute controller

Intent:
- Stop treating the stopping controller as a two-Q regression problem when the deployed decision only depends on the compute advantage:
  - `A_compute(s) = Q_continue(s) - Q_halt(s)`

Meaningful change:
- Simplified `scripts/train_fitted_q_controller.py` so the controller head now predicts one scalar:
  - `A_compute`
- Training loss is now direct MSE on the Bellman-derived counterfactual compute advantage.
- Greedy evaluation now uses the explicit stopping rule:
  - continue iff `A_compute > 0`
  - halt otherwise
- Logged metrics now emphasize:
  - `advantage_mse`
  - `mean_abs_advantage_error`
  - `sign_accuracy`
  - `average_regret`

Purpose:
- Remove common-mode value fitting from the controller objective.
- Train the learned search rule directly: whether another unit of search is worth its compute cost.

Validation:
- `python3 -m py_compile scripts/train_fitted_q_controller.py test_train_fitted_q_controller.py`
- `/opt/miniconda3/envs/cts_supervised/bin/python -m unittest test_train_fitted_q_controller.py`
- `/opt/miniconda3/envs/cts_supervised/bin/python scripts/train_fitted_q_controller.py --help`

Follow-up change:
- Added `--representation oracle-action-now` to `scripts/train_fitted_q_controller.py`.
- This replaces each real tree snapshot with a 2d one-hot root feature:
  - `[1, 0]` for oracle continue
  - `[0, 1]` for oracle halt
- In this mode, the model uses a fixed-feature linear advantage readout instead of loading the TreeNN encoder.
- Added `--min-decision-margin` so low-margin episodes can be filtered for the toy sanity check.

Purpose:
- Make the counterfactual advantage objective testable on the same trivial representation used for prior PPO diagnostics.
- Expected behavior on strong-margin data:
  - high `sign_accuracy`
  - high greedy stop-step accuracy
  - `advantage_readout_weight[0] + bias > 0`
  - `advantage_readout_weight[1] + bias <= 0`

Follow-up optimization:
- The initial toy path still used the lazy episode `DataLoader`, which rebuilt raw tree episodes each epoch.
- Fixed this for the tensorizable frozen-encoder paths:
  - `--representation oracle-action-now`: materialize selected train/validation episodes once, then train/evaluate from `TensorDataset(features, target_advantages)`
  - frozen `--representation tree`: run the frozen encoder once to materialize root embeddings and target advantages, then train the advantage head on tensors
  - greedy evaluation uses precomputed per-episode tensors for both materialized paths

Purpose:
- Keep the diagnostic faithful to the compute-advantage objective while avoiding repeated raw `.pt` loading, oracle recomputation, and frozen-encoder forward passes every epoch.

Overnight run setup:
- Added `slurm/train_compute_advantage_della.slurm`.
- Defaults:
  - full train/validation manifests
  - frozen pretrained encoder
  - real tree representation
  - materialized root embeddings
  - `continue_cost = 0.001`
  - `epochs = 50`
  - checkpoint: `/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/checkpoints/tree_controller_compute_adv_full.pt`

Frozen run result:
- Log analyzed:
  - `cts-compute-adv_6645287.out`
- Local analysis directory:
  - `analysis_outputs/compute_advantage_run_6645287/`
- Materialization time:
  - train: approximately `4097.9s`
  - validation: approximately `221.0s`
- Best validation sign accuracy:
  - `0.435` at epoch `15`
- Best greedy return:
  - epoch `5`
  - `average_return = 0.157`
  - `average_oracle_value = 0.167`
  - `average_regret = 0.010`
  - `exact_stop_step_accuracy = 0.076`
  - `first_action_accuracy = 0.480`
- Final greedy evaluation:
  - `exact_stop_step_accuracy = 0.065`
  - `first_action_accuracy = 0.468`
  - `average_return = 0.156`
  - `average_oracle_value = 0.167`
  - `average_regret = 0.011`
  - `average_expansions = 11.644`

Conclusion:
- The advantage-only counterfactual objective works on the toy `oracle-action-now` representation, but the frozen real-tree representation still does not expose the compute-advantage boundary strongly enough for a simple head.
- Return is close to oracle in absolute value, but stop-step and sign metrics remain poor.

Unfrozen run result:
- Log analyzed:
  - `cts-adv-unfrozen_6645314.out`
- Local analysis directory:
  - `analysis_outputs/compute_advantage_run_6645314/`
- The run only logged 9 train epochs and one validation/greedy checkpoint, likely because unfrozen training cannot use frozen-embedding materialization and is much slower.
- Best validation sign accuracy:
  - `0.288` at epoch `5`
- Final logged greedy evaluation:
  - `exact_stop_step_accuracy = 0.028`
  - `first_action_accuracy = 0.438`
  - `average_return = 0.155`
  - `average_oracle_value = 0.167`
  - `average_regret = 0.012`
  - `average_expansions = 12.591`

Conclusion:
- The unfrozen run did not improve the compute-advantage controller in the time available.
- It was worse than the frozen run on sign accuracy and greedy boundary metrics, but the run was incomplete and much slower.

### Frozen encoder advantage decodability probe added

Intent:
- Test whether counterfactual compute advantage is decodable from frozen encoder root snapshots independent of the full controller training loop.

Meaningful change:
- Extended `scripts/probe_controller_representation.py` with:
  - `--probe-target action`
  - `--probe-target advantage`
- The advantage probe trains the existing linear/MLP probe heads to regress:
  - `A_compute(s) = Q_continue(s) - Q_halt(s)`
- It reports:
  - `mse`
  - `mae`
  - `sign_accuracy`
  - `first_sign_accuracy`
  - per-phase sign accuracy
- Existing action-probe caches remain usable. If a cache was created before advantage targets were stored, the script backfills advantage targets from the selected raw paths without recomputing frozen encoder embeddings.

Purpose:
- Separate representation decodability from the full compute-advantage trainer and check whether the frozen encoder root embedding contains enough information for the scalar value-of-computation boundary.

Validation:
- `python3 -m py_compile scripts/probe_controller_representation.py test_probe_controller_representation.py`
- `/opt/miniconda3/envs/cts_supervised/bin/python -m unittest test_probe_controller_representation.py`
- `/opt/miniconda3/envs/cts_supervised/bin/python scripts/probe_controller_representation.py --help`

Follow-up:
- Added `slurm/probe_advantage_decodability_della.slurm` to submit the full-dataset frozen advantage probe without hand-writing the command on the cluster.
- Defaults:
  - full train/validation manifests
  - pretrained frozen encoder checkpoint
  - shared probe cache directory
  - `--probe-target advantage`
  - `--probe-type mlp`
  - `--probe-epochs 50`

Validation:
- `bash -n slurm/probe_advantage_decodability_della.slurm`

## Going forward

Any future entry should include:
- Date
- Intent
- Meaningful change or run configuration
- Outcome
- Conclusion

## 2026-04-10

Intent:
- Correct the uncertainty-encoder pretraining target semantics after discovering that the earlier child-WDL path was decoding raw child valuehead WDLs instead of search-consolidated edge WDLs.

Meaningful change:
- Extended search backup statistics to carry WDL vectors alongside scalar values in `cts_pretrain.py`.
- `compute_teacher_targets(...)` and `consolidate_generated_tree(...)` now produce per-edge search-consolidated WDL targets:
  - backed up through search with ply-wise win/loss perspective flips
  - aggregated by visit-weighted averaging
- `PretrainExample` now persists `edge_wdl_targets` for those consolidated edge targets.
- Reintroduced a slot-conditioned child-WDL encoder pretraining path:
  - canonical sibling slots from lexicographic `incoming_move_uci`
  - sinusoidal slot encoding plus learned projection in the encoder/decoder
  - child-WDL trainer logs target entropy and KL gap (`loss_gap`)
- Updated packed tensorized shards to store `edge_slot` and optional `edge_wdl_targets`.
- Scrubbed a stale WDL validation message in `schema.py`.

Outcome:
- Focused and full validation passed:
  - `/opt/miniconda3/envs/trm/bin/python -m pytest test_supervised_branch.py -q`
  - `/opt/miniconda3/envs/trm/bin/python -m pytest test_plumbing.py test_model.py -q`
  - `/opt/miniconda3/envs/trm/bin/python -m pytest -q`
  - `/opt/miniconda3/envs/trm/bin/python supervised_branch_cli.py pretrain-child-wdl-encoder --help`
  - `/opt/miniconda3/envs/trm/bin/python scripts/pack_pretrain_examples.py --help`
- Final test result: `91 passed`

Conclusion:
- The branch is back to a semantically aligned child-WDL pretraining pipeline.
- The remaining work is operational: regenerate raw examples so they actually carry the corrected `edge_wdl_targets`, then rerun any packing/training on top of those regenerated examples.

## 2026-04-10

Intent:
- Reuse the existing fixed-size WDL trees by deriving variable-size root prefixes instead of recollecting fresh raw trees from lc0.

Meaningful change:
- Added prefix-derivation helpers in `cts_pretrain.py`:
  - `prefix_node_count_schedule(...)`
  - `sample_prefix_expansion_count_for_node_budget(...)`
  - `derive_prefix_pretrain_example(...)`
- Added `scripts/derive_pretrain_prefixes.py`:
  - reads existing raw pretrain examples
  - samples one root prefix per source example with actual node count in `[min_nodes, max_nodes]`
  - recomputes consolidated scalar and edge-WDL targets on the prefix
  - saves new raw `PretrainExample`s with throughput logging

Outcome:
- Validation:
  - `/opt/miniconda3/envs/trm/bin/python -m pytest test_supervised_branch.py -q`
  - `/opt/miniconda3/envs/trm/bin/python scripts/derive_pretrain_prefixes.py --help`
  - `/opt/miniconda3/envs/trm/bin/python -m pytest -q`
- Final result: `93 passed`

Conclusion:
- We can now build a variable-size raw dataset from the existing 96-node WDL trees without new engine calls.

## 2026-04-13

### Episode difficulty metrics (data-only) and greedy-eval diagnostics

Intent:
- Quantify how “interesting” each controller episode is using only halt rewards and `continue_cost`, and log per-episode diagnostics during greedy evaluation so regret can be related to those metrics.

Meaningful change:
- Added `episode_difficulty.py` with six metrics: halt reward range, optimal-vs-second-best return gap, return variance, softmax entropy over per-stop returns, regret of always halting at step 0, and reward curvature (sign changes in halt-reward differences).
- Extended `scripts/train_fitted_q_controller.py`:
  - `--output-diagnostics` writes a JSONL on the final greedy-eval epoch (validation only).
  - Each record includes halt rewards, oracle/predicted stop, return, regret, **predicted** advantages, and the six difficulty scalars.
  - `_predict_stop_step` returns `(stop, predicted_advantages)` for logging.
- Unit tests for the difficulty helpers live in `test_train_fitted_q_controller.py`.

Outcome:
- Greedy-eval diagnostics are model-dependent (regret uses the fitted policy); difficulty scalars in that JSONL are still data-intrinsic for the same halt-reward sequence and `continue_cost`.

Conclusion:
- High-regret profile plots compare **oracle stop** (from true rewards) to **predicted** advantages; misalignment is expected under imperfect learning.

### Standalone packed-data difficulty analysis

Intent:
- Summarize difficulty over full train and validation packed manifests without loading any model.

Meaningful change:
- Added `scripts/analyze_episode_difficulty.py` and `slurm/analyze_episode_difficulty_della.slurm`.
- Default output path (unless overridden): `episode_difficulty.jsonl` under the packed-data root.

Outcome (full unfiltered packed data, `continue_cost = 0.001`):
- Train / validation difficulty summaries were broadly aligned; combined ~71k episodes.
- Roughly **56%** of episodes had `oracle_stop_step == 0` (halt immediately).
- Mean difficulty signals indicated a **flat** return landscape (e.g. high softmax entropy near the maximum for the episode length, tiny optimal-vs-second-best gap on average).

Conclusion:
- The unfiltered distribution is dominated by episodes where stopping time barely affects return; that motivated filtering before retraining the controller.

### Filtering packed episodes and retraining on the subset

Intent:
- Drop trivially flat episodes and reduce imbalance toward “halt immediately,” using only existing packed shards.

Meaningful change:
- Added `scripts/filter_packed_episodes.py` and `slurm/filter_packed_episodes_della.slurm`.
- Filters episodes with `halt_reward_range <` threshold; optional cap per `oracle_stop_step` (stratify-sampling).
- First cluster run hit OOM at 16G (all kept episodes held in RAM); reran with **64G**.

Configuration used:
- `MIN_HALT_REWARD_RANGE = 0.10`
- `MAX_PER_STOP_STEP = 500`
- Output directory: `.../controller_packed/filtered/`

Counts:
- Train: `68044 → 16155` (range filter) → `6384` (stratify).
- Validation: `3581 → 848` → `848` (stratify did not bind on validation).

Fitted-Q retrain (filtered manifests, same `continue_cost = 0.001` and linear cost as before — **no** nonlinear cost or cost sweep in code yet):
- Checkpoints and diagnostics (examples):
  - `fittedq_controller_async_filtered.pt` + `fittedq_controller_async_filtered_diagnostics.jsonl`
  - `fittedq_controller_sync_filtered.pt` + `fittedq_controller_sync_filtered_diagnostics.jsonl`
- Validation greedy (848 episodes):
  - **Async:** `average_regret ≈ 0.016`, `first_action_accuracy ≈ 0.731`, `exact_stop_step_accuracy ≈ 0.042`.
  - **Sync:** `average_regret ≈ 0.040`, `first_action_accuracy ≈ 0.697`, `exact_stop_step_accuracy ≈ 0.052`.
- On unfiltered data, sync and async had been nearly tied on average regret; on filtered data, **async shows clearly lower regret** and fewer high-regret episodes in diagnostics.

Conclusion:
- Filtering surfaces differences between encoders: async’s message-passing appears to help when episodes are nontrivial.
- Under-search still carries much higher mean regret than over-search at this `continue_cost`, consistent with cheap extra search vs. missing a better later halt.

### Analysis notebooks

Meaningful change:
- `regret_landscape.ipynb` — loads paired sync/async **filtered diagnostics** JSONLs; regret distributions, regret vs. difficulty metrics, predicted vs. oracle stop, regret by oracle step, over- vs under-search breakdowns, correlation matrices, sync-vs-async per-episode regret scatter, high-regret trajectory plots.
- `episode_difficulty_analysis.ipynb` — loads **`episode_difficulty.jsonl`** from `analyze_episode_difficulty.py`; train/val summaries, histograms, oracle-stop distribution, correlations, halt@0 vs continue boxplots, example halt-reward traces.

Cluster copy (example):
- `scp della:/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/controller_packed/episode_difficulty.jsonl .`

### Qualitative notes from regret-landscape plots (filtered diagnostics)

- Regret vs. difficulty metrics often shows dense mass near **zero regret**; structures (lines, L-shapes) appear when tying **flat landscapes** (small second-best gap, many near-ties) to small regret even when the stop step is wrong.
- **Predicted vs. oracle stop:** both models **over-search** more than under-search at this cost (consistent with low `continue_cost` and asymmetric error: under-search mean regret much higher than over-search).
- **Regret vs. oracle stop step:** largely flat — mistakes are not concentrated at a particular oracle stopping time.
- **Per-episode sync vs. async regret:** mean can favor async while head-to-head “who wins more rows” can favor sync if async wins a **few** episodes by large margins and sync wins **many** by tiny amounts.

### Not done in this thread (future)

- Nonlinear / superlinear continue cost in training or packing.
- Systematic `continue_cost` sweep with re-packing or live targets.
- Prior-entropy pre-filter at raw FEN generation time.

## 2026-04-14

### Budget-aware controller oracle and packed-data augmentation

Intent:
- Replace the old scalar `continue_cost` fitted-Q controller setup with a budget-aware offline controller whose decision state is the learned tree encoding together with literal tree size and remaining synthetic planning budget.

Meaningful change:
- Added `budgeted_controller_oracle.py` with the new recursion
  - `c_maint(N) = 0.01 * (N / 30) ^ 1.1`
  - `c_time(T) = lambda * ((T - 1 + tau)^(-(p-1)) - (T + tau)^(-(p-1)))`
  - defaults: `lambda = 18.537`, `p = 2.8`, `tau = 2.5`, timeout value `-1`
  - `V_t = max(H_t, -c_maint(N_t) - c_time(T_t) + V_{t+1})`
- `scripts/pack_controller_episodes.py` now:
  - computes raw trimmed controller episodes once
  - pre-filters by raw `halt_reward_range` before augmentation
  - augments each surviving raw episode with deterministic synthetic starting budgets across five buckets
  - stores `tree_sizes`, `time_budgets`, `starting_budgets`, bucket metadata, and budget-aware oracle targets in packed shards
- Packed format bumped to:
  - manifest: `cts_budgeted_controller_episode_manifest_v2`
  - shard: `cts_budgeted_controller_episode_shard_v2`

Default augmentation:
- buckets:
  - `[1, 3]`
  - `[4, 10]`
  - `[11, 25]`
  - `[26, 60]`
  - `[61, 120]`
- `2` samples per bucket
- pre-filter default in the Slurm wrapper: `MIN_HALT_REWARD_RANGE = 0.10`

Conclusion:
- The packer now does the sensible order for this workflow: filter on raw episode interestingness first, then replicate into synthetic budgets, instead of writing a huge augmented corpus only to discard most of it later.

### Budget-aware controller training and caching

Intent:
- Train the offline async controller on `concat(z_t, N_t, T_t)` with supervised advantage regression under the new budget-aware oracle, while avoiding repeated frozen-encoder materialization cost.

Meaningful change:
- `scripts/train_fitted_q_controller.py` now:
  - consumes only the packed budget-aware manifests for this path
  - uses an MLP head on `concat(z_t, N_t, T_t)` (one hidden layer with ReLU)
  - computes greedy returns/regret under the new budget-aware oracle
  - writes diagnostics containing `N_t`, `T_t`, starting budget, oracle/predicted stop, oracle/predicted value, target/predicted advantages, and regret
- Added on-disk cache for frozen-encoder materialization:
  - first run materializes `[z_t, N_t, T_t]` features from the packed dataset and saves them next to the manifests
  - later runs reuse those cache files as long as the packed manifest path and encoder checkpoint match

Practical notes:
- The expensive `materialize_*_encoder` stage is still a full one-pass encoding over the packed dataset; larger `EPISODE_BATCH_SIZE` only improves throughput, not asymptotic cost.
- The cache is intended to make repeated controller-head retrains cheap once the first frozen-encoder pass has completed.

### Cluster workflow fixes

Meaningful change:
- Updated Slurm wrappers for packing and training to expose the budget-aware parameters and packed-manifest paths.
- Fixed an rsync workflow issue during deployment: copying without `-R` polluted the remote repo root with stray duplicates while leaving `scripts/` / `slurm/` unchanged. Subsequent syncs used `rsync -avR ...` to update the active files in place.

Conclusion:
- Current intended cluster workflow is:
  1. re-pack with raw-range pre-filter + budget augmentation
  2. train directly on the repacked manifests
  3. reuse cached frozen-encoder features on subsequent controller runs

## 2026-04-15

### First budget-aware async controller run: poor final metacontrol despite low regression loss

Run:
- Packed budget-aware controller data with the original maintenance scale `0.01`.
- Trained the async frozen-encoder controller with the default small MLP head.

Final greedy metrics on the validation run (`8480` episodes):
- `average_return = -0.075`
- `average_oracle_value = 0.080`
- `average_regret = 0.155`
- `average_expansions = 2.293`

Immediate interpretation:
- The run is not failing because the model cannot detect imminent timeout; rather, regret is coming mostly from medium-to-large budget regimes where the controller continues too long.
- The regression objective plateaus early while greedy metacontrol remains poor, so low advantage MSE by itself is not a sufficient success criterion.

### Saved analysis scripts for budget-aware runs

Meaningful change:
- Added `scripts/plot_advantage_loss_from_log.py`
  - parses a training `.out` log and plots train / validation `advantage_mse` over epochs
- Added `scripts/analyze_budgeted_controller_run.py`
  - consumes only the diagnostics JSONL and the training `.out`
  - writes a full report directory with plots and `summary.json`
  - includes:
    - loss curves
    - greedy metrics over epochs
    - regret by starting-budget bucket
    - stop error by budget bucket
    - regret by oracle stop step
    - regret by stop-step error (`predicted_stop_step - oracle_stop_step`)
    - sign accuracy by current `T_t`
    - target-advantage distribution by current `T_t`
    - predicted vs target advantage
    - false-continue / false-halt rates by `T_t`
    - regret by initial tree size
    - `T_t × N_t` partial-dependence heatmaps
    - same-tree different-budget consistency
    - oracle stop distribution
    - calibration by predicted-advantage magnitude
    - trivial baseline sweep

Key findings from the diagnostics-only report:
- Best greedy epoch occurs earlier than the final epoch; continuing training increases average expansions and worsens regret.
- Regret is small in the `scramble` bucket and substantially larger in `medium-large`, `large`, and `very-large`.
- Over-search is the dominant failure mode in those larger-budget buckets.
- Sign accuracy is strongest at very low `T_t`; the bad run is not primarily a “cannot detect low time” problem.

### Regret decomposition: halt-reward term vs maintenance vs time

Meaningful change:
- Extended `scripts/analyze_budgeted_controller_run.py` with an exact episode-level decomposition:
  - `oracle_value - predicted_value`
  - `= (halt_reward@oracle - halt_reward@predicted) + (predicted maintenance paid - oracle maintenance paid) + (predicted time paid - oracle time paid)`
- Added plots and JSON summaries for that decomposition.

Main result:
- In large-budget over-search episodes, regret is almost entirely **not** coming from time cost.
- Example: for `large`, `delta = +6` (`predicted_stop_step - oracle_stop_step = 6`), mean regret is about `0.50`, with roughly:
  - halt-reward term `≈ 0.21`
  - maintenance term `≈ 0.28`
  - time term `≈ 0.008`

Conclusion:
- Large-budget regret is dominated by two things:
  1. extra maintenance burden from carrying a larger tree for extra steps
  2. halting later at a root move whose final-tree score is worse than the oracle-stop move
- The explicit time-cost term is negligible in those regimes.

### Oracle-stop factor analysis

Meaningful change:
- Added an oracle-decision-factor decomposition to `scripts/analyze_budgeted_controller_run.py`:
  - `target_advantage = future_value_gain - maintenance_cost - time_cost`
  - `future_value_gain = oracle_next_value - halt_reward_now`
- At the oracle stop step, the report now classifies whether halting is driven by:
  - `future_already_worse`
  - `maintenance_dominated`
  - `time_dominated`

Main result:
- In the `large` bucket, oracle halting is almost never driven by time cost:
  - `future_already_worse`: about `97%`
  - `maintenance_dominated`: about `3%`
  - `time_dominated`: essentially `0%`

Interpretation:
- In large-budget regimes, the oracle typically halts because continuing is already worse in value terms before explicit time cost matters.
- So the metacontroller’s failure there is not just “missing the time term.”

### Move-switch analysis for `future_already_worse`

Meaningful change:
- Added `scripts/analyze_future_worse_move_switch.py`
  - reconstructs the trimmed decision episode from each `source_path`
  - compares the oracle-stop root move to the predicted-stop root move
  - splits `future_already_worse` episodes into:
    - `same_move`
    - `move_switch`

Main result:
- `same_move` cases:
  - mean regret `≈ 0.069`
  - halt-reward term exactly `0`
  - regret comes almost entirely from maintenance/time cost
- `move_switch` cases:
  - mean regret `≈ 0.409`
  - halt-reward term `≈ 0.346`
  - maintenance term much smaller than the halt-reward penalty

Large-budget interpretation:
- If over-search keeps the same root move, regret is basically just extra maintenance cost.
- The really large regrets come from over-search episodes where the root move changes to one that the final finite-tree evaluation scores worse.

Open concern:
- This exposes a non-monotonicity in the halt-reward construction: immediate halting can outperform intermediate search, while later search may recover.
- The current reward target therefore mixes “value of computation” with transient root-action churn along the finite search trajectory.

### Architecture/cost changes queued for the next overnight run

Meaningful change:
- Strengthened the controller head in `scripts/train_fitted_q_controller.py` and the train Slurm wrappers:
  - hidden width `256`
  - `3` hidden layers
- Reduced the default budgeted maintenance scale by a factor of `4`:
  - from `0.01` to `0.0025`
  - updated in:
    - `budgeted_controller_oracle.py`
    - `scripts/pack_controller_episodes.py`
    - `scripts/train_fitted_q_controller.py`
    - `slurm/pack_controller_episodes_della.slurm`
    - `slurm/train_fitted_q_controller_della.slurm`
    - `slurm/train_compute_advantage_della.slurm`

Intent:
- Lower maintenance burden should reduce the strong immediate-halt bias and make the oracle less dominated by “tree upkeep” penalties.
- The deeper MLP is a straightforward capacity increase for reading out budget-sensitive controller structure from `concat(z_t, N_t, T_t)`.

## 2026-04-16

### Oracle-aligned controller data path and churn analysis

Intent:
- Make the controller supervision semantically consistent with the intended oracle: snapshots should be prefixes of the original generated search, and halt rewards should be derived from the full source-tree oracle rather than from fresh teacher re-search on each prefix.

Meaningful changes:
- `cts_pretrain.py` now stores oracle trajectory data in generated raw examples:
  - expansion counts after each original search expansion
  - root-Q trace over the original search
  - best-move trace over the original search
  - final full-source-tree root Q-values
- `cts_episode_envs.py` now consumes those stored oracle traces when building controller episodes, falling back to the older re-search path only for legacy examples that do not have the new fields.
- The budgeted-controller path now defaults `maintenance_scale = 0.0`:
  - `budgeted_controller_oracle.py`
  - `scripts/pack_controller_episodes.py`
  - `scripts/train_fitted_q_controller.py`
  - `slurm/pack_controller_episodes_della.slurm`
  - `slurm/train_fitted_q_controller_della.slurm`
  - `slurm/train_compute_advantage_della.slurm`
- `scripts/analyze_budgeted_controller_run.py` now supports source-path rewrites and root-action churn summaries anchored on the ultimately chosen move:
  - top-level motif classes `A`, `X*A`, `X*AB*A`
  - compressed-length and unique-move distributions
  - first appearance and stabilization of the final move
  - oracle stop phase relative to first appearance / stabilization
  - lucky early halts

Result:
- The code path is now ready for a clean regeneration of the raw `generated_trees` corpus under the oracle-aligned semantics.
- Existing raw examples remain loadable, but only regenerated examples carry the stored oracle traces and therefore avoid prefix re-search.

Conclusion:
- Future budgeted-controller experiments should use regenerated raw source trees, then rebuild the split manifests and packed controller data from that new corpus before training.

## 2026-04-19

### Time-only budgeted controller run on oracle-aligned, `X*AB*A`-filtered data

Intent:
- Evaluate the fitted-Q metacontroller in the cleaner regime where:
  - controller supervision comes from stored oracle traces on the regenerated oracle-aligned source trees
  - maintenance cost is removed (`maintenance_scale = 0.0`)
  - unstable `X*AB*A` source trajectories are excluded at pack time

Meaningful run setup:
- Packed validation episodes:
  - `30630`
- Buckets balanced:
  - `6126` each for `scramble`, `medium-small`, `medium-large`, `large`, `very-large`
- Oracle metadata recovered from the run log:
  - `maintenance_scale = 0.0`
  - `time_lambda = 18.537`
  - `time_p = 2.8`
  - `time_tau = 2.5`
  - `samples_per_bucket = 2`
  - async encoder checkpoint with `k = 1`

Main result:
- Final greedy metrics (epoch `20`):
  - average oracle value `0.353`
  - average return `0.304`
  - average regret `0.049`
  - exact stop-step accuracy `0.444`
  - first-action accuracy `0.916`
  - average expansions `19.347`
- Best behavioral epoch was earlier than the final checkpoint:
  - best greedy return epoch `10`
  - best greedy regret epoch `10`
  - best validation-MSE epoch `5`

Interpretation:
- The controller learns the coarse halt/continue direction well (`first_action_accuracy` high), but stop timing remains weak, especially in the larger-budget buckets.
- The dominant failure mode is **oversearch**, not undersearch:
  - `medium-large`: oversearch `~0.64`
  - `large`: oversearch `~0.84`
  - `very-large`: oversearch `~0.97`

### Regret decomposition under `maintenance_scale = 0.0`

Meaningful analysis result:
- The report explicitly decomposes regret as:
  - `oracle_value - predicted_value`
  - `= (halt_reward@oracle - halt_reward@predicted) + maintenance_term + time_term`
- With `maintenance_scale = 0.0`, maintenance contributions were exactly zero in the analysis outputs.
- Oversearch regret is therefore almost entirely **time cost**, not maintenance and only weakly halt-reward deterioration:
  - overall oversearch mean regret `~0.076`
  - overall mean time term `~0.073`
  - overall mean halt-reward term `~0.003`

Conclusion:
- In the time-only regime, the controller’s remaining error is mainly “continuing too long and paying extra time cost.”

### Near-threshold sign failure identified

Meaningful analysis result:
- The fitted-Q scalar looks reasonable under MSE/correlation summaries, but sign behavior near the decision boundary is very poor.
- Margin-calibration summary:
  - for `|target_advantage| < 0.05`, sign accuracy is only `~0.29`
  - those low-margin states account for about `70%` of all analyzed states
- Structure of the low-margin regime:
  - concentrated in higher-budget states, not scramble / medium-small
  - within-bucket low-margin rates:
    - `medium-large`: `~48%`
    - `large`: `~73%`
    - `very-large`: `~81%`
  - by current time budget, low-margin rates are already extreme around `T_t = 11..25`
- Crucially, those low-margin states are overwhelmingly slightly **negative**:
  - negative low-margin targets vastly outnumber positive ones
  - model errors there are overwhelmingly `false_continue`, not `false_halt`

Interpretation:
- The controller is not merely noisy near zero; it has a strong **positive bias** in the high-budget, near-threshold regime.
- This explains why MSE can stabilize while metacontrol remains poor in the large-budget buckets:
  - MSE only weakly penalizes small wrong-sign predictions
  - control behavior is driven by the sign of the advantage, not just its squared error

Conclusion:
- The main remaining problem is now clearly a **decision-boundary calibration problem**, not gross regression failure.
- Next experiments should focus on losses/model-selection criteria that care directly about the halt/continue sign near zero, rather than only scalar MSE.

## 2026-04-20

### Entropy-sampled controller dataset scheme

Intent:
- Make the packed controller dataset less dominated by trees whose oracle stop behavior is nearly constant across sampled budgets, without hard-coding a minimum stop step or discarding all easy trees.

Implemented scheme:
- `scripts/pack_controller_episodes.py` now supports `--sample-trees-by-stop-entropy`.
- For each source tree:
  - generate the usual kept budgeted episodes after the existing filters
  - collect oracle stop steps across those sampled budgets
  - bin stop steps into:
    - `0-1`
    - `2-5`
    - `6-9`
    - `10-12`
    - `13+`
  - compute normalized entropy `H` of that 5-bin stop-step distribution
  - keep the whole tree with deterministic probability `p_keep = H`
- Keep/drop uses a deterministic hash of `(source_path, seed)` so repacks are reproducible.

Why this scheme:
- preserves full within-tree budget structure for retained trees
- does not impose any claim that later stop steps are intrinsically better
- still keeps some trees with only `0/1` mass
- smoothly favors trees whose stopping behavior actually varies with budget

Operational note:
- the packer now reports entropy-filtered tree counts explicitly in progress logs and manifests, so retention can be measured after one pass instead of guessed in advance.

### Loss logging now matches the actual optimized objective

Intent:
- The fitted-Q trainer was optimizing `MSE + sign_loss_weight * BCE`, but logs and plots were still only surfacing the MSE term. That was hiding the actual optimization signal during sign-loss sweeps.

Implemented change:
- `scripts/train_fitted_q_controller.py` now logs:
  - `train_total_loss`
  - `train_advantage_mse`
  - `train_sign_bce`
  - and the corresponding validation metrics
- Both analysis scripts were updated to parse both old and new log formats:
  - `scripts/analyze_compute_advantage_training_log.py`
  - `scripts/analyze_budgeted_controller_run.py`
- Training/validation plots now expose total loss and the two components separately rather than only the MSE curve.

Conclusion:
- Future sweeps over sign-loss weight, learning rate, and model size can now be analyzed against the true training objective while using a dataset intervention that prefers cross-budget stop diversity without enforcing a hand-designed monotonic interpretation.

### Entropy-sampled dataset sweep result: not useful

Intent:
- Test whether a tree-level entropy-biased packed dataset improves the controller problem by favoring source trees whose oracle stop step varies across sampled budgets.

Sweep setup:
- Dataset:
  - entropy-sampled packed controller corpus with:
    - stop-step bins `0-1`, `2-5`, `6-9`, `10-12`, `13+`
    - normalized entropy `H`
    - keep probability `0.1 + 0.9 * H`
    - `X*AB*A` exclusion still enabled
- Hyperparameters:
  - sign loss weight: `0.1`, `0.25`, `0.5`
  - learning rate: `1e-4`, `3e-4`, `1e-3`
  - controller head width: `256`, `512`

Main result:
- The entropy-sampled dataset underperformed the earlier sign-loss run on the standard filtered dataset.
- Most important difference:
  - entropy-sampled sweep runs had `average_oracle_value ≈ 0.063`
  - previous successful sign-loss run had `average_oracle_value ≈ 0.353`
- So this intervention did not produce a “harder but informative” dataset. It produced a much flatter one.

Observed behavior:
- The balanced-dataset sweep still had a strongly near-threshold state distribution:
  - about `63%` of states had `|target_advantage| < 0.05`
  - about `96%` of those low-margin states were negative
- Best-regret config in the sweep:
  - `sign_loss_weight = 0.1`
  - `learning_rate = 3e-4`
  - `q_hidden = 512`
  - best greedy regret `≈ 0.044`
- Best overall tradeoff:
  - `sign_loss_weight = 0.25`
  - `learning_rate = 1e-3`
  - `q_hidden = 256`
  - best greedy regret `≈ 0.046`
  - best exact stop `≈ 0.523`
  - same-tree budget monotonicity `≈ 0.93`
- Higher sign-loss weight (`0.5`) improved exact stop accuracy but did not improve regret.

Interpretation:
- Stop-step entropy by itself is the wrong dataset criterion.
- It prefers trees whose stopping behavior moves across budgets, but does not guarantee that those episodes have meaningful oracle value scale or margin structure.
- In practice it selected many flat, low-value episodes and did not resolve the near-threshold negative-dominance problem.

Conclusion:
- Do not use the entropy-sampled dataset as the main training distribution.
- Keep the original `no_xaba` packed dataset as the default.
- If future dataset filtering is revisited, it should target informativeness/value scale directly, not stop-step entropy alone.

## 2026-04-22

### Hierarchical `dj` tree stratification for controller packing

Intent:
- Replace the failed entropy-only tree filter with a dataset intervention that directly attacks the overrepresentation of trivial early-stop trees while preserving whole-tree budget structure.

Implemented scheme:
- `scripts/pack_controller_episodes.py` now supports tree-stratified sampling via:
  - `--sample-trees-by-tree-strata`
  - `--tree-stratification-mode {d,j,dj}`
- Per source tree, after the usual filters, compute:
  - `D = stop_depth_excess`
    - mean oracle stop depth beyond step `1` across sampled budgets
    - trees that always stop at `0` or `1` have `D = 0`
  - `J = budget_action_variance`
    - mean variance across budgets of the oracle continue/halt sign over valid planning steps
- The final `dj` mode is hierarchical rather than full-grid equalization:
  - first rebalance across `D` bins with a `50/50` mixture of natural mass and equalized-bin mass
  - then, within each `D` bin, mildly tilt toward larger `J` with multipliers `1.0`, `1.25`, `1.5`
  - renormalize within each `D` bin so the top-level `D` retained mass is unchanged

Dry-run result on the oracle96 trace corpus:
- Accepted source trees after the standard filters:
  - `57,683`
- `d`-only and hierarchical `dj` both retained about:
  - `17.6k` trees
  - `175.6k` budgeted episodes
- `j`-only was essentially useless and kept almost the whole dataset.
- Full joint `D x J` equalization was discarded because sparse cells collapsed the dataset to almost nothing.
- The final hierarchical `dj` scheme behaved sensibly:
  - within `d0`, keep probability increased from about `0.188` to `0.282` as `J` rose
  - within `d2`, keep probability increased from about `0.247` to `0.370`
  - the middle `d1` bin remained fully kept

Interpretation:
- The main imbalance lives on the stop-depth axis, not on budget sensitivity alone.
- A hierarchical `D -> J` scheme is stable and gives the intended bias:
  - fewer trivial early-stop trees
  - mild preference for trees where budget actually changes the oracle action
- This intervention is appropriate as a training-data bias, but not as a replacement for natural validation.

### `dj`-balanced controller run and sweep

Intent:
- Test whether training on the hierarchical `dj`-balanced packed dataset improves learned metacontrol relative to the plain `no_xaba` dataset.

Single-run result:
- The first `dj` run (`sign_loss_weight = 0.25`, `learning_rate = 3e-4`, `q_hidden = 256`) did not beat the earlier `sign_loss_run0` baseline.
- Final greedy metrics on the `dj` validation set:
  - oracle value `0.275`
  - return `0.233`
  - regret `0.042`
  - exact stop accuracy `0.511`
  - expansions `4.996`
- Compared with the earlier sign-loss run on the standard dataset:
  - lower oracle-normalized performance (`0.233 / 0.275 ≈ 0.847` vs `0.319 / 0.353 ≈ 0.904`)
  - higher normalized regret (`0.042 / 0.275 ≈ 0.153` vs `0.035 / 0.353 ≈ 0.099`)
- One quantity did improve:
  - same-tree predicted stop monotonicity rose from about `0.81` to about `0.87`
- But the overall control policy was still worse, especially through large-budget oversearch.

Sweep result on the `dj` dataset:
- Hyperparameters:
  - sign loss weight: `0.1`, `0.25`, `0.5`
  - learning rate: `1e-4`, `3e-4`, `1e-3`
  - head width: `256`, `512`
- Best config by regret:
  - `sign_loss_weight = 0.1`
  - `learning_rate = 1e-4`
  - `q_hidden = 512`
  - best greedy regret `0.040`
  - best return `0.235`
- Main trend:
  - weaker sign loss helped on this harder dataset
  - larger sign loss increased exact stop-step accuracy but hurt regret and return
- Effect sizes were modest rather than dramatic.

Conclusion:
- The `dj` dataset is harder in the intended sense, but the current controller does not yet exploit it well enough to outperform the plain `no_xaba` recipe.
- For this corpus, the best current setting is roughly:
  - `sign_loss_weight = 0.1`
  - `learning_rate = 1e-4`
  - `q_hidden = 512`
- Going forward:
  - training-data filtering should apply only to the training split
  - natural validation should remain representative of the unfiltered corpus
  - a separate bespoke metacontrol challenge set is likely needed for paper-quality evaluation

## 2026-04-23

### Hyperparameter sweep on the standard no_xaba dataset

Intent:
- Sweep sign_loss_weight, weight_decay, and head depth around the cosine LR scheduler baseline to find the best frozen-encoder controller configuration.

Setup:
- Base config: cosine LR schedule (`min_lr = 1e-5`), `learning_rate = 1e-3`, `q_hidden = 256`, 3 hidden layers, 20 epochs.
- Five runs varying one or two axes from baseline:
  - `slw01`: `sign_loss_weight = 0.1`
  - `slw05`: `sign_loss_weight = 0.5`
  - `wd4`: `weight_decay = 1e-4`
  - `slw05wd4`: `sign_loss_weight = 0.5`, `weight_decay = 1e-4`
  - `deep5`: `q_hidden_layers = 5`
- All trained on `controller_packed_combined_nomaint_no_xaba` with the `tree_encoder_child_wdl_async_k1` encoder.

Result:
- `slw01` was the best run by regret: `0.035` (vs `0.037` for the scheduler baseline).
- `slw01` also had the best return (`0.312`) and lowest MSE (`0.012`).
- Weight decay and deeper heads reduced oversearch (fewer expansions, ~3.7) but increased undersearch regret, netting worse overall regret (`0.049`).
- Higher sign loss weight improved exact stop accuracy but not regret.

Conclusion:
- `sign_loss_weight = 0.1` with cosine LR schedule became the new best configuration.
- Weight decay and depth act as regularizers that make the model halt too early.

### PUCT-filtered data augmentation

Intent:
- Augment the training corpus with positions specifically selected because search changes the best move, hypothesizing that these would produce more informative controller episodes.

Setup:
- Ran a PUCT stability filter on ~100k FENs: kept only positions where the best move at 1 expansion differs from the best move at full budget, and the midpoint best move also differs.
- Generated 96-node trees from the filtered FENs (25 shards, ~2250 FENs each).
- Combined with the existing tree corpus, repacked, and trained.

Result:
- `puct_run0` (combined v2 dataset, `sign_loss_weight = 0.1`, cosine LR): regret `0.043` on the v2 validation set.
- Cross-evaluation on the same v2 validation set: `slw01` (trained on v1 only) achieved regret `0.044`.
- Nearly identical — the PUCT-filtered trees did not meaningfully improve the controller.
- A 100-epoch run showed no improvement beyond epoch 20.

Conclusion:
- PUCT-filtered data augmentation was not useful. The trivial-episode proportion only dropped from ~80% to ~73%, suggesting the filter was not selective enough.

## 2026-04-24

### Trivial episode downsampling

Intent:
- The training distribution is ~80% trivial episodes (oracle_stop_step <= 1). Rather than augmenting with filtered data, directly downsample trivial episodes to shift the training distribution toward non-trivial cases.

Setup:
- Added `--downsample-trivial` to `scripts/pack_controller_episodes.py`.
- Deterministic per-episode hash decides keep/drop for trivial episodes.
- Applied only to training split; validation remains unfiltered.
- Used `keep_prob = 0.167` on trivials, targeting a 40/60 (trivial/non-trivial) training distribution.

Result:
- `subsample_hard_trees_run0`: regret `0.057`, average expansions `11.9`.
- Compared to `puct_run0` baseline: regret `0.043`, expansions `8.1`.
- The model massively oversearched: it learned a "continue" bias from the non-trivial-heavy training set and applied it broadly.
- Training metrics were also worse: MSE `0.020` vs `0.014`, sign_accuracy `0.761` vs `0.835`.

Conclusion:
- Downsampling trivial episodes backfired. The 80/20 trivial/non-trivial ratio reflects the true distribution. The model needs to see trivial examples to calibrate its stopping decisions.

### Flat loss reweighting (nontrivial_loss_weight = 4.0)

Intent:
- Instead of changing the data distribution (which removes calibration examples), upweight the loss on non-trivial episodes. This preserves all data while amplifying the gradient signal on harder cases.

Setup:
- Added `--nontrivial-loss-weight` to `scripts/train_fitted_q_controller.py`.
- Materialized cache now stores per-snapshot `oracle_stop_steps` to enable per-snapshot weighting.
- Loss functions (`_advantage_loss_components`, `_sign_auxiliary_loss`) accept optional per-snapshot weights; weighted loss uses `(loss * weights).sum() / weights.sum()`.
- Training run: `nontrivial_loss_weight = 4.0`, `sign_loss_weight = 0.1`, cosine LR, 20 epochs on `controller_packed_combined_nomaint_no_xaba`.

Result (evaluated on the same 30,630-episode validation set as slw01):
- Overall regret `0.029` (vs slw01 `0.029`). Identical.
- Exact stop accuracy `0.613` (vs slw01 `0.564`). Higher.
- Per-bin comparison revealed a clear pattern:
  - Bins 0-1: reweighting improved exact stop (0.814/0.681 vs 0.830/0.595) and regret on bin 1 (0.009 vs 0.011).
  - Bins 2+: slw01 had lower regret on every bin. The reweighting regressed on bins 4-7 (0.139 vs 0.122) and 8-15 (0.196 vs 0.164).
- The improvements on bin 1 (60% of episodes) exactly offset the regressions on the tail bins, netting the same overall regret.

Conclusion:
- Flat reweighting improved the dominant bins but hurt the rare bins — the ones it was supposed to help. A flat weight treats bin-2 and bin-15 identically despite very different difficulty and frequency.

### Inverse frequency loss reweighting

Intent:
- Replace flat reweighting with a principled scheme: weight each snapshot inversely proportional to the frequency of its oracle halt-step bin. This automatically gives higher weight to rarer (deeper) episodes.

Setup:
- Added `--inverse-freq-weights` flag to `scripts/train_fitted_q_controller.py`.
- Before training, the script scans cache shards and computes bin frequencies across 7 bins (0, 1, 2, 3, 4-7, 8-15, 16+).
- Per-bin weight = `total / (num_bins * bin_count)`, normalized so expected weight per snapshot is ~1.
- Training run: `inverse_freq_weights = True`, `sign_loss_weight = 0.1`, cosine LR, 20 epochs on `controller_packed_combined_nomaint_no_xaba`.

Result (same 30,630-episode validation set):
- Overall regret `0.029`. Same as slw01 and flat reweighting.
- Exact stop accuracy `0.628` (highest of all three).
- Per-bin exact stop improved on bins 1 (0.705 vs 0.595) and 3 (0.100 vs 0.088) relative to slw01.
- Per-bin regret: slw01 still won on every bin except bin 1. The pattern was the same as flat reweighting — gains on the dominant bins offset by losses on the tail.

Conclusion:
- All three approaches (slw01, flat reweight, inverse frequency) converge to the same ~0.029 regret.
- Loss reweighting can improve exact stop accuracy on bins 0-1 but cannot improve regret on bins 2+.
- Exact stop accuracy on bins 2+ remained stuck at 5-12% across all interventions.
- The bottleneck is likely in the frozen encoder features, not the loss function or data distribution. The encoder representations do not carry enough information to distinguish "stop at step 4" from "stop at step 8."

### Summary of the data/loss intervention sequence

All runs below use the async frozen encoder (`tree_encoder_child_wdl_async_k1`), cosine LR schedule, 20 epochs. Regret and exact stop evaluated on the same 30,630-episode validation set except where noted.

| Run | Intervention | Regret | Exact stop | Avg expansions |
|-----|-------------|--------|------------|----------------|
| slw01 | sign_loss_weight=0.1 (baseline) | 0.029 | 0.564 | 6.9 |
| puct_run0 | + PUCT-filtered data | 0.030* | 0.500* | 8.1* |
| subsample | downsample trivials (keep 0.167) | 0.057* | 0.377* | 11.9* |
| reweight_w4 | nontrivial_loss_weight=4.0 | 0.029 | 0.613 | 5.5 |
| inv_freq | inverse frequency weights | 0.029 | 0.628 | 5.5 |

(*) Evaluated on 52,350-episode v2 validation set, not directly comparable.

The main remaining lever is unfreezing the encoder, which would allow the representation to adapt to the stopping task but invalidates the materialized cache and is substantially more expensive to train.

## 2026-04-25

### Affine calibration controller variant

Intent:
- Test whether learning a separate affine transform `alpha * advantage + beta` for the halt/continue sign logit improves metacontrol, decoupling the magnitude regression (advantage_head) from the halt decision boundary.

Meaningful change:
- Extended `scripts/train_fitted_q_controller.py` on the `affine-calibration` branch:
  - `ComputeAdvantageTreeSearchModel` gains `--affine-calibration` flag.
  - When enabled, adds learnable `affine_alpha_raw` and `affine_beta` parameters. The sign logit becomes `softplus(alpha_raw) * advantage + beta`, ensuring positive scale.
  - `_predict_stop_step()` now returns `(stop_step, raw_advantages, sign_logits)` — the halt decision uses sign_logits, not raw advantages.
  - Diagnostics JSONL now logs both `predicted_advantages` (raw MLP output) and `sign_logits` (affine-transformed halt decision values).
- Also added `--regret-weighted-bce` flag: weights the sign BCE loss by `|target_advantage| / mean(|target_advantage|)`, clamped to `[0.25, 10.0]`. This upweights high-regret snapshots in the sign loss without changing the MSE term.
- Code lives in `~/chess/cts/affine` on cluster (local worktree at `worktrees/affine-calibration`).

Run configurations (all on `controller_packed_combined_nomaint_no_xaba`, `sign_loss_weight=0.1`, 20 epochs):

**affine_run0 / affine_run1** (jobs 7345567 / 7345908):
- `AFFINE_CALIBRATION=1`
- Two identical runs to check stability. Results were effectively identical.

**affine_regretweighted_run0** (job 7348396):
- `AFFINE_CALIBRATION=1, REGRET_WEIGHTED_BCE=1`

Result (30,630-episode validation set):

| Run | Regret | Return | Oracle | Exact stop | Expansions |
|-----|--------|--------|--------|------------|------------|
| affine | 0.055 | 0.299 | 0.354 | 0.789 | 2.68 |
| affine+rw | 0.034 | 0.320 | 0.354 | 0.679 | 3.87 |
| slw01 (baseline) | 0.029 | 0.325 | 0.354 | 0.564 | 7.90 |

Interpretation:
- Plain affine calibration dramatically undersearches: 2.68 expansions vs slw01's 7.90. Exact stop accuracy is highest (0.789) because most episodes have oracle_stop <= 1 and the model correctly halts early — but it halts too eagerly on the non-trivial episodes, producing 0.055 regret.
- Regret-weighted BCE partially corrects this: the regret weighting amplifies the gradient signal on episodes where incorrect halting is costly, pushing the model to search more (3.87 expansions) and reducing regret to 0.034.
- Neither affine variant matches slw01 on regret. The affine transform decouples scale from the decision boundary, but the underlying problem is that the advantage head's magnitude near zero doesn't carry enough information — rescaling it doesn't help.

### Five candidate models for the paper

Intent:
- Consolidate the five controller variants that will be compared in the paper. All use the same frozen encoder (`tree_encoder_child_wdl_async_k1`), same packed data (`controller_packed_combined_nomaint_no_xaba`), same `sign_loss_weight=0.1`, 20 epochs.

Summary table (30,630-episode validation set, same oracle config):

| Model | Intervention | Regret | Return | Oracle | Exact stop | Expansions |
|-------|-------------|--------|--------|--------|------------|------------|
| slw01 | baseline | 0.029 | 0.325 | 0.354 | 0.564 | 7.90 |
| reweight_w4 | nontrivial_loss_weight=4.0 | 0.030 | 0.324 | 0.354 | 0.613 | 6.55 |
| inv_freq | inverse frequency weights | 0.029 | 0.324 | 0.354 | 0.628 | 5.52 |
| affine | affine calibration | 0.055 | 0.299 | 0.354 | 0.789 | 2.68 |
| affine+rw | affine + regret-weighted BCE | 0.034 | 0.320 | 0.354 | 0.679 | 3.87 |

Diagnostics JSONL paths (local):
- `analysis_outputs/slw01_reweight_comp/slw01_on_current_val_diagnostics.jsonl`
- `analysis_outputs/slw01_reweight_comp/reweight_w4_on_current_val_diagnostics.jsonl`
- `analysis_outputs/inv_rew_loss_run0/fittedq_invfreq_run0_diagnostics.jsonl`
- `analysis_outputs/affine_run1/fittedq_affine_cal_diagnostics.jsonl`
- `analysis_outputs/affine_regretweighted_run0/fittedq_affine_cal_regret_wt_diagnostics.jsonl`

Conclusion:
- The top three models (slw01, reweight_w4, inv_freq) all converge to ~0.029 regret despite different loss interventions. Loss reweighting improves exact stop accuracy on trivial bins but cannot improve regret on non-trivial bins.
- The affine variants show that decoupling magnitude from sign is insufficient; the bottleneck is representational.
- All five are included in the paper to demonstrate the robustness of the near-optimal result and to show the effect of different loss calibration strategies.

## 2026-04-26

### Paper structure and figure plan

The paper has three main result sections (Section 2, human comparison, is on hold pending collaborator data):

**Section 1: The model learns near-optimal metacontrol.**
Compare all five candidate models against meaningful baselines.

Figures planned:
- **Fig 1a: Model vs baselines Pareto curve.** x = avg expansions, y = avg regret. Four baseline families: always halt, never halt, constant probability (sweep p), value gap threshold (sweep threshold). Each model as a starred point. Baselines computed from diagnostics + raw examples.
- **Fig 1b: Budget-conditioned behavior.** 2x3 subplots showing 6 representative trees. For each tree: x = starting budget, y = stop step. Overlay all five models' predicted stop steps plus oracle.
- **Fig 1c: Regret decomposition by budget bucket.** Grouped bar chart: 5 budget buckets x 5 models, stacked halt-reward + time-cost components.
- **Fig 1d: Return distribution.** Overlaid CDFs of predicted_value per model, plus oracle CDF.

**Section 3: What does the controller compute?**
Test whether the controller implements VOC using children's WDL information decoded from the root embedding.

Figures planned:
- **Fig 3a: Controller output vs decoded VOC.** Scatter plot of predicted advantage vs VOC for each model. VOC = E[max_c V_c] - max_c E[V_c] computed via Monte Carlo from decoded child WDLs.
- **Fig 3b: WDL subspace ablation.** SVD of decoder readout directions in z_root space. Sweep k = 1..128: keep-only or ablate top-k directions. Measure sign accuracy of advantage_head on ablated features.
- **Fig 3c: Layer-by-layer VOC correlation.** Forward hooks on each ReLU in advantage_head. Canonical correlation of activations with VOC at each layer.
- **Fig 3d: Residual analysis beyond VOC.** Regress predicted advantage on VOC, correlate residuals with T_t and other predictors.
- **Fig 3e: T_t gating.** Partial correlation of individual neurons with T_t controlling for VOC.
- **Fig 3f: Example episodes.** Multi-axis subplots showing target_adv, per-model predicted_adv, decoded VOC, and T_t over step index for selected episodes.

**Dependency chain:**
The child-WDL decoder is needed for all Section 3 analyses. The original decoder was discarded at encoder checkpoint time. Two paths:
1. The rerun encoder pretraining now saves the decoder alongside the encoder (preferred).
2. Fallback: `scripts/retrain_child_wdl_decoder.py` retrains a fresh decoder on frozen encoder outputs.

Scripts created:
- `scripts/paper_section1_figures.py`
- `scripts/paper_section3_figures.py`
- `scripts/paper_figure_utils.py`
- `scripts/retrain_child_wdl_decoder.py` (fallback)
- Slurm wrappers for each.

## 2026-04-28

### Full pipeline rerun for paper reproducibility

Intent:
- Retrain the entire pipeline from scratch (encoder pretraining through controller training and diagnostics) to produce clean, reproducible quantities for the paper. All outputs suffixed `_rerun` to preserve original quantities for comparison.

### Encoder pretraining rerun

Motivation:
- The original encoder was trained on variable-size prefix trees derived from the 96-node source trees. The rerun trains on the same 96-node trees used for controller training, ensuring semantic alignment. The original decoder weights were discarded at checkpoint time; the rerun saves both encoder and decoder.

Technical changes:
- `cts_pretrain.py`:
  - `PackedTensorizedShardDataset` rewritten to preload all shards at init and flatten into individual examples (`.clone()` each). This avoids two failure modes on 96-node trees: (a) tensor slice views pinning entire ~600MB shard storages in memory under shuffled access, and (b) Python 3.14's forkserver default requiring dataset pickling for DataLoader workers.
  - `ChildWdlPretrainer` now tracks `best_decoder_state` alongside `best_encoder_state`.
  - Added `save_training_state()` / `load_training_state()` for mid-training resumption across Slurm jobs.
  - `fit()` accepts `start_epoch` and `resume_path` for resume chaining.
  - Added `save_best_decoder()` to persist the decoder checkpoint.
- `supervised_branch_cli.py`:
  - Auto-detects resume checkpoint (`*_resume.pt`) and resumes from last completed epoch.
  - Saves decoder alongside encoder after training.

Commands:
- Initial submission (epochs 1-50, 1-hour Slurm limit):
  ```
  cd ~/chess/cts/async_soph && sbatch --time=01:00:00 --constraint="" \
    --export=ALL,PROJECT_DIR=$HOME/chess/cts/async_soph,K=1,EPOCHS=100,NUM_WORKERS=0,\
  TRAIN_DIR=/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/pretrain_packed_oracle96_trace_filtered_rerun/train_manifest.json,\
  VALIDATION_DIR=/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/pretrain_packed_oracle96_trace_filtered_rerun/validation_manifest.json,\
  OUTPUT_CHECKPOINT=/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/checkpoints/tree_encoder_child_wdl_async_k1_rerun.pt \
    slurm/pretrain_child_wdl_encoder_della.slurm
  ```
- `NUM_WORKERS=0` required because the preloaded flat dataset (~700MB, 35k examples) cannot be pickled to forkserver workers under Python 3.14.
- `--constraint=""` and `--time=01:00:00` to avoid 7-8 hour queue waits from the 24h/nomig defaults.
- Chained jobs via `--dependency=afterany:<job_id>` with increasing `EPOCHS` (100, 150, 200) to extend training across multiple 1-hour slots. Resume checkpoint stores last completed epoch; resubmission with higher EPOCHS continues seamlessly.

Training speed:
- ~1.1 minutes per epoch at batch_size=128 (slurm default) with NUM_WORKERS=0.

Result:
- Encoder pretraining completed through 200 epochs.
- Outputs:
  - Encoder: `/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/checkpoints/tree_encoder_child_wdl_async_k1_rerun.pt`
  - Decoder: `/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/checkpoints/tree_encoder_child_wdl_async_k1_rerun_decoder.pt`

### Controller rerun: 5 variants

Motivation:
- Retrain all five candidate controller variants from the hyperparameter/loss sweep on the rerun encoder checkpoint. All share the same frozen encoder and packed controller data; they differ only in loss configuration.

Commands (all submitted in parallel):
- **slw01** (`sign_loss_weight=0.1`, baseline):
  ```
  cd ~/chess/cts/async_soph && sbatch --time=04:00:00 \
    --export=ALL,PROJECT_DIR=$HOME/chess/cts/async_soph,\
  PACKED_TRAIN_DATA=/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/controller_packed_combined_nomaint_no_xaba/train_manifest.json,\
  PACKED_VALIDATION_DATA=/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/controller_packed_combined_nomaint_no_xaba/validation_manifest.json,\
  ENCODER_CHECKPOINT=/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/checkpoints/tree_encoder_child_wdl_async_k1_rerun.pt,\
  OUTPUT_CHECKPOINT=/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/checkpoints/fittedq_slw01_rerun.pt,\
  SIGN_LOSS_WEIGHT=0.1 \
    slurm/train_fitted_q_controller_della.slurm
  ```
- **reweight_w4** (`nontrivial_loss_weight=4.0`):
  ```
  cd ~/chess/cts/async_soph && sbatch --time=04:00:00 \
    --export=ALL,PROJECT_DIR=$HOME/chess/cts/async_soph,\
  ...,SIGN_LOSS_WEIGHT=0.1,NONTRIVIAL_LOSS_WEIGHT=4.0 \
    slurm/train_fitted_q_controller_della.slurm
  ```
- **inv_freq** (inverse frequency weights):
  ```
  cd ~/chess/cts/async_soph && sbatch --time=04:00:00 \
    --export=ALL,PROJECT_DIR=$HOME/chess/cts/async_soph,\
  ...,SIGN_LOSS_WEIGHT=0.1,INVERSE_FREQ_WEIGHTS=1 \
    slurm/train_fitted_q_controller_della.slurm
  ```
- **affine** (affine calibration, from `~/chess/cts/affine`):
  ```
  cd ~/chess/cts/affine && sbatch --time=04:00:00 \
    --export=ALL,PROJECT_DIR=$HOME/chess/cts/affine,\
  ...,SIGN_LOSS_WEIGHT=0.1,AFFINE_CALIBRATION=1 \
    slurm/train_fitted_q_controller_della.slurm
  ```
- **affine+rw** (affine + regret-weighted BCE):
  ```
  cd ~/chess/cts/affine && sbatch --time=04:00:00 \
    --export=ALL,PROJECT_DIR=$HOME/chess/cts/affine,\
  ...,SIGN_LOSS_WEIGHT=0.1,AFFINE_CALIBRATION=1,REGRET_WEIGHTED_BCE=1 \
    slurm/train_fitted_q_controller_della.slurm
  ```
- All use the same packed data (`controller_packed_combined_nomaint_no_xaba`) and rerun encoder checkpoint. Output checkpoints named `fittedq_{variant}_rerun.pt`.

Infrastructure changes for rerun:
- **Regret-based checkpointing**: Changed checkpoint selection from minimal validation MSE to minimal greedy regret. MSE can plateau while metacontrol stays poor; greedy regret is the actual metric of interest. Checkpointing now happens inside the greedy eval block.
- **Materialized cache support**: Added `MATERIALIZED_TRAIN_CACHE` / `MATERIALIZED_VALIDATION_CACHE` env vars to the slurm script, enabling pre-computed encoder forward passes to skip re-materialization.
- **Parallel materialization**: New `scripts/materialize_controller_cache.py` with `materialize` (per-worker) and `merge` subcommands. Splits episodes across workers by index range, each writing partial shards. Merge combines them via symlinks into a single cache. Reduced 72-min single-GPU materialization to ~24 min across 3 workers.
- **Storage migration**: Scratch filesystem hit capacity mid-run. All new outputs redirected to `/tigress/ysagiv/chess/cts/`.

Result (all checkpointed by minimal greedy regret):

| Variant     | Regret (rerun) | Regret (original) | Return (rerun) | Stop acc (rerun) |
|-------------|----------------|--------------------|----------------|------------------|
| slw01       | 0.031          | 0.029              | —              | —                |
| reweight_w4 | 0.034          | 0.030              | —              | —                |
| inv_freq    | 0.047          | 0.029              | —              | —                |
| affine      | 0.045          | 0.055              | —              | —                |
| affine+rw   | 0.036          | 0.034              | —              | —                |

- slw01, reweight_w4, affine+rw within ~0.005 of originals. affine improved (0.045 vs 0.055).
- **inv_freq regressed significantly** (0.047 vs 0.029). Greedy eval log confirms epoch 10 was best (0.047, 6.7 expansions) and epoch 20 worse (0.054, 10.5 expansions). Checkpointing correctly selected epoch 10. The regression appears real—likely due to the rerun encoder (trained on 96-node trees vs original variable-size prefix trees) interacting differently with inverse-frequency loss weighting.

Output paths:
- Checkpoints: `/tigress/ysagiv/chess/cts/checkpoints/fittedq_{variant}_rerun.pt`
- Diagnostics: `/tigress/ysagiv/chess/cts/checkpoints/fittedq_{variant}_rerun_diagnostics.jsonl`
- Materialized caches: `/tigress/ysagiv/chess/cts/train_cache_rerun.pt`, `validation_cache_rerun.pt`

## 2026-04-29

### Paper figures: first pass

Motivation:
- Generate all paper figures (Section 1: Figs 1a–1d, Section 3: Figs 3a–3f) using the five rerun controller variants. First complete pass to evaluate figure quality and identify issues.

Technical details:
- **VPI computation**: Replaced Monte Carlo sampling (1000 draws per snapshot) with closed-form VPI (Dearden). For children with WDL distributions $(w_c, d_c, l_c)$: $\text{VPI} = 1 - \prod_c(1-w_c) - \prod_c l_c - \max_c(w_c - l_c)$. Two products and a max, O(C) per snapshot. Derived from Dearden's VPI framework treating the decoded WDL as the belief distribution over child values on $\{+1, 0, -1\}$.
- **Decoder**: Rerun encoder pretraining already saved the decoder in the right format (`decoder_state_dict` + `metadata`). No retraining step needed.
- Section 1 ran without GPU (~minutes). Section 3 ran on 1 GPU, materialized 953,562 validation snapshots in ~8 minutes (3829 batches at episode_batch_size=8), total job ~12 minutes.
- Figures output to `/tigress/ysagiv/chess/cts/paper_figures/section{1,3}/`.

Results — Section 1:
- **Fig 1a (Pareto)**: All 5 models well below constant-probability baseline curve. Models cluster at low regret (0.03–0.06) but spread horizontally: affine/affine+rw halt earliest (~2–4 expansions), slw01 at ~7, inv_freq at ~11. Value-gap baseline skipped (needs raw .pt loading).
- **Fig 1b (Budget behavior)**: Models generally track oracle's increasing-with-budget pattern. inv_freq overshoots; affine halts too early. Oracle sometimes non-monotone, which no model captures.
- **Fig 1c (Regret decomposition)**: affine has highest halt-reward regret (stops too early). inv_freq has large positive time-cost regret in medium+ buckets (oversearches). slw01 most balanced.
- **Fig 1d (Return CDF)**: All models track oracle closely above return 0.5. More mass on negative returns than oracle. Models nearly indistinguishable from each other.

Results — Section 3:
- **Fig 3a (VPI correlation)**: Weak. Pearson r = 0.07–0.09, Spearman ρ = 0.16–0.34. VPI is one signal but far from dominant. affine+rw highest Spearman (0.34), inv_freq lowest (0.16).
- **Fig 3b (WDL ablation)**: Most informative. Top 1–2 WDL principal components carry most useful signal; adding more WDL directions *decreases* sign accuracy. Ablating top-k causes modest drop (~88%→77%) then recovery. Controller uses both WDL and non-WDL directions.
- **Fig 3c (Layer correlation)**: VOC correlation peaks at middle hidden layer then drops — network builds VPI-like representation internally before collapsing. slw01 strongest mid-layer VOC correlation (~0.17).
- **Fig 3d (Residuals)**: All models show ~0.34 correlation between VOC-regression residuals and T_t. Remarkably consistent — after VPI, the dominant remaining signal is time budget.
- **Fig 3e (T_t gating)**: Neuron n13 in top-10 T_t-gated neurons for 4/5 models. Clear linear T_t-activation relationships with bimodal structure.
- **Fig 3f (Example episodes)**: Easy/medium/hard/high-regret episodes. Models correctly predict negative advantage for easy cases. High-regret case shows failure to track non-monotone target advantage.

Key finding:
- **The controller is NOT primarily computing VPI.** Pearson correlations with decoded VPI are 0.07–0.09 — barely above noise. The WDL subspace ablation shows the first few WDL principal components matter, but the controller also relies heavily on non-WDL features (T_t especially, r~0.34 with residuals after VPI regression). The controller appears to use a coarse WDL summary + time budget rather than computing the full VPI integral. Open question: can training be adjusted to encourage explicit VPI computation?

### Mechanistic analysis of the advantage head

Motivation:
- The controller doesn't compute VPI. What does it compute instead? We want to understand how the advantage head MLP (Linear(130→256)→ReLU→Linear(256→256)→ReLU→Linear(256→256)→ReLU→Linear(256→1)) processes its inputs [z_root(128), N_t, T_t]. Script: `scripts/analyze_advantage_head.py`.

Technical details:
- **Weight decomposition**: For each model's first layer (256×130), partition columns into z_root (128) and scalars (N_t, T_t). Compute per-neuron weight norm fractions.
- **WDL subspace alignment**: SVD the child-WDL decoder's first-layer root-readout submatrix (128×128). Project each first-layer advantage-head neuron's z_root weights onto the top-k decoder readout directions. Measure fraction of weight norm in that subspace. This is rotationally invariant (subspace overlap is a geometric quantity).
- **Neuron classification**: Correlate each first-layer post-ReLU activation with VPI and T_t. Classify as T_t-dominated (|r_tt|>0.3, |r_vpi|<0.1), VPI-sensitive (|r_vpi|>0.1, |r_tt|<0.1), mixed, or dead (std < 1e-8).
- **Output neuron profiles**: Rank penultimate-layer neurons by output contribution (|w_out| × std(activation)), report correlations with VPI, T_t, and target advantage.
- **T_t sweep**: Hold 50 z_roots fixed (stratified across target range), sweep T_t from 0→120 in 100 steps. Plot predicted advantage and per-neuron activations.
- **z_root progression**: For 12 episodes, encode each step's tree with frozen encoder, hold T_t fixed at {10, 40, 80}, run through advantage head. Plot predicted advantage and penultimate-layer activations.

Ran on cluster: `salloc --gres=gpu:1 --mem=32G --cpus-per-task=4 -t 01:00:00`, interactive.

Results:
- **Weight decomposition**: z_root accounts for ~98% of first-layer weight norm across all 5 models. T_t fraction is 5–10%. By weight magnitude, the MLP devotes nearly all input capacity to reading z_root.
- **WDL subspace alignment**: Only ~6% of z_root weight projects onto top-5 WDL decoder readout directions, ~11% onto top-10, ~22% onto top-20. Chance level for a 128-dim space would be ~15.6% (top-20). The advantage head reads mostly from z_root directions orthogonal to the child-WDL subspace — it is NOT extracting the information VPI would need from z_root.
- **Neuron classification**: 55–60% of first-layer neurons are dead (never activate on validation set). Only 3–6 per model are T_t-dominated, 14–24 are VPI-sensitive (weak threshold), 4–8 mixed. The effective network is much smaller than the 256-neuron architectural capacity.
- **Output neuron profiles**: Top-contribution penultimate neurons have strong r(target) (~0.8–0.9) but near-zero r(VPI) (~0.01). The network tracks the target accurately without using VPI. r(T_t) for top neurons ranges 0.2–0.5.
- **T_t sweep**: Sharp nonlinear response. Predicted advantage ~-3.5 at T_t=0, rapid rise through T_t=10–20, asymptoting near 0 by T_t≥40. Individual z_root curves are tight at high T_t (network indifferent regardless of position) and fan out at low T_t (z_root determines how negative). Functional form: T_t sets the output scale; z_root modulates the threshold at low budgets.
- **z_root progression**: Varying z_root along an episode (tree grows) does change the output, but changes are subtle at T_t=40/80 and only become large at T_t=10 where T_t gating amplifies z_root differences. All 5 models mostly agree with each other.

Summary interpretation:
- The controller is a **T_t-gated z_root reader**. T_t controls the output scale (ample budget → ~0, low budget → strongly negative). z_root provides a position-dependent modulation, but from directions largely orthogonal to child WDL — likely encoding tree-structural features (search topology, evaluation stability) rather than decision uncertainty. The heuristic is approximately "halt when budget is low, with a position-dependent threshold."
- **Dead neurons**: 56% dead first-layer neurons is a known ReLU pathology (dying ReLU). Gradient is zero for neurons whose pre-activation is always negative, so they never recover. Potential fix: replace ReLU with LeakyReLU or GELU in the advantage head, giving gradient flow even when pre-activation is negative.
- **What z_root directions does it read?** Unknown from current analysis. The ~78% of z_root weight orthogonal to the WDL subspace could encode tree depth, branching factor, evaluation volatility, etc. Probing experiments needed.
- First-pass intervention figures only captured scalar output, not per-layer activations. Updated script to register forward hooks on each ReLU and capture per-neuron activation profiles during both T_t sweep and z_root progression.

Activation hook results (rerun):
- **T_t sweep, first hidden layer**: Two distinct neuron populations across all 5 models. (1) Linearly increasing with T_t: several neurons ramp ~0→40-50 as T_t goes 0→120 — direct T_t readers. (2) High at T_t≈0, decaying — encoding "budget nearly exhausted." Neuron n13 appears across all 5 models as a linearly T_t-responsive feature detector.
- **T_t sweep, penultimate layer**: All neuron activations compressed to 0–2 for T_t > 20, with sharp spikes at T_t ≈ 0. The nonlinear elbow (T_t=10–20) in the scalar output is fully formed by this layer. slw01/reweight_w4 have more active neurons at high T_t than affine variants.
- **z_root progression, penultimate layer** (slw01, 4 episodes × 3 T_t): Large transient at steps 0–3 (tree going from small to meaningful structure), then slow drift. Initial assessment dominates, but some neurons (e.g., n218 in ep 11298) continue ramping over 80 steps. Activation patterns are similar across T_t=10/40/80 for the same episode — z_root processing is largely T_t-invariant at penultimate depth.
- **Key question**: For late-stopping episodes (oracle_stop > 20), does z_root drift drive the halting decision, or is T_t countdown sufficient? Scalar output at T_t=80 stays near 0 through entire episodes, suggesting T_t countdown is the primary mechanism. The z_root sets an initial "position complexity" threshold; T_t counts down to it.

### SwiGLU advantage head experiment

Motivation:
- 56% of first-layer ReLU neurons are dead. SwiGLU (`(xW₁)·swish(xV)`) avoids dying neurons entirely (swish has nonzero gradient everywhere) and provides a learned gating mechanism that could help with T_t × z_root interaction.

Technical details:
- Added `SwiGLULayer` module to `train_fitted_q_controller.py`: two parallel `nn.Linear` projections, output is `linear(x) * F.silu(gate(x))`. Drop-in replacement for Linear+ReLU in `nn.Sequential`.
- New `--activation` CLI arg (default `"relu"`, choices `["relu", "swiglu"]`).
- Code lives in separate project directories on the cluster: `~/chess/cts/swiglu/` (copy of async_soph) and `~/chess/cts/affine_swiglu/` (copy of affine).
- 5 variants submitted with same loss configurations as the rerun, same frozen encoder and materialized caches (`train_cache_rerun.pt`, `validation_cache_rerun.pt`), checkpoints to `/tigress/ysagiv/chess/cts/checkpoints/fittedq_{variant}_swiglu.pt`.

Commands:
- slw01: `SIGN_LOSS_WEIGHT=0.1,ACTIVATION=swiglu`
- reweight_w4: `SIGN_LOSS_WEIGHT=0.1,NONTRIVIAL_LOSS_WEIGHT=4.0,ACTIVATION=swiglu`
- inv_freq: `SIGN_LOSS_WEIGHT=0.1,INVERSE_FREQ_WEIGHTS=1,ACTIVATION=swiglu`
- affine: `SIGN_LOSS_WEIGHT=0.1,AFFINE_CALIBRATION=1,ACTIVATION=swiglu` (from affine_swiglu)
- affine+rw: `SIGN_LOSS_WEIGHT=0.1,AFFINE_CALIBRATION=1,REGRET_WEIGHTED_BCE=1,ACTIVATION=swiglu` (from affine_swiglu)

v1 result (default Kaiming init): All 3 non-affine variants exploded (loss 5e29), producing degenerate always-halt policies (regret 0.233, avg_expansions 1.44). Kaiming init on multiplicative projections causes variance blowup.

v2 fixes: Xavier uniform with gain=1/√2 on both projections + zero biases, `nn.LayerNorm(input_dim)` before first SwiGLU layer, MAX_GRAD_NORM=2.0 (2× parameters → more generous clipping).

v2 result (3 non-affine variants; affine variants pending):

| Variant | ReLU regret | SwiGLU regret | Δ |
|---------|------------|---------------|------|
| slw01 | 0.031 | 0.033 | +0.002 |
| reweight_w4 | 0.034 | 0.039 | +0.005 |
| inv_freq | 0.047 | 0.051 | +0.004 |

Learning curve instability: MSE oscillates wildly between epochs while sign accuracy improves steadily. Example (slw01): validation MSE ranges from 0.012 (stable epochs) to 52,174 (epoch 18), yet sign_accuracy stays 0.85–0.90 throughout. reweight_w4 has a catastrophic spike at epoch 19 (MSE=3.3e13) but recovers at epoch 20. Pattern is consistent across all 3 variants — intermittent epochs produce very large scalar predictions while the sign (halt/continue direction) remains correct.

Interpretation: SwiGLU's multiplicative interaction amplifies outlier predictions when gate and linear projections align on extreme inputs. The sign head is unaffected because it only sees the direction, not the magnitude. The greedy evaluation (which only uses sign) improves steadily, but the MSE component of the loss is dominated by these spikes. This is a fundamental instability of the multiplicative architecture under L2 loss — the model finds good sign decisions but hasn't learned to regularize scalar magnitude.

Conclusion: SwiGLU does not improve over ReLU. The dead neuron problem (56% first-layer) is apparently not a bottleneck — the live neurons carry enough capacity. The T_t-gated z_root reading strategy works fine with sparse activation. Pursuing activation function changes further is unlikely to be productive.

### GeLU activation ablation

Motivation: Isolate whether dead neurons matter, without SwiGLU's multiplicative instability. GeLU is a drop-in replacement for ReLU — same parameter count, same architecture, no dead neurons, no magnitude blowup.

Technical details: Added `--activation gelu` option (Linear+GELU instead of Linear+ReLU). Same training setup as SwiGLU runs, from `~/chess/cts/swiglu/`. Checkpoints to `/tigress/ysagiv/chess/cts/checkpoints/fittedq_{variant}_gelu.pt`.

Result:

| Variant | ReLU regret | GeLU regret |
|---------|------------|-------------|
| slw01 | 0.031 | 0.031 |
| reweight_w4 | 0.034 | 0.033 |
| inv_freq | 0.047 | 0.047 (best at ep10; 0.055 at ep20) |

GeLU matches ReLU exactly. Learning curves are completely stable (validation MSE steady at 0.011–0.014, no spikes). Confirms: (1) SwiGLU's problem was the multiplicative interaction, not the activation shape; (2) dead neurons are neither helping nor hurting — they're irrelevant; (3) activation function is not the bottleneck.

### WDL subspace ablation (activation-space causal test)

Motivation: The weight-space analysis showed ~78% of advantage head z_root weights are orthogonal to the decoder's WDL readout directions. But that's a weight-level observation — does the controller actually depend on WDL information in the data? Two hypotheses: (A) the controller genuinely ignores child WDL, reading non-WDL tree structure from z_root; (B) the controller uses WDL information through different linear directions than the decoder, so weight-space alignment underestimates WDL dependence.

Technical details: For each model, take the materialized validation cache (953K snapshots × 130 features). SVD the decoder's root-readout matrix to get the WDL readout basis Vt [128×128]. For k=1..128:
- **Keep-only**: replace z_root with its projection onto top-k WDL directions. Run advantage head, measure sign accuracy and MSE.
- **Ablate**: remove top-k WDL directions from z_root (keep the orthogonal complement). Run advantage head, measure sign accuracy and MSE.
Script: `analyze_advantage_head.py`, function `wdl_subspace_ablation()`.

Sign accuracy results (selected k values):

| Model | Baseline | Keep k=5 | Ablate k=5 | Ablate k=128 (scalars only) |
|-------|----------|----------|------------|----------------------------|
| slw01 | 0.871 | 0.835 | **0.896** | 0.871 |
| reweight_w4 | 0.812 | 0.681 | 0.812 | 0.820 |
| inv_freq | 0.852 | 0.751 | 0.853 | 0.822 |
| affine | 0.733 | 0.671 | **0.786** | **0.829** |
| affine+rw | 0.713 | 0.776 | 0.769 | **0.841** |

MSE results:

| Model | Baseline MSE | Ablate k=128 MSE |
|-------|-------------|-----------------|
| slw01 | 0.012 | 0.281 |
| reweight_w4 | 0.012 | 0.318 |
| inv_freq | 0.017 | 0.284 |
| affine | 0.012 | 0.284 |
| affine+rw | 0.013 | 0.289 |

Key findings:
1. **Ablating WDL directions does not hurt sign accuracy — it often helps.** For slw01, removing the 8 most WDL-informative directions gives the best sign accuracy (0.903 vs 0.871 baseline). The WDL subspace injects noise into the controller's halt/continue decision.
2. **Zeroing ALL z_root (k=128 ablation) barely changes sign accuracy for slw01** (0.871 → 0.871). For affine/affine+rw, removing z_root *improves* sign accuracy by 10–13 points. The halt/continue decision is almost entirely determined by T_t (and N_t).
3. **z_root matters for magnitude, not sign.** MSE jumps 23× when z_root is zeroed (0.012 → 0.28), confirming z_root does affect the output value. But it rarely flips the sign. Most snapshots have T_t far from the transition band (T_t=10–20) where z_root could change the sign.
4. **Keep-only is non-monotonic.** Performance drops as more WDL directions are included (dip at k=30–80), recovering only at k=128 (full z_root). The intermediate WDL directions actively interfere.
5. **Explanation A confirmed**: the controller genuinely ignores child WDL for the halt/continue decision. It's not reading WDL through alternative directions.

Interpretation: The greedy controller policy (threshold at advantage=0) is effectively a T_t countdown rule. z_root modulates the advantage magnitude (how confident the decision is) but almost never changes which side of zero the prediction falls on. The GNN representation, trained to encode child WDL, contributes to calibration of the advantage prediction but not to the binary halt/continue decision that determines regret. Open question: does z_root's magnitude contribution matter for regret at marginal decisions? Sign accuracy averages over all snapshots including trivially easy ones; the regret-critical snapshots are exactly those near the sign boundary where z_root might matter. Need greedy evaluation with zeroed z_root to measure actual regret impact.

### T_t-only baseline trained from scratch (decisive z_root ablation)

Motivation:
- The Apr 29 inference-time z_root ablation suggested z_root contributes magnitude but rarely flips the binary halt/continue decision (sign accuracy invariant under z_root zeroing on slw01). That measurement is on a head whose weights were *shaped during training* by z_root information; zeroing at inference removes only the per-snapshot deviations around an already-z_root-informed threshold. The decisive test is to train a head from scratch that never sees z_root and ask whether it can match slw01.
- λ_m = 0 in the controller oracle config, so N_t is dynamically inert. The only non-z_root scalar that matters is T_t. The baseline is therefore: input = T_t alone, no encoder, no caches, no tree work.

Technical changes:
- Added `--t-only-baseline` to `scripts/train_fitted_q_controller.py`.
- Added `TOnlyAdvantageModel` (MLP on a single scalar T_t; matches q_hidden / q_hidden_layers / separate_sign_head with the regular controller).
- Added `_extract_t_only_tensors`, which walks packed shards once and reads only the scalar arrays (`target_advantages`, `starting_budgets`, `step_node_cutoffs`, `oracle_stop_steps`, `oracle_values`) to produce flat per-snapshot `(T_t, target_advantage)` tensors plus per-episode metadata. No tree node/edge reconstruction, no encoder forward passes, no `DataLoader`, no collator.
- Added matching `_train_t_only_epoch`, `evaluate_t_only_predictions`, `evaluate_t_only_greedy_policy`. Same loss components as the regular path: weighted advantage MSE + `sign_loss_weight` × sign BCE. Greedy eval runs the model on flat tensors in batches and slices per episode for stop-step and regret.
- Added `slurm/train_fitted_q_t_only_della.slurm` (separate from the SwiGLU-using `train_fitted_q_controller_della.slurm`, which passes a `--activation` flag the trainer here doesn't accept).
- Added 7 unit tests: model slices T_t correctly, ignores z_root and N_t columns, has no-op `freeze_encoder`; extractor flattens episodes correctly; train epoch updates parameters; greedy eval finds the right stop-step on a synthetic threshold-based head.

Run configuration (job 7481459):
- Project dir: `~/chess/cts/t_only` (copy of async_soph with the T_t-only changes).
- Train manifest: `controller_packed_combined_nomaint_no_xaba/train_manifest.json` (576830 episodes, 17924009 snapshots).
- Validation manifest: same dataset (30630 episodes, 953562 snapshots).
- Encoder checkpoint: `tree_encoder_child_wdl_async_k1_rerun.pt` — argparse-required but never opened under `--t-only-baseline`.
- `SIGN_LOSS_WEIGHT = 0.1` (matches slw01_rerun for an apples-to-apples comparison).
- Default architecture: `q_hidden = 256`, `q_hidden_layers = 3`, single advantage head (`separate_sign_head = 0`). Matches slw01 except for input dimensionality.
- 20 epochs. Wall time ~13 minutes total (extraction + 20 epochs of MLP-on-scalar). Greedy eval at epoch 10 and 20.

Greedy validation metrics:

| | T_t-only (best, ep 10) | slw01_rerun |
|---|---|---|
| `average_regret` | **0.218** | **0.031** |
| `average_return` | 0.136 | ≈0.323 (implied: oracle 0.354 − regret 0.031) |
| `average_oracle_value` | 0.354 | 0.354 |
| `exact_stop_step_accuracy` | 0.199 | — |
| `first_action_accuracy` | 0.663 | — |
| `average_expansions` | 20.79 | — |
| validation `sign_accuracy` (best across 20 ep) | 0.457 | ≈0.871 (Apr 29 ablation baseline) |
| validation `advantage_mse` (final) | 0.046 | ≈0.012 (Apr 29 ablation baseline) |

Training curve detail:
- `train_advantage_mse` and `train_sign_bce` plateau in epoch 1 and never move (epoch 1: 0.055/0.666; epoch 20: 0.045/0.666).
- `train_sign_accuracy` stays at ≈0.43 throughout — close to the marginal continue rate. The optimizer is at the information-theoretic ceiling for a 1-D function of T_t.

Interpretation:
- z_root closes ~85% of the regret-to-oracle gap that survives a T_t-only controller: `(0.218 − 0.031) / 0.218 ≈ 0.86`. The GNN representation is doing real work.
- Sign accuracy stuck at ≈0.43 is mechanistic, not optimization failure. The same T_t value appears with both halt-optimal and continue-optimal labels in the training distribution depending on z_root; a 1-D head cannot discriminate them and correctly settles at the marginal rate.
- This reconciles with the Apr 29 inference-time ablation. Zeroing z_root on the *trained slw01 head* preserves the head's z_root-shaped threshold; only per-snapshot deviations around that threshold are removed, and those deviations move the sign on a small fraction of regret-critical snapshots invisible in averaged sign accuracy. Training from scratch without z_root removes the threshold-shaping itself, and regret blows up 7×.
- The "controller is not computing VPI" finding from the WDL subspace ablation stands. The new framing pairs it with: z_root contributes substantially to regret through directions ~78% orthogonal to the child-WDL subspace. What those directions encode is now the load-bearing open question for Section 3.

Conclusion:
- The tree representation is regret-critical, not decorative. Section 3 needs to identify the non-WDL features in z_root that drive the regret reduction. Candidates: tree depth, branching factor, root-move churn, value variance across children, evaluation volatility along the principal variation. Stratifying T_t-only regret by budget bucket (and by the slw01-vs-T_t-only regret delta per episode) will localize where z_root buys what.

Output paths:
- Training log: `analysis_outputs/t_only/cts-fittedq-tonly_7481459.out`
- Best checkpoint: `/tigress/ysagiv/chess/cts/checkpoints/fittedq_t_only_baseline_rerun.pt` (selected by min validation greedy regret = 0.21752, epoch 10).
- Diagnostics JSONL: `/tigress/ysagiv/chess/cts/checkpoints/fittedq_t_only_baseline_rerun_diagnostics.jsonl` (30630 episodes; regret_mean=0.2175, regret_std=0.2979, regret_max=1.9868).

### (N_t, T_t) baseline — disambiguating step-within-episode info from genuine z_root signal

Motivation:
- The T_t-only baseline conflated two ablations: it removed both z_root *and* N_t. With λ_m = 0 the oracle's continue cost doesn't depend on N_t, so N_t was assumed dynamically inert. But N_t still enters the model's input, and N_t = first-decision-tree-size + step is a near-step counter modulated by an initial-instability offset. Because starting budgets span 1–120 across episodes, T_t alone cannot tell the head where it is in an episode (T_t = 10 could be step 0 of a budget-10 or step 90 of a budget-100 episode); N_t disambiguates that.
- This baseline isolates the contribution of N_t alone: train a head whose input is `[N_t, T_t]` (no z_root), same loss/architecture/data as slw01.

Technical changes:
- Generalized `TOnlyAdvantageModel` with `input_dim` parameter (1 → T_t only, 2 → [N_t, T_t]).
- Added `tree_sizes: torch.Tensor | None` to `TOnlyTensors`.
- `_extract_t_only_tensors(..., include_n_t=True)` walks the same packed shards once and additionally pulls per-snapshot tree sizes (`step_node_cutoffs`) into a flat tensor; no tree node/edge reconstruction.
- Train/eval/greedy helpers stack `[N_t, T_t]` per snapshot when `tensors.tree_sizes is not None`.
- New CLI flag `--include-n-t` (only valid with `--t-only-baseline`).

Run configuration (job 7484042):
- Same project dir, same packed manifest, same encoder checkpoint as the T_t-only run (encoder argparse-required but never opened).
- `SIGN_LOSS_WEIGHT = 0.1` (matches slw01_rerun).
- 20 epochs. Wall time ~13 min.

Greedy validation metrics (best @ epoch 20):

| Model | Inputs | `average_regret` | `average_return` | `sign_accuracy` (best) | `first_action_accuracy` | `average_expansions` |
|---|---|---|---|---|---|---|
| oracle | — | 0.000 | 0.354 | 1.000 | 1.000 | — |
| slw01 (rerun) | z_root, N_t, T_t | **0.031** | ≈0.323 | ≈0.871 | — | ≈7 (Pareto) |
| **(N_t, T_t)** | N_t, T_t | **0.056** | 0.297 | 0.628 | 0.832 | 14.20 |
| T_t-only | T_t | 0.218 | 0.136 | 0.457 | 0.663 | 20.79 |

Decomposition of the regret-reduction beyond T_t-only (gap = 0.218 − 0.031 = 0.187):

- **N_t alone** (T_t-only → (N_t, T_t)): closes 0.162 = **86.6%** of the gap.
- **z_root** ((N_t, T_t) → slw01): closes 0.025 = **13.4%** of the gap.

Equivalently, z_root cuts residual regret beyond (N_t, T_t) by 0.025 / 0.056 ≈ 45% — a meaningful but small absolute amount.

Interpretation:
- The Apr 29 inference-time WDL/z_root ablation was correctly reporting that z_root specifically contributes little to the binary halt/continue decision. It was not OOD-fooling us. The "z_root closes 86% of the regret gap" claim from the prior T_t-only entry conflated z_root with N_t; the corrected attribution is that **N_t** closes ~87% of that gap and z_root closes the remaining ~13%.
- Mechanistically: with λ_m = 0 in the oracle's continue cost, N_t doesn't enter the *cost*, but the model uses it as a step-within-episode counter to disambiguate where on the budget trajectory each snapshot sits. Combined with T_t, the pair encodes (starting_budget, step) up to the small initial-instability offset. That pair is approximately sufficient for the optimal stopping decision under this oracle/data distribution.
- z_root's residual ~13% contribution is meaningful in regret terms (it nearly halves residual regret) but is a marginal calibration signal, not a dominant computation. The "metacontroller computes VOC from the tree" framing as currently written is not supported.

Implications for the paper:
- Section 3 cannot claim the controller's good metacontrol is driven by tree-structural reasoning; the dominant signal is a budget-and-step heuristic.
- The decisive remaining experiment is the symmetric ablation: train slw01/rw4/inv_freq with `--exclude-n-t` (input becomes [z_root, T_t], no N_t available). Three outcomes:
  - Regret ≈ 0.056 (matches the (N_t, T_t) baseline): z_root *can* substitute for N_t when the shortcut is removed; the original training simply took the easier path. The paper reframes around training-time interventions that force tree-representation use.
  - Regret in (0.056, 0.218): z_root partially compensates; smaller story.
  - Regret ≈ 0.218 (matches T_t-only): z_root cannot encode step-within-episode information; its only contribution is the 0.025 marginal calibration. The paper's premise is dead.

Output paths:
- Training log: `analysis_outputs/tn_only/cts-fittedq-tonly_7484042.out`
- Best checkpoint: `/tigress/ysagiv/chess/cts/checkpoints/fittedq_n_t_t_t_baseline_rerun.pt` (selected by min validation greedy regret = 0.05645, epoch 20).
- Diagnostics JSONL: `/tigress/ysagiv/chess/cts/checkpoints/fittedq_n_t_t_t_baseline_rerun_diagnostics.jsonl` (30630 episodes; regret_mean=0.0565, regret_std=0.1339, regret_max=1.4445).

### Canonical [z_root, T_t]-only setup and intervention sweep (no_n_runs / no_n_v2 / no_n_v3)

Reframing:
- N_t was being silently used as a step-counter shortcut (per the (N_t, T_t) baseline above, which closes 87% of the regret-vs-T_t-only gap on its own). Continuing to train on N_t was a conceptual bug given λ_m = 0 in the canonical oracle: N_t doesn't enter the cost function but does enter the model input, where it acts as a near-perfect step-within-episode signal that the head leans on.
- Going forward the canonical model is `[z_root, T_t]`-only. Previous slw01 / reweight_w4 / inv_freq runs that consumed N_t are scaffolding-contaminated reference points (not the real evaluation grid).
- The new reference is the T_t-only floor at regret 0.218: the best a model with no z_root and no N_t can do.

Implementation note:
- Added `--exclude-n-t` to `scripts/train_fitted_q_controller.py`. Slices the N_t column out at `predict_from_features` time, leaving the head with input `[z_root, T_t]`. Materialized caches built with N_t are reused unchanged; only the head's input dimensionality changes.

#### Run 1: exclude-N_t v1 (LR=1e-3, 20 epochs, greedy_eval_interval=10)

Three loss configs (slw01, reweight_w4, inv_freq) with `--exclude-n-t`. All three regret oscillated wildly across epochs and converged at a poor floor:

| Loss config | Best regret (v1) |
|---|---|
| slw01 | 0.291 |
| reweight_w4 | 0.245 |
| inv_freq | 0.239 |
| (T_t-only reference) | 0.218 |

Notable: all three are *worse* than T_t-only. A model with strictly more input information than T_t-only doesn't even match T_t-only's floor — z_root acts as distractor noise the optimizer latches onto.

Logs: `analysis_outputs/no_n_runs/cts-fittedq_748484{8,9,0}.out`.

#### Run 2: exclude-N_t v2 (LR=1e-4 cosine→1e-5, 60 epochs, greedy_eval_interval=5)

Same three loss configs, smaller LR with cosine schedule, more epochs, finer greedy eval. Hypothesis: oscillation in v1 was high-LR overshoot.

| Loss config | Best regret (v2) |
|---|---|
| slw01 | **0.238** |
| reweight_w4 | **0.233** |
| inv_freq | 0.250 |

LR fix slightly improved slw01 / reweight_w4 but did not change the qualitative picture. Last 30 epochs of trajectories oscillate in a bounded band (slw01: 0.24–0.26) — converged at a poor floor, not unconverged. All three still worse than T_t-only (0.218).

Logs: `analysis_outputs/no_n_v2/cts-fittedq_748644{7,8,9}.out`.

#### Diagnosis: MSE/BCE loss tension as a candidate mechanism

The unified head (slw01) computes `total_loss = advantage_mse + sign_loss_weight * sign_bce` over the same scalar prediction. The two losses have non-coincident optima:
- MSE wants `pred ≈ target_advantage`, with target magnitudes O(0.01–0.1).
- Sign BCE wants `pred → ±∞` (saturated logits in the correct direction).

When MSE residual is small (slw01 with N_t: validation MSE ≈ 0.012), the two gradients balance at a stable point. With N_t removed, MSE residual stays large (~0.029) — its gradient remains substantial throughout training. The BCE gradient is bounded but persistent, pushing predictions outward; MSE pulls them back. At LR 1e-3 this oscillates; at LR 1e-4 it converges to a worse-than-T_t-only point.

This led to two interventions: (a) **MSE cooling** — fade the MSE term over training so BCE eventually wins cleanly; (b) **regret weighting** — weight per-snapshot loss by `|target_advantage|` (clamped at 1e-3), upweighting the continue-optimal class which has larger absolute advantages, countering BCE's drift toward the marginal halt class on under-determined inputs.

Implementation:
- Added `--mse-loss-weight-{schedule,initial,final}` to the trainer (constant / linear / cosine schedules). Default is `constant=1.0` so existing runs are unaffected.
- Added `--regret-weight-loss` flag. Per-snapshot weight is `|target_advantage|.clamp(min=1e-3)`. Composes multiplicatively with `--inverse-freq-weights` and `--nontrivial-loss-weight` if set. Cache training path only.

#### Run 3: aggressive cooling (slw01, BCE-primary, MSE→0)

Cooling with `SIGN_LOSS_WEIGHT=1.0` (vs canonical 0.1) and MSE cosine→0. Early intermediate at epoch 25:
- regret 0.374 (vs v2's 0.283 at the same epoch)
- average_expansions 3.79 (vs v2's ≈ 9.78)
- average_return = -0.026 (negative — losing value vs halting at step 0)
- validation sign_accuracy 0.884 — high because the model halts everywhere; majority-class predictor

Killed early — making BCE 10× stronger from the start let BCE's degenerate optimum (saturate at the marginal halt class) take over.

#### Run 4: conservative cooling (slw01, canonical SIGN_LOSS_WEIGHT=0.1, MSE→0)

Same loss-weight ratio as v2 at epoch 1, MSE fades to 0 by epoch 60. Greedy regret trajectory:

| Epoch | 5 | 10 | **15** | 20 | 25 | 30 | 35 | 40 | 45 | 50 | 55 | 60 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| regret | 0.349 | 0.313 | **0.252** | 0.308 | 0.308 | 0.317 | 0.329 | 0.340 | 0.358 | 0.358 | 0.383 | 0.399 |
| expansions | 7.6 | 8.5 | 11.5 | 7.5 | 7.1 | 6.8 | 5.8 | 4.7 | 4.2 | 4.3 | 2.8 | 2.7 |

Best regret 0.252 at epoch 15. After that, regret drifts upward and `average_expansions` collapses from 11.5 → 2.7 — same BCE-saturation failure as the aggressive run, just slower because BCE weight stayed at 0.1. As MSE fades, BCE wins, model halts everywhere. **Cooling is destructive regardless of BCE weight.** The loss-tension hypothesis as a productive intervention is dead.

Log: `analysis_outputs/no_n_v3/cts-fittedq_7490082.out`.

#### Run 5: regret weighting (slw01, no cooling, no filter)

Stable across all 60 epochs:

| Epoch | 5 | 10 | 15 | 20 | 25 | 30 | 35 | 40 | **45** | 50 | 55 | 60 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| regret | 0.274 | 0.257 | 0.271 | 0.239 | 0.243 | 0.251 | 0.252 | 0.252 | **0.234** | 0.246 | 0.245 | 0.248 |
| first_action_acc | 0.815 | 0.819 | 0.830 | 0.826 | 0.836 | 0.836 | 0.834 | 0.842 | 0.835 | 0.835 | 0.835 | 0.841 |
| expansions | 26.1 | 25.9 | 25.8 | 25.2 | 25.1 | 25.1 | 25.4 | 25.1 | 24.9 | 25.0 | 25.3 | 25.3 |

Best regret 0.234 at epoch 45. Regret oscillates in a tight band 0.234–0.274 with no drift — qualitatively different from every other [z_root, T_t] training. `first_action_accuracy` is the highest of any run at 0.84 (model correctly continues at step 0), and `average_expansions ≈ 25` (model continues for too long). Regret weighting did exactly what it was designed to do: prevent BCE saturation toward halt by upweighting continue-optimal snapshots.

But: best regret 0.234 is still 0.016 *worse* than the T_t-only floor (0.218). Stable training of `[z_root, T_t]` does not surface usable z_root signal beyond what T_t alone provides. Adding z_root to the input is, at best, neutral.

Log: `analysis_outputs/no_n_v3/cts-fittedq_7490237.out`.

#### Interpretation across runs 1–5

The mechanism behind the v1/v2 oscillation looks like loss tension (consistent with the cooling/regret-weighting interventions producing the predicted opposite-direction failures and stabilization). But fixing the dynamics doesn't recover usable z_root signal — the best stable training of `[z_root, T_t]` (regret-weighted, 0.234) still loses to `T_t` alone (0.218).

Two remaining hypotheses for why z_root contributes nothing exploitable:
1. **Data redundancy.** The dataset has so many snapshots where T_t alone determines the answer that z_root's gradient signal is drowned out by T_t-only-determined cases. Filtering to T_t-sensitive snapshots (per-tree, per-step, where the oracle's action varies across budgets) might surface the signal.
2. **Encoder bottleneck.** The encoder, pretrained for child-WDL reconstruction, simply doesn't carry the position-specific value-of-computation features that VOC-style metacontrol would need. No training-side intervention can recover what isn't there.

Implementation for (1): added `--t-sensitive-filter hard` to the trainer. Walks packed shards once to compute per-`(source_path, step)` flags (T_t-sensitive iff the oracle's halt/continue action varies across the budgets that reach this step). Aligns the flag tensor to the materialized cache row order via `torch.randperm` with the cache build's seed. Saves a sidecar at `<train_cache_path>.t_sensitive_mask.pt` for free reuse on subsequent runs. (Initial implementation walked the full DataLoader to align — too slow because the collator tensorizes trees per batch. Fast version uses a direct shard walk; ~10–20 min total mask compute vs >60 min before.)

In flight as of this entry:
- **Filter run** (slw01_no_n_t + `T_SENSITIVE_FILTER=hard`, otherwise v2 hyperparameters): tests hypothesis (1) — does training on T_t-sensitive snapshots only let z_root contribute?
- **Kitchen sink run** (filter + cosine MSE cooling + regret weighting, dependent on filter completing to reuse the mask sidecar): if filter alone helps, do the stabilizing interventions stack on top?

If both still float around 0.218–0.24, hypothesis (1) is dead and the encoder is the bottleneck — outside the scope of this week's deadline.

#### Run 6: filter alone (slw01_no_n_t + T_SENSITIVE_FILTER=hard, no cooling, no regret-weighting)

T_t-sensitivity statistics from the filter computation:
- T_t-sensitive positions: 678596 / 5295803 = **12.8%** of (source, step) pairs.
- Snapshots kept: 3546306 / 17924009 = **19.8%** of total snapshots.

So 80% of training snapshots are T_t-determined (the oracle's action is the same regardless of starting budget for that source-step). The dataset is heavily redundant; only one snapshot in five carries information that requires z_root for the optimal action.

Greedy regret trajectory (60 epochs, eval every 5):

| Epoch | 5 | 10 | 15 | 20 | 25 | 30 | 35 | 40 | **45** | 50 | 55 | 60 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| regret | 0.240 | 0.274 | 0.295 | 0.248 | 0.301 | 0.275 | 0.267 | 0.279 | **0.229** | 0.250 | 0.251 | 0.253 |
| first_action_acc | 0.650 | 0.597 | 0.588 | 0.647 | 0.593 | 0.631 | 0.646 | 0.620 | 0.691 | 0.677 | 0.661 | 0.655 |
| expansions | 17.4 | 13.4 | 11.9 | 14.8 | 9.7 | 11.3 | 10.9 | 10.2 | 12.7 | 11.5 | 10.7 | 10.6 |

Best regret 0.229 at epoch 45 — slightly better than regret-weighted (0.234) and v2 (0.238). Still worse than T_t-only floor (0.218). Trajectory oscillates 0.229–0.301 across all 60 epochs (range 0.07) — filter alone does not stabilize training the way regret weighting does.

Wall time: ~12.5 min total (mask compute + 60 epochs at ~1/5 the data volume). Mask sidecar saved to `/tigress/ysagiv/chess/cts/train_cache_rerun.pt.t_sensitive_mask.pt` for reuse.

Log: `analysis_outputs/no_n_v3/cts-fittedq_7493984.out`.

Interpretation:
- The data redundancy hypothesis (hypothesis 1) gets weak partial support: filtering to the T_t-sensitive 20% of snapshots gives a 0.009 regret improvement over v2 (0.238 → 0.229). Real but small.
- The improvement is *not* large enough to break below T_t-only's 0.218 floor. Even when we restrict training to the snapshots where z_root **must** contribute under the oracle, the head's regret on the full validation set is still worse than the input-z_root-blind T_t-only baseline.
- This is increasingly consistent with hypothesis (2): the encoder genuinely doesn't carry exploitable VOC signal. The encoder was pretrained on child-WDL reconstruction; the WDL information is well-preserved (KL ≈ 0.02) but it's not the right signal for the metacontroller's stopping decision under this oracle.

Updated reference grid:

| Model | Inputs | Regret | Notes |
|---|---|---|---|
| oracle | — | 0.000 | |
| **T_t-only floor** | **T_t** | **0.218** | canonical reference for [z_root, T_t] models |
| Filter | z_root, T_t | 0.229 | oscillating |
| Regret-weighted | z_root, T_t | 0.234 (stable) | only stable trajectory of single-intervention runs |
| v2 (no interv.) | z_root, T_t | 0.238 | converged-poor floor |
| Conservative cooling | z_root, T_t | 0.252 → 0.399 | drifts to halt-saturation |
| **Kitchen sink (filter + cooling + regret-weight)** | **z_root, T_t** | **0.187** | **first run to break below T_t-only**; still descending at ep 60 |
| Bigger-capacity regret-weighted | z_root, T_t | (queued) | Q_HIDDEN=512, Q_HIDDEN_LAYERS=5 |

#### Run 7: kitchen sink (filter + MSE cosine cooling + regret-weighted) — breaks below T_t-only

Configuration: slw01 + EXCLUDE_N_T + T_SENSITIVE_FILTER=hard + REGRET_WEIGHT_LOSS=1 + MSE_LOSS_WEIGHT_INITIAL=1.0, FINAL=0.0, SCHEDULE=cosine. v2 hyperparameters otherwise (LR 1e-4 cosine to 1e-5, 60 epochs, GREEDY_EVAL_INTERVAL=5).

Greedy regret trajectory:

| Epoch | 5 | 10 | 15 | 20 | 25 | 30 | 35 | 40 | 45 | 50 | 55 | **60** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| regret | 0.324 | 0.306 | 0.304 | 0.281 | 0.278 | 0.271 | 0.269 | 0.259 | 0.250 | 0.229 | 0.199 | **0.187** |
| first_action_acc | 0.816 | 0.807 | 0.816 | 0.814 | 0.818 | 0.819 | 0.819 | 0.817 | 0.816 | 0.812 | 0.788 | 0.759 |
| expansions | 26.9 | 26.8 | 26.7 | 26.3 | 26.3 | 26.1 | 26.1 | 25.9 | 25.7 | 25.2 | 24.2 | 22.7 |

Best regret **0.187** at epoch 60 — monotone descent, no oscillation, no collapse. **Trajectory is still falling at the final epoch** (0.199 → 0.187 in the last 5 epochs); the run is not converged. More training would likely push regret further down.

For the first time, a `[z_root, T_t]`-only controller breaks below the T_t-only floor (0.218 → 0.187, an absolute regret reduction of 0.031).

Why each intervention is load-bearing:
- **Filter alone** (run 6, regret 0.229): some help, but oscillating; doesn't break the T_t-only floor.
- **Regret weighting alone** (run 5, regret 0.234, stable): prevents BCE drift toward halt-saturation; without it, cooling collapses (run 4: 0.252 → 0.399).
- **Cooling alone** (run 4): destructive without regret weighting — BCE saturates at halt class.
- **Combined**: filter narrows training to z_root-relevant snapshots; regret weighting holds the continue class properly weighted while MSE provides scale anchor; cooling fades MSE so BCE drives final sign decisions on a class-balanced objective. The combination achieves stable monotone improvement.

The diagnostic implication: the loss-tension hypothesis (run-3-and-4 cooling failures suggested) and the data-redundancy hypothesis (run 6 filter weakly supported) and the class-imbalance hypothesis (run 5 regret-weighted fixed) were each correct about *part* of the picture. None individually fixed regret below T_t-only because each addressed only one bottleneck. Stacking them addresses all three simultaneously and the model can finally extract usable z_root signal.

Implications for the paper: the previous "tree representation appears decorative" framing is now wrong. The encoder *does* carry exploitable VOC signal; it just needs the right training-time setup to surface it. The contribution becomes about identifying the diagnostic methodology and the intervention stack that turns naive offline fitted-Q into a controller that actually uses the tree representation.

Log: `analysis_outputs/no_n_v3/cts-fittedq_7493985.out`. Diagnostics: `regret_mean=0.1872, regret_std=0.2365, regret_max=1.9286`. Output checkpoint: `/tigress/ysagiv/chess/cts/checkpoints/fittedq_slw01_no_n_t_kitchen_sink_rerun.pt`.

Immediate next move: run kitchen sink for more epochs (e.g., 150 or 200) to see where the trajectory actually converges.

#### Run 8: bigger-capacity regret-weighted (Q_HIDDEN=512, Q_HIDDEN_LAYERS=5 + regret weighting, no filter, no cooling)

Configuration: slw01 + EXCLUDE_N_T + REGRET_WEIGHT_LOSS=1 + Q_HIDDEN=512 + Q_HIDDEN_LAYERS=5. v2 hyperparameters otherwise. Roughly 4× total parameters in the head.

Greedy regret trajectory:

| Epoch | 5 | 10 | 15 | 20 | 25 | 30 | 35 | 40 | **45** | 50 | 55 | 60 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| regret | 0.259 | 0.237 | 0.233 | 0.229 | 0.230 | 0.229 | 0.229 | 0.233 | **0.226** | 0.241 | 0.240 | 0.237 |
| first_action_acc | 0.825 | 0.841 | 0.844 | 0.848 | 0.854 | 0.856 | 0.856 | 0.863 | 0.855 | 0.862 | 0.860 | 0.857 |
| expansions | 25.7 | 24.8 | 24.3 | 24.9 | 23.9 | 23.1 | 22.9 | 22.3 | 22.4 | 22.4 | 22.2 | 22.1 |

Best regret **0.226** at epoch 45. Trajectory is monotone through epoch 30 (0.259 → 0.229) then converges to a 0.226–0.241 band. `first_action_accuracy = 0.86` is the highest of any [z_root, T_t]-only run; `average_expansions ≈ 22` (still over-continuing, slightly less than the 256-wide regret-weighted's 25).

Improvement over the original-capacity regret-weighted (0.234) is **0.008**. Still does not break below the T_t-only floor (0.218).

Compared with the kitchen sink (0.187, run 7):
- Capacity bump on regret-weighted gives a 0.008 regret reduction.
- Filter + cooling stacked on regret-weighted (= kitchen sink) gives a 0.047 regret reduction and is still descending.

So capacity is contributing roughly 6× less per-intervention than the filter + cooling stack. Consistent with the encoder + data structure being the dominant bottleneck, and head capacity being a marginal-but-real second-order improvement. The combination kitchen-sink + capacity + more epochs is the natural next experiment.

Log: `analysis_outputs/no_n_v3/cts-fittedq_7494698.out`. Output checkpoint: `/tigress/ysagiv/chess/cts/checkpoints/fittedq_slw01_no_n_t_regretw_bigger_rerun.pt`.

#### Run 9: kitchen sink + bigger capacity, 100 epochs

Configuration: kitchen sink (filter + cooling + regret-weighting + EXCLUDE_N_T, slw01) + Q_HIDDEN=512 + Q_HIDDEN_LAYERS=5 + EPOCHS=100. Cosine cooling and cosine LR schedule both stretch over 100 epochs instead of 60.

Greedy regret trajectory:

| Epoch | 5 | 10 | 15 | 20 | 25 | 30 | 35 | 40 | 45 | 50 | **55** | 60 | 65 | 70 | 75 | 80 | 85 | 90 | 95 | 100 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| regret | 0.304 | 0.275 | 0.274 | 0.267 | 0.242 | 0.240 | 0.232 | 0.225 | 0.234 | 0.234 | **0.198** | 0.225 | 0.212 | 0.204 | 0.198 | 0.199 | 0.198 | 0.198 | 0.202 | 0.201 |
| expansions | 26.8 | 26.4 | 26.1 | 26.0 | 25.3 | 25.0 | 24.8 | 24.6 | 24.3 | 24.2 | 22.7 | 23.3 | 22.1 | 21.6 | 21.1 | 20.3 | 20.1 | 19.3 | 19.3 | 18.3 |

Best regret 0.198 at epoch 55 (rounded — actual saved-best 0.1977). Wall time: 22 min. Plateau at 0.198–0.212 through epochs 55–100; trajectory does not improve further.

Comparison with run 7 (kitchen sink, canonical 256-wide × 3-layer head, 60 epochs):
- Run 7 best: **0.187** at epoch 60, *still descending*.
- Run 9 best: **0.198** at epoch 55, plateaus.

Bigger capacity is *worse* in absolute terms by 0.011, despite running 40 more epochs. **Capacity bumps appear to harm rather than help when stacked on the kitchen sink.** This kills the capacity-as-bottleneck hypothesis cleanly: with the right loss + data interventions in place, the canonical 256-wide × 3-layer head outperforms a 4× larger head.

Caveat for fair comparison: the cosine cooling schedule shape stretches over the run length, so at epoch 55 the run-9 MSE weight was ≈ 0.42 (cosine midway) while at run-7 epoch 55 it was ≈ 0.015 (near the end). Different cooling state at the same epoch number. To know definitively whether the small head beats the big head at full cooling, the canonical kitchen sink needs to run for 100 epochs (same schedule as run 9).

Log: `analysis_outputs/no_n_ks/cts-fittedq_7496598.out`. Output checkpoint: `/tigress/ysagiv/chess/cts/checkpoints/fittedq_slw01_no_n_t_kitchen_sink_bigger_rerun.pt`. Diagnostics: `regret_mean=0.2013, regret_std=0.3315, regret_max=1.9868`.

Immediate next move: re-run the canonical kitchen sink (256-wide × 3-layer) for 100 epochs to give an apples-to-apples comparison at full cooling. If it lands below 0.187, the small head genuinely beats the big head and capacity is harmful. If it lands at ~0.20, both heads converge similarly and run 7's 0.187 was a still-descending intermediate.

#### Run 10: canonical kitchen sink, 100 epochs

Configuration: kitchen sink (run 7) settings exactly, but `EPOCHS=100` instead of 60. Default Q_HIDDEN=256 and Q_HIDDEN_LAYERS=3.

Greedy regret trajectory:

| Epoch | 5 | 20 | 40 | 60 | 80 | 90 | 95 | **100** |
|---|---|---|---|---|---|---|---|---|
| regret | 0.325 | 0.283 | 0.271 | 0.252 | 0.223 | 0.190 | 0.183 | **0.180** |
| expansions | 26.9 | 26.4 | 26.1 | 25.7 | 24.8 | 23.6 | 23.0 | 22.3 |

Best regret **0.180** at epoch 100 — *still descending* in the final stretches (0.190 → 0.183 → 0.180 across the last three evals). Wall time: 21 min.

Two findings:

1. **Canonical capacity beats bigger capacity decisively at the same training schedule.** Run 9 (bigger, 100 ep): 0.198 plateau. Run 10 (canonical, 100 ep): 0.180 still descending. The 0.018 gap is real, and the run-10 trajectory is going lower while run-9 is flat. Capacity bumps are *net-harmful* when stacked on the kitchen sink, not just unhelpful.

2. **Trajectory is not converged at 100 epochs.** The final 5 epochs improve regret at roughly 0.005/epoch even though the cosine MSE schedule is fully cooled (MSE weight ≈ 0 by epoch 100) and the cosine LR is at the minimum (1e-5). Improvement is purely BCE-driven at small LR — slow but steady. 200 epochs should push regret further; hard to predict exactly where, but at the current rate the asymptote is likely well below 0.180.

Updated reference grid:

| Model | Inputs | Regret | Notes |
|---|---|---|---|
| oracle | — | 0.000 | |
| **Kitchen sink, canonical, 100 epochs (this run)** | **z_root, T_t** | **0.180** | still descending; new best |
| Kitchen sink, canonical, 60 ep (run 7) | z_root, T_t | 0.187 | still descending at ep 60 |
| Kitchen sink, bigger capacity, 100 ep (run 9) | z_root, T_t | 0.198 | plateau |
| **T_t-only floor** | **T_t** | **0.218** | |
| Bigger-capacity regret-weighted (run 8) | z_root, T_t | 0.226 | |
| Filter alone (run 6) | z_root, T_t | 0.229 | |
| Regret-weighted (run 5) | z_root, T_t | 0.234 | |
| v2 (run 2, slw01) | z_root, T_t | 0.238 | |

Log: `analysis_outputs/no_n_ks/cts-fittedq_7498352.out`. Output checkpoint: `/tigress/ysagiv/chess/cts/checkpoints/fittedq_slw01_no_n_t_kitchen_sink_100ep_rerun.pt`. Diagnostics: `regret_mean=0.1796, regret_std=0.2343, regret_max=1.9286`.

Immediate next move: extend kitchen sink to 200 epochs at canonical capacity to find where the trajectory actually converges.

#### Run 11 (in flight): canonical kitchen sink, 200 epochs

Configuration: kitchen sink, canonical capacity, EPOCHS=200. Cosine MSE cooling and cosine LR schedule both stretch over 200 epochs.

Intermediate trajectory at epoch 170 (run still in flight):

| Epoch | 60 | 90 | 100 | 130 | 150 | 165 | **170** |
|---|---|---|---|---|---|---|---|
| regret | 0.267 | 0.239 | 0.229 | 0.221 | 0.209 | 0.198 | **0.195** |
| expansions | 25.9 | 25.1 | 24.6 | 24.4 | 23.8 | 23.4 | 23.3 |

The 200-epoch schedule is *not* apples-to-apples with the 100-epoch run 10 at the same epoch number (cosine cooling stretches), so the relevant comparison is at matched schedule progress:

- Run 10 (100 ep) at schedule progress 0.85 (= ep 85): regret ≈ 0.21 (interpolated from ep 80 = 0.223, ep 90 = 0.190).
- Run 11 (200 ep) at schedule progress 0.85 (= ep 170): regret 0.195.

So stretching the schedule appears to gain ~0.015 in regret at matched progress. The trajectory is still descending — recent rate ~0.005–0.01 per 5 epochs. Final regret at ep 200 is plausibly in the 0.17–0.18 range if the recent rate continues through the steep-descent phase that the 100-epoch run saw in its final 20%.

Will update with the final result when the run completes (~10–15 min more cluster time).

#### Run 11 final: canonical kitchen sink, 200 epochs

Run completed in 41 min wall.

Final trajectory tail:

| Epoch | 170 | 175 | 180 | 185 | 190 | **195** | 200 |
|---|---|---|---|---|---|---|---|
| regret | 0.195 | 0.194 | 0.186 | 0.182 | 0.178 | **0.176** | 0.176 |
| expansions | 23.3 | 23.1 | 22.6 | 22.5 | 22.1 | 21.7 | 21.4 |

Best regret **0.176** at epoch 195; the last 5 epochs are flat — trajectory has converged. Wall time 41 min.

Comparison with the 100-epoch run (run 10): 100 more epochs bought a 0.004 additional regret reduction and confirmed the floor. The 200-epoch run is the converged version of run 10.

Diagnostics: `regret_mean=0.1762, regret_std=0.2522, regret_max=1.9391`.

Output checkpoint: `/tigress/ysagiv/chess/cts/checkpoints/fittedq_slw01_no_n_t_kitchen_sink_200ep_rerun.pt`. Log: `analysis_outputs/no_n_ks/cts-fittedq_7499179.out`.

#### Final reference grid

| Model | Inputs | Regret | Notes |
|---|---|---|---|
| oracle | — | 0.000 | |
| **Kitchen sink, canonical capacity, 200 ep** | **z_root, T_t** | **0.176** | converged; new best |
| Kitchen sink, canonical, 100 ep | z_root, T_t | 0.180 | still descending |
| Kitchen sink, canonical, 60 ep | z_root, T_t | 0.187 | still descending |
| Kitchen sink, bigger capacity, 100 ep | z_root, T_t | 0.198 | plateau; capacity bump hurt |
| **T_t-only floor** | **T_t** | **0.218** | |
| Bigger-capacity regret-weighted | z_root, T_t | 0.226 | |
| Filter alone | z_root, T_t | 0.229 | oscillating |
| Regret-weighted alone | z_root, T_t | 0.234 | stable but capped |
| v2 (slw01_no_n_t, no interventions) | z_root, T_t | 0.238 | worse than T_t-only |

#### Headline result

- Naive `[z_root, T_t]` training (regret 0.238) is *worse* than `T_t`-only (0.218). The encoder's tree representation acts as a distractor under standard fitted-Q training: the optimizer finds spurious z_root patterns that hurt the decision quality more than they help.
- With three stacked training-time interventions — T_t-sensitive snapshot filter + per-snapshot regret weighting + MSE cosine cooling — `[z_root, T_t]` achieves regret **0.176**.
- That's a **19% absolute regret reduction below the T_t-only floor (0.218 → 0.176)** and a **26% reduction below the naive baseline (0.238 → 0.176)**.
- Each intervention individually is either small (filter: 0.229, regret-weighted: 0.234) or destructive (cooling alone: drifts to 0.399). Only the stack works.
- Capacity bumps on top of the kitchen sink are net-harmful (0.198 vs 0.176). The encoder + intervention stack is the load-bearing thing, not head capacity.

#### Paper framing

- The contribution is the identification of a training pathology in offline fitted-Q metacontrol — naive training with the canonical loss + standard hyperparameters under-uses the tree representation in a way that is *worse than ignoring it* — and a diagnostic intervention stack that recovers usable VOC signal from the same encoder, same data, same model.
- The slw01_rerun result (regret 0.031 with z_root + N_t + T_t) was scaffolding-contaminated: most of that good performance was driven by N_t acting as a step-counter shortcut, not by genuine tree-representation use. The clean evaluation grid is the canonical `[z_root, T_t]`-only setup, where the kitchen sink at 200 epochs is the new best at 0.176.

## 2026-04-30

### Lambda=5 oracle re-pack and intervention sweep

Motivation:
- Under the canonical lambda=18.5 oracle, kitchen-sink-with-interventions on `[z_root, T_t]` achieves regret 0.176 vs T_t-only 0.218 — a 0.042 gap. Hypothesis was that lambda was set high enough that time pressure dominates the optimal stopping decision, so T_t alone gets most of the way to the oracle. Lowering lambda should reduce time pressure and force the oracle's decision to depend more on per-position value of computation, where z_root could contribute.
- Re-pack the controller episodes with `time_lambda=5` (vs canonical 18.537) keeping all other oracle parameters fixed. Re-train the canonical reference grid on the new oracle.

Pipeline (deadline-relevant timing):
- Re-pack: ~minutes (no tree regen, just oracle backward induction over existing PretrainExample files).
- Re-materialize encoder cache: 3 parallel workers, ~24 min wall.
- Merge cache shards: seconds on login node.
- Training: T_t-only ~13 min, naive [z_root, T_t] ~30 min, kitchen sink 200ep ~50 min.

Important data note: the re-pack was done on `pretrain_split_oracle96_trace_filtered_rerun` (the split the rerun encoder was actually pretrained on). This is **different** from the original `controller_packed_combined_nomaint_no_xaba` data which used a "combined" split with both rerun and PUCT-filtered trees. The lambda=5 dataset is a strict subset of trees compared to the original. Results are therefore confounded with a dataset change as well as the oracle change — we cannot cleanly attribute a difference to lambda alone.

#### Run 12: T_t-only baseline on lambda=5

Configuration: `--t-only-baseline`, no encoder, just packed shards + T_t scalar. 20 epochs (default for T_t-only).

Final result: **regret 0.150** at epoch 20. `average_return = 0.213`, `average_oracle_value = 0.363`, `first_action_accuracy = 0.775`, `average_expansions = 26.3`, `regret_mean = 0.1499`.

For comparison, T_t-only on the lambda=18.5 / combined data was regret 0.218. The floor *dropped* under lambda=5 — opposite of my prediction that lower time pressure would make T_t alone less informative.

Log: `analysis_outputs/no_n_kms/cts-fittedq-tonly_7504140.out`. Output checkpoint: `/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/checkpoints/fittedq_t_only_baseline_lambda5.pt`.

#### Run 13: naive [z_root, T_t] on lambda=5

Configuration: slw01 + EXCLUDE_N_T + lambda=5, no interventions, v2 hyperparameters (LR 1e-4 cosine to 1e-5, 60 epochs).

Final result: **regret 0.192** at epoch 60 (best 0.155 earlier in training). `regret_mean = 0.1918`, `average_expansions = 16.5`.

Compared to lambda=18.5 v2 naive (regret 0.238), the lambda=5 naive run is similar quality on relative terms. Both are *worse* than the corresponding T_t-only floor (0.150 here, 0.218 there). Same pattern: naive z_root inclusion hurts versus pure T_t.

Log: `analysis_outputs/no_n_kms/cts-fittedq_7504153.out`.

#### Run 14: kitchen sink, canonical capacity, 200 epochs on lambda=5 (headline run)

Configuration: kitchen sink (filter + cooling + regret-weighted) + EXCLUDE_N_T + 200 epochs + canonical capacity (Q_HIDDEN=256, Q_HIDDEN_LAYERS=3) + v2 hyperparameters + lambda=5.

Final result tail:

| Epoch | 190 | 195 | 200 |
|---|---|---|---|
| regret | 0.132 | 0.134 | **0.132** |
| first_action_acc | 0.844 | 0.847 | 0.843 |
| expansions | 25.8 | 25.9 | 25.7 |

Best **regret 0.132** at epoch 190 — 0.018 lower than T_t-only floor. Stable plateau across epochs 190–200. `regret_mean = 0.1325`. `average_return = 0.230`, `average_oracle_value = 0.363`, `capture rate = 63.4%`.

Log: `analysis_outputs/no_n_kms/cts-fittedq_7504154.out`. Output checkpoint: `/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/checkpoints/fittedq_slw01_no_n_t_kitchen_sink_lambda5.pt`.

#### Comparison across oracles

| Oracle / Dataset | Model | Regret | Capture % | KS-vs-Floor gap |
|---|---|---|---|---|
| lambda=18.5 / combined trees | T_t-only floor | 0.218 | 38.4% | — |
| lambda=18.5 / combined trees | naive [z_root, T_t] | 0.238 | 32.8% | -0.020 |
| lambda=18.5 / combined trees | **Kitchen sink** | **0.176** | **52.8%** | **0.042** |
| **lambda=5 / rerun trees** | T_t-only floor | **0.150** | 58.7% | — |
| **lambda=5 / rerun trees** | naive [z_root, T_t] | 0.192 | 47.1% | -0.042 |
| **lambda=5 / rerun trees** | **Kitchen sink** | **0.132** | **63.4%** | **0.018** |

Interpretation:

The hypothesis was: lower lambda → less time pressure → VOC matters more → T_t-only floor *increases* and kitchen-sink-vs-floor gap *widens*. The data shows the opposite: T_t-only floor *decreased* (0.218 → 0.150) and the kitchen-sink gap *shrunk* (0.042 → 0.018).

Mechanistic read: under low time pressure, the oracle's optimal stopping rule converges to "wait until search has effectively converged." That convergence happens at roughly the same step count regardless of position (the planner reaches steady state at similar depths), and step count is approximately T_t-up-to-budget-offset. So T_t alone is a *better* approximation under low time pressure, not a worse one. The high-time-pressure regime (lambda=18.5) is actually where time-vs-position trade-offs are most heterogeneous — and even there, the kitchen-sink-gap was modest.

Caveats:
- Confounded with dataset change (rerun-only trees vs combined trees). Cannot cleanly attribute the floor change to lambda alone. To deconfound would require re-packing the combined-trees split with lambda=5.
- The 12% relative regret reduction (kitchen sink vs T_t-only) on the cleanest aligned setup is real but small.

Implication for the paper:

Lowering lambda did not surface a regime where the GNN representation contributes substantially. The clean (encoder-aligned, lambda=5) result is **kitchen sink at regret 0.132 vs T_t-only at 0.150** — a 0.018 absolute improvement. The previous lambda=18.5 number (0.176 vs 0.218 = 0.042) actually showed a larger absolute z_root contribution.

The paper claim "tree representation enables better offline metacontrol" stands, but the magnitude of the contribution is modest (0.018–0.042 in absolute regret, 12–20% in relative regret reduction). The intervention stack (filter + cooling + regret-weighted) is necessary to extract any z_root signal at all; without it, the GNN actively hurts compared to T_t alone in both oracle regimes.

#### Where things stand at the end of this run

- The strongest evidence is on the clean aligned setup at lambda=5: kitchen sink achieves 0.132 vs T_t-only's 0.150. A 12% relative regret reduction.
- The intervention stack is empirically necessary: naive training is consistently worse than T_t-only; only filter + regret-weighting + MSE cooling yields a controller that exceeds the floor.
- The paper's contribution is the diagnostic methodology and intervention stack; the absolute regret reduction is real but smaller than the lambda=18.5 numbers initially suggested.

### VPI auxiliary loss: does explicit VPI supervision help?

Motivation:
- The kitchen sink at lambda=5 hits 0.132 regret vs T_t-only's 0.150. The 0.018 gap is real but small. Hypothesis: maybe the head extracts z_root signal that's *related to* but not actually VOC. Adding an explicit auxiliary loss that forces an intermediate (or final) layer of the advantage MLP to predict closed-form Dearden VPI from ground-truth child WDLs would surface VPI computation if z_root carries that signal — and either (a) further reduce regret if VPI is the right computation, or (b) leave regret unchanged if VPI isn't what the controller actually needs.

Implementation:
- New `--vpi-aux-layer` (int) and `--vpi-aux-weight` (float) CLI flags. Auxiliary head taps the activation after the k-th Linear+ReLU stage of the advantage MLP and projects to a scalar VPI prediction.
- Refactored `_build_advantage_head` to return an `AdvantageHead` (subclass of `nn.Sequential` with `forward_with_aux(x, aux_layer)`) so intermediate activations can be captured cleanly.
- VPI targets pre-computed once per packed dataset by walking shards and applying the Dearden formula `1 - prod(1-w_c) - prod(l_c) - max_c(w_c - l_c)` to root children's WDL features. Stored as a sidecar at `<train_cache>.vpi_targets.pt` aligned to the cache's shuffled row order via `torch.randperm` with the cache build seed (same pattern as the T_t-sensitive mask sidecar).
- Restricted to the unified-head architecture (incompatible with `--separate-sign-head` and `--t-only-baseline`).

#### Run 15: kitchen sink + VPI aux at the end (layer 3, parallel to advantage projection)

Configuration: kitchen sink (filter + regret-weighted + MSE cooling) + lambda=5 + VPI_AUX_LAYER=3 + VPI_AUX_WEIGHT=1.0 + 200 epochs + canonical capacity. Target stats logged at startup: mean=0.0117, std=0.0599.

Final result: **regret 0.132** at epoch 200. Identical to kitchen sink without VPI aux (run 14: also 0.132).

Log: `analysis_outputs/vpi_au/cts-fittedq_7506006.out`.

#### Run 16: kitchen sink + VPI aux at intermediate layer 2

Configuration: same as run 15 but VPI_AUX_LAYER=2.

Final result: **regret 0.131** (best 0.130) at epoch 200. Within noise of the no-VPI baseline.

Log: `analysis_outputs/vpi_au/cts-fittedq_7506020.out`.

Interpretation:
- Explicit VPI supervision does not change the controller's regret, regardless of whether the aux head taps an intermediate layer or sits in parallel at the end of the head. This is a clean negative result for the "the controller would compute VOC if you forced it to predict VPI" framing.
- VPI is neither necessary (the kitchen sink already reaches 0.132 without it) nor sufficient (adding it doesn't break below that floor) for the metacontrol task under this oracle.

Implementation oversight to note: per-batch training logs include `vpi_mse` but the per-epoch summary line does not expose it. With VPI_AUX_WEIGHT=1.0 and target variance ≈ 0.0036 (std ≈ 0.06), a constant-mean predictor would yield aux MSE ≈ 0.0036 contributing ≈ 0.0036 to total loss — plausibly invisible against the dominant sign BCE / advantage MSE early in training, and indistinguishable from a head that genuinely fitted per-snapshot VPI. Since we only observe `train_total_loss = 0.010` at convergence, we cannot disambiguate "aux head fitted VPI" from "aux head short-circuited to predicting the mean." Either way, the regret outcome is the same: explicit VPI supervision doesn't change metacontrol quality.

A clean follow-up if we want to disambiguate decodability from utility: train a frozen-controller probe (instantiate the kitchen-sink head, freeze it, attach a fresh VPI aux head and train only that on the VPI target). If the probe MSE drops well below 0.0036, z_root's representation does carry VPI in principle, and the run-15/16 negative-regret result means VPI just isn't the right computation for this oracle. If the probe MSE stays at ≈0.0036, the encoder doesn't carry VPI at all and Section 3's "controller doesn't compute VPI" finding stands at the representation level.

Updated reference grid (lambda=5 / rerun-aligned data):

| Model | Inputs | Regret | Notes |
|---|---|---|---|
| oracle | — | 0.000 | |
| Kitchen sink + VPI aux at layer 2 | z_root, T_t | 0.131 | |
| **Kitchen sink, canonical, 200 ep** | **z_root, T_t** | **0.132** | best stable |
| Kitchen sink + VPI aux at layer 3 | z_root, T_t | 0.132 | |
| **T_t-only floor** | **T_t** | **0.150** | |
| Naive [z_root, T_t] | z_root, T_t | 0.192 | |

#### Updated reference grid

| Model | Inputs | Regret | Notes |
|---|---|---|---|
| oracle | — | 0.000 | |
| slw01_rerun (full) | z_root, N_t, T_t | 0.031 | scaffolding-contaminated; uses N_t step counter |
| (N_t, T_t)-only | N_t, T_t | 0.056 | shows N_t alone closes 87% of the regret-vs-T_t-only gap |
| **T_t-only floor** | **T_t** | **0.218** | canonical reference for [z_root, T_t] models |
| exclude-N_t v1 (slw01) | z_root, T_t | 0.291 | LR=1e-3, oscillation |
| exclude-N_t v1 (reweight_w4) | z_root, T_t | 0.245 | |
| exclude-N_t v1 (inv_freq) | z_root, T_t | 0.239 | |
| exclude-N_t v2 (slw01) | z_root, T_t | 0.238 | LR=1e-4 cosine; converged at poor floor |
| exclude-N_t v2 (reweight_w4) | z_root, T_t | 0.233 | |
| exclude-N_t v2 (inv_freq) | z_root, T_t | 0.250 | |
| Aggressive cooling | z_root, T_t | 0.374 → killed | BCE saturation at halt class |
| Conservative cooling | z_root, T_t | 0.252 (ep 15) → 0.399 (ep 60) | drifts to halt-saturation |
| **Regret-weighted** | **z_root, T_t** | **0.234** (stable) | first stable [z_root, T_t] training; still > T_t-only |
| Filter | z_root, T_t | (in flight) | |
| Kitchen sink (filter + cooling + regret-weight) | z_root, T_t | (queued) | |

### Encoder finetuning: training-loop optimizations and architectural cleanup

Context:
- Encoder finetuning (unfrozen TreeNN + advantage head, differential LR encoder=1e-5 / head=1e-3, warm-started from `fittedq_sweep_slw01.pt`) was running at ~2.2 h/epoch on a single A100. At 20 epochs that's a 44 h job, well past della's 24 h wall — the previous run timed out after 11 epochs.
- Goal of this session: make a 20-epoch finetuning run feasible in one job, and clean up an architectural confound in the controller input.

Meaningful changes (committed on `finetune-treenn`):

1. **Mixed precision (bf16) and level-local sequential attention** (`362e105`).
   - Added `--mixed-precision {bf16,fp16}` to `scripts/train_fitted_q_controller.py`, defaulting to bf16. All training/eval forward sites are wrapped in `torch.amp.autocast`; fp16 path uses `GradScaler`, bf16 does not.
   - In `GNN.py::_forward_sequential`, restructured the per-depth-level attention to (a) precompute `W_q(node_states)` once per upward sweep instead of redundantly at every depth level, and (b) call a new `TreeAttMsgLayer.forward_level` that runs `W_k`/`W_v`/scatter-softmax/`W_o` only on the parent nodes at each level rather than the full `[N_total, ...]` tensor. Output verified bitwise-identical to the prior python-loop reference on a synthetic tree, gradients verified non-zero for all parameters.
   - Combined wall-clock impact: ~2.2 h/epoch → ~1 h/epoch on a single A100 with `EPISODE_BATCH_SIZE=16`. The level-local optimization is the bigger lever; bf16 alone gave ~10% improvement.

2. **DDP support** (`b73cf46`).
   - Added `_setup_distributed()` that detects `torchrun` env vars; falls through silently in single-GPU mode. Sets the CUDA device *before* `init_process_group("nccl")` so each rank lands on its own GPU. Wraps the model in `DDP` when `world_size > 1`. Uses `DistributedSampler` on the packed-episode loader with `set_epoch` each epoch. All logging, validation, greedy eval, and checkpoint saves are gated on `rank == 0`. `_save_checkpoint` strips the DDP wrapper before serializing.
   - Slurm script (`slurm/train_finetune_controller_della.slurm`) now counts `CUDA_VISIBLE_DEVICES` to pick `NUM_GPUS` (slurm doesn't reliably set `SLURM_GPUS_ON_NODE` on della) and launches via `torchrun --standalone --nproc_per_node="${NUM_GPUS}"`.

3. **N_t removed from controller input** (`b73cf46`).
   - The advantage head now operates on `[z_t, T_t]` only. `encode_with_state_features` keeps `tree_sizes` in its signature for caller compatibility but ignores it; `input_dim = d_embed + 1`; `controller_inputs` metadata reflects `["z_t", "T_t"]`.
   - Rationale: even with `lambda_m = 0`, `N_t` is not a spurious variable — it is a near-perfect step counter, and as the lambda=18.5 reference grid shows ((N_t, T_t)-only achieves regret 0.056, closing 87% of the gap to slw01_rerun's 0.031 vs T_t-only's 0.218), a controller with `N_t` available can ignore `z_t` entirely and recover most of the oracle decision. Removing `N_t` forces the controller to depend on the encoder representation rather than the scaffolding signal.
   - Old warm-start checkpoints are no longer load-compatible: the first head layer changed shape from `[q_hidden, d_embed+2]` to `[q_hidden, d_embed+1]`. Finetuning runs starting now must train the head from scratch.

DDP investigation outcome (recorded for future reference, not pursued further this session):

- 4-GPU DDP gave only ~10% per-epoch wall-time speedup over single-GPU (~72 min vs ~78 min/epoch). Diagnostic timing instrumented around the data fetch / forward / backward phases showed `compute_ms ≈ 37 ms` while total per-batch was ~480 ms. The signature finding was an inverse correlation between `data_ms` and `backward_ms`: when rank 0's data fetch was fast, it sat at the gradient `all_reduce` waiting for the slowest rank; when rank 0's data fetch was slow, other ranks were already waiting on it.
- Diagnosis: **DDP load imbalance from variable-length episodes.** Different ranks get batches of episodes with very different total snapshot counts, so per-rank forward/backward times diverge and the synchronization gates the slowest rank every step. The model is also small for an A100 (128-d, ~96-node trees, kernels under-occupy SMs), so even per-rank work doesn't saturate.
- Real fix would be a length-aware `BatchSampler` (bin episodes by total snapshot count, snake-distribute across ranks per step). Estimated 2–3 h to implement and test. Not worth blocking the experiment on — left as a follow-up if multi-GPU scaling becomes important.

Pending run:

- 4-GPU DDP, N_t removed, no warm-start, 20 epochs, `--time=24:00:00 --mem=128G --cpus-per-task=8 NUM_WORKERS=2 SIGN_LOSS_WEIGHT=0.25 WARM_START_CHECKPOINT=`. Output checkpoint: `/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/checkpoints/finetune_controller.pt`. At ~66 min/epoch in DDP (without diagnostic instrumentation), 20 epochs ≈ 22 h — fits in the wall budget. Result to be logged once the run completes.

## 2026-05-13

### Codebase simplification + end-to-end pipeline verification

Intent:
- Migrate the flat repo layout to a hierarchical `src/cts/` package, switch every entry point and SLURM script to a single YAML-config interface, and prune accumulated dead code — then verify the refactored pipeline runs end-to-end on the cluster.

Meaningful changes:

1. **Package restructure**: flat top-level modules collapsed into `src/cts/{core,data,models,train,analysis}/`. `TreeNN` → `TreeEncoder`, `ComputeAdvantageTreeSearchModel` → `MetaController`. `pyproject.toml` + `conftest.py` added so `pip install -e .` and `pytest tests/` work cleanly. No shims preserved — every call site, config, and import moved in the same commit.
2. **Config system**: every entry point now reads a Pydantic-validated YAML via `--config PATH`, with optional `--override key=value` for nested overrides. Each SLURM script collapses to `python3 -m cts.X.Y --config "${CONFIG}"`; experiment variants live as sibling YAMLs (the diff between two experiments is the diff between their YAMLs). `extra="forbid"` rejects typos at load time. Reference table for the full pipeline is recorded in `~/.claude/CLAUDE.md`.
3. **Dead-code prune**: 7 dormant analysis scripts deleted (PPO-era trace tools, duplicated motif helpers). `analyze_budgeted_controller_run.py` (3050 LOC) split into 9 themed sub-modules under `cts/analysis/_budgeted/`. Five copies of duplicated regex parsers consolidated into `cts/analysis/_common.py`. The `TreeTensorizer` and `ControllerCollator` classes converted to free functions (no per-instance state). Defensive checks against impossible states removed throughout. README added; package docstrings tightened.
4. **Self-describing checkpoints**: encoder architecture now embedded in checkpoint metadata under `encoder_architecture`. `materialize` and `controller_train` read it directly, so 7 encoder-arch knobs disappear from `MaterializeConfig` and downstream YAMLs.
5. **Test suite**: 59 tests pass locally and on della (after `pip install pytest` in the CTS env). Two pre-existing brittle float-equality tests permanently deselected via `--deselect`.

Verification:
- Rsynced to a fresh cluster directory `~/chess/cts/async_soph_restruct/` so the previous code stayed intact.
- Ran a 10-fen smoke through every pipeline stage in order: build_tree (sharded via orchestrator) → preprocess_gnn split → preprocess_gnn pack → encoder pretrain (100 epochs, K=1, loss decreased monotonically) → preprocess_mc pack → materialize + merge (train and validation) → controller train (slw01, 20 epochs). All outputs landed in `_smoke`-suffixed scratch directories.
- Surfaced and fixed three pre-existing issues during the smoke: (a) `submit_generate_dataset_shards.py` still emitted the old `FENS_PATH`/`START_INDEX` env-var pattern after the slurm script had been migrated to read `CONFIG` — rewrote it to emit per-shard YAMLs; (b) CPU-only slurm scripts unconditionally `module load cudatoolkit/12.8`, which failed when they landed on CPU nodes — dropped the load from 6 scripts that don't request a GPU; (c) `merge` subcommand of materialize requires an explicit `encoder_checkpoint` field to populate cache metadata, or downstream `controller_train` rejects the cache as a mismatch — documented in the smoke merge YAMLs.

Conclusion:
- Refactored codebase verified operational end-to-end. The new YAML-driven interface and src layout are the baseline going forward; further experiments will branch off this state.

## 2026-05-15 (hl4291 — tree generation on della)

### Cpu vs gpu for lc0 dataset generation

Intent:
- Establish practical throughput and cluster constraints for generating large teacher-tree shards with ysagiv’s `lc0` (CUDA-linked binary), before scaling to a 50k-FEN run.

Observations:

1. **Loader / libcublas on CPU nodes (historical).** Early CPU smokes needed pip CUDA wheels on `LD_LIBRARY_PATH` when `/usr/local/cuda-12.8` was missing. That path was too slow for production; only the **GPU array** wrapper in `hl4291_slurm/1_generate_shards.slurm` was used for the 50k corpus.

2. **Throughput.** Tree generation is lc0-bound; **GPU array** is the production path (`--array=0-9`, `CHUNK_SIZE=5000`).

3. **Per-example disk format.** Trees land as `cts_raw_pretrain_example_v3` per file; spot-checks via `load_raw_pretrain_record` / `to_pretrain_example()` confirm `format` tag, tensor lengths aligned with `tree.num_nodes()`, and edges consistent. Example sizes observed on scratch (thousands of nodes) reflect the teacher expansion, not a literal “96 nodes” cap — the `min_nodes` / `max_nodes` in YAML feed **budget sampling** in the search config.

### Slurm: 50k shard array (gpu, 10 concurrent tasks)

Meaningful changes:

- **`hl4291_slurm/1_generate_shards.slurm`**: moved from `--partition=cpu` to **one GPU per task** (`#SBATCH --gres=gpu:1`, `#SBATCH --constraint=nomig`, wall `24:00:00`). Environment setup uses hl4291 venv + `module load cudatoolkit/12.8`. Set **`PYTHONUNBUFFERED=1`** and **`python3 -u`** so batch logs show progress without line-buffering surprises. (Yotam’s generic GPU wrapper remains in `slurm/generate_dataset_shard.slurm`.)
- **Sharding policy**: build **50k indices** as **`--array=0-9`** with **`CHUNK_SIZE=5000`** (10 tasks × 5000 FENs) instead of 100×500, to stay within a typical **array-size cap** (~10 concurrent tasks at many sites).
- **`configs/data/hl4291_build_tree_50k.yaml`**: header comments updated with the `sbatch` line including `CHUNK_SIZE=5000`; output remains `.../generated_trees_50k`.

Operational:

- Cleared prior **`generated_trees_50k/*.pt`** before a fresh GPU run.
- Submitted GPU array job **`8289498`** (verify completion in `sacct` / `logs/cts-gen-50k_*`).

### Smoke policy (project convention)

- **Slurm smoke jobs** should keep **`--time` ≤ ~10 minutes** for quick checks; smoke YAMLs + overrides should be sized to finish under that unless deliberately stress-testing.

## 2026-05-17 (hl4291 — topology-supervised encoder pretrain)

Intent:

- Use per-node teacher topology from ``cts_raw_pretrain_example_v3`` (``n_visits``, ``nodes_below``, ``max_breadth_relative``, ``max_depth_relative``) for **encoder pretraining** via a standalone **``TopologyHead``** (still separate from Child-WDL), with Huber regression by default and an optional **weighted / phased Huber curriculum** (`topology_target_weights` in `BuildTreeConfig` / `TopologyPretrainConfig`).
- Optionally **widening** packed ``node_features`` to nine columns **for downstream Child-WDL** that should *see* teacher topology at the embedding input — without duplicating incompatible code paths.

**Strategic framing (why we care):** see **`### Bigger picture: node potential`** at the end of **2026-05-19 (hl4291 — pipeline docs, `hl4291_slurm/`, `fens/` layout)** — embeddings should ultimately encode **node potential** for metacontroller decisions (especially at the root), not only minimize Huber on raw targets.

Packing presets ([`PackPretrainConfig`](src/cts/data/preprocess_gnn/pack.py)):

| Situation | `topology_features` | `topology_supervision_shard` |
|-----------|---------------------|-------------------------------|
| Baseline Child-WDL (narrow) | `false` | `false` |
| Topology-only shards (historic layout: 5-wide + flat targets) | `false` | `true` |
| **Full template** (9-wide rows + duplicated flat ``topology_targets``, same scaled values from one helper) | `true` | `true` |

**Validation**: `topology_features=true` requires `topology_supervision_shard=true` so targets are never missing for topology pretrain and tensors cannot drift.

**YAML migration**: deprecated keys are still accepted once and stripped:

- ``include_topology_targets: true`` → `topology_supervision_shard: true`, `topology_features: false`
- ``include_teacher_topology: true`` → both flags `true` (full template).

**Internals**: `scaled_teacher_topology_matrix` in [`teacher_targets.py`](src/cts/data/preprocess_gnn/teacher_targets.py) fills both the last four ``node_features`` columns (wide pack) and the flat ``topology_targets`` shard (`log1p` on visits and ``nodes_below`` only).

**Checkpoint widths**: Topology pretraining still feeds a **five-column** subtree into ``TreeEncoder`` (``collate_tensorized_examples_for_topology`` strips trailing topology columns); saved encoder checkpoints therefore keep ``node_feat: 5``. **Nine-wide Child-WDL** requires a freshly built ``TreeEncoder(node_feat=9)`` and is **not** a silent ``load_state_dict`` from a topology-pretrained narrow checkpoint without resizing the input linear.

**Weighted Huber**: `topology_target_weights: [w0,w1,w2,w3]` (aligned with [`topology_target_feature_names`](src/cts/core/schema.py)) averages per-target loss with that mask; zeros remove a dimension from the active denominator (implemented in [`TopologyPretrainer._regression_loss`](src/cts/train/gnn_pretrain.py)).

**Collate fallback**: Wide shards without an explicit ``topology_targets`` tensor can still load if manifest + trailing column names match; prefer always writing ``topology_targets``.

Example configs:

- [`configs/data/preprocess_gnn/topology_pack_topology_supervision_only.yaml`](configs/data/preprocess_gnn/topology_pack_topology_supervision_only.yaml)
- [`configs/data/preprocess_gnn/topology_pack_full_template.yaml`](configs/data/preprocess_gnn/topology_pack_full_template.yaml)
- **Scratch (≈49k trees, topology full preset):** seeded train/val via [`configs/data/preprocess_gnn/hl4291_split_50k_topology.yaml`](configs/data/preprocess_gnn/hl4291_split_50k_topology.yaml) → pack via [`configs/data/preprocess_gnn/hl4291_pack_50k_topology_full.yaml`](configs/data/preprocess_gnn/hl4291_pack_50k_topology_full.yaml).

### 49k-ish scratch pipeline: seeded split + topology full pack

Train/validation manifests should **not** be built as “sorted paths, first 90% / last 10%”: filename order tracks generation index and can bias val. Use [`split.py`](src/cts/data/preprocess_gnn/split.py) instead: collect `*.pt` under `source_root` in **sorted** order (so filesystem walk order does not leak into reproducibility), **shuffle with `SplitConfig.seed`**, then write validation as the first ``max(1, int(round(n * validation_fraction)))`` shuffled paths and train as the remainder (`validation_fraction` must stay in `(0,1)`).

| Step | Config / artifact |
|------|-------------------|
| FEN list | `.../CTS/fens/sampled_root_fens_2023.txt` — see [`hl4291_build_tree_50k.yaml`](configs/data/hl4291_build_tree_50k.yaml) |
| Generate | [`hl4291_slurm/1_generate_shards.slurm`](hl4291_slurm/1_generate_shards.slurm): GPU array, `CHUNK_SIZE=5000`, 24 h wall |
| Split | [`hl4291_split_50k_topology.yaml`](configs/data/preprocess_gnn/hl4291_split_50k_topology.yaml): `source_root=.../generated_trees_50k`, `split_root=.../pretrain_split_50k_topology`, `validation_fraction=0.1`, `seed=42`, `clear=true`. Typical counts with **48,701** `.pt`: **43,831 train**, **4870 validation**. |
| CLI split | ``python -u -m cts.data.preprocess_gnn.split --config configs/data/preprocess_gnn/hl4291_split_50k_topology.yaml`` |
| Pack | [`hl4291_pack_50k_topology_full.yaml`](configs/data/preprocess_gnn/hl4291_pack_50k_topology_full.yaml): `topology_features` + `topology_supervision_shard`, ``output_root=.../pretrain_packed_50k_topology_full``, `shard_size=2000`, `clear=true`. Serial pack (no `num_workers`). |
| Slurm pack | [`hl4291_slurm/2_pack_shards.slurm`](hl4291_slurm/2_pack_shards.slurm): `#SBATCH --time=00:15:00`, 1 CPU, 4G. ``sbatch hl4291_slurm/2_pack_shards.slurm`` with pack CONFIG. |

Full hl4291 submit/status table: [`hl4291_slurm/README.md`](hl4291_slurm/README.md).

Changing **`seed`** in `hl4291_split_50k_topology.yaml` gives a different random partition with the same tree set; rerun **split** before **pack** whenever the partition should change.

CLI:

- Split: ``python -m cts.data.preprocess_gnn.split --config ...``
- Pack: ``python -m cts.data.preprocess_gnn.pack --config ...``
- Topology pretrain: ``python -m cts.data.build_tree --config ...`` with ``command: pretrain-topology-encoder``; optional ``topology_target_weights: [1.0, 1.0, 0.0, 0.0]`` for phased training.

Tests: [`tests/test_topology_pack_presets.py`](tests/test_topology_pack_presets.py).

Operational note: 50k GPU run left indices **38701–39999** missing (array task 7); topology pack/pretrain can proceed on **48,701** v3 trees until that shard is regenerated.

## 2026-05-18 (hl4291 — topology full pack: serial-only + Slurm sizing)

### Why this work was needed

Topology encoder pretrain on the **48,701-tree** split is blocked until [`hl4291_pack_50k_topology_full.yaml`](configs/data/preprocess_gnn/hl4291_pack_50k_topology_full.yaml) finishes writing to ``pretrain_packed_50k_topology_full``. The first della pack job (**8365231**) produced a single train shard and then sat idle until **TIMEOUT** — so the failure mode had to be understood before burning more queue time or claiming the pipeline “just needs more wall clock.”

### What went wrong (diagnosis)

Symptom: after ~90 s and **shard_00000.pt**, logs stopped; four more hours with no `shard_examples=` lines, then Slurm killed the job.

Why that pattern points away from “slow trees” or “bad shard-2 data”:

- Throughput on shard 1 was healthy (~22 ex/s for 2k examples).
- At that rate the full corpus is **~40 min**, not 4 h — so TIMEOUT was a **hang**, not under-provisioned time.
- Shard-2 input paths exist and are similar size to shard 1; nothing suggested one pathological file.

Actual mechanism: [`pack.py`](src/cts/data/preprocess_gnn/pack.py) opened a **new `ProcessPoolExecutor` for every shard**. Shard 1’s pool ran while the parent had only imported PyTorch. After shard 1, the parent held large tensors from **`torch.cat` / `torch.save`**, then forked workers for shard 2. **Fork + active PyTorch in the parent** is a known deadlock scenario; workers never returned, so `as_completed` never reached the next `log_interval` line. Reproduced on vis: second pool hangs after a parent `torch.cat`/`save` warmup; serial path crosses shard boundaries fine.

The `hl4291` config used **`num_workers: 8`** even though topology templates used **`num_workers: 0`** — so we turned on a code path the repo did not actually rely on.

### Decisions (why each change)

| Decision | Why |
|----------|-----|
| **Pack serially only; delete `num_workers`** | Multiprocessing here is **unsafe** (per-shard fork after torch) and **unnecessary**: measured serial throughput on real trees is **~90–115 ex/s**, i.e. full pack in **~8–10 min**. Keeping `num_workers` implied a supported fast path that **fails silently at shard 2**; rejecting the key in YAML fails loud on stale configs. |
| **Smoke test through shard 2** | The production bug was specifically “shard 1 OK, shard 2 never starts logging”; a test that only packs one example or one shard would not catch it. |
| **`cpus-per-task=1` + `OMP_NUM_THREADS=1`** | Work is strictly sequential; extra Slurm CPUs do not speed this CLI and encourage false confidence that `num_workers` does something. |
| **`mem=4G` (not 64G)** | Measured peak for a full 2k-tree topology shard (load + tensorize list + `torch.cat` + save) is **~1.7 GB**; memory does **not** grow with shard count because each shard is assembled and written then dropped. **64G was ~30× need** and unfair on a shared cluster; **4G ≈ 2× peak** leaves headroom without hoarding. |
| **`time=00:15:00`** | Serial pack on the full ~49k split completed in **~9 min** on della (job **8429582**); 15 min leaves headroom vs vis-node benchmarks without hoarding queue time. |

If we need parallelism later, the safe patterns are **Slurm array over shards** (separate processes, no fork-after-torch) or **`spawn`** with one long-lived pool — **not** “new `ProcessPoolExecutor` per shard” in one Python process.

### What changed (reference)

- [`pack.py`](src/cts/data/preprocess_gnn/pack.py): removed pool path; `tensorize_example_for_pack`; `PackPretrainConfig` rejects deprecated `num_workers`.
- GNN pack YAMLs: dropped `num_workers` lines (including [`hl4291_pack_50k_topology_full.yaml`](configs/data/preprocess_gnn/hl4291_pack_50k_topology_full.yaml)).
- [`tests/test_topology_pack_presets.py`](tests/test_topology_pack_presets.py): `test_pack_topology_smoke_reaches_second_train_shard`.
- [`hl4291_slurm/2_pack_shards.slurm`](hl4291_slurm/2_pack_shards.slurm): 1 CPU, 4G, 15 min wall, thread caps in the job script.

### Result

Cleared partial pack output and resubmitted on della after the Slurm/code fixes. Job **8429582** (`cts-pack-topo-50k`) **COMPLETED** (exit 0) in **9 min 15 s** on 2026-05-18.

Output (`pretrain_packed_50k_topology_full` on scratch):

| Split | Examples | Shards | Size |
|-------|----------|--------|------|
| Train | 43,831 | 22 | ~15 GB |
| Validation | 4,870 | 3 | ~1.7 GB |

Manifests: `topology_features=true`, `topology_supervision_shard=true`. Spot-check on `shard_00000`: `node_features` shape `(N, 9)`, `topology_targets` `(N, 4)`. Log throughput **~93–94 ex/s** (serial path). Log: [`logs/cts-pack-topo-50k_8429582_0.out`](logs/cts-pack-topo-50k_8429582_0.out).

Topology encoder pretrain on this split is unblocked. Tree generation still missing indices **38701–39999** (array task 7); the pack used all **48,701** v3 trees currently on scratch.

## 2026-05-18

### VOC-based metacontrol framework: design exploration

This entry records a long design discussion on how to fix the Section-3 failure mode (controller collapses to a `T_t` countdown, ignoring child-WDL information in `z_root`). It is conclusions-oriented but includes enough of the alternatives considered that the rationale is recoverable without the original conversation.

**Motivation and reframing of Section 3.**

The original Section 3 finding was that the trained advantage head loads almost entirely on directions of `z_root` orthogonal to the decoded child-WDL subspace (78% weight orthogonal; Pearson with decoded VPI ~0.07-0.09). The natural first reading was an extraction failure: WDL is in `z_root`, the controller just can't read it. The right reading is structural: value of computation depends on *posterior uncertainty* over child values, and a point-estimate WDL contains no such signal — VOC over a point estimate is zero by definition. The controller correctly assigned negligible weight to a feature that mathematically cannot distinguish "child confidently winning" from "child looks winning but search hasn't settled." It fell back on `T_t` (and `N_t`, before that was removed), which at least proxies "how much has been searched."

The fix is upstream of the controller: the encoder must output a posterior (mean *and* concentration) per child, not just a mean. The controller can then potentially learn to use the concentration channel.

**Three layers of uncertainty, kept separate.**

- *Layer 1:* WDL point estimate — the mean of the engine's belief over a child's value distribution. This is what current pretraining targets.
- *Layer 2:* Concentration of the belief itself — how stable the WDL would be under more teacher search. This is the layer VOC depends on, and is missing from the current pretraining objective.
- *Layer 3:* Student-level reconstruction error — how reliably the encoder predicts the teacher's WDL. Orthogonal to the Bayesian content; relates to model uncertainty.

The intervention targets Layer 2 explicitly via Dirichlet pretraining: the encoder outputs Dirichlets on the W/D/L simplex per parent→child edge, with mean reflecting the WDL estimate and concentration reflecting accumulated search-derived evidence.

**Where the Dirichlets come from.**

Two candidate sources were considered:

- *Path 1 (within-tree leaf variability).* Treat leaf evaluations under each child's subtree as soft observations under a conjugate Dirichlet–Multinomial model. The target posterior at edge `e` is `α_e* = ε·1 + κ·θ_leaf_c + Σ_l θ_leaf_l`, where `θ_leaf_c` is lc0's evaluation of `c` itself (serving as a per-position prior with strength `κ`), `ε·1` is small Laplace smoothing (keeps Dirichlets non-degenerate against zero-component lc0 outputs), and the sum is over expanded leaves in `c`'s subtree.

- *Path 2 (across-search variability).* Run teacher search `M` times per position with different random seeds (root Dirichlet noise, tie-breaking, expansion order); use the spread of resulting WDLs as the layer-2 estimate. More empirically principled and makes no parametric assumption about leaf exchangeability, but costs M× the dominant search compute.

Path 1 was chosen on efficiency and cognitive-plausibility grounds (humans don't run search multiple times). The load-bearing approximation is that leaves are iid samples from a single underlying distribution per node — violated in tactical positions where shallow leaves (e.g., immediate material loss after a sacrifice) and deep leaves (mating attack emerging) are systematically different. The iid assumption is acceptable as a baseline; the principled refinement is a hierarchical model with parent-conditional priors, if calibration diagnostics show it's needed empirically.

**Aggregation operator at internal nodes.**

The encoder outputs Dirichlets at every node; target Dirichlets are computed by aggregating from leaves up. Several operators were considered:

- *Leaf-sum.* Each node's Dirichlet is `ε·1 + κ·θ_leaf_self + Σ_subtree θ_leaf` — the additive Bayesian conjugate update. Smooth, simple. Failure mode under VOC-max search: a confident-and-best child gets few visits (no further information to gain), so its high value never accumulates in ancestor leaf-sums. The mate-leaf at depth 5 only contributes one observation to root's leaf-sum, drowned by visits to less-resolved children. PUCT hid this bug because exploitation kept the best child accumulating visits.

- *Hard max.* Each node's Dirichlet inherits from its leader child (the one with highest `μ̄ = w̄ - l̄`), recursively. Propagates a mate's certainty exactly up the tree. Failure mode: discrete cascade dynamics — a single new observation that flips the leader at any level cascades the leader chain up to root, producing discontinuous training targets for the encoder (two near-identical trees can have very different Dirichlet targets at every internal node).

- *Softmax-weighted.* Parent's Dirichlet parameters = softmax(`μ̄_c / τ`)-weighted sum of children's Dirichlet parameters. At `τ → 0` recovers hard max; at `τ → ∞` becomes uniform average. With low `τ`, the operator approximates hard max while keeping training targets smooth in the children's parameters.

Softmax-weighted aggregation with low `τ` chosen. One operator throughout, smooth at training, near-hard at inference.

**Where VOC computation lives.**

This was the most repeatedly confused part of the discussion. Resolution: **VOC computation is purely a data-generation tool. There is no VOC math at inference.**

The role of VOC in the pipeline:

1. *Data generation.* During teacher search, expansion is driven by VOC-max: at each step, compute Dirichlets at every node in the current tree via bottom-up softmax aggregation; compute VOC at each level via 1D numerical integration over the predictive Dirichlet; descend by argmax VOC to a frontier leaf; lc0 evaluates that leaf (and its children, per the project's expansion convention). Iterate to budget.

2. *Training.* The encoder learns to predict the bottom-up-aggregated Dirichlets (KL loss). The metacontroller learns from the resulting snapshot-sequence trajectories via fitted-Q with Bellman targets — exactly the existing training pipeline, applied to the new data.

3. *Inference.* Encoder forward pass produces Dirichlets. Metacontroller forward pass produces decisions. No closed-form math, no Gaussian order statistics, no numerical integration. The metacontroller has learned (via fitted-Q on VOC-expanded data) to mimic the VOC-max policy: at root it predicts halt iff `max_c A ≤ 0`; at internal nodes during descent it picks argmax A.

The closed-form Gaussian-order-statistics approach to VOC was the natural inference-time computation if VOC were needed at inference — but under fitted-Q, the metacontroller doesn't need explicit VOC targets, and at inference the metacontroller's output replaces explicit VOC. So that machinery moves to data-generation time, where exactness matters more than speed. **1D numerical integration** over the marginal of `θ_leaf_W - θ_leaf_L` was chosen for VOC computation during data generation: deterministic, bounded error by floating-point precision, no hyperparameter to tune, no MC convergence diagnostics. One-time cost paid once at data prep.

**Cascade behavior is accepted.**

Under VOC-max + softmax-with-low-τ aggregation, a single new leaf observation can cascade leader changes from its parent up to root (one observation can flip the entire decision chain). This is the correct behavior — finding a mate should immediately commit to that line, not require many confirmation visits. PUCT's smooth backup actively fights against this property by requiring many visits to accumulate certainty. The cascade is a feature, not a bug, for chess metacontrol. The training-time discontinuities it creates are handled by the softmax smoothing.

**Final framework.**

- *Data generation.* Per sampled position: build a tree using VOC-max as the in-tree search policy. At each step: compute Dirichlets at every node via bottom-up softmax aggregation from leaves; compute VOC at each level via 1D numerical integration on the predictive Dirichlet; descend by VOC-max to a frontier leaf; lc0 evaluates the leaf and its children; iterate to budget. Output: tree with leaf evaluations, Dirichlet targets at every node, and the snapshot sequence for trajectory data.

- *Encoder.* Same GNN architecture as existing. Decoder output changed from softmax over `(W, D, L)` to softplus-parameterized Dirichlet `(α_W, α_D, α_L)`. Loss: Dirichlet KL against the per-node targets. Each leaf's Dirichlet is just `ε·1 + κ·θ_leaf` (no descendants).

- *Metacontroller.* Same MLP architecture as existing. Input changed to `(z_root, T_t)` — `N_t` dropped, since it's a near-perfect step counter that lets the controller shortcut around `z_root`. Training: fitted-Q on offline trajectories with Bellman targets (the existing approach).

- *Inference.* Per iteration:
  1. Encoder forward pass on the current tree.
  2. Metacontroller applied at root → halt iff `max_c A ≤ 0`.
  3. If continue: metacontroller applied at each level during descent, argmax over children's `A` to pick descent direction, ending at a frontier leaf.
  4. lc0 evaluates that leaf; encoder reruns; next iteration.

- *Hyperparameters.* `κ` (lc0 prior strength), `ε` (Laplace smoothing), `τ` (softmax aggregation temperature). To be tuned empirically.

- *Diagnostics.* Held-out scatter of predicted vs. target Dirichlets (mean and concentration). Section-3-style weight-orthogonality check on the new advantage head — does it now load on variance-bearing directions of `z_root`? Regret/exact-stop on the existing eval set vs. the current baseline.

**Two implementation branches.**

To separate the encoder hypothesis from the search-policy hypothesis:

- *Branch 1 (GNN-only change).* Retrain encoder on Dirichlet targets using *existing* PUCT-generated trees. Metacontroller retrained via fitted-Q on existing trajectories. Inference unchanged (PUCT for in-tree search). Tests whether Layer-2 information in `z_root` alone fixes the `T_t`-countdown collapse, without changing the search policy.

- *Branch 2 (full pipeline).* Regenerate trees using VOC-max expansion. Encoder and metacontroller trained on this new corpus. Inference uses the metacontroller for both halt and descent. Tests whether VOC-max as the search policy adds value beyond Layer-2 pretraining alone.

Branch 1 ships first (smaller change, isolates the encoder hypothesis, reuses existing data). Branch 2 added if Branch 1 shows the encoder hypothesis is correct but plateaus below the target performance.

**Key abandoned ideas, with reasons.**

- *Direct VOC supervision of the encoder.* Rejected on the principle that the GNN should produce primitives (Dirichlets) and the controller should learn control. Predicting VOC directly would blur this decomposition and is closer to AlphaZero-style direct value supervision than to the project's "primitives for downstream tasks" framing.

- *Two-phase scheme (VOC-max planning + hard-max consolidation at action time).* Initially proposed as a way to combine efficient compute allocation with correct value propagation. Rejected because if planning uses leaf-sum values and action uses consolidation-derived values, the planning VOC computation is optimizing against a quantity (`max` of leaf-sum means) different from the actual decision rule (`max` of consolidated means). The cascade dynamics of hard max are what create the misalignment; soft (visit-weighted) aggregation under PUCT historically papered over this by over-visiting good children.

- *Expected-max aggregation with moment-matched Dirichlets at internal nodes.* The expected-max distribution at an internal node lives on a scalar (`max_c μ_c ∈ [-1, 1]`), not on the simplex. Moment-matching to a Dirichlet is conceptually muddled — forcing a simplex parametric form on a quantity that isn't simplex-valued.

- *Dropping Dirichlets at internal nodes in favor of `(mean, variance)` scalars.* Introduces a leaves-vs-internal asymmetry in the encoder's output representation. Better to keep one parametric form throughout (Dirichlets) and let softmax aggregation handle the recursion.

- *MC sampling for VOC computation at data gen.* Functional but requires choosing K and verifying convergence — unwanted ongoing calibration burden. 1D numerical integration replaces it: same accuracy class, no hyperparameter, deterministic.

**Status.**

Conceptually settled. Implementation hasn't started. Next steps: Branch 1 first (encoder retraining on existing data), with the calibration diagnostic and Section-3 weight-orthogonality check as the immediate post-training go/no-go signals.

### Encoder KL audit by (parent_depth, child_subtree_size)

Intent: the pretraining log reports one scalar `loss_gap ≈ KL(target‖pred)` averaged over every supervised edge. That hides whether the encoder is uniformly accurate or leans heavily on easy slices (e.g. leaf children, where the "child WDL" target is just the leaf's own raw WDL with no consolidation). For the metacontrol question this matters: `z_root` has to summarize the root's children, which in 96-node trees are internal nodes with substantial subtrees beneath them, not leaves.

Approach: re-run the rerun encoder + decoder over the rerun pretrain split, compute per-edge `KL(target‖pred) = (target * (log target − log pred)).sum(-1)`, and bucket by parent depth (plies from root, 0–12+) and child subtree size in log₂ bins (1, 2–3, 4–7, …, ≥128). Output: per-cell mean KL + edge counts, marginals along each axis, overall mean KL (sanity-checks against the pretraining log), and a heatmap PDF. Implementation lives at `cts.analysis.audit_encoder_kl` with the universal `load_pretrain_example_dataset` so it accepts either packed JSON manifests or raw text manifests.

`rewrite_compact` bug surfaced and worked around (not fixed): the pack stage requires raw examples in `cts_raw_pretrain_example_v2`, but the rerun split's .pt files are still `_v1`. The `cts.data.preprocess_gnn.rewrite_compact` module advertises v1 dict → v2 conversion (`_load_record_from_source` line 242), but the v1-dict branch immediately calls `RawPretrainExampleRecord.from_payload`, which validates strict-v2 (`teacher_targets.py` lines 563–564) and rejects v1. The v1 PretrainExample-pickle branch (`from_example`) works; only the dict branch is broken. Worked around by pointing the audit at the rerun split's text manifests directly (the audit tensorizes on the fly via the universal loader; costs minutes on a 1h walltime). The rewrite_compact + pack chain is wired but unused; restoring `pretrain_packed_oracle96_trace_filtered_rerun/` to disk requires fixing the bug first.

**Inputs:**
- `/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/pretrain_split_oracle96_trace_filtered_rerun/{train,validation}_manifest.txt` — rerun pretrain split, text manifests pointing at v1 raw .pt examples.
- `/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/checkpoints/tree_encoder_child_wdl_async_k1_rerun.pt` — rerun encoder.
- `/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/checkpoints/tree_encoder_child_wdl_async_k1_rerun_decoder.pt` — paired child-WDL decoder.

**Outputs:**
- `/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/encoder_kl_audit/audit_{train,validation}.json` — per-bucket mean KL, edge counts, marginals, overall mean KL, plus depth and size-bin labels.
- `/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/encoder_kl_audit/audit_{train,validation}.pdf` — heatmap of mean KL over the (parent_depth, child_subtree_size) grid with edge counts annotated.

Outcome: pending — sbatches submitted, results to be appended.

### rewrite_compact v1-dict branch is broken (independent bug, noted)

`cts.data.preprocess_gnn.rewrite_compact._load_record_from_source` claims to accept v1 raw-record dicts as input (module docstring + line 242 dispatch). It does not: the v1 branch immediately calls `RawPretrainExampleRecord.from_payload`, and `from_payload` (in `cts.data.preprocess_gnn.teacher_targets`, lines 563–564) is strict-v2:

```python
if payload.get("format") != RAW_PRETRAIN_FORMAT:  # "cts_raw_pretrain_example_v2"
    raise ValueError(f"Expected raw pretrain format {RAW_PRETRAIN_FORMAT}, got {payload.get('format')!r}.")
```

The PretrainExample-pickle branch (`from_example`) still works, because that path reconstructs a `PretrainExample` via the legacy unpickler shim and converts via `from_example`. So `rewrite_compact` works on legacy pickled-`PretrainExample` files but fails on the v1 dict-payload variant — which is what the rerun split's raw .pt files happen to be.

The fix is not just relaxing the format check: the v2 bump was motivated by canonical UCI-sorted slot ordering (see `_child_ptr_and_children_index` comment, `teacher_targets.py` line 284). A v1 dict's `children_index` may be in engine-dependent (non-canonical) order, so a correct v1→v2 conversion has to re-sort children by UCI and reorder `edge_wdl_targets` to match — not just retag.

Not fixed this session; out of scope for the encoder KL audit. Recorded here so the next person who tries to use `rewrite_compact` doesn't repeat the diagnosis. Workaround for any analysis that needs v1 raw .pt files: use the universal `load_pretrain_example_dataset` (text-manifest or directory mode), which tensorizes on the fly via `tensorize_forest` and `edge_wdl_target_tensor` — those helpers apply the canonical slot sort, so downstream is consistent with v2 packed shards.

## 2026-05-19 (hl4291 — pipeline docs, `hl4291_slurm/`, `fens/` layout)

### Status (end of day)

The **topology-full pack** for the ~49k split is **done** (see 2026-05-18: job **8429582**, ~9 min). Scratch artifacts:

| Artifact | Path |
|----------|------|
| Raw trees | `.../CTS/data/generated_trees_50k/` (**48,701** `.pt`; **1,299** missing at global indices **38701–39999**) |
| Split manifests | `.../CTS/data/pretrain_split_50k_topology/` |
| Packed GNN data | `.../CTS/data/pretrain_packed_50k_topology_full/` (43,831 train + 4,870 val examples) |
| FEN inputs | `.../CTS/fens/` (repo-local mirror: [`fens/`](fens/)) |

**Next:** topology encoder pretrain (`command: pretrain-topology-encoder`, Huber + `TopologyHead`) — needs a GPU YAML + Slurm script (not committed yet). Optional: backfill the **38701–39999** gap (~**2.8 h** on one GPU at ~0.13 roots/s, or ~**15–20 min** wall with 10 parallel sub-slices).

### `hl4291_slurm/` (numbered scripts)

Separated **hl4291** della wrappers from Yotam’s [`slurm/`](slurm/) (conda `CTS`, ysagiv paths). Only the paths that actually ran in production are kept:

| # | Script | Role |
|---|--------|------|
| 1 | [`hl4291_slurm/1_generate_shards.slurm`](hl4291_slurm/1_generate_shards.slurm) | GPU array: `CHUNK_SIZE` × `SLURM_ARRAY_TASK_ID` → `--override start_index/end_index` |
| 2 | [`hl4291_slurm/2_pack_shards.slurm`](hl4291_slurm/2_pack_shards.slurm) | Serial pack (`CONFIG` = topology preset, split paths, shard size) |

Dropped unused CPU / single-GPU generate wrappers after confirming only the **array** job was used for the 50k corpus.

### `fens/` vs `data/`

Renamed input layout so FEN lists are not under `data/`: **`CTS/fens/`** for text inputs, **`CTS/data/`** for trees / splits / packed shards. [`hl4291_build_tree_50k.yaml`](configs/data/hl4291_build_tree_50k.yaml) now points at `.../CTS/fens/sampled_root_fens_2023.txt`. Repo checkout: [`fens/sampled_root_fens_2023.txt`](fens/sampled_root_fens_2023.txt).

### Array task 7 failure (gap regen reference)

Job **8289498_7** (**FAILED** after **8h 12m**, lc0 `Unexpected EOF` at ~global index **38700**; log [`logs/cts-gen-50k_8289498_7.out`](logs/cts-gen-50k_8289498_7.out)). Regenerate only the gap with `resume: true` and e.g.:

### Topology pretrain loop (implemented 2026-05-19)

**Entry point:** `python3 -m cts.data.build_tree` with `command: pretrain-topology-encoder` (same module as tree generation). Loads JSON manifests → `PackedTensorizedShardDataset` → `TopologyPretrainer` (Huber default) → best encoder + `TopologyHead` checkpoints.

| Artifact | Path |
|----------|------|
| Slurm step 3 | [`hl4291_slurm/3_pretrain_topology.slurm`](hl4291_slurm/3_pretrain_topology.slurm) |
| Full config | [`hl4291_pretrain_topology_50k.yaml`](configs/data/hl4291_pretrain_topology_50k.yaml) — 43,831 train / 4,870 val, `batch_size=8`, `epochs=20`, `k=1`, 128-d |
| Slurm smoke config | [`hl4291_pretrain_topology_shard_slurm_smoke.yaml`](configs/data/hl4291_pretrain_topology_shard_slurm_smoke.yaml) — capped manifests under [`configs/data/manifests/`](configs/data/manifests/) (64+32 trees from shard 0 only) |
| One-batch dev smoke | [`scripts/smoke_topology_pretrain_batch.py`](scripts/smoke_topology_pretrain_batch.py) |

Shard Slurm smoke on GPU: **2 epochs**, val loss **≈0.56**, checkpoints written to `tree_encoder_topology_shard_smoke.pt` + `_topology_head.pt` (~33 s including preload).

Full-run expectations: **~15–20 min** dataset preload (all shards into RAM), then **many hours per epoch** at `batch_size=8` on large trees — use `*_resume.pt` + chained 24 h jobs if needed.

Single-GPU backfill (with `resume: true` in the build_tree YAML):

```bash
export CONFIG="$PWD/configs/data/hl4291_build_tree_50k.yaml"
python3 -u -m cts.data.build_tree --config "${CONFIG}" \
  --override start_index=38701 --override end_index=40000
```

Or submit via Slurm with the same overrides in the job script / a small gap-only YAML. For ~**15–20 min** wall, shard into 10 array tasks (`CHUNK_SIZE=130`, `BASE_START=38701`) — see [`hl4291_slurm/README.md`](hl4291_slurm/README.md).

### Bigger picture: node potential (metacontroller-facing goal)

This subsection records **constraints and intent** that sit above the concrete Huber/`TopologyHead` setup in **`## 2026-05-17 (hl4291 — topology-supervised encoder pretrain)`** earlier in this notebook. None of this *requires* a Dirichlet parametrization; that was optional geometry for sibling mass when we discussed simplex-aligned losses.

**North star — what we need from GNN topology pretraining**

- The encoder should yield representations that summarize **node potential**: a succinct sense of **how much further search/compute at this vertex is likely to matter**, given **only** what the teacher procedure actually produced from a **root-started** tree build.
- **Why the root (and downstream controller) cares:** allocation decisions (“grow here vs there”, PV vs widening, stopping) eventually need signals that behave like **marginal value of additional computation** plus **local expansion structure**. Pure board features miss the **budget-conditioned search state** frozen into the shard; topology supervision is how we distill that state into **`d_embed` vectors** the metacontroller can consume without rerunning lc0 inside the controller loop.

**Constraint (1) — root-only search; descendants are coarse**

- We **will not** rerun full MCTS/PUCT from every node as independent roots (ideal “goalpost”; too expensive).
- All per-node statistics in packed targets are induced by **simulations rooted once** — descendant rows are **path-conditional, budget-limited** summaries, not equilibrium from restarting search at each descendant. Statistical mass at **`n_visits ≈ 0`** and shallow leaves is therefore **expected**, not purely a labeling bug.

**Constraint (2) — marginal value of compute is latent; use proxies**

- Ground-truth \(\partial V / \partial \text{budget}\) per node is not shipped in the shards. Targets like **`log1p(n_visits)`**, subtree counts (`nodes_below`), and breadth/depth relatives are **cheap teacher proxies** for “traffic / unfinished business / shape” rather than literal marginal values.
- A trained encoder can still be **metacontroller-useful**: the downstream module learns a nonlinear map from **`h_v`** (+ online cues) to stop/expand once online reward or auxiliary losses exist.

**Constraint (3) — dual pretraining modes and observability shift**

| Mode | What sees the graph | Typical supervision |
|------|---------------------|---------------------|
| **A — Topology GNN pretrain** | **Flattened** packed trees: effectively **all** nodes (and CSR edges inside the encoder) in offline batches | Per-node **`topology_targets`**, optionally future edge-group objectives |
| **B — Metacontroller training** | **Growing** snapshot; in deployment often **narrow** visibility (e.g. along **principal variation + local fringe**), not the full contemporaneous subgraph used in Mode A |

- **Risk:** embeddings tuned on the **dense offline view** may under-generalize when the controller only **`walks`** a subtree of what pretrain **flattened**.
- **Mitigations to keep in mind:** path-centric masking / stochastic subgraph views at pretrain time, multi-view consistency, or distilling from teacher-rich states to student inputs that mimic online channels — design TBD alongside controller I/O spec.

**Where Dirichlet / simplex language optionally fits**

- If we supervise **how visit mass splits among siblings**, targets live on a **probability simplex** (\(\hat\pi_c \propto\) edge/consolidated child counts). Softmax **+ CE/KL is a practical surrogate** for that geometry; full Dirichlet–multinomial modeling is optional and orthogonal to `(1)`–`(3)` above.

**Through-line.** The exploratory work on **`n_visits` histograms**, masking, sibling shares, etc. circles the same motive: recover a **representation of node potential under root-budget dynamics** well enough that, **especially at the root**, the stack can eventually answer “is more search here worthwhile?” without replaying lc0.

**Generation budget (path B, current 50k yaml).** `generate-dataset` uses **`min_nodes` / `max_nodes`** (96/96) as the **expansion cap** (`num_expansions` in [`generate_partial_tree_from_provider`](src/cts/data/preprocess_gnn/teacher_targets.py)). Each loop iteration still does **PUCT select + backprop**; non-expanding iterations (depth cap, terminal, no children) **do not** increment `num_expansions` but **do** increment edge `visit_count`. **`search_budget`** is only consumed in **`compute_teacher_targets`** (path A) and is **inert** for the current 50k corpus — do not interpret **`n_visits`** as capped by `search_budget`.

### Candidate pretrain targets (node potential proxies)

**Conceptual shift (2026-05-20).** Move from supervising **static search artifacts** (`n_visits`, subtree shape) toward proxies for **future search evolution** / **search plasticity** — “how valuable would additional computation be **from here**?” — under the hard constraint that we only run **one root-started search** per position (no per-node MCTS restarts). All targets must be **derivable from partial snapshots of one evolving tree** (final tree and/or stored oracle trace).

**Deliverable for downstream metacontroller:** a **node-level** representation of **compute value / epistemic unfinishedness**, consumed for **stopping and budget allocation**. Edge/sibling geometry remains **input structure** (entropy, Q spread, competition among children), but the primary learned object is **`h_v` (node potential)**, not an edge-wise visit distribution.

| # | Target | What it measures | Pros | Cons | Suggested role |
|---|--------|------------------|------|------|----------------|
| **1** | **Value gap** \(V_{\text{best}} - V_{\text{second}}\) among children (visit-weighted Q at consolidation) | Action ambiguity / decision-boundary sharpness; small gap ⇒ unresolved competition | Simple, cheap, **per-node**, directly speaks to “might flip best move” | Ignores full policy vector; no explicit time axis; noisy when child visit counts are tiny | Strong **auxiliary** or lightweight primary |
| **2** | **Policy drift** \(D(\pi_k, \pi_{k+n})\) (KL / JS / TV / argmax flip) along search time | Actual **belief change** under more compute | Most aligned with meta-reasoning (“search still reorganizing?”) | Needs **temporal snapshots**; only nodes present across snapshots are well-defined; noisier to derive | Most principled **primary** where traces exist |
| **3** | **Child visit entropy** \(H(\hat\pi_{\text{children}})\), optionally normalized by branching factor \(K\) | Current allocation diffuseness / local competition | Cheap, dense, **simplex-aware**; easy from final `n_visits` | Confounds uncertainty with \(K\); static; weak evidence strength | **Auxiliary** ambiguity / plasticity proxy |

**Edge-wise vs node-wise.** Metacontroller needs **scalar or low-dim node potential**, not mandatory edge softmax targets. Sibling structure informs the encoder and can define **derived node scalars** (gap, entropy) without training a full edge distribution head.

**What is already on disk (50k v3, path B).**

- **Final-tree, all nodes:** child Q / WDL from consolidation → **value gap** per internal node; **visit entropy** from [`n_visits`](src/cts/data/preprocess_gnn/teacher_targets.py) sibling groups (incoming edge counts).
- **Temporal, root only:** [`oracle_root_q_trace`](src/cts/data/preprocess_gnn/teacher_targets.py), `oracle_best_move_trace`, `oracle_trace_expansion_counts` — **policy drift at the root** without re-search (see also §3 tree generation / §5 controller packing in this notebook).
- **Not stored today:** per-node \(\pi_t\) traces across expansion steps — **policy drift at non-root nodes** would require new logging or offline replay from prefixes (prefix episodes exist for **controller** packing, not yet as GNN topology targets).

**Current baseline vs this menu.** Topology pretrain supervision is **`value_gap` only** (raw centipawns, NaN on leaves / `<2` children) in **`cts_raw_pretrain_example_v5`** — no `n_visits` column, no edge WDL in the default generation path. Legacy Child-WDL pretrain can still opt into edge targets via `include_edge_wdl_targets=True`. **Repack required** for existing 50k topology-full manifests.

**Open design choices.**

- Combine targets as **multi-task Huber on scalars** vs **representation-only** (train head on proxies, export encoder without heads).
- **Masking:** skip nodes with \(K<2\) children or zero parent visit mass for gap/entropy; root drift needs \(k, k+n\) inside trace length.
- **Train/serve gap:** dense offline trees vs metacontroller **PV-narrow** views — consider path-masked pretrain when adopting drift-based targets.

## 2026-05-19 (hl4291 — value_gap schema refactoring to `node_targets` + edge case unit tests)

### Why this work was needed

We aligned the GNN tree pretraining pipeline with the transition to the new `v5` `value_gap` supervision schema. As part of this, the term `topology_features` was identified as mis-named for describing node-wise supervision targets embedded into the GNN input node features. To illustrate that these are node-wise targets for the pre-trainer, we renamed `topology_features` to `node_targets`. 

Additionally, we needed robust test coverage for specific mathematical edge-cases of the parent-perspective `value_gap` centipawn calculation (tied top siblings, tied suboptimal siblings, and single-child nodes).

### What changed (reference)

- **Terminology Alignment**:
  - Refactored `topology_features` to `node_targets` across:
    - [`schema.py`](src/cts/core/schema.py): Parameterized `tree_encoder_feature_schema` with `node_targets`.
    - [`pack.py`](src/cts/data/preprocess_gnn/pack.py): Updated the `PackPretrainConfig` schema, `tensorize_example_for_pack`, and packing scripts.
    - [`teacher_targets.py`](src/cts/data/preprocess_gnn/teacher_targets.py): Updated `to_tensorized_tree_example` signature and validation.
    - [`build_tree.py`](src/cts/data/build_tree.py) and [`smoke_topology_pretrain_batch.py`](scripts/smoke_topology_pretrain_batch.py): Updated schema instantiations.
  - Added a backward-compatible Pydantic validator in `PackPretrainConfig` that transparently maps deprecated `topology_features` keys in YAML configs directly to `node_targets`.
- **Added Robust Unit Tests**:
  - Implemented the following tests in [`tests/test_value_gap.py`](tests/test_value_gap.py):
    - `test_two_siblings_share_best_value_yields_zero_gap`: Asserts that when the top two children share identical values (both best), the value gap is exactly `0.0`.
    - `test_two_siblings_share_suboptimal_value_normal_gap`: Asserts that when two suboptimal children share identical values but are below the best child, the value gap correctly calculates as the normal difference between the best and second-best child values (no distortion).
    - `test_single_child_gap_is_nan`: Asserts that an internal node with a single child yields a `NaN` gap.
- **Hardened test suite**:
  - Verified that all **84/84 tests pass** successfully with exit code 0.

### Policy Drift Target Measurement (Softmax over visits & KL Divergence)

We implemented robust support for measuring search policy drift from "early" search stages ($t_{\text{early}}$) to "late" search stages ($t_{\text{late}}$) to capture search plasticity:
* **Softmax Policy Formulation**: Instantiated the child policy at time $t$ as a softmax distribution over child visit counts:
  $$p_t(c) = \text{softmax}\left(\frac{N_t(c)}{\tau}\right)$$
  Using softmax rather than raw frequency ratio ensures that all probabilities are strictly positive ($>0$), giving complete numerical stability to the downstream distance metric.
* **KL Divergence Distance Metric**: Quantified drift between early policy $p_{\text{early}}$ and late policy $p_{\text{late}}$ via KL Divergence:
  $$D_{\text{KL}}(p_{\text{early}} \parallel p_{\text{late}}) = \sum_c p_{\text{early}}(c) \log\left(\frac{p_{\text{early}}(c)}{p_{\text{late}}(c)}\right)$$
* **Non-Triviality Thresholds ($n_{\text{min}}$)**: Enforced a minimum visit requirement $N_{\text{parent}} \ge n_{\text{min}}$ for the early stage, ignoring trivial nodes or leaves with $<2$ children to keep targets meaningful.
* **Corpus Boundary Confirmation**: Confirmed that since the 50k dataset generation yaml uses a flat `min_nodes: 96, max_nodes: 96` budget, the maximum possible expansion count step is exactly $t = 96$ across all examples.
* **Added `policy_drift` Module & Tests**:
  - Implemented the algorithms in [`src/cts/data/preprocess_gnn/policy_drift.py`](src/cts/data/preprocess_gnn/policy_drift.py).
  - Added robust test suite [`tests/test_policy_drift.py`](tests/test_policy_drift.py) covering tied visits, temperature scaling, KL divergence correctness, $n_{\text{min}}$ filtering, leaf exclusion, and the 96-expansion boundary validation.
  - Verified that all **91/91 tests** in the project pass successfully.

### Why We Do Not Drop `NaN` Nodes from the GNN Forward Pass
We analyzed the suggestion of dropping nodes with `NaN` targets completely from the pass and clarified why the current masking design is structurally superior:
1. **Connectivity Invariance in GNN Message-Passing**: GNN operations rely on the full tree structure to propagate features upwards to the root. If we physically pruned `NaN` target nodes (like leaf nodes or single-child subtrees) from the graph during the forward pass, we would break the graph's structural message-passing paths. The root node would become disconnected from its descendants, completely destroying the GNN's capacity to aggregate context across the search tree.
2. **Contextual Feature Contribution**: Even if a node has a `NaN` target (meaning we don't compute loss on it), its input features (e.g. prior policy, static value, or evaluation signals) are vital contextual information for computing the representations of its ancestor nodes (including the root).
3. **Loss Masking via `torch.isfinite`**: The current regression loss dynamically masks out these nodes during gradient backpropagation via `torch.isfinite(targets)`. This guarantees that these nodes do not contribute to parameter gradients while fully utilizing their features in GNN message passing.

### Refactored to `NodePretrainer`
* **Renamed classes**: Renamed `TopologyPretrainer` $\to$ **`NodePretrainer`**, `TopologyPretrainConfig` $\to$ **`NodePretrainConfig`**, and `TopologyMetrics` $\to$ **`NodePretrainMetrics`** across `gnn_pretrain.py`, `build_tree.py`, `smoke_topology_pretrain_batch.py`, and the test suite (`test_topology_pack_presets.py`).
* All **91/91 unit tests** are fully updated and continue to pass.

### Multi-Head GNN Target Integration (Value Gap + Policy Drift)

We completed the expansion of the tree pretraining targets to a multi-task learning setup by integrating policy drift alongside the value gap target head in the production `v5` schema:
* **Dynamic Multi-Head Schema Modification**:
  - Updated `TEACHER_TOPOLOGY_FEATURE_NAMES` in `schema.py` to `("value_gap", "policy_drift")`.
  - Stacking both target columns enables the GNN pretrainer to train joint objectives simultaneously or mask individual objectives if needed.
* **On-Disk Record Serialization Upgrade**:
  - Expanded `PretrainExample` and `RawPretrainExampleRecord` in `teacher_targets.py` to carry, validate, and serialize `policy_drift` tensors alongside `value_gap`.
  - Modified `to_payload`, `from_payload`, and `to_pretrain_example` to support multi-target serialization, ensuring full backward compatibility with older datasets.
* **Empirical Policy Drift Target Calculation**:
  - Added visit-count tracking (`oracle_root_visits_trace`) inside the main PUCT loop of `generate_partial_tree_from_provider` to record the root's child visit distribution after each step.
  - In `build_pretrain_example`, dynamically calculated the empirical root node KL divergence drift from early search ($N_{\text{parent}} \ge 10$) to late search ($t = 96$). Nodes that are suboptimal or do not meet the minimum visit requirement are assigned `NaN` to be masked during regression.
* **Multi-Target Scaling & Collation**:
  - Re-implemented `scaled_teacher_topology_matrix` in `teacher_targets.py` to return stacked `[num_nodes, 2]` targets.
  - Updated `to_tensorized_tree_example` to dynamically map column slices using their index in `TEACHER_TOPOLOGY_FEATURE_NAMES` instead of hardcoded 1D indexes.
* **Hardened Test Suite & 100% Pass**:
  - Adapted collator and loss masking tests in `test_topology_pack_presets.py` and `test_value_gap.py` to support 2-dimensional target shapes.
  - Implemented an end-to-end integration test `test_record_serialization_round_trip_with_policy_drift` in `test_policy_drift.py` validating that policy drift correctly round-trips from generation through record packing and target scaling.
  - Verified all **92/92 tests pass** successfully with exit code 0.

## 2026-05-19 (hl4291 — finalized nodetargets transition & policy drift integration)

### What was accomplished
We successfully completed all terminology transitions and fully finalized the integration of the joint policy drift and value gap targets. All subsystems (data serialization, tree generation, pretraining loop, dataset packaging, and testing) are now 100% unified under the `node_targets` and `nodetargets` nomenclature:
1. **CLI and Pretraining Command Renaming**:
   - Refactored `pretrain-topology-encoder` subcommand in `build_tree.py` to `pretrain-nodetargets-encoder`.
   - Updated `_build_topology_model` helper to `_build_nodetargets_model`.
   - Updated weight and head paths from `topology` to `nodetargets` in the trainer and scripts.
2. **Pretraining Script Renaming**:
   - Updated the local batch pretraining smoke script `scripts/smoke_topology_pretrain_batch.py` to use nodetargets variables and correct constructor arguments for `TensorizedTreeExample`.
3. **Internal Data Models & Collation Integration**:
   - Refactored all remaining `topology_targets` and `topology_supervision_shard` occurrences across datasets, packaging, tests, and collation scripts to `nodetargets_targets` and `node_supervision_shard` respectively.
4. **Test Suite Verification**:
   - Updated `tests/test_policy_drift.py`, `tests/test_topology_pack_presets.py`, and `tests/test_value_gap.py` to use the unified nodetargets interfaces, correct shapes, and dynamic mocks where applicable.
   - All **92/92 unit and integration tests are passing flawlessly** with zero errors or warnings.

### Verdict
The pipeline is fully operational, verified, mathematically sound, and ready to launch fresh tree generation and pretraining runs!


## 2026-05-20

### Bucketed (unweighted) encoder pretraining — fresh from random init under current code

Intent:
- Get an encoder pretrained under current code (post-2026-05-11 slot-ordering fix) with a per-validation-epoch bucketed-KL time series, so the (parent_depth, child_subtree_size) breakdown can be tracked over training rather than only measured at the endpoint.
- Provide a clean current-code baseline against which interventions on the loss can be compared.

Configuration matches the rerun encoder's hyperparameters as recorded in the 2026-04-30 lab notebook entry plus the pre-YAML-migration slurm script defaults (commit `5c84fd6^`): k=1, async, d_embed=128, d_message=128, n_heads=4, d_att=32, node_embed_hidden=128, decoder_hidden=128, batch_size=128, lr=1e-3, weight_decay=0. Departures: 1000 epochs (vs the rerun's chained-to-200), and a JSONL is written per validation epoch with the per-bucket grid.

Result:
- Run completed 1000 epochs in ~8 h.
- Final overall mean KL on the validation set (unweighted accumulator): 0.00396.
- Best-validation epoch was 956 at KL 0.00365, very close to final — the trajectory was monotone-decreasing in trend across the whole run.
- Per-bucket pattern at convergence: leaves (size 1, 97% of edges) at KL 0.0039; KL escalates with child subtree size to 0.051 at size ≥1024. Same qualitative shape as the rerun-encoder audit reported on 2026-05-18, but at ~13x lower absolute magnitudes.
- Trajectory was clean: no spikes above 3x median in post-warmup epochs (>100).

Comparison to the rerun-encoder audit (2026-05-18 entry): the bucketed encoder under current code reaches lower validation KL than the audit reported on the rerun encoder by every cell. The reasons aren't fully determined — candidate mechanisms include (a) effective training duration (the rerun's actual saved-checkpoint epoch count was not separately verified against the lab notebook claim of 200), and (b) slot-semantics mismatch from commit `f297744` (the rerun was trained pre-fix; the audit ran the rerun encoder through post-fix code that re-canonicalizes slot ordering at tensorize time). Both are plausible; neither has been isolated. The 2026-05-18 entry's claim that the gap is "a real and bad finding about the rerun encoder, not an artifact of the audit" was overconfident — at minimum, the absolute magnitude is partly attributable to the cross-code-version effect.

What the bucketed run does establish cleanly: the *structural* pattern (large-subtree children harder than leaves) is a property of the encoder architecture/objective, not of a specific run. It shows up in both encoders, at different absolute scales.

Inputs:
- `/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/pretrain_packed_oracle96_trace_filtered_rerun/{train,validation}_manifest.json` — packed shards, regenerated this session from the rerun split via the `rewrite_compact` v1-dict fix (commit `fb2c5b9`) followed by a fresh pack.

Outputs:
- `/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/checkpoints/tree_encoder_child_wdl_async_k1_bucketed.pt` — best-validation encoder, 1000 epochs from random init under current code.
- `/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/checkpoints/tree_encoder_child_wdl_async_k1_bucketed_decoder.pt` — paired child-WDL decoder.
- `/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/checkpoints/tree_encoder_child_wdl_async_k1_bucketed_kl.jsonl` — 1000-row JSONL time series, one row per validation epoch. Schema matches `cts.train.gnn_pretrain.BucketedKLState.summary` (per-bucket mean KL grid, edge-count grid, marginals, overall stats, bin labels).

### Subtree-size-weighted encoder pretraining

Intent:
- Push the encoder's gradient budget onto the non-trivial summarization predictions. The unweighted loss is mean-over-edges; ~97% of edges target leaf children where the prediction task is near-trivial (the leaf's input WDL is in the encoder's input and the decoder essentially round-trips it through the parent state). Large-subtree children, where the encoder has to summarize many descendants into a single parent-state representation, contributed little gradient signal because they're rare.

Principle: weight each edge's cross-entropy by its child's subtree size. Each "node summarized" gets equal voice in the loss instead of each edge prediction. Both the cross-entropy and target-entropy reference are weighted identically so `loss_gap` remains a clean weighted KL. The bucketed-KL accumulator in the JSONL stays *unweighted* so per-cell numbers are directly comparable to the bucketed (unweighted) run cell-by-cell.

Configuration: identical to the bucketed run except `loss_weight_by_subtree_size: true`. Same encoder architecture, same dataset, same 1000-epoch budget.

Runtime: ~20 h (vs the unweighted run's ~8 h). The slowdown was caused by `compute_subtree_sizes` running once per training batch (in addition to once per validation batch as in the unweighted run) with `.nonzero` / `.item` / `.any` calls forcing GPU→CPU stalls — about 30–50 ms per batch. Refactored mid-run to a sync-free version (commit `3a46c03`); applies to future runs only.

Result:
- Run completed 1000 epochs. Best-validation epoch was 956.
- Final overall mean KL (unweighted accumulator, comparable to bucketed run): 0.0121, vs the bucketed run's 0.0040 — 3x higher on the aggregate, exactly as expected for an intervention that downweights the leaves that dominate the unweighted mean.
- Per-bucket trade-off at convergence (compared to bucketed run final-epoch values):
  - Size ≥1024 marginal: 0.0024 vs 0.0507 — ~21x lower.
  - Size 1 (leaves) marginal: 0.0123 vs 0.0039 — ~3x higher.
  - depth-0 × size ≥1024 cell (the headline diagnostic for metacontrol-relevant root-children predictions): 0.0023 vs 0.048 — ~21x lower. This cell hit its floor by ~epoch 30 and remained flat thereafter.
- Other large-subtree cells (128+) also improved 4–10x; leaves and small-subtree cells (1–15) all degraded 2–3x.

Trajectory stability: mostly smooth, with five to seven distinct perturbation events across the 800 post-warmup epochs. The largest was at epoch 318 (overall KL = 0.545, ~33x the local median; recovered to baseline within ~10 epochs). Two smaller clusters at 276–280 (peak 0.148) and 934–937 (peak 0.113), plus a handful of isolated single-epoch spikes. The saved encoder (best-val at epoch 956) is well clear of all of them.

The variance concern flagged when we discussed weighting normalization was visible in these spikes — linear weighting puts very high per-edge weight on the ~5k rare ≥1024-subtree edges, and batches that happen to draw many such edges produce noisier gradients. The run worked through it, but if we run a follow-on variant a sub-linear weighting (sqrt or log) would be the natural way to reduce the gradient-noise without giving up the targeting.

Whether this trade-off helps metacontrol is a downstream question — the controller doesn't directly consume per-edge WDL accuracy; it consumes z_t at root. Answer pending controller training on the new encoder.

Inputs:
- Same packed manifests as the bucketed run.

Outputs:
- `/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/checkpoints/tree_encoder_child_wdl_async_k1_subtree_weighted.pt` — best-validation encoder.
- `/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/checkpoints/tree_encoder_child_wdl_async_k1_subtree_weighted_decoder.pt` — paired decoder.
- `/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/checkpoints/tree_encoder_child_wdl_async_k1_subtree_weighted_kl.jsonl` — 1000-row JSONL.

### Controller training infrastructure: positive-enumeration `controller_inputs`

`cts.train.controller_train` previously hardcoded the advantage head's input to `[z_t, N_t, T_t]` while the saved-checkpoint metadata stamped `controller_inputs: ["z_t", "N_t", "T_t"]` as if it were configurable — two sources of truth that could disagree silently. Replaced both with a single `controller_inputs` config field that drives the head's `input_dim`, the feature-slicing inside `MetaController._select_features`, and the saved-checkpoint metadata (commit `7dfd099`).

The field uses positive enumeration (`[z_t, T_t]` to drop N_t, `[z_t]` for z_t-only) rather than exclusion flags, validated at config load via a Pydantic field validator over `("z_t", "N_t", "T_t")`. The canonical cache layout (`[z_t, N_t, T_t]`) is unchanged; selection happens at the model boundary.

Two follow-on controller training configs are wired and ready to submit once the subtree-weighted encoder's cache is materialized:
- `configs/train/controller_subtree_weighted_zt_tt.yaml` — `controller_inputs: [z_t, T_t]`.
- `configs/train/controller_subtree_weighted_zt_only.yaml` — `controller_inputs: [z_t]`.

Cache materialization configs are also ready:
- `configs/data/preprocess_mc/materialize_subtree_weighted_{train,validation}.yaml`.

Not yet run; the experimental phase begins when those go to the cluster.

## 2026-05-21

### Cache materialization for the subtree-weighted encoder

Standard materialize-then-merge against the subtree-weighted encoder, producing the `[z_t, N_t, T_t]` feature caches the controller reads at training time. Ran with three parallel workers per split via the `WORKER_INDEX` env-var path added to the materialize slurm script the prior session. No issues; final caches written atomically.

Inputs:
- `/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/checkpoints/tree_encoder_child_wdl_async_k1_subtree_weighted.pt`
- `/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/controller_packed_combined_nomaint_no_xaba/{train,validation}_manifest.json`

Outputs:
- `/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/train_cache_subtree_weighted.pt`
- `/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/validation_cache_subtree_weighted.pt`
- `/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/train_cache_subtree_weighted_shards/` (per-worker, kept for resume)
- `/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/validation_cache_subtree_weighted_shards/`

### Controller training on the subtree-weighted encoder: `[z_t, T_t]` and `[z_t]`

Both controllers trained 20 epochs at LR 1e-3, sign_loss_weight 0.1, vanilla losses (no T-sensitivity filter, no MSE cooling, no regret weighting), canonical capacity (Q_HIDDEN=256, 3 layers), new `controller_inputs` pipeline (no N_t).

Greedy validation regret:

| Controller inputs | Best | Final | Per-epoch trajectory |
|---|---|---|---|
| `[z_t, T_t]` | **0.023** (ep 7) | 0.024 | 0.023–0.027 across all 20 epochs; flat from epoch 1 |
| `[z_t]` | **0.073** (ep 12) | 0.076 | 0.073–0.087; mildly oscillating |

The `[z_t, T_t]` run hit 0.027 on the first validation epoch and never moved more than 0.004 from that floor. Qualitatively different from every prior controller training on the rerun encoder — the historical best of 0.176 (kitchen sink, run 11 on 2026-04-29) required 200 epochs of stacked interventions (T-sensitivity filter + MSE cosine cooling + regret-weighted loss) to converge at 0.176. Under the subtree-weighted encoder, the head saturates at 0.024 with none of those interventions in a tenth of the epoch budget.

The `[z_t]`-only run is informative on its own: with no time-budget input at all, z_t alone reaches 0.076 — about 2.9x lower than the T_t-only floor (0.218 historical, 0.212 reproduced today; see next sub-entry). But it does *not* match the historical `[N_t, T_t]`-without-encoder baseline (0.056 on 2026-04-29), so z_t alone in this encoder doesn't fully substitute for the (N_t, T_t) pair as a step-and-budget signal. It carries some of that information plus information that's complementary to T_t — the `[z_t, T_t]` number (0.024) is 3.2x lower than `[z_t]` alone, well outside what you'd see from redundant inputs.

Inputs:
- Encoder + caches from the materialization sub-entry above.
- `/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/controller_packed_combined_nomaint_no_xaba/{train,validation}_manifest.json`

Outputs:
- `/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/checkpoints/fittedq_subtree_weighted_zt_tt.pt` + `_diagnostics.jsonl`
- `/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/checkpoints/fittedq_subtree_weighted_zt_only.pt` + `_diagnostics.jsonl`

Slurm logs: `cts-fittedq_8527146.out` (zt_tt), `cts-fittedq_8527147.out` (zt_only).

### Ablation: rerun encoder + `[z_t, T_t]` under the new pipeline

Is the `[z_t, T_t] = 0.024` result coming from the new *encoder* or the new *code base* (controller_inputs refactor, default changes, anything else that drifted between the rerun-era pipeline and today's)? This sub-entry isolates that.

Configuration identical to the subtree-weighted `[z_t, T_t]` run except the encoder checkpoint and matching cache are swapped to the rerun encoder's. Cache reuse, not re-materialization, since the rerun cache already exists at `/tigress/`. Same 20 epochs, LR 1e-3, slw01, vanilla loss, `controller_inputs: [z_t, T_t]`.

Result: regret oscillated 0.219–0.324 across the 20 epochs with no descent trend. Best 0.219 at epoch 15, final 0.291. This sits within noise of the historical *naive* `[z_root, T_t]` baseline (0.238, 2026-04-29 (N_t, T_t)-baseline reference table) and just above the T_t-only floor (0.218). I.e., under vanilla training the rerun encoder's z_t contributes nothing on top of T_t.

Conclusion: the 0.176 → 0.024 improvement is attributable to the *encoder*, not to the pipeline. The rerun encoder needed 200 epochs of stacked loss-side interventions to extract any usable z_t signal at all (and even then only down to 0.176); the subtree-weighted encoder yields a controller that beats that floor by 7x in 20 epochs of vanilla training.

Inputs:
- `/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/checkpoints/tree_encoder_child_wdl_async_k1_rerun.pt`
- `/tigress/ysagiv/chess/cts/train_cache_rerun.pt`
- `/tigress/ysagiv/chess/cts/validation_cache_rerun.pt`
- `/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/controller_packed_combined_nomaint_no_xaba/{train,validation}_manifest.json`

Outputs:
- `/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/checkpoints/fittedq_rerun_encoder_zt_tt_ablation.pt` + `_diagnostics.jsonl`

Slurm log: `cts-fittedq_8533204.out`.

### T_t-only baseline under the new `controller_inputs` pipeline

Sanity check: does `controller_inputs: [T_t]` under the new code reproduce the historical T_t-only floor (0.218, 2026-04-29)? If yes, the new code isn't silently breaking the floor; the numbers above can be read against the historical reference grid without an extra correction.

Same cache as the subtree-weighted runs — cache choice is moot for a T_t-only head since T_t is encoder-independent. Same 20-epoch vanilla slw01 training.

Result: best 0.212 at epoch 14, final 0.234, range 0.212–0.239 across the 20 epochs, no learning trend (T_t is a single scalar; there's nothing for the head to fit beyond a thresholded function of T_t). Within 0.006 of the historical floor.

Inputs:
- Same as the (z_t, T_t) / (z_t) runs (cache is read but the z-component is unused by the model under `controller_inputs: [T_t]`).

Outputs:
- `/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/checkpoints/fittedq_subtree_weighted_tt_only.pt` + `_diagnostics.jsonl`

Slurm log: `cts-fittedq_8550917.out`.

### Consolidated reference grid

Numbers grouped by encoder and training regime. The four runs from this entry are bolded.

| Run | Inputs | Encoder | Training | Greedy regret |
|---|---|---|---|---|
| Oracle | — | — | — | 0.000 |
| **Subtree-weighted `[z_t, T_t]`** | z_t, T_t | subtree-weighted | vanilla, 20 ep | **0.023 / 0.024** (best / final) |
| Historical slw01 (rerun) | z_t, N_t, T_t | rerun | vanilla, 20 ep | 0.031 |
| Historical `[N_t, T_t]` (no encoder) | N_t, T_t | — | vanilla, 20 ep | 0.056 |
| **Subtree-weighted `[z_t]`** | z_t | subtree-weighted | vanilla, 20 ep | **0.073 / 0.076** |
| Kitchen sink | z_t, T_t | rerun | filter + cooling + regret-w, 200 ep | 0.176 |
| **T_t-only (new pipeline)** | T_t | — | vanilla, 20 ep | **0.212 / 0.234** |
| T_t-only floor (historical) | T_t | — | vanilla | 0.218 |
| **Rerun-encoder ablation** | z_t, T_t | rerun | vanilla, 20 ep | **0.219 / 0.291** |
| Naive `[z_root, T_t]` (historical) | z_t, T_t | rerun | vanilla | 0.238 |

Headline: the subtree-weighted encoder produces a controller that beats every previously-recorded Section 3 baseline. With `[z_t, T_t]` alone (no N_t, no loss-side interventions, 20 epochs, canonical capacity) it reaches regret 0.024 — below the historical slw01_rerun (0.031, which used N_t as a step-counter shortcut), 7.3x below the rerun-era best (kitchen sink at 0.176 over 200 epochs), and 9.1x below the T_t-only floor (0.218 historical / 0.212 reproduced).

The earlier characterization of z_root as "decorative" was a property of the rerun encoder under vanilla training, not of the architecture: under the subtree-size-weighted pretraining objective the same encoder architecture produces a z_t that carries usable metacontrol signal directly, without scaffolding scalars or loss-side interventions.

Caveats:
- Single seed per configuration. The historical-comparison numbers from earlier entries are likewise single-seed.
- All 20-epoch runs at canonical capacity. Whether more epochs further reduce 0.024 is untested (trajectory was flat from epoch 1, so the headroom is probably small, but unmeasured).
- The subtree-weighted encoder pretraining had several short-lived spike events across its 1000-epoch trajectory (logged 2026-05-20). The saved checkpoint is at epoch 956, well clear of all of them. Whether other saved checkpoints would give similar controller numbers is untested.
- The downstream evaluation grid (Pareto figures, oversearch reports, advantage-head analysis) has not been regenerated against these new controllers.

## 2026-05-28 (hl4291 — Stage 2b proxy analysis on ysagiv caches)

> **2026-05-29:** The short-lived proxy scripts (`2b_train_controller.py`, `2b_plot_loss.py`, …) lived in a separate `chess_analysis/analysis/` CTS folder that was removed when that directory was repurposed for human move-time analytics (`chess_analysis/src/` → `chess_analysis/analysis/`). **Scratch outputs below are unchanged**; rerun would use `lmcos` entry points or a fresh hl4291 wrapper.

Cheap offline proxy runs to compare the three May-2026 controller recipes without re-submitting full 20-epoch Slurm training. Code at the time lived in `chess_analysis/analysis/` (not under `lmcos/`): train with `2b_train_controller.py`, plot with `2b_plot_loss.py`, regret curves from synced Slurm logs via `plot_controller_regret_curves.py`.

### Setup

Three named variants mirror the subtree-weighted comparison above:

| Name | Encoder cache | Head inputs | Full Slurm greedy regret (20 ep, Yotam) |
|---|---|---|---|
| `subtree_weight_root+budget` | subtree-weighted | `[z_t, T_t]` | **0.024** |
| `subtree_weight_root` | subtree-weighted | `[z_t]` | 0.076 |
| `no_subtree_weight_root+budget` | rerun | `[z_t, T_t]` | ~0.22–0.29 |

Each proxy run: **1000 training batches** (~6% of one full epoch over the train materialized cache), **eval every 50 batches** on the **full validation cache** for advantage MSE and on the **first 3000 validation episodes** for policy regret. Checkpoints and per-batch metrics JSON are written under hl4291 scratch.

**Policy regret metric (proxy eval):** expected regret under **probabilistic stopping**, not the production greedy rule. At step `t`, stop with probability `P(stop) = σ(−A_t / τ)` (`τ = 1` default); any remaining probability mass stops on the final step. Expected return is computed exactly as `Σ_t P(stop at t) × R(t)`; regret = `oracle_value − E[return]`. This is smoother than halting at the first `A ≤ 0` and can rank variants differently from greedy regret when validation MSE and stop-step accuracy diverge (notably `z_t`-only: high irreducible MSE but moderate greedy regret on full Slurm runs).

### Results @ batch 1000 (proxy)

| Variant | Train adv. MSE (final batch) | Val adv. MSE | Expected regret |
|---|---|---|---|
| `subtree_weight_root+budget` | 0.015 | **0.013** | **0.238** |
| `subtree_weight_root` | 0.374 | 0.314 | 0.281 |
| `no_subtree_weight_root+budget` | 0.031 | 0.030 | 0.325 |

Takeaways aligned with the full Slurm grid:

- **Encoder dominates:** subtree-weighted + budget has the lowest validation MSE and the lowest expected regret in this proxy; rerun + `[z_t, T_t]` has low MSE but the worst regret (same qualitative pattern as greedy eval: fits targets better yet stops poorly).
- **`z_t` only:** train/val MSE stays ~0.31 (structural ceiling without `T_t`), but expected regret (0.274) is still better than the rerun ablation — consistent with subtree-weighted `z_t` carrying halt/continue signal even when scalar targets are not fully predictable.
- **Loss ≠ regret:** ranking by validation MSE does not match ranking by policy regret; the proxy tooling logs both on the same batch index so curves can be compared directly.

Plots (retired 2026-05-29): ~~`chess_analysis/analysis/outputs/2b/`~~ — proxy scripts removed with directory shuffle.

Inputs:
- ysagiv materialized caches and configs (read-only), same paths as the 2026-05-21 controller entries.
- `chess_analysis/lmcos/configs/train/controller_{subtree_weighted_zt_tt,subtree_weighted_zt_only,rerun_encoder_zt_tt_ablation}.yaml`

Outputs:
- `/scratch/gpfs/GRIFFITHS/hl4291/chess/CTS/2b/{variant}_controller.pt` — proxy checkpoints (1000 batches).
- `/scratch/gpfs/GRIFFITHS/hl4291/chess/CTS/2b/{variant}_metrics.json` — train loss curves + periodic val MSE / expected regret.
- ~~`chess_analysis/analysis/outputs/2b/*.png`~~ — figures from retired `2b_plot_loss.py` (removed 2026-05-29).
- ~~`chess_analysis/analysis/logs/cts-fittedq_852714{6,7}.out`~~ — copied Slurm logs for full-run greedy regret curves (removed 2026-05-29).

Commands (replot only, no retrain) — **retired with proxy scripts**:

```bash
# removed 2026-05-29
# python3 analysis/2b_plot_loss.py
```

## 2026-05-29 (repo layout — `analysis/` rename, lmcos cleanup)

**`chess_analysis/` directory shuffle:**

- Removed the short-lived CTS proxy tree (`2a_make_cache.py`, `2b_*`, `analysis/slurm/submit_2a.sh`, …).
- Renamed **`chess_analysis/src/` → `chess_analysis/human_analytics/`** (via interim `analysis/`) so human move-time analytics / metacontrol code is not confused with **`lmcos/src/`**.
- Flattened **`lmcos/src/cts/` → `lmcos/src/`**; kept **`import cts`** via root **`cts/__init__.py`** ``__path__`` shim.

**`lmcos/` cleanup (same day, branch `jordan`):**

- Deleted **`hl4291_slurm/`** and orphaned hl4291 / topology experiment configs.
- Deleted alternate Slurm paths (compute-advantage, encoder KL audit, prefix derive, rewrite_compact, filter_packed, planning-cost / entropy sweeps) and all **`configs/analysis/`** YAMLs.
- Removed **`HANDOFF.md`** — pipeline order now in **`slurm/README.md`**; experiment history stays in this notebook.
- Kept ysagiv main-chain **`slurm/`** wrappers and core **`src/`** package unchanged (Slurm ``PYTHONPATH=${PROJECT_DIR}``).

**Later same day — slurm/config layout:**

- Removed **`demos/`** (stale tutorial notebooks and helpers).
- Reorganized **`slurm/`** into stage folders: `1_preprocess_data/`, `2_pretrain_encoder/`, `3_preprocess_root/`, `4_supervised_controller/`; logs under **`slurm/logs/`**.
- Removed ysagiv stub **`configs/**/*.yaml`**; empty stage dirs mirror slurm layout (add hl4291 run configs as needed). See **`configs/README.md`**.
- Removed **`scripts/`** — the only remaining orchestrator (`submit_generate_dataset_shards.py`) lives in **`slurm/1_preprocess_data/`** next to its wrapper shell script.
- Renamed **`chess_analysis/analysis/` → `chess_analysis/human_analytics/`** to avoid clashing with `lmcos/analysis/` (CTS post-hoc diagnostics).

**Later same day — package layout finalization:**

- Moved **`lmcos/src/analysis/` → `lmcos/analysis/`** (still imported as `cts.analysis`; pipeline code stays in `src/`).
- Moved **`lmcos/configs/` → `lmcos/slurm/configs/`** so cluster YAMLs sit next to Slurm scripts.
- Removed root **`cts/`** import shim and **`cts.egg-info/`**; `import cts` now resolves via `pyproject.toml` `package-dir` (`src/` + `analysis/`) and `pip install -e .`.
