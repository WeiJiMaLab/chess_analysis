# CTS — Handoff

Compute Tree Search: a meta-controller for chess MCTS-style search that learns
*when to stop searching*. Given a partially-expanded search tree and a
remaining time budget, a small neural network predicts halt vs continue. The
controller is fit by regression onto a budgeted oracle that simulates the
policy a perfect search-stopper would have used.

## 1. Repo layout

```
src/cts/
  core/                  shared substrate
    tree.py                SearchTree (append-only, UCI-lex slot ordering)
    tensorizer.py          tree → flat tensor batch
    schema.py              node feature schema
    providers/             UCI/lc0 engine integration
  data/                  data generation + preprocessing
    sample_fens.py         draw root FENs from Lichess
    validate_fens.py       PUCT-stability filter on FENs
    build_tree.py          PUCT-bounded tree generation via lc0
    filter_packed_episodes.py
    preprocess_gnn/        encoder-pretraining target chain (split, pack)
    preprocess_mc/         controller-training target chain (pack, materialize)
  models/                neural network modules
    gnn.py                 TreeEncoder + child-WDL head
    tree_mha.py            slot-conditioned multi-head attention
    mc.py                  MetaController (encoder + advantage MLP)
  train/                 training loops
    gnn_pretrain.py        encoder pretraining (child-WDL targets)
    controller_train.py    fitted-Q controller training
  analysis/              diagnostics, plots, evaluation tools
    _budgeted/             themed sub-modules for the main analyzer
    _common.py             shared regex/parse helpers
configs/                 YAML configs (one per entry point + variants)
slurm/                   SLURM job scripts (della-specific defaults; adjust
                         module loads + paths for other clusters)
scripts/                 orchestrators (just submit_generate_dataset_shards.py)
tests/                   pytest suite (~60 tests, run with `pytest tests/`)
LAB_NOTEBOOK.md          running experimental log; chronological
data/                    sampled_root_fens_2023.txt and similar
demos/                   April-era tutorial notebooks (some are stale — see §4)
```

## 2. Setup

Requirements:
- Python 3.10+
- PyTorch, pydantic v2, PyYAML, python-chess, matplotlib
- lc0 with a weights file — only needed for tree generation (stages 1–3 below).
  Training and analysis don't need lc0.
- pytest (for the test suite)

Install:
```
pip install -e .
```
The `pyproject.toml` declares the package and puts `src/cts/` on the import
path. `conftest.py` does the equivalent for `pytest`. Run tests with
`pytest tests/`; 59 should pass.

## 3. The config + sbatch interface

Every entry point reads a Pydantic-validated YAML config:
```
python3 -m cts.<module> --config configs/.../some.yaml
python3 -m cts.<module> --config configs/.../some.yaml --override seed=42 \
                                                       --override epochs=10
```
Override keys support dot-notation for nested fields. Values are parsed as
YAML scalars (`true` → bool, `42` → int, etc.). Unknown fields are rejected
at load time — typos in YAML keys fail loudly rather than silently.

Every SLURM script collapses to:
```
python3 -m cts.<module> --config "${CONFIG}"
```
…driven by a single `CONFIG` env var, e.g.
```
sbatch --export=ALL,PROJECT_DIR=$HOME/path/to/repo,CONFIG=configs/data/build_tree.yaml \
  slurm/generate_dataset_shard.slurm
```
SLURM scripts default to the della cluster's modules (`anaconda3/2023.3`,
`cudatoolkit/12.8` on GPU jobs) and a `CTS` conda env. Adjust the module
load section for a different cluster.

Variant experiments: copy the base YAML to a sibling
(`configs/train/controller_train/<variant>.yaml`) and tweak. The diff
between two experiments is the diff between their YAMLs.

## 4. Pipeline (in order)

Each stage waits on its predecessor (data dep). Sample YAMLs are in
`configs/`; the cluster-side commands below assume the della layout —
update `PROJECT_DIR` and scratch paths in the YAMLs to match your env.

| # | Stage | Slurm script | Default YAML | Module |
|---|---|---|---|---|
| 1 | sample root FENs | `slurm/sample_root_fens_della.slurm` | `configs/data/sample_fens.yaml` | `cts.data.sample_fens` |
| 2 | validate FENs (PUCT stability filter) | (runs in-line, no slurm) | `configs/data/validate_fens.yaml` | `cts.data.validate_fens` |
| 3 | build trees (lc0 + PUCT) | `slurm/generate_dataset_shard.slurm` (via orchestrator) | `configs/data/build_tree.yaml` | `cts.data.build_tree` |
| 4a | split train/val | `slurm/prepare_pretrain_split_della.slurm` | `configs/data/preprocess_gnn/split.yaml` | `cts.data.preprocess_gnn.split` |
| 4b | pack pretrain shards | `slurm/pack_pretrain_examples_della.slurm` | `configs/data/preprocess_gnn/pack.yaml` | `cts.data.preprocess_gnn.pack` |
| 5 | encoder pretrain (child-WDL) | `slurm/pretrain_child_wdl_encoder_della.slurm` | `configs/data/pretrain_encoder_smoke.yaml` (template) | `cts.data.build_tree` (`command: pretrain-child-wdl-encoder`) |
| 6 | pack controller episodes (budgeted oracle) | `slurm/pack_controller_episodes_della.slurm` | `configs/data/preprocess_mc/pack.yaml` | `cts.data.preprocess_mc.pack` |
| 7a | materialize encoder cache | `slurm/materialize_controller_cache_della.slurm` | `configs/data/preprocess_mc/materialize.yaml` | `cts.data.preprocess_mc.materialize` (`command: materialize`) |
| 7b | merge worker caches | same slurm | same module | same module (`command: merge`) |
| 8 | controller train (fitted-Q) | `slurm/train_fitted_q_controller_della.slurm` | `configs/train/controller_train.yaml` | `cts.train.controller_train` |

### Stage 3 orchestrator
Build_tree is the only stage with an orchestrator (because lc0 is slow and
shard parallelism is essential). The orchestrator reads a base YAML, slices
the FEN file into shards, emits a per-shard YAML overriding `start_index`/
`end_index`, and sbatches one job per shard:
```
python3 scripts/submit_generate_dataset_shards.py \
  --config configs/data/build_tree.yaml \
  --project-dir $HOME/path/to/repo \
  --shard-size 2253 --time 06:00:00 --submit
```
Drop `--submit` for a dry-run (writes per-shard YAMLs + prints sbatch
commands without queuing).

### Stage 7 (materialize) two subcommands
Materialize runs the frozen encoder over the packed controller episodes to
cache embeddings. Two subcommands:
- `command: materialize` — one worker writes its shards to
  `output_dir/worker_XX/`. Run `num_workers` of these in parallel.
- `command: merge` — combines all worker dirs into a single `final_cache`
  .pt file. Run once after all materialize jobs complete.

For train + validation you run materialize+merge separately (4 jobs total
in serial mode, or 2 parallel pairs).

### Optional / non-standard stages
- `cts.data.preprocess_gnn.derive_prefixes` — samples variable-size root
  prefixes from full 96-node trees. Not currently in the standard chain
  (production runs split → pack directly on full trees).
- `cts.data.filter_packed_episodes` — filters packed controller episodes
  by some criterion. Used selectively.
- `cts.data.preprocess_mc.merge` (the *pack* merge, not the materialize
  merge) — merges multiple packed controller episode dirs.

### Demos
`demos/` is April-era and partially stale (tutorial 05 imports a
`HaltController` class that was removed in the refactor). Tutorials 01–04
should still be useful as orientation to the SearchTree / encoder /
provider interfaces; tutorial 05 needs rewriting against the new
`MetaController` interface before it can be run.

## 5. Reproducing a known-good run

To verify the install works end-to-end on a tiny dataset (called a "smoke"
in our lab notebook), every pipeline stage has a sibling `*_smoke.yaml`
that points at smoke-suffixed scratch directories and a 10-FEN input file.
Suggested order:
1. `head -n 10 /path/to/sampled_root_fens.txt > /path/to/sampled_root_fens_smoke.txt`
2. Update `fens:` in `configs/data/build_tree_smoke.yaml` to point at the
   smoke FEN file.
3. Run stages 3 → 4a → 4b → 5 → 6 → 7a → 7b → 8 with the corresponding
   `*_smoke.yaml` configs. Each stage should complete in a few minutes.

The end state is `fittedq_smoke.pt` + `fittedq_smoke_diagnostics.jsonl`
under the checkpoints dir. The diagnostics .jsonl has per-epoch loss
curves and validation metrics — the controller's `train_total_loss`
should trend downward.

## 6. Notes for adapting to a different cluster

The SLURM scripts in `slurm/` are della-specific in:
- Module loads (`anaconda3/2023.3`, `cudatoolkit/12.8` on GPU jobs).
- Conda env name (`CTS`).
- Default `PROJECT_DIR=/home/ysagiv/chess/cts/...`.
- Constraint flags (`--constraint=nomig` on tree generation).

For another cluster, edit the module load section + the `PROJECT_DIR`
default. The python invocation (`python3 -m cts.X.Y --config "${CONFIG}"`)
stays the same. For non-SLURM environments, invoke the python module
directly with `--config`.

## 7. Project-standard parameters

Recorded for reproducibility:
- Tree generation: `SEARCH_BUDGET=64`, `MIN_NODES=96`, `MAX_NODES=96`
  (fixed 96-node trees), `MAX_DEPTH=30`, `MULTIPV=8`.
- Encoder: `k=1`, `node_embed_hidden=d_embed=d_message=128`, `n_heads=4`,
  `d_att=32`, `decoder_hidden=128`.
- Controller training default: `sign_loss_weight=0.1` (slw01), 20 epochs,
  `batch_size=1024`, `learning_rate=1e-3`, cosine LR (default).

See `LAB_NOTEBOOK.md` for the experimental history; entries are dated and
include rationale, command, and outcome for each meaningful run.
