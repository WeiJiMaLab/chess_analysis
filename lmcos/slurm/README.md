# Slurm scripts

Cluster wrappers for the CTS pipeline on della. Each script takes a `CONFIG` env var pointing at a YAML under `slurm/configs/<stage>/`.

**Logs:** `slurm/logs/` — flat Slurm stdout/stderr (gitignored).

**Outputs:** `slurm/outputs/<stage>/` — flat per-stage artifacts (tracked in git). Stage 4 example: `<run>.yaml`, `<run>.png`, `comparison.png` under `slurm/outputs/4_supervised_controller/`.

## Stage folders

### `1_preprocess_data/`

| Script | Module |
|--------|--------|
| `sample_root_fens_della.slurm` | `cts.data.sample_fens` |
| `generate_dataset_shard.slurm` | `cts.data.build_tree` (`generate-dataset`) |
| `submit_generate_dataset.sh` | Local orchestrator → `submit_generate_dataset_shards.py` (slices FEN YAML, submits N shard jobs) |
| `prepare_pretrain_split_della.slurm` | `cts.data.preprocess_gnn.split` |
| `pack_pretrain_examples_della.slurm` | `cts.data.preprocess_gnn.pack` |

Tree generation example:

```bash
cd /home/hl4291/chess_analysis/lmcos
mkdir -p slurm/logs slurm/outputs/4_supervised_controller
export CONFIG="$PWD/slurm/configs/1_preprocess_data/build_tree.yaml"
./slurm/1_preprocess_data/submit_generate_dataset.sh --config "$CONFIG" --submit
```

### `2_pretrain_encoder/`

| Script | Module |
|--------|--------|
| `pretrain_child_wdl_encoder_della.slurm` | `cts.data.build_tree` (`pretrain-child-wdl-encoder`) |

### `3_preprocess_root/`

| Script | Module |
|--------|--------|
| `pack_controller_episodes_della.slurm` | `cts.data.preprocess_mc.pack` |
| `materialize_controller_cache_della.slurm` | `cts.data.preprocess_mc.materialize` / `merge` |

Materialize uses parallel workers via `WORKER_INDEX` env override (see script comments).

### `4_supervised_controller/`

| Script | Module |
|--------|--------|
| `controller_train_supervised.slurm` | `cts.train.controller_train` |
| `submit_configs.sh` | Submit all configs in `slurm/configs/4_supervised_controller/` in parallel |

#### Ablation configs (hl4291, ysagiv read-only caches)

| YAML | Display name | Encoder | Controller inputs |
|------|--------------|---------|-------------------|
| `legacy_root_budget.yaml` | `legacy[root+budget]` | rerun async k=1 | `[z_t, T_t]` |
| `subtree_weighting_root_budget.yaml` | `subtree-weighting[root+budget]` | subtree-weighted async k=1 | `[z_t, T_t]` |
| `subtree_weighting_root.yaml` | `subtree-weighting[root]` | subtree-weighted async k=1 | `[z_t]` |

All three use `epochs: 3`, `batch_size: 18000`, `metrics_log_interval: 20`, `validation_step_interval: 100`, and `greedy_eval_step_interval: 100`. `ControllerTrainMetricsLogger` writes `slurm/outputs/4_supervised_controller/<run>.yaml` and refreshes `<run>.png` every 100 steps.

Submit all configs in parallel (comparison plot auto-submits with `afterok` when training finishes):

```bash
cd /home/hl4291/chess_analysis/lmcos
export VENV_DIR=/home/hl4291/venv
./slurm/4_supervised_controller/submit_configs.sh
# → slurm/outputs/4_supervised_controller/comparison.png
```

Slurm stdout/stderr land in `slurm/logs/` (job-level `%x_%j.out`).

## Environment

Set for hl4291 runs:

```bash
export PROJECT_DIR=/home/hl4291/chess_analysis/lmcos
export VENV_DIR=/home/hl4291/venv
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:$PYTHONPATH}"
```

See [`labnotebook.md`](../../labnotebook.md) and [`reports/`](../../reports/) for experiment history and [`../../README.md`](../../README.md) for workspace overview.
