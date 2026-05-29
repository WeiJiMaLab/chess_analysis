# Slurm scripts

Cluster wrappers for the CTS pipeline on della. Each script takes a `CONFIG` env var pointing at a YAML under `slurm/configs/<stage>/`.

**Logs:** `slurm/logs/` (gitignored except `.gitkeep`).

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
mkdir -p slurm/logs
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
| `train_fitted_q_controller_della.slurm` | `cts.train.controller_train` |

## Environment

Scripts default to ysagiv's della layout (`PROJECT_DIR`, conda `CTS`). Adjust module loads and `PROJECT_DIR` for hl4291 runs. Set:

```bash
export PROJECT_DIR=/home/hl4291/chess_analysis/lmcos
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:$PYTHONPATH}"
```

See [`LAB_NOTEBOOK.md`](../LAB_NOTEBOOK.md) for experiment history and [`REPO_STRUCTURE.md`](../REPO_STRUCTURE.md) for layout rationale (2026-05-29 refactor).
