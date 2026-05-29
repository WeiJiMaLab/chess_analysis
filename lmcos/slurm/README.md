# Slurm scripts (Yotam / ysagiv defaults)

Cluster wrappers for the main CTS pipeline on della with **ysagiv** paths and the **conda `CTS`** environment (`/home/ysagiv/chess/cts/async_soph`).

Each script takes a `CONFIG` env var pointing at a YAML under `configs/`.

## Pipeline order

| # | Stage | Slurm script | Module |
|---|-------|--------------|--------|
| 1 | sample root FENs | `sample_root_fens_della.slurm` | `cts.data.sample_fens` |
| 2 | validate FENs | (CLI) | `cts.data.validate_fens` |
| 3 | build trees | `generate_dataset_shard.slurm` | `cts.data.build_tree` |
| 4a | split train/val | `prepare_pretrain_split_della.slurm` | `cts.data.preprocess_gnn.split` |
| 4b | pack pretrain | `pack_pretrain_examples_della.slurm` | `cts.data.preprocess_gnn.pack` |
| 5 | encoder pretrain | `pretrain_child_wdl_encoder_della.slurm` | `cts.data.build_tree` |
| 6 | pack controller episodes | `pack_controller_episodes_della.slurm` | `cts.data.preprocess_mc.pack` |
| 7 | materialize + merge cache | `materialize_controller_cache_della.slurm` | `cts.data.preprocess_mc.materialize` |
| 8 | train controller | `train_fitted_q_controller_della.slurm` | `cts.train.controller_train` |

Tree generation uses `scripts/submit_generate_dataset_shards.py` to shard FEN lists across array jobs.

See [`LAB_NOTEBOOK.md`](../LAB_NOTEBOOK.md) for experiment history and canonical artifact paths.
