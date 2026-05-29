# Configs

YAML configs for `python3 -m cts.<module> --config …`. Each file is a Pydantic-validated spec for one pipeline stage.

**Layout mirrors `slurm/` stage folders.** Add run-specific YAMLs here as hl4291 experiments are defined — the old ysagiv stub configs were removed (2026-05-29). See [`REPO_STRUCTURE.md`](../REPO_STRUCTURE.md) §6 for rationale.

| Directory | Stages | Typical modules |
|-----------|--------|-----------------|
| `1_preprocess_data/` | Sample FENs, build lc0 trees, GNN split/pack | `cts.data.sample_fens`, `cts.data.build_tree`, `cts.data.preprocess_gnn.split`, `cts.data.preprocess_gnn.pack` |
| `2_pretrain_encoder/` | Child-WDL (or other) encoder pretrain | `cts.data.build_tree` (`command: pretrain-child-wdl-encoder`) |
| `3_preprocess_root/` | Pack controller episodes, materialize/merge embedding caches | `cts.data.preprocess_mc.pack`, `cts.data.preprocess_mc.materialize` |
| `4_supervised_controller/` | Fitted-Q controller training | `cts.train.controller_train` |

Example:

```bash
export CONFIG="$PWD/slurm/configs/3_preprocess_root/materialize_validation.yaml"
sbatch --export=ALL,CONFIG slurm/3_preprocess_root/materialize_controller_cache_della.slurm
```

Override keys at the CLI: `--override epochs=10 --override batch_size=512`.
