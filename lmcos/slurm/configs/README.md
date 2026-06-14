# Configs

YAML configs for `python3 -m cts.<module> --config …`. Each file is a Pydantic-validated spec for one pipeline stage.

**Layout mirrors `slurm/` stage folders.** Add run-specific YAMLs here as hl4291 experiments are defined — the old ysagiv stub configs were removed (2026-05-29).

| Directory | Stages | Typical modules |
|-----------|--------|-----------------|
| `1_preprocess_data/` | Sample FENs, build lc0 trees, GNN split/pack | `cts.data.process_fens`, `cts.data.build_tree`, `cts.data.preprocess_gnn.split`, `cts.data.preprocess_gnn.pack` |
| `2_pretrain_encoder/` | Child-WDL (or other) encoder pretrain | `cts.data.build_tree` (`command: pretrain-child-wdl-encoder`) |
| `3_preprocess_root/` | Pack controller episodes, materialize/merge embedding caches | `cts.data.preprocess_mc.pack`, `cts.data.preprocess_mc.materialize` |
| `4_supervised_controller/` | Fitted-Q controller training | `cts.train.controller_train` |

### Stage 4 ablation configs

| File | Display name | Purpose |
|------|--------------|---------|
| `legacy_root_budget.yaml` | `legacy[root+budget]` | Rerun encoder, inputs `[z_t, T_t]` |
| `subtree_weighting_root_budget.yaml` | `subtree-weighting[root+budget]` | Subtree-weighted encoder, inputs `[z_t, T_t]` |
| `subtree_weighting_root.yaml` | `subtree-weighting[root]` | Subtree-weighted encoder, inputs `[z_t]` only |

Common keys: `epochs`, `validation_step_interval`, `greedy_eval_step_interval`, `metrics_log_interval`, `metrics_path` (`slurm/outputs/4_supervised_controller/<run>.yaml`), `metrics_plot_path` (`slurm/outputs/4_supervised_controller/<run>.png`), `metrics_run_name`. `ControllerTrainMetricsLogger` writes metrics and refreshes curves on eval steps; replot with `python3 -m cts.train.controller_train plot-metrics …`.

Example:

```bash
export CONFIG="$PWD/slurm/configs/4_supervised_controller/subtree_weighting_root_budget.yaml"
python3 -m cts.train.controller_train --config "${CONFIG}"
```

Override keys at the CLI: `--override epochs=10 --override batch_size=512`.
