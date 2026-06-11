# R-LMCOS-OVERVIEW — LMCOS pipeline and repo layout

**Ref:** `R-LMCOS-OVERVIEW` · [Index](README.md)

## Summary (after implementation)

| | |
|---|---|
| **Description** | Offline meta-controller over lc0 search trees: FEN sample → filter → tree gen → GNN pretrain → controller pack → fitted-Q training (`cts` package). |
| **Rationale** | Learn halt/continue from teacher oracle traces before full planning-head self-play. |
| **Expectation** | Six-stage pipeline with Pydantic YAML configs and `slurm/<stage>/` mirrors. |
| **Finding** | Layout stable since 2026-05-29 refactor; hl4291 Della uses `VENV_DIR=/home/hl4291/venv`. |

## Notes

### Package map

| Module | Role |
|---|---|
| `cts.core` | `SearchTree`, tensorizer, lc0 providers |
| `cts.data` | FEN sample/filter, `build_tree`, GNN + MC preprocess |
| `cts.models` | `TreeEncoder`, `MetaController` |
| `cts.train` | `gnn_pretrain`, `controller_train` |
| `cts.analysis` | Diagnostics (`analyze_budgeted_controller_run`, A0/A1 scripts) |

### Pipeline stages

1. FEN sampling (DuckDB / Lichess SQL)
2. PUCT stability filter
3. Tree generation (96-node budget, oracle trace)
4. Encoder pretraining (child-WDL)
5. Controller episode pack (+ budget augmentation)
6. Fitted-Q controller training

**Commands:** `lmcos/slurm/README.md`, `lmcos/slurm/configs/README.md`.  
**Historical experiments (Apr–May 2026):** [(R-ARCH-LMCOS)](archive-lmcos-notebook-legacy.md).
