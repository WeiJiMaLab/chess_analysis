# Repository structure (2026-05-29)

This document describes the **layout refactor** on branch `jordan`: what changed, why, and what tradeoffs it introduces. For experiment history, see [`LAB_NOTEBOOK.md`](LAB_NOTEBOOK.md). For day-to-day Slurm commands, see [`slurm/README.md`](slurm/README.md).

---

## 1. Two trees in `chess_analysis/`

The workspace intentionally separates **human chess analytics** from **neural search-control (CTS)** code:

| Path | Purpose |
|------|---------|
| **`chess_analysis/human_analytics/`** | DuckDB move-time ETL, figures, Slidev deck, modular `metacontrol/` refactor. *Not* the CTS training pipeline. |
| **`chess_analysis/lmcos/`** | CTS package (`cts`), cluster Slurm wrappers, run configs, tests. |

Previously both were confused under names like `src/`, `analysis/` (CTS proxy scripts, then briefly `analysis/` again), and `lmcos/src/cts/`. The refactor makes the split explicit:

```
chess_analysis/
├── human_analytics/          # human analytics (renamed from `src/`, then `analysis/`)
│   ├── movetime_analysis.py
│   ├── metacontrol/
│   ├── slurm/                # DuckDB / engine-eval jobs
│   └── figures/
└── lmcos/                    # CTS (this repo)
    ├── src/                  # cts package (core, data, models, train)
    ├── analysis/             # cts.analysis — post-hoc diagnostics (Python only)
    ├── slurm/
    │   ├── configs/<stage>/  # run YAMLs (empty until hl4291 configs are added)
    │   ├── logs/
    │   └── <stage>/          # cluster wrappers + local orchestrators
    ├── tests/
    └── fens/
```

**Benefit:** A new contributor can tell immediately which directory answers “how do humans allocate time?” vs “how do we train the halt/continue controller?” without parsing nested `src/` paths.

**Cost:** The old CTS proxy tree (formerly under `chess_analysis/analysis/`, removed 2026-05-29) (`2a_make_cache.py`, `2b_train_controller.py`, etc.) was removed. Scratch artifacts under `/scratch/.../hl4291/chess/CTS/` remain, but local rerun scripts do not until re-wired against `lmcos` entry points.

---

## 2. `lmcos` package layout

### Before

```
lmcos/
├── src/cts/{core,data,models,train,analysis}/   # double nesting
├── scripts/                                     # legacy CLIs + one orchestrator
├── slurm/*.slurm                                # flat list, ysagiv defaults
├── configs/data/... configs/train/...           # 34 ysagiv YAML stubs
├── hl4291_slurm/                                # separate 50k pipeline
├── demos/                                       # stale notebooks
├── logs/                                        # top-level Slurm output
└── HANDOFF.md                                   # onboarding doc (partially stale)
```

### After

```
lmcos/
├── src/
│   ├── core/
│   ├── data/
│   ├── models/
│   ├── train/
│   └── _config.py
├── analysis/                # post-hoc diagnostics (imported as cts.analysis)
│   └── _budgeted/
├── slurm/
│   ├── configs/
│   │   ├── README.md
│   │   └── {1_preprocess_data,2_pretrain_encoder,3_preprocess_root,4_supervised_controller}/
│   ├── logs/
│   ├── 1_preprocess_data/
│   ├── 2_pretrain_encoder/
│   ├── 3_preprocess_root/
│   └── 4_supervised_controller/
├── tests/
└── fens/
```

**Benefit — flattened `src/`:** Implementation files live directly under `lmcos/src/` instead of `lmcos/src/cts/`, aligning with the sibling `chess_analysis/human_analytics/` tree and reducing “which `src`?” confusion.

**Benefit — diagnostics separated:** `analysis/` sits beside `src/` (not inside it). Pipeline code (`core`, `data`, `models`, `train`) stays in `src/`; offline plotting and evaluation live in `analysis/` but still import as `cts.analysis.*`.

**Benefit — stable imports:** `import cts`, `python3 -m cts.data.build_tree`, and Slurm invocations are unchanged. `pyproject.toml` maps `cts` → `src/` and `cts.analysis` → `analysis/` via `package-dir`. Use `pip install -e .` on compute nodes, or rely on `conftest.py` / `PYTHONPATH=${PROJECT_DIR}` for tests and Slurm.

**Cost — no root `cts/` shim:** The old one-file `cts/__init__.py` import shim was removed; package resolution is entirely via setuptools / editable install.

---

## 3. Pipeline stages (mental model)

Stages are numbered consistently in **`slurm/<stage>/`** and **`slurm/configs/<stage>/`**:

| Stage | Folder | What happens | Primary `cts` modules |
|-------|--------|--------------|------------------------|
| **1** | `1_preprocess_data/` | Sample FENs → lc0 trees → GNN split/pack | `sample_fens`, `build_tree`, `preprocess_gnn.split`, `preprocess_gnn.pack` |
| **2** | `2_pretrain_encoder/` | Encoder pretrain (child-WDL, etc.) | `build_tree` (`pretrain-child-wdl-encoder`) |
| **3** | `3_preprocess_root/` | Pack controller episodes → materialize `[z_t, N_t, T_t]` caches | `preprocess_mc.pack`, `preprocess_mc.materialize` |
| **4** | `4_supervised_controller/` | Fitted-Q controller training | `train.controller_train` |

Every cluster job follows the same pattern:

```bash
export CONFIG="$PWD/slurm/configs/<stage>/<run>.yaml"
export PROJECT_DIR=/home/hl4291/chess_analysis/lmcos
sbatch --export=ALL,CONFIG slurm/<stage>/<job>.slurm
# inside the .slurm script:
python3 -m cts.<module> --config "${CONFIG}"
```

**Benefit:** Folder names encode **dependency order**. Config paths mirror Slurm paths, so “which YAML for this job?” is obvious.

**Cost:** Stage **3** name (`preprocess_root`) is shorthand for “controller/root-step preprocessing” (episode pack + embedding materialization), not “preprocess the repo root directory.” The name may need refinement if the pipeline grows.

---

## 4. Slurm vs local orchestrators

| Artifact | Runs where | Role |
|----------|------------|------|
| `slurm/<stage>/*.slurm` | Compute node (GPU/CPU) | One `sbatch` job → one `python3 -m cts.*` invocation |
| `slurm/1_preprocess_data/submit_generate_dataset.sh` | Login node | Shell wrapper |
| `slurm/1_preprocess_data/submit_generate_dataset_shards.py` | Login node | Slices a FEN file into shard YAMLs and submits **many** `generate_dataset_shard.slurm` jobs |

There is **no separate `scripts/` directory** anymore. The only orchestrator that is not itself a Slurm job lives next to the stage-1 Slurm scripts it submits.

**Benefit:** One place to look for “how do I launch stage 1 at scale?” — `slurm/1_preprocess_data/`.

**Cost:** Login-node Python helpers are mixed with cluster `.slurm` files in the same folder. If more orchestrators appear, consider a `slurm/<stage>/local/` subfolder.

---

## 5. Logs

| Before | After |
|--------|-------|
| `lmcos/logs/*.out` | `lmcos/slurm/logs/*.out` |

All `#SBATCH --output` / `--error` paths and `mkdir -p` lines were updated. `.gitignore` ignores `slurm/logs/`.

**Benefit:** Logs sit beside the code that produces them; top-level `lmcos/` is not cluttered with job output.

---

## 6. Configs

All **ysagiv scratch-path stub YAMLs** were removed (~34 files under the old `configs/data/` and `configs/train/` trees). Empty stage directories now live under **`slurm/configs/`** as placeholders.

**Benefit:**
- No misleading paths to `/scratch/.../ysagiv/...` or `/home/ysagiv/...`.
- New hl4291 runs start from configs that match **your** scratch layout and naming.
- Diff between experiments = diff between YAML siblings in the same stage folder.

**Cost:**
- **Nothing is runnable out of the box** until you add configs (e.g. `slurm/configs/1_preprocess_data/build_tree.yaml`).
- Historical LAB_NOTEBOOK entries that cite deleted YAML paths are archival only.
- Reference controller configs used by the removed CTS proxy (`controller_subtree_weighted_zt_tt.yaml`, etc.) are gone; ysagiv checkpoints on scratch are still readable if paths are known.

---

## 7. Removed paths (intentional)

| Removed | Rationale |
|---------|-----------|
| **`hl4291_slurm/`** + hl4291 YAMLs | Separate node-targets / 50k experiment track; not wired into current CTS comparison pipeline |
| **`demos/`** | April-era notebooks; stale (`HaltController` removed), duplicated concepts now in `src/` + LAB_NOTEBOOK |
| **`HANDOFF.md`** | Overlapped with Slurm README + notebook; contained stale topology/hl4291 sections |
| **`configs/analysis/`** + audit Slurm | Post-hoc tooling; not on stages 1–4 path (`analysis/` Python retained) |
| Alternate Slurm (compute-advantage, sweeps, filter, rewrite_compact, …) | Experimental / inactive branches |
| CTS proxy under old `chess_analysis/analysis/` | Superseded by consolidating execution in `lmcos`; scratch outputs preserved |

**Benefit:** Smaller surface area; active path is stages 1–4 + `tests/` + `LAB_NOTEBOOK.md`.

**Cost:** Reviving a one-off experiment (e.g. encoder KL audit, entropy sweep) requires restoring scripts from git history.

---

## 8. What stayed the same

- **`import cts` / `python3 -m cts.*`** — public Python API unchanged.
- **`tests/`** — 141 tests pass (1 pre-existing `test_audit_encoder_kl` failure).
- **`analysis/`** — offline diagnostics library at repo root (imported as `cts.analysis`; no Slurm wrappers after cleanup).
- **ysagiv Slurm script bodies** — still default to ysagiv `PROJECT_DIR` and conda `CTS`; update env block when running as hl4291 (see [`slurm/README.md`](slurm/README.md)).
- **Upstream ysagiv artifacts** on scratch — still valid inputs for comparison runs; only local wiring/docs changed.

---

## 9. Benefits summary

| Goal | How the new structure helps |
|------|-----------------------------|
| **Disambiguate `src/`** | `chess_analysis/human_analytics/` vs `lmcos/src/`; no `lmcos/src/cts/` |
| **Avoid `analysis` name clash** | `human_analytics/` vs `lmcos/analysis/` (CTS diagnostics, imported as `cts.analysis`) |
| **Single execution model** | Cluster = `slurm/<stage>/` + `slurm/configs/<stage>/`; logic = `python3 -m cts.*` only |
| **Ordered pipeline** | Numbered folders match dependency chain 1 → 2 → 3 → 4 |
| **hl4291 ownership** | Removed ysagiv/hl4291_slurm stubs; empty config slots for your YAMLs |
| **Less clutter** | No demos, no duplicate scripts tree, logs under `slurm/logs/` |
| **Stable imports** | `cts` package name preserved via `pyproject.toml` `package-dir` (`src/` + `analysis/`) |

---

## 10. Costs and follow-ups

1. **Write hl4291 configs** under `slurm/configs/<stage>/` before any production Slurm submit.
2. **Update Slurm env blocks** (`PROJECT_DIR`, venv vs conda, module loads) for hl4291 della — wrappers still carry ysagiv defaults in the script body.
3. **Re-wire stage 2a/2b proxy** (if desired) as thin `slurm/3_*` / `slurm/4_*` configs calling `materialize` and `controller_train`, instead of the deleted proxy scripts.
4. **Optional:** rename `3_preprocess_root` → clearer name (e.g. `3_preprocess_controller`) if it causes confusion.
5. **Optional:** add `lmcos/README.md` pointer to this file (currently only LAB_NOTEBOOK + slurm/configs READMEs).

---

## 11. Quick reference

```bash
# Repo root for CTS work
cd /home/hl4291/chess_analysis/lmcos

# Tests (no install required; conftest adds repo root to path)
python3 -m pytest tests/

# Stage 1 tree generation (after writing slurm/configs/1_preprocess_data/build_tree.yaml)
mkdir -p slurm/logs
export CONFIG="$PWD/slurm/configs/1_preprocess_data/build_tree.yaml"
./slurm/1_preprocess_data/submit_generate_dataset.sh --config "$CONFIG"   # dry-run
./slurm/1_preprocess_data/submit_generate_dataset.sh --config "$CONFIG" --submit

# Single Slurm stage
export CONFIG="$PWD/slurm/configs/2_pretrain_encoder/pretrain.yaml"
sbatch --export=ALL,PROJECT_DIR=$PWD,CONFIG slurm/2_pretrain_encoder/pretrain_child_wdl_encoder_della.slurm
```

---

*Last updated: 2026-05-29 (branch `jordan`).*
