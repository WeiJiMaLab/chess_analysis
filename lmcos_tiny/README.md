# lmcos_tiny — self-contained Stockfish mini-GNN / mini-MC halt-model pipeline

A **standalone fork** of just the pieces needed to take chess positions (FENs) through to a
plot profiling four *halt policies* on a Stockfish "strength ladder" (Elo 1800 / 2000 / 2200).
It is the small (`tiny` encoder, 50K-root) sibling of the full lc0 pipeline in `../lmcos`.

> **Why this exists.** The full `lmcos/src/cts` package is large and shared with the
> production lc0 run. This directory copies *only* the modules this pipeline actually
> imports (the runtime import closure — **38 files**), folded into one importable `cts`
> package, plus the orchestration and **one merged config per rung**, so the whole thing
> reads top-to-bottom in one place. Code under `src/cts/` is **byte-identical** to the
> parent — nothing was rewritten, only extracted.

## The goal (everything else is plumbing)
FENs → a plot comparing four halt policies on **Regret** and **Stop-Accuracy** `P(stop==OSS)`:

| policy | what it does | parameters |
|---|---|---|
| **NeverHalt** | search to budget | none |
| **AlwaysHalt** | stop at step 0 | none |
| **FractionHalt** | stop once tree size ≥ `θ·B` | one scalar `θ` (fit on train regret) |
| **MCHalt** | the trained GNN meta-controller ("readout") | learned (encoder + MLP head) |

`OSS` = the cost-aware oracle's optimal stop step. The "ladder" sweeps teacher strength
(Stockfish Elo 1800/2000/2200) to see how the policies move with it.

## The 4-phase pipeline

```
 PHASE 1  tree gen + pack trees   (CPU)     PHASE 2  train encoder + pack root reps  (GPU)
 ───────────────────────────────            ──────────────────────────────────────────────
   gen_trees ─ Stockfish search trees         encoder ───── tiny child-WDL encoder
   split ───── 80/20 train/val                materialize ─ frozen encoder → z_t cache
   gnn_pack ── encoder training examples                      (the "root representations")
   mc_pack ─── budgeted-DP oracle episodes
                                            PHASE 3  train readout (MCHalt)  (GPU)
                                              train ─────── frozen encoder + advantage MLP on [z_t,T_t]

                                            PHASE 4  eval  (CPU)
                                              run_eval ──── 4-model regret/stop-acc + cross-Elo ladder plot
```
Data flow (two branches off `split`, rejoined at `materialize`):
```
gen_trees → split ┬─ gnn_pack ─ encoder ┐
                  └─ mc_pack ───────────┴─ materialize → train → eval
```
Each rung is independent → the three run as parallel SLURM array tasks; ladder wall-clock ≈ one rung.

### "What's the difference between the old `pack_cpu` and `pack_gpu`?"
That earlier split was **by resource (does the job need a GPU?), not by phase** — which is
why it was confusing. It maps cleanly onto the phases above:
- old **`pack_cpu`** = split + gnn_pack + mc_pack  → **Phase 1's "pack trees"** (all CPU).
- old **`pack_gpu`** = encoder + materialize        → **Phase 2** (both GPU; materialize is
  literally "pack root representations" — it runs the frozen encoder over each tree to emit `z_t`).

`mc_pack` lives in Phase 1 (it's the *second* CPU "pack trees" op) even though its episodes
are consumed by the readout in Phase 3.

## Configs: single source-of-truth base file
`configs/core.yaml` holds **one section per stage**
(`treegen, split, gnn_pack, mc_pack, encoder, materialize, train, eval`). All paths are parameterized via the `globals` section.

Each `cts` entry point reads that merged config **directly** — no intermediary rendered YAML.
The shared loader (`cts._config`) accepts `--stage` to slice a section and `--set` to patch it
(dotted keys hit `globals` before `${}` interpolation; bare keys patch the section):
```bash
python -m cts.data.preprocess_mc.pack --config configs/core.yaml --stage mc_pack \
    --set globals.sf_elo=1800 --set num_workers=32
#         └ merged config              └ section   └ pick rung (pre-interp)  └ patch field
```
`configs/render_stage.py` remains as a shell helper for **querying** resolved values
(`--get globals.materialized_dir`), used by `pipeline/helpers/setup_env.sh` to derive paths.

## Directory layout
```
lmcos_tiny/
├── README.md
├── pyproject.toml          installable `cts` package (src-layout)
├── env.sh                  `source env.sh` → PYTHONPATH=src (this fork wins over the editable install)
├── configs/
│   ├── core.yaml           merged source-of-truth base config file with path templates
│   └── render_stage.py     shell helper to QUERY resolved values (--get), used by setup_env.sh
├── pipeline/               orchestration, one slurm file per STAGE (helpers/ aside)
│   ├── helpers/
│   │     setup_env.sh      (Shared helper to load modules, activate venv, and export paths)
│   ├── submit_all.sh                           (chains every stage for all three rungs)
│   ├── 1a_gen_trees.slurm                      (CPU array: treegen)
│   ├── 1b_filter_trees.slurm                   (CPU array: PUCT∩monotone filter shards)
│   ├── 1c_pack_trees.slurm                     (CPU: merge shards + split + gnn_pack + mc_pack)
│   ├── 2a_train_encoder.slurm                  (GPU: encoder)
│   ├── 2b_pack_root.slurm                      (GPU array: materialize cached root reps)
│   ├── 2c_merge_root.slurm                     (CPU: stitch worker shards → z_t cache)
│   ├── 3_train_readout.slurm                   (GPU: controller_train)
│   └── 4_eval.sh                               (CPU: 4-model eval + ladder plot)
├── slurm/logs/             job logs
└── src/cts/                the 38-file import closure (byte-identical to ../lmcos)
    ├── _config.py                       YAML→pydantic loader (the `--config FILE` contract)
    ├── core/        tree, schema, tensorizer, kl_buckets, providers/{stockfish,lc0,…}
    ├── data/        build_tree (gen + encoder pretrain)
    │   ├── preprocess_gnn/  split, pack, teacher_targets
    │   └── preprocess_mc/   pack, oracle, materialize
    ├── models/      gnn, tree_mha, mc (MetaController), readout (the 4-policy family)
    ├── train/       gnn_pretrain, controller_train
    └── analysis/_budgeted/  alt_models_eval, mchalt_scorer, ladder_plot, baselines
```

## How to run

```bash
source lmcos_tiny/env.sh        # PYTHONPATH=src (this fork) + venv
```

> `bash pipeline/submit_all.sh` chains every stage below for all three rungs. To run a single stage:

| stage | command (per rung unless noted) | resource | measured time¹ |
|---|---|---|---|
| 1a gen-trees | `ELO=1800 SHARD_SIZE=2500 BASE_START=0 LANE_END=50000 sbatch --array=0-19 pipeline/1a_gen_trees.slurm` | CPU ×20 | 35 min–1h11 / task |
| 1b filter | `ELO=1800 sbatch --dependency=afterok:<gen> --array=0-99 pipeline/1b_filter_trees.slurm` | CPU ×100 | ~10 min / task |
| 1c pack-trees | `ELO=1800 sbatch --dependency=afterok:<filter> pipeline/1c_pack_trees.slurm` | 16 CPU | 18–24 min |
| 2a train-enc | `ELO=1800 sbatch --dependency=afterok:<pack> pipeline/2a_train_encoder.slurm` | 1 GPU | ~2 min |
| 2b pack-reps | `ELO=1800 NWORKERS=40 sbatch --dependency=afterok:<enc> --array=0-39 pipeline/2b_pack_root.slurm` | 40 GPU | ~25–45 min |
| 2c merge-reps | `ELO=1800 NWORKERS=40 sbatch --dependency=afterok:<reps> pipeline/2c_merge_root.slurm` | CPU | ~5 min |
| 3 readout | `ELO=1800 sbatch --dependency=afterok:<merge> pipeline/3_train_readout.slurm` | 1 GPU | ~10–15 min |
| 4 eval | `bash pipeline/4_eval.sh` | CPU | < 5 min |

¹ Measured on Della from the actual runs (jobs 10323846 / 10324132 / 10368943); phase 3/4 from
the lc0-prod reference (same code, 12 min train). The long poles are **phase 1 gen-trees** and
**phase 2 materialize**; everything else is minutes. Materialize is single-worker here —
3 workers/split (the lc0 pattern) would cut it to ~8–10 min.

## Data locations (on `/scratch/gpfs/GRIFFITHS/hl4291`, not copied here)
```
sf_ladder_roots_50k.txt          50k root FENs (uniform sample of fens.txt)
sf_trees/elo{ELO}/               phase 1 — 50k Stockfish search trees / rung
sf_split/elo{ELO}/               phase 1 — 80/20 manifests
sf_packed/elo{ELO}/              phase 1+2 — gnn pack + tiny_encoder.pt (+ sf_mchalt_controller.pt from phase 3)
sf_mc_packed/elo{ELO}/           phase 1 — ~360k oracle-labelled episodes / rung
sf_mc_materialized/elo{ELO}/     phase 2 — {train,validation}_cache.pt  (z_t root reps)
sf_pack_configs/elo{ELO}/        rendered per-stage YAMLs (render_stage output)
figures/lmcos_tiny/              phase 4 — regret/oss/ladder plots + results JSON
```

## Two gotchas baked into the scripts (so you don't rediscover them)
1. **`--config` is a file path, never stdin.** `_config.load_config` does `Path(path).read_text()`;
   piping a heredoc via `--config -` raises `FileNotFoundError: '-'` — the overnight materialize
   crashed on exactly this. Every stage here renders a real file first.
2. **Tiny encoder arch must be set explicitly.** `controller_train` builds the encoder from the
   `train` section's `k/d_embed/...` fields (not checkpoint metadata) then `load_state_dict`s the
   weights; the schema defaults are the 128-d prod values. The configs set the **tiny** arch
   (`k=1, d_embed=32, d_message=32, n_heads=2, d_att=16, node_embed_hidden=32`) to match
   `tiny_encoder.pt`; a mismatch fails loudly on the state-dict load.

## Relationship to the parent repo
This is an **extract for legibility**, not the source of truth. Authoritative code + lineage live
in `../lmcos` (`mc_pipeline.md`, `mc_minimal_plan.md`, `labnotebook.md`). If the parent's `cts`
modules change, this fork does **not** auto-update — re-extract the closure if you need it current.
