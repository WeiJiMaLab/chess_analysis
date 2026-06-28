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

## Configs: one merged source-of-truth per rung
`configs/elo{1800,2000,2200}.yaml` each hold **one section per stage**
(`treegen, split, gnn_pack, mc_pack, encoder, materialize, train, eval`), with absolute
paths spelled out — open one file and you see the entire rung. Across rungs only `sf_elo`
and the `elo####` path component differ.

Each `cts` entry point still takes a single flat `--config FILE` (pydantic, `extra="forbid"`),
so the slurm scripts slice the relevant section with **`configs/render_stage.py`**:
```bash
python configs/render_stage.py configs/elo1800.yaml mc_pack --out mc_pack.yaml --set num_workers=32
#                              └ merged config        └ stage  └ flat YAML the entry point eats
```
`--set` injects the few per-invocation values that aren't per-rung (e.g. materialize's
per-split `output_dir`/`packed_data`, or mc_pack's worker count). `render_stage.py` is the
*only* glue — the copied `src/cts` code is untouched.

## Directory layout
```
lmcos_tiny/
├── README.md
├── pyproject.toml          installable `cts` package (src-layout)
├── env.sh                  `source env.sh` → PYTHONPATH=src (this fork wins over the editable install)
├── configs/
│   ├── elo1800.yaml        merged source-of-truth, one section per stage  ┐ only sf_elo +
│   ├── elo2000.yaml                                                       │ elo#### path
│   ├── elo2200.yaml                                                       ┘ differ
│   └── render_stage.py     slice a section → flat per-stage YAML (+ --set overrides)
├── pipeline/               orchestration, one dir per PHASE
│   ├── 1_treegen_pack_trees/
│   │     gen_trees.slurm                       (CPU array: treegen)
│   │     pack_trees.slurm                       (CPU: split + gnn_pack + mc_pack)
│   ├── 2_train_encoder_pack_rootreps/
│   │     train_encoder_pack_rootreps.slurm      (GPU: encoder + materialize)
│   │     pack_rootreps_rerun.slurm              (GPU array: materialize-only, reuse encoders)
│   ├── 3_train_readout/
│   │     train_readout.slurm                    (GPU: controller_train)
│   └── 4_eval/
│         run_eval.sh                            (CPU: 4-model eval + ladder plot)
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

| phase | command (per rung unless noted) | resource | measured time¹ |
|---|---|---|---|
| 1 gen-trees | `ELO=1800 SHARD_SIZE=2500 BASE_START=0 LANE_END=50000 sbatch --array=0-19 pipeline/1_treegen_pack_trees/gen_trees.slurm` | CPU ×20 | 35 min–1h11 / task |
| 1 pack-trees | `ELO=1800 sbatch --dependency=afterok:<gen> pipeline/1_treegen_pack_trees/pack_trees.slurm` | 32 CPU | 18–24 min |
| 2 enc+reps | `ELO=1800 sbatch --dependency=afterok:<pack> pipeline/2_train_encoder_pack_rootreps/train_encoder_pack_rootreps.slurm` | 1 GPU | enc ~2 min · materialize ~25–45 min |
| 2 reps-rerun | `sbatch --array=0-2 pipeline/2_train_encoder_pack_rootreps/pack_rootreps_rerun.slurm` (materialize all rungs, reuse encoders) | 1 GPU ×3 | ~25–45 min |
| 3 readout | `ELO=1800 sbatch --dependency=afterok:<reps> pipeline/3_train_readout/train_readout.slurm` | 1 GPU | ~10–15 min |
| 4 eval | `bash pipeline/4_eval/run_eval.sh` | CPU | < 5 min |

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
figures/lmcos/                   phase 4 — regret/oss/ladder plots + results JSON
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
