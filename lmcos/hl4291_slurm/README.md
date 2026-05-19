# hl4291 Slurm scripts (della)

Numbered wrappers for the **hl4291** topology-encoder data pipeline on scratch.

**Scratch root:** `/scratch/gpfs/GRIFFITHS/hl4291/chess/CTS/`

| Directory | Contents |
|-----------|----------|
| **`fens/`** | Root FEN lists (inputs) — e.g. `sampled_root_fens_2023.txt` |
| **`data/`** | Generated trees, split manifests, packed tensorized shards (outputs) |

Yotam’s main CTS pipeline (conda `CTS`, ysagiv paths) lives in [`../slurm/`](../slurm/).

## Scripts

| # | Script | Step |
|---|--------|------|
| 1 | [`1_generate_shards.slurm`](1_generate_shards.slurm) | GPU array: lc0 teacher trees (`CONFIG` + `CHUNK_SIZE` → FEN slice) |
| 2 | [`2_pack_shards.slurm`](2_pack_shards.slurm) | CPU: tensorize + pack for GNN pretrain (`CONFIG` = split + topology preset) |
| 3 | [`3_pretrain_topology.slurm`](3_pretrain_topology.slurm) | GPU: topology encoder + `TopologyHead` (Huber); `CONFIG` = pretrain YAML |

Between **1** and **2**, run **split** on the CLI (no Slurm script):

```bash
python3 -u -m cts.data.preprocess_gnn.split \
  --config configs/data/preprocess_gnn/hl4291_split_50k_topology.yaml
```

## Current status (2026-05-19)

| Step | Status |
|------|--------|
| 1 — 50k tree generation | **48,701 / 50,000** trees on scratch; array task **7** failed — indices **38701–39999** missing (~**1,299** FENs) |
| Split | Done — `pretrain_split_50k_topology` (43,831 train / 4,870 val) |
| 2 — topology-full pack | **Done** — job **8429582**, ~9 min → `pretrain_packed_50k_topology_full` |
| 3 — topology pretrain | **Ready to run** — see configs below |

**Pretrain configs**

| Run | YAML |
|-----|------|
| CLI / Slurm smoke (64 train / 32 val trees, one shard) | [`hl4291_pretrain_topology_shard_slurm_smoke.yaml`](../configs/data/hl4291_pretrain_topology_shard_slurm_smoke.yaml) |
| Full ~49k packed corpus | [`hl4291_pretrain_topology_50k.yaml`](../configs/data/hl4291_pretrain_topology_50k.yaml) |

**One-batch code smoke (no `build_tree` CLI):** [`scripts/smoke_topology_pretrain_batch.py`](../scripts/smoke_topology_pretrain_batch.py)

## Submit

```bash
cd /home/hl4291/chess_analysis/lmcos && mkdir -p logs

# 1) trees (full 50k: 10 × 5000 FENs)
export CONFIG="$PWD/configs/data/hl4291_build_tree_50k.yaml"
sbatch --array=0-9 --export=ALL,CHUNK_SIZE=5000 hl4291_slurm/1_generate_shards.slurm

# 1b) optional — backfill gap only (~2.8 h one GPU; resume skips existing files)
# sbatch --export=ALL,CONFIG hl4291_slurm/1_generate_shards.slurm  # with overrides start_index=38701 end_index=40000

# split (see above), then:

# 2) pack (~15 min wall)
export CONFIG="$PWD/configs/data/preprocess_gnn/hl4291_pack_50k_topology_full.yaml"
sbatch hl4291_slurm/2_pack_shards.slurm

# 3) topology pretrain (smoke first)
export CONFIG="$PWD/configs/data/hl4291_pretrain_topology_shard_slurm_smoke.yaml"
sbatch hl4291_slurm/3_pretrain_topology.slurm
# Full run: CONFIG=configs/data/hl4291_pretrain_topology_50k.yaml
```

## Key configs

| Purpose | YAML |
|---------|------|
| Tree generation | [`configs/data/hl4291_build_tree_50k.yaml`](../configs/data/hl4291_build_tree_50k.yaml) |
| Train/val split | [`configs/data/preprocess_gnn/hl4291_split_50k_topology.yaml`](../configs/data/preprocess_gnn/hl4291_split_50k_topology.yaml) |
| Topology-full pack | [`configs/data/preprocess_gnn/hl4291_pack_50k_topology_full.yaml`](../configs/data/preprocess_gnn/hl4291_pack_50k_topology_full.yaml) |
| Topology pretrain (full) | [`configs/data/hl4291_pretrain_topology_50k.yaml`](../configs/data/hl4291_pretrain_topology_50k.yaml) |
| Topology pretrain (shard smoke) | [`configs/data/hl4291_pretrain_topology_shard_slurm_smoke.yaml`](../configs/data/hl4291_pretrain_topology_shard_slurm_smoke.yaml) |

Logs: [`../logs/`](../logs/). Chronological notes: [`../LAB_NOTEBOOK.md`](../LAB_NOTEBOOK.md) (2026-05-17 – 2026-05-19).
