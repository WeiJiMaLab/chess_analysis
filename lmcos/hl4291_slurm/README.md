# hl4291 Slurm scripts (della)

Numbered wrappers for the **hl4291** data pipeline on scratch (`/scratch/gpfs/GRIFFITHS/hl4291/`).

Yotam’s cluster defaults live in [`../slurm/`](../slurm/).

| # | Script | Step |
|---|--------|------|
| 1 | `1_generate_shards.slurm` | GPU array: lc0 teacher trees (`CONFIG` + `CHUNK_SIZE` → FEN slice) |
| 2 | `2_pack_shards.slurm` | CPU: tensorize + pack for GNN pretrain (`CONFIG` = split + topology preset) |

Between 1 and 2, run **split** on the CLI (no Slurm script):

```bash
python3 -u -m cts.data.preprocess_gnn.split --config configs/data/preprocess_gnn/hl4291_split_50k_topology.yaml
```

## Submit

```bash
cd /home/hl4291/chess_analysis/lmcos && mkdir -p logs

# 1) trees (example: 10 × 5000 FENs)
export CONFIG="$PWD/configs/data/hl4291_build_tree_50k.yaml"
sbatch --array=0-9 --export=ALL,CHUNK_SIZE=5000 hl4291_slurm/1_generate_shards.slurm

# split (see above), then:

# 2) pack
export CONFIG="$PWD/configs/data/preprocess_gnn/hl4291_pack_50k_topology_full.yaml"
sbatch hl4291_slurm/2_pack_shards.slurm
```

Logs: [`../logs/`](../logs/).
