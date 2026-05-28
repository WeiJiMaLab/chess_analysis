# Analysis (Yotam controller / encoder outputs)

## Stage 2b named variants

| Variant name | Encoder | Head inputs | Yotam greedy regret (20 ep) |
|--------------|---------|-------------|------------------------------|
| `subtree_weight_root+budget` | subtree-weighted | z_t + T_t | **~0.024** (best) |
| `subtree_weight_root` | subtree-weighted | z_t only | ~0.076 |
| `no_subtree_weight_root+budget` | rerun (no subtree weighting) | z_t + T_t | ~0.22 |

Artifacts on scratch (`/scratch/gpfs/GRIFFITHS/hl4291/chess/CTS/2b/`):

- `{name}_controller.pt`
- `{name}_metrics.json`

Plots (`analysis/outputs/2b/`):

- `{name}_loss.png` — per variant (two panels)
- `overlay_loss.png` — total loss overlay

```bash
cd /home/hl4291/chess_analysis

# Train all three (1000-batch proxy each)
python3 analysis/2b_train_controller.py --all --max-batches 1000 --save

# Or one variant (default = best)
python3 analysis/2b_train_controller.py --variant subtree_weight_root+budget --max-batches 1000 --save

# Plot all metrics in 2b/ + overlay
python3 analysis/2b_plot_loss.py
```

## Regret curves (full Slurm runs)

```bash
./analysis/sync_logs.sh
python3 analysis/plot_controller_regret_curves.py
```
