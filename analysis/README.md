# Analysis (Yotam controller / encoder outputs)

Offline analysis of ysagiv's della artifacts. Full Slurm training lives in [`../lmcos/`](../lmcos/); this folder holds **cheap proxies** and plots.

## Layout

| Path | Role |
|------|------|
| [`config.py`](config.py) | All scratch paths, controller runs, caches |
| [`2b_train_controller.py`](2b_train_controller.py) | Stage 2b train proxy (`--max-batches 1000`) + loss plots |
| [`plot_controller_regret_curves.py`](plot_controller_regret_curves.py) | Regret vs epoch from Slurm logs |
| [`sync_logs.sh`](sync_logs.sh) | Copy `cts-fittedq_*.out` from ysagiv's repo |
| [`smoke_benchmark.py`](smoke_benchmark.py) | Time I/O only |
| `logs/` | Local Slurm logs (gitignored) |
| `outputs/` | Figures (gitignored) |

## Stage 2b proxy train

```bash
cd /home/hl4291/chess_analysis
python3 analysis/2b_train_controller.py --max-batches 1000
```

Output: `analysis/outputs/2b_train_controller_loss.png` (two panels: total loss vs batch; MSE vs sign BCE).

**Loss terms** (same as `cts.train.controller_train`):

- `advantage_mse` — regress the oracle advantage `A = Q_continue − Q_halt` (how much better continuing is than stopping).
- `sign_bce` — classify `sign(A)` (continue vs halt); weighted by `sign_loss_weight` (0.1 in the best run).
- `total_loss = advantage_mse + 0.1 × sign_bce`

MSE is usually small (~0.01) when fits are good; sign BCE is often larger (~0.6) because it's a binary cross-entropy, not because the model is "worse" on that term.

## Regret curves (finished Slurm runs)

```bash
./analysis/sync_logs.sh
python3 analysis/plot_controller_regret_curves.py
```

Output: `analysis/outputs/controller_regret_curves.png`
