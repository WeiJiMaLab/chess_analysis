# `lmcos_small/src/analysis/` — deliberation / RT analysis

The reaction-time (RT) vs tree-signal analysis behind reports **R-VOC** (`reports/voc.md`)
and **R-CONSTRUAL** (`reports/construal.md`). Relocated here from the buried `scratch_halt/`
so it has a visible, documented home (mirrors `lmcos_small/src/human/`'s layout).

The thesis: human log-RT on a position tracks the **size of the rationally-built
consideration set** (legal moves / action gap / VOC family), and is *separable* from the
hindsight value of computation. These scripts compute per-tree signals from the Stockfish
search trees, then correlate them against human log-RT (Spearman ρ, percentile-bootstrap
95% CIs — see `[[bootstrap-cis-always]]`).

## Layout

```
analysis/
  README.md                 # this file
  compute_voc_signals.py    # cts; per-tree signals  -> voc_signals.parquet     (sbatch)
  compute_voc_tau_sweep.py  # cts; softmax-VOC τ sweep -> voc_tau_sweep.parquet  (sbatch)
  halter_vs_rt.py           # cts; causal tree-stats halter vs hindsight oracle  (sbatch)
  make_rt_figures.py        # THE figure source for R-VOC / R-CONSTRUAL          (local)
  concavity_test.py         # saturation / plateau-shift diagnostic              (local)
  litmus_strength.py        # engine-strength diagnostic (SF-1350 vs SF-2000)    (local)
  construal_proxy.py        # proxy for construal level analysis                 (local)
  gap_reconcile.py          # reconciles action gaps                             (local)
  normative_curves.py       # fits normative curves                              (local)
  oss_nodecost.py           # calculates node cost for one-step search           (local)
  prune_proxy.py            # value-pruning proxy (step 1, no-regen)             (local)
  prune_regen_analyze.py    # analyzes regenerated pruned trees                  (local)
  ratio_test.py             # ratio test diagnostic                              (local)
  sf_n1_vs_n100.py          # compares Stockfish nodes=1 vs nodes=100            (local)
  
Note: All Slurm scripts have been consolidated to the unified slurm directory:
  ../slurm/analysis/
    voc_signals.slurm       # runs compute_voc_signals.py (VOC_ELO env, default 2000)
    voc_tau_sweep.slurm     # runs compute_voc_tau_sweep.py
    halter_vs_rt.slurm      # runs halter_vs_rt.py
    dumbeval_gen.slurm      # near-static (nodes=1) dumb-eval tree generation -> elo2000_n1
    logs/                   # slurm .out/.err logs (at ../slurm/analysis/logs/)
```

## Compute → figure flow

1. **`compute_voc_signals.py`** (sbatch `../slurm/analysis/voc_signals.slurm`) — reads the filtered SF
   trees, computes per-tree signals (legal moves, action gap, n_good_*, oracle gain /
   regret / step\*, etc.) keyed by 4-field FEN → `voc_signals.parquet`.
   `VOC_ELO` (default 2000) selects the `sf_filtered/elo{ELO}` directory.
2. **`compute_voc_tau_sweep.py`** (sbatch `../slurm/analysis/voc_tau_sweep.slurm`) — softmax-VOC over a
   temperature (τ) sweep keyed by FEN → `voc_tau_sweep.parquet`.
3. **`make_rt_figures.py`** (local; ~1 min — reads the ~63k parquet + DuckDB) — the single
   source for every ρ-vs-RT figure: `rt_headline`, `cost_sweep_rt`, `voc_tau_sweep`,
   `rt_partials`, `good_moves_signflip`, `oss_dist_elo2000`. Cited in the
   `reports/voc.md` / `reports/construal.md` appendices.

Diagnostics (local, read the same parquets):

- **`concavity_test.py`** — saturation / plateau-shift test (`rt_vs_n_concavity`).
- **`litmus_strength.py`** — does the legal-moves/RT effect change at a weaker engine?
  Compares SF-2000 vs SF-1350 voc parquets (`litmus_strength`).
- **`halter_vs_rt.py`** (sbatch `slurm/halter_vs_rt.slurm`) — a causal tree-stats halter
  (PG-trained StatsReadout) vs the hindsight oracle, correlated against log-RT.

`dumbeval_gen.slurm` is a generation job (not analysis): it sources
`lmcos_small/slurm/helpers/setup_env.sh` and calls `cts.data.build_tree` with
`sf_search_limit_nodes=1` into a separate `sf_trees/elo2000_n1` dir (H1/H3 test).

## Dependencies

Most scripts are **pure analytics** (pandas / duckdb / matplotlib) — no GNN/MC. Only the
three compute/halter scripts (`compute_voc_signals.py`, `compute_voc_tau_sweep.py`,
`halter_vs_rt.py`) need `cts`; they add `lmcos_small/src` to `sys.path` and import
`cts.data.preprocess_mc.*` / `cts.analysis._budgeted`.

## Data & figure paths (absolute)

| What | Path |
| :--- | :--- |
| Filtered SF trees / signals | `/scratch/gpfs/GRIFFITHS/hl4291/sf_filtered/elo{ELO}/` |
| Raw SF trees | `/scratch/gpfs/GRIFFITHS/hl4291/sf_trees/elo{ELO}/` |
| `voc_signals.parquet` | `…/sf_filtered/elo{ELO}/voc_signals.parquet` |
| `voc_tau_sweep.parquet` | `…/sf_filtered/elo2000/voc_tau_sweep.parquet` |
| DuckDB (human RT) | `/scratch/gpfs/GRIFFITHS/hl4291/personal.db` |
| Figures | `/home/hl4291/chess_analysis/figures/normative/` |

Interpreter: slurm jobs use `/home/hl4291/venv/bin/python` (`dumbeval_gen` instead sources
`setup_env.sh`). Run local diagnostics with the project venv from the repo root.
