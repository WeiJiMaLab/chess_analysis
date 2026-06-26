# mc_pipeline.md — Authoritative end-to-end lineage of the current GNN/MC controller

**Purpose.** The single reproducibility record for *this* GNN/MC instance: where every datum came
from, every filter, every training step, the architecture, the packing, and the cost/loss math —
a **current reflection of what the code does**, kept in sync with the code. History, diagnoses, and
resolved bugs belong in the **labnotebook** (archival), never here. If a number can't be traced to a
stage below, that's a bug in this doc — fix it here.

**The instance this describes:** `controller_filtered.pt`. Forward work / live experiments:
`mc_minimal_plan.md`.

---

## 0. Lineage at a glance (stage → artifact)

```
ysagiv human_trees (640,605 *.pt)
  └─[tree filter: PUCT-stable ∩ monotone]→ clean_trees.txt (101,812)            §2  feb712e
       └─[50/50 partition, seed 0]→ split_filtered/{train 50,674 | val 51,138}  §3
            ├─[GNN pack]→ packed/gnn/{train,val}_manifest.json                  §4
            │    └─[child-WDL pretrain, 45 ep]→ GNN/encoder_best.pt             §5  dc08907
            └─[MC pack, budgeted oracle, halt-range≥0.05]→ packed/mc/ (374,750 | 378,920 ep) §6
                 └─[materialize frozen encoder]→ materialized/{train,val}_cache.pt (11.65M | 11.78M) §7
                      └─[controller train, [z_t,T_t], MSE+0.1·signBCE, 3 ep]→ GNN/controller_filtered.pt §8
                           └─[eval vs baselines]→ figures/{regret,stop_eq_oss}_by_model.png §9
```

---

## Definitions (verified against code — every row cross-checked to `file:line`)

**These mechanisms have never changed since the inception of this code.** Do not infer them — this is
the authoritative statement, each row read from the cited construction site (not the labnotebook).
`t` = expansion step (0-indexed, post-trim); `B` = starting budget.

| Symbol | Definition | Code (verified) |
|---|---|---|
| `bmi[t]` (`oracle_best_move_index`) | index of the **best-so-far root move** at step `t` (argmax of the root Q estimate *at that expansion*) | raw tree field |
| `oracle_final_root_q_values[m]` | **depth-96 (full-search) Q** of root move `m` — fixed teacher value, not evolving | raw tree field |
| **`halt_reward[t]`** (= `halt_value[t]`) | `oracle_final_root_q_values[ bmi[t] ]` — **full-search Q of the move best-so-far at `t`** (what you'd get committing now, scored under the *complete* search). **NOT** the evolving shallow estimate. | [pack.py:660](lmcos/src/data/preprocess_mc/pack.py#L660), [oracle.py:270](lmcos/src/data/preprocess_mc/oracle.py#L270) |
| `T_t` (remaining budget) | `B − t` | [oracle.py:259](lmcos/src/data/preprocess_mc/oracle.py#L259) |
| `N_t` (tree size) | node count at step `t` | [pack.py:660](lmcos/src/data/preprocess_mc/pack.py#L660) |
| `time_cost(T_t)` | power-law: `λ·[(T_t−δ+τ)^{−(p−1)} − (T_t+τ)^{−(p−1)}]`; linear: constant `λ` | [oracle.py:167-183](lmcos/src/data/preprocess_mc/oracle.py#L167) |
| `maintenance_cost(N_t)` | `scale·(N_t/ref)^exp` — **DISABLED** (`scale=0`) | [oracle.py:160-164](lmcos/src/data/preprocess_mc/oracle.py#L160) |
| `continue_cost` | `maintenance_cost + time_cost` | [oracle.py:186-188](lmcos/src/data/preprocess_mc/oracle.py#L186) |
| `continue_value[t]` | `−continue_cost(N_t,T_t) + next_value`; `next_value` = `V*(t+1)`, else `timeout_value` if `T_t≤δ`, else `halt_reward[t]` at the data-truncation boundary | [oracle.py:277-285](lmcos/src/data/preprocess_mc/oracle.py#L277) |
| `V*(t)` (oracle value) | `max(halt_value, continue_value)`, **tie → halt** (`≥`) — backward induction | [oracle.py:290-295](lmcos/src/data/preprocess_mc/oracle.py#L290) |
| **`target_advantage[t]`** (controller regression target) | `continue_value[t] − halt_value[t]` (positive ⇒ continue) | [oracle.py:286](lmcos/src/data/preprocess_mc/oracle.py#L286) |
| `OSS` (optimal stop step) | cost-aware DP stop from `t`: `t` if halting, else propagate | [oracle.py:293,297](lmcos/src/data/preprocess_mc/oracle.py#L293) |
| greedy decision rule | stop at **first** `t` with `advantage(t) ≤ 0`, else last step | [oracle.py:152-157](lmcos/src/data/preprocess_mc/oracle.py#L152) |
| `GSS` (greedy stopping step) | `argmax(bmi == bmi[-1])` — first step the eventual-best move appears (zero-cost stop) | [filter_trees_by_trace.py:63](lmcos/src/data/filter_trees_by_trace.py#L63) |
| PUCT-stable filter | `bmi[0] ≠ bmi[-1]  AND  bmi[mid] ≠ bmi[-1]` | [filter_trees_by_trace.py:62](lmcos/src/data/filter_trees_by_trace.py#L62) |
| monotone-convergence filter | `all(bmi[gss:] == bmi[-1])` — best move, once found, never abandoned (acts on **move identity**, not value) | [filter_trees_by_trace.py:64](lmcos/src/data/filter_trees_by_trace.py#L64) |
| `value_gain[t]` (VG, new) | `V*_zerocost(t) − halt_reward[t] ≥ 0` — DP value with cost disabled (reuses the oracle, captures reversals) | [value_gain.py:74](lmcos/src/data/preprocess_mc/value_gain.py#L74) |
| `Gain` / VOC (analysis-side) | `Q_final[bmi[-1]] − Q_final[bmi[1]]` (human-analytics, not the MC target) | labnotebook:190 |

**Known consequence of the `halt_reward` definition** (verified, not inferred): once `bmi[t]` locks onto
the final move the curve is flat at the trajectory max (gain ≡ 0 after convergence; 100% of `packed/mc`
trajectories); any non-monotonicity is **pre-convergence only** (best-so-far move identity hopping among
candidates of non-monotone final Q).

---

## 1. Source data
- **Origin:** `/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/human_trees` — Lc0 search trees, one
  `.pt` per human board position (root FEN), generated by ysagiv. Live/growing set.
- **Snapshots used (NOTE the drift):** the **tree filter** ran over **640,605** files; the
  **partition** was taken at **640,928** files (a few hundred more, written between the two steps).
  This is a real provenance wart — the two stages were not pinned to one frozen snapshot.
- **Per-tree payload (fields we use):** `oracle_best_move_index` (per-expansion argmax trace, 96
  steps), `oracle_final_root_q_values`, `node_features` (5 feats), `parent_index`, `oracle_root_q_trace`,
  `oracle_root_moves`. Node features `node_feat=5`.

## 2. Tree filtering → "informative-stopping" subset
- **Code:** [filter_trees_by_trace.py](lmcos/src/data/filter_trees_by_trace.py). **Commit:** feb712e.
- **Criterion = INTERSECTION of two tree-level tests** on `bmi = oracle_best_move_index`,
  `final = bmi[-1]`, `gss = argmax(bmi == final)`:
  1. **PUCT-stability** — `bmi[0] != final AND bmi[len//2] != final` (search materially changes the
     chosen move). → 209,496 (**32.7%**).
  2. **Monotone-convergence** — `all(bmi[gss:] == final)` (once the eventual-best is first found it's
     never abandoned). → 418,374 (**65.3%**).
- **Result:** intersection **101,812 clean (15.9%)**. Stats: `filtered/filter_stats.json`.
- **Rationale + provenance note:** a post-hoc 96-step read of ysagiv's PUCT-stability filter
  (`oracle96_trace_filtered`); the monotone test is ours. **Deviation flag:** ysagiv's original ran
  a separate 16-node search; we approximate from the 96-step trace.

## 3. Train/test split
- **Code:** [partition/make_partition.py](partition/make_partition.py). Example-level 50/50, **seed 0**.
- Full partition over 640,928 trees → 320,464 train / 320,464 test (`partition/`, marked TEMPORARY).
- **Filtered split = clean_trees ∩ partition side:** **50,674 train / 51,138 test**
  (`filtered/{train,test}_clean.txt`; full-path manifests `packed/split_filtered/{train,validation}_manifest.txt`).

## 4. GNN packing (encoder training examples)
- **Code:** `cts.data.preprocess_gnn.pack`. **Config:** `packed/configs/gnn_pack_filtered.yaml`
  (`split_root=split_filtered`, `output_root=packed/gnn`, `shard_size=2000`, `clear=true`).
- Direct split→pack (no `derive_prefixes`). Output: `packed/gnn/{train,validation}_manifest.json` +
  sharded examples. **Run:** SLURM array `gnn_pack_array.slurm` (20 chunks) → `gnn_merge.slurm`.

## 5. Encoder pretraining (child-WDL)
- **Code:** `cts.data.build_tree pretrain-child-wdl-encoder`. **Config:** `GNN/configs/gnn_train.yaml`.
  **Run:** `GNN/gnn_pretrain.slurm` (job 10140737, COMPLETED, 1h46m). **Commit (per-epoch ckpts):** dc08907.
- **Architecture (`TreeEncoder`, from `encoder_best.pt` metadata):** `k=1, d_embed=128, d_message=128,
  n_heads=4, d_att=32, node_embed_hidden=128, node_feat=5`. Decoder hidden 128 (discarded after pretrain).
- **Objective:** `search_consolidated_edge_wdl_v1` — per-edge child-WDL cross-entropy;
  **`loss_weight_by_subtree_size=true`** (edges weighted by subtree size). KL = CE − target_entropy.
- **Optimization:** 45 epochs, batch 128, lr 1e-3, weight_decay 0, `num_workers=0`.
- **Result (epoch 45):** train CE 0.364 / val 0.373; KL (`loss_gap`) train 0.078 / val 0.082;
  train≈val (no overfit). Checkpoint **selected by val loss** → `encoder_best.pt` (+ per-epoch `encoder_epochNNN.pt`).

## 6. MC episode packing (budgeted oracle)
- **Code:** `cts.data.preprocess_mc.pack` ([pack.py](lmcos/src/data/preprocess_mc/pack.py),
  [oracle.py](lmcos/src/data/preprocess_mc/oracle.py)). **Config:** `packed/configs/mc_array_base.yaml`
  (per-chunk) — built into `packed/mc/{train,validation}_manifest.json`. **Run:** array `mc_pack_array.slurm`
  (job 10167793, 20 chunks) → inline merge.
- **Oracle:** `budgeted_controller_v1`. Per tree, a budget bucket is drawn; the oracle runs DP over
  expansion steps to get, per step, `halt_reward`, `continue_value` (post-DP), `tree_size`, and the
  **optimal stop step (OSS)**.
- **`halt_reward[t]` definition (load-bearing; [pack.py:660](lmcos/src/data/preprocess_mc/pack.py#L660)):**
  `halt_reward[t] = oracle_final_root_q_values[ oracle_best_move_index[t] ]` — the **depth-96 (full-search)
  Q-value of the move that is best-so-far at step t**, i.e. what you'd get by committing now to your
  current best move, scored under the *complete* search. It is **NOT** the evolving shallow Q-estimate at
  step t. Consequences (verifiable, not inferred): once `bmi[t]` reaches the final best move the curve is
  **flat at the trajectory max** (gain ≡ 0 after convergence); any non-monotonicity is **pre-convergence
  only**, from the best-so-far move's *identity* hopping among candidates with non-monotone final Q.
- **Cost model (the meta-controller's "cost"; [oracle.py:157-180](lmcos/src/data/preprocess_mc/oracle.py#L157)):**
  - **time cost** `= time_lambda · (convex term in remaining_budget)`, params `time_lambda=18.537,
    time_p=2.8, time_tau=2.5, time_delta=1`.
  - **maintenance cost** `= maintenance_scale · (num_nodes/ref_nodes)^exponent`, **DISABLED**
    (`maintenance_scale=0`; ref 30, exp 1.1).
  - `reward_scale=1.0`, `timeout_value=-1.0`. **`continue_cost = maintenance + time`.**
  - **Episode `oracle_value`** = the value of stopping at the OSS (best achievable under the cost).
- **Episode-level filters:** `min_halt_reward_range=0.05` (**OURS**; ysagiv used **0.1** — deviation),
  `min_decision_margin=0.0` (matches ysagiv). Everything else matches ysagiv's
  `controller_packed_combined_nomaint_no_xaba` manifest (verified field-by-field).
- **Budget buckets (time units):** scramble 1–3, medium-small 4–10, medium-large 11–25, large 26–60,
  very-large 61–120; `samples_per_bucket=2`.
- **Result:** **374,750 train / 378,920 val** episodes (≈3.5% skipped by the halt-range filter).

## 7. Materialize (frozen encoder → z_root cache)
- **Code:** `cts.data.preprocess_mc.materialize`. **Config:** `packed/configs/mc_mat_base.yaml`
  (`encoder_checkpoint=encoder_best.pt`, device cuda; `hidden_dim/hidden_layers` configure a head
  that is **discarded** — only encoder outputs are kept). **Run:** GPU arrays 10168238/9 (16 workers/split)
  → `command=merge`.
- For each MC snapshot, runs the **frozen** encoder over the tree → `z_root` (128-d), then
  **concatenates raw `N_t` (tree size) and `T_t` (remaining budget)** → cache row `[z_t(128), N_t, T_t]`
  (`encode_with_state_features`, [mc.py](lmcos/src/models/mc.py)).
- **Result caches:** `materialized/{train,validation}_cache.pt` = **11,653,016 / 11,780,624** snapshots,
  48 shards each (shards symlinked in `*_cache.pt.d/`).

## 8. Controller training (the readout head)
- **Code:** [controller_train.py](lmcos/src/train/controller_train.py), model
  [MetaController](lmcos/src/models/mc.py). **Config:** `GNN/configs/controller_filtered.yaml`.
  **Run:** `GNN/controller_filtered.slurm` (job 10176390, COMPLETED, 12 min).
- **Model:** frozen encoder + small ReLU MLP (`hidden_dim=256, hidden_layers=3`). **Inputs:**
  `controller_inputs=[z_t, T_t]` — uses root embedding + remaining budget, **drops N_t**.
- **Regression target:** `target_advantage = continue_value − halt_value` (the oracle's **post-DP**
  advantage of continuing; [oracle.py:129](lmcos/src/data/preprocess_mc/oracle.py#L129)).
- **Loss:** `total = advantage_mse + sign_loss_weight · sign_bce` with **`sign_loss_weight=0.1`**;
  `advantage_mse = MSE(pred_adv, target_adv)`, `sign_bce = BCE(sign(target_adv))`.
  ([controller_train.py:232](lmcos/src/train/controller_train.py#L232)).
- **Decision rule (eval):** `predicted_stop_from_advantages` — STOP at the first step with predicted
  `advantage ≤ 0` (a **fixed, untuned** threshold; [oracle.py:143](lmcos/src/data/preprocess_mc/oracle.py#L143)).
- **Optimization:** 3 epochs, batch 18000, lr 1e-3. **Checkpoint selected by min validation
  `average_regret`.**
- **Result:** **best (selected) val regret 0.209**, P(stop==OSS) 0.314; final-epoch 0.288/0.339.
  train sign_accuracy plateaued ≈0.69. (Matches ysagiv's `subtree_weighting_root_budget.yaml`
  hyperparams except data.)

## 9. Evaluation vs baselines
- **Code:** [alt_models_eval.py](lmcos/analysis/_budgeted/alt_models_eval.py). **Metric (all models):**
  `regret = oracle_value − return_for_stop_step(stop_step)`, lower better; + P(stop==OSS).
- **Baselines:** Always-Stop (stop@0), Never-Stop (full budget), Fraction-of-Budget `f*` (stop iff
  `N_t ≥ f·B`, `f` fit on **train** regret, evaluated on **val**).
- **Result (filtered val):** Fraction `f*=0.17` regret **0.077**; Always 0.467; Never 1.388;
  GNN/MC **0.209** (selected). Forward work and the live experiment status: `mc_minimal_plan.md`.

## 10. Reproduce (exact commands)
```bash
# env
source /home/hl4291/venv/bin/activate; export PYTHONPATH=/home/hl4291/chess_analysis/lmcos/src
# 2. filter
python -m cts.data.filter_trees_by_trace --config <filter.yaml>           # -> clean_trees.txt
# 3. split
python partition/make_partition.py --seed 0                                # -> partition/
# 4-5. gnn pack + encoder (SLURM)
sbatch --array=0-19 packed/gnn_pack_array.slurm; sbatch packed/gnn_merge.slurm
sbatch GNN/gnn_pretrain.slurm                                              # -> encoder_best.pt
# 6-7. mc pack + materialize (SLURM)
sbatch --array=0-19 packed/mc_pack_array.slurm   # merge inline
sbatch --array=0-15 --export=ALL,PACKED=.../train_manifest.json,OUTDIR=.../materialized/train,NWORKERS=16 GNN/mc_materialize_array.slurm  # +val; then command=merge
# 8. controller
sbatch GNN/controller_filtered.slurm                                       # -> controller_filtered.pt
# 9. eval
python -m analysis._budgeted.alt_models_eval --packed-root packed/mc --out-dir figures
```

## 11. File map (where everything lives)
| what | path |
|---|---|
| source trees | `/scratch/.../ysagiv/.../human_trees/*.pt` |
| filter stats | `/scratch/.../hl4291/filtered/filter_stats.json` |
| filtered split | `/scratch/.../hl4291/filtered/{train,test}_clean.txt`, `packed/split_filtered/*` |
| GNN packs | `/scratch/.../hl4291/packed/gnn/` |
| encoder + per-epoch ckpts | `/scratch/.../hl4291/GNN/encoder_best.pt`, `encoder_epochNNN.pt` |
| MC packs | `/scratch/.../hl4291/packed/mc/` (+ `mc_chunks/`) |
| materialized caches | `/scratch/.../hl4291/packed/materialized/{train,validation}_cache.pt` |
| controller + metrics | `/scratch/.../hl4291/GNN/controller_filtered.{pt,_metrics.json,_diagnostics.jsonl}` |
| all configs | `/scratch/.../hl4291/{packed/configs,GNN/configs}/*.yaml` |
| figures | `/home/hl4291/chess_analysis/figures/{regret,stop_eq_oss}_by_model.png` |
```
```
