# gnn_train.md — Part 4: profile + train the child-WDL GNN encoder

**Status:** plan draft. Awaiting **[Q#]** answers before profiling/training. The
model is **already implemented** — this is profiling, a loss dashboard, and a full
train on the part-1 split, not new modeling.

Related: [[partition]], [[eval]], [[tree-set-and-gss]], [[tree_gen]].

---

## 0. What exists today

- **Model:** `ChildWdlModel` (`cts.models.gnn`), a message-passing encoder. Trainer
  [gnn_pretrain.py](lmcos/src/train/gnn_pretrain.py): Adam, per-edge **cross-entropy**
  of predicted child WDL vs teacher WDL targets. This matches your description —
  from the parent edge embedding `E(parent, v)` predict `WDL(v)`, `v` a sinusoidal
  child-slot embedding.
- **Metrics already logged** per pass: `total_loss` (CE averaged over supervised
  edges), `target_entropy` (the CE's information floor), and `loss_gap =
  total_loss − target_entropy` — **the real KL we drive toward zero**. Optional
  per-bucket KL grid (`log_bucketed_kl`).
- **Data pipeline:** [preprocess_gnn/](lmcos/src/data/preprocess_gnn/): `split.py`
  → `derive_prefixes.py` → `pack.py` → `teacher_targets.py`
  (`PackedTensorizedShardDataset`).
- **Already smoke-tested:** `/scratch/.../hl4291/tmp/pretrain_smoke/` holds
  `smoke_bucketed_kl.jsonl` (loss log), `smoke_encoder.pt` /
  `_encoder_decoder.pt` / `_resume.pt` checkpoints, `smoke_gpu.yaml`, and
  `smoke.slurm`. Launch path:
  [pretrain_child_wdl_encoder_della.slurm](lmcos/slurm/2_pretrain_encoder/pretrain_child_wdl_encoder_della.slurm).
  So loss-decreases-on-small-batches is already observed; this plan scales it up.

---

## 1. Deliverable A — in-flight loss dashboard

The trainer already emits metrics; we need them **visible and monitorable** mid-run:
1. Ensure each train/val pass appends a JSONL row (`step`/`epoch`, `total_loss`,
   `target_entropy`, `loss_gap`, `num_supervised_edges`, wall-clock). The smoke run
   already writes `*_bucketed_kl.jsonl`; reuse that path.
2. A small **live plot** script (`tail` the JSONL → `loss_gap` vs step + a rolling
   ΔCE) refreshed on an interval, written to `figures/` so progress is glanceable.
3. Print a one-line-per-interval log: `step, CE, KL(loss_gap), ΔCE, edges/s, ETA`.

## 2. Deliverable B — time-to-converge estimate

Estimate wall-clock for CE to converge to **ΔCE < 1e-5** on the current ysagiv
dataset, **without a full run**: from a short profiling run measure (a)
edges/sec throughput on the target GPU, (b) the empirical CE-decay curve over the
first K steps, then extrapolate steps-to-threshold × time/step. Smoke profiling
runs stay ≤ ~2 min on data subsets (subagent-eligible, per your note).

## 3. Deliverable C — full train on the part-1 split

1. **Match the partition.** Reuse [[partition]] 1A manifests so GNN train/test ==
   the FEN split parts 1–3 share. (**[Q1] RESOLVED:** source = `human_trees`. The
   `generated_trees` default dir doesn't exist; per user we pack `human_trees`,
   never create `generated_trees`.)
2. Pack tensorized shards from `human_trees` for train/test (the partition
   manifests) into **`/scratch/gpfs/GRIFFITHS/hl4291/packed/gnn/{train,test}/`**.
3. Train with checkpointing; on convergence (or budget) **save the final encoder**
   to `/scratch/gpfs/GRIFFITHS/hl4291/GNN/` (per your suggestion) with
   `save_encoder_checkpoint` (arch + state + metadata: dataset snapshot, split
   seed, config hash).

---

## Open questions
- **[Q1] Tree population.** Train the GNN on `human_trees` (the canonical analysis
  set, what the partition splits) or `generated_trees` (the `split.py`/packer
  default)? If they differ, which is ground truth and do they share FENs? *Pending
  — blocks B and C.*
  - packer default
- **[Q2] Convergence criterion.** "ΔCE < 1e-5 step-to-step" — per **minibatch step**
  (very noisy) or per **epoch** (stable)? And on `total_loss` (CE, has an entropy
  floor so it won't reach 0) or on `loss_gap` (KL → 0)? *Assumption:* per-epoch ΔKL
  (`loss_gap`) < 1e-5, with CE reported alongside.
  - epoch
- **[Q3] Hyperparameters / config.** Reuse the smoke config (`smoke_gpu.yaml`)
  architecture (`k`, `d_embed`, `d_message`, `n_heads`, lr, batch, epochs), or is
  there a target "production" config? Profiling and the final train should use the
  same arch. *Need the intended config.*
  - use whichever config ysagiv used before (see either smoke or git history)
- **[Q4] Hardware.** Which della GPU partition for profiling and for the full run
  (the slurm already targets della)? Throughput/ETA are GPU-specific. *Assumption:*
  same GPU as `smoke.slurm`.
  - gpu-short, *Not* gpu-test
- **[Q5] Checkpoint cadence + retention.** Every N steps / epochs; keep all or
  best-by-val? *Assumption:* every epoch + best-by-val, all retained under
  `/scratch/.../hl4291/GNN/`.
  - every n-steps + best-by-val
- **[Q6] Stop condition for the full run.** Train to the ΔCE threshold, or a fixed
  epoch/time budget with the threshold as an early-stop? *Assumption:* fixed-epoch
  budget with ΔKL early-stop, whichever first.
  - first, the KL early stop + fixed-epoch budget (can extend this if need be, report thoroughly).

## Tests
- **`test_loss_logging_monotone_on_smoke`** — a few steps on the smoke subset: CE
  and `loss_gap` are finite, decreasing on average, `loss_gap ≥ 0`.
- **`test_edge_wdl_target_shapes`** — teacher WDL targets are valid distributions
  (sum to 1, ≥ 0) and align with supervised edges.
- **`test_checkpoint_roundtrip`** — `save_encoder_checkpoint` →
  `load_encoder_architecture`/`load_encoder_checkpoint` reconstructs an
  identically-shaped encoder with equal weights.
- **`test_split_matches_partition`** — the GNN packer's train/test FEN sets equal
  the [[partition]] manifests (shared leakage guard with [[eval]]).
- **`test_eta_extrapolator`** — the time-to-converge estimator, given a synthetic
  decaying-CE curve + fixed throughput, returns the analytically known ETA.

---

## Post-profiling execution plan + timing (2026-06-19)

**Status:** pre-pack DONE (`packed/gnn/{train,validation}`, 202,830 / 202,829 examples,
loader-validated); profiling DONE; **training HELD** pending go.

### Phase 1 — encoder training (child-WDL CE → KL)
- **Config:** ysagiv smoke arch — `k=2, d_embed=128, d_message=128, n_heads=4`, Adam
  `lr=1e-2`, subtree-size-weighted CE. Data = our partition's packed shards.
- **Throughput:** packed → GPU-bound (~100k edges/s, A100-class). Train split ≈ 202,830×
  ~1,640 ≈ **3.3e8 edges/epoch ⇒ ~1–1.3 h/epoch** (+ a cheaper val pass).
- **Convergence:** per-epoch ΔKL(`loss_gap`) < 1e-5 ([Q2]); ~16 epochs is an optimistic
  lower bound (the smoke plateau is overfit noise), **budget 30–60 epochs** with KL early-stop.
- **ETA:** ~18 h (16 ep) · ~1.4 d (30 ep) · ~2.75 d (60 ep) ⇒ **~1–3 days** wall, gated by
  GPU queue + walltime.
- **Mechanics:** della GPU walltime is bounded (gpu-short = hours), so this is a
  **checkpoint+resume chain** of jobs (ckpt every N steps + best-by-val [Q5]; KL early-stop +
  fixed-epoch cap [Q6], whichever first). Save best+final encoder to
  `/scratch/gpfs/GRIFFITHS/hl4291/GNN/`. Monitor with `human_analytics/gnn_kl_dashboard.py`.
- **Note:** `packed/gnn/` shards are symlinks into `packed/gnn_chunks/`; for safekeeping,
  materialize real copies (or keep `gnn_chunks/`) before any cleanup.

### Phase 2 — benchmarking (controller + fill the reserved Part-3 slot)
1. **Materialize controller features** — forward the *frozen* encoder over the MC snapshots to
   cache `z_root` per snapshot (the often-forgotten step;
   `cts.data.preprocess_mc.materialize`). Forward-only pass over the MC packed snapshots,
   **~hours GPU**.
2. **Train the MC controller** — fitted-Q advantage head (MSE + sign-BCE, selected by
   `average_regret`; `controller_train.py`) on the cached features. Small MLP ⇒ **fast
   (~minutes–1 h)**.
3. **Evaluate on the TEST split** — Regret + P(stop==OSS); drop the trained GNN/MC controller
   into the **reserved greyed slot** in `regret_by_model.png` / `stop_eq_oss_by_model.png`
   (vs Always Stop / Never Stop / Fraction `f*`). Encoder's own quality = the KL-convergence
   curve + per-bucket KL (`encoder_kl_audit`).
- **ETA:** ~0.5–1 day (materialize dominates).

### End-to-end: **~1.5–4 days** wall, mostly Phase 1.

### Decisions needed before launch
- **Go/no-go** to start Phase 1 now.
- Confirm the **gpu-short walltime** (sets the resume cadence) — or use a longer GPU QOS if
  available, for fewer restarts.
- Confirm the **epoch cap** (30 vs 60) for the ΔKL early-stop budget.
