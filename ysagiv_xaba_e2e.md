# ysagiv `human_trees`, xaba branch -- Phase 2 Agent 6: does e2e (joint encoder+head) training help?

Record of Phase 2 Agent 6. Question: Agent 1 (`ysagiv_trees.md`) found the strongest positive
z_t result in this whole investigation on the ysagiv xaba-filtered population -- a FROZEN
encoder z_t-Controller significantly beats both SingleHalt\* and Stats-Controller at
`lambda=0.0005, m=0`. Separately, Agent 2 (`our_trees_continued.md`, sub-line A) found that
**e2e training** (jointly unfreezing the encoder and training against the real PG
expected-regret objective, `cts.train.e2e_controller_train`, instead of freezing the encoder and
only fitting a readout on top) is the ONLY lever anywhere in this investigation that beat
SingleHalt\* even once on our OWN corpus -- weak, non-replicated evidence (1 win out of 12
regime x epoch points), but the only one. **Nobody had tried e2e training on the ysagiv
xaba-filtered population.** This track does that: does e2e training improve on the already-strong
frozen-encoder result there, the way it (weakly) did on our own corpus, hurt it, or wash out?

**Status: PAUSED by the coordinator (user request to pause the whole investigation and review
results so far). Zero epochs finished. No results exist yet — none were ever produced, not
even a partial one.** The job chain below was submitted and epoch-1 training (`10847125`) was
mid-run when the coordinator cancelled it. Verified directly, not assumed:

- `sacct -j 10847125,10847134,10847135,10847136` shows all 4 jobs `CANCELLED` (epoch-1 training
  cancelled by signal after `00:05:38` elapsed; the other 3 jobs never started — `CANCELLED` with
  `00:00:00` elapsed, since each was gated behind a since-cancelled `afterok` dependency).
- Epoch-1 training's own log shows it had completed only **batch 600 of 1047** (~57%) of the
  train-split pass of its single planned epoch when it was killed (`running_mean_loss=0.172029
  elapsed_s=318.4`, then nothing further; stderr: `JOB 10847125 ... CANCELLED ... DUE to SIGNAL
  Terminated`). It had not even reached the end-of-epoch validation pass, let alone the
  checkpoint-save step that only fires after both train and val passes complete
  (`e2e_controller_train.main`'s `_save_checkpoint` calls happen strictly after `run_epoch` returns
  for both phases).
- `find /scratch/gpfs/GRIFFITHS/hl4291/lmcos/ysagiv_xaba_e2e -type f` returns **nothing** —
  the checkpoint dir (`e2e_checkpoints/`) and materialized-cache dir were never populated.
  **No checkpoint of any kind — full, per-epoch, "latest", or encoder-only sidecar — exists on
  disk.** There is genuinely nothing to evaluate yet, not a partially-usable artifact.

So: not "epoch 1 done, epoch 2 pending" and not "results pending" in the sense of "just waiting on
a queued job" — this is a clean, complete stop with **zero training progress banked**. Per the
coordinator's explicit instruction, **nothing has been or will be resubmitted**. The methodology,
config, and scripts below are all still valid/ready and require no changes to resume later — only
re-submission (starting again from `train epoch 1`, since no checkpoint survived to resume from
partway) is needed if/when this track is picked back up.

## Methodology

**What e2e training changes vs. the frozen baseline.** Everywhere in `ysagiv_trees.md`
(and `ysagiv_xaba_regime_sweep.md`, `ysagiv_xaba_capacity_sweep.md`), the "z_t" comparison
works by: (1) materializing `z_t` from a FROZEN, already-trained child-WDL encoder
(`tiny_encoder.pt`, trained via `pretrain-child-wdl-encoder`, 6 epochs, never updated again), then
(2) `analysis.ysagiv_sig`'s `compute_pairwise_significance` (via `analysis.evaluate`'s
`_fit_stop_controllers`) fits a FRESH 200-epoch PG readout on top of that frozen `z_t` for the
"zt" controller. **The encoder itself never sees the downstream regret objective** in that
pipeline -- only the readout head is ever optimized against it. e2e training removes exactly that
restriction: `cts.train.e2e_controller_train` builds a `MetaController` WITHOUT calling
`freeze_encoder()`, and trains encoder + head JOINTLY against the same closed-form expected-regret
loss (`pg_controller_train.expected_regret_batched`) via one joint optimizer. Concretely, this
track's contribution is a NEW ENCODER (the e2e run's `unfreeze_encoder=true` output), which is
then run back through the SAME `ysagiv_sig` pipeline (fresh 200-epoch readout on top) used for
every other number in this whole ysagiv track -- so the comparison isolates exactly one variable,
whether the encoder itself was ever exposed to the regret objective, holding the readout-fitting
procedure fixed.

**Checkpoint initialization.** `train.encoder_checkpoint` in the new, additive
`config_ysagiv_xaba_e2e.yaml` points read-only at Agent 1's already-trained frozen encoder:
`/scratch/gpfs/GRIFFITHS/hl4291/lmcos/ysagiv_xaba/packed/tiny_encoder.pt` (verified to exist via
`find`/`ls` before use, not assumed). `train.packed_{train,validation}_data` likewise point
read-only at Agent 1's already-packed mc-training data
(`/scratch/gpfs/GRIFFITHS/hl4291/lmcos/ysagiv_xaba/mc_packed/{train,validation}_manifest.json`,
8,293 train / 2,093 validation episodes per `ysagiv_trees.md`) -- **no trees regenerated, no
re-splitting, no re-packing.** All of THIS track's own outputs (e2e checkpoints, its own
materialized validation cache, figures) live under a fresh, isolated
`/scratch/gpfs/GRIFFITHS/hl4291/lmcos/ysagiv_xaba_e2e/` scratch dir and
`outputs/figures/ysagiv/xaba_e2e/` figures dir -- confirmed distinct from Agent 1's `ysagiv_xaba`
scratch dir (never written to) and from Agent 4/5's `ysagiv_xaba_regime_sweep`/wide/deep dirs.
Architecture hyperparameters (`d_embed=32, k=1, node_embed_hidden=32, d_message=32, n_heads=2,
d_att=16, hidden_dim=32, hidden_layers=2`) and cost-model defaults are copied unchanged from
Agent 1's `config_ysagiv_xaba.yaml` so the only thing that differs from the frozen baseline is
`unfreeze_encoder: true` plus e2e's own optimizer/objective.

**Config-loading footgun, checked before submitting anything**: `analysis.utils.helpers` reads a
`human_analysis` section at import time (same issue Agent 1 documented); `config_ysagiv_xaba_e2e.yaml`
carries the same stub Agent 1's config uses. `slurm/helpers/setup_env.sh` also unconditionally
queries `encoder.output_checkpoint` -- a stub `encoder:` section (pointing, informationally only,
at Agent 1's `tiny_encoder.pt`) was added so `setup_env.sh` doesn't crash; nothing in this track's
actual chain reads or writes through it. Verified via
`CONFIG=config_ysagiv_xaba_e2e.yaml python3 -c "import analysis.evaluate"` (passed) and by
validating both `ControllerTrainConfig` (stage=`train`) and `MaterializeConfig` (stage=`materialize`)
directly against the merged config before any SLURM submission.

**Training recipe** (`slurm/agent6_ysagiv_xaba_e2e_train_1ep.slurm`, `qos=gpu-test`, routed there
from the start per the documented `gpu-short` contention Agent 2 hit): one epoch per submission,
chained via the existing `resume_checkpoint` hook (`ControllerTrainConfig.resume_checkpoint`,
added by Agent 2 for `our_trees_continued.md` and reused here unmodified) -- each step resumes the
FULL `model_state_dict` (encoder + head) from the previous step's checkpoint, not just the base
encoder. `pg_max_episodes=15000` (matches Agent 2's cap; xaba's train split is 8,293 episodes, so
this cap does not actually truncate an epoch -- included only for parity/safety). Each training
job also writes an encoder-only sidecar checkpoint (`{stem}_encoder.pt`, in
`gnn_pretrain.save_encoder_checkpoint`'s format) that `materialize.py` consumes directly, same
mechanism `our_trees_continued.md` used to re-materialize per-epoch.

**Evaluation.** `slurm/agent6_ysagiv_xaba_e2e_materialize_eval.slurm`: for each epoch's encoder
sidecar, re-materializes `z_t` on Agent 1's read-only ysagiv-xaba VALIDATION manifest (own scratch
output, `ysagiv_xaba_e2e/materialized/validation_cache_epoch{N}.pt`), then runs
`python -m analysis.ysagiv_sig --packed-root $MCP --cache <new cache> --out-dir
outputs/figures/ysagiv/xaba_e2e --tag epoch{N}_<regime>` at the SAME 3 REGIME-confirmed regimes
`ysagiv_trees.md` uses (`lambda=0.01,m=0`; `lambda=0.001,m=0.001`; `lambda=0.0005,m=0`) -- the
exact same script/methodology (paired-bootstrap, `n_boot=2000`, percentile CI, never
normal-theory) Agent 1/4 used, so numbers are directly comparable without re-deriving anything.

**Scope, per direct instruction ("nothing fancy")**: 2 chained epochs (not Agent 2's full 4), each
epoch's checkpoint evaluated at all 3 regimes -- 6 (epoch, regime) points total, not a sweep. If
only epoch 1 clears before this doc is finalized, that is reported plainly as a single data point,
not over-interpreted.

## Job chain (submitted, then CANCELLED by the coordinator before completion — see Status)

| step | job | depends on | what it does | outcome |
|---|---|---|---|---|
| train epoch 1 | `10847125` | -- (fresh from `tiny_encoder.pt`) | `unfreeze_encoder=true`, 1 epoch, `qos=gpu-test` | **CANCELLED** at batch 600/1047 of the train pass (57%), `00:05:38` elapsed, never reached val pass or checkpoint save |
| materialize+eval epoch 1 | `10847134` | `afterok:10847125` | re-materialize val z_t from epoch 1's encoder sidecar, `ysagiv_sig` x 3 regimes | **CANCELLED**, never started (`00:00:00` elapsed) — dependency (epoch 1 training) never reached `afterok` |
| train epoch 2 | `10847135` | `afterok:10847125` | resumes epoch 1's FULL checkpoint, 1 more epoch | **CANCELLED**, never started — no epoch-1 checkpoint ever existed to resume from |
| materialize+eval epoch 2 | `10847136` | `afterok:10847135` | re-materialize val z_t from epoch 2's encoder sidecar, `ysagiv_sig` x 3 regimes | **CANCELLED**, never started |

Checkpoints dir `/scratch/gpfs/GRIFFITHS/hl4291/lmcos/ysagiv_xaba_e2e/e2e_checkpoints/` and
materialized-cache dir `/scratch/gpfs/GRIFFITHS/hl4291/lmcos/ysagiv_xaba_e2e/materialized/` are
both **empty** (verified via `find ... -type f`, zero files). No figures/raw JSON were ever
written to `outputs/figures/ysagiv/xaba_e2e/` either — that directory does not contain any
`ysagiv_sig_*.json` from this track.

**To resume later**: nothing needs to change. `config_ysagiv_xaba_e2e.yaml` and both slurm
scripts (`slurm/agent6_ysagiv_xaba_e2e_train_1ep.slurm`,
`slurm/agent6_ysagiv_xaba_e2e_materialize_eval.slurm`) are unmodified, still valid, and were
verified working (config import/validation checks all passed; the training job itself ran
cleanly for 5m38s with correct data — `train_episodes=8293 val_episodes=2093`, matching
`ysagiv_trees.md` exactly — before being cancelled for scheduling reasons, not a bug). Resuming
means literally re-running the same submission sequence from "train epoch 1" (no partial
checkpoint survived to resume from mid-epoch); see the Job chain section above for the exact
`sbatch`/`--dependency=afterok` sequence used.

## Results

**NONE. Zero epochs completed, zero checkpoints produced, nothing evaluated.** This is not a
"waiting on a queued job" pending state — the chain was cancelled before any training epoch
finished, so there is no e2e-trained encoder of any kind to materialize or score. The table below
is left as a placeholder for when/if this track resumes; every cell is genuinely empty, not
omitted.

| epoch | regime | k\* | SingleHalt\* | Stats | z_t (e2e) | SH-zt (CI) | Stats-zt (CI) | vs. frozen z_t (`ysagiv_trees.md`) |
|---|---|---|---|---|---|---|---|---|
| 1 | lambda=0.01, m=0 | -- | -- | -- | -- | -- | -- | frozen: 0.0805 |
| 1 | lambda=0.001, m=0.001 | -- | -- | -- | -- | -- | -- | frozen: 0.0515 |
| 1 | lambda=0.0005, m=0 | -- | -- | -- | -- | -- | -- | frozen: 0.0217 (frozen z_t's win here: SH-zt +0.0179 [+0.0148,+0.0202], Stats-zt +0.0143 [+0.0104,+0.0185]) |
| 2 | lambda=0.01, m=0 | -- | -- | -- | -- | -- | -- | -- |
| 2 | lambda=0.001, m=0.001 | -- | -- | -- | -- | -- | -- | -- |
| 2 | lambda=0.0005, m=0 | -- | -- | -- | -- | -- | -- | -- |

**Frozen-encoder baseline for reference (reused verbatim from `ysagiv_trees.md`, not
re-derived)**: `n_fit=1,465 / n_eval=628` at every regime.

| regime | k\* | SingleHalt\* | Stats | z_t (frozen) | SH-Stats (CI) | SH-zt (CI) | Stats-zt (CI) |
|---|---|---|---|---|---|---|---|
| lambda=0.01, m=0 | 3 | 0.0834 | 0.0718 | 0.0805 | +0.0116 [+0.0035,+0.0210] Stats wins | +0.0029 [-0.0037,+0.0108] ns | -0.0087 [-0.0183,+0.0004] ns (barely) |
| lambda=0.001, m=0.001 | 4 | 0.0533 | 0.0462 | 0.0515 | +0.0072 [-0.0026,+0.0175] ns | +0.0019 [-0.0031,+0.0078] ns | -0.0053 [-0.0147,+0.0042] ns |
| lambda=0.0005, m=0 | 95 | 0.0396 | 0.0360 | 0.0217 | +0.0036 [+0.0001,+0.0060] Stats wins (barely) | **+0.0179 [+0.0148,+0.0202] z_t WINS** | **+0.0143 [+0.0104,+0.0185] z_t WINS** |

(diffs are `a - b`; a positive, CI-excludes-0 value means `b` significantly beats `a`, i.e. lower
regret. "z_t WINS" = the frozen z_t significantly beats that baseline -- the headline win this
track is testing whether e2e training preserves, improves, or erodes.)

## Verdict

**No verdict possible — this track was paused before producing any data.** Not "inconclusive
results," not "a wash," not "one data point pending interpretation" — literally zero e2e training
epochs completed, zero checkpoints exist, and zero significance tests were run. The motivating
question ("does e2e training improve on the strong frozen-encoder xaba result the way it weakly
did on our own corpus?") remains completely open on this population. If/when this track resumes,
the plan is unchanged from before the pause: 1-2 chained 1-epoch e2e runs (per the original
"nothing fancy" instruction), each evaluated at the same 3 regimes, with the same explicit caution
against over-interpreting a single significant point that `our_trees_continued.md` sub-line A's
epoch-2-then-non-replication history already demonstrates is necessary.
