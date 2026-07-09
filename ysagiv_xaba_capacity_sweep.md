# ysagiv `human_trees`, xaba branch — encoder capacity sweep (Agent 5)

Follow-up to Agent 1's `ysagiv_trees.md`: on the xaba-filtered population, that track found z_t
significantly beats BOTH SingleHalt* and Stats-Controller at the λ=0.0005 regime — "the
strongest positive z_t result anywhere in this investigation" — while tying (not losing) at the
other two regimes. This track asks a narrower, direct question: **is the production tiny GNN
encoder's fixed size (`d_embed=32, k=1`) capping that result?** `hypotheses.md`'s S-2 finding
already argues capacity is *probably* not the bottleneck (the same 32-dim architecture, trained
on a stats-reconstruction objective instead of the frozen child-WDL objective, recovers
structural signal at R²=0.84-0.95 vs. 0.07-0.17 under the shipped objective — i.e. the
*objective*, not the *size*, looks limiting) — but that evidence is indirect (different
objective, different corpus). This track tests capacity directly, on the exact population where
z_t's best result lives.

**Status: PAUSED, INCOMPLETE.** The coordinator cancelled every queued/running job for both arms
(2026-07-08) so the user could pause and review the investigation so far — this is not a
"still running, check back later" state, it is a deliberate stop. Neither arm produced eval
results. See the "Status" section at the bottom for exactly how far each arm got (confirmed via
`sacct` and the on-disk artifacts, not just recollection) and what resuming requires. Nothing is
queued or running right now; do not resubmit without explicit direction to resume this track.

## Design

Two independent arms, each varying exactly ONE architectural knob off the baseline
(`config_ysagiv_xaba.yaml`'s encoder: `d_embed=32, k=1`), so any effect is attributable to one
change, not a confound:

| arm | config | d_embed | k | what changed |
|---|---|---|---|---|
| **Baseline** (Agent 1, reused, not retrained) | `config_ysagiv_xaba.yaml` | 32 | 1 | — |
| **Arm B — wider** | `config_ysagiv_xaba_wide64.yaml` | 64 | 1 | embedding width doubled |
| **Arm C — deeper** | `config_ysagiv_xaba_deep_k2.yaml` | 32 | 2 | +1 message-passing round (2-hop receptive field; `cts/models/gnn.py`'s `TreeEncoder` docstring: "Receptive field after k rounds is k hops") |

**Width-linked knobs, scaled in lockstep for Arm B only.** The baseline config ties
`d_embed = d_message = node_embed_hidden = decoder_hidden = 32` AND `n_heads*d_att = 2*16 = 32`
— every width knob in the encoder equals the same value by convention. Nothing in the code
*requires* this (`TreeEncoder`/`TreeAttMsgLayer` accept independent `d_embed`/`d_message`/`d_att`
via plain `nn.Linear`/`nn.GRUCell` — verified by reading `cts/models/gnn.py` and
`cts/models/tree_mha.py`), but doubling only `d_embed` while leaving the others at 32 would
introduce an uncontrolled internal bottleneck (message/attention capacity unchanged while
embedding width doubles) rather than a clean "wider encoder" test. Arm B therefore doubles all
five together (`d_embed=d_message=node_embed_hidden=decoder_hidden=64`, `d_att=32`,
`n_heads=2` unchanged), preserving the baseline's own uniform-width design. Arm C changes only
`k`; every width knob stays at the baseline value.

**Readout head held fixed across all three arms** (`hidden_dim=32, hidden_layers=2` in both
`materialize:` and `train:`) — only encoder capacity is the manipulated variable, not readout
capacity, in either arm.

## What was reused vs. re-run

Per the coordinator's brief: tree generation, `split`, and `mc_pack` are filter/data operations
independent of encoder architecture, so they are **not** re-run. Only the architecture-dependent
stages are:

| stage | re-run? | why |
|---|---|---|
| split / gnn_pack / mc_pack | **no** — reused from Agent 1 | architecture-independent; both new configs' `split_dir`/`mc_packed_dir` point directly at Agent 1's `/scratch/gpfs/GRIFFITHS/hl4291/lmcos/ysagiv_xaba/{split,mc_packed}` (confirmed read-only-safe: `cts.data.preprocess_mc.materialize` only ever *reads* `packed_data`/`manifest_path` from `mc_packed_dir` — all its writes go to the arm's own new `materialized_dir`; see `materialize_worker`/`merge_workers` in `materialize.py`) |
| `encoder.train_dir`/`validation_dir` (gnn_pack's output manifests) | **no** — reused from Agent 1 | encoder pretraining DATA doesn't depend on encoder architecture; both new configs point `encoder.train_dir`/`validation_dir` directly at Agent 1's `.../ysagiv_xaba/packed/{train,validation}_manifest.json` |
| `train_encoder` | **yes**, per arm | architecture-dependent — this is the whole point of the sweep |
| `pack_root` (materialize) | **yes**, per arm | embeds trees through the arm's newly-trained encoder; z_t is architecture-dependent |
| `train_readout_pg` | **yes**, per arm | trains a fresh PG readout head on the arm's own z_t cache |
| `eval` | **yes**, per arm | same 3 regimes as Agent 1, on the arm's own controller + cache |

Every output each arm produces (encoder checkpoint, z_t cache, controller checkpoint, figures)
lives under a NEW isolated scratch subdirectory
(`/scratch/gpfs/GRIFFITHS/hl4291/lmcos/ysagiv_xaba_{wide64,deep_k2}/`,
`outputs/figures/ysagiv/xaba_{wide64,deep_k2}/`) — Agent 1's `ysagiv_xaba/` artifacts and
`ysagiv_trees.md` are untouched.

## Pipeline

Same scripts Agent 1 used, unmodified, driven by the two new configs
(`slurm/pipeline/{train_encoder,pack_root,pack_root_merge,train_readout_pg}.slurm`), chained via
`--dependency=afterok:<jobid>` exactly as `submit_all.sh`'s pattern does. One new script,
`slurm/pipeline/ysagiv_eval_multi_capacity.slurm` — a copy of Agent 1's
`ysagiv_eval_multi.slurm` (not modified) with a single addition: it threads `eval.d_embed`
through as an explicit `--d-embed` flag to both `analysis.evaluate` and `analysis.ysagiv_sig`.
This matters because **both scripts' `--d-embed` CLI default is 32** — correct for the baseline
and Arm C, but silently WRONG for Arm B (`d_embed=64`): a mismatch there would misread the
`T_t` column out of the wrong index of the raw feature array
(`analysis/evaluate.py`'s `_load_zt_by_episode` does `allf[:, :d_embed]` / `allf[:, d_embed]`)
rather than crash, so it would not have been obvious without this fix. Verified before
submitting: `CONFIG=config_ysagiv_xaba_wide64.yaml python3 -c "import analysis.evaluate"` and
the same for `config_ysagiv_xaba_deep_k2.yaml` both succeed (the `human_analysis:` stub
footgun, inherited from `config_ysagiv_xaba.yaml`, is present and correct in both).

Both arms' `pack_root` materialize step ran as a 10-way array (`--array=0-9`), matching Agent
1's actual xaba-branch array size (not the 40-way production default in `pipeline.yaml`, which
targets the much larger production corpus). Same 3 REGIME-confirmed cost points as Agent 1
(`λ=0.01,m=0`; `λ=0.001,m=0.001`; `λ=0.0005,m=0`), so results are directly row-for-row
comparable.

## Results

### Baseline (Agent 1, reused verbatim from `ysagiv_trees.md` — not retrained here)

n_fit=1,465 / n_eval=628 (70/30 split of 2,093 validation episodes).

| regime | k* | SingleHalt\* | Stats | z_t | SH−Stats (CI) | SH−zt (CI) | Stats−zt (CI) |
|---|---|---|---|---|---|---|---|
| λ=0.01, m=0 | 3 | 0.0834 | 0.0718 | 0.0805 | +0.0116 [+0.0035,+0.0210] **Stats wins** | +0.0029 [−0.0037,+0.0108] ns | −0.0087 [−0.0183,+0.0004] ns (barely) |
| λ=0.001, m=0.001 | 4 | 0.0533 | 0.0462 | 0.0515 | +0.0072 [−0.0026,+0.0175] ns | +0.0019 [−0.0031,+0.0078] ns | −0.0053 [−0.0147,+0.0042] ns |
| λ=0.0005, m=0 | 95 | 0.0396 | 0.0360 | **0.0217** | +0.0036 [+0.0001,+0.0060] **Stats wins** (barely) | **+0.0179 [+0.0148,+0.0202] z_t WINS** | **+0.0143 [+0.0104,+0.0185] z_t WINS** |

Decodability (n≈2093, R² predicting R(t)): steps alone=0.105, tree-stats alone=0.238, **z_t
alone=0.339**, all combined=0.531. (diffs are `a − b`; positive + CI excludes 0 means `b`
significantly beats `a`, i.e. lower regret.)

### Arm B — wider (d_embed=64, k=1)

**INCOMPLETE — cancelled mid-`pack_root`.** `train_encoder` (job `10847083`) completed cleanly
(1m20s, exit 0:0; `tiny_encoder.pt` + all 6 `encoder_epoch*.pt` checkpoints are on disk at
`/scratch/gpfs/GRIFFITHS/hl4291/lmcos/ysagiv_xaba_wide64/packed/`). The `pack_root` materialize
array (job `10847088`, 10-way) was running when cancelled — all 10 workers show `CANCELLED`
after 3m34s-5m07s of partial progress (confirmed via `sacct`); `worker_00`..`worker_09`
directories exist under `materialized/train_work/` with SOME shards written, but the array never
finished the `train` split (let alone reached `validation`) for any worker. `pack_root_merge`
(`10847089`), `train_readout_pg` (`10847090`), and `ysagiv_eval_multi_capacity` (`10847091`)
never started (0:00 elapsed, `CANCELLED` while pending on the dependency chain). No z_t cache,
no controller checkpoint, no eval output exists for this arm.

| regime | k* | SingleHalt\* | Stats | z_t | SH−Stats (CI) | SH−zt (CI) | Stats−zt (CI) |
|---|---|---|---|---|---|---|---|
| λ=0.01, m=0 | — | — | — | — | — | — | — |
| λ=0.001, m=0.001 | — | — | — | — | — | — | — |
| λ=0.0005, m=0 | — | — | — | — | — | — | — |

Decodability: not computed.

### Arm C — deeper (d_embed=32, k=2)

**INCOMPLETE — cancelled mid-`pack_root`.** `train_encoder` (job `10847105`) also completed
cleanly (1m22s, exit 0:0; `tiny_encoder.pt` + all 6 `encoder_epoch*.pt` checkpoints are on disk
at `/scratch/gpfs/GRIFFITHS/hl4291/lmcos/ysagiv_xaba_deep_k2/packed/`) before the cancellation
reached this arm. The `pack_root` materialize array (job `10847106`, 10-way) was running when
cancelled — all 10 workers show `CANCELLED` after 1m29s-3m34s of partial progress (shorter than
Arm B's, consistent with the cancellation catching this arm earlier in its run); `worker_00`..
`worker_09` directories exist under `materialized/train_work/` with partial shards, `train`
split unfinished, `validation` split not started. `pack_root_merge` (`10847107`),
`train_readout_pg` (`10847108`), and `ysagiv_eval_multi_capacity` (`10847109`) never started.
No z_t cache, no controller checkpoint, no eval output exists for this arm either.

| regime | k* | SingleHalt\* | Stats | z_t | SH−Stats (CI) | SH−zt (CI) | Stats−zt (CI) |
|---|---|---|---|---|---|---|---|
| λ=0.01, m=0 | — | — | — | — | — | — | — |
| λ=0.001, m=0.001 | — | — | — | — | — | — | — |
| λ=0.0005, m=0 | — | — | — | — | — | — | — |

Decodability: not computed.

## Verdict

**Cannot be determined yet — this track was paused before any eval ran.** Neither arm produced
a single regret number, so nothing below can be answered from data collected so far. This is
NOT a "results are ambiguous" situation; it's a "no results exist" situation. Once resumed and
both arms' eval jobs finish, the question to answer at each regime (especially λ=0.0005, the
winning regime) is whether z_t's regret and its paired-significance margin vs.
SingleHalt\*/Stats move meaningfully off the baseline row:

- If **neither arm** moves the λ=0.0005 result (or the other two regimes) beyond noise, that is
  a real, if negative, answer to "is the tiny GNN the bottleneck?" — consistent with, and
  strengthening, `hypotheses.md`'s S-2 finding (objective, not size, is what's limiting the
  encoder) by showing it holds under a DIRECT capacity manipulation, not just an indirect
  cross-objective comparison.
- If **one arm** moves the result and the other doesn't, that identifies which knob (width vs.
  depth) matters, which is itself informative about the encoder's failure mode (e.g. depth
  mattering would suggest the 1-hop receptive field is starving z_t of subtree structure beyond
  immediate children; width mattering would suggest a raw representational-capacity ceiling).
- If **both arms** move it, that would be evidence against S-2's reading and worth flagging as a
  genuine update, not smoothed over.

This section will be filled in with the actual verdict once the pending jobs complete.

## Status (for anyone resuming this track)

**PAUSED 2026-07-08 by explicit coordinator instruction** (user wanted to stop and review the
whole investigation) — all queued/running jobs for both arms were cancelled via `scancel`.
**Nothing is running. Do not resubmit without a fresh instruction to resume.** This section
records exactly how far each arm got so a resume can pick up from the config files below
instead of restarting from scratch.

**Arm B (wide64)** — chain was `train_encoder`=10847083 → `pack_root`(array 0-9)=10847088 →
`pack_root_merge`=10847089 → `train_readout_pg`=10847090 → `ysagiv_eval_multi_capacity`=10847091.
Confirmed final state via `sacct`:
- `train_encoder` (10847083): **COMPLETED** (0:0, 1m20s). Encoder checkpoint is valid and
  usable: `/scratch/gpfs/GRIFFITHS/hl4291/lmcos/ysagiv_xaba_wide64/packed/tiny_encoder.pt`
  (plus `encoder_epoch001-006.pt`, `encoder_latest.pt`, `tiny_encoder_decoder.pt`,
  `tiny_encoder_resume.pt` — all present).
- `pack_root` array (10847088, tasks 0-9): **CANCELLED mid-run** (each task ran 3m34s-5m07s
  before the signal landed). Partial `worker_00`..`worker_09` shard directories exist under
  `/scratch/gpfs/GRIFFITHS/hl4291/lmcos/ysagiv_xaba_wide64/materialized/train_work/` — do NOT
  treat these as usable; `materialize.py`'s default `RESUME=0` behavior clears a worker's shard
  directory and starts fresh, so simply re-submitting `pack_root.slurm` with the same config is
  safe and will not silently graft onto stale partial shards.
- `pack_root_merge` (10847089), `train_readout_pg` (10847090), `ysagiv_eval_multi_capacity`
  (10847091): **CANCELLED, never started** (0:00 elapsed each — killed while pending on the
  dependency chain, no partial output to worry about).

**Arm C (deep_k2)** — chain was `train_encoder`=10847105 → `pack_root`(array 0-9)=10847106 →
`pack_root_merge`=10847107 → `train_readout_pg`=10847108 → `ysagiv_eval_multi_capacity`=10847109.
Confirmed final state via `sacct`:
- `train_encoder` (10847105): **COMPLETED** (0:0, 1m22s) — this finished before the cancellation
  reached this arm, despite the coordinator's cancellation note describing it as "running when
  cancelled"; `sacct` shows a clean exit. Encoder checkpoint is valid and usable:
  `/scratch/gpfs/GRIFFITHS/hl4291/lmcos/ysagiv_xaba_deep_k2/packed/tiny_encoder.pt` (plus the
  same epoch/decoder/resume checkpoint files as Arm B).
- `pack_root` array (10847106, tasks 0-9): **CANCELLED mid-run** (each task ran 1m29s-3m34s —
  shorter than Arm B's, consistent with the cancellation reaching this arm slightly earlier in
  its run). Same partial-shard situation as Arm B, same fix (fresh resubmit, default
  `RESUME=0` clears and redoes).
- `pack_root_merge` (10847107), `train_readout_pg` (10847108), `ysagiv_eval_multi_capacity`
  (10847109): **CANCELLED, never started.**

**To resume**: both arms' encoder training is DONE and does not need to be repeated — the
checkpoints on disk are complete, valid 6-epoch runs. Resuming is exactly: re-submit
`pack_root.slurm` (`--array=0-9`, `--dependency=` none needed since `train_encoder` already
succeeded) → `pack_root_merge.slurm` → `train_readout_pg.slurm` →
`ysagiv_eval_multi_capacity.slurm`, chained via `afterok` as before, `CONFIG=` pointed at
`config_ysagiv_xaba_wide64.yaml` / `config_ysagiv_xaba_deep_k2.yaml` respectively. No config or
script changes are needed; both configs and `slurm/pipeline/ysagiv_eval_multi_capacity.slurm`
are already correct and were verified (`import analysis.evaluate` under each `CONFIG`) before
the original submission. Once `ysagiv_eval_multi_capacity` completes for both arms, results
live at `outputs/figures/ysagiv/xaba_wide64/ysagiv_sig_regime_*.json` and
`outputs/figures/ysagiv/xaba_deep_k2/ysagiv_sig_regime_*.json` (same JSON shape Agent 1's
`analysis.ysagiv_sig` always produces) — pull the `k_star`/regret/CI fields from those into the
tables above, plus `decodability_data.json` for the R² line, and fill in the Verdict section
honestly (including a null/mixed result if that's what the numbers show — do not overstate a
marginal difference as "capacity matters" without checking the CI, per this project's
bootstrap-CI-always convention).
