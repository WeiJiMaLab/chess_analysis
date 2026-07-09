# SLURM directory cleanup — closed diagnosis phase

Archived the one-off scripts and logs belonging to the now-closed Diagnosis
phase (`SIG`, `CP`, `REGIME`, `G`, and the original `E1/E2/S1/S2/S3/Z2/Z3`
diagnostic runs) out of `slurm/` into `slurm/archive/` (scripts) and
`slurm/archive/logs/` (matching `.out`/`.err`). Nothing under `slurm/pipeline/`
was touched, and nothing matching `agent1*`/`agent2*`/`agent3*`/`ysagiv*`/
`cp_regen*` was touched. Files are untracked in git (`git status --short`
showed `??` for every `.slurm` moved; the `.out`/`.err` logs are gitignored
via `*.out`/`*.err` in `.gitignore`), so plain `mv` was used throughout, no
`git mv` and no whole-repo git commands.

Verification method for every item below: cross-referenced against
`squeue -u hl4291` (confirmed absent from the queue at time of archiving —
only `cp-regen-gen-trees` (10842167, RUNNING), `train-encoder` (10842093,
PENDING, unrelated script not in scope), and `agent2-e2e-continue-1ep`
(10842808, RUNNING) were live) and `sacct -u hl4291 -X` (confirmed each
job's `State` was `COMPLETED`/`FAILED`/`CANCELLED`, never `RUNNING`/`PENDING`).

## Scripts archived (10) — `slurm/` → `slurm/archive/`

| Script | Job name(s) / IDs | sacct state | Notes |
|---|---|---|---|
| `diagnosis_e1_encoder_longtrain.slurm` | `diag-e1-encoder` (10827818) | COMPLETED 10:30→10:41 | |
| `diagnosis_e2_zt_stats_decodability.slurm` | `diag-e2-zt-stats` (10831415) | COMPLETED 12:06→12:20 | |
| `diagnosis_s1_stats_ablation.slurm` | `diag-s1-stats-ablation` (10832187) | COMPLETED 12:45→12:47 | Result folded into `plan.md`'s S1 section |
| `diagnosis_s2_stats_regression_pretrain.slurm` | `diag-s2-stats-encoder` (10833402) | COMPLETED 13:24→14:05 | Track explicitly marked "dropped, superseded" in `plan.md` line 152 |
| `diagnosis_s2b_stats_regression_pretrain_structural.slurm` | `diag-s2b-structural-encoder` (10834317) | COMPLETED 13:41→14:35 | Same S2/S3-track drop as above |
| `diagnosis_s3_stats_materialize_and_eval.slurm` | `diag-s3-stats-eval` | **never appears in sacct** (job-name never submitted, or ran under a different name) | `plan.md` line 152 explicitly says the S2/S3 track was "dropped, superseded, not revisited" — Z's e2e success tests the same question more directly, so this script is confirmed dead regardless of whether it ever ran |
| `z2_e2e_controller_train.slurm` | `z2-e2e-controller` (10829588 CANCELLED, 10831440 COMPLETED 12:08→12:33) | closed | The *checkpoint* it produced is still used by Agent 2's continuation on scratch — only the script (not needed to rerun) was archived, checkpoint untouched |
| `z3_e2e_materialize_and_eval.slurm` | `z3-e2e-eval` (10831803 FAILED, 10833368 FAILED) | closed (both attempts failed) | Per task brief, superseded by `evaluate_e2e_z3.py` + `sig_s_eval.slurm`/`z3_eval_only.slurm` |
| `sig_s_eval.slurm` | `sig-s-eval` (10839167) | COMPLETED 15:59→16:00 | Feeds `sig_significance.md`, already written up |
| `z3_eval_only.slurm` | `sig-z-eval` (10839166) | COMPLETED 15:59→16:13 | Feeds `sig_significance.md`, already written up |

## Logs archived (22 files, 11 job IDs) — `slurm/logs/` → `slurm/archive/logs/`

`.out`+`.err` pairs for: `diag-e1-encoder_10827818`, `diag-e2-zt-stats_10831415`,
`diag-s1-stats-ablation_10832187`, `diag-s2-stats-encoder_10833402`,
`diag-s2b-structural-encoder_10834317`, `z2-e2e-controller_10831440`,
`z3-e2e-eval_10831803`, `z3-e2e-eval_10833368`, `sig-s-eval_10839167`,
`sig-z-eval_10839166`, and `regime_10639795` (see below — no diag-s3 logs
exist since that job never ran).

### REGIME

No `.slurm` script for the REGIME grid-search was found anywhere in
`slurm/` — it appears to have been run via direct Bash / `src/analysis/regime_select.py`
(as `plan.md` line 15/148 describes) or submitted with `sbatch --wrap` rather
than a checked-in script file, so there was nothing to archive there. Its
job `regime` (10639795) is COMPLETED, from 2026-07-03 (5 days old, long
closed, `regime_select.md` write-up references it) — archived its
`slurm/logs/regime_10639795.{out,err}` alongside the others since the task
asked to also sweep logs for closed diagnosis-phase jobs, even without a
paired script.

## Left alone (found, deliberately not touched)

- **`slurm/cp_regen_gen_trees.slurm`, `slurm/cp_regen_submit_downstream.sh`** — matches `cp_regen*` exclusion (rule 2); `cp-regen-gen-trees` (job 10842167, array) is **RUNNING right now**.
- **All `agent2_*.slurm` files (11 total)** — matches `agent2*` exclusion (rule 2); one (`agent2_e2e_continue_train_1ep.slurm`, job-name `agent2-e2e-continue-1ep`, 10842808) is **RUNNING right now**; the rest are Agent 2's recently-completed debug/eval scripts from the last ~40 minutes, clearly still part of Agent 2's active working set.
- **`z2-pulse-check` (job 10829540, COMPLETED 11:08) and its log `slurm/logs/z2-pulse-check_10829540.{out,err}`** — no corresponding `.slurm` script exists anywhere in `slurm/` (already removed or never checked in), and it wasn't in the candidate list from the task brief. Left alone since there's no script to archive and I couldn't confirm from `plan.md` alone that this specific one-off is fully subsumed elsewhere — listed here as "found but not moved" for the record, not archived.
- **`t3-decode-check` (job 10831713, COMPLETED 12:22) and its log** — same situation: no matching `.slurm` file in `slurm/` to archive, not in the candidate list, left untouched.
- **`train-encoder` (job 10842093, PENDING)** — belongs to `slurm/agent2_e2e_continue_train_1ep.slurm` (confirmed via `--job-name` grep), i.e. Agent 2's file; out of scope, untouched.
- **`slurm/pipeline/`** — not entered/touched at all, per rule 1.

## Result

10 scripts + 22 log files moved into `slurm/archive/` and `slurm/archive/logs/` respectively. No whole-repo git commands were run; each move was a plain `mv` preceded by a `git status --short` check (all `.slurm` files were untracked `??`; log files are gitignored). `squeue -u hl4291` re-checked immediately before and after the moves — no archived job's name reappeared.
