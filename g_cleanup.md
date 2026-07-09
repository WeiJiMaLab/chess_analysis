# G — plotting/pipeline cleanup: record of work (2026-07-08)

Self-contained cleanup pass per `plan.md`'s "G — plotting/pipeline cleanup" section, run independently of the parallel REGIME/SIG/CP/Z-EXT investigation. Nothing under `outputs/figures/minply15_maxply75/diagnosis/{regime_select,sig_significance,sig_s_significance,cp_recalibration}.*` or `/scratch/gpfs/GRIFFITHS/hl4291/lmcos/minply15_maxply75/diagnosis/` was touched.

## Summary of new files

| file | purpose |
|---|---|
| `src/analysis/encoder_loss_plot.py` | Sub-item 1: simplified E1 encoder loss plot |
| `src/analysis/controller_learning_curve_plot.py` | Sub-item 2: restyled PG controller curve + baselines |
| `src/analysis/t1_render_trees_svg.py` | Sub-item 3: T1 tree SVG regeneration |
| `g_cleanup.md` (this file) | write-up |

## 1. Encoder proof-of-principle plot — simplified + relocated

**Finding:** grepping the repo for `e1_encoder_loss_vs_epoch` turned up only a docstring cross-reference in `src/cts/train/e2e_controller_train.py` (that module's `_write_history` generates a *different*, Z2-specific plot titled "Z2: end-to-end (encoder-unfrozen) MetaController loss vs epoch"). **No live script produces `e1_encoder_loss_vs_epoch`** — `cts/data/build_tree.py`'s `pretrain_child_wdl_encoder_command` (the real training entry point, `slurm/pipeline/train_encoder.slurm`) only prints per-epoch summary lines to stdout and writes checkpoints; it never plots. `tree_diagnostics.py` and everything T1/T2-adjacent in `src/analysis/` is untracked in git, consistent with a pattern of one-off scripts run once and lost.

**Recovery:** the original run's per-epoch `train_total_loss`/`val_total_loss` survive as stdout in `slurm/logs/train-encoder_10805127.out`, identified by matching mtimes against the current production checkpoint set (`packed/encoder_epoch001..006.pt`, `packed/tiny_encoder.pt`, all written ~34s apart) against ~2000 historical `train-encoder_*.out` logs. Confirmed by viewing the old PDF/PNG directly: metric was "child-WDL total loss," title read "original 6-epoch run vs extended 30-epoch rerun" (task brief's "60-epoch" was an overestimate; real extended run = job `10827818`, `slurm/logs/diag-e1-encoder_10827818.out`, 30 epochs — matches plan.md's "epochs 20-30" plateau text; not touched/replotted, only used to confirm the original run's log).

**Granularity:** per-batch train-loss lines exist, but validation loss is only computed once per epoch, so per-epoch is the finest resolution where train/val are comparable — used per-epoch (6 points), per the task's explicit allowance.

**Wrote:** `src/analysis/encoder_loss_plot.py` — `parse_epoch_summary_lines()` (pure regex parser) + `render_encoder_loss_plot()` (board house style) + CLI `main()` defaulting to the recovered log.

**Before:** `diagnosis/{pdf,png}/e1_encoder_loss_vs_epoch.*` (4 lines: train/val × original/extended). Deleted.
**After:** `normative/{pdf,png}/e1_encoder_loss_vs_epoch.*` (2 lines: train+val, original 6-epoch run only). Verified: PDF 12,503B, PNG 4023×2823 RGBA, visually confirmed sharp drop then plateau.

## 2. PG training curve — baselines + board styling + relocated

**Correcting my initial guess:** `e2e_controller_train.py` does not generate `controller_learning_curve` (same as item 1). Grepped fresh and found the real production trainer: **`src/cts/train/pg_controller_train.py`**, invoked by `slurm/pipeline/train_readout_pg.slurm` (`--set epochs=20 --set pg_episode_batch=1024`) — the exact "full 20-epoch run" plan.md describes.

Its `_save_training_curves()` writes to `/scratch/.../packed/mchalt_controller_training_curve.csv`, whose 20 rows match plan.md exactly (epoch 1 regret=0.8001, plateau at 0.8505 epochs 2-10, down to 0.11753 at epoch 20). No live script promotes this CSV to the styled `diagnosis/{pdf,png}/controller_learning_curve.*` — also untraceable.

**Baselines:** computed Always Stop (k=0) / Always Continue (k=max) directly via `pg_controller_train`'s own `EpisodeMetadata`/`_oracle_config`/`return_for_stop_step` machinery against the same validation split + oracle config that produced the CSV (`time_mode=linear, time_lambda=0.01, maintenance_scale=0.0`). Result: Always Stop=**0.1767**, Always Continue=**0.8505** — the latter is an exact byte-match to the CSV's degenerate plateau, strong independent confirmation of correctness.

**Wrote:** `src/analysis/controller_learning_curve_plot.py` — `always_stop_continue_regret()` (pure, testable) + `_load_history()` + `render_controller_learning_curve()` (board house style: `apply_poster_style()`, `MAIN_COLOR`/`PHASE_COLORS`/`LEGEND_FONTSIZE`) + CLI `main()`.

**Before:** `diagnosis/{pdf,png}/controller_learning_curve.*` (2 lines, ad hoc colors). Deleted.
**After:** `normative/{pdf,png}/controller_learning_curve.*` (4 series: train E[regret], val regret, Always Continue dashed line, Always Stop dotted line). Verified: PDF 17,594B, PNG 4060×2823 RGBA, visually confirmed crossing pattern matches plan.md's narrative. (Task asked only for Always Stop/Continue, not SingleHalt* — no third line added.)

## 3. T1 trees — SVG-only, relocated to `trees/`

**Finding:** `render_tree_graphviz()` in `tree_diagnostics.py` already renders natively to SVG (docstring explains why: raster montages make many-small-boards trees illegible). No driver script survives anywhere (repo, `.claude/worktrees/`, scratch) that calls it or montages outputs into `t1_tree_panel.png`.

**Reconstruction, not copy:** regenerated the 5 trees from `diagnosis/t1_summary.json` (each entry's `src` `.pt` path, `node_cutoff`, recorded red-flag results) plus `tree_diagnostics.py`'s surviving pure helpers. Reverse-engineered the original node-selection budget from surviving asset-folder counts (4/7/12/20/40 files ÷ 2 per node = 4/7/12/20/40 nodes, matching `t1_summary.json`'s `n_branching_nodes` exactly for 4 trees and capped at 40 for `max`'s 89) → `prune_topk_subtree(parent_index, budget=40, intermediate_only=has_children)`.

**Verified before trusting the render:** recomputed `duplicate_fen_check`/`linear_chain_check`/`illegal_move_check` for all 5 trees and asserted exact match against `t1_summary.json` (including `max`'s 133 duplicate FEN pairs) — all passed byte-for-byte, and selected-node counts matched exactly (4/7/12/20/40) before any SVG was written.

**Path bug found+fixed:** first attempt used a relative `out_dir`; graphviz's `dot` subprocess runs with cwd set to the rendered file's own directory (observed: `dot -Kdot -Tsvg -O t1_tree_p10`, bare relative name), breaking the `<IMG SRC>` references. Fixed by resolving `out_dir` to an absolute path — likely why the task brief flagged the old assets' baked-in paths as move-fragile.

**Wrote:** `src/analysis/t1_render_trees_svg.py` — `load_truncated_tree()`, `select_render_nodes()`, `verify_against_summary()` (pure/testable), `render_all()` + CLI `main()`.

**Before:** `diagnosis/t1_tree_{p10,p50,p75,p90,max}.png` + `..._assets/` + `diagnosis/{pdf,png}/t1_tree_panel.*`. All deleted.
**After:** `trees/t1_tree_{p10,p50,p75,p90,max}.svg` + sibling `..._assets/` folders (absolute-path image references). Verified: all 5 well-formed XML (`xmllint --noout`), sizes 3.3KB–31.8KB scaling with node count, one round-tripped through `rsvg-convert` to a valid 435×638 PNG confirming embedded boards resolve correctly.

`t1_summary.json` left in place (still cited by plan.md prose). Confirmed T2/T3/E2/S1/Z artifacts in `diagnosis/` untouched — folder not cleared wholesale. Existing `test_tree_diagnostics.py` (11 tests) still passes.

## 4. Pipeline — `engine_analysis_cp`/`engine_analysis_pwin` commented out

Edited `slurm/pipeline/pipeline.yaml`: commented out both step entries (not deleted, with re-enable comment), updated the ASCII DAG header to mark both `[DISABLED]`. Confirmed `submit_all.sh` reads `steps:` dynamically via `yaml.safe_load` — verified the edited YAML parses cleanly with both steps absent and all 10 other steps present.

Note: `pipeline.yaml` had other unrelated uncommitted changes already present at session start (the labnotebook 2026-07-07 PUCT pivot: `befs1cp_md36`→`puctvalue_md36`, `array: 0-99`→`0-159`, new `argmax_filter` step) — not mine; my diff is scoped to the two commented-out entries and `[DISABLED]` annotations.

## plan.md image embeds

Per the coordinator's confirmation, already updated (lines 16, 20, T1 section split into 5 per-tree SVG embeds) — not re-touched here.

## Open items

- Original driver scripts for all three relocated/regenerated plots are genuinely lost (untracked in git, not in `.claude/worktrees/` or scratch) — this write-up documents the recovery trail used to reconstruct each faithfully.
- No new pytest files for the two plotting scripts beyond their pure functions being individually testable — judged lower priority than T1's stronger correctness bar (exact match against `t1_summary.json`), since a silently-wrong tree reconstruction is much harder to notice than a wrong loss curve.
