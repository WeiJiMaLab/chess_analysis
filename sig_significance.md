# SIG — paired significance checks (SIG-S, SIG-Z)

Both jobs completed successfully before the interruption (`sacct`: `10839167` sig-s-eval, 56s, exit 0:0; `10839166` sig-z-eval, 13m53s, exit 0:0). This file reconstructs the write-up from the SLURM logs (`slurm/logs/sig-s-eval_10839167.out`, `slurm/logs/sig-z-eval_10839166.out`) and the JSON artifacts they wrote, since the agent that ran them was interrupted before writing this report itself.

Methodology for both: paired bootstrap 95% CI (percentile bootstrap, via `cts.stats.bootstrap_ci`, per repo convention — never normal-theory) on the per-episode regret **difference** (baseline − comparison) on the same held-out episodes, at the primary validated regime `time_mode=linear, time_lambda=0.01, maintenance_scale=0.0`. A CI entirely > 0 means the first term significantly has *higher* (worse) regret, i.e. the second method wins; CI straddling 0 means not distinguishable at this sample size.

## SIG-S — Stats-Controller vs SingleHalt*

New script `src/analysis/evaluate_sig_s.py`, run via `slurm/sig_s_eval.slurm`.

| | regret |
|---|---|
| SingleHalt* (k=10) | 0.1204 |
| Stats-Controller | 0.1137 |

Paired diff (SingleHalt* − Stats): **mean = +0.0067, 95% CI [+0.0026, +0.0112] — CONFIRMED, excludes 0.**

**Stats-Controller significantly beats SingleHalt\* at the validated regime.** This was previously only an eyeballed point-estimate win (plan.md flagged the marginal CIs as "nearly touching"); the paired test resolves it cleanly in Stats-Controller's favor. Plot: `outputs/figures/minply15_maxply75/diagnosis/{pdf,png}/sig_s_significance.*`. Raw result: `outputs/figures/minply15_maxply75/diagnosis/sig_s_result.json`.

## SIG-Z — e2e z_t (1 epoch, joint encoder+head) vs the three baselines

Ran the pre-existing (previously unrun) `src/analysis/evaluate_e2e_z3.py`, which already implemented the paired-diff methodology and the decodability tick-label fix — confirmed both work correctly now that it's actually been executed.

| method | regret |
|---|---|
| SingleHalt* (k=10) | 0.1204 |
| Stats-Controller | 0.1137 |
| original (frozen-encoder) z_t | 0.1207 |
| e2e z_t (1 epoch) | 0.1161 |

Paired diffs (baseline − e2e z_t), primary regime `m=0.0`:

| vs. | mean diff | 95% CI | verdict |
|---|---|---|---|
| SingleHalt* | +0.0042 | [−0.0004, +0.0090] | **NOT significant** (straddles 0, barely) |
| Stats-Controller | −0.0024 | [−0.0079, +0.0030] | **NOT significant** (e2e directionally behind, not confirmed) |
| original frozen z_t | +0.0046 | [+0.0000, +0.0091] | **CONFIRMED** (excludes 0) |

**Headline, corrected from plan.md's earlier "promising, not confirmed" framing: e2e z_t significantly beats the OLD frozen-encoder z_t, but does NOT yet significantly beat SingleHalt\* — the point-estimate win over SingleHalt\* was noise, not signal, at 1 epoch of training.** It also doesn't significantly beat or lose to Stats-Controller. This matters for prioritization: `Z-EXT` (more e2e training) is well-motivated by the confirmed win over the frozen encoder, but "e2e beats SingleHalt*" is not yet a real result — more epochs are needed before that claim can be tested for real, not just re-tested at the same sample size.

**Decodability**: e2e z_t alone decodes R(t) at MLP R²=0.111, vs the frozen encoder's R²=0.013 — an order-of-magnitude improvement, consistent with (though not dependent on) the regret story above. Plots: `outputs/figures/minply15_maxply75/diagnosis/z3_e2e_comparison/{pdf,png}/z3_frontier_e2e_vs_baselines_m0p0.*`, `z3_decodability_with_e2e.*`. Raw: `z3_paired_diffs_primary.json`, `z3_maintenance_sweep.json`.

## Bonus finding: independent confirmation of the maintenance_scale collapse

`evaluate_e2e_z3.py`'s built-in sweep at `maintenance_scale ∈ {0.0, 0.05, 0.1}` reproduces T3/Z's known collapse *exactly*, with a new data point: at `m=0.05` **all four methods** (SingleHalt*, Stats, orig z_t, e2e z_t) collapse to identical `0.0090`; at `m=0.1` all four collapse to identical `0.0023`. This is a clean, independent confirmation of the pathology `REGIME` is investigating — and rules out "only 3 of the methods collapse, one is different" as a hypothesis: literally all four converge to the same value, consistent with `SingleHalt*`'s own `k*` collapsing to a fixed degenerate point that every method's fitted/learned policy converges to trivially.

## Bottom line for plan.md

- SIG-S: **done, CONFIRMED** — Stats-Controller > SingleHalt*, real.
- SIG-Z: **done, PARTIALLY confirmed** — e2e z_t > frozen z_t (real), e2e z_t vs SingleHalt*/Stats (not yet significant, needs `Z-EXT`).
- The decodability-plot bug plan.md flagged as open is already fixed in `evaluate_e2e_z3.py` (confirmed working, not just claimed).
