# Lab Notebook

This file is the running experimental record for the project.

What belongs here:
- Meaningful changes to objectives, training setups, diagnostics, or evaluation.
- Summaries of important runs, including intent, key parameters, and outcome.

What does not belong here:
- Tiny plumbing edits with no experimental consequence.
- Incidental refactors or formatting-only changes.

## 2026-04-04

### Encoder pretraining path repaired

Intent:
- Get the tree encoder pretraining path working reliably on packed tensorized data.

Meaningful change:
- Fixed the encoder pretraining path and produced a usable checkpoint.

Result:
- Encoder pretraining completed successfully.
- Checkpoint used afterward:
  - `/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/checkpoints/tree_encoder_pretrain_interactive.pt`

Notes:
- This became the reference encoder checkpoint for later controller experiments.

## 2026-04-05

### Synthetic PPO diagnostic added and cleaned up

Intent:
- Determine whether PPO/controller learning is fundamentally broken independent of the real tree representation.

Meaningful changes:
- Added `scripts/controller_synthetic_diagnostic.py`.
- Switched the synthetic diagnostic to bypass the learned encoder and use deterministic one-hot timestep features.
- Added a multi-scenario synthetic suite.

Result:
- PPO learned the correct halt/continue policy on clean synthetic tasks.
- This ruled out a gross end-to-end PPO failure in the simplest setting.

Conclusion:
- The controller pipeline can learn correct stopping rules on well-posed toy problems.

### Evaluation/env sequencing bug fixed

Intent:
- Verify that controller evaluation and deterministic control diagnostics were not being contaminated by env-handling bugs.

Meaningful changes:
- Fixed `evaluate_controller(...)` so it does not recreate the first episode every time.
- Fixed deterministic env sequencing in `scripts/controller_representation_control.py`.

Result:
- Oracle-control evaluation became trustworthy.

Conclusion:
- Some earlier controller-control failures were polluted by evaluation/env bugs.

### Oracle-stop-step control diagnostic built

Intent:
- Test the production controller path with trivial representations.

Meaningful changes:
- Added `scripts/controller_representation_control.py`.
- Added exact stop-step accuracy, first-action accuracy, confusion matrix, and trace support.

Result:
- With the old weak oracle dataset, performance was seed-sensitive and fragile.
- Full traces showed divergence concentrated at the state:
  - `label_1 | phase_1`
- Bad runs received early value/advantage pressure toward `continue` at that state and then stopped sampling `halt`.

Conclusion:
- The failure was localized to the second decision in a weak-margin diagnostic, not a broad controller collapse.

### Weakness of the original oracle dataset identified

Intent:
- Understand why the oracle-stop-step control task was so fragile.

Finding:
- The old regret-style oracle dataset often produced cases like:
  - `oracle_0`: `[0, 0, 0]`
  - `oracle_1`: `[-x, 0, 0]`
  - `oracle_2`: `[-x, -x, 0]`

Interpretation:
- The critical `oracle_1` second decision was often only better by the continue-cost margin.

Conclusion:
- The original oracle diagnostic was too weak to be a decisive controller test.

### Controller objective changed from regret-only to absolute full-reference move quality

Intent:
- Use a better-founded reward: score halting by the quality of the selected move under the full planner, not by regret relative to the final best move.

Meaningful changes:
- In `supervised_branch.py`, halt reward changed from:
  - `Q_full(best_move_now) - Q_full(best_move_full)`
  to:
  - `Q_full(best_move_now)`
- The same objective change was propagated to:
  - `scripts/controller_representation_control.py`
  - `scripts/generate_controller_oracle_dataset.py`
  - `scripts/analyze_controller_optimal_actions.py`

Conclusion:
- Controller reward is now in absolute full-reference move-quality units.

### Strong oracle dataset regenerated under the new objective

Intent:
- Rebuild the oracle control diagnostic under the new absolute-value reward with clearer decision margins.

Meaningful change:
- Generated:
  - `/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/controller_oracle_strong_abs`

Result:
- Example halt-reward sequences became meaningfully separated.
- Oracle-stop-step control sweeps on this dataset achieved exact stop-step accuracy `1.0` across all tested seeds.

Conclusion:
- Under the stronger objective and stronger diagnostic dataset, the control path behaves reliably.

## 2026-04-06

### Real-data oracle analysis under the new absolute-value objective

Intent:
- Measure whether the real collected dataset actually contains a meaningful adaptive stopping signal under the new reward.

Run:
- `scripts/analyze_controller_optimal_actions.py`
- Data: real train/validation manifests

Important parameter sweep:
- `continue_cost = 0.1`
  - Real-data oracle was mostly immediate halt.
- `continue_cost = 0.001`
  - Real-data oracle became meaningfully adaptive.

Validation-set oracle summary for `continue_cost = 0.001`:
- `total_examples = 3581`
- `mean_first_halt_reward = 0.107454`
- `mean_optimal_value = 0.167093`
- `should_continue_rate = 0.437308`
- Oracle mean expansions from the stop-step histogram:
  - `1.8073`

Conclusion:
- The real dataset does contain a meaningful adaptive stopping signal under the new objective.
- `continue_cost = 0.001` is the appropriate regime for a genuinely adaptive controller.

### Real controller training under the new absolute-value objective

Intent:
- Train the real controller on the real collected dataset under the new objective with low continue cost.

Run configuration:
- Command: `supervised_branch_cli.py frozen-rl-generated`
- Train data:
  - `/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/pretrain_split/train_manifest.txt`
- Validation data:
  - `/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/pretrain_split/validation_manifest.txt`
- Encoder checkpoint:
  - `/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/checkpoints/tree_encoder_pretrain_interactive.pt`
- Continue cost:
  - `0.001`

Early finding:
- With the old rollout collector, `num_envs` semantics were wrong.

### Major PPO rollout collection bug found and fixed

Intent:
- Compare the implementation to standard PPO semantics.

Meaningful change:
- In `FrozenEncoderControllerTrainer._collect_rollout()`, rollout collection was changed from:
  - stepping one env per rollout step
  to:
  - stepping every env on every rollout step

Why this mattered:
- Before the fix:
  - `num_envs = 16`, `rollout_steps = 256` collected only `256` total transitions.
- After the fix:
  - it correctly collects `4096` transitions.

Regression protection:
- Added a test in `test_supervised_branch.py` asserting collected rollout size is:
  - `num_envs * rollout_steps`

Conclusion:
- Prior real PPO runs used a substantially smaller and noisier effective batch than intended.

### PPO diagnostics added

Intent:
- Track whether PPO is behaving healthily instead of relying only on surrogate losses.

Meaningful changes:
- Added per-update PPO diagnostics:
  - `approx_kl`
  - `clipfrac`
  - `explained_variance`
- Added `--validation-interval` to the controller CLI and Slurm job.

Conclusion:
- Training logs now include both PPO-health metrics and periodic validation metrics.

### Interactive smoke test after the rollout fix

Intent:
- Verify that the corrected rollout semantics and new diagnostics run cleanly on GPU before launching a long Slurm job.

Run configuration:
- `num_envs = 16`
- `rollout_steps = 256`
- `minibatch_size = 128`
- `continue_cost = 0.001`
- `num_updates = 5`
- `validation_interval = 1`

Outcome:
- No crashes.
- PPO diagnostics looked healthy:
  - `approx_kl` stayed small
  - `clipfrac` stayed modest
  - `explained_variance` quickly became strongly positive
- Validation average expansions moved from very small to around `3.0` during the short smoke run.

Conclusion:
- The corrected PPO implementation and diagnostics were healthy enough to justify a full cluster run.

### Full Slurm run: job `6590724`

Intent:
- Run the real controller at scale under the corrected PPO semantics and new objective.

Key parameters:
- Objective: absolute full-reference move quality
- `continue_cost = 0.001`
- `num_envs = 16`
- `rollout_steps = 256`
- `minibatch_size = 128`
- `validation_interval = 50`
- Requested walltime:
  - `12:00:00`

Outcome:
- The job timed out at update `480 / 1000`.
- Controller remained numerically stable through the end.

Last logged training metrics:
- `update = 480`
- `policy_loss = -0.001246`
- `value_loss = 0.007091`
- `entropy = 0.044301`
- `approx_kl = 0.000626`
- `clipfrac = 0.008179`
- `explained_variance = 0.935112`
- `mean_episode_return = 0.133493`
- `mean_episode_length = 9.441`
- `halt_rate = 0.372`

Last logged validation metrics:
- `update = 450`
- `average_return = 0.120145`
- `average_expansions = 9.125000`
- `average_terminal_quality = 0.194199`

Interpretation:
- Training was technically healthy and stable.
- Relative to the offline validation oracle:
  - controller return was below oracle (`0.120` vs `0.167`)
  - controller used much more search (`9.125` expansions vs oracle `1.807`)
- The learned policy is still over-continuing.

Additional local analysis:
- Downloaded and analyzed:
  - `logs/cts-controller_6590724.out`
- Analysis artifacts:
  - `analysis_outputs/controller_run_6590724/report/summary.json`
  - `analysis_outputs/controller_run_6590724/report/updates.csv`
  - `analysis_outputs/controller_run_6590724/report/validations.csv`
  - `analysis_outputs/controller_run_6590724/report/training_metrics.png`
  - `analysis_outputs/controller_run_6590724/report/validation_metrics.png`

Notable summary numbers:
- Best logged validation return:
  - `0.121552` at update `50`
- Last logged validation return:
  - `0.120145` at update `450`
- Oracle return ratio:
  - about `0.719`
- Oracle expansion ratio:
  - about `5.05`

Conclusion:
- The full run did not fail numerically.
- It learned a stable but still over-searching controller.
- Later training did not improve validation return over the early checkpoint.

### Representation probe added

Intent:
- Test whether frozen encoder root states contain enough information to predict the offline oracle halt/continue decision on real snapshots without involving PPO.

Meaningful change:
- Added:
  - `scripts/probe_controller_representation.py`

What it does:
- Loads real snapshot states from raw pretrain examples.
- Computes offline oracle halt/continue actions under the current absolute-value controller objective.
- Extracts frozen encoder root embeddings in batches.
- Trains a small MLP probe on those embeddings.
- Reports:
- overall accuracy
- first-action accuracy
- per-phase accuracy

Follow-up improvement:
- Added cached dataset support so the expensive oracle-label + embedding pass can be paid once and reused across probe sweeps.
- Added progress logging during sample loading and embedding extraction.

Conclusion:
- This is the next discriminating test if PPO remains flat: determine whether the representation carries the oracle signal at all.

### RL sanity harness added

Intent:
- Directly verify RL gradient wiring instead of inferring correctness from long noisy training curves.

Meaningful change:
- Added:
  - `scripts/rl_sanity_checks.py`
- Added regression coverage in:
  - `test_supervised_branch.py`

What it does:
- Runs three controller-RL sanity checks against the real PPO and REINFORCE code paths:
- one-batch manual-sign tests
- gradient and parameter norm inspection
- fixed-state halt-logit drift over several updates on simple toy environments

Why this matters:
- The PPO run was numerically stable but flat on validation.
- The REINFORCE baseline also showed suspiciously little movement early.
- This harness is intended to answer a narrower question: are updates flowing with the correct sign and reaching the intended modules at all.

Initial result:
- Ran the harness locally on CPU with short settings (`drift_updates=2`).
- PPO:
  - manual-sign tests passed for both positive and negative halt updates
  - encoder, halt head, and value head all had nonzero gradient norms
  - halt-favored logit drifted strongly positive; continue-favored drifted negative
- REINFORCE:
  - manual-sign tests passed for both positive and negative halt updates
  - encoder and halt head had nonzero gradient norms
  - value head correctly had zero gradient because REINFORCE does not use it
  - halt-favored logit drifted positive; continue-favored drifted negative

Conclusion:
- There is now direct evidence that both RL implementations are wired correctly at the local gradient-update level.
- The flat long-run training curves are therefore more likely to reflect optimization/task issues than a gross sign or disconnected-gradient bug.

Cluster result:
- Ran the same harness on the cluster with the real encoder checkpoint and default controller architecture.
- JSON artifact:
  - `/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/analysis/rl_sanity_checks.json`

Observed outcomes:
- PPO:
  - manual-sign tests passed
  - encoder, halt head, and value head all had nonzero gradient norms and parameter deltas
  - halt-favored fixed-state logit drifted strongly positive over 5 updates
  - continue-favored fixed-state logit drifted strongly negative over 5 updates
- REINFORCE:
  - manual-sign tests passed
  - encoder and halt head had nonzero gradient norms and parameter deltas
  - value head correctly remained unused (`0` grad, `0` parameter delta)
  - halt-favored fixed-state logit drifted strongly positive over 5 updates
  - continue-favored fixed-state logit drifted negative over 5 updates

Interpretation:
- This is strong evidence against a gross gradient-sign or disconnected-parameter bug in either RL pathway.
- The remaining failures on real training runs are more likely due to optimization dynamics, distribution shift, objective mismatch, or representation limits than basic gradient wiring.

### Unfrozen encoder RL run analyzed (`6621831`)

Intent:
- Test whether unfreezing the tree encoder helps the real controller objective, since the frozen PPO run flatlined early.

Run summary:
- Job:
  - `6621831`
- Local artifacts:
  - `analysis_outputs/controller_run_6621831/cts-controller_6621831.out`
  - `analysis_outputs/controller_run_6621831/report/summary.json`
  - `analysis_outputs/controller_run_6621831/report/training_metrics.png`
  - `analysis_outputs/controller_run_6621831/report/validation_metrics.png`
- Frozen vs unfrozen comparison artifacts:
  - `analysis_outputs/controller_run_6621831/comparison/comparison_summary.json`
  - `analysis_outputs/controller_run_6621831/comparison/training_comparison.png`
  - `analysis_outputs/controller_run_6621831/comparison/validation_comparison.png`

Key results:
- Unfrozen best validation:
  - return `0.118722` at update `50`
  - expansions `12.125`
- Unfrozen last validation:
  - return `0.116597` at update `150`
  - expansions `14.15625`
- Frozen best validation:
  - return `0.121552` at update `50`
  - expansions `7.5625`

Conclusion:
- Unfreezing the encoder did not help on this run.
- It was worse than the frozen baseline on validation return, terminal quality, and search efficiency.
- Like the frozen run, the best validation point still appeared very early and then deteriorated.

## 2026-04-06

### Real-data trivial control: `oracle_action_now`

Intent:
- Remove representation ambiguity on the real episode distribution by feeding the controller only the current offline oracle halt/continue action.

Meaningful change:
- Extended:
  - `scripts/controller_representation_control.py`
- Added regression coverage in:
  - `test_controller_representation_control.py`

What changed:
- New representation mode:
  - `oracle-action-now`
- Each observation is now a 2d one-hot:
  - `[1, 0]` for continue
  - `[0, 1]` for halt
- This representation intentionally hides phase and episode identity, so it is a stricter real-data control than `oracle-stop-step`.

Validation:
- Targeted unit tests passed for the new observation encoding and env behavior.

Purpose:
- If PPO succeeds with `oracle-action-now` on real data, the remaining bottleneck is much more likely to be representation/state construction than the RL machinery itself.

Follow-up:
- An independent audit found that `evaluate_oracle_agreement(...)` was miswired for `oracle-action-now`.
- The evaluation env was accidentally instantiated with the default representation instead of the requested one, so the oracle-agreement footer from the first `oracle-action-now` run was invalid.
- Fixed the evaluator wiring and added a regression test for `oracle-action-now` agreement evaluation.

Additional change:
- Added `--min-decision-margin` to the control script.
- This allows real-data control runs to filter episodes to only those with strong optimal-action margins under the current objective.

Follow-up change:
- Replaced the control diagnostic policy/value heads with linear readouts on top of the fixed-feature encoder.
- This makes `oracle-action-now` directly interpretable:
  - halt logit = `w_continue * action_0 + w_halt * action_1 + b`
- The control script now prints the learned halt/value readout weights for small feature sets.

Additional diagnostic:
- Added `--force-oracle-actions` to `scripts/controller_representation_control.py`.
- This diagnostic collects PPO rollouts by stepping the environment with the oracle action encoded in `oracle-action-now` observations, while still scoring the model log-probability and value for that forced action.

Purpose:
- Separate local gradient plumbing from on-policy exploration.
- If ordinary `oracle-action-now` PPO collapses because continuing only pays off under correct future actions, forced-oracle rollouts test whether PPO moves the linear readout correctly when future actions are constrained to be optimal.

Follow-up diagnostic:
- Added `--one-step-oracle-bandit`.
- This converts real oracle-action states into a balanced one-step contextual bandit:
  - observation `[1, 0]` means continue
  - observation `[0, 1]` means halt
  - matching the encoded action gives reward `+1`
  - the opposite action gives reward `-1`
  - the episode terminates immediately

Purpose:
- This is the clean PPO plumbing diagnostic for the oracle-action representation.
- It removes sequential stopping dynamics, future-policy dependence, and transition-count imbalance, while still using the same PPO trainer, tensorizer, and linear policy/value model.

Result:
- The one-step bandit diagnostic achieved:
  - `average_return = 1.000`
  - `bandit_action_accuracy = 1.000`
- Learned linear halt logits:
  - `logit([1, 0]) = -6.854`
  - `logit([0, 1]) = 6.997`

Conclusion:
- PPO plumbing, action encoding, tensorization, Bernoulli sign convention, and linear readout capacity are correct in the immediate-reward setting.

Additional diagnostic:
- Added `--supervised-oracle-action`.
- This trains the same linear halt readout with supervised BCE on balanced `oracle-action-now` states extracted from the selected sequential real episodes.

Purpose:
- Check whether the sequential oracle-action labels are internally coherent and linearly learnable once RL temporal credit assignment is removed.
- The diagnostic also evaluates the supervised readout through the tensorized sequential stop-step evaluator.

Result:
- The supervised sequential oracle-action diagnostic achieved perfect reported accuracies.
- Learned linear halt logits in the latest run:
  - `logit([1, 0]) = -0.990`
  - `logit([0, 1]) = 1.118`

Conclusion:
- The sequential oracle-action labels are coherent and linearly learnable.
- The remaining `oracle-action-now` PPO failure is therefore isolated to sequential on-policy credit assignment/exploration, not diagnostic wiring or representation capacity for the toy control.

### Offline fitted-Q controller path added

Intent:
- Add an RL-flavored alternative to PPO that avoids the on-policy continuation-trajectory discovery problem.

Meaningful change:
- Added:
  - `scripts/train_fitted_q_controller.py`
- The script trains a frozen-encoder controller to predict Bellman action values:
  - `Q(s_t, continue) = -continue_cost + V*(s_{t+1})`
  - `Q(s_t, halt) = halt_reward_t`
  - action order is `[continue, halt]`
- The greedy policy halts iff:
  - `Q_halt >= Q_continue`

Purpose:
- Use the known deterministic snapshot trajectory structure directly.
- Preserve an RL/value-learning framing while removing PPO’s on-policy exploration trap.

Validation:
- Added focused regression tests in:
  - `test_train_fitted_q_controller.py`
- Tests cover Bellman target construction, episode-batch collation, and greedy stop-step behavior.

Initial smoke result:
- On a 1000-train / 500-validation smoke run, plain two-Q MSE achieved:
  - `validation_q_mse = 0.024`
  - `validation_action_accuracy = 0.507`
  - `greedy average_return = 0.148`
  - `greedy average_oracle_value = 0.164`
  - `greedy exact_stop_step_accuracy = 0.166`

Interpretation:
- Return was reasonably close to oracle, but action/boundary accuracy was weak.
- This indicates that plain Q-value MSE can fit common value level while missing the decision margin.

Follow-up change:
- Reparameterized the Q head as:
  - raw output 0: `Q_continue - Q_halt`
  - raw output 1: `Q_halt`
- `forward()` still returns comparable action values in `[Q_continue, Q_halt]` order.
- Training loss now optimizes:
  - halt-value MSE
  - continue-advantage MSE, weighted by `--advantage-loss-weight`

Purpose:
- Train the actual stopping decision variable directly:
  - continue iff `Q_continue - Q_halt > 0`

### 2026-04-07: Advantage-only compute controller

Intent:
- Stop treating the stopping controller as a two-Q regression problem when the deployed decision only depends on the compute advantage:
  - `A_compute(s) = Q_continue(s) - Q_halt(s)`

Meaningful change:
- Simplified `scripts/train_fitted_q_controller.py` so the controller head now predicts one scalar:
  - `A_compute`
- Training loss is now direct MSE on the Bellman-derived counterfactual compute advantage.
- Greedy evaluation now uses the explicit stopping rule:
  - continue iff `A_compute > 0`
  - halt otherwise
- Logged metrics now emphasize:
  - `advantage_mse`
  - `mean_abs_advantage_error`
  - `sign_accuracy`
  - `average_regret`

Purpose:
- Remove common-mode value fitting from the controller objective.
- Train the learned search rule directly: whether another unit of search is worth its compute cost.

Validation:
- `python3 -m py_compile scripts/train_fitted_q_controller.py test_train_fitted_q_controller.py`
- `/opt/miniconda3/envs/cts_supervised/bin/python -m unittest test_train_fitted_q_controller.py`
- `/opt/miniconda3/envs/cts_supervised/bin/python scripts/train_fitted_q_controller.py --help`

Follow-up change:
- Added `--representation oracle-action-now` to `scripts/train_fitted_q_controller.py`.
- This replaces each real tree snapshot with a 2d one-hot root feature:
  - `[1, 0]` for oracle continue
  - `[0, 1]` for oracle halt
- In this mode, the model uses a fixed-feature linear advantage readout instead of loading the TreeNN encoder.
- Added `--min-decision-margin` so low-margin episodes can be filtered for the toy sanity check.

Purpose:
- Make the counterfactual advantage objective testable on the same trivial representation used for prior PPO diagnostics.
- Expected behavior on strong-margin data:
  - high `sign_accuracy`
  - high greedy stop-step accuracy
  - `advantage_readout_weight[0] + bias > 0`
  - `advantage_readout_weight[1] + bias <= 0`

Follow-up optimization:
- The initial toy path still used the lazy episode `DataLoader`, which rebuilt raw tree episodes each epoch.
- Fixed this for the tensorizable frozen-encoder paths:
  - `--representation oracle-action-now`: materialize selected train/validation episodes once, then train/evaluate from `TensorDataset(features, target_advantages)`
  - frozen `--representation tree`: run the frozen encoder once to materialize root embeddings and target advantages, then train the advantage head on tensors
  - greedy evaluation uses precomputed per-episode tensors for both materialized paths

Purpose:
- Keep the diagnostic faithful to the compute-advantage objective while avoiding repeated raw `.pt` loading, oracle recomputation, and frozen-encoder forward passes every epoch.

Overnight run setup:
- Added `slurm/train_compute_advantage_della.slurm`.
- Defaults:
  - full train/validation manifests
  - frozen pretrained encoder
  - real tree representation
  - materialized root embeddings
  - `continue_cost = 0.001`
  - `epochs = 50`
  - checkpoint: `/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/checkpoints/tree_controller_compute_adv_full.pt`

## Going forward

Any future entry should include:
- Date
- Intent
- Meaningful change or run configuration
- Outcome
- Conclusion
