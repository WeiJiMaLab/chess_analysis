# Lab Notebook

## Project overview

The motivation is to build a planning model that keeps the abstract tree-search scaffolding but replaces hand-written decision rules with neural networks at each decision point. Conceptually: a meta-controller chooses act in the real environment vs plan inside an internal tree. The act path uses a policy at the root of the current tree; the plan path uses a planning head over the tree representation to choose planning actions—navigation (e.g. move focus in the tree) and expansion / evaluation steps that update the tree via a learned world model and value feedback—before committing to a real move. In principle such a model would be trained AlphaZero-style with self-play.

Current work is intentionally narrower: meta-control of search only (e.g. when to continue expanding vs when to halt), on teacher-generated search trees and offline targets. Given snapshots of a growing search tree, a controller decides whether to continue expanding search or halt and act on the current best move. The setting is offline: trajectories come from teacher search on positions (e.g. drawn from the Lichess database with simple filters), and supervision is derived from counterfactual value-of-computation—how halting at each expansion step compares to continuing under a fixed continue cost and halt rewards defined from the search state.

The main representation pipeline is neural encoding of search trees. A TreeNN-style encoder embeds each snapshot; training combines encoder pretraining (e.g. child-WDL and related targets) with fitted Q-style training of a scalar compute-advantage head that predicts whether continuing is better than halting, rather than policy-gradient RL on the same objective. The scientific questions include whether a simple meta-controller can learn optimal control given this TreeNN representation, what tree encoding features or control features (e.g. continue cost architecture) support optimal control learning, and to what extent the controller's behaviour matches real human choices. That is the first slice that must work on real tree encodings and real halt/continue tradeoffs.

Later, the same tree representation is meant to support a full planning head whose action space includes concrete planning operations (which node to expand, which to evaluate, etc.), still inside the same overall loop sketched above. The lab notebook records experiments along that path—encoders, packing, fitted-Q controller training, diagnostics, and evaluation—not incidental refactors.

### Pipeline stages

The full data-to-model pipeline has six stages. Each stage's output feeds directly into the next.

#### 1. FEN sampling

Starting positions are sampled from a Lichess game database via SQL (DuckDB). The query selects ~200k games from 2023 with both players rated 1800–2600 and at least 20 half-moves. From each sampled game, one board position is drawn uniformly at random (subject to ply 8–120, 2–60 legal moves, 8–32 pieces). A final reservoir sample reduces the set to ~100k FENs. The output is a flat text file of FEN strings and an accompanying Parquet table with game metadata (ELO, time control, opening, piece counts). See `sql/download_FENs.py`.

#### 2. FEN filtering

Not all positions produce informative search trees. A PUCT-based stability filter runs a short tree search (default 16 nodes, same PUCT logic used for full tree generation) on each FEN and records the best root move at 1 expansion, at a midpoint (~5 expansions), and at the full budget. A FEN is kept only if the best move at 1 expansion differs from the best move at the full budget **and** the best move at the midpoint also differs from the full-budget best move. This selects positions where search materially changes the chosen action, so the resulting trees will have nontrivial stopping structure for the controller. This step is embarrassingly parallel and is submitted as sharded Slurm jobs. See `scripts/filter_fens_by_puct_stability.py` and `scripts/submit_puct_filter_shards.py`.

#### 3. Tree generation

Each filtered FEN becomes a PUCT search tree. The expansion loop starts from the root position and repeatedly selects a leaf via the PUCT formula (`Q(s,a) + c_puct * P(a|s) * sqrt(N(s)) / (1 + N(s,a))`), expands it by querying an lc0 engine for child priors, values, and WDL vectors, then backpropagates the leaf's value up the selection path (negating at each ply for alternating perspective). WDL vectors are backpropagated alongside scalar values. Expansion continues until a node budget is reached (currently fixed at 96 nodes).

During expansion, an oracle trace is recorded: after every expansion step, the current best root move and Q-values for all root children are saved. This trace allows later stages to reconstruct any prefix of the search trajectory without re-running the engine.

After expansion completes, teacher targets are extracted. Node value targets are visit-weighted averages of child Q-values. Edge WDL targets are visit-weighted averages of the WDL vectors accumulated during backpropagation, perspective-flipped to the parent's viewpoint. These targets, together with the tree structure, per-node scalar features (value, WDL, prior), and the oracle trace, are saved as a serialized `PretrainExample`.

Variable-size trees can also be derived from a full 96-node tree by sampling a prefix node count from a log-uniform distribution and recomputing targets on the truncated tree, avoiding additional engine calls. See `cts_pretrain.py` (core logic) and `supervised_branch_cli.py` (CLI entry points).

#### 4. Encoder pretraining

The tree encoder is a GNN (class `TreeNN` in `GNN.py`) that operates on variable-size trees packed into a single flat batch. Each node starts with a 5-dimensional feature vector (value, WDL win/draw/loss, WDL variance) embedded through a two-layer MLP into a `d_embed`-dimensional state. The encoder then runs `k` rounds of alternating upward (children→parent) and downward (parent→children) message passing. In the upward pass, each parent aggregates its children's states via a multi-head attention layer (`TreeAttMsgLayer`) conditioned on sinusoidal slot embeddings that encode each child's position among its siblings (sorted by UCI move string). In the downward pass, each child receives a linear projection of its parent's state. Both directions update node states through a shared GRU cell.

Two propagation modes exist. In **synchronous** mode, all nodes update simultaneously from the previous round's states, giving a receptive field of `k` hops. In **asynchronous** (sequential) mode, the upward pass processes nodes from leaves to root in depth order and the downward pass from root to leaves, so each node sees already-updated neighbors. With `k = 1` in asynchronous mode, every node's receptive field covers the entire tree regardless of depth. The current encoder uses asynchronous mode with `k = 1`.

The encoder output is the set of all node states plus the root state for each tree in the batch. Trees are batched by flattening all nodes into a single tensor with a CSR (compressed sparse row) child-pointer structure and a tree-index vector that maps each node back to its source tree.

Pretraining uses the **child-WDL objective**: a slot-conditioned decoder MLP takes the concatenation of a parent node's state and a sinusoidal slot embedding and predicts a 3-way softmax over win/draw/loss for each parent→child edge. The loss is cross-entropy against the search-consolidated edge WDL targets from tree generation. The pretraining loop iterates over packed tensorized shards with Adam, tracking loss gap (cross-entropy minus target entropy) as the primary validation metric. The best encoder checkpoint (by validation loss) is saved and used as the frozen backbone for controller training. See `GNN.py`, `cts_pretrain.py` (class `ChildWdlPretrainer`), and `supervised_branch_cli.py` (`pretrain-child-wdl-encoder` subcommand).

#### 5. Controller episode packing

Raw `PretrainExample` trees are converted into budget-augmented controller episodes for offline training. This is a multi-step process.

First, each source tree's oracle trace is used to reconstruct a trimmed **snapshot episode**: a sequence of tree snapshots at each expansion step, together with the best root move and its Q-value at each snapshot. The episode is trimmed to start at the first expansion where the root has at least one child (i.e., the first real decision point). The **halt reward** at each snapshot is the full-tree Q-value of the root move that is best at that snapshot—measuring what you would get if you stopped searching there and committed to that move, evaluated under the complete search.

Second, each source tree is replicated across synthetic **starting budgets**. Five budget buckets span the range from scramble (1–3 time steps) to very-large (61–120 time steps), with 2 deterministic samples per bucket (seeded by a hash of the source path). For each sampled starting budget, the budgeted oracle computes the optimal stopping policy via backward induction:

- At each planning step `t`, **halt value** = halt reward at step `t`.
- **Continue value** = −`c_continue(N_t, T_t)` + `V*(t+1)`, where `c_continue` is the sum of a maintenance cost (proportional to `(N_t / 30)^1.1`, currently scaled to 0) and a time cost (a decreasing-marginal function of remaining budget `T_t`, parameterized by `λ`, `p`, `τ`).
- The oracle halts at step `t` iff halt value ≥ continue value; otherwise it continues.
- The **target advantage** at each step is `continue_value − halt_value`. Positive means "continue is better"; negative means "halt is better."

If the starting budget expires before the episode ends, a large negative timeout value (−1) is used as the continuation value.

Before packing, source trees are filtered: trees whose halt-reward range across the episode falls below a threshold (default 0.10) are discarded, and trees exhibiting `X*AB*A` root-move churn patterns (where the best move oscillates back to a previously abandoned move) can be excluded. An optional hierarchical `dj` stratification rebalances the training set across stop-depth-excess bins (how deep the oracle searches on average) and budget-action-variance bins (how much the oracle's halt/continue decisions vary across budgets), to reduce overrepresentation of trivially easy trees.

The output is packed tensorized shards: each shard stores per-step node features, parent/child indices, depth, edge slots, halt rewards, target advantages, tree sizes, time budgets, oracle stop steps, and starting budgets for a batch of episodes. See `scripts/pack_controller_episodes.py`, `budgeted_controller_oracle.py`, and `cts_episode_envs.py`.

#### 6. Controller training

The controller predicts the **compute advantage** `A(s) = Q_continue(s) − Q_halt(s)` at each planning step and halts when `A(s) ≤ 0`. The model (class `ComputeAdvantageTreeSearchModel` in `scripts/train_fitted_q_controller.py`) consists of the pretrained TreeNN encoder (typically frozen) plus an MLP advantage head. The encoder processes each step's tree snapshot and produces a root embedding; this is concatenated with two scalar state features (current tree size `N_t` and remaining time budget `T_t`) to form the advantage head's input. The MLP head (configurable width and depth; default 3 layers of 256 units) outputs a scalar predicted advantage. An optional separate sign head shares the MLP backbone but has its own final projection for a binary continue/halt logit.

Training minimizes a weighted combination of advantage MSE and sign BCE (binary cross-entropy on the sign of the advantage). The sign loss directly targets the decision boundary rather than relying on squared-error alone, which can be insensitive to small wrong-sign predictions near zero.

To avoid repeated frozen-encoder forward passes, the training script supports **materialization**: on the first run for a given packed dataset and encoder checkpoint, all per-step root embeddings are computed once and cached to disk alongside the scalar features and targets. Subsequent runs load these cached tensors directly and train only the advantage head on a flat `TensorDataset`.

**Greedy evaluation** runs the trained model's stopping rule on validation episodes: starting from the first planning step, it predicts advantages and halts at the first step where `A(s) ≤ 0` (or at the episode's last step). The resulting stop step determines the achieved return (halt reward minus accumulated continue costs), which is compared to the oracle's optimal return. Reported metrics include exact stop-step accuracy, first-action accuracy, average return, average oracle value, average regret, and average number of expansions used. Per-episode diagnostics (predicted advantages, oracle/predicted stop steps, difficulty scalars, regret decomposition) can be written to JSONL for offline analysis.

See `scripts/train_fitted_q_controller.py` for the full training loop, and `scripts/analyze_budgeted_controller_run.py` for post-hoc analysis of trained controllers.

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

Frozen run result:
- Log analyzed:
  - `cts-compute-adv_6645287.out`
- Local analysis directory:
  - `analysis_outputs/compute_advantage_run_6645287/`
- Materialization time:
  - train: approximately `4097.9s`
  - validation: approximately `221.0s`
- Best validation sign accuracy:
  - `0.435` at epoch `15`
- Best greedy return:
  - epoch `5`
  - `average_return = 0.157`
  - `average_oracle_value = 0.167`
  - `average_regret = 0.010`
  - `exact_stop_step_accuracy = 0.076`
  - `first_action_accuracy = 0.480`
- Final greedy evaluation:
  - `exact_stop_step_accuracy = 0.065`
  - `first_action_accuracy = 0.468`
  - `average_return = 0.156`
  - `average_oracle_value = 0.167`
  - `average_regret = 0.011`
  - `average_expansions = 11.644`

Conclusion:
- The advantage-only counterfactual objective works on the toy `oracle-action-now` representation, but the frozen real-tree representation still does not expose the compute-advantage boundary strongly enough for a simple head.
- Return is close to oracle in absolute value, but stop-step and sign metrics remain poor.

Unfrozen run result:
- Log analyzed:
  - `cts-adv-unfrozen_6645314.out`
- Local analysis directory:
  - `analysis_outputs/compute_advantage_run_6645314/`
- The run only logged 9 train epochs and one validation/greedy checkpoint, likely because unfrozen training cannot use frozen-embedding materialization and is much slower.
- Best validation sign accuracy:
  - `0.288` at epoch `5`
- Final logged greedy evaluation:
  - `exact_stop_step_accuracy = 0.028`
  - `first_action_accuracy = 0.438`
  - `average_return = 0.155`
  - `average_oracle_value = 0.167`
  - `average_regret = 0.012`
  - `average_expansions = 12.591`

Conclusion:
- The unfrozen run did not improve the compute-advantage controller in the time available.
- It was worse than the frozen run on sign accuracy and greedy boundary metrics, but the run was incomplete and much slower.

### Frozen encoder advantage decodability probe added

Intent:
- Test whether counterfactual compute advantage is decodable from frozen encoder root snapshots independent of the full controller training loop.

Meaningful change:
- Extended `scripts/probe_controller_representation.py` with:
  - `--probe-target action`
  - `--probe-target advantage`
- The advantage probe trains the existing linear/MLP probe heads to regress:
  - `A_compute(s) = Q_continue(s) - Q_halt(s)`
- It reports:
  - `mse`
  - `mae`
  - `sign_accuracy`
  - `first_sign_accuracy`
  - per-phase sign accuracy
- Existing action-probe caches remain usable. If a cache was created before advantage targets were stored, the script backfills advantage targets from the selected raw paths without recomputing frozen encoder embeddings.

Purpose:
- Separate representation decodability from the full compute-advantage trainer and check whether the frozen encoder root embedding contains enough information for the scalar value-of-computation boundary.

Validation:
- `python3 -m py_compile scripts/probe_controller_representation.py test_probe_controller_representation.py`
- `/opt/miniconda3/envs/cts_supervised/bin/python -m unittest test_probe_controller_representation.py`
- `/opt/miniconda3/envs/cts_supervised/bin/python scripts/probe_controller_representation.py --help`

Follow-up:
- Added `slurm/probe_advantage_decodability_della.slurm` to submit the full-dataset frozen advantage probe without hand-writing the command on the cluster.
- Defaults:
  - full train/validation manifests
  - pretrained frozen encoder checkpoint
  - shared probe cache directory
  - `--probe-target advantage`
  - `--probe-type mlp`
  - `--probe-epochs 50`

Validation:
- `bash -n slurm/probe_advantage_decodability_della.slurm`

## Going forward

Any future entry should include:
- Date
- Intent
- Meaningful change or run configuration
- Outcome
- Conclusion

## 2026-04-10

Intent:
- Correct the uncertainty-encoder pretraining target semantics after discovering that the earlier child-WDL path was decoding raw child valuehead WDLs instead of search-consolidated edge WDLs.

Meaningful change:
- Extended search backup statistics to carry WDL vectors alongside scalar values in `cts_pretrain.py`.
- `compute_teacher_targets(...)` and `consolidate_generated_tree(...)` now produce per-edge search-consolidated WDL targets:
  - backed up through search with ply-wise win/loss perspective flips
  - aggregated by visit-weighted averaging
- `PretrainExample` now persists `edge_wdl_targets` for those consolidated edge targets.
- Reintroduced a slot-conditioned child-WDL encoder pretraining path:
  - canonical sibling slots from lexicographic `incoming_move_uci`
  - sinusoidal slot encoding plus learned projection in the encoder/decoder
  - child-WDL trainer logs target entropy and KL gap (`loss_gap`)
- Updated packed tensorized shards to store `edge_slot` and optional `edge_wdl_targets`.
- Scrubbed a stale WDL validation message in `schema.py`.

Outcome:
- Focused and full validation passed:
  - `/opt/miniconda3/envs/trm/bin/python -m pytest test_supervised_branch.py -q`
  - `/opt/miniconda3/envs/trm/bin/python -m pytest test_plumbing.py test_model.py -q`
  - `/opt/miniconda3/envs/trm/bin/python -m pytest -q`
  - `/opt/miniconda3/envs/trm/bin/python supervised_branch_cli.py pretrain-child-wdl-encoder --help`
  - `/opt/miniconda3/envs/trm/bin/python scripts/pack_pretrain_examples.py --help`
- Final test result: `91 passed`

Conclusion:
- The branch is back to a semantically aligned child-WDL pretraining pipeline.
- The remaining work is operational: regenerate raw examples so they actually carry the corrected `edge_wdl_targets`, then rerun any packing/training on top of those regenerated examples.

## 2026-04-10

Intent:
- Reuse the existing fixed-size WDL trees by deriving variable-size root prefixes instead of recollecting fresh raw trees from lc0.

Meaningful change:
- Added prefix-derivation helpers in `cts_pretrain.py`:
  - `prefix_node_count_schedule(...)`
  - `sample_prefix_expansion_count_for_node_budget(...)`
  - `derive_prefix_pretrain_example(...)`
- Added `scripts/derive_pretrain_prefixes.py`:
  - reads existing raw pretrain examples
  - samples one root prefix per source example with actual node count in `[min_nodes, max_nodes]`
  - recomputes consolidated scalar and edge-WDL targets on the prefix
  - saves new raw `PretrainExample`s with throughput logging

Outcome:
- Validation:
  - `/opt/miniconda3/envs/trm/bin/python -m pytest test_supervised_branch.py -q`
  - `/opt/miniconda3/envs/trm/bin/python scripts/derive_pretrain_prefixes.py --help`
  - `/opt/miniconda3/envs/trm/bin/python -m pytest -q`
- Final result: `93 passed`

Conclusion:
- We can now build a variable-size raw dataset from the existing 96-node WDL trees without new engine calls.

## 2026-04-13

### Episode difficulty metrics (data-only) and greedy-eval diagnostics

Intent:
- Quantify how “interesting” each controller episode is using only halt rewards and `continue_cost`, and log per-episode diagnostics during greedy evaluation so regret can be related to those metrics.

Meaningful change:
- Added `episode_difficulty.py` with six metrics: halt reward range, optimal-vs-second-best return gap, return variance, softmax entropy over per-stop returns, regret of always halting at step 0, and reward curvature (sign changes in halt-reward differences).
- Extended `scripts/train_fitted_q_controller.py`:
  - `--output-diagnostics` writes a JSONL on the final greedy-eval epoch (validation only).
  - Each record includes halt rewards, oracle/predicted stop, return, regret, **predicted** advantages, and the six difficulty scalars.
  - `_predict_stop_step` returns `(stop, predicted_advantages)` for logging.
- Unit tests for the difficulty helpers live in `test_train_fitted_q_controller.py`.

Outcome:
- Greedy-eval diagnostics are model-dependent (regret uses the fitted policy); difficulty scalars in that JSONL are still data-intrinsic for the same halt-reward sequence and `continue_cost`.

Conclusion:
- High-regret profile plots compare **oracle stop** (from true rewards) to **predicted** advantages; misalignment is expected under imperfect learning.

### Standalone packed-data difficulty analysis

Intent:
- Summarize difficulty over full train and validation packed manifests without loading any model.

Meaningful change:
- Added `scripts/analyze_episode_difficulty.py` and `slurm/analyze_episode_difficulty_della.slurm`.
- Default output path (unless overridden): `episode_difficulty.jsonl` under the packed-data root.

Outcome (full unfiltered packed data, `continue_cost = 0.001`):
- Train / validation difficulty summaries were broadly aligned; combined ~71k episodes.
- Roughly **56%** of episodes had `oracle_stop_step == 0` (halt immediately).
- Mean difficulty signals indicated a **flat** return landscape (e.g. high softmax entropy near the maximum for the episode length, tiny optimal-vs-second-best gap on average).

Conclusion:
- The unfiltered distribution is dominated by episodes where stopping time barely affects return; that motivated filtering before retraining the controller.

### Filtering packed episodes and retraining on the subset

Intent:
- Drop trivially flat episodes and reduce imbalance toward “halt immediately,” using only existing packed shards.

Meaningful change:
- Added `scripts/filter_packed_episodes.py` and `slurm/filter_packed_episodes_della.slurm`.
- Filters episodes with `halt_reward_range <` threshold; optional cap per `oracle_stop_step` (stratify-sampling).
- First cluster run hit OOM at 16G (all kept episodes held in RAM); reran with **64G**.

Configuration used:
- `MIN_HALT_REWARD_RANGE = 0.10`
- `MAX_PER_STOP_STEP = 500`
- Output directory: `.../controller_packed/filtered/`

Counts:
- Train: `68044 → 16155` (range filter) → `6384` (stratify).
- Validation: `3581 → 848` → `848` (stratify did not bind on validation).

Fitted-Q retrain (filtered manifests, same `continue_cost = 0.001` and linear cost as before — **no** nonlinear cost or cost sweep in code yet):
- Checkpoints and diagnostics (examples):
  - `fittedq_controller_async_filtered.pt` + `fittedq_controller_async_filtered_diagnostics.jsonl`
  - `fittedq_controller_sync_filtered.pt` + `fittedq_controller_sync_filtered_diagnostics.jsonl`
- Validation greedy (848 episodes):
  - **Async:** `average_regret ≈ 0.016`, `first_action_accuracy ≈ 0.731`, `exact_stop_step_accuracy ≈ 0.042`.
  - **Sync:** `average_regret ≈ 0.040`, `first_action_accuracy ≈ 0.697`, `exact_stop_step_accuracy ≈ 0.052`.
- On unfiltered data, sync and async had been nearly tied on average regret; on filtered data, **async shows clearly lower regret** and fewer high-regret episodes in diagnostics.

Conclusion:
- Filtering surfaces differences between encoders: async’s message-passing appears to help when episodes are nontrivial.
- Under-search still carries much higher mean regret than over-search at this `continue_cost`, consistent with cheap extra search vs. missing a better later halt.

### Analysis notebooks

Meaningful change:
- `regret_landscape.ipynb` — loads paired sync/async **filtered diagnostics** JSONLs; regret distributions, regret vs. difficulty metrics, predicted vs. oracle stop, regret by oracle step, over- vs under-search breakdowns, correlation matrices, sync-vs-async per-episode regret scatter, high-regret trajectory plots.
- `episode_difficulty_analysis.ipynb` — loads **`episode_difficulty.jsonl`** from `analyze_episode_difficulty.py`; train/val summaries, histograms, oracle-stop distribution, correlations, halt@0 vs continue boxplots, example halt-reward traces.

Cluster copy (example):
- `scp della:/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/controller_packed/episode_difficulty.jsonl .`

### Qualitative notes from regret-landscape plots (filtered diagnostics)

- Regret vs. difficulty metrics often shows dense mass near **zero regret**; structures (lines, L-shapes) appear when tying **flat landscapes** (small second-best gap, many near-ties) to small regret even when the stop step is wrong.
- **Predicted vs. oracle stop:** both models **over-search** more than under-search at this cost (consistent with low `continue_cost` and asymmetric error: under-search mean regret much higher than over-search).
- **Regret vs. oracle stop step:** largely flat — mistakes are not concentrated at a particular oracle stopping time.
- **Per-episode sync vs. async regret:** mean can favor async while head-to-head “who wins more rows” can favor sync if async wins a **few** episodes by large margins and sync wins **many** by tiny amounts.

### Not done in this thread (future)

- Nonlinear / superlinear continue cost in training or packing.
- Systematic `continue_cost` sweep with re-packing or live targets.
- Prior-entropy pre-filter at raw FEN generation time.

## 2026-04-14

### Budget-aware controller oracle and packed-data augmentation

Intent:
- Replace the old scalar `continue_cost` fitted-Q controller setup with a budget-aware offline controller whose decision state is the learned tree encoding together with literal tree size and remaining synthetic planning budget.

Meaningful change:
- Added `budgeted_controller_oracle.py` with the new recursion
  - `c_maint(N) = 0.01 * (N / 30) ^ 1.1`
  - `c_time(T) = lambda * ((T - 1 + tau)^(-(p-1)) - (T + tau)^(-(p-1)))`
  - defaults: `lambda = 18.537`, `p = 2.8`, `tau = 2.5`, timeout value `-1`
  - `V_t = max(H_t, -c_maint(N_t) - c_time(T_t) + V_{t+1})`
- `scripts/pack_controller_episodes.py` now:
  - computes raw trimmed controller episodes once
  - pre-filters by raw `halt_reward_range` before augmentation
  - augments each surviving raw episode with deterministic synthetic starting budgets across five buckets
  - stores `tree_sizes`, `time_budgets`, `starting_budgets`, bucket metadata, and budget-aware oracle targets in packed shards
- Packed format bumped to:
  - manifest: `cts_budgeted_controller_episode_manifest_v2`
  - shard: `cts_budgeted_controller_episode_shard_v2`

Default augmentation:
- buckets:
  - `[1, 3]`
  - `[4, 10]`
  - `[11, 25]`
  - `[26, 60]`
  - `[61, 120]`
- `2` samples per bucket
- pre-filter default in the Slurm wrapper: `MIN_HALT_REWARD_RANGE = 0.10`

Conclusion:
- The packer now does the sensible order for this workflow: filter on raw episode interestingness first, then replicate into synthetic budgets, instead of writing a huge augmented corpus only to discard most of it later.

### Budget-aware controller training and caching

Intent:
- Train the offline async controller on `concat(z_t, N_t, T_t)` with supervised advantage regression under the new budget-aware oracle, while avoiding repeated frozen-encoder materialization cost.

Meaningful change:
- `scripts/train_fitted_q_controller.py` now:
  - consumes only the packed budget-aware manifests for this path
  - uses an MLP head on `concat(z_t, N_t, T_t)` (one hidden layer with ReLU)
  - computes greedy returns/regret under the new budget-aware oracle
  - writes diagnostics containing `N_t`, `T_t`, starting budget, oracle/predicted stop, oracle/predicted value, target/predicted advantages, and regret
- Added on-disk cache for frozen-encoder materialization:
  - first run materializes `[z_t, N_t, T_t]` features from the packed dataset and saves them next to the manifests
  - later runs reuse those cache files as long as the packed manifest path and encoder checkpoint match

Practical notes:
- The expensive `materialize_*_encoder` stage is still a full one-pass encoding over the packed dataset; larger `EPISODE_BATCH_SIZE` only improves throughput, not asymptotic cost.
- The cache is intended to make repeated controller-head retrains cheap once the first frozen-encoder pass has completed.

### Cluster workflow fixes

Meaningful change:
- Updated Slurm wrappers for packing and training to expose the budget-aware parameters and packed-manifest paths.
- Fixed an rsync workflow issue during deployment: copying without `-R` polluted the remote repo root with stray duplicates while leaving `scripts/` / `slurm/` unchanged. Subsequent syncs used `rsync -avR ...` to update the active files in place.

Conclusion:
- Current intended cluster workflow is:
  1. re-pack with raw-range pre-filter + budget augmentation
  2. train directly on the repacked manifests
  3. reuse cached frozen-encoder features on subsequent controller runs

## 2026-04-15

### First budget-aware async controller run: poor final metacontrol despite low regression loss

Run:
- Packed budget-aware controller data with the original maintenance scale `0.01`.
- Trained the async frozen-encoder controller with the default small MLP head.

Final greedy metrics on the validation run (`8480` episodes):
- `average_return = -0.075`
- `average_oracle_value = 0.080`
- `average_regret = 0.155`
- `average_expansions = 2.293`

Immediate interpretation:
- The run is not failing because the model cannot detect imminent timeout; rather, regret is coming mostly from medium-to-large budget regimes where the controller continues too long.
- The regression objective plateaus early while greedy metacontrol remains poor, so low advantage MSE by itself is not a sufficient success criterion.

### Saved analysis scripts for budget-aware runs

Meaningful change:
- Added `scripts/plot_advantage_loss_from_log.py`
  - parses a training `.out` log and plots train / validation `advantage_mse` over epochs
- Added `scripts/analyze_budgeted_controller_run.py`
  - consumes only the diagnostics JSONL and the training `.out`
  - writes a full report directory with plots and `summary.json`
  - includes:
    - loss curves
    - greedy metrics over epochs
    - regret by starting-budget bucket
    - stop error by budget bucket
    - regret by oracle stop step
    - regret by stop-step error (`predicted_stop_step - oracle_stop_step`)
    - sign accuracy by current `T_t`
    - target-advantage distribution by current `T_t`
    - predicted vs target advantage
    - false-continue / false-halt rates by `T_t`
    - regret by initial tree size
    - `T_t × N_t` partial-dependence heatmaps
    - same-tree different-budget consistency
    - oracle stop distribution
    - calibration by predicted-advantage magnitude
    - trivial baseline sweep

Key findings from the diagnostics-only report:
- Best greedy epoch occurs earlier than the final epoch; continuing training increases average expansions and worsens regret.
- Regret is small in the `scramble` bucket and substantially larger in `medium-large`, `large`, and `very-large`.
- Over-search is the dominant failure mode in those larger-budget buckets.
- Sign accuracy is strongest at very low `T_t`; the bad run is not primarily a “cannot detect low time” problem.

### Regret decomposition: halt-reward term vs maintenance vs time

Meaningful change:
- Extended `scripts/analyze_budgeted_controller_run.py` with an exact episode-level decomposition:
  - `oracle_value - predicted_value`
  - `= (halt_reward@oracle - halt_reward@predicted) + (predicted maintenance paid - oracle maintenance paid) + (predicted time paid - oracle time paid)`
- Added plots and JSON summaries for that decomposition.

Main result:
- In large-budget over-search episodes, regret is almost entirely **not** coming from time cost.
- Example: for `large`, `delta = +6` (`predicted_stop_step - oracle_stop_step = 6`), mean regret is about `0.50`, with roughly:
  - halt-reward term `≈ 0.21`
  - maintenance term `≈ 0.28`
  - time term `≈ 0.008`

Conclusion:
- Large-budget regret is dominated by two things:
  1. extra maintenance burden from carrying a larger tree for extra steps
  2. halting later at a root move whose final-tree score is worse than the oracle-stop move
- The explicit time-cost term is negligible in those regimes.

### Oracle-stop factor analysis

Meaningful change:
- Added an oracle-decision-factor decomposition to `scripts/analyze_budgeted_controller_run.py`:
  - `target_advantage = future_value_gain - maintenance_cost - time_cost`
  - `future_value_gain = oracle_next_value - halt_reward_now`
- At the oracle stop step, the report now classifies whether halting is driven by:
  - `future_already_worse`
  - `maintenance_dominated`
  - `time_dominated`

Main result:
- In the `large` bucket, oracle halting is almost never driven by time cost:
  - `future_already_worse`: about `97%`
  - `maintenance_dominated`: about `3%`
  - `time_dominated`: essentially `0%`

Interpretation:
- In large-budget regimes, the oracle typically halts because continuing is already worse in value terms before explicit time cost matters.
- So the metacontroller’s failure there is not just “missing the time term.”

### Move-switch analysis for `future_already_worse`

Meaningful change:
- Added `scripts/analyze_future_worse_move_switch.py`
  - reconstructs the trimmed decision episode from each `source_path`
  - compares the oracle-stop root move to the predicted-stop root move
  - splits `future_already_worse` episodes into:
    - `same_move`
    - `move_switch`

Main result:
- `same_move` cases:
  - mean regret `≈ 0.069`
  - halt-reward term exactly `0`
  - regret comes almost entirely from maintenance/time cost
- `move_switch` cases:
  - mean regret `≈ 0.409`
  - halt-reward term `≈ 0.346`
  - maintenance term much smaller than the halt-reward penalty

Large-budget interpretation:
- If over-search keeps the same root move, regret is basically just extra maintenance cost.
- The really large regrets come from over-search episodes where the root move changes to one that the final finite-tree evaluation scores worse.

Open concern:
- This exposes a non-monotonicity in the halt-reward construction: immediate halting can outperform intermediate search, while later search may recover.
- The current reward target therefore mixes “value of computation” with transient root-action churn along the finite search trajectory.

### Architecture/cost changes queued for the next overnight run

Meaningful change:
- Strengthened the controller head in `scripts/train_fitted_q_controller.py` and the train Slurm wrappers:
  - hidden width `256`
  - `3` hidden layers
- Reduced the default budgeted maintenance scale by a factor of `4`:
  - from `0.01` to `0.0025`
  - updated in:
    - `budgeted_controller_oracle.py`
    - `scripts/pack_controller_episodes.py`
    - `scripts/train_fitted_q_controller.py`
    - `slurm/pack_controller_episodes_della.slurm`
    - `slurm/train_fitted_q_controller_della.slurm`
    - `slurm/train_compute_advantage_della.slurm`

Intent:
- Lower maintenance burden should reduce the strong immediate-halt bias and make the oracle less dominated by “tree upkeep” penalties.
- The deeper MLP is a straightforward capacity increase for reading out budget-sensitive controller structure from `concat(z_t, N_t, T_t)`.

## 2026-04-16

### Oracle-aligned controller data path and churn analysis

Intent:
- Make the controller supervision semantically consistent with the intended oracle: snapshots should be prefixes of the original generated search, and halt rewards should be derived from the full source-tree oracle rather than from fresh teacher re-search on each prefix.

Meaningful changes:
- `cts_pretrain.py` now stores oracle trajectory data in generated raw examples:
  - expansion counts after each original search expansion
  - root-Q trace over the original search
  - best-move trace over the original search
  - final full-source-tree root Q-values
- `cts_episode_envs.py` now consumes those stored oracle traces when building controller episodes, falling back to the older re-search path only for legacy examples that do not have the new fields.
- The budgeted-controller path now defaults `maintenance_scale = 0.0`:
  - `budgeted_controller_oracle.py`
  - `scripts/pack_controller_episodes.py`
  - `scripts/train_fitted_q_controller.py`
  - `slurm/pack_controller_episodes_della.slurm`
  - `slurm/train_fitted_q_controller_della.slurm`
  - `slurm/train_compute_advantage_della.slurm`
- `scripts/analyze_budgeted_controller_run.py` now supports source-path rewrites and root-action churn summaries anchored on the ultimately chosen move:
  - top-level motif classes `A`, `X*A`, `X*AB*A`
  - compressed-length and unique-move distributions
  - first appearance and stabilization of the final move
  - oracle stop phase relative to first appearance / stabilization
  - lucky early halts

Result:
- The code path is now ready for a clean regeneration of the raw `generated_trees` corpus under the oracle-aligned semantics.
- Existing raw examples remain loadable, but only regenerated examples carry the stored oracle traces and therefore avoid prefix re-search.

Conclusion:
- Future budgeted-controller experiments should use regenerated raw source trees, then rebuild the split manifests and packed controller data from that new corpus before training.

## 2026-04-19

### Time-only budgeted controller run on oracle-aligned, `X*AB*A`-filtered data

Intent:
- Evaluate the fitted-Q metacontroller in the cleaner regime where:
  - controller supervision comes from stored oracle traces on the regenerated oracle-aligned source trees
  - maintenance cost is removed (`maintenance_scale = 0.0`)
  - unstable `X*AB*A` source trajectories are excluded at pack time

Meaningful run setup:
- Packed validation episodes:
  - `30630`
- Buckets balanced:
  - `6126` each for `scramble`, `medium-small`, `medium-large`, `large`, `very-large`
- Oracle metadata recovered from the run log:
  - `maintenance_scale = 0.0`
  - `time_lambda = 18.537`
  - `time_p = 2.8`
  - `time_tau = 2.5`
  - `samples_per_bucket = 2`
  - async encoder checkpoint with `k = 1`

Main result:
- Final greedy metrics (epoch `20`):
  - average oracle value `0.353`
  - average return `0.304`
  - average regret `0.049`
  - exact stop-step accuracy `0.444`
  - first-action accuracy `0.916`
  - average expansions `19.347`
- Best behavioral epoch was earlier than the final checkpoint:
  - best greedy return epoch `10`
  - best greedy regret epoch `10`
  - best validation-MSE epoch `5`

Interpretation:
- The controller learns the coarse halt/continue direction well (`first_action_accuracy` high), but stop timing remains weak, especially in the larger-budget buckets.
- The dominant failure mode is **oversearch**, not undersearch:
  - `medium-large`: oversearch `~0.64`
  - `large`: oversearch `~0.84`
  - `very-large`: oversearch `~0.97`

### Regret decomposition under `maintenance_scale = 0.0`

Meaningful analysis result:
- The report explicitly decomposes regret as:
  - `oracle_value - predicted_value`
  - `= (halt_reward@oracle - halt_reward@predicted) + maintenance_term + time_term`
- With `maintenance_scale = 0.0`, maintenance contributions were exactly zero in the analysis outputs.
- Oversearch regret is therefore almost entirely **time cost**, not maintenance and only weakly halt-reward deterioration:
  - overall oversearch mean regret `~0.076`
  - overall mean time term `~0.073`
  - overall mean halt-reward term `~0.003`

Conclusion:
- In the time-only regime, the controller’s remaining error is mainly “continuing too long and paying extra time cost.”

### Near-threshold sign failure identified

Meaningful analysis result:
- The fitted-Q scalar looks reasonable under MSE/correlation summaries, but sign behavior near the decision boundary is very poor.
- Margin-calibration summary:
  - for `|target_advantage| < 0.05`, sign accuracy is only `~0.29`
  - those low-margin states account for about `70%` of all analyzed states
- Structure of the low-margin regime:
  - concentrated in higher-budget states, not scramble / medium-small
  - within-bucket low-margin rates:
    - `medium-large`: `~48%`
    - `large`: `~73%`
    - `very-large`: `~81%`
  - by current time budget, low-margin rates are already extreme around `T_t = 11..25`
- Crucially, those low-margin states are overwhelmingly slightly **negative**:
  - negative low-margin targets vastly outnumber positive ones
  - model errors there are overwhelmingly `false_continue`, not `false_halt`

Interpretation:
- The controller is not merely noisy near zero; it has a strong **positive bias** in the high-budget, near-threshold regime.
- This explains why MSE can stabilize while metacontrol remains poor in the large-budget buckets:
  - MSE only weakly penalizes small wrong-sign predictions
  - control behavior is driven by the sign of the advantage, not just its squared error

Conclusion:
- The main remaining problem is now clearly a **decision-boundary calibration problem**, not gross regression failure.
- Next experiments should focus on losses/model-selection criteria that care directly about the halt/continue sign near zero, rather than only scalar MSE.

## 2026-04-20

### Entropy-sampled controller dataset scheme

Intent:
- Make the packed controller dataset less dominated by trees whose oracle stop behavior is nearly constant across sampled budgets, without hard-coding a minimum stop step or discarding all easy trees.

Implemented scheme:
- `scripts/pack_controller_episodes.py` now supports `--sample-trees-by-stop-entropy`.
- For each source tree:
  - generate the usual kept budgeted episodes after the existing filters
  - collect oracle stop steps across those sampled budgets
  - bin stop steps into:
    - `0-1`
    - `2-5`
    - `6-9`
    - `10-12`
    - `13+`
  - compute normalized entropy `H` of that 5-bin stop-step distribution
  - keep the whole tree with deterministic probability `p_keep = H`
- Keep/drop uses a deterministic hash of `(source_path, seed)` so repacks are reproducible.

Why this scheme:
- preserves full within-tree budget structure for retained trees
- does not impose any claim that later stop steps are intrinsically better
- still keeps some trees with only `0/1` mass
- smoothly favors trees whose stopping behavior actually varies with budget

Operational note:
- the packer now reports entropy-filtered tree counts explicitly in progress logs and manifests, so retention can be measured after one pass instead of guessed in advance.

### Loss logging now matches the actual optimized objective

Intent:
- The fitted-Q trainer was optimizing `MSE + sign_loss_weight * BCE`, but logs and plots were still only surfacing the MSE term. That was hiding the actual optimization signal during sign-loss sweeps.

Implemented change:
- `scripts/train_fitted_q_controller.py` now logs:
  - `train_total_loss`
  - `train_advantage_mse`
  - `train_sign_bce`
  - and the corresponding validation metrics
- Both analysis scripts were updated to parse both old and new log formats:
  - `scripts/analyze_compute_advantage_training_log.py`
  - `scripts/analyze_budgeted_controller_run.py`
- Training/validation plots now expose total loss and the two components separately rather than only the MSE curve.

Conclusion:
- Future sweeps over sign-loss weight, learning rate, and model size can now be analyzed against the true training objective while using a dataset intervention that prefers cross-budget stop diversity without enforcing a hand-designed monotonic interpretation.

### Entropy-sampled dataset sweep result: not useful

Intent:
- Test whether a tree-level entropy-biased packed dataset improves the controller problem by favoring source trees whose oracle stop step varies across sampled budgets.

Sweep setup:
- Dataset:
  - entropy-sampled packed controller corpus with:
    - stop-step bins `0-1`, `2-5`, `6-9`, `10-12`, `13+`
    - normalized entropy `H`
    - keep probability `0.1 + 0.9 * H`
    - `X*AB*A` exclusion still enabled
- Hyperparameters:
  - sign loss weight: `0.1`, `0.25`, `0.5`
  - learning rate: `1e-4`, `3e-4`, `1e-3`
  - controller head width: `256`, `512`

Main result:
- The entropy-sampled dataset underperformed the earlier sign-loss run on the standard filtered dataset.
- Most important difference:
  - entropy-sampled sweep runs had `average_oracle_value ≈ 0.063`
  - previous successful sign-loss run had `average_oracle_value ≈ 0.353`
- So this intervention did not produce a “harder but informative” dataset. It produced a much flatter one.

Observed behavior:
- The balanced-dataset sweep still had a strongly near-threshold state distribution:
  - about `63%` of states had `|target_advantage| < 0.05`
  - about `96%` of those low-margin states were negative
- Best-regret config in the sweep:
  - `sign_loss_weight = 0.1`
  - `learning_rate = 3e-4`
  - `q_hidden = 512`
  - best greedy regret `≈ 0.044`
- Best overall tradeoff:
  - `sign_loss_weight = 0.25`
  - `learning_rate = 1e-3`
  - `q_hidden = 256`
  - best greedy regret `≈ 0.046`
  - best exact stop `≈ 0.523`
  - same-tree budget monotonicity `≈ 0.93`
- Higher sign-loss weight (`0.5`) improved exact stop accuracy but did not improve regret.

Interpretation:
- Stop-step entropy by itself is the wrong dataset criterion.
- It prefers trees whose stopping behavior moves across budgets, but does not guarantee that those episodes have meaningful oracle value scale or margin structure.
- In practice it selected many flat, low-value episodes and did not resolve the near-threshold negative-dominance problem.

Conclusion:
- Do not use the entropy-sampled dataset as the main training distribution.
- Keep the original `no_xaba` packed dataset as the default.
- If future dataset filtering is revisited, it should target informativeness/value scale directly, not stop-step entropy alone.

## 2026-04-22

### Hierarchical `dj` tree stratification for controller packing

Intent:
- Replace the failed entropy-only tree filter with a dataset intervention that directly attacks the overrepresentation of trivial early-stop trees while preserving whole-tree budget structure.

Implemented scheme:
- `scripts/pack_controller_episodes.py` now supports tree-stratified sampling via:
  - `--sample-trees-by-tree-strata`
  - `--tree-stratification-mode {d,j,dj}`
- Per source tree, after the usual filters, compute:
  - `D = stop_depth_excess`
    - mean oracle stop depth beyond step `1` across sampled budgets
    - trees that always stop at `0` or `1` have `D = 0`
  - `J = budget_action_variance`
    - mean variance across budgets of the oracle continue/halt sign over valid planning steps
- The final `dj` mode is hierarchical rather than full-grid equalization:
  - first rebalance across `D` bins with a `50/50` mixture of natural mass and equalized-bin mass
  - then, within each `D` bin, mildly tilt toward larger `J` with multipliers `1.0`, `1.25`, `1.5`
  - renormalize within each `D` bin so the top-level `D` retained mass is unchanged

Dry-run result on the oracle96 trace corpus:
- Accepted source trees after the standard filters:
  - `57,683`
- `d`-only and hierarchical `dj` both retained about:
  - `17.6k` trees
  - `175.6k` budgeted episodes
- `j`-only was essentially useless and kept almost the whole dataset.
- Full joint `D x J` equalization was discarded because sparse cells collapsed the dataset to almost nothing.
- The final hierarchical `dj` scheme behaved sensibly:
  - within `d0`, keep probability increased from about `0.188` to `0.282` as `J` rose
  - within `d2`, keep probability increased from about `0.247` to `0.370`
  - the middle `d1` bin remained fully kept

Interpretation:
- The main imbalance lives on the stop-depth axis, not on budget sensitivity alone.
- A hierarchical `D -> J` scheme is stable and gives the intended bias:
  - fewer trivial early-stop trees
  - mild preference for trees where budget actually changes the oracle action
- This intervention is appropriate as a training-data bias, but not as a replacement for natural validation.

### `dj`-balanced controller run and sweep

Intent:
- Test whether training on the hierarchical `dj`-balanced packed dataset improves learned metacontrol relative to the plain `no_xaba` dataset.

Single-run result:
- The first `dj` run (`sign_loss_weight = 0.25`, `learning_rate = 3e-4`, `q_hidden = 256`) did not beat the earlier `sign_loss_run0` baseline.
- Final greedy metrics on the `dj` validation set:
  - oracle value `0.275`
  - return `0.233`
  - regret `0.042`
  - exact stop accuracy `0.511`
  - expansions `4.996`
- Compared with the earlier sign-loss run on the standard dataset:
  - lower oracle-normalized performance (`0.233 / 0.275 ≈ 0.847` vs `0.319 / 0.353 ≈ 0.904`)
  - higher normalized regret (`0.042 / 0.275 ≈ 0.153` vs `0.035 / 0.353 ≈ 0.099`)
- One quantity did improve:
  - same-tree predicted stop monotonicity rose from about `0.81` to about `0.87`
- But the overall control policy was still worse, especially through large-budget oversearch.

Sweep result on the `dj` dataset:
- Hyperparameters:
  - sign loss weight: `0.1`, `0.25`, `0.5`
  - learning rate: `1e-4`, `3e-4`, `1e-3`
  - head width: `256`, `512`
- Best config by regret:
  - `sign_loss_weight = 0.1`
  - `learning_rate = 1e-4`
  - `q_hidden = 512`
  - best greedy regret `0.040`
  - best return `0.235`
- Main trend:
  - weaker sign loss helped on this harder dataset
  - larger sign loss increased exact stop-step accuracy but hurt regret and return
- Effect sizes were modest rather than dramatic.

Conclusion:
- The `dj` dataset is harder in the intended sense, but the current controller does not yet exploit it well enough to outperform the plain `no_xaba` recipe.
- For this corpus, the best current setting is roughly:
  - `sign_loss_weight = 0.1`
  - `learning_rate = 1e-4`
  - `q_hidden = 512`
- Going forward:
  - training-data filtering should apply only to the training split
  - natural validation should remain representative of the unfiltered corpus
  - a separate bespoke metacontrol challenge set is likely needed for paper-quality evaluation
