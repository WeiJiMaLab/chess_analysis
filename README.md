# Chess Meta-control (CMC): workspace overview

This repository is the working root for **Chess Meta-control (CMC)**—research that combines **large-scale human chess analytics** in `human_analytics/` with a neural **meta-controller** in **`lmcos/`** (learned metacontrol over search). The chronological experiment log is [`labnotebook.md`](labnotebook.md); stable analysis write-ups live under [`reports/`](reports/). This README is written so that a reader (or an AI agent doing literature search) can recover **intent, formal objectives, training protocols, and connections to prior work** without re-deriving them from the code alone.

**Objective (control layer).** Build a **meta-controller** that manages the trade-off between **thinking** (expanding a Leela/lc0 search tree) and **acting** (playing a move). At the control layer this is an **optimal stopping** problem: is the move-quality we might discover worth the compute we are about to spend? The `lmcos` stack implements this as **representation learning first** (GNN over search trees), then **offline fitted-Q / advantage regression** on teacher traces—not yet full self-play PPO at production scale.

**Primary code locations**

| Path | Role |
| :--- | :--- |
| `chess_analysis/` | DuckDB, figures; Slidev deck lives under `human_analytics/presentations/` |
| `chess_analysis/human_analytics/` | **Human analytics** entry points (`movetime_analysis.py`, …) and **`utils/`** library |
| `chess_analysis/human_analytics/slurm/scripts/` | **Pipeline CLIs** (preprocess, engine eval, joins) |
| `chess_analysis/human_analytics/slurm/` | Shell/Sbatch orchestration that calls `slurm/scripts/*.py` |
| `chess_analysis/labnotebook.md` | Chronological log (Description · Rationale · Status · Reference) |
| `chess_analysis/reports/` | Stable analysis reports (`R-*` refs) |
| `chess_analysis/lmcos/` | Tree encoder, offline controller training; **`src/`** (`cts` package), **`analysis/`** (`cts.analysis`), **`slurm/`** (stage scripts + **`slurm/configs/`** run YAMLs) |
| `chess_analysis/human_analytics/presentations/lmcos-overview/` | Slidev deck: motivation, method, human validation |

For environment setup, Stockfish paths, and notebook entry points, see this file and `chess_analysis/human_analytics/README.md`. The latter documents **code layout** (`human_analytics/` vs `slurm/scripts/`), the behavioral pipeline, and figure conventions.

**Pipeline / DuckDB (`preprocess.py`):** DuckDB spill and staged parquet files follow **one directory per step** (`work_dir` for `get_games` and `process_moves`, `staging_dir` for shard extract **and** `merge` into `moves`). After parquets land, **`merge`** only builds **`moves`**; **`process_moves`** builds **`processed_moves`** / **`processed_moves_nonzero`**. **`preprocess.sh`** runs both. Defaults live in **`preprocess.py` `main()` `config`**, not `**kwargs` plumbing—see `chess_analysis/human_analytics/README.md` §4.

---

## 1. Scientific intent

### 1.1 Long-term goal (planning model)

The motivating picture is a **planning system** that keeps explicit **tree search** as scaffolding but replaces hand-written search-control rules with **neural decision modules**. At a high level:

- A **meta-controller** chooses: **act** in the real environment (play a move) vs **plan** inside an internal search tree.
- The **act** path uses a policy at the root of the current tree; the **plan** path uses a **planning head** over a learned **tree representation** to choose **planning operations** (navigation, expansion, evaluation) that update the tree through a **learned world model** and value feedback, before committing to a real move.
- In principle, such a system could be trained in an **AlphaZero-style** loop with self-play.

### 1.2 Current research slice (this codebase)

The implementation is **deliberately narrower**: **meta-control of search only** (when to keep expanding vs when to **halt** and play), on **teacher-generated** search trees and **offline** targets. Trajectories are snapshots of a growing search (e.g. from **Leela / lc0**-style search on positions sampled from Lichess with simple filters). Supervision is derived from **counterfactual value-of-computation**: comparing halting at each expansion step to continuing, under a defined **continue cost** and **halt rewards** from the search state (see §3).

**Central empirical questions** include: Can a **simple** halt/continue policy learn (near-)optimal control given a **TreeNN** encoding? Which encoding or **cost architecture** (linear vs budget-aware) supports learning? How does behavior relate to **human** time allocation and engine-based **VOC** (value of computation) from the behavioral track?

A future layer is a **full planning head** (which node to expand, etc.) on the same representation; see [`labnotebook.md`](labnotebook.md) and [(R-LMCOS-OVERVIEW)](reports/lmcos-pipeline-overview.md) for the roadmap.

---

## 2. Human behavioral track (context for “broad implications”)

Work under `chess_analysis/human_analytics/` treats chess as a natural experiment in **resource allocation**: move time is heavy-tailed; **remaining clock** and **position complexity** both predict thinking time, with a stable **VOC** effect (prospective engine gain vs shallow eval) and characteristic **ply-stage** “arc” of deliberation. Slides in `human_analytics/presentations/lmcos-overview/` connect this to **resource-rational** meta-control: humans adapt budgets to time pressure and to estimated benefit of search.

The `lmcos` line asks the complementary question: if we **teach a network** the statistics of a search tree, can it **approximate the stopping rule** implied by a formal cost–benefit model? That links behavioral VOC curves to **machine metareasoning** on trees.

---

## 3. Formal problem: halting, costs, and oracles

> ⚠️ **Filename note (2026-06-16).** §§3–5 below were written against an aspirational
> "metacontrol" module refactor (`analysis/metacontrol/`, `controller_oracle.py`,
> `planning_cost.py`, `budgeted_controller_oracle.py`, `episode_difficulty.py`,
> `GNN.py`/`TreeMHA.py`, `cts_pretrain.py`, `supervised_branch.py`, `cts_rl.py`, …)
> that was **never landed and is now abandoned** — none of those files/dirs exist.
> The **math and objectives in these sections are still accurate**; treat the file
> names as *conceptual labels only*. The **authoritative code map is §11** (the real
> `cts.*` package: `cts.data.build_tree`, `cts.data.preprocess_mc`,
> `cts.train.controller_train`, `cts.models.gnn`, `cts.core.providers`, …) and
> [`labnotebook.md`](labnotebook.md).

### 3.1 Snapshots and halt rewards

Along one search episode, let snapshots be \(s_0,\ldots,s_{K-1}\) with step index \(t\) over expansions. The **halt reward** at \(t\), written \(h_t\), scores the **quality of the move selected if the agent stops at \(t\)**. The project moved from a **regret** formulation to an **absolute** full-reference target: \(h_t\) is driven by the **value of the best move at snapshot \(t\)** under the full teacher search (e.g. \(Q\)-full), not by difference to the final best move only. That avoids degenerate small margins on intermediate steps (see [(R-ARCH-LMCOS)](reports/archive-lmcos-notebook-legacy.md), 2026-04-05).

### 3.2 Linear continue cost and DP oracle (scalar cost)

Let **continue cost** be \(c > 0\) per expansion (possibly extended to **linear vs power** cumulative costs via `planning_cost.PlanningCostConfig`). If the agent **halts** at step \(j\), a natural return is
\[
R(j) = h_j - C(j),
\]
where \(C(j)\) is the **cumulative** planning cost paid for **continues** before stopping (e.g. \(C(j) = c\cdot j\) for a linear per-step cost with \(j\) continues before halt at the same index in the lab’s indexing conventions—implementation details in `planning_cost.py` and `controller_oracle.py`).

The **offline oracle** for “halt vs continue” is obtained by **backward dynamic programming** on the finite horizon (see `compute_oracle_policy` in `controller_oracle.py`):
- At the last index, the value is the halt payoff.
- For each earlier \(t\), compare **halt now** vs **continue** (pay incremental cost, transition to \(t+1\)).

Actions are typically encoded with **0 = continue**, **1 = halt**. The optimal **stop step** is \(\arg\max_j R(j)\) under the same cost model.

**Role of \(c\).** The lab found that for real data, **very small** \(c\) (e.g. order \(10^{-3}\)) yields a **meaningfully adaptive** oracle (nontrivial continuation rate), while larger \(c\) can collapse the oracle to **immediate halt**—so the same dataset supports different “economies of thought” depending on the cost scale.

### 3.3 Value-of-computation and advantage

Define **Q-style** targets:
\[
Q_{\mathrm{halt}}(s_t) = h_t, \qquad
Q_{\mathrm{continue}}(s_t) = -c + V^*(s_{t+1}),
\]
with \(V^*\) the oracle value of following the optimal policy from the next snapshot (or the DP value in tabular form on the chain). The **compute advantage** (scalar actually trained in the fitted-**Q** / advantage path) is
\[
A_{\mathrm{compute}}(s_t) = Q_{\mathrm{continue}}(s_t) - Q_{\mathrm{halt}}(s_t).
\]
A **greedy** policy **continues** iff \(A_{\mathrm{compute}}(s_t) > 0\). Training minimizes MSE to Bellman-derived \(A_{\mathrm{compute}}\) to avoid “common-mode” value fitting that matches levels but not the **decision boundary** (see [(R-ARCH-LMCOS)](reports/archive-lmcos-notebook-legacy.md), fitted-Q and advantage-only sections).

### 3.4 Budget-aware oracle (state: tree size and time budget)

The **scalar-\(c\)** model ignores literal **tree size** and **remaining time budget**. The **budgeted** extension (`budgeted_controller_oracle.py`) augments the state with **node count** \(N_t\) and **remaining planning budget** \(T_t\). A step cost combines:

- **Maintenance** (superlinear in tree size by default), e.g.
  \[
  c_{\mathrm{maint}}(N) = s \left(\frac{N}{N_0}\right)^\eta
  \]
  with learnable/defaults for \(s, N_0, \eta\).
- **Time** cost from a **discrete** budget consumed each step, e.g. using a **power-law**-style **incremental** cost
  \[
  c_{\mathrm{time}}(T) = \lambda\Big[ (T-1+\tau)^{-(p-1)} - (T+\tau)^{-(p-1)} \Big]
  \]
  (see `time_cost` in code; when budget is exhausted, a **timeout value** is used).

The Bellman backup matches the same halt-vs-continue pattern with **state-dependent** continue cost. Packed episodes augment raw trajectories with **synthetic starting budgets** across buckets so the controller sees diverse \((N,T)\) (manifest format `v2` in the lab). The trained head often uses **concat\((z_t, N_t, T_t)\)** with encoder embedding \(z_t\).

### 3.5 Episode difficulty (data-only diagnostics)

`episode_difficulty.py` defines metrics (halt reward **range**, optimal vs second-best **return gap**, **entropy** over per-stop returns, **regret** of “halt at 0,” **curvature** of halt-reward differences) to characterize whether stopping time matters. Analysis showed many episodes have **nearly flat** return landscapes, motivating **filtering** (e.g. by halt reward range) before retraining.

---

## 4. Tree representation and pretraining (TreeNN / GNN)

### 4.1 Encoding and two-phase message passing

Search trees are **tensorized** for GPU batching (`tensorizer.py`): a **flat-forest** layout with parent/child pointers, packed into **fat shards** for I/O efficiency. A **TreeNN**-style model (`GNN.py`, `TreeMHA.py`) runs **rounds** of message passing in **topological** order (not one synchronous blur of all neighbors):

1. **Upward sweep (child → parent):** children aggregate via **multi-head attention**; each parent updates with a **GRU** (“what did my children discover?”).
2. **Downward sweep (parent → child):** the parent summary is **linearly projected** and broadcast; each child updates with the **same GRU** (“given global context, how should my local move-vector change?”).

**Slot encodings** (sinusoidal / learned) disambiguate **child order** (canonical UCI ordering) for **per-edge** prediction heads—central to **child-WDL** pretraining, where **every parent→child edge** in a batch contributes a cross-entropy signal (not just the root), giving dense gradients for subtree structure.

### 4.2 Oracle data construction (pretrain)

- **Mode (research):** **dynamic growth** — run a full **oracle** search (e.g. large node budget), then take a **prefix** of the expansion sequence as input and **consolidate** deep statistics from the full tree as **supervised targets** (prefix / deep targets in `cts_pretrain.py`).
- **Targets:** Scalar value backups and, after fixes in 2026-04-10, **search-consolidated per-edge WDL** targets (visit-weighted, perspective-correct) stored as `edge_wdl_targets`, not raw value-head slices at a node in isolation.
- **Prefix derivation:** `derive_pretrain_prefixes.py` can subsample **variable-size prefixes** from existing fixed full trees without re-querying the engine (see [(R-ARCH-LMCOS)](reports/archive-lmcos-notebook-legacy.md)).

### 4.3 Packing and Slurm

Large-scale flow: **generate** many `.pt` **PretrainExample** / raw examples (cluster) → **pack** to shards → **pretrain** encoder (e.g. child-WDL) → **pack controller episodes** (with budget augmentation) → **train** halt/continue head. Job templates and run YAMLs live under `lmcos/slurm/` (`slurm/configs/<stage>/`). Slurm **stdout/stderr** go to flat `slurm/logs/`; per-run **metrics YAML**, **curve PNGs**, and **comparison plots** go to flat `slurm/outputs/<stage>/` (tracked in git). Stage **4** ablation configs are submitted via `./slurm/4_supervised_controller/submit_configs.sh` (glob all YAMLs in `slurm/configs/4_supervised_controller/`). See `lmcos/slurm/README.md` and [(R-LMCOS-STAGE4)](reports/lmcos-stage4-ablation.md).

### 4.4 Engine providers and abstraction (`cts.core.providers`)

Tree generation uses a UCI engine-provider abstraction under `cts.core.providers`
(package, not a single `providers.py`): a common UCI subprocess/communication layer
with engine-specific parsing for **lc0** (`lc0.py` — `VerboseMoveStats`, WDL, the
valuehead/classic two-engine split) and a generic UCI path. This keeps the PUCT
search logic engine-agnostic.

**Perspective / sign correctness.** Each child's `value`/WDL come from a **separate
value-engine query on the child's own FEN** (`_value_features_for_fen`), already in
the child's side-to-move frame — there is **no `-Q` flip at storage**. The single
negamax negation happens during `_backpropagate_path`. Per-edge WDL targets are
parent-perspective-flipped when consolidated into `edge_wdl_targets`. Net result is
correct negamax; an earlier README description of a "flip at storage, flip again at
backprop" two-flip mechanism was **inaccurate** and has been removed.

### 4.5 Testing

The real test suite lives in **`lmcos/tests/`** (run `pytest tests/` from `lmcos/`
with `PYTHONPATH=src`). It covers provider parsing, terminal-state WDL, PUCT search
quality on tactical mates, tensorizer/packing, the DP oracle, fitted-Q, and probes.
*(An earlier `analysis/metacontrol/tests/` suite described here never existed; the
authoritative test location is `lmcos/tests/`.)*

---

## 5. Training protocols: controller

### 5.1 On-policy RL (PPO / REINFORCE)

`supervised_branch.py` / `cts_rl.py` implement **PPO** (and baselines) on **frozen-encoder** or **unfrozen** encoders, with **Bernoulli** halt/continue, **value** head, and diagnostics (**approx kl**, **clipfrac**, **explained variance**). A **rollout collection bug** was fixed so batch size is \( \texttt{num\_envs} \times \texttt{rollout\_steps} \) (regression test in `test_supervised_branch.py`).

`scripts/rl_sanity_checks.py` validates **sign** of updates and logit **drift** on toy tasks—used to rule out **gradient wiring** bugs when long runs are flat.

### 5.2 Offline fitted advantage regression

`scripts/train_fitted_q_controller.py` fits **scalar** \(A_{\mathrm{compute}}\) (or budget-aware MLP on features) with **MSE** to DP targets, with **greedy** evaluation and optional **diagnostics** JSONL. **Materialization caches** for frozen-encoder **root embeddings** avoid repeating heavy encoding each epoch when manifests/checkpoints are unchanged.

### 5.3 Probes and controls

- **`probe_controller_representation.py`:** Train a small **MLP/linear** probe on **frozen root embeddings** to predict **oracle action** or regress **advantage**—tests whether the **representation** carries the **halting** signal *without* full RL.
- **`controller_representation_control.py`:** Trivial or oracle-derived features (e.g. **oracle-action-now** one-hot) to isolate **PPO** vs **representation**; includes **one-step bandit** and **supervised** checks.

### 5.4 Empirical patterns (from the lab record)

- **PPO** on real trees can be **stable** but **over-search** relative to the offline oracle at low \(c\); validation return can **plateau early**.
- **Unfreezing** the encoder in one large run did **not** improve validation vs frozen.
- **Fitted advantage** on frozen embeddings can get **return** near oracle but **poor** exact stop-step / sign unless data are **filtered** to nontrivial episodes; **async** vs **sync** encoders can differ on filtered data.
- **Budgeted** packing and training are the **current** intended path for state-aware costs.

These are *hypothesis-generating* outcomes; see [`labnotebook.md`](labnotebook.md) and [(R-A0)](reports/oracle-stop-vs-human-rt.md) for numbers and run IDs.

---

## 6. Talks and notebooks

| Resource | Content |
| :--- | :--- |
| `human_analytics/presentations/lmcos-overview/` | Motivation, architecture slides, **human** clock/VOC figures |
| `labnotebook.md` | Chronological experiment log |
| `reports/` | Stable `R-*` analysis reports |

Analysis notebooks mentioned in the lab (`regret_landscape.ipynb`, `episode_difficulty_analysis.ipynb`) live alongside packed diagnostics on analysis machines.

---

## 7. Literature search guide (for agents or humans)

The project sits at the intersection of several named research areas. Useful **query families** and **paper clusters**:

- **Metareasoning / rational metareasoning** (Horvitz, Russell & Wefald, Han & Stewart): deciding when to stop deliberation; **anytime** algorithms; **value of computation** as an explicit object.
- **Optimal stopping** on finite horizons: **backward induction**, **threshold policies**, **regret** vs **reward** parameterizations (connections to §3).
- **GNNs on trees / graphs** for search: **TreeLSTM**, **GAT**, **graph transformers**, **UCT** + learned value (AlphaZero, MuZero, **Sampled MuZero**); **neural MCTS** and **search policy** learning.
- **Reinforcement learning** for **budgeted** or **horizon** problems: **PPO**, **REINFORCE**, **contextual bandits** (one-step diagnostics in this repo); **Offline RL** and **imitation of oracles**; **Fitted Q-Iteration** / **Bellman** regression for **discrete** control on fixed datasets.
- **Chess and game AI**: **MCTS + NN** (Leela, Stockfish NNUE as non-learned control); **adaptive** search time in engines.
- **Cognitive** / **psychophysics** of **chess** (if linking to human data): time pressure, **depth of search** vs **task difficulty**; the behavioral scripts formalize **VOC** and **clock elasticity**.

**Contrast this work:** offline **teacher trees**, **no** full self-play loop yet, **explicit** cost models (linear or budgeted) with **DP oracles** rather than end-game outcome only.

---

## 8. How to work in this workspace

- **Environment:** hl4291 della uses `/home/hl4291/venv` for stage-4 jobs (`VENV_DIR`); ysagiv jobs may use `cts_supervised` conda. Activate locally with `source .venv/bin/activate` from repo root when available.
- **Tests:** `lmcos/tests/` cover plumbing, oracles, fitted-Q, probes; run `pytest tests/` from `lmcos/` with `PYTHONPATH` set.
- **Sync:** When copying to clusters, the lab notes using **`rsync -avR`** to avoid sparse directory mistakes.

For day-to-day commands: **`human_analytics/README.md`** (DuckDB ETL, figures); **`lmcos/slurm/README.md`** (pipeline Slurm); **`labnotebook.md`** (experiments).

---

## 9. Division of responsibilities

| Area | Lead | Scope |
| :--- | :--- | :--- |
| **Infrastructure & integration** | Yotam | lc0/tree export, GNN core (bidirectional sweeps, slot encodings, batching), RL training paths |
| **Validation & psychology** | Jordan | Baselines, human alignment (move time, clock regimes), controller ablations, diagnostics |

---

## 10. Development roadmap (high level)

1. **Data path** — Leela/lc0 trees → tensorization → GNN forward pass (stages 1–3 under `lmcos/slurm/`).
2. **Supervised encoder** — prefix vs oracle WDL; frozen `tree_encoder` checkpoint.
3. **Offline meta-control** — fitted advantage / budgeted oracle on materialized caches; greedy regret eval.
4. **Psychometric validation** — compare learned stopping to human Lichess move-time and engine VOC.

Current production focus is **stage 3–4** on ysagiv read-only caches (hl4291 della, `VENV_DIR=/home/hl4291/venv`).

---

## 11. Code map (`lmcos`)

| Concept | Module / path |
| :--- | :--- |
| Tree generation & encoder pretrain CLI | `cts.data.build_tree` |
| Controller episode pack / materialize | `cts.data.preprocess_mc` |
| Fitted-Q controller train + metrics plots | `cts.train.controller_train` |
| Slurm stage wrappers + run YAMLs | `lmcos/slurm/`, `lmcos/slurm/configs/` |
| Post-hoc budgeted-run analysis | `cts.analysis.analyze_budgeted_controller_run` |
| Human analytics pipeline & figures | `human_analytics/` (see `human_analytics/README.md`) |

---

*Last updated 2026-06-04: experiment log at `labnotebook.md` + `reports/`; see [(R-LMCOS-STAGE4)](reports/lmcos-stage4-ablation.md) for the 2026-05-29 controller harness.*
