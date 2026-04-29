# Workspace overview: chess behavior, value of computation, and learned search control

This home directory is the working root for a research thread that combines **large-scale human chess analytics** in `chess_analysis/` with a neural **meta-controller** codebase **`chess_analysis/lmcos/`** (learned metacontrol over search). The long-form experimental record is `chess_analysis/lmcos/LAB_NOTEBOOK.md`. This README is written so that a reader (or an AI agent doing literature search) can recover **intent, formal objectives, training protocols, and connections to prior work** without re-deriving them from the code alone.

**Primary code locations**

| Path | Role |
| :--- | :--- |
| `chess_analysis/` | DuckDB, figures, presentations |
| `chess_analysis/src/` | **Analysis** entry points (`movetime_analysis.py`, …) and **`utils/`** library |
| `chess_analysis/src/slurm/scripts/` | **Pipeline CLIs** (preprocess, engine eval, VOC parquet, joins) |
| `chess_analysis/src/slurm/` | Shell/Sbatch orchestration that calls `slurm/scripts/*.py` |
| `chess_analysis/lmcos/` | Tree encoder, offline controller training, Slurm job definitions |
| `chess_analysis/lmcos/LAB_NOTEBOOK.md` | Dated experiments, cluster run IDs, and conclusions |
| `chess_analysis/lmcos/demos/` | Tutorial notebooks (`01_`–`05_`) and `understanding.md` |
| `chess_analysis/presentations/cmc-overview/` | Slidev deck: motivation, method, human validation |

For environment setup, Stockfish paths, and notebook entry points, see this file and `chess_analysis/src/README.md`. The latter documents **code layout** (`src/` vs `slurm/scripts/`), the behavioral pipeline, and figure conventions.

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

A future layer is a **full planning head** (which node to expand, etc.) on the same representation; the lab notebook and `demos/05_meta_controller_tutorial.ipynb` are aligned with that roadmap.

---

## 2. Human behavioral track (context for “broad implications”)

Work under `chess_analysis/src/` treats chess as a natural experiment in **resource allocation**: move time is heavy-tailed; **remaining clock** and **position complexity** both predict thinking time, with a stable **VOC** effect (prospective engine gain vs shallow eval) and characteristic **ply-stage** “arc” of deliberation. Slides in `presentations/cmc-overview/` connect this to **resource-rational** meta-control: humans adapt budgets to time pressure and to estimated benefit of search.

The `lmcos` line asks the complementary question: if we **teach a network** the statistics of a search tree, can it **approximate the stopping rule** implied by a formal cost–benefit model? That links behavioral VOC curves to **machine metareasoning** on trees.

---

## 3. Formal problem: halting, costs, and oracles

### 3.1 Snapshots and halt rewards

Along one search episode, let snapshots be \(s_0,\ldots,s_{K-1}\) with step index \(t\) over expansions. The **halt reward** at \(t\), written \(h_t\), scores the **quality of the move selected if the agent stops at \(t\)**. The project moved from a **regret** formulation to an **absolute** full-reference target: \(h_t\) is driven by the **value of the best move at snapshot \(t\)** under the full teacher search (e.g. \(Q\)-full), not by difference to the final best move only. That avoids degenerate small margins on intermediate steps (see `LAB_NOTEBOOK`, 2026-04-05).

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
A **greedy** policy **continues** iff \(A_{\mathrm{compute}}(s_t) > 0\). Training minimizes MSE to Bellman-derived \(A_{\mathrm{compute}}\) to avoid “common-mode” value fitting that matches levels but not the **decision boundary** (see `LAB_NOTEBOOK`, fitted-Q and advantage-only sections).

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

### 4.1 Encoding

Search trees are **tensorized** for GPU batching (`tensorizer.py`): a **flat-forest** layout with parent/child pointers, packed into **fat shards** for I/O efficiency. A **TreeNN**-style model (`GNN.py`, `TreeMHA.py`) runs **rounds** of message passing: **upward** (children → parent, attention + **GRU** updates) and **downward** (parent → children), in **topological** order, so evidence aggregates to the **root** representation used for readouts and control.

**Slot encodings** (sinusoidal / learned) disambiguate **child order** (canonical ordering by UCI) for **per-edge** prediction heads—relevant to **child-WDL** pretraining (predict consolidated **win/draw/loss** targets per edge).

### 4.2 Oracle data construction (pretrain)

- **Mode (research):** **dynamic growth** — run a full **oracle** search (e.g. large node budget), then take a **prefix** of the expansion sequence as input and **consolidate** deep statistics from the full tree as **supervised targets** (prefix / deep targets: `cts_pretrain.py`, demos `01`–`02`).
- **Targets:** Scalar value backups and, after fixes in 2026-04-10, **search-consolidated per-edge WDL** targets (visit-weighted, perspective-correct) stored as `edge_wdl_targets`, not raw value-head slices at a node in isolation.
- **Prefix derivation:** `derive_pretrain_prefixes.py` can subsample **variable-size prefixes** from existing fixed full trees without re-querying the engine (see `LAB_NOTEBOOK`).

### 4.3 Packing and Slurm

Large-scale flow: **generate** many `.pt` **PretrainExample** / raw examples (cluster) → **pack** to shards → **pretrain** encoder (e.g. child-WDL) → **pack controller episodes** (with budget augmentation) → **train** halt/continue head. Job templates live under `chess_analysis/lmcos/slurm/`.

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

These are *hypothesis-generating* outcomes; see `LAB_NOTEBOOK.md` for numbers and run IDs.

---

## 6. Demos, talks, and notebooks

| Resource | Content |
| :--- | :--- |
| `lmcos/demos/01_prefix_tutorial.ipynb` | Prefix sampling and deep target consolidation |
| `02_tensorization_tutorial.ipynb` | Flat-forest batching |
| `03_gnn_tutorial.ipynb` | Bidirectional GNN “heartbeat” |
| `04_pretrain_tutorial.ipynb` | Supervised pretraining and losses |
| `05_meta_controller_tutorial.ipynb` | Halt/continue and economy of thought |
| `lmcos/demos/understanding.md` | GNN wiring, slot encodings, dense recursive WDL head |
| `presentations/cmc-overview/` | Motivation, architecture slides, **human** clock/VOC figures |

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

- **Environment:** Prefer a dedicated venv/conda (see `chess_analysis/README.md`; cluster jobs may use `trm` / `cts_supervised`-style envs as in the lab).
- **Tests:** `chess_analysis/lmcos/test_*.py` cover plumbing, oracles, fitted-Q, probes; run with `python -m pytest` from a configured environment.
- **Sync:** When copying to clusters, the lab notes using **`rsync -avR`** to avoid sparse directory mistakes.

For day-to-day commands and paths inside `chess_analysis`, use **`chess_analysis/src/README.md`** (pipeline CLIs under **`src/slurm/scripts/`**); for meta-controller and tree visualization, see **`src/performance/visualize_tree_expansion.py`** and **`lmcos/`**.

---

*Last updated to reflect `lmcos/LAB_NOTEBOOK.md` and demos through 2026-04-14 (budgeted oracle and packing v2).*
