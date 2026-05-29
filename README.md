# Workspace overview: chess behavior, value of computation, and learned search control

This home directory is the working root for a research thread that combines **large-scale human chess analytics** in `chess_analysis/` with a neural **meta-controller** codebase **`chess_analysis/lmcos/`** (learned metacontrol over search). The long-form experimental record is `chess_analysis/lmcos/LAB_NOTEBOOK.md`. This README is written so that a reader (or an AI agent doing literature search) can recover **intent, formal objectives, training protocols, and connections to prior work** without re-deriving them from the code alone.

**Primary code locations**

| Path | Role |
| :--- | :--- |
| `chess_analysis/` | DuckDB, figures; Slidev deck lives under `human_analytics/presentations/` |
| `chess_analysis/human_analytics/` | **Human analytics** entry points (`movetime_analysis.py`, …) and **`utils/`** library |
| `chess_analysis/human_analytics/metacontrol/` | Modular tree-search export (refactored from `lmcos/`) |
| `chess_analysis/human_analytics/slurm/scripts/` | **Pipeline CLIs** (preprocess, engine eval, joins) |
| `chess_analysis/human_analytics/slurm/` | Shell/Sbatch orchestration that calls `slurm/scripts/*.py` |
| `chess_analysis/lmcos/LAB_NOTEBOOK.md` | Dated experiments, cluster run IDs, and conclusions |
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

A future layer is a **full planning head** (which node to expand, etc.) on the same representation; see `lmcos/LAB_NOTEBOOK.md` for the roadmap.

---

## 2. Human behavioral track (context for “broad implications”)

Work under `chess_analysis/human_analytics/` treats chess as a natural experiment in **resource allocation**: move time is heavy-tailed; **remaining clock** and **position complexity** both predict thinking time, with a stable **VOC** effect (prospective engine gain vs shallow eval) and characteristic **ply-stage** “arc” of deliberation. Slides in `human_analytics/presentations/lmcos-overview/` connect this to **resource-rational** meta-control: humans adapt budgets to time pressure and to estimated benefit of search.

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

- **Mode (research):** **dynamic growth** — run a full **oracle** search (e.g. large node budget), then take a **prefix** of the expansion sequence as input and **consolidate** deep statistics from the full tree as **supervised targets** (prefix / deep targets in `cts_pretrain.py`).
- **Targets:** Scalar value backups and, after fixes in 2026-04-10, **search-consolidated per-edge WDL** targets (visit-weighted, perspective-correct) stored as `edge_wdl_targets`, not raw value-head slices at a node in isolation.
- **Prefix derivation:** `derive_pretrain_prefixes.py` can subsample **variable-size prefixes** from existing fixed full trees without re-querying the engine (see `LAB_NOTEBOOK`).

### 4.3 Packing and Slurm

Large-scale flow: **generate** many `.pt` **PretrainExample** / raw examples (cluster) → **pack** to shards → **pretrain** encoder (e.g. child-WDL) → **pack controller episodes** (with budget augmentation) → **train** halt/continue head. Job templates and run YAMLs live under `chess_analysis/lmcos/slurm/` (`slurm/configs/<stage>/`). Stage **4** smoke configs (`legacy_root_budget.yaml`, `subtree_weighting_root_budget.yaml`, `subtree_weighting_root.yaml`; display names `legacy[root+budget]`, etc.) train 1000 steps on ysagiv materialized caches with validation every 100 steps; see `lmcos/slurm/README.md` and `lmcos/LAB_NOTEBOOK.md` (2026-05-29 entry).

### 4.4 Modular Metacontrol Pipeline (`analysis/metacontrol/`)

The project has been refactored into a modular structure under `analysis/metacontrol/` to enforce strict decoupling and didactic clarity:

- **`core/`**: Fundamental data structures (`tree.py`, `tensorizer.py`) and schemas (`schemas.py`).
- **`data/`**: Pipeline logic for tree generation (`generator.py`), meta-control DP derivation (`targets_mc.py`), and GNN target computation (`targets_gnn.py`).
- **`tutorials/`**: Didactic notebooks and generation scripts demonstrating the full pipeline.
- **`tests/`**: Comprehensive integration and unit tests for the pipeline.

The new `TreeSearch` class in `generator.py` provides a clean, method-based API for tree growth, while `targets_mc.py` and `targets_gnn.py` separate the derivation of training targets for the controller and the GNN respectively.

**Root sampling and single-tree export (2026-05):**

- **Lichess roots:** `analysis/metacontrol/scripts/sample.py` (`ChessSampler`) writes filtered FEN rows; an example row is kept at `/scratch/gpfs/GRIFFITHS/hl4291/data/metacontrol_example.csv` when integration tests run.
- **One-tree pipeline:** `analysis/metacontrol/scripts/generate_and_profile.py` reads that CSV (or `--csv`), runs `TreeSearch` + targets + `TreeTensorizer`, and saves `/scratch/gpfs/GRIFFITHS/hl4291/data/trees/example_tree_00001.pt` by default. Use `--no-profile` for wall time only; otherwise it prints cumulative `cProfile` stats and writes `generate_and_profile.pstats` alongside the shard.
- **`.pt` layout vs `ysagiv`:** Legacy controller shards on the cluster use `format="cts_budgeted_controller_episode_shard_v4"` with RL replay pointer tensors (`trajectory_node_ptr`, `episode_step_ptr`, …) and flattened `target_advantages` across many trajectories. Metacontrol’s Phase~1 export uses `format="metacontrol_single_tree_v1"`, the same five-wide `node_features` (`value`, WDL, `wdl_var` pad), `edge_child` as `int32`, and **omits** trajectory pointers. It adds explicit tensors `mc_halt_rewards`, `mc_dp_values`, `edge_wdl_targets`, and per-snapshot `target_advantages`; `oracle_values` here are **per-node** GNN consolidated values (not the legacy per-episode oracle scalars). See `analysis/metacontrol/README.md` and `analysis/metacontrol/migration.md`.

### 4.5 Engine Providers and Abstraction (`core/providers.py`)

The pipeline now features a generalized engine provider system that supports both **LC0** and **Stockfish** (and any other UCI-compatible engine). 
- **`UciExpansionProvider`**: Centralizes subprocess management and UCI communication.
- **`LC0ExpansionProvider`**: Parses specialized statistics like `VerboseMoveStats` and WDL.
- **`StockfishExpansionProvider`**: Handles standard UCI info and uses `multipv` rank-based priors.

This abstraction ensures that the `TreeSearch` logic remains engine-agnostic, allowing research to focus on the search dynamics rather than engine-specific parsing.
 
- **Mathematical Parity and Perspective Correctness**: 
  The pipeline enforces strict sign and perspective consistency. Historical discrepancies in teacher targets (where winning positions were sometimes recorded with negative values) have been resolved.
  - **Expansion**: Engine Q-values (reported from parent perspective) are flipped (`-Q`) when stored in child nodes to maintain local consistency.
  - **Backpropagation**: Values are flipped again during path traversal, restoring the original engine intent at the parent edge.
  - **WDL Targets**: Normalized WDL probabilities are flipped (`win <-> loss`) at each level to ensure they always reflect the side-to-move.
  - **Parity Tests**: A dedicated suite (`test_legacy_parity.py`) ensures that new generation logic matches the mathematical core of the legacy system while correcting its sign conventions.

### 4.6 Testing and Validation

The project maintains a comprehensive test suite (`analysis/metacontrol/tests/`) that covers unit, engine, and integration scenarios (run `pytest analysis/metacontrol/tests -m "not integration"` for a fast slice; include `test_sampler.py` for DuckDB smoke tests):
- **Core Logic & Target Derivation**:
  - `test_targets_mc.py` verifies the DP algorithm, including a simulation demonstrating that when a tree expansion discovers a mate-in-2, the backward DP properly assigns a massive positive "continue advantage" to earlier snapshots.
  - `test_targets_gnn.py` ensures target consolidation is accurate, validating that terminal checkmate states correctly map to a WDL of `(1.0, 0.0, 0.0)` from the parent's perspective.
- **Provider Accuracy (`test_providers.py`)**: Engine-invariant tests for both **LC0** and **Stockfish**. These confirm that terminal states (like Fool's Mate or Stalemate) yield strictly consistent WDL evaluations regardless of which side is checkmated, validating the local `ChildPerspective` invariant.
- **Search Efficacy (`test_search_quality.py`)**: End-to-end validation using real engine binaries to confirm the search process correctly identifies tactical wins (**Scholar's Mate**, **Back Rank Mate**, **Arabian Mate**, **Damiano's Mate**, **Morphy's Puzzle**, **Philidor's Smothered Sequence**). It explicitly tests the PUCT exploration mechanics, for instance showing that a quiet mate-in-2 might be overlooked with a low `c_puct` but is confidently surfaced when `c_puct` is raised.
- **Divergence Analysis**: Documented evidence proving that the fixed pipeline is structurally superior to the legacy system, which previously avoided winning moves due to a sign-inversion bug.

### 4.7 Hardening and Engine Stability

The pipeline has been "hardened" for production-scale data generation:
- **Perspective Parity**: Confirmed via `test_legacy_mates.py` that the pipeline correctly implements `ChildPerspective = -ParentQ`, fixing the "search blindness" of the legacy code.
- **Engine Robustness**: Implemented dynamic `MultiPV` capping and explicit UCI flushing to prevent engine segmentation faults (notably in Stockfish 15) and ensure stable communication.
- **Engine-Invariant Testing**: Both LC0 and Stockfish are verified through the same parameterized test suites for both terminal states and deep tactical searches.

Run the full validation suite:
```bash
pytest analysis/metacontrol/tests/core/test_providers.py analysis/metacontrol/tests/data/test_search_quality.py analysis/metacontrol/tests/data/test_pipeline.py analysis/metacontrol/tests/data/test_generator.py
```

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

## 6. Talks and notebooks

| Resource | Content |
| :--- | :--- |
| `analysis/presentations/lmcos-overview/` | Motivation, architecture slides, **human** clock/VOC figures |
| `lmcos/LAB_NOTEBOOK.md` | CTS experiment log, pipeline stages, cluster run IDs |

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

For day-to-day commands and paths inside `chess_analysis`, use **`chess_analysis/human_analytics/README.md`** (pipeline CLIs under **`human_analytics/slurm/scripts/`**); for meta-controller code see **`lmcos/`** (`src/`, **`analysis/`**, stage-organized **`slurm/`**, **`slurm/configs/`**).

---

*Last updated to reflect `lmcos/LAB_NOTEBOOK.md`, `lmcos/slurm/` smoke controller configs, and `human_analytics/` layout (2026-05-29).*
