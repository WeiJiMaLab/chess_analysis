# Migration Plan: lmcos to src/metacontrol

This document outlines the migration plan to move the `lmcos` codebase into a more structured `src/metacontrol` directory with a clear separation of concerns.

## 1. Project Overview
The goal of the `metacontrol` (formerly `lmcos`) project is to develop a neural meta-controller that decides when to continue or halt an internal tree search based on a learned representation of the search tree.

## 2. Target Directory Structure
We aim for the following separation of concerns in `src/metacontrol`:

```text
src/metacontrol/
├── core/
│   ├── tree.py             # Basic SearchTree and Node classes (from tree.py)
│   └── tensorizer.py       # Tree to tensor conversion (from tensorizer.py)
├── data/
│   ├── gnn/
│   │   ├── generator.py    # Tree building via Teacher Search (from cts_pretrain.py)
│   │   └── dataset.py      # Child-WDL example packing (from pack_pretrain_examples.py)
│   ├── mc/
│   │   ├── snapshots.py    # Deriving snapshots from full trees (from cts_episode_envs.py)
│   │   └── targets.py      # DP calculation for counterfactual advantage (new/refactored)
│   └── common.py           # Shared dataset/loader utilities
├── models/
│   ├── gnn.py              # TreeNN with bi-directional sweeps (from GNN.py)
│   └── controller.py       # Halt/Advantage heads (from GNN.py)
├── tasks/
│   ├── gnn_pretrain/
│   │   ├── trainer.py      # Supervised training for Child-WDL (from cts_pretrain.py)
│   │   └── eval.py         # WDL prediction metrics
│   └── metacontrol/
│       ├── ppo_trainer.py  # PPO implementation (from cts_rl.py)
│       ├── q_trainer.py    # Fitted-Q/Advantage training (from train_fitted_q_controller.py)
│       └── env.py          # RL Environment for stop/continue (from cts_episode_envs.py)
└── scripts/
    ├── generate_trees.py   # CLI for tree generation
    ├── run_pretrain.py     # CLI for GNN pretraining
    ├── run_mc_train.py     # CLI for MC training
    └── analyze.py          # Unified analysis script
```

## 3. Implementation Details

### GNN: Bi-directional Sweeps
The `TreeNN` architecture currently implements bi-directional sweeps in `_forward_sequential`.
- **Upward Pass**: Propagates information from leaves to root using a GRU and Attention Message Layer.
- **Downward Pass**: Propagates root information back to leaves.
- **Refactor**: Ensure these sweeps are explicitly modular and can be configured for different `k` iterations.

### MC: DP Targets for Advantage
The meta-controller should be trained on DP-derived targets.
- **Halt Reward**: The value of the best move at the current snapshot.
- **Continue Value**: The value of the best move in the *future* (from the full tree) minus the `continue_cost`.
- **DP**: For each snapshot $s_t$, we calculate the counterfactual advantage $A(s_t) = Q_{continue}(s_t) - Q_{halt}(s_t)$.
- **Refactor**: Move this logic into `data/mc/targets.py` to decouple target generation from the training loop.

## 3. Migration Roadmap

### Phase 1: Data Generation & Targets
- **Goal**: Clean up the tree building and target calculation logic.
- **Source**: `cts_pretrain.py`, `cts_episode_envs.py`, `tree.py`, `tensorizer.py`.
- **Refactor**:
    - Consolidate tree building into a single `NodeBudget`-based path. Remove the legacy `build_tree_from_provider` path.
    - Simplify `TeacherSearchConfig` and `NodeBudgetDistribution` into a unified `SearchGenerator` config.
    - Implement a clean DP-based target generator for MC in `data/mc/targets.py`.
    - Flatten `PretrainExample` and `GeneratedTree` into a single, well-documented `SearchSnapshot` structure.
- **Readability**: Avoid "optional-everything" constructors. Use explicit, required arguments for core logic.

### Phase 2: Core Representations & GNN Architecture
- **Goal**: Modularize the GNN and core data structures.
- **Source**: `tree.py`, `schema.py`, `GNN.py`, `TreeMHA.py`.
- **Refactor**:
    - Move `SearchTree` and `Node` to `core/tree.py`.
    - Merge `TreeMHA.py` into `models/gnn.py`.
    - Simplify `TreeNN` bi-directional sweeps. Remove `sequential=False` (synchronous) mode if not needed.
    - Consolidate the four different model wrappers in `GNN.py` into a single `ModularTreeModel` that takes a list of heads.
- **Readability**: Use clear names like `upward_pass` and `downward_pass`.

### Phase 3: Training Tasks & CLI
- **Goal**: Implement clean training loops and unified CLI.
- **Source**: `cts_rl.py`, `train_fitted_q_controller.py`, `supervised_branch_cli.py`.
- **Refactor**:
    - Create `tasks/gnn_pretrain/trainer.py` for Child-WDL training.
    - Create `tasks/metacontrol/ppo_trainer.py` and `tasks/metacontrol/q_trainer.py`.
    - Simplify PPO logic: remove `Reinforce` if it's no longer used.
    - Replace the massive `supervised_branch_cli.py` with smaller, task-focused scripts.
- **Readability**: Log metrics in a consistent format. Remove complex "trace_callback" logic if simple logging suffices.

## 4. Immediate Next Steps
1. [x] Audit `cts_pretrain.py` to strip out legacy tree-building modes.
2. [x] Define the `SearchSnapshot` data format for Phase 1.
3. [x] Create `src/metacontrol/core/` and move the baseline tree logic.
4. [x] Harden engine providers (LC0, Stockfish), fix perspective bugs, and establish engine-invariant tests.
5. [ ] Implement `src/metacontrol/data/sampler.py` for root FEN extraction from Lichess DB.
6. [ ] Implement SLURM job orchestration for clustered data generation.
