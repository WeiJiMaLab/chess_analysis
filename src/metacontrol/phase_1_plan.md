# Metacontrol Data Generation Pipeline: Stage 2 & Beyond

This document outlines the remaining steps for the `metacontrol` pipeline following the successful implementation and modularization of the core tree and target derivation logic.

## 1. Current State (Completed)
- [x] **Modular Architecture**: Established `core/` for schemas and data structures, and `data/` for pipeline logic.
- [x] **Core Tree**: Implemented `SearchNode` and `SearchTree` with ASCII rendering.
- [x] **Tree Growth**: Implemented `TreeSearch` (formerly `SearchGenerator`) with PUCT selection and backpropagation.
- [x] **Target Derivation**: Implemented deterministic DP for meta-control advantages (`targets_mc.py`) and GNN target consolidation (`targets_gnn.py`).
- [x] **Tensorization**: Integrated `TreeTensorizer` for batching trees into GNN-compatible formats.
- [x] **Validation**: 100% pass rate on 50 integration and unit tests.

---

## 2. Upcoming Stage: Root Selection & Dataset Generation

### Stage 2.5: Root Sampling Logic
- **Objective**: Create a script to sample root FENs from the Lichess database mirror (`lichess.db`) matching original research criteria.
- **Implementation**:
    - Build a DuckDB-based sampler in `src/metacontrol/data/sampler.py`.
    - Apply filters: 
        - ELO [1800, 2600]
        - Ply [8, 120]
        - Legal Moves [2, 60]
        - Piece Count [8, 32]
- **Parameters (Faithful to Original)**:
    - `max_nodes`: **64** (Teacher search budget)
    - `min_nodes`: **5** (Minimum snapshot size)
    - `max_depth`: **10**
    - `search_budget`: **64** (Engine sims per expansion)

---

## 3. Stage 5: Large-Scale Production Pipeline
- **Objective**: Parallelize the generation of millions of snapshots across the cluster.
- **Tasks**:
    - [ ] **Engine Integration**: Connect the pipeline to the `lc0` binary via a production-ready `TreeExpansionProvider`.
    - [ ] **Sharding**: Implement logic to save trees in packed shards (`.pt` files) for high-throughput training.
    - [ ] **Slurm Orchestration**: Develop job templates for distributed generation.

---

## 4. Stage 6: Training Orchestration
- **Objective**: Rebuild the training loops for the GNN and the Meta-Control head.
- **Tasks**:
    - [ ] **GNN Pretraining**: Supervised learning on `node_target_values` and `edge_wdl_targets`.
    - [ ] **Controller Training**: Fitted-Q / Advantage regression using the Bellman-derived targets from `targets_mc.py`.
    - [ ] **Evaluation Suite**: Measure oracle agreement and "economy of thought" metrics.

---

## 5. Performance Goals
| Optimization | Method | Purpose |
| :--- | :--- | :--- |
| **Vectorization** | PyTorch/NumPy | Optimized memory layout for tree batches. |
| **I/O Efficiency** | Sharded Packing | Fast loading during GPU training. |
| **Parallelism** | Slurm Job Arrays | Multi-node generation throughput. |

---

## 6. Visualization & Debugging
- **Tree Inspector**: Extend `SearchTree.render()` to include derived targets (advantages/WDLs).
- **Trajectory Analysis**: Plot halt rewards vs. expansion steps to verify "value of computation" landscapes.
