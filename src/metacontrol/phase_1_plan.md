# Metacontrol Data Generation Pipeline: Stage 2 & Beyond

This document outlines the remaining steps for the `metacontrol` pipeline following the successful implementation and modularization of the core tree and target derivation logic.

## 1. Current State (Completed)
- [x] **Modular Architecture**: Established `core/` for schemas and data structures, and `data/` for pipeline logic.
- [x] **Core Tree**: Implemented `SearchNode` and `SearchTree` with ASCII rendering.
- [x] **Tree Growth**: Implemented `TreeSearch` (formerly `SearchGenerator`) with PUCT selection and backpropagation.
- [x] **Target Derivation**: Implemented deterministic DP for meta-control advantages (`targets_mc.py`) and GNN target consolidation (`targets_gnn.py`).
- [x] **Tensorization**: Integrated `TreeTensorizer` for batching trees into GNN-compatible formats.
- [x] **Engine Integration**: Connected to real `lc0` and `stockfish` binaries, fixed perspective bugs, and added dynamic MultiPV capping for stability.
- [x] **Validation**: 100% pass rate on 67 integration and unit tests, including rule-based terminal logic, deep tactical searches (mate-in-1, mate-in-2), and legacy parity comparisons.

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
- [x] **Engine Integration**: Connected the pipeline to `lc0` and `stockfish` via the generalized `UciExpansionProvider`.
- [x] **Visualization & Debugging**: Created `plotting.py` for Graphviz tree rendering and advantage landscape analysis.
- [ ] **Sharding**: Implement logic to save trees in packed shards (`.pt` files) for high-throughput training.
- [ ] **Slurm Orchestration**: Develop job templates for distributed generation.
