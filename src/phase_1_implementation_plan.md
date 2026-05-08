# Phase 1: Implementation Plan (Object-First Strategy)

This document details the staged implementation strategy for the `metacontrol` data generation pipeline, prioritizing readability and correctness before performance.

## 1. Directory Structure (Flat Data Layout)

```text
src/metacontrol/
├── core/
│   ├── tree.py             # SearchNode, SearchTree, and ASCII render()
│   ├── tensorizer.py       # TreeBatch and SnapshotTensorizer
│   └── schema.py           # Hard-coded chess feature definitions
├── data/
│   ├── engine.py           # Engine abstraction (LC0, Stockfish)
│   ├── generator.py        # Search logic (select, expand, backprop, record)
│   ├── targets_gnn.py      # GNN-specific target calculation (Edge WDLs)
│   ├── targets_mc.py       # MC-specific target calculation (DP Advantages)
│   └── dataset.py          # PyTorch Dataset and Snapshot packing
└── tests/
    ├── core/
    │   ├── test_tree.py
    │   └── test_tensorizer.py
    ├── data/
    │   ├── test_generator.py
    │   ├── test_targets_gnn.py
    │   └── test_targets_mc.py
    └── tactical/
        └── test_chess_logic.py
```

---

## 2. Class Schemas & Data Structures

### `core/tree.py`
- **`class SearchNode`**:
    - `node_id: int`
    - `fen: str`
    - `parent: Optional[SearchNode]`
    - `children: Dict[str, SearchNode]`
    - `visit_count: int`
    - `total_value: float`
    - `q_value: float`
    - `depth: int`
    - `is_terminal: bool`
    - `features: Dict[str, float]`

- **`class SearchTree`**:
    - `root: SearchNode`
    - `nodes_by_id: Dict[int, SearchNode]`
    - `expansion_history: List[SearchNode]`

---

## 3. Stage 1: Core Tree (`core/tree.py`)

### Proposed Tests
- **`test_single_leaf`**
- **`test_child_of_parent_is_self`**
- **`test_parent_of_child_is_self`**
- **`test_root_is_depth_zero`**
- **`test_tree_depth_after_adding_child`**
- **`test_no_cycles`** (Transposition unfolding)
- **`test_checkmate_is_leaf`**
- **`test_render_with_depth`** (ASCII validation)

---

## 4. Stage 1.5: Tensorization Link (`core/tensorizer.py`)

### Proposed Tests
- **`test_single_leaf_tensorization`**
- **`test_tensorization_different_sized_trees`**
- **`test_ids_unique`** (Flattened batch uniqueness)
- **`test_no_cycles_in_batch`**
- **`test_move_slot_stability`** (UCI sorting)

---

## 5. Stage 2: Search Generator (`data/generator.py`)

### Proposed Tests
- **`test_puct_selection`**
- **`test_backprop_averages`**
- **`test_alternating_turns`**
- **`test_discovery_dynamics`**
- **`test_expand_on_checkmate`**
- **`test_PUCT_should_converge_on_checkmate`**
- **`test_sacrifice_queen_for_checkmate`**
- **`test_Q_leaf_is_V`**

---

## 6. Stage 3: Target Logic (`data/targets_gnn.py` & `data/targets_mc.py`)

### Proposed Tests
- **`test_linear_dp_math`** (Targets MC)
- **`test_optimal_stop_logic`** (Targets MC)
- **`test_edge_wdl_consolidation`** (Targets GNN)
- **`test_build_snapshots_integrity`**

---

## 7. Stage 4: Performance Transition (Future)

| Optimization | Method | Purpose |
| :--- | :--- | :--- |
| **Profiling** | `cProfile` | Identify bottlenecks. |
| **Vectorization** | NumPy Arrays | Optimized memory layout. |
| **JIT** | Numba | Fast PUCT selection. |

## 8. Visualization & Debugging
- **`SearchTree.render()`**: Indented ASCII tree view.
- **`SearchNode.to_dict()`**: Dictionary view for inspection.
