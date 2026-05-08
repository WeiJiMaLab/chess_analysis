# Phase 1: Implementation Plan & Design (Object-First Strategy)

This document details the staged implementation strategy and granular design for the `metacontrol` data generation pipeline, prioritizing readability and correctness before performance. It replaces the redundant search-per-snapshot logic with a unified search and a single DP pass.

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
    │   ├── test_targets_mc.py
    │   └── test_pipeline.py
    └── tactical/
        └── test_chess_logic.py
```

## 2. Class Schemas & Data Structures

### `core/tree.py`
We will simplify the `SearchTree` to be a pure data structure without search logic.
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
    - `features: Dict[str, float]` (Hard-coded keys: `value`, `wdl_win`, `wdl_draw`, `wdl_loss`)

- **`class SearchTree`**:
    - `root: SearchNode`
    - `nodes_by_id: Dict[int, SearchNode]`
    - `expansion_history: List[SearchNode]`

### `core/tensorizer.py`
- **`class TreeTensorizer`**:
    - `def tensorize(tree: SearchTree) -> TreeBatch`: Converts a tree to the packed format used by the GNN.

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

## 4. Stage 1.5: Tensorization Link (`core/tensorizer.py`)

### Proposed Tests
- **`test_single_leaf_tensorization`**
- **`test_tensorization_different_sized_trees`**
- **`test_ids_unique`** (Flattened batch uniqueness)
- **`test_no_cycles_in_batch`**
- **`test_move_slot_stability`** (UCI sorting)
- **`test_round_trip`**: Verify that tensorized data contains the same node counts and edge relationships.

## 5. Stage 2: Search Generator (`data/generator.py`)

We will consolidate the disparate tree-building logic into a single class.

### API
```python
@dataclass
class GeneratorConfig:
    max_nodes: int
    max_depth: int
    c_puct: float

class SearchGenerator:
    """Unified generator that grows a tree and maintains search stats."""
    def __init__(self, provider: TreeExpansionProvider, config: GeneratorConfig):
        self.provider = provider
        self.config = config

    def generate(self, root_fen: str) -> SearchTree:
        """
        Grows a tree up to max_nodes using PUCT selection.
        Updates edge_stats (Q-values and WDLs) during backpropagation.
        """
```

### Proposed Tests
- **`test_puct_selection`**
- **`test_backprop_averages`**
- **`test_alternating_turns`**
- **`test_discovery_dynamics`**
- **`test_expand_on_checkmate`**
- **`test_PUCT_should_converge_on_checkmate`**
- **`test_sacrifice_queen_for_checkmate`**
- **`test_Q_leaf_is_V`**
- **`test_convergence`**: Assert that with a very high budget, the tree approximates minimax values.
- **`test_invariants`**: Assert `node_count <= max_nodes` and `depth <= max_depth`.

## 6. Stage 3: Target Logic (`data/targets_gnn.py` & `data/targets_mc.py`)

This replaces the redundant search-per-snapshot logic with a single DP pass.

### API
```python
@dataclass
class SearchSnapshot:
    tree_prefix: SearchTree
    halt_reward: float        # Value of best move at current snapshot
    continue_value: float     # Optimal value from future search - cost
    advantage: float          # continue_value - halt_reward

def calculate_dp_values(tree: SearchTree, continue_cost: float) -> Dict[int, float]:
    """
    Performs a bottom-up DP pass over the tree to find the optimal value 
    of each node under the stop/continue cost.
    V(node) = max(V_halt(node), V_continue(node))
    """

def derive_snapshots(full_tree: SearchTree, continue_cost: float) -> List[SearchSnapshot]:
    """
    Iterates through the expansion sequence of the full_tree.
    For each prefix, creates a SearchSnapshot with pre-calculated DP targets.
    """
```

### Proposed Tests
- **`test_linear_dp_math`** (Targets MC): Create a manual 3-node tree and verify advantages for different costs.
- **`test_optimal_stop_logic`** (Targets MC)
- **`test_monotonicity`** (Targets MC): Assert `V_halt` is non-decreasing as more nodes are added.
- **`test_edge_wdl_consolidation`** (Targets GNN)
- **`test_build_snapshots_integrity`**

## 7. Stage 4: Integration & Pipeline (`test_pipeline.py`)

### Proposed Tests
- **`test_pipeline.py`**:
    - End-to-end test: `FEN` $\rightarrow$ `SearchGenerator` $\rightarrow$ `derive_snapshots` $\rightarrow$ `DataLoader`.
    - Verify that the resulting batch of snapshots has valid `advantage` targets.

## 8. Stage 5: Performance Transition (Future)

| Optimization | Method | Purpose |
| :--- | :--- | :--- |
| **Profiling** | `cProfile` | Identify bottlenecks. |
| **Vectorization** | NumPy Arrays | Optimized memory layout. |
| **JIT** | Numba | Fast PUCT selection. |

## 9. Visualization & Debugging
- **`SearchTree.render()`**: Indented ASCII tree view.
- **`SearchNode.to_dict()`**: Dictionary view for inspection.

## 10. Why this is better
- **Efficiency**: One search builds the tree; one DP pass builds all targets. No redundant LC0 calls or re-searches.
- **Readability**: Clear separation between *growing* the tree and *calculating* what the optimal meta-control decision should have been.
- **Simplicity**: No more "if node_budget is None" branches.
