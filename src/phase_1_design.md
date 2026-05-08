# Phase 1 Design: Data Generation & Targets

This document provides a granular design for the first phase of the `lmcos` to `metacontrol` migration.

## 1. Core Representations (`core/`)

### `core/tree.py`
We will simplify the `SearchTree` to be a pure data structure without search logic.
- **`class Node`**:
    - `node_id: int`
    - `fen: str`
    - `depth: int`
    - `is_terminal: bool`
    - `scalar_features: Dict[str, float]` (Hard-coded keys: `value`, `wdl_win`, `wdl_draw`, `wdl_loss`)
- **`class SearchTree`**:
    - `nodes: Dict[int, Node]`
    - `edges: List[Tuple[int, int]]`
    - `edge_metadata: Dict[Tuple[int, int], Dict[str, Any]]` (Includes `visit_count`, `q_value`, `mean_wdl`)

### `core/tensorizer.py`
- **`class TreeTensorizer`**:
    - `def tensorize(tree: SearchTree) -> TreeBatch`: Converts a tree to the packed format used by the GNN.

---

## 2. Tree Generation (`data/gnn/generator.py`)

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

---

## 3. Snapshot & Target Derivation (`data/mc/targets.py`)

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

---

## 4. Testing Plan

### Unit Tests
1.  **`test_generator.py`**:
    - **Convergence**: Assert that with a very high budget, the tree approximates minimax values.
    - **Invariants**: Assert `node_count <= max_nodes` and `depth <= max_depth`.
2.  **`test_targets.py`**:
    - **Toy Tree DP**: Create a manual 3-node tree and verify that `calculate_dp_values` returns the expected advantages for different `continue_cost` values.
    - **Monotonicity**: Assert that `V_halt` (best move value) is non-decreasing as more nodes are added (or verify it matches the teacher's behavior).
3.  **`test_tensorizer.py`**:
    - **Round-trip**: Verify that tensorized data contains the same node counts and edge relationships as the source tree.

### Integration Tests
1.  **`test_pipeline.py`**:
    - End-to-end test: `FEN` $\rightarrow$ `SearchGenerator` $\rightarrow$ `derive_snapshots` $\rightarrow$ `DataLoader`.
    - Verify that the resulting batch of snapshots has valid `advantage` targets.

## 5. Why this is better
- **Efficiency**: One search builds the tree; one DP pass builds all targets. No redundant LC0 calls or re-searches.
- **Readability**: Clear separation between *growing* the tree and *calculating* what the optimal meta-control decision should have been.
- **Simplicity**: No more "if node_budget is None" branches.
