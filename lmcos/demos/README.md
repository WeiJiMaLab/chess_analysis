## lmcos demos

This directory contains small, self-contained scripts intended to make the `lmcos` components easier to understand one-by-one.

### Tutorial Series: GNN Training Pipeline

This directory is organized into a didactic curriculum to help you understand how `lmcos` processes irregular search trees.

| Order | Tutorial | Mechanistic Focus | Key File |
| :--- | :--- | :--- | :--- |
| 1 | [01_prefix_tutorial.ipynb](./01_prefix_tutorial.ipynb) | **Data:** Prefix Sampling & Consolidation | `cts_pretrain.py` |
| 2 | [02_tensorization_tutorial.ipynb](./02_tensorization_tutorial.ipynb) | **Representation:** Flat-forest Tensorization | `tensorizer.py` |
| 3 | [03_gnn_tutorial.ipynb](./03_gnn_tutorial.ipynb) | **Compute:** Bidirectional Sequential Sweep | `GNN.py` |

## Design Intuition
See [understanding.md](./understanding.md) for deep-dives into:
*   The **"Flattened Forest"** memory layout.
*   The **"Bidirectional Synchronized Round"** (Top-to-bottom and bottom-to-top propagation).
*   The **"Search Accelerator"** (GNN) architecture details.

---

### Data Generation Modes

`build_pretrain_example` in `cts_pretrain.py` creates teacher targets in two ways. **Mode 2 is the primary method used for research.**

#### Mode 1: Fixed Skeleton (Vestigial / Legacy)
*   **Workflow:** `build_tree_from_provider` → `compute_teacher_targets`
*   **Mechanism:** It builds a "complete" skeleton up to `max_depth` first. Then, it runs a fixed number of PUCT rollouts on that **fixed topology** to refine edge statistics.
*   **Intuition:** "Cast a wide net, then simulate to see which parts of the net are strongest." This is largely vestigial and was used for initial value definitions.

#### Mode 2: Dynamic Growth (Search-Driven / Current)
*   **Workflow:** `generate_partial_tree_from_provider` → `consolidate_generated_tree`
*   **Mechanism:** The tree structure is an **output of the search**. It starts with a root and grows one node at a time based on PUCT selection until a budget is hit. The statistics are collected **online** during growth.
*   **Intuition:** "Explore where it's promising until I run out of budget." This produces the lopsided, realistic trees needed for training the meta-controller.

---

### `prefix_demo.py`

Visualizes **Prefix Sampling**—how we create training pairs by "chopping" an Oracle tree.

*   **Unified Tree:** Renders a single search context that blends the past (GNN Input) and the future (Oracle Signal).
*   **Blue Solid Nodes:** Nodes available to the GNN. Shows both the **INPUT** (static heuristic) and the **TARGET** (final search value).
*   **Gray Dashed Nodes:** The "future" expansions that justify the targets in the blue nodes.

**Example:**
```bash
python demos/prefix_demo.py \
  --oracle-nodes 12 \
  --prefix-nodes 4 \
  --out-svg demos/figures/prefix_demo.svg
```

