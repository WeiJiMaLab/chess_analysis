## lmcos demos

This directory contains small, self-contained scripts intended to make the `lmcos` components easier to understand one-by-one.

### How `lmcos` fits together

The repo is a **data pipeline → tensorization → encoder → controller** stack.

| Stage | Name | Description | Demo |
| :--- | :--- | :--- | :--- |
| **1** | **Tree Growth** | Building a search tree using PUCT and `lc0`. | `demo_partial_tree_generation.py` |
| **2** | **Packing** | Saving/loading `PretrainExample` datasets. | *(Internal scripts)* |
| **3** | **Tensorization** | Turning trees into batched PyTorch tensors. | *(Planned: demo_tensorization.py)* |
| **4** | **Tree Encoder** | Message passing over the tree via `TreeNN`. | *(Planned: demo_tree_nn.py)* |
| **5** | **Controller** | RL environment for halt/continue decisions. | *(Planned: demo_meta_controller.py)* |

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

### `demo_partial_tree_generation.py`

Builds a small **partial tree** (Mode 2) from a root FEN and renders the WDL statistics (static vs. visit-weighted teacher targets) as an SVG.

*   **Move:** The move that reached this node.
*   **wdl(s):** Static win/draw/loss from the `lc0` value-head.
*   **teacher WDL:** The search-weighted target.
    *   **Blue**: Consolidated from child edge statistics (at least one visit).
    *   **Gray**: Static fallback (leaf or node with zero visits).

**Example:**
```bash
python demos/demo_partial_tree_generation.py \
  --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" \
  --max-depth 3 \
  --min-nodes 2 \
  --max-nodes 3 \
  --out-svg demos/figures/demo_partial_tree_generation.svg
```

