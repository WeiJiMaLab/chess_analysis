## lmcos demos

This directory contains a didactic curriculum and self-contained scripts designed to make the `lmcos` components transparent and mechanistically clear.

### Tutorial Series: GNN Training Pipeline

Follow this sequence to understand how raw search trees are transformed into a neural training signal.

| Order | Tutorial | Mechanistic Focus | Key File |
| :--- | :--- | :--- | :--- |
| 1 | [01_prefix_tutorial.ipynb](./01_prefix_tutorial.ipynb) | **Data:** Prefix Sampling & Deep Target Consolidation | `cts_pretrain.py` |
| 2 | [02_tensorization_tutorial.ipynb](./02_tensorization_tutorial.ipynb) | **Representation:** Flat-forest "Folding" for GPUs | `tensorizer.py` |
| 3 | [03_gnn_tutorial.ipynb](./03_gnn_tutorial.ipynb) | **Compute:** Bidirectional Sequential Swaps | `GNN.py` |
| 4 | [04_pretrain_tutorial.ipynb](./04_pretrain_tutorial.ipynb) | **Learning:** Supervised Loss & Readout Heads | `cts_pretrain.py` |
| 5 | [05_meta_controller_tutorial.ipynb](./05_meta_controller_tutorial.ipynb) | **Decision:** The Halt/Continue RL Policy | `cts_rl.py` |

---

### Visual Language (The "Formal Graph" Aesthetic)

All tutorials and helper scripts use a unified visual system to distinguish between **Logic** and **Data**:

*   **Indigo (#4338ca):** Represents **Upward** flow, "Self" features, or **Root/Parent** summary states.
*   **Emerald (#059669):** Represents **Downward** flow, Global Context, or **Child** update states.
*   **Hollow Rounded Box:** Represents **Data** (Hidden State Vectors, $h$).
*   **Filled Rounded Box:** Represents **Logic/Modules** (GRU, Attention, Projections).
*   **Slate/Gray:** Represents the past, future, or high-level structure (The Search Tree).

---

### Core Data Workflows

`build_pretrain_example` in `cts_pretrain.py` handles the creation of GNN training pairs. We primarily use **Mode 2 (Dynamic Growth)** for research:

1. **Oracle Tree Generation:** A full search is performed (usually with a budget of 64+ expansions).
2. **Prefix Sampling:** That tree is "chopped" at a random expansion count (e.g., $k=10$) to create the **Input Prefix**.
3. **Consolidation:** The statistics from the *full* search are propagated back to the nodes in the *prefix* to create **Deep Targets**.
4. **Tensorization:** The lopsided prefix tree is flattened into a `TreeBatch` for the GNN.

---

### Command Line Tools

#### `prefix_demo.py`
Visualizes **Prefix Sampling**—showing exactly how we blend the GNN's partial view with the "Future Signal" from the Oracle.

```bash
python demos/prefix_demo.py --oracle-nodes 12 --prefix-nodes 4
```

See [understanding.md](./understanding.md) for a deep dive into the GNN's internal wiring.
