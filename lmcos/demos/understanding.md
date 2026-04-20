# Conceptual Understanding: lmcos Meta-Controller

The `lmcos` architecture is a **meta-controller** suite. It uses Graph Neural Networks (GNNs) to decide optimally when to "halt" search and execute a move.

## 1. The GNN "Search Accelerator"

The GNN performs **representation learning** over search trees. Its goal is to compress the lopsided information of a search tree into a dense summary at the root node.

### The Two-Phase Round (Mechanistic Truth)
A single round of GNN processing consists of two sequential, depth-wise sweeps:

1.  **Phase 1: Upward - "Evidence Funnel"**
    *   Information flows from **Leaves to Parent**.
    *   Uses **Multi-Head Attention (MHA)** to aggregate child vectors.
    *   The Parent's memory is updated via a **GRU**.
    *   *Intuition:* "What have my children discovered that changes my own value?"

2.  **Phase 2: Downward - "Strategy Broadcast"**
    *   Information flows from **Parent to Children**.
    *   The Parent's summary is projected via a Linear layer and sent back down.
    *   Each Child's memory is updated via the *same* **GRU**.
    *   *Intuition:* "Given what the root now knows about the whole tree, how should my local move-vector change?"

### 3. The State Update (GRU)
Each node's representation is updated via a **GRU (Gated Recurrent Unit)**.
*   **Input:** The incoming message (Upward or Downward).
*   **Hidden State:** The node's current understanding.
*   **Why?** The GRU helps the node "remember" its original heuristic value while integrating new information from its neighbors, preventing the signal from fading.

---

## Design Intuition: Sinusoidal Slot Encodings

How does the GNN know which branch is the "Best Move" (Slot 0) vs. the "Alternative" (Slot 10)? It uses a **Continuous Address Space**.

### The Truth Table Analogy
Think of the slot encoding like a **Digital Bitmap** in a binary truth table. To represent numbers $0$ to $7$, you use bits that flip at different frequencies:
*   **LSB (Bit 0):** High Frequency ($0, 1, 0, 1, \dots$)
*   **Middle (Bit 1):** Medium Frequency ($0, 0, 1, 1, \dots$)
*   **MSB (Bit 2):** Low Frequency ($0, 0, 0, 0, 1, 1, 1, 1, \dots$)

### The Sinusoidal Upgrade
In `GNN.py`, we replace these sharp "bits" with smooth **Sine and Cosine waves**:
1.  **Differentiability:** Waves are smooth, so the GNN can learn to interpolate between "addresses."
2.  **Multiresolution:** High-frequency waves tell the GNN the **exact** move number (high-res), while low-frequency waves provide the **global context** (low-res, e.g., "I am one of the top 3 moves").
3.  **Numerical Stability:** No matter how many moves are in a position, every value in the address vector stays between $-1$ and $1$.

This "Digital Address" allows the **ChildWdlHead** to combine a generic state summary (`h_parent`) with a specific target address (`slot_encoding`) to generate a precise prediction for that specific branch.

---

### Topological Sequential Propagation
Unlike standard GNNs that "blur" information with neighbors, the `TreeNN` iterates node-by-node in **Topological Order**. This allows information to travel from the deepest leaf to the absolute root in a single Indigo pulse.

---

## 3. Supervised Pre-training: Representation Learning

The first stage of training is designed to build a powerful "Intuitive Engine." The GNN must learn to predict the **future** of a search tree before it ever learns to **control** it.

### The Learning Task
We generate training pairs using **Prefix Sampling**:
*   **Input ($T_{prefix}$):** A partial, shallow search tree.
*   **Target ($WDL_{oracle}$):** The eventual Win/Draw/Loss probabilities discovered by a deep Oracle search.
*   **Loss:** Cross-Entropy between the GNN's predicted logits and the Oracle's consolidated probabilities.

### 5. The Thinking Curve: Solving the Economy of Thought

The Meta-Controller (Tutorial 05) solves an **Optimal Stopping Problem**. It answers one question: *"Is the move-quality I'm about to discover worth the electricity I'm about to spend?"*

#### The Reward Signal ($R$)
The training logic uses **Dynamic Programming (DP)** to look at a full trace of search snapshots $T_0, T_1, \dots, T_K$. For each snapshot, we calculate a **Net Reward**:
$$R(k) = \text{Value}(T_k) - (C \cdot k)$$
*   **Value:** The expected win rate (WDL) of the best move at that moment.
*   **Cost ($C$):** A constant penalty for each unit of search (expansion).

#### The DP Oracle
The **Oracle Action** is derived by finding the global peak of this $R(k)$ curve. If search step $k$ is before the peak, the correct action is **CONTINUE**. If we are at or after the peak, the correct action is **HALT**.

By training on thousands of these "Thinking Curves," the Meta-Controller learns to recognize the visual patterns in the tree (e.g. high volatility, close move-scores) that flag a tactical breakthrough is worth the extra wait.

### 6. The Engineering Pipeline: Scaling to Production

While the tutorials focus on single "fragments," the production codebase is optimized to keep the GPU fully saturated. The journey from a raw board position to a trained controller follows this 4-stage pipeline:

#### Stage 1: Massive Dataset Generation (`generate-dataset`)
We use the `lc0` engine to perform thousands of deep searches (usually 800+ nodes). Each search results in a `.pt` file containing the full `SearchTree` and its eventual outcomes.

#### Stage 2: Data Packing (`pack_pretrain_examples.py`)
Reading 100,000 tiny files is a massive I/O bottleneck. In production, we "shard" the data. This script tensorizes thousands of trees into giant **"Fat Tensors"** and saves them with a pointer system (`node_ptr`, `edge_ptr`). This allows the trainer to load massive batches in a single disk read.

#### Stage 3: GNN Backbone Training (`pretrain-child-wdl-encoder`)
We train the GNN core (Tutorial 03-04) on these packed shards. This stage is computationally expensive and is typically run on high-performance clusters (see `slurm/`). The output is a frozen **`tree_encoder.pt`** backbone.

#### Stage 4: Meta-Control Optimization (`frozen-rl-generated`)
Finally, we attach the `HaltController` (Tutorial 05) to the frozen GNN. We use **PPO (Proximal Policy Optimization)** to help the agent explore the "Thinking Space" and learn the optimal balance between compute-cost and move-quality across live search traces.

### Mechanistic Truth: The Dense Recursive Signal
A crucial design choice in `lmcos` is that the **ChildWDL Head is not anchored solely to the root.**
*   **Any Node can be a Parent:** During pre-training, the readout head is applied to **every edge** in the tree batch.
*   **Dense Gradients:** If a tree has $N$ nodes, we get $N-1$ training signals in a single forward pass. Every node acts as a "Local Root" that must justify its representation by predicting its children's future.
*   **Recursive Depth Invariance:** This forces the GNN to learn board patterns and subtree relationships that work regardless of whether the node is at depth 0 or depth 10.

---

## Code Map & Curriculum

| Concept | Production File | Tutorial / Demo |
| :--- | :--- | :--- |
| **Prefix Sampling** | `cts_pretrain.py` | [01_prefix_tutorial.ipynb](./01_prefix_tutorial.ipynb) |
| **Tensorization** | `tensorizer.py` | [02_tensorization_tutorial.ipynb](./02_tensorization_tutorial.ipynb) |
| **Bidirectional Sweep**| `GNN.py` | [03_gnn_tutorial.ipynb](./03_gnn_tutorial.ipynb) |
| **Supervised Loop** | `cts_pretrain.py` | [04_pretrain_tutorial.ipynb](./04_pretrain_tutorial.ipynb) |
| **Halt/Continue RL** | `cts_rl.py` | [05_meta_controller_tutorial.ipynb](./05_meta_controller_tutorial.ipynb) |

---

## Visual Vocabulary

When reading the diagrams in this suite, keep these formal distinctions in mind:

| Visual Element | Meaning | Mechanistic Part |
| :--- | :--- | :--- |
| **Hollow Box** | **Data Container** | Hidden State ($h$), Messages ($m$) |
| **Filled Box** | **Neural Module** | GRUCell, Linear, MH-Attention |
| **Solid Lines** | **Active Search** | Search edges ($T_k$) |
| **Dashed Lines** | **Future Potential** | Oracle Nodes (not yet seen by GNN) |

---

## Glossary

*   **Oracle ($\omega$):** A deeply searched tree used as ground truth for supervised training.
*   **Thinking Cost ($C$):** The linear penalty used to train the meta-controller's halting policy.
*   **Sinusoidal Slot Encoding:** A method for telling the GNN *which* move an edge represents (e.g., move 1 vs move 3) using periodic functions.
*   **TreeBatch:** The flattened tensor representation used for parallel GPU execution of tree sweeps.
