# Conceptual Understanding: lmcos Meta-Controller

The `lmcos` architecture is a **meta-controller** suite. It uses Graph Neural Networks (GNNs) to decide optimally when to "halt" search and execute a move.

## 1. The GNN "Search Accelerator"

The GNN performs **representation learning** over search trees. Its goal is to compress the lopsided information of a search tree into a dense summary at the root node.

### The Two-Phase Round (Mechanistic Truth)
A single round of GNN processing consists of two sequential, depth-wise sweeps:

1.  **Phase 1: Upward (Indigo) - "Evidence Funnel"**
    *   Information flows from **Leaves to Parent**.
    *   Uses **Multi-Head Attention (MHA)** to aggregate child vectors.
    *   The Parent's memory is updated via a **GRU**.
    *   *Intuition:* "What have my children discovered that changes my own value?"

2.  **Phase 2: Downward (Emerald) - "Strategy Broadcast"**
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
| **Indigo Color** | **Primary/Summary** | Upward Pass, Root States |
| **Emerald Color** | **Context/Broadcast** | Downward Pass, Child Updates |
| **Dashed Lines** | **Future Potential** | Oracle Nodes (not yet seen by GNN) |

---

## Glossary

*   **Oracle ($\omega$):** A deeply searched tree used as ground truth for supervised training.
*   **Thinking Cost ($C$):** The linear penalty used to train the meta-controller's halting policy.
*   **Sinusoidal Slot Encoding:** A method for telling the GNN *which* move an edge represents (e.g., move 1 vs move 3) using periodic functions.
*   **TreeBatch:** The flattened tensor representation used for parallel GPU execution of tree sweeps.
