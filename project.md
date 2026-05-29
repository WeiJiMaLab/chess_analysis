# Project: Chess Meta-control (CMC)
> [!IMPORTANT]
> To run code in this repository, you MUST activate the `.venv` first:
> `source .venv/bin/activate` (or equivalent for your shell).

## 1. Objective

To develop a **Meta-Controller (MC)** that manages the trade-off between "thinking" (expanding a Leela Chess Engine search tree) and "acting" (executing a move). The model aims at **Policy Compression**—maximizing win reliability while minimizing computational cost—and, at the control layer, solves an **optimal stopping** problem: *is the move-quality we are about to discover worth the compute we are about to spend?*

The `lmcos` stack implements this as a **meta-controller suite**: GNNs over search trees learn dense representations; a **Halt/Continue** policy is trained on full search traces so the agent recognizes when extra search pays off (e.g. volatile, close scores) versus when halting is best.

---
## 2. Architecture: Two-Phase Tree GNN (Mechanistic Truth)

The MC operates over the search tree $\mathcal{T}$ produced by Leela. The GNN **compresses** lopsided tree information into a **dense summary at the root** (and, during training, uses **dense supervision** at every parent–child edge—see Section 3).

Unlike a single synchronous "round" that mixes parent and child messages in one step, **one round of GNN processing is two sequential, depth-wise sweeps**:

### Phase 1: Upward (Indigo) — "Evidence funnel"

- Information flows **from leaves toward the root** (child $\to$ parent).
- Children are aggregated with **multi-head attention (MHA)**.
- The parent's memory is updated with a **GRU**.
- *Intuition:* "What have my children discovered that changes my own value?"

### Phase 2: Downward (Emerald) — "Strategy broadcast"

- Information flows **from parent to children**.
- The parent's summary is projected (e.g. **linear**) and sent down.
- Each child's memory is updated with the **same GRU** as in the upward phase.
- *Intuition:* "Given what the root knows about the whole tree, how should my local move-vector change?"

### State update (GRU)

Each node's representation is updated via a **GRU**: **incoming message** (upward or downward) as input, **current hidden state** as memory. That lets the node retain its original heuristic signal while integrating neighborhood information and limits signal fade.

### Topological sequential propagation

**TreeNN** advances **node-by-node in topological order**, not by blurring with all neighbors at once. In one upward **pulse**, information can travel from the deepest leaf to the root along valid tree paths.

### Sinusoidal slot encodings

The GNN must know **which legal move** an edge represents (e.g. best move vs. tenth alternative). **Sinusoidal slot encodings** provide a continuous "address": smooth sine/cosine features (multiresolution: high frequency $\approx$ fine move index, low frequency $\approx$ coarse rank), values bounded for stability, and differentiable interpolation between slots. The **ChildWDL head** combines a generic parent summary $h_{\mathrm{parent}}$ with this address so each branch gets a precise prediction.

---

## 3. Training Story: Supervision First, Then Control

### Supervised pre-training (representation learning)

The first stage builds an **intuitive engine** that predicts the **future** of a tree before learning to **control** search.

- **Prefix sampling:** input is a **partial, shallow** tree $T_{\mathrm{prefix}}$; target is **Win/Draw/Loss** from a **deep oracle** search, $WDL_{\mathrm{oracle}}$.
- **Loss:** cross-entropy between predicted logits and oracle probabilities.

**Dense ChildWDL signal:** the readout is **not** only at the root. **Any node can act as parent:** the head is applied to **every edge** in the batch, so a tree with $N$ nodes yields $N-1$ supervision signals per forward pass—**recursive depth invariance** and strong gradients for subtree structure.

### Meta-control (optimal stopping on the "thinking curve")

Given snapshots $T_0, T_1, \ldots, T_K$ along a search trace, define **net reward** at step $k$:

$$
R(k) = \mathrm{Value}(T_k) - C \cdot k
$$

- **Value:** expected win rate (WDL) of the best move at that snapshot.
- **Cost ($C$):** penalty per unit of search (e.g. expansion).

A **DP oracle** labels actions from the **global** peak of $R(k)$: **CONTINUE** before the peak, **HALT** at or after. Training on many such curves teaches the **HaltController** to recognize tree patterns that justify extra search vs. stopping.

*(This aligns with a cost-sensitive RL view: maximize outcome while penalizing total search; the snapshot formulation makes the stopping rule explicit.)*

---

## 4. Production Pipeline (Scaling)

1. **Massive dataset generation** — deep `lc0` searches (e.g. 800+ nodes); each run yields a serialized **SearchTree** and outcomes (e.g. `.pt`).
2. **Data packing** — shard many trees into **fat tensors** with pointer layouts (`node_ptr`, `edge_ptr`) so the trainer does few large reads instead of many small files.
3. **GNN backbone pre-training** — supervised ChildWDL / encoder training on packed data; output a frozen **`tree_encoder.pt`** (cluster jobs often via `slurm/`).
4. **Frozen RL meta-control** — attach **HaltController** to the frozen encoder; **PPO** explores the thinking space and learns compute vs. quality trade-offs on live traces.

---

## 5. Division of Responsibilities

### **Infrastructure & Integration (Lead: Yotam)**

- **Leela wrapper:** pause/resume search, export tree tensors.
- **GNN core:** bidirectional sweeps (MHA + GRU upward; linear broadcast + GRU downward), slot encodings, **TreeBatch** / topological execution.
- **RL training:** PPO (or related) on halt/continue with cost $C$; optional alignment with DP oracle labels from thinking curves.

### **Validation & Psychology (Lead: Jordan)**

- **Baselines:** TreeLSTM (bottom-up only), log-entropy stopping, linear CP vs. move time, etc.
- **Human alignment:** RT regression vs. human move times; clock-scramble regimes; feature analysis (reliability, volatility, controllability) vs. "think" triggers.

---

## 6. Development Roadmap

- **Phase 1:** End-to-end data path — Leela trees $\to$ tensorization $\to$ GNN forward pass.
- **Phase 2:** Supervised pre-training (prefix vs. oracle WDL); frozen **tree_encoder**.
- **Phase 3:** Meta-control RL (HaltController + PPO) with cost $C$; DP/thinking-curve evaluation.
- **Phase 4:** Psychometric validation vs. Lichess/FICS human move data.

---

## 7. Reference: `lmcos` Code Map (High Level)

| Concept | Production direction |
| :--- | :--- |
| Prefix sampling / supervised loop | `cts_pretrain.py` |
| Tensorization | `tensorizer.py` |
| Bidirectional sweep | `GNN.py` |
| Halt/Continue RL | `cts_rl.py` |

See **`lmcos/LAB_NOTEBOOK.md`** and **`lmcos/slurm/README.md`** for the CTS pipeline curriculum and stage layout.
# Coding style in `human_analytics/` (chess_analysis)

This note describes how Python (and notebook) code in `human_analytics/` is written. It is descriptive, not a strict linter profile.

## Design priorities

Code is optimized for **modularity** and **readability**. Preference is given to clarity of data flow and obviously named operations.

## Plotting Standards (Poster Style)

All visualizations aimed at analysis and presentation must follow the **Poster Design System**:

- **Aesthetics**:
    - **No Subplot Titles**: Remove repeating or internal titles. Context is provided by axis labels and the surrounding narrative.
    - **Minimal Borders**: Remove top and right axes spines (`ax.spines['top'].set_visible(False)`).
    - **XY Grid**: Use subtle background grids where sensible (`ax.grid(True, alpha=0.3)`).
    - **Color Palette**: Use `MAIN_COLOR` (#2E86C1) for primary signals and distinct green/red for positive/negative controls.
- **Typography and Scale**:
    - **Labels**: `22pt` (e.g. `ax.set_xlabel(..., fontsize=22)`).
    - **Ticks**: `18pt` (e.g. `plt.rcParams['xtick.labelsize'] = 18`).
- **Mathematical Notation**:
    - **Natural Log**: Use natural logarithms (`np.log`) universally. Label as $\log(\cdot)$ or $\log T$.
    - **Explicit Math**: Define residuals explicitly using LaTeX math in the label (e.g. $\log T - \log \text{med}_{ply}$).
    - **Naming**: Spell out labels like "Clock Time" in full. Use $T$ or "Move Time" for the dependent variable $y$.

## Role of the code

`human_analytics/` mixes **small runnable scripts**, **importable analysis utilities**, and **Jupyter notebooks**. Domain logic and operational safety stay visible rather than hidden behind abstractions.

## Language and typing

- **Python 3** with modern union syntax (`str | None`).
- **Type hints** are used for public helpers; omitted for small analysis lambdas.

## Naming

- **Functions and variables:** `snake_case`.
- **Module-level tuning constants:** `SCREAMING_SNAKE` (e.g. `MAIN_COLOR`, `FONT_SIZE_LABEL`).
- **Files:** Prefer short, descriptive names (e.g. `clocktime_movetime.py` over `clock_move_analysis.py`).

## Imports

1. **`from __future__ import annotations`** if needed.
2. **Standard library** (alphabetical).
3. **Blank line.**
4. **Third-party** (alphabetical by package: `chess`, `dask`, `duckdb`, `matplotlib`, `numpy`, `pandas`, `seaborn`, `statsmodels`, `tqdm`).
5. **Blank line.**
6. **Local project** (`from utils import …`, `from _bootstrap import …` in `slurm/scripts/`).

## Formatting and structure

- **Indentation:** 4 spaces.
- **Control flow:** Early guard returns; `if __name__ == "__main__":` entry points for scripts.

## Summary

The codebase favors **readable, modular analysis code**—clear separation of utilities (`utils/`) and scripts, consistent import layout, and explicit domain naming. **Visual excellence** is a primary requirement for all generated analysis figures.
