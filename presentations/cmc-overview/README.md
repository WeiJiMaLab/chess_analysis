# ♟️ Resource Rational Meta-Control (Slidev)

High-fidelity presentation deck for the **Chess Meta-Control (CMC)** project. This deck synthesizes the motivation, mechanistic methods, and empirical validation of learned search stopping rules.

## 🚀 Quick Start

```bash
# 1. Install dependencies
npm install

# 2. Fix figure symlinks (if needed)
mkdir -p public
ln -sf ../../../src/figures public/figures

# 3. Launch interactive dev server
npm run dev
```

- **Dev Mode**: `npm run dev` (Hot-reloading at http://localhost:5173)
- **Build**: `npm run build` (Static export to `dist/`)
- **Export**: `npm run export` (Generates PDF)

---

## 🏗️ Presentation Structure

The presentation is organized into three core narrative arcs:

### **Part 1: The Problem (Motivation)**
Focuses on why fixed-budget search is computationally wasteful.
- **The Forced Move Problem**: Highlighting the inefficiency of standard engines.
- **The Stopping Problem**: Formalizing the trade-off between move quality and computational cost.

### **Part 2: The Methods (Mechanics)**
Technical deep-dive into the CMC architecture and training pipeline.
- **Control Flow**: How the Meta-Controller sits atop the MCTS loop.
- **Bidirectional GNN**: Visualizing the "Heartbeat" sweeps (Attention Up / MLP Down).
- **The Halt Controller**: Mechanics of the policy gradient optimization against a DP Oracle.
- **Economy of Thought**: Defining the optimal inflection point $k^*$ on the "Thinking Curve."

### **Part 3: Empirical Results (Behavioral Validation)**
Validation using human behavioral data and engine-derived VOC.
- **Clock Elasticity**: Resolving the "Ply Paradox" to show how humans adapt thinking budgets to available time.
- **Response Time Arc**: Tracking cognitive demand across game stages (Opening vs. Middle vs. Endgame).
- **VOC Demand**: Correlating human deliberation time with the "Value of Computation."

---

## 🎨 Design & Components

This deck uses a custom **premium design system** defined in `style.css` with a focus on dark-mode aesthetics and typography (Inter & Space Grotesk).

### **Custom Interactive Components** (`/components`)
- `<ChessBackground />`: Animated Three.js background with floating chess pieces.
- `<GnnTwoSweeps />`: Interactive diagram showing the bidirectional GNN message passing.
- `<PolicyPretrainDiagram />`: Tree-based visualization of the DP Oracle logic.
- `<MetaControllerZoom />`: Mechanistic readout of the root hidden state.

### **Formatting Rules** (Slidev Gotchas)
Do **not** insert a blank line between the slide divider `---` and the frontmatter block.
```md
---
layout: two-cols
---
# Correct
```

---

## 📊 Figures & Data Integration

Figures are served from the core `chess_analysis/src/figures` directory via a symlink in `public/figures`. 

To update the figures, run the analysis scripts from the repository root (after `selected_moves` is loaded, e.g. via `src/slurm/script_load_moves.sh` and `load_data.py`).

**Default** (`combined.png`) and **nonzero_T** (`combined_nonzero_T.png`, premoves / `move_time = 0` removed) for the main timing / branching scripts:
```bash
python src/move_time_summary.py
python src/move_time_summary.py --nonzero_T
python src/ply_movetime.py
python src/ply_movetime.py --nonzero_T
python src/clock_movetime.py
python src/clock_movetime.py --nonzero_T
python src/clock_movetime.py --opp
python src/clock_movetime.py --nonzero_T --opp
python src/npossiblemoves_movetime.py
python src/npossiblemoves_movetime.py --nonzero_T
python src/voc_movetime.py
```

The deck reads PNGs from the symlinked `public/figures` → `src/figures` (e.g. `combined.png` and `combined_nonzero_T.png` under `move_time_summary/`, `ply_movetime/`, and `npossiblemoves_movetime/`; under `clock_movetime/` add `combined_opp.png` and `combined_nonzero_T_opp.png` for the opponent-clock slides; plus `/figures/voc_movetime/standard_voc_single_sample.png`).
