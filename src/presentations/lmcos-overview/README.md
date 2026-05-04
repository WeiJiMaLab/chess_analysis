# ♟️ Resource Rational Meta-Control (Slidev)

High-fidelity presentation deck for the **Chess Meta-Control (CMC)** project. This deck synthesizes the motivation, mechanistic methods, and empirical validation of learned search stopping rules using human timing data.

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

### **Part 3: Behavioral Validation (Human Data)**
Human move timing as an existence proof and target distribution for efficient search (histograms, ply arc, clock pressure, branching, material, quantile heatmaps from the shared `Analyzer` path).

---

## 🎨 Design & Components

This deck uses a custom **premium design system** defined in `style.css` with a focus on dark-mode aesthetics and typography (Inter & Space Grotesk).

### **Custom Interactive Components** (`components/`)
Used on-slide (see `slides.md`): `<ChessBackground />`, `<PaperSketch />`, `<LeelaSearchLoop />`, `<MetaControllerZoom />`, `<GnnTwoSweeps />`, `<GnnPretrainDiagram />`, `<ChildWdlDiagram />`, `<PolicyPretrainDiagram />`, `<DpOracleDiagram />`.

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

To update the figures, run the analysis scripts from the repository root after `preprocess.py merge` has built `processed_moves` / `processed_moves_nonzero` (see `src/slurm/preprocess.sh` and `utils/selected_db.py`).

**Move-time histograms** (both variants): deliberation-only vs including premoves:
```bash
python src/move_time_summary.py
python src/move_time_summary.py --include_zeroT
```

**Dashboards** (`movetime_analysis.py`; default table `processed_moves_nonzero` → one PNG per analysis):
```bash
python src/movetime_analysis.py --only clock clock_opp npossiblemoves pieces_exc self_pieces_exc ply
```

The symlink `public/figures` → `../../../src/figures` (from this directory) serves `move_time_summary/{combined.png,combined_include_zeroT.png}`, `ply_movetime/combined.png`, `clock_movetime/{combined.png,combined_opp.png,...}`, `npossiblemoves_movetime/`, `n_pieces_exc_pawns_movetime/`, `n_self_pieces_exc_pawns_movetime/`, plus companion `*_quantile_heatmap.png` files where `movetime_analysis` requests them. Literature PNG/SVG under `figures/lit/` are checked into `src/figures` with the analysis repo.
