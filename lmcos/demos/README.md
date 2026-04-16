## lmcos demos

This directory contains small, self-contained scripts intended to make the `lmcos` components easier to understand one-by-one.

### `demo_partial_tree_generation.py`

Builds a small **partial tree** from a root FEN using:
- provider-backed lc0 queries for node features/expansions
- PUCT-driven leaf selection/expansion
- a sampled node-budget cap

Defaults assume shared assets live at:
- `chess_analysis/engines/lc0-src/build/release/lc0`
- `chess_analysis/weights/t1-256x10-distilled-swa-2432500.pb.gz`

Example (from `chess_analysis/lmcos/`):

```bash
python demos/demo_partial_tree_generation.py \
  --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" \
  --max-depth 3 \
  --min-nodes 2 \
  --max-nodes 3 \
  --out-svg demos/out.svg
```

