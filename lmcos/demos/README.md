## lmcos demos

This directory contains small, self-contained scripts intended to make the `lmcos` components easier to understand one-by-one.

### `demo_partial_tree_generation.py`

Builds a small **partial tree** from a root FEN using:
- provider-backed lc0 queries for node features/expansions
- PUCT-driven leaf selection/expansion
- a sampled node-budget cap

Didactic intuition (matching how this demo works):
- Think of the provider as a wrapper around lc0 that gives a heuristic state evaluation plus expansion candidates.
- The raw node `value` is a heuristic `v(s)` signal at each visited state.
- The iterative teacher-style loop then uses PUCT to decide where to search next.
- As it runs, it backs up path statistics into edge/action values (`Q(s,a)`), so search decisions become action-aware rather than only state-heuristic.
- In this demo visualization, node labels show the raw `v(s)` values; backed-up `Q(s,a)` values are used internally for selection.

Value semantics in this demo:
- Node `value` shown in the SVG is the **raw lc0 state-evaluation feature** at that node.
- During the iterative PUCT loop, the code also maintains backed-up **edge/action values** (`Q(s,a)`) in search stats.
- Those backed-up `Q` values are used for selection, but are not currently rendered in node labels.

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

