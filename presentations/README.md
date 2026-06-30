# ♟️ Chess Meta-Control — Presentations (Slidev)

Multiple Slidev decks for the **Chess Meta-Control (CMC)** project, each runnable
on its own. Every deck is a flat entry file under **`src/<deck>.md`**:

| Deck | Entry | Topic |
| --- | --- | --- |
| `main` | `src/main.md` | Learned meta-control of tree search — motivation, methods, human-timing validation (was `lmcos-overview`) |
| `tree-search` | `src/tree-search.md` | Tree search & human deliberation — the linear story (mermaid map, PUCT/BFS/MCTS animation, lc0→SF, satisficing) |
| `modelrecovery` | `src/modelrecovery.md` | Parameter recovery / identifiability *(stub)* |
| `modelcomparison` | `src/modelcomparison.md` | Cross-model fit & selection *(stub)* |
| `gazeanalysis` | `src/gazeanalysis.md` | Eye-movement / deliberation-window analyses *(stub)* |

The intent is didactic: page through focused, per-topic decks slide-by-slide
rather than reading one long `.md`. More decks will be added over time. This
mirrors the multi-deck layout in the sibling `monkey_4iar/presentations`.

## Run

```bash
cd presentations
npm install
npm run dev                 # runs the default deck (main)
npm run dev modelrecovery   # run a specific deck by name
```

Build or export a specific deck:

```bash
npm run build modelcomparison   # static export to dist/
npm run export gazeanalysis     # PDF
```

The deck name is the first argument; it defaults to `main` when omitted. An
unknown name prints the list of available decks. Extra Slidev flags pass through
after the deck name, e.g. `npm run dev modelrecovery -- --port 3030`. A leading
`--` separator is tolerated (`npm run dev -- modelrecovery`). `dev` opens the
browser automatically.

`scripts/run-deck.js` maps `<deck>` to `src/<deck>.md`, lists the decks it finds
in `src/`, and forwards remaining flags to the Slidev CLI.

## Shared resources (addon, components, CSS, figures)

All decks share one set of resources. The Slidev project root is `src/`, and
`src/` contains symlinks back to the shared assets at the presentations root, so
every deck resolves them identically:

| In a deck | Resolves to |
| --- | --- |
| `@/shared/slidev-addon-base` (frontmatter addon) | `src/shared` → `../shared` |
| auto-loaded custom components | `shared/slidev-addon-base/components/*.vue` |
| `css: ./style.css` | `src/style.css` → `../style.css` |
| `/figures/...` (absolute image `src`) | `src/public` → `../public`, then `public/figures` → `../../lmcos_small/human/figures` |

The shared addon `shared/slidev-addon-base/` provides:

- **`components/`** — the custom interactive Vue diagrams, auto-registered for
  every deck: `<ChessBackground />`, `<PaperSketch />`, `<LeelaSearchLoop />`,
  `<MetaControllerZoom />`, `<DpOracleDiagram />`, `<GnnTwoSweeps />`,
  `<GnnPretrainDiagram />`, `<ChildWdlDiagram />`, `<PolicyPretrainDiagram />`.
- **`global-bottom.vue`** — page-number footer shown on every slide.
- **`vite.config.ts`** — widens `server.fs.allow` so the dev server can serve
  the `lmcos_small/human/figures` directory (outside `src/`).

So components, styling (`style.css`), and figures are edited in one place and
used by all decks.

### Add a new deck

1. Create `src/<deck>.md` with the same frontmatter as the stubs
   (`theme: default`, `addons: ["@/shared/slidev-addon-base"]`, `css: ./style.css`).
2. Run it: `npm run dev <deck>`.

No other wiring is needed; `run-deck.js` discovers `src/*.md` automatically.

### Formatting rule (Slidev gotcha)

Do **not** insert a blank line between the slide divider `---` and the
frontmatter block:

```md
---
layout: two-cols
---
# Correct
```

## Figures & data integration

Figures are served from the core `chess_analysis/lmcos_small/human/figures`
directory via the symlink chain `src/public/figures → ../public/figures →
../../lmcos_small/human/figures`. Decks reference them with absolute `/figures/...`
URLs (e.g. `/figures/log_movetime_histogram.png`).

To update the figures, run the analysis scripts from the repository root after
`preprocess.sh` has run `merge` and `process_moves` (building `processed_moves` /
`processed_moves_nonzero`; see `utils/selected_db.py`).

**Move-time histograms** (both variants): deliberation-only vs including premoves:

```bash
python lmcos_small/human/move_time_summary.py
python lmcos_small/human/move_time_summary.py --include_zeroT
```

**Dashboards** (`movetime_analysis.py`; default table `processed_moves_nonzero` → one PNG per analysis):

```bash
python analysis/movetime_analysis.py --only clock clock_opp legal_moves pieces_exc own_material ply
```

## Deployment

`netlify.toml` / `vercel.json` build the **default** deck (`npm run build` →
`dist/`). To deploy a specific deck instead, set the build command to
`npm run build <deck>`.
