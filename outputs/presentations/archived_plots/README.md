# archived_plots

Plots recovered from git history to keep `src/tree-search.md` rendering while the
figures are regenerated.

- **Source commit:** `8cb5fc0` (parent of `67ba327` "clean slate: remove all plots"),
  original path `figures/lmcos_tiny/*.png`.
- **Status: STALE.** These were built on the lossy depth-4 trees (the reason `67ba327`
  removed them). They are placeholders only — regenerate via
  `src/analysis/make_rt_figures.py` and drop the fresh PNGs into
  `public/figures/normative/`, then repoint the deck's `<img src>` paths back.

The deck (`src/tree-search.md`) currently points its `<img>` tags at this folder
(`../archived_plots/…`). Ten of the 18 archived plots are referenced; the rest are
kept for reference during reconstruction.
