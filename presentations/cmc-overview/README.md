# CMC overview (Slidev)

Slides synthesize **Chess Meta-Control**: motivation, `lmcos` methods (step-by-step, aligned with `lmcos/demos/`), and human-game validation.

## Slidev: `layout` and `---` (important)

Do **not** insert a **blank line** between a slide divider `---` and the next line when that line starts **slide frontmatter** (`layout:`, `class:`, `layoutClass:`, etc.). Slidev will mis-parse the slide and layouts break.

**Good:**

```md
Previous slide body ends.

---
layout: two-cols
layoutClass: gap-8
---

# Next slide title
```

**Bad** (blank line between the first `---` and `layout:`):

```md
---

layout: two-cols
```

A blank line **after** the previous slide’s content and **before** `---` is fine. See the HTML comment near the top of `slides.md`.

## Run

```bash
cd chess_analysis/presentations/cmc-overview
npm install
npm run dev
```

Optional: `npm run build` (static `dist/`), `npm run export` (PDF).

## Figures must load from `/figures/...`

Vite only reliably serves assets under this package’s **`public/`** directory. This project uses a **symlink**:

`public/figures` → `../../../src/figures` (i.e. `chess_analysis/src/figures`).

If images 404 after clone:

```bash
cd presentations/cmc-overview
mkdir -p public
ln -sf ../../../src/figures public/figures
```

Then regenerate PNGs from **`chess_analysis/`** (with data and venv as needed):

```bash
cd chess_analysis
source .venv/bin/activate
python src/clock_move_analysis.py
python src/ply_analysis.py
python src/voc_analysis.py
```

Slides reference paths like `/figures/clock_move_analysis/move_time_distribution.png`.
