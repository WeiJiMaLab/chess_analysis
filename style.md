# Coding style in `src/` (chess_analysis)

This note describes how Python (and notebook) code in `src/` is written. It is descriptive, not a strict linter profile.

## Design priorities

Code is meant to be optimized for **modularity** and **readability**: small, named pieces with obvious data flow and domain meaning (chess, VOC, timing). Prefer that clarity over **flexibility**—fewer layers of indirection, fewer generic configuration surfaces, and explicit scripts or functions are fine when they make the analysis easier to follow and reuse.

## Role of the code

`src/` mixes **small runnable scripts**, **importable analysis utilities**, and **Jupyter notebooks**. Domain logic and operational safety (engine lifecycle, parallelism, DB paths) stay visible rather than hidden behind abstractions unless there is a clear reuse win.

## Language and typing

- **Python 3** with modern union syntax where used (`str | None` in `utils.get_stockfish_engine`).
- **`from __future__ import annotations`** appears in `utils.py` only; other modules do not use it.
- **Type hints** are used for some public helpers (`get_db_connection`, `preprocess_data`) and omitted for many small analysis functions (`bootstrapped_ci`, `compute_metrics_by_bin`, etc.).

## Naming

- **Functions and variables:** `snake_case`.
- **Module-level tuning constants:** `SCREAMING_SNAKE` (e.g. `SHALLOW_DEPTH`, `STOCKFISH_PATH`, `DEPTH_DEEP` in notebooks).
- **Private-by-convention paths:** leading underscore on module globals meant as implementation detail (`_STOCKFISH_HOME`).

## Imports

Use a single convention everywhere (`.py` files and notebook code cells):

1. **`from __future__ import annotations`** only when needed (currently `utils.py`); must be first, before any other imports.
2. **Standard library** — one block, alphabetical (`import` and `from … import` sorted by module name; `from collections import …` before `import os`).
3. **Blank line.**
4. **Third-party** — one block, alphabetical by top-level package (`chess` / `chess.*`, then `dask`, `duckdb`, `joblib`, `matplotlib`, `numpy`, `pandas`, `seaborn`, `statsmodels`, `tqdm`, etc.). For the same package, `import chess` then related `from chess.engine import …` is fine.
5. **Blank line.**
6. **Local project** — `from features import …`, `from utils import …` (alphabetical by module name: `features` before `utils`). Multi-line `from features import (` lists are sorted alphabetically by imported symbol.
7. **Blank line** (in notebooks).
8. **IPython magics** (`%load_ext`, `%autoreload`) after imports, not mixed into them.

Do not duplicate the same import in a file. Flat local imports (`from utils import …`) assume `src/` or the project layout is on the path.

## Documentation

- **Module docstring:** present on `utils.py` (one short paragraph).
- **Function docstrings:** used for heavier setup (`get_stockfish_engine`, `get_db_connection`, `compute_voc`, `score_to_wp`) with domain context (Russek, Stockfish versions, caller responsibilities).
- **Lighter functions** often rely on a **one-line comment** or no docstring (`features.py`).

## Formatting and structure

- **Indentation:** 4 spaces.
- **Line length:** generally moderate; **long SQL** is kept in triple-quoted / f-string blocks inside `conn.sql(...)`.
- **Spacing in signatures:** mixed habits (e.g. `depth = 14` vs `depth=14` in `features.py`).
- **Control flow:** early `continue` / guard returns in loops; `if __name__ == "__main__":` entry points for scripts.

## Comments

Comments explain **why** and **operational constraints**: hash clearing between engine evaluations, default parallelism and `VOC_N_JOBS`, core-dump limits, in-memory DuckDB defaults, sampling logic in SQL. Paper references (e.g. Russek) appear in docstrings and plot titles.

## Error handling and robustness

- **Scripts** print **warnings** when attaching read-only DBs fails, then continue when possible.
- **Engine usage:** `try` / `finally` with `engine.quit()` in worker-style code; some paths use broad `except Exception` or bare `except` to **skip bad positions** rather than fail the whole batch.
- **Parallel runs:** retry loops that reduce `n_jobs` after worker termination (`TerminatedWorkerError`).

## Data and tooling conventions

- **pandas** for in-memory tables; **Dask** (`dask.dataframe`) for larger parquet pipelines in `features.py` / notebooks.
- **DuckDB** for SQL over attached Lichess DBs; **parquet/CSV** for caches and outputs.
- **joblib** `Parallel` / `delayed` for CPU-bound Stockfish work.
- **statsmodels** formula API for quick OLS summaries; **seaborn** for plotting in VOC replication.

## Notebooks (`timing_analysis.ipynb`, `modelchecking.ipynb`)

- Follow the **Imports** section above in each cell that imports modules.
- **Constants** defined in cells alongside imports where appropriate.
- **Procedural cells:** load parquet → preprocess → `compute()` → ad-hoc analysis; local `def` helpers and lambdas are acceptable when they keep the narrative readable.

## Summary

The codebase favors **readable, modular analysis code**—clear separation of utilities, features, and scripts, consistent import layout, and explicit domain naming. Exhaustive typing and strict exception taxonomies are not required everywhere; **clarity of structure** is the main goal, even when that limits how generic a piece of code is.
