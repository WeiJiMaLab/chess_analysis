# Chess analysis

Analysis of chess games and positions using DuckDB, the python-chess library, and Stockfish.

## Requirements

- Python 3 (tested with 3.11)
- [DuckDB](https://duckdb.org/)
- [python-chess](https://python-chess.readthedocs.io/)
- [pandas](https://pandas.pydata.org/)
- A **Stockfish** binary (UCI engine) for position evaluation

Optional: Jupyter for running `test_db.ipynb`, and matplotlib for plots.

## Setup

1. Create a virtual environment and install dependencies:

   ```bash
   cd chess_analysis
   python -m venv .venv
   source .venv/bin/activate   # or .venv\Scripts\activate on Windows
   pip install duckdb pandas chess matplotlib jupyter
   ```

2. **Stockfish** (defaults in `utils.py`, under your home directory on Della):
   - **SF14**: `~/stockfish/src/stockfish` (SF14 tree, portable NNUE bundled in the binary).
   - **SF15**: `~/stockfish-sf_15/src/stockfish` with NNUE `nn-6877cd24400e.nnue` in the same `src/` directory (clone `sf_15` from [official-stockfish/Stockfish](https://github.com/official-stockfish/Stockfish), then `cd src && make -j build ARCH=x86-64-avx2`).
   - Override by passing `path=` to `get_stockfish_engine()` or by editing `STOCKFISH_SF14_PATH` / `STOCKFISH_SF15_PATH` in `utils.py`.

3. **Lichess database** (optional): The notebook can attach a read-only DuckDB database (e.g. `lichess.db`). Set the path in the notebook when attaching.

## Project layout

- `test_db.ipynb` – Exploratory queries and analysis (games, endgame positions, FEN display, Stockfish evaluation).
- `utils.py` – Shared helpers: `display_fen`, DB connection, Stockfish engine, evaluation.
- `personal.db` – Local DuckDB database (created on first run; ignored by git).

## Usage

Run the notebook from the project root so that `import utils` works:

```bash
cd chess_analysis
jupyter notebook test_db.ipynb
```

Or in Jupyter/Lab, ensure the working directory is `chess_analysis` when running the notebook.

## License

Use and adapt as you like.
