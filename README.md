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

2. **Stockfish**: Use a Stockfish executable that works on your system (e.g. built from [official-stockfish/Stockfish](https://github.com/official-stockfish/Stockfish) with `make build ARCH=x86-64-avx2`). Set the path in the notebook or in `utils.STOCKFISH_PATH`.

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
