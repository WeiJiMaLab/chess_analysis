import pandas as pd
import chess
from chess.engine import EngineTerminatedError
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from utils import get_db_connection, get_stockfish_engine
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
import os

SHALLOW_DEPTH = 1   # proxy for "no computation" — what you'd play immediately
DEEP_DEPTH = 15     # proxy for "full computation" — Russek at depth 15
TIME_LIMIT = 5.0   # hard ceiling per analysis call to prevent engine hangs

def score_to_wp(score, board_after):
    """Convert score to win probability from the original player's perspective.
    After pushing a move, board_after.turn is the opponent.
    So original player's turn = not board_after.turn.
    """
    try:
        wdl = score.white().wdl(ply=board_after.ply())
        wp_white = wdl.wins / 1000.0
        return wp_white if not board_after.turn == chess.WHITE else 1.0 - wp_white
    except:
        return None

def compute_voc(board, engine, candidate_moves, deep_depth, time_limit):
    """
    Compute VOC by evaluating each candidate move's resulting position independently.
    Matches Russek et al.'s approach: push move, evaluate resulting position at depth-15,
    negate score to get value from original player's perspective.
    """
    scores = {}
    for move in candidate_moves:
        # push move to get resulting position
        board_after = board.copy()
        board_after.push(move)

        # skip if resulting position is stalemate (score is 0 by definition)
        if board_after.is_stalemate():
            scores[move] = 0.5
            continue

        # clear hash before each candidate — independent evaluation, no interference
        engine.configure({"Clear Hash": True})
        try: info = engine.analyse(board_after, chess.engine.Limit(depth=deep_depth, time=time_limit))
        except Exception: continue
        wp = score_to_wp(info["score"], board_after)
        if wp is not None: scores[move] = wp
    return scores



def analyze_position(position):
    engine = None
    try:
        engine = get_stockfish_engine(version=14)
        board = chess.Board(position.board_position)
        # 1. Get top 5 moves at depth 1 (consideration set)
        info_shallow = engine.analyse(board, chess.engine.Limit(depth=SHALLOW_DEPTH), multipv=5)
        candidate_moves = [entry["pv"][0] for entry in info_shallow]
        shallow_move = candidate_moves[0]  # The move played at depth 1 (no computation/baseline)

        # 2. Evaluate all 5 candidates at depth 15 (deep search)
        scores = compute_voc(board, engine, candidate_moves, DEEP_DEPTH, TIME_LIMIT)
        
        if shallow_move not in scores or len(scores) == 0:
            return None

        # 3. VOC = max(deep evaluation of top 5 depth-1 moves) - deep evaluation of move 1
        voc = max(scores.values()) - scores[shallow_move]
        if voc is None:
            return None

        return pd.Series({
            "board_position": position.board_position,
            "player_white":   position.player_white,
            "voc":            voc,
            "move_time":      position.move_time,
            "elo":            position.white_elo if position.player_white else position.black_elo,
            "n_candidates":   len(candidate_moves),
            "ply":            board.ply()
        })

    except EngineTerminatedError:
        print(f"Engine crashed on position {position.board_position}")
        return None
    except Exception as e:
        print(f"Skipping position {position.board_position}: {e}")
        return None
    finally:
        if engine is not None:
            try:
                engine.quit()
            except Exception:
                pass

def analyze_data(df):
    print(f"\nComputed VOC for {len(df)} positions")
    print(df[["voc", "move_time", "elo", "n_candidates"]].describe())

    m_full = smf.ols("move_time ~ voc", data=df).fit()
    print(f"Full data R²: {m_full.rsquared:.4f}")
    print(f"VOC coef: {m_full.params['voc']:.4f}, p={m_full.pvalues['voc']:.4f}")

    m_sqrt = smf.ols("move_time ~ voc_sqrt", data=df).fit()
    print(f"\nSqrt VOC R²: {m_sqrt.rsquared:.4f}")
    print(f"VOC_sqrt coef: {m_sqrt.params['voc_sqrt']:.4f}, p={m_sqrt.pvalues['voc_sqrt']:.4f}")

    df_nonzero = df[df["voc"] > 0].copy()
    m_nonzero = smf.ols("move_time ~ voc_sqrt", data=df_nonzero).fit()
    print(f"\nNonzero only R²: {m_nonzero.rsquared:.4f} (n={len(df_nonzero)})")
    print(f"VOC_sqrt coef: {m_nonzero.params['voc_sqrt']:.4f}, p={m_nonzero.pvalues['voc_sqrt']:.4f}")

    print(f"\nDelta AIC (linear - sqrt): {m_full.aic - m_sqrt.aic:.2f}")

    return df

def plot_data(df):
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # --- Plot 1: Russek Figure 1b replication — binned mean move time ---
    df_plot = df.copy()
    df_plot["voc_bin"] = pd.cut(df_plot["voc"], bins=10)
    binned = df_plot.groupby("voc_bin")["move_time"].agg(["mean", "count"])
    axes[0].plot(range(len(binned)), binned["mean"].values, marker="o", color="black")
    axes[0].set_xlabel("VOC bin (equal-width, low → high)")
    axes[0].set_ylabel("Mean move time (s)")
    axes[0].set_title("Russek Fig 1b replication")
    # annotate bin counts so you can see how sparse the tail is
    for j, (mean, count) in enumerate(zip(binned["mean"], binned["count"])):
        if not np.isnan(mean):
            axes[0].annotate(f"n={int(count)}", (j, mean), textcoords="offset points",
                            xytext=(0, 6), ha="center", fontsize=7)


    # --- Plot 2: All positions scatter ---
    m_sqrt = smf.ols("move_time ~ voc_sqrt", data=df).fit()
    sns.scatterplot(x="voc_sqrt", y="move_time", data=df, alpha=0.4, color="blue", ax=axes[1])
    sns.regplot(x="voc_sqrt", y="move_time", data=df, scatter=False, color="red", ax=axes[1])
    axes[1].text(0.05, 0.95, f"R² = {m_sqrt.rsquared:.3f}\nCoef = {m_sqrt.params['voc_sqrt']:.3f}",
                transform=axes[1].transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.7))
    axes[1].set_xlabel("sqrt(VOC)")
    axes[1].set_ylabel("Move time (s)")
    axes[1].set_title("All positions")

    # --- Plot 3: Nonzero VOC only ---
    df_nonzero = df[df["voc"] > 0].copy()
    m_nonzero = smf.ols("move_time ~ voc_sqrt", data=df_nonzero).fit()
    sns.scatterplot(x="voc_sqrt", y="move_time", data=df_nonzero, alpha=0.4, color="blue", ax=axes[2])
    sns.regplot(x="voc_sqrt", y="move_time", data=df_nonzero, scatter=False, color="red", ax=axes[2])
    axes[2].text(0.05, 0.95, f"R² = {m_nonzero.rsquared:.3f}\nCoef = {m_nonzero.params['voc_sqrt']:.3f}\nn={len(df_nonzero)}",
                transform=axes[2].transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.7))
    axes[2].set_xlabel("sqrt(VOC)")
    axes[2].set_title("Nonzero VOC only")

    plt.suptitle(f"VOC replication — SF14, n={len(df)} positions", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("voc_sqrt_vs_move_time.png", dpi=300)
    plt.close()
    print("Plot saved: voc_sqrt_vs_move_time.png")



if __name__ == "__main__":
    POSITIONS_CACHE = "positions_cache.parquet"
    if os.path.exists(POSITIONS_CACHE):
        positions = pd.read_parquet(POSITIONS_CACHE)
    else:
        conn = get_db_connection(threads=32)
        start_date = "2022-01-01"
        end_date = "2022-12-31"
        db_path = '/scratch/gpfs/GRIFFITHS/chess-db/lichess.db'
        try:
            conn.execute(f"ATTACH '{db_path}' AS core (READ_ONLY)")
        except Exception as e:
            print(f"Warning attaching database: {e}")

        positions = conn.sql(f"""
            SELECT m.gid, m.board_position, m.move_time, m.move_ply, m.player_white,
                g.white_elo, g.black_elo, g.initial_clock
            FROM core.moves m
            JOIN core.games g ON m.gid = g.gid
            WHERE g.utc_datetime BETWEEN '{start_date}' AND '{end_date}'
            AND g.initial_clock >= 300
            AND m.move_ply BETWEEN 15 AND 75
            AND m.move_time > 0
            AND g.white_elo >= 2000
            AND g.black_elo >= 2000
            LIMIT 10000
        """).df()
        positions.to_parquet(POSITIONS_CACHE)

    results = Parallel(n_jobs=128, prefer="processes")(
        delayed(analyze_position)(position) for position in tqdm(positions.itertuples(), total=len(positions))
    )
    data = [r for r in results if r is not None]
    df = pd.DataFrame(data)
    df.to_csv("voc_results.csv", index=False)
    
    df["voc_sqrt"] = np.sqrt(df["voc"])
    df = analyze_data(df)
    plot_data(df)
