#%%
import pandas as pd
import chess
from chess.engine import EngineTerminatedError
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from utils import get_db_connection, get_stockfish_engine

# --- Database setup ---
conn = get_db_connection(threads=10)
start_date = "2023-12-01"
end_date = "2023-12-02"

db_path = '/scratch/gpfs/GRIFFITHS/chess-db/lichess.db'
try:
    conn.execute(f"ATTACH '{db_path}' AS core (READ_ONLY)")
except Exception as e:
    print(f"Warning attaching database: {e}")

#%%
# Sample 1000 random midgame positions from rapid/classical games between strong players.
# Random ordering via RANDOM() ensures we don't accidentally oversample any particular
# opening or time period.
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
    ORDER BY RANDOM()
    LIMIT 2000
""").df()

#%%
SHALLOW_DEPTH = 1   # proxy for "no computation" — what you'd play immediately
DEEP_DEPTH = 15     # proxy for "full computation" — Russek's deep search depth
TIME_LIMIT = 10.0   # hard ceiling per analysis call to prevent engine hangs

engine = get_stockfish_engine()
engine.configure({"Skill Level": 5}) 
data = []

for i in tqdm(range(500)):
    position = positions.iloc[i]
    board = chess.Board(position.board_position)
    ply = board.ply()  # needed for WDL calibration — win prob varies with game phase

    try:
        # --- Step 1: Stream depth-1 through depth-15 ---
        # We collect every move that was ever the best move at any depth.
        # This gives us the candidate set for re-evaluation.
        # We also record the depth-1 best move specifically as our "shallow" move.
        candidate_moves = set()
        shallow_move = None

        # Clear hash BEFORE streaming so each position starts fresh.
        # This prevents contamination from previous positions.
        engine.configure({"Clear Hash": True})
        with engine.analysis(
            board,
            chess.engine.Limit(depth=DEEP_DEPTH, time=TIME_LIMIT)
        ) as analysis:
            for info in analysis:
                pv = info.get("pv")
                if not pv:
                    continue
                depth = info.get("depth")
                candidate_moves.add(pv[0])
                if depth == SHALLOW_DEPTH:
                    shallow_move = pv[0]

        if shallow_move is None or len(candidate_moves) == 0:
            continue

        # --- Step 2: Re-evaluate all candidates at depth-15 ---
        # Do NOT clear hash here — we want to reuse the search tree
        # from Step 1, which already explored these lines.
        # multipv lets us score all candidates in one pass.
        infos = engine.analyse(
            board,
            chess.engine.Limit(depth=DEEP_DEPTH, time=TIME_LIMIT),
            multipv=len(candidate_moves),
            root_moves=list(candidate_moves)
        )

        # --- Step 3: Convert scores to win probability via WDL ---
        # wdl(ply=ply) uses Stockfish's built-in calibration which accounts
        # for game phase — early positions have more uncertain outcomes than
        # late positions with the same centipawn score.
        # We always work from the active player's perspective so VOC is
        # always a gain (never negative by construction).
        scores = {}
        for info in infos:
            pv = info.get("pv")
            if not pv:
                continue
            score = info["score"]
            # Use active player's perspective so VOC = improvement for the player to move
            wdl = score.white().wdl(ply=ply) if board.turn == chess.WHITE \
                  else score.black().wdl(ply=ply)
            scores[pv[0]] = wdl.wins / 1000.0

        if shallow_move not in scores or not scores:
            continue

        # --- Step 4: Compute VOC ---
        # v_shallow: how good is the depth-1 move when evaluated deeply?
        # v_deep: how good is the best move found by deep search?
        # VOC = the gain from thinking deeper. Should always be >= 0.
        v_shallow = scores[shallow_move]
        v_deep = max(scores.values())
        voc = max(v_deep - v_shallow, 0.0)  # clip tiny negative values from numerical noise

        data.append(pd.Series({
            "board_position": position.board_position,
            "player_white":   position.player_white,
            "score_shallow":  v_shallow,
            "score_deep":     v_deep,
            "voc":            voc,
            "move_time":      position.move_time,
            "elo":            position.white_elo if position.player_white else position.black_elo,
            "n_candidates":   len(candidate_moves),  # useful diagnostic
            "ply":            ply
        }))

    except EngineTerminatedError:
        print(f"Engine crashed on position {i}: {position.board_position}")
        try:
            engine.quit()
        except Exception:
            pass
        engine = get_stockfish_engine()

    except Exception as e:
        print(f"Skipping position {i}: {e}")
        continue

engine.quit()

df = pd.DataFrame(data)
print(f"\nComputed VOC for {len(df)} positions")
print(df[["voc", "move_time", "elo", "n_candidates"]].describe())

import statsmodels.formula.api as smf

# R-squared on full data
m_full = smf.ols("move_time ~ voc", data=df).fit()
print(f"Full data R²: {m_full.rsquared:.4f}")
print(f"VOC coef: {m_full.params['voc']:.4f}, p={m_full.pvalues['voc']:.4f}")

# R-squared with sqrt VOC (Russek's preferred form)
df["voc_sqrt"] = np.sqrt(df["voc"])
m_sqrt = smf.ols("move_time ~ voc_sqrt", data=df).fit()
print(f"\nSqrt VOC R²: {m_sqrt.rsquared:.4f}")
print(f"VOC_sqrt coef: {m_sqrt.params['voc_sqrt']:.4f}, p={m_sqrt.pvalues['voc_sqrt']:.4f}")

# R-squared on nonzero VOC only
df_nonzero = df[df["voc"] > 0].copy()
df_nonzero["voc_sqrt"] = np.sqrt(df_nonzero["voc"])
m_nonzero = smf.ols("move_time ~ voc_sqrt", data=df_nonzero).fit()
print(f"\nNonzero only R²: {m_nonzero.rsquared:.4f} (n={len(df_nonzero)})")
print(f"VOC_sqrt coef: {m_nonzero.params['voc_sqrt']:.4f}, p={m_nonzero.pvalues['voc_sqrt']:.4f}")

# Delta AIC between linear and sqrt
print(f"\nDelta AIC (linear - sqrt): {m_full.aic - m_sqrt.aic:.2f}")


#%% Plot sqrt(VOC) vs Move Time with regression stats
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

# Color scheme
scatter_color = "blue"
line_color = "red"

# --- Subplot 1: All positions ---
sns.scatterplot(x="voc_sqrt", y="move_time", data=df, alpha=0.6, color=scatter_color, ax=axes[0])
sns.regplot(x="voc_sqrt", y="move_time", data=df, scatter=False, color=line_color, ax=axes[0])

# Regression stats
r2_full = m_sqrt.rsquared
coef_full = m_sqrt.params["voc_sqrt"]
axes[0].text(0.05, 0.95, f"R² = {r2_full:.3f}\nCoef = {coef_full:.3f}", 
             transform=axes[0].transAxes, fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
axes[0].set_xlabel("sqrt(VOC)")
axes[0].set_ylabel("Move time (s)")
axes[0].set_title("All positions")

# --- Subplot 2: Nonzero VOC only ---
sns.scatterplot(x="voc_sqrt", y="move_time", data=df_nonzero, alpha=0.6, color=scatter_color, ax=axes[1])
sns.regplot(x="voc_sqrt", y="move_time", data=df_nonzero, scatter=False, color=line_color, ax=axes[1])

# Regression stats
r2_nonzero = m_nonzero.rsquared
coef_nonzero = m_nonzero.params["voc_sqrt"]
axes[1].text(0.05, 0.95, f"R² = {r2_nonzero:.3f}\nCoef = {coef_nonzero:.3f}", 
             transform=axes[1].transAxes, fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
axes[1].set_xlabel("sqrt(VOC)")
axes[1].set_title("Nonzero VOC only")

plt.suptitle("Move time vs sqrt(VOC) with regression stats", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("voc_sqrt_vs_move_time.png", dpi=300)
plt.close()

print("Plot saved: voc_sqrt_vs_move_time.png")


