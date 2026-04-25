import chess
import dask.dataframe as dd
import numpy as np

def score_to_ev(score, pov): 
    score_obj = score.pov(pov).wdl()
    return (score_obj.wins + 0.5 * score_obj.draws) / score_obj.total()

def ev_board(engine, board, depth = 14): 
    score = engine.analyse(board, chess.engine.Limit(depth=depth))["score"]
    return score_to_ev(score, board.turn)

def cp_board(engine, board, depth = 14): 
    score = engine.analyse(board, chess.engine.Limit(depth=depth))["score"]
    return score.pov(board.turn).cp

def ev_after_move(engine, board, move, depth = 14):
    board_after = board.copy()
    board_after.push(move)
    score = engine.analyse(board_after, chess.engine.Limit(depth=depth))["score"]
    return score_to_ev(score, board.turn)


#####
def best_two_diff(engine, board, depth = 14): 
    candidates = engine.analyse(board, chess.engine.Limit(depth=depth), multipv = 2)
    if len(candidates) < 2: return 0
    v1, v2 = score_to_ev(candidates[0]["score"], board.turn), score_to_ev(candidates[1]["score"], board.turn)
    return np.abs(v1 - v2)

def voc_consideration_set(engine, board, n_candidates = 5):
    # Vdeep(a_deep | a_deep \in A) - Vdeep(a_shallow | a_shallow \in A)
    shallow_candidates = engine.analyse(board, chess.engine.Limit(depth=1), multipv = n_candidates)
    shallow_moves = [c["pv"][0] for c in shallow_candidates]
    best_shallow_move = shallow_moves[0]
    scores = {move: ev_after_move(engine, board, move) for move in set(shallow_moves)}

    # voc = difference between value of best (deep) move and value of shallow move
    voc = max(scores.values()) - scores[best_shallow_move]
    return voc

def voc_single_sample(engine, board):
    # V_deep(a_deep) - V_deep(a_shallow)
    best_shallow_move = engine.analyse(board, chess.engine.Limit(depth=1))["pv"][0]
    best_deep_move = engine.analyse(board, chess.engine.Limit(depth=14))["pv"][0]

    if best_shallow_move == best_deep_move: 
        return 0
    else: 
        return ev_after_move(engine, board, best_deep_move) - ev_after_move(engine, board, best_shallow_move)

def row_to_fen(row): 
    """FEN for engines: includes board, turn, castling rights, and en passant targets."""
    turn = 'w' if row['player_white'] else 'b'
    
    # Handle optional columns (Dask/Pandas Series or dict)
    try:
        cr = row['castling_rights'] if row['castling_rights'] is not None else '-'
    except (KeyError, ValueError, TypeError):
        cr = '-'
        
    try:
        ep = row['en_passant_targets'] if row['en_passant_targets'] is not None else '-'
    except (KeyError, ValueError, TypeError):
        ep = '-'
        
    return f"{row['board_position']} {turn} {cr} {ep}"

# Preprocess sample games to add time_left and time_left_after columns
def preprocess_data(df: dd.DataFrame) -> dd.DataFrame:
    df = df.sort_values(["gid", "move_ply"], kind="mergesort").reset_index(drop=True)
    df = df.set_index("gid") #most of our work is done on gids

    # clean up move time
    df["move_time"] = df["move_time"].fillna(0).clip(lower=0) 

    # group by gid and player number to get player-wise move time left
    grouped = df.groupby(["gid", "player_white"], sort=False)
    df["spent_prior"] = grouped["move_time"].cumsum() - df["move_time"]
    df["n_prior"] = grouped.cumcount()
    inc = df["clock_increment"]
    df["player_time_left"] = (df["initial_clock"] - df["spent_prior"] + df["n_prior"] * inc).clip(lower=0) 

    # append active player to fen
    df["fen"] = df.apply(row_to_fen, axis=1, meta=("object", "object"))
    df["move_number"] = (df["move_ply"] + 1) // 2
    df["total_time_left"] = df.groupby(["gid", "move_number"])["player_time_left"].transform("sum", meta=("total_time_left", "float64"))

    # Shifts within groups
    df["move_time(t-1)"] = df.groupby("gid")["move_time"].shift(1, meta=("move_time", "float64"))
    df["move_time(t-2)"] = df.groupby("gid")["move_time"].shift(2, meta=("move_time", "float64"))
    df["fen(t-1)"] = df.groupby("gid")["fen"].shift(1, meta=("fen", "object"))
    return df


