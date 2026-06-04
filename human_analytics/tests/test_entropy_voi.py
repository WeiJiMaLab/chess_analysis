import math
import numpy as np
import chess
import chess.engine

from human_analytics.entropy_voi_analysis import (
    score_to_pwin,
    shannon_entropy,
    softmax,
    compute_stopping_depth
)

def test_score_to_pwin():
    # Monotonicity test
    cp_minus_1000 = chess.engine.PovScore(chess.engine.Cp(-1000), chess.WHITE)
    cp_0 = chess.engine.PovScore(chess.engine.Cp(0), chess.WHITE)
    cp_plus_1000 = chess.engine.PovScore(chess.engine.Cp(1000), chess.WHITE)
    
    p_minus = score_to_pwin(cp_minus_1000, chess.WHITE)
    p_0 = score_to_pwin(cp_0, chess.WHITE)
    p_plus = score_to_pwin(cp_plus_1000, chess.WHITE)
    
    assert p_minus < p_0 < p_plus
    
    # Draw value
    assert math.isclose(p_0, 0.5, abs_tol=1e-5)
    
    # Mate values
    mate_win = chess.engine.PovScore(chess.engine.Mate(1), chess.WHITE)
    mate_loss = chess.engine.PovScore(chess.engine.Mate(-1), chess.WHITE)
    
    assert score_to_pwin(mate_win, chess.WHITE) == 1.0
    assert score_to_pwin(mate_loss, chess.WHITE) == 0.0

def test_shannon_entropy():
    # Uniform distribution
    N = 10
    probs = np.ones(N) / N
    h = shannon_entropy(probs)
    assert math.isclose(h, math.log(N), rel_tol=1e-5)
    
    # Deterministic distribution
    probs_det = np.array([1.0, 0.0, 0.0])
    h_det = shannon_entropy(probs_det)
    assert math.isclose(h_det, 0.0, abs_tol=1e-5)
    
    # Non-negativity
    probs_rand = softmax(np.random.randn(5), beta=1.0)
    assert shannon_entropy(probs_rand) >= 0.0

def test_compute_stopping_depth():
    max_depth = 8
    n_moves = 5
    
    # A perfectly uniform q_matrix where all evaluations are the same
    # Entropy should not change over depths (IG = 0)
    # The stopping rule H(d-1) - H(d) < theta should trigger immediately at d=1
    q_matrix = np.zeros((max_depth, n_moves))
    
    d_star_B = compute_stopping_depth(q_matrix, approach="B", beta=1.0, theta=0.01)
    d_star_A = compute_stopping_depth(q_matrix, approach="A", beta=1.0, theta=0.01)
    
    assert 1 <= d_star_B <= max_depth
    assert 1 <= d_star_A <= max_depth
    assert d_star_B == 1
    
    # Test approach A with missing values (NaNs)
    q_matrix_A = np.full((max_depth, n_moves), np.nan)
    # Depth 0 (i.e. depth=1)
    q_matrix_A[0, :2] = 0.5
    # Depth 1 (i.e. depth=2)
    q_matrix_A[1, :2] = [0.6, 0.4]
    
    d_star_A_missing = compute_stopping_depth(q_matrix_A, approach="A", beta=1.0, theta=0.001)
    assert 1 <= d_star_A_missing <= max_depth
