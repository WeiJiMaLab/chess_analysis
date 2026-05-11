import pytest
import pandas as pd
from pathlib import Path
from src.metacontrol.data.sampler import ChessSampler

@pytest.fixture(scope="module")
def sampler():
    return ChessSampler(memory_limit="40GB", threads=8)

@pytest.fixture(scope="module")
def sampled_data(sampler):
    # Expanded smoke test - 1 day, 100 games
    return sampler.sample_positions(
        year=2023,
        start_date="2023-01-01",
        end_date="2023-01-02",
        n_games=100,
        n_final_fens=50,
        seed=42
    )

def test_elo_bounds(sampled_data):
    assert (sampled_data["white_elo"] >= 1800).all()
    assert (sampled_data["white_elo"] <= 2600).all()
    assert (sampled_data["black_elo"] >= 1800).all()
    assert (sampled_data["black_elo"] <= 2600).all()

def test_ply_bounds(sampled_data):
    assert (sampled_data["move_ply"] >= 8).all()
    assert (sampled_data["move_ply"] <= 120).all()

def test_legal_moves_bounds(sampled_data):
    assert (sampled_data["n_possible_moves"] >= 2).all()
    assert (sampled_data["n_possible_moves"] <= 60).all()

def test_piece_count_bounds(sampled_data):
    assert (sampled_data["n_pieces"] >= 8).all()
    assert (sampled_data["n_pieces"] <= 32).all()

def test_one_position_per_game(sampled_data):
    # Check that each gid is unique in the sample
    assert sampled_data["gid"].is_unique

def test_output_format(sampled_data):
    expected_columns = {
        "full_fen", "gid", "move_ply", "n_pieces", "n_possible_moves",
        "utc_datetime", "time_control_type", "opening", "white_elo", "black_elo"
    }
    assert expected_columns.issubset(set(sampled_data.columns))

def test_export_example(sampled_data):
    if sampled_data.empty:
        pytest.skip("No data sampled")
    out_dir = Path("/scratch/gpfs/GRIFFITHS/hl4291/data")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "metacontrol_example.csv"
    
    # Just export one row as an example
    example = sampled_data.head(1)
    example.to_csv(out_path, index=False)
    assert out_path.exists()
    print(f"\nExported example to {out_path}")
