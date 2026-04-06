-- Sample root positions from the Lichess DuckDB tables.
--
-- Expected tables:
--   games(gid, white_title, black_title, white_elo, black_elo, n_moves, time_control_type)
--   moves(gid, move_ply, fen, board_position, castling_rights, en_passant_targets,
--         player_white, n_pieces, n_possible_moves, white_n_games, black_n_games,
--         white_tenure, black_tenure)
--
-- Outputs:
--   sampled_root_fens_balanced.txt
--   sampled_root_fens_natural.txt
--
-- Usage:
--   duckdb /path/to/lichess.duckdb < sql/sample_root_positions.sql
--
-- If your DB is already open in a DuckDB shell, run:
--   .read sql/sample_root_positions.sql

CREATE OR REPLACE TEMP TABLE sampling_params AS
SELECT
    20260327::BIGINT AS hash_seed,
    20::INTEGER AS min_game_halfmoves,
    8::INTEGER AS min_move_ply,
    120::INTEGER AS max_move_ply,
    2::INTEGER AS min_legal_moves,
    60::INTEGER AS max_legal_moves,
    6::INTEGER AS min_pieces,
    30::INTEGER AS max_pieces,
    50::INTEGER AS min_player_games,
    30::INTEGER AS min_player_tenure_days,
    1200::INTEGER AS min_elo,
    2600::INTEGER AS max_elo,
    200::INTEGER AS elo_bucket_width,
    5000::INTEGER AS balanced_per_bucket_cap,
    200000::INTEGER AS natural_total_cap,
    3::INTEGER AS max_examples_per_structure;

CREATE OR REPLACE TEMP TABLE eligible_games AS
SELECT
    g.gid,
    g.time_control_type,
    g.opening,
    g.white_elo,
    g.black_elo,
    (g.white_elo + g.black_elo) / 2.0 AS avg_elo
FROM games AS g
CROSS JOIN sampling_params AS p
WHERE COALESCE(g.white_title, '') <> 'BOT'
  AND COALESCE(g.black_title, '') <> 'BOT'
  AND g.n_moves >= p.min_game_halfmoves
  AND g.time_control_type IN ('Blitz', 'Rapid', 'Classical')
  AND g.white_elo BETWEEN p.min_elo AND p.max_elo
  AND g.black_elo BETWEEN p.min_elo AND p.max_elo;

CREATE OR REPLACE TEMP TABLE candidate_moves AS
SELECT
    m.fen,
    m.gid,
    m.move_ply,
    m.board_position,
    CASE WHEN m.player_white THEN 'w' ELSE 'b' END AS side_to_move,
    COALESCE(NULLIF(m.castling_rights, ''), '-') AS castling_rights_norm,
    COALESCE(NULLIF(m.en_passant_targets, ''), '-') AS en_passant_targets_norm,
    m.n_pieces,
    m.n_possible_moves,
    g.time_control_type,
    g.opening,
    CAST(FLOOR(g.avg_elo / p.elo_bucket_width) * p.elo_bucket_width AS INTEGER) AS elo_bucket,
    CASE
        WHEN m.move_ply <= 20 THEN 'opening'
        WHEN m.n_pieces >= 12 THEN 'middlegame'
        ELSE 'endgame'
    END AS phase,
    hash(m.gid, m.move_ply, p.hash_seed) AS sample_hash
FROM moves AS m
JOIN eligible_games AS g USING (gid)
CROSS JOIN sampling_params AS p
WHERE m.fen IS NOT NULL
  AND m.fen <> ''
  AND m.move_ply BETWEEN p.min_move_ply AND p.max_move_ply
  AND m.n_possible_moves BETWEEN p.min_legal_moves AND p.max_legal_moves
  AND m.n_pieces BETWEEN p.min_pieces AND p.max_pieces
  AND m.white_n_games >= p.min_player_games
  AND m.black_n_games >= p.min_player_games
  AND m.white_tenure >= p.min_player_tenure_days
  AND m.black_tenure >= p.min_player_tenure_days;

CREATE OR REPLACE TEMP TABLE deduped_moves AS
WITH exact_ranked AS (
    SELECT
        cm.*,
        ROW_NUMBER() OVER (
            PARTITION BY cm.fen
            ORDER BY cm.sample_hash
        ) AS fen_rank
    FROM candidate_moves AS cm
),
structural_ranked AS (
    SELECT
        er.*,
        ROW_NUMBER() OVER (
            PARTITION BY
                er.board_position,
                er.side_to_move,
                er.castling_rights_norm,
                er.en_passant_targets_norm
            ORDER BY er.sample_hash
        ) AS structural_rank
    FROM exact_ranked AS er
    WHERE er.fen_rank = 1
)
SELECT
    sr.*
FROM structural_ranked AS sr
CROSS JOIN sampling_params AS p
WHERE sr.structural_rank <= p.max_examples_per_structure;

CREATE OR REPLACE TEMP TABLE balanced_sample AS
WITH ranked AS (
    SELECT
        dm.*,
        ROW_NUMBER() OVER (
            PARTITION BY dm.time_control_type, dm.elo_bucket, dm.phase
            ORDER BY dm.sample_hash
        ) AS bucket_rank
    FROM deduped_moves AS dm
)
SELECT
    ranked.fen,
    ranked.time_control_type,
    ranked.elo_bucket,
    ranked.phase,
    ranked.move_ply,
    ranked.n_pieces,
    ranked.n_possible_moves,
    ranked.gid
FROM ranked
CROSS JOIN sampling_params AS p
WHERE ranked.bucket_rank <= p.balanced_per_bucket_cap;

CREATE OR REPLACE TEMP TABLE natural_sample AS
SELECT
    dm.fen,
    dm.time_control_type,
    dm.elo_bucket,
    dm.phase,
    dm.move_ply,
    dm.n_pieces,
    dm.n_possible_moves,
    dm.gid
FROM deduped_moves AS dm
CROSS JOIN sampling_params AS p
ORDER BY dm.sample_hash
LIMIT (SELECT natural_total_cap FROM sampling_params);

COPY (
    SELECT fen
    FROM balanced_sample
    ORDER BY time_control_type, elo_bucket, phase, move_ply, gid
) TO 'sampled_root_fens_balanced.txt' (
    FORMAT CSV,
    HEADER FALSE,
    DELIMITER '\t'
);

COPY (
    SELECT fen
    FROM natural_sample
    ORDER BY time_control_type, elo_bucket, phase, move_ply, gid
) TO 'sampled_root_fens_natural.txt' (
    FORMAT CSV,
    HEADER FALSE,
    DELIMITER '\t'
);

-- Optional sanity checks to run manually after the exports:
--
-- SELECT time_control_type, elo_bucket, phase, COUNT(*) AS n
-- FROM balanced_sample
-- GROUP BY 1, 2, 3
-- ORDER BY 1, 2, 3;
--
-- SELECT phase, COUNT(*) AS n
-- FROM natural_sample
-- GROUP BY 1
-- ORDER BY 1;
