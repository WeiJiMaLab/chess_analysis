# R-HUMAN-DATA — Human Lichess dataset

**Ref:** `R-HUMAN-DATA` · [Index](README.md)

## Summary (after implementation)

| | |
|---|---|
| **Description** | Lichess 10+0 games (Oct–Dec 2023), preprocessed via `human_analytics/slurm/`, stored in DuckDB `personal.db`. |
| **Rationale** | Large-scale human RT and move-quality signals for VOC/MQ and oracle comparison. |
| **Expectation** | Filtered tables support ply 15–75, opp clock ≥ 60s analyses aligned with Russek et al. |
| **Finding** | **1.97M** games; **135M** nonzero-RT moves in `processed_moves_nonzero`; board-feature columns and `ply_tertiles` available. |

## Notes

### Filters (preprocessing)

- Time control: **10+0** (600s, no increment)
- Min ELO: **2000** (both players)
- Excluded: negative `move_time`, berserk, extra-time grants

### Tables (`/scratch/gpfs/GRIFFITHS/hl4291/personal.db`)

| Table | Rows | Notes |
|---|---|---|
| `games` | 1,971,698 | Selected games |
| `moves` | 145,142,731 | Raw moves |
| `processed_moves` | 145,142,731 | + FEN, piece counts, `ply_tertiles` |
| `processed_moves_nonzero` | 135,482,903 | `move_time > 0` |

`processed_moves` adds: `n_pieces_on_board_{inc,exc}_pawns`, `n_self_pieces_exc_pawns`, `n_opp_pieces_exc_pawns`, `fen` (4-field), `ply_tertiles`.
