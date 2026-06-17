# R-MOVETIME-PRIOR — Move-time dashboards (pre-2026-06-02)

**Ref:** `R-MOVETIME-PRIOR` · [Index](README.md)

## Summary (after implementation)

| | |
|---|---|
| **Description** | Six bivariate `move_time` analyses via `movetime_analysis.py` / `slurm/analysis.sh`, plus `move_time_summary.py` and `ply_premove.py`. |
| **Rationale** | Establish baseline RT structure before engine VOC/MQ and lmcos oracle work. |
| **Expectation** | RT rises with clock, branching, and material; non-monotone ply arc; premoves concentrated in opening/late endgame. |
| **Finding** | All dashboards regenerated 2026-06-02; `ln(move_time)` ~ log-normal; key correlations documented in legacy notebook § Prior Analyses. |

## Notes

| Analysis | X variable | Key finding |
|---|---|---|
| `clock` | `player_clock_time` | RT ↑ with remaining clock |
| `clock_opp` | `opponent_clock_time` | Weaker opponent-clock effect |
| `npossiblemoves` | `n_possible_moves` | RT ↑ with branching |
| `pieces_exc` / `self_pieces_exc` | piece counts | RT ↑ in richer positions |
| `ply` | `move_ply` | Fast openings → slower mid → faster end |

Figures: `human_analytics/figures/`. Full tables: [(R-ARCH-HUMAN)](archive-lmcos-notebook-legacy.md).
