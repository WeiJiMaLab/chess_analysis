# R-VOC-MQ — Engine VOC / MQ pipeline

**Ref:** `R-VOC-MQ` · [Index](README.md)

## Summary (after implementation)

| | |
|---|---|
| **Description** | `engine_analysis.py` (MQ, VOC), unified eval pipeline `build_pos_with_engine_eval.py`, plotting in `voc_mq_analysis.py`. |
| **Rationale** | Russek-style value-of-computation and move-quality on human positions before normative oracle comparison. |
| **Expectation** | MQ ≤ 0; VOC ≥ 0; multipv=2 reduces extra engine calls; deterministic at fixed depth. |
| **Finding** | Pipeline merged and tested (9 unit tests); MQ clamped for aspiration-window edge cases; lc0 VOC≈0 at depth=1 vs Stockfish — engine choice matters. |

## Procedure

| Step | Status |
|------|--------|
| `engine_analysis.py` MQ/VOC contracts | ✅ done |
| `build_pos_with_engine_eval.py` eval + merge | ✅ done |
| Unit tests `test_engine_analysis.py` | ✅ done |
| E[ΔUC] (Russek Figures 4–5) | ⬜ incomplete |
| Full-dataset loop via `build_selected_moves_with_engine` | ⬜ incomplete |

## Notes

### Definitions

- **MQ** = `e_win_taken − e_win_best` (≤ 0; 0 = optimal)
- **VOC** = `V_deep(a_deep) − V_deep(a_shallow)` with shallow best at depth 1

### Russek alignment (gaps)

| Item | Status |
|---|---|
| Ply 15–75, opp clock ≥ 60s | Partially applied in 100K run; DB filters still open |
| E[ΔUC] over top-5 depth-1 candidates | Not implemented |
| Win prob | WDL direct (not centipawn logistic) |

See [(R-VOC-100K)](human-voc-mq-100k.md) for 100K Stockfish results.
