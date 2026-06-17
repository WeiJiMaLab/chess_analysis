# R-HUMAN-BACKLOG — Human analytics follow-ups

**Ref:** `R-HUMAN-BACKLOG` · [Index](README.md)

## Summary (during active implementation)

| | |
|---|---|
| **Description** | Open human-data tasks not tied to a single numbered analysis (filters, E[ΔUC], full-dataset engine loop). |
| **Rationale** | Russek alignment and production-scale VOC/MQ annotation deferred during A0–A4 focus. |
| **Expectation** | Filters match Russek ply 15–75 + opp clock ≥ 60s globally; E[ΔUC] enables Figures 4–5 style analyses. |
| **Open questions / notes** | lc0 depth for meaningful non-zero VOC still TBD. |

## Procedure

| Step | Status |
|------|--------|
| Align DB filters: ply **15–75**, `opponent_clock_time >= 60s` | ⬜ incomplete |
| Full MQ vs clock at SF depth 10+, 10K+ positions | ⬜ incomplete |
| Full log(RT) vs VOC at depth 10+ | ⬜ incomplete |
| Implement **E[ΔUC]** (top-5 depth-1 candidates) | ⬜ incomplete |
| Loop VOC/MQ into `build_selected_moves_with_engine` | ⬜ incomplete |
| Required lc0 depth for non-trivial VOC | ⬜ incomplete |

## Notes

See [(R-VOC-MQ)](human-voc-mq.md) and [(R-ARCH-HUMAN)](archive-lmcos-notebook-legacy.md) for design context.
