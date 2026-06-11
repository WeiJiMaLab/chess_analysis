# R-VOC-100K — 100K Stockfish VOC/MQ run

**Ref:** `R-VOC-100K` · [Index](README.md) · Prerequisite: [(R-VOC-MQ)](human-voc-mq.md)

## Summary (after implementation)

| | |
|---|---|
| **Description** | 100K positions: Stockfish depth **5/1**, ply **15–75**, `opponent_clock_time ≥ 60s`; flat `figures/` conventions. |
| **Rationale** | Scale VOC–RT and MQ–clock effects beyond 10K smoke. |
| **Expectation** | r(log RT, VOC) > 0; MQ–clock sign needs VOC-mediated interpretation. |
| **Finding** | r(log RT, VOC) = **+0.097**; r(MQ, clock) = **−0.098** (negative within every ply tertile); VOC > 0.005 in **33%** of positions. |

## Notes

### Summary stats (n=100K)

| Column | Mean | p50 |
|---|---|---|
| VOC | +0.100 | 0.000 |
| MQ | −0.116 | 0.000 |
| toptwo | +0.123 | 0.013 |

### MQ vs clock

Players with **more** clock at a ply tended to have **worse** MQ — consistent with having rushed through low-VOC positions earlier. Effect may differ at depth=15 (Russek).

### Plot conventions (2026-06-02)

- Flat `figures/`; `x_vs_y.png` naming; 50-bin histograms; Analyzer 2×2 dashboards; axis labels use **log** not ln.

Legacy detail: [(R-ARCH-HUMAN)](archive-human-analytics-notebook-legacy.md).
