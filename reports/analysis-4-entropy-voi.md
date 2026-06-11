# R-A4 — Analysis 4: entropy value-of-information stopping

**Ref:** `R-A4` · [Index](README.md)

## Summary (after implementation)

| | |
|---|---|
| **Description** | Rule B: stop when marginal Shannon-entropy reduction at root falls below θ; Stockfish multi-depth traces on human positions (`entropy_voi_analysis.py`). |
| **Rationale** | Test information-theoretic stopping on **same** human DB positions as RT analyses (CPU-feasible vs GPU tree gen). |
| **Expectation** | d* correlates with branching and log(RT); Approach B (neutral imputation) more stable than top-K-only softmax. |
| **Finding** | **6,494** traces (10K job); r(d*, branching) ~**+0.10** at θ=0.001; r(d*, log RT) ~**+0.01** (near zero) — entropy rule does **not** explain human RT at scale. |

## Procedure

| Step | Status |
|------|--------|
| `query_stockfish_multidepth` + CP→WDL (depth 1–8) | ✅ done |
| Sample human positions + cache traces | ✅ done |
| Approach A/B entropy + θ sweep | ✅ done |
| SLURM `entropy_voi_10k.slurm` (6,494 positions) | ✅ done |
| Figures: `entropy_voi_vs_human_rt.png`, trajectories, correlation table | ✅ done |
| `test_entropy_voi.py` | ✅ done |
| Exhaustive β sweep (if revisiting) | ⬜ incomplete |

## Notes

### Timing (10K scale)

- **238.6 s** for 6,494 traces, 16 CPU workers (~**0.037 s/position**)

### Correlations (θ sweep; see `figures/entropy_voi_correlations.md`)

| θ | r(d*_B, log RT) | r(d*_B, branching) |
|---|---|---|
| 0.001 | +0.014 | +0.101 |
| 0.005 | −0.005 | +0.039 |

### Scripts

- `human_analytics/entropy_voi_analysis.py`
- `human_analytics/slurm/entropy_voi_10k.slurm`

### Interpretation

Branching alignment partial; RT alignment absent — supports investigating engine-strength mismatch [(R-A3)](analysis-3-weaker-engine.md) rather than this entropy rule alone.
