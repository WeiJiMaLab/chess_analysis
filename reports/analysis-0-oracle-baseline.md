# R-A0 — Analysis 0: oracle direction + minimal MC

**Ref:** `R-A0` · [Index](README.md) · Next: [(R-A1)](analysis-1-human-oracle.md), [(R-A2)](analysis-2-minimal-model.md)

## Summary (after implementation)

| | |
|---|---|
| **Description** | **0a:** `oracle_stop_step` vs board features on lmcos `filtered_shard` trees. **0b:** 4-feature MLP vs GNN+MC halt/continue sign accuracy. |
| **Rationale** | De-risk A1 (human FEN trees) and A2 (skip GNN pretrain) before cluster spend. |
| **Expectation** | ≥3/4 feature directions match human RT proxies; val sign acc ≥ 80% without GNN. |
| **Finding** | **A0a go** (3/4; toptwo mismatch by design). **A0b go** (val **86.4%** vs GNN+MC **90.1%**). Tests **25/25**. |

## Procedure

| Step | Status |
|------|--------|
| A0a `oracle_stop_step_features.py` (n=5K, budget=43) | ✅ done |
| A0b `minimal_mc_baseline.py` (2K train / 500 val) | ✅ done |
| Supplementary `min_expansions_analysis.py` | ✅ done |
| Unit tests | ✅ done |

## Notes

### A0a correlations (budget=43)

| Feature | Oracle r | Human r | Match |
|---|---|---|---|
| branching | +0.165 | +0.195 | ✓ |
| material | +0.154 | +0.039 | ✓ |
| gain_depth | +0.797 | +0.096 | ✓ |
| toptwo | +0.436 | −0.064 | ✗ |

`oracle_stop_step`: mean=11.7, std=10.3.

### A0b sign accuracy

| Model | Val sign acc |
|---|---|
| GNN+MC | 90.1% |
| Minimal MLP | **86.4%** |

**Scripts:** `lmcos/analysis/oracle_stop_step_features.py`, `minimal_mc_baseline.py`.  
**Figures:** `lmcos/analysis/figures/`.  
**Oracle:** `compute_budgeted_oracle()` — WDL halt rewards from `oracle_root_q_trace`.
