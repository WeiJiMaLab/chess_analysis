# R-THEORY — Stopping proxies and normative comparison

**Ref:** `R-THEORY` · [Index](README.md)

## Summary (after implementation)

| | |
|---|---|
| **Description** | Framework linking human RT to normative computation: proxies (`gain_depth`, `min_expansions`, E[ΔUC]) vs lmcos DP oracle `oracle_stop_step`. |
| **Rationale** | Clarify what claim each analysis supports (directional feature match vs same-position RT vs trained controller). |
| **Expectation** | DP oracle should correlate with RT better than crude two-depth gaps when measured on matched positions. |
| **Finding** | Q-refinement stopping is locally rational but biased (“chasing tails”); min_expansions ≠ DP oracle; A0a superseded packed-shard directional check — see [(R-ORACLE-RT)](oracle-stop-vs-human-rt.md). |

## Notes

### Target correlation

$$r(\text{DP-Oracle opt depth},\; \text{human RT}) > r(\text{gain\_depth},\; \text{human RT})$$

### Three validation tiers

| Tier | Claim | Requires |
|---|---|---|
| A | Same features predict oracle stop and RT directionally | lmcos trees (A0a) |
| B | Same FEN: oracle stop ↔ human RT | Human FEN trees (A1) |
| C | Trained controller ↔ human RT | Controller on human positions |

### Epistemology (min_expansions vs oracle)

Agents chase Q tails in dominant positions (high gain_depth, small min_expansions) while stopping early in ambiguous ones — opposite of efficient allocation. E[ΔUC] targets unobservable “would search change my move?”

Full prose: [(R-ARCH-HUMAN)](archive-lmcos-notebook-legacy.md) § The epistemology of stopping.
