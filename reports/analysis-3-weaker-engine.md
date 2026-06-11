# R-A3 — Analysis 3: weaker engine (SF ELO 2000)

**Ref:** `R-A3` · [Index](README.md) · Prerequisite: [(R-A1)](analysis-1-human-oracle.md) — **conditional**

## Summary (prior to implementation)

| | |
|---|---|
| **Description** | Regenerate oracle trees on **same human FENs** using Stockfish `UCI_LimitStrength`, ELO=2000; compare r(oracle_stop_step, log RT) vs Lc0 oracle. |
| **Rationale** | Lc0 ~3000+ ELO may “see through” positions; human pool is ≥2000 ELO — strength-matched engine may align deliberation better. |
| **Expectation** | SF2000 oracle correlates more strongly with human RT than full-strength Lc0, especially on tactical errors. |
| **Open questions / notes** | **On hold** until A1 reports matched-position r. CP→WDL via logistic `pwin = 1/(1+exp(-cp/400))`. Retraining GNN+MC on SF trees is optional follow-on. |

## Procedure

| Step | Status |
|------|--------|
| A1 gives directional r(oracle_stop_step, log RT) | ⬜ incomplete (gate) |
| `build_tree.py` Stockfish + ELO option + CP→WDL halt rewards | ⬜ incomplete |
| Trees on 10K human FENs → `human_trees_sf2000/` | ⬜ incomplete |
| `compute_budgeted_oracle()` + correlation vs A1 | ⬜ incomplete |
| Side-by-side correlation figure (Lc0 vs SF2000) | ⬜ incomplete |
| `test_sf2000_oracle.py` | ⬜ incomplete |

## Notes

### If SF2000 also fails

Strengthens case that engine VOC does not capture human deliberation (fundamental mismatch, not strength artifact).

### Output paths

- Trees: `/scratch/gpfs/GRIFFITHS/hl4291/tmp/human_trees_sf2000/`
- Stockfish: `/home/hl4291/stockfish/src/stockfish` (SF14)
