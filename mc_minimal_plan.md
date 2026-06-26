# mc_minimal_plan.md — The goal: FENs → halt-model loss profile

**The goal (everything else is fluff).** One pipeline: a subset of FENs → a plot profiling the four
halt models — **NeverHalt, AlwaysHalt, FractionHalt, MCHalt** — on **Regret**, **Stop-Accuracy**, and
**Loss** (where defined). `MCHalt` = the trained GNN meta-controller; the other three are parameter-free
or one-parameter baselines. Current code/data lineage: [mc_pipeline.md](mc_pipeline.md). History:
labnotebook.

---

## Pipeline stages (status — verified 2026-06-26)

| # | stage | code | lc0 prod | SF ladder |
|---|---|---|---|---|
| 1 | FENs → trees | `cts.data.build_tree` | ✅ | ✅ 150/rung smoke |
| 2 | filter | `filter_trees_by_trace` | ✅ informative | ⚠️ valid-FEN guard only |
| 3 | GNN pack | `preprocess_gnn.pack` | ✅ | ✅ elo1800 only |
| 4 | encoder pretrain (child-WDL) | `gnn_pretrain` | ✅ `encoder_best.pt` | ✅ `tiny_encoder.pt` |
| 5 | MC pack (oracle labels) | `preprocess_mc.pack` | ✅ `packed/mc` | ❌ not run |
| 6 | materialize root rep | `preprocess_mc.materialize` | ✅ | ❌ not run |
| 7 | controller train (MCHalt) | `controller_train` | ✅ `controller_filtered.pt` | ❌ not run |
| 8 | **eval + plot 4 models** (Regret/StopAcc/Loss) | `alt_models_eval` (+driver) | ⚠️ partial | ❌ |

**The goal is reachable NOW on lc0 prod** (stages 1–7 all done) — it only needs stage 8 wired. SF needs
stages 5–7 before it can feed stage 8.

## What's built

- **Readout family** ([readout.py](lmcos/src/models/readout.py)): `AlwaysStop`, `NeverStop`,
  `FractionStop`, `GnnMetaController` (= MCHalt), all behind one decision rule. *(Also contains
  `StatsReadout` + scrapped `ValueGainReadout`/`HaltCurveReadout` — to delete.)*
- **Trained MCHalt** on lc0 prod: `controller_filtered.pt` — regret **0.209**, P(stop==OSS) **0.314**
  (recorded; independently spot-checked through one harness).
- **Single-harness scoring proven**: a driver scores Always/Never/Fraction + the trained controller over
  one prod-val episode set; the oracle baseline matches across both paths (alignment verified at 3000 ep).
  Headline it produces: **MCHalt (~0.21) loses to FractionHalt (~0.068)** — the result the plot will show.

## What's still needed for the goal (checklist)

- [ ] **Stage 8 — one eval/plot entrypoint.** Load all four Readout models incl. **MCHalt from a trained
      checkpoint** (not a passed-in scalar), score over one episode set, emit **Regret-by-model** +
      **Stop-Accuracy-by-model** as horizontal bars → `figures/lmcos/regret_by_model.png`, `oss_by_model.png`.
      (Fold the verified driver `scratchpad/profile_controller.py` logic into `alt_models_eval.py`.)
- [ ] **Define "Loss by model" — NEEDS A DECISION.** Only trainable readouts have a training loss;
      Always/Never/Fraction don't. Options: (a) report **Regret** as the universal metric + training-loss
      only for MCHalt, or (b) drop the Loss panel. Pick one.
- [ ] **Single FENs → plot driver.** Stages 1–8 are 8 separate CLI/SLURM steps; wrap them so a FEN subset
      runs straight through to the plot (the actual "end-to-end" ask).
- [ ] **(SF path) run stages 5–7 on `sf_packed/elo1800`** — MC pack → materialize → train a tiny MCHalt —
      if the demo is to be on Stockfish trees rather than lc0 prod.
- [ ] **Clean `readout.py`**: delete `ValueGainReadout`, `HaltCurveReadout` (VG scrapped); decide whether
      `StatsReadout` stays (it's a 5th "tree-stats" tier, not in the 4-model goal).

## Decisions for the user
1. **Demo on lc0 prod (ready now) or push SF through stages 5–7 first?**
2. **What is "Loss by model"** — MCHalt-only training loss alongside universal Regret, or drop it?

## File map
| what | path |
|---|---|
| Readout family | [readout.py](lmcos/src/models/readout.py) |
| eval + plot (stage 8, to finish) | [alt_models_eval.py](lmcos/analysis/_budgeted/alt_models_eval.py) |
| verified scoring driver (to fold in) | `scratchpad/profile_controller.py` |
| trained MCHalt (lc0 prod) | `/scratch/.../GNN/controller_filtered.pt` |
| SF trees / packs / tiny encoder | `/scratch/.../sf_trees/`, `/scratch/.../sf_packed/elo1800/` |
| code/data lineage + definitions | [mc_pipeline.md](mc_pipeline.md) |
