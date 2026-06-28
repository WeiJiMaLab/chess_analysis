# P1-prep — cost-regime oracle relabel (R-MC-COST)

Relabels the budgeted DP oracle over the **cached** MC episode traces with overridden
cost params — NO tree generation, NO encoder materialization. Module:
`cts.data.preprocess_mc.costsweep_relabel`.

Output shards keep the source schema (graph/feature tensors untouched); only
`oracle_values`, `oracle_stop_steps`, `target_advantages` and the oracle metadata
change. They load directly through `alt_models_eval` / the P0 regret-eval harness.

## Regimes produced (capped smoke, 4000 episodes/split)
Output root: `/scratch/gpfs/GRIFFITHS/hl4291/packed/mc_costsweep/<regime>/`

| regime         | time_mode | time_lambda (=c_bar for linear) | p   | tau |
|----------------|-----------|---------------------------------|-----|-----|
| lambda_0.1     | power_law | 0.1                             | 2.8 | 2.5 |
| lambda_1.0     | power_law | 1.0                             | 2.8 | 2.5 |
| lambda_5.0     | power_law | 5.0                             | 2.8 | 2.5 |
| lambda_18.537  | power_law | 18.537 (production)             | 2.8 | 2.5 |
| linear_0.0     | linear    | 0.0                             |  -  |  -  |
| linear_0.003   | linear    | 0.003                           |  -  |  -  |
| linear_0.02    | linear    | 0.02                            |  -  |  -  |

Each regime dir holds `train/` + `validation/` shards, `{train,validation}_manifest.json`,
and `costsweep_manifest.json` (cost params + self-contained Always/Never/Fraction regret).

## Reproduce one regime (capped smoke)
```bash
source /home/hl4291/venv/bin/activate
export PYTHONPATH=/home/hl4291/chess_analysis/lmcos/src
python -m cts.data.preprocess_mc.costsweep_relabel \
  --source-root /scratch/gpfs/GRIFFITHS/hl4291/packed/mc \
  --output-root /scratch/gpfs/GRIFFITHS/hl4291/packed/mc_costsweep/lambda_1.0 \
  --time-mode power_law --time-lambda 1.0 \
  --max-episodes-per-split 4000
```

## Full run (all ~375K episodes/split)
Same command, **drop** `--max-episodes-per-split`. Single-process, CPU-only, reads/writes
40 shards/split; ~minutes per regime. Not auto-submitted to SLURM. To run the whole sweep
just loop the table above over `--time-mode`/`--time-lambda` and a matching `--output-root`.

## Final measurement
The regime regret numbers here are a quick sanity preview. The authoritative regret/OSS
measurement runs the relabeled dirs through the P0 regret-eval harness (`alt_models_eval`
and the Readout family), which is built/owned by the P0 agent.
