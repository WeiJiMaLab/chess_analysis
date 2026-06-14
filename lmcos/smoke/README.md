# Operational smoke tests — batched tree generation (R-BATCHGEN §5)

These scripts implement the **§5 "Smoke tests (operational)"** table of
[`reports/batched-tree-generation.md`](../../reports/batched-tree-generation.md).
They are run **by hand, before scale-up**, to gate the new in-process batched
search-tree generator (`cts.data.batched_gen`) against the legacy lc0-UCI path.

They are *operational* checks (wiring, throughput, memory, parity evidence), not
the committed unit tests in `tests/` (the §4 `T-*` table). Run them on the
cluster where the lc0 binary, weights, and FEN pool live.

## Defaults (real cluster assets)

- lc0 binary: `/scratch/gpfs/GRIFFITHS/ysagiv/tools/lc0/build/release/lc0`
- lc0 weights: `/scratch/gpfs/GRIFFITHS/ysagiv/chess/weights/t1-256x10-distilled-swa-2432500.pb.gz`
- FEN pool: `/scratch/gpfs/GRIFFITHS/hl4291/lmcos/fens.txt` (4-field FENs; scripts append `" 0 1"`)

Every script samples FENs from the pool unless `--fens FILE` is given, takes
`--seed` (reproducible sampling), and accepts `--out DIR` for a JSON result
file. Run any script with `--help` for its full flag set.

If `cts.data.batched_gen` (or one of its symbols) isn't implemented yet, the
scripts print a clear `[SCAFFOLD NOT READY]` / `[NOT IMPLEMENTED]` message and
exit non-zero (code 2) instead of crashing.

## Scripts → §5 ID → one-line run

| Script | §5 ID | Proves | Run command |
|---|---|---|---|
| `s_1tree.py` | **S-1tree** | end-to-end wiring + L1/L3 on one tree | `python smoke/s_1tree.py --out smoke_out` |
| `s_n_lockstep.py` | **S-Nlock** | batching preserves each tree's order | `python smoke/s_n_lockstep.py --n 8 --out smoke_out` |
| `s_throughput.py` | **S-thru** | the 10–100× trees/s claim; pick batch N | `python smoke/s_throughput.py --n 100 --batch-sizes 1,32,256,1024 --device cuda --out smoke_out` |
| `s_memory.py` | **S-mem** | peak RSS with 1,000 live trees | `python smoke/s_memory.py --n 1000 --device cuda --out smoke_out` |
| `s_backend.py` | **S-backend** | L2 numeric drift + determinism | `python smoke/s_backend.py --n 256 --out smoke_out` |
| `s_parity1k.py` | **S-parity1k** | L3 go/no-go: % trajectory-identical | `python smoke/s_parity1k.py --n 1000 --device cuda --out smoke_out` |
| `s_cpu_throughput.py` | **S-cpu** | CPU-fleet lane trees/s vs cores | `python smoke/s_cpu_throughput.py --n 100 --threads 1,2,4,8,16 --out smoke_out` |

Run from the `lmcos/` directory so `cts` and the shared `smoke/_*.py` helpers
import cleanly. Exit codes: `0` pass / measure-only, `1` parity FAIL, `2`
backend scaffold not ready.

## Shared helpers (not smoke tests themselves)

- `_smoke_common.py` — FEN load/normalize, FEN sampling, scaffold-safe imports,
  the `SmokeResult` envelope, and the `PretrainExample` diff used by the parity
  scripts.
- `_lc0_baseline.py` — the two-engine `Lc0DirectEvalProvider` + sequential
  `build_pretrain_example` baseline, wired exactly like `build_tree.py`.
- `_gpu_util.py` — best-effort `nvidia-smi` utilization sampling.

## Recommended order

1. `s_backend.py` (L2 — is the evaluator close to lc0 and deterministic?)
2. `s_1tree.py` then `s_n_lockstep.py` (L1 — does the search logic match?)
3. `s_parity1k.py` (L3 — end-to-end trajectory identity, the green-light read)
4. `s_throughput.py` / `s_memory.py` / `s_cpu_throughput.py` (speedup + sizing)
