# AGENTS.md

## Repo Purpose

This repository is a Python research codebase for chess tree-search/controller experiments.
Most source files live at the repo root, with analysis and data-generation utilities in `scripts/`,
cluster jobs in `slurm/`, SQL helpers in `sql/`, and tests in root-level `test_*.py` files.

## Working Rules

- Start by reading the relevant module and its nearest tests before editing.
- Prefer small, local fixes over broad refactors unless the user explicitly asks for restructuring.
- Preserve experimental intent. If a change affects training semantics, reward definitions, rollout logic,
  evaluation procedure, or dataset generation, call that out clearly.
- Do not touch heavyweight artifacts in `weights/` or external scratch-path data unless the user asks.
- When changing experiment behavior in a meaningful way, update `LAB_NOTEBOOK.md` with intent,
  meaningful change, and result once the outcome is known.

## Project Map

- `supervised_branch.py`, `supervised_branch_cli.py`: controller training loop and CLI entrypoints.
- `cts_pretrain.py`, `tensorizer.py`, `TreeMHA.py`, `GNN.py`: encoder/pretraining/model components.
- `controller_oracle.py`, `planning_state.py`, `tree.py`: planning/oracle/tree utilities.
- `cts_rl.py`, `cts_episode_envs.py`: RL environment and rollout-related logic.
- `cts_uci_common.py`, `cts_uci_parsers.py`, `cts_uci_process.py`, `uci_provider.py`: engine/UCI plumbing.
- `scripts/`: diagnostics, dataset generation, analysis, and experiment helpers.
- `test_*.py`: pytest coverage for training, diagnostics, plumbing, and data generation.
- `LAB_NOTEBOOK.md`: running record of meaningful experimental changes and outcomes.

## Validation Expectations

- For targeted code changes, run the smallest relevant pytest file first.
- For broad or cross-cutting changes, run `python -m pytest`.
- If a change affects scripts only, prefer a focused script `--help` or a narrow regression test over
  launching a long training job unless the user explicitly wants that.
- If you cannot run the relevant validation, say so explicitly and explain why.

## Common Commands

- Run all tests:
  - `python -m pytest`
- Run one test file:
  - `python -m pytest test_supervised_branch.py`
- Inspect CLI options:
  - `python supervised_branch_cli.py --help`
- Inspect script options:
  - `python scripts/controller_synthetic_diagnostic.py --help`

## Codex Workflow For This Repo

When asked to implement something:

1. Restate the concrete outcome and identify the likely files.
2. Read the relevant implementation and tests.
3. Edit only the files needed for the task.
4. Run focused validation.
5. Summarize behavior changes, validation run, and any residual risk.

When asked for an experiment or analysis task:

1. Identify whether this is code, a script invocation, or interpretation-only work.
2. Avoid changing training logic and analysis logic in the same step unless necessary.
3. Record meaningful experimental changes in `LAB_NOTEBOOK.md` after results are established.

## Prompting Guidance For Users

Codex works best here when requests include:

- The desired outcome.
- The file or subsystem if known.
- The validation target.
- Any constraints on scope.

Good example:

`Fix PPO rollout collection in supervised_branch.py and add a regression test in test_supervised_branch.py. Run the smallest relevant pytest file afterward.`

Weaker example:

`Can you look at my training code?`
