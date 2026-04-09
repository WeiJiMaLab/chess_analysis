# Repo Lessons

- Diagnostic summaries for this project should default to about 3 decimal places unless more precision is needed to debug a numerical issue.
- ML/AI experiment utilities should use proper scalable data flow by default: cache expensive datasets, use DataLoaders/tensor batches where appropriate, and avoid ad hoc Python loops that undermine repeatability.
- When providing cluster experiment instructions, separate local-terminal sync commands from cluster-terminal run commands; do not combine them into one executable block.
- Toy representations such as `oracle-action-now` and frozen-encoder diagnostics should be precomputed into tensor datasets; do not route them through repeated raw-tree episode construction or frozen-encoder passes each epoch.
- Distinguish semantic readiness from operational readiness. Do not tell the user a long ML/data job is “ready to run” without explicitly calling out known speed/throughput debt, expected walltime risk, and whether an optimization pass is still warranted.
- When the user asks whether further optimization is possible, provide a comprehensive, prioritized inventory of remaining optimization levers (algorithmic, vectorization/data-structure, and job orchestration), not just the first few ideas that come to mind.
- In this repo/thread, optimize for completeness over speed in explanations. Define every term/symbol before using it, state assumptions explicitly, motivate equations and design choices, and prefer a complete answer over a fast but underspecified one.
- When explaining a system or algorithm, do not stop at the first plausible summary. Include: what problem it solves, how the pieces are defined, why the formulation is correct, what remains approximate/suboptimal, and what meaningful alternatives exist when relevant.
