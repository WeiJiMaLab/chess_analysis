# Repo Lessons

- Diagnostic summaries for this project should default to about 3 decimal places unless more precision is needed to debug a numerical issue.
- ML/AI experiment utilities should use proper scalable data flow by default: cache expensive datasets, use DataLoaders/tensor batches where appropriate, and avoid ad hoc Python loops that undermine repeatability.
- When providing cluster experiment instructions, separate local-terminal sync commands from cluster-terminal run commands; do not combine them into one executable block.
- Toy representations such as `oracle-action-now` should be precomputed into tensor datasets for training diagnostics; do not route them through repeated raw-tree episode construction each epoch.
