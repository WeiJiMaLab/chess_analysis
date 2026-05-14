"""CTS — Compute Tree Search meta-controller research codebase.

Train a small neural network that decides *when to stop searching* in a
chess MCTS-style policy: given a partially-expanded tree and a remaining
time budget, predict halt vs continue. The training data is produced by a
budgeted oracle; the controller fits the oracle's choices by regression.

Subpackages:

- ``cts.core``     — shared substrate (``SearchTree``, tensorizer, feature schema, lc0 providers)
- ``cts.data``     — tree generation + preprocessing (encoder pretrain chain, controller-data chain)
- ``cts.models``   — neural-network modules (``TreeEncoder`` + ``MetaController``)
- ``cts.train``    — training loops (encoder pretrain, controller fitted-Q)
- ``cts.analysis`` — diagnostics, plotting, evaluation tools

Every entry point reads a Pydantic-validated YAML config via
``--config PATH`` plus optional ``--override key=value`` flags.
``LAB_NOTEBOOK.md`` is the running experimental log.
"""
