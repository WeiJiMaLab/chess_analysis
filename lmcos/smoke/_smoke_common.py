"""Shared scaffolding for the batched-tree-generation smoke tests.

WHY this module exists: every smoke script in this directory needs the same
three things and nothing in the codebase yet provides them in one place —

1. A *robust* import of the still-being-built ``cts.data.batched_gen``
   subpackage. The package is a scaffold: symbols may be missing, or present
   but raising ``NotImplementedError``. A smoke test should report "backend
   not ready yet" clearly rather than crash with an opaque ``ImportError``.
2. FEN sampling from the ground-truth pool with the 4->6-field normalization
   (``fens.txt`` stores 4-field FENs; lc0 / python-chess want 6 fields).
3. The real engine/asset paths and the canonical search knobs, so each
   script's ``--help`` shows sensible defaults that point at the cluster.

Keeping load/normalization/wiring here lets each script stay a thin
load -> compute -> report driver, per the repo's analysis-script ethos.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence

# Smoke scripts live in ``lmcos/smoke``; ``cts`` is the editable install rooted
# at ``lmcos/src``. Mirror the analysis scripts and make the package importable
# even when the environment hasn't been ``pip install -e``'d.
_LMCOS_ROOT = Path(__file__).resolve().parent.parent
if str(_LMCOS_ROOT) not in sys.path:
    sys.path.insert(0, str(_LMCOS_ROOT))


# --- Real cluster paths and canonical search knobs (defaults for --help) ---

DEFAULT_LC0_BINARY = "/scratch/gpfs/GRIFFITHS/ysagiv/tools/lc0/build/release/lc0"
DEFAULT_LC0_WEIGHTS = (
    "/scratch/gpfs/GRIFFITHS/ysagiv/chess/weights/t1-256x10-distilled-swa-2432500.pb.gz"
)
DEFAULT_FEN_POOL = "/scratch/gpfs/GRIFFITHS/hl4291/lmcos/fens.txt"

# Search knobs mirrored from build_tree.py's generate-dataset defaults so the
# smoke trees match what scale-up will actually produce.
DEFAULT_MAX_DEPTH = 4
DEFAULT_SEARCH_BUDGET = 64
DEFAULT_C_PUCT = 1.0
DEFAULT_MIN_NODES = 16
DEFAULT_MAX_NODES = 128
DEFAULT_MULTIPV = 8


class ScaffoldNotReady(RuntimeError):
    """Raised when ``cts.data.batched_gen`` is absent or still a stub.

    Smoke scripts catch this at the top level and print a clear "the batched
    generator isn't implemented yet" message with a non-fatal exit, so the
    operator learns *what* is missing instead of reading a traceback.
    """


def import_batched_gen() -> Any:
    """Import ``cts.data.batched_gen`` or raise :class:`ScaffoldNotReady`.

    Returns the package module; callers reach into ``.search``,
    ``.evaluator``, ``.net_evaluator`` submodules from there.
    """
    try:
        return importlib.import_module("cts.data.batched_gen")
    except ImportError as exc:  # subpackage not created yet
        raise ScaffoldNotReady(
            "cts.data.batched_gen is not importable yet "
            f"(expected under {_LMCOS_ROOT / 'src' / 'data' / 'batched_gen'}). "
            f"Underlying import error: {exc}"
        ) from exc


def import_symbol(module_name: str, symbol: str) -> Any:
    """Import one public symbol from a ``cts.data.batched_gen`` submodule.

    ``module_name`` is the leaf name (e.g. ``"search"``, ``"evaluator"``,
    ``"net_evaluator"``); the full dotted path is built here so every script
    references the same contract.
    """
    import_batched_gen()  # ensures the parent package exists / gives a clean error
    full_name = f"cts.data.batched_gen.{module_name}"
    try:
        module = importlib.import_module(full_name)
    except ImportError as exc:
        raise ScaffoldNotReady(f"{full_name} is not importable yet: {exc}") from exc
    try:
        return getattr(module, symbol)
    except AttributeError as exc:
        raise ScaffoldNotReady(
            f"{full_name}.{symbol} is not defined yet (contract symbol missing)."
        ) from exc


# --- FEN loading / normalization ---


def normalize_fen(raw_fen: str) -> str:
    """Normalize a pool FEN to 6 fields by appending the halfmove/fullmove pair.

    ``fens.txt`` stores 4-field FENs (board, side, castling, en-passant). lc0
    and python-chess expect 6 fields; the spec says to append ``" 0 1"``. If a
    FEN already has 5+ fields we leave it untouched.
    """
    fields = raw_fen.split()
    if len(fields) == 4:
        return raw_fen + " 0 1"
    return raw_fen


def load_fen_pool(path: str) -> List[str]:
    """Read newline-separated FENs from ``path``, normalized to 6 fields."""
    with open(path, "r", encoding="utf-8") as handle:
        return [normalize_fen(line.strip()) for line in handle if line.strip()]


def sample_fens(
    *,
    explicit_fens_path: Optional[str],
    pool_path: str,
    count: int,
    seed: int,
) -> List[str]:
    """Return ``count`` FENs: from ``--fens`` if given, else sampled from the pool.

    Sampling is *without replacement* and seeded so a given ``--seed`` always
    selects the same held-out set — smoke runs must be reproducible to compare
    new vs old. If the pool is smaller than ``count`` we take all of it.
    """
    if explicit_fens_path:
        fens = load_fen_pool(explicit_fens_path)
        return fens[:count] if count > 0 else fens
    pool = load_fen_pool(pool_path)
    if count <= 0 or count >= len(pool):
        return pool
    rng = random.Random(seed)
    return rng.sample(pool, count)


# --- Reporting ---


@dataclass
class SmokeResult:
    """Uniform result envelope every smoke script prints and (optionally) saves.

    ``passed`` is None for scripts that only *measure* (throughput, memory)
    rather than *assert* parity; True/False for the parity scripts.
    """

    smoke_id: str
    passed: Optional[bool]
    summary: dict

    def to_stdout(self) -> None:
        """Print a human-scannable summary block."""
        verdict = "MEASURE-ONLY" if self.passed is None else ("PASS" if self.passed else "FAIL")
        print(f"\n=== {self.smoke_id}: {verdict} ===")
        for key, value in self.summary.items():
            print(f"  {key}: {value}")

    def write_json(self, out_dir: Optional[str]) -> Optional[str]:
        """Write the result to ``<out_dir>/<smoke_id>.json`` if ``out_dir`` set."""
        if not out_dir:
            return None
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{self.smoke_id}.json")
        payload = {"smoke_id": self.smoke_id, "passed": self.passed, "summary": self.summary}
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)
        return path


def add_common_arguments(parser: argparse.ArgumentParser, *, default_n: int) -> None:
    """Attach the flags every smoke script shares: FEN source, paths, seed, out.

    WHY centralized: keeps each script's ``--help`` consistent and ensures the
    real cluster paths are the defaults everywhere (operator can just run the
    script bare on the cluster).
    """
    parser.add_argument(
        "--n", type=int, default=default_n, help="Number of FENs / trees to use."
    )
    parser.add_argument(
        "--fens", type=str, default=None,
        help="Path to a FEN file (one per line). Defaults to sampling the ground-truth pool.",
    )
    parser.add_argument(
        "--fen-pool", type=str, default=DEFAULT_FEN_POOL,
        help="Ground-truth FEN pool sampled when --fens is not given.",
    )
    parser.add_argument(
        "--lc0-binary", type=str, default=DEFAULT_LC0_BINARY, help="lc0 release binary."
    )
    parser.add_argument(
        "--lc0-weights", type=str, default=DEFAULT_LC0_WEIGHTS, help="lc0 network weights (.pb.gz)."
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed (FEN sampling + budget buckets).")
    parser.add_argument(
        "--out", type=str, default=None, help="Optional directory for a JSON result file."
    )


def run_smoke(main_fn: Callable[[argparse.Namespace], SmokeResult], args: argparse.Namespace) -> int:
    """Execute ``main_fn``, handle the scaffold-not-ready case, return an exit code.

    Exit codes: 0 = pass/measure-only, 1 = parity FAIL, 2 = scaffold not ready.
    Centralizing this means every script degrades gracefully and uniformly when
    the batched generator is still a stub.
    """
    try:
        result = main_fn(args)
    except ScaffoldNotReady as exc:
        print(f"\n[SCAFFOLD NOT READY] {exc}")
        print("This smoke test cannot run until the batched generator is implemented.")
        return 2
    except NotImplementedError as exc:
        print(f"\n[NOT IMPLEMENTED] A batched-gen symbol raised NotImplementedError: {exc}")
        print("Backend is still a scaffold; re-run once the method is implemented.")
        return 2
    result.to_stdout()
    saved = result.write_json(getattr(args, "out", None))
    if saved:
        print(f"\n  wrote {saved}")
    if result.passed is False:
        return 1
    return 0


# --- Tree/example diffing (used by parity smoke tests) ---


def diff_pretrain_examples(new_example: Any, old_example: Any) -> List[str]:
    """Return a list of human-readable mismatch strings between two examples.

    Empty list == byte-faithful match on the fields the L1 contract cares
    about: tree topology + node FENs/moves, per-step oracle trace, and the
    targets. We compare the rehydrated ``PretrainExample`` objects directly
    rather than the on-disk records so a script can diff in memory.

    The comparison is deliberately explicit (field by field) so the *first*
    mismatch is informative — "trees diverged at expansion k" is the signal
    the operator needs, not a single opaque ``!=``.
    """
    mismatches: List[str] = []

    new_tree = new_example.tree
    old_tree = old_example.tree
    if new_tree.num_nodes() != old_tree.num_nodes():
        mismatches.append(
            f"num_nodes: new={new_tree.num_nodes()} old={old_tree.num_nodes()}"
        )
    # Expansion order is the time axis of the stopping problem — compare it
    # node by node so we can name the first index that forks.
    new_nodes = list(new_tree.iter_nodes())
    old_nodes = list(old_tree.iter_nodes())
    for index, (new_node, old_node) in enumerate(zip(new_nodes, old_nodes)):
        if new_node.incoming_move_uci != old_node.incoming_move_uci:
            mismatches.append(
                f"node[{index}].incoming_move: new={new_node.incoming_move_uci!r} "
                f"old={old_node.incoming_move_uci!r}"
            )
            break
        if new_node.fen != old_node.fen:
            mismatches.append(f"node[{index}].fen: new={new_node.fen!r} old={old_node.fen!r}")
            break
        if new_node.parent_id != old_node.parent_id:
            mismatches.append(
                f"node[{index}].parent_id: new={new_node.parent_id} old={old_node.parent_id}"
            )
            break

    if list(new_example.oracle_best_move_trace) != list(old_example.oracle_best_move_trace):
        first = _first_diff_index(
            new_example.oracle_best_move_trace, old_example.oracle_best_move_trace
        )
        mismatches.append(f"oracle_best_move_trace first diff at step {first}")
    if not _close_nested(new_example.oracle_root_q_trace, old_example.oracle_root_q_trace):
        mismatches.append("oracle_root_q_trace differs")
    if not _close_seq(new_example.node_target_values, old_example.node_target_values):
        mismatches.append("node_target_values differ")
    if not _close_seq(new_example.value_gap, old_example.value_gap):
        mismatches.append("value_gap differs")

    return mismatches


def _first_diff_index(left: Sequence[Any], right: Sequence[Any]) -> int:
    """Index of the first differing element (or len if one is a prefix)."""
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return min(len(left), len(right))


def _close_seq(left: Sequence[float], right: Sequence[float], tol: float = 1e-6) -> bool:
    """Length + elementwise approximate equality for a flat float sequence."""
    if len(left) != len(right):
        return False
    return all(_close(a, b, tol) for a, b in zip(left, right))


def _close_nested(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]], tol: float = 1e-6) -> bool:
    """Approximate equality for a 2-D float sequence (e.g. the Q trace)."""
    if len(left) != len(right):
        return False
    return all(_close_seq(a, b, tol) for a, b in zip(left, right))


def _close(a: float, b: float, tol: float) -> bool:
    """NaN-aware approximate scalar equality (NaN == NaN here, by design)."""
    import math

    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return abs(float(a) - float(b)) <= tol
