"""In-process batched lc0 net evaluator (Phase 2 scaffold, report §2 L2 / T-fwd).

This is the evaluator that delivers the actual speedup: instead of querying lc0
one position at a time over a UCI text pipe, it runs the lc0 network *in
process* on a batched input tensor and returns priors + value + WDL for the
whole batch in one forward pass (report §1.2-1.3).

Two backends are contemplated (report §6):

  - **lczerolens + PyTorch (preferred):** loads the Leela ``.pb.gz`` net into a
    PyTorch module and ships the lc0-faithful board encoding, so encoding and
    weights come from one lc0-aligned source (least parity risk for T-enc/T-fwd).
  - **ONNX Runtime + leela2onnx:** ``leela2onnx`` exports the net to ONNX;
    ``onnxruntime`` runs the same graph on CPU or GPU (helps the CPU-fleet
    lane). Encoding must then be reproduced ourselves (see ``encoding.py``).

The heavy libraries (lczerolens / torch-for-inference / onnxruntime / numpy)
are **lazy-imported inside methods**, so ``import cts.data.batched_gen`` never
requires them. The class/signature is complete; the forward pass is a clearly
marked TODO. Net weights: reuse ``t1-256x10-distilled-swa-2432500.pb.gz`` so L2
compares against the same network the lc0-UCI baseline uses (report §6).
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional, Sequence

from .evaluator import Evaluator, PositionEval


class NetBackend(str, Enum):
    """Which inference backend ``NetEvaluator`` loads the net into."""

    LCZEROLENS = "lczerolens"  # PyTorch via lczerolens (preferred, report §6)
    ONNX = "onnx"  # onnxruntime over a leela2onnx export


def quantize_prior_to_uci_text_resolution(probability: float) -> float:
    """Round a policy prior to lc0's verbose-move-stats text resolution.

    The lc0-UCI pipeline reads priors from text like ``(P: 12.34%)`` and
    reparses them as ``float(text) / 100`` ([parse_no_search_analysis]), which
    silently discards all precision below 1e-4 — a quirk of the UCI text path,
    not the network. With ``re_baseline=False`` we reproduce it exactly so the
    in-process net's priors match the lc0-UCI trees bit-for-bit; with
    ``re_baseline=True`` we keep the net's full-precision policy.

    NOTE: exact parity also depends on lc0's C++ ``printf`` rounding mode
    matching Python's; T-fwd must confirm against captured lc0 text.
    """
    return float(f"{probability * 100.0:.2f}") / 100.0


def quantize_wdl_to_uci_permille(
    win: float, draw: float, loss: float
) -> "tuple[float, float, float]":
    """Snap a WDL triple to lc0's integer per-mille UCI grid, then renormalize.

    lc0's UCI value engine reports WDL as integer per-mille (``wdl W D L``,
    0-1000), which the baseline reparses and normalizes by their sum
    ([parse_root_value_features_from_lines] -> [value_features_from_wdl]). So
    the in-process net's full-precision softmax must be rounded to the same
    1e-3 grid for ``re_baseline=False`` parity. Rounding to per-mille also
    absorbs the sub-1e-3 numeric gap between the ONNX/torch backend and lc0's
    CUDA kernels, which is what makes discrete-trajectory parity attainable
    (report §2 "honest scope"). Residual forks can only occur when a component
    straddles a per-mille boundary.
    """
    w, d, l = round(win * 1000.0), round(draw * 1000.0), round(loss * 1000.0)
    total = w + d + l
    if total <= 0:
        raise ValueError("WDL must have positive mass after per-mille rounding.")
    return (w / total, d / total, l / total)


# lc0's default search-time PolicyTemperature (UCI ``PolicyTemperature``). The
# verbose-move-stats prior P the lc0-UCI pipeline parses is the policy head
# softmaxed at this temperature, so re_baseline=False applies it (empirically
# reproduces lc0's reported P to ~1e-4); re_baseline=True uses 1.0 (raw policy).
LC0_DEFAULT_POLICY_TEMPERATURE = 1.359


class NetEvaluator(Evaluator):
    """Batched, in-process lc0 net evaluator.

    Args:
        weights_path: path to the Leela ``.pb.gz`` net (same weights as the
            lc0-UCI baseline for a fair L2 comparison).
        backend: which inference stack to load the net into.
        device: torch/onnx device string (``"cpu"``, ``"cuda"``, ...). fp32 is
            the parity-safe default (report §2 / §9).
        max_batch_size: largest forward-pass batch; the batched search loop may
            hand more FENs than this in one ``evaluate`` call, so they are
            chunked to bound device memory.
        re_baseline: the single fidelity switch (report §2). ``False`` (default,
            priority) replicates the lc0-UCI pipeline bit-for-bit: value/WDL is
            lc0's 1-ply best-child ``valuehead`` minimax snapped to the per-mille
            grid, priors are the policy head at ``PolicyTemperature`` 1.359
            snapped to the 1e-4 grid, history is fen_only. ``True`` uses the raw
            root value head + temperature 1.0 (no 1-ply lookahead, ~30x cheaper,
            a cleaner training target) and requires re-running the A0
            oracle-direction checks (re-baseline).
    """

    def __init__(
        self,
        weights_path: str,
        *,
        backend: NetBackend = NetBackend.LCZEROLENS,
        device: str = "cpu",
        max_batch_size: int = 1024,
        re_baseline: bool = False,
    ) -> None:
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive.")
        self._weights_path = weights_path
        self._backend = backend
        self._device = device
        self._max_batch_size = max_batch_size
        self._re_baseline = re_baseline
        # The loaded model is built lazily on first ``evaluate`` so constructing
        # a NetEvaluator (e.g. in config wiring) never imports the heavy libs.
        self._model = None
        # Per-FEN memoization so a position queried as a child then re-queried
        # as a parent costs one forward pass — mirrors the lc0 provider caches
        # and keeps the batched loop's per-FEN read counts predictable.
        self._eval_cache: Dict[str, PositionEval] = {}

    def evaluate(self, fens: Sequence[str]) -> List[PositionEval]:
        """Return one ``PositionEval`` per FEN, scoring cache-misses in batches.

        1. partition ``fens`` into cache hits and unique misses;
        2. run the net on the misses in ``max_batch_size`` chunks (one forward
           pass each — the speedup) and decode the WDL + policy heads;
        3. populate the cache and return results in input order.
        """
        missing = [fen for fen in dict.fromkeys(fens) if fen not in self._eval_cache]
        if missing:
            for fen, position_eval in zip(missing, self._evaluate_uncached(missing)):
                self._eval_cache[fen] = position_eval
        return [self._eval_cache[fen] for fen in fens]

    def _evaluate_uncached(self, fens: Sequence[str]) -> List[PositionEval]:
        """Faithfully replicate lc0's ``valuehead`` per position.

        lc0's ``valuehead`` is NOT the raw root value head — it does a **1-ply
        best-child minimax** (confirmed: its output equals
        ``max over legal children of (-raw_value(child))`` exactly). So for
        ``re_baseline=False`` a position's value/WDL is that 1-ply backup,
        snapped to the per-mille grid; priors are the policy head softmaxed at
        lc0's ``PolicyTemperature`` (1.359) and snapped to the 1e-4 grid.
        Children are built by pushing each move (1 ply of history) — matching how
        lc0's valuehead search evaluates them. Terminal positions (no legal
        moves) fall back to the raw root value head. With ``re_baseline=True``
        the temperature is 1.0 and no grid-snapping is applied.

        Batching: every position's children are pooled into one forward pass
        (chunked by ``max_batch_size``) — the speedup over per-child UCI calls.
        ``value``/``wdl`` are side-to-move perspective; ``wdl`` sums to 1.
        """
        from lczerolens import LczeroBoard  # lazy

        self._ensure_model_loaded()
        temperature = 1.0 if self._re_baseline else LC0_DEFAULT_POLICY_TEMPERATURE

        root_boards = [LczeroBoard(fen) for fen in fens]
        root_policy, root_wdl = self._forward_heads(root_boards, want_policy=True)

        # re_baseline=True: raw root value head, no 1-ply lookahead (~30x cheaper).
        if self._re_baseline:
            results: List[PositionEval] = []
            for i, board in enumerate(root_boards):
                priors = self._policy_logits_to_move_priors(
                    board, root_policy[i], temperature
                )
                win, draw, loss = (float(x) for x in root_wdl[i])
                results.append(
                    PositionEval(priors=priors, value=win - loss, wdl=(win, draw, loss))
                )
            return results

        # re_baseline=False: faithful lc0 1-ply best-child valuehead minimax.
        legal_per_fen: List[list] = []
        child_boards: list = []
        for board in root_boards:
            moves = list(board.legal_moves)
            legal_per_fen.append(moves)
            for move in moves:
                child = board.copy()
                child.push(move)  # 1-ply history, as lc0's valuehead search does
                child_boards.append(child)

        _, child_wdl = self._forward_heads(child_boards, want_policy=False)

        results = []
        child_pos = 0
        for i, board in enumerate(root_boards):
            moves = legal_per_fen[i]
            priors = self._policy_logits_to_move_priors(board, root_policy[i], temperature)
            if not moves:
                # Terminal: no children to back up; use the raw root value head.
                win, draw, loss = (float(x) for x in root_wdl[i])
            else:
                # 1-ply minimax: choose the child worst for the opponent; the
                # node value (our perspective) is that child's WDL, flipped.
                best_value = None
                win = draw = loss = 0.0
                for j in range(len(moves)):
                    child_win, child_draw, child_loss = (
                        float(x) for x in child_wdl[child_pos + j]
                    )
                    our_win, our_draw, our_loss = child_loss, child_draw, child_win
                    value = our_win - our_loss
                    if best_value is None or value > best_value:
                        best_value = value
                        win, draw, loss = our_win, our_draw, our_loss
                child_pos += len(moves)
            if not self._re_baseline:
                win, draw, loss = quantize_wdl_to_uci_permille(win, draw, loss)
            results.append(
                PositionEval(priors=priors, value=win - loss, wdl=(win, draw, loss))
            )
        return results

    def _forward_heads(self, boards, *, want_policy):
        """Run the net over ``boards`` in ``max_batch_size`` chunks.

        Returns ``(policy, wdl)`` concatenated across chunks; ``policy`` is
        ``None`` when ``want_policy`` is False (children only need WDL), and both
        are ``None`` for an empty board list.
        """
        import torch  # lazy
        from lczerolens.board import InputEncoding  # lazy

        if not boards:
            return None, None
        # lc0's HistoryFill=fen_only (the default the pipeline inherited) repeats
        # the current position into the history planes for a bare FEN; the
        # matching lczerolens encoding is REPEATED (confirmed: reproduces lc0's
        # policy + valuehead on ply-15-75 positions). The genuine game-start
        # position is special-cased by lc0 and would not match, but our data is
        # always midgame, so this is correct for every FEN we generate on.
        encoding = InputEncoding.INPUT_CLASSICAL_112_PLANE_REPEATED
        policy_chunks = []
        wdl_chunks = []
        for start in range(0, len(boards), self._max_batch_size):
            chunk = boards[start : start + self._max_batch_size]
            planes = torch.stack(
                [board.to_input_tensor(input_encoding=encoding) for board in chunk]
            )
            with torch.no_grad():
                out = self._model(planes)
            wdl_chunks.append(out["wdl"].detach())
            if want_policy:
                policy_chunks.append(out["policy"].detach())
        wdl = torch.cat(wdl_chunks, dim=0)
        policy = torch.cat(policy_chunks, dim=0) if want_policy else None
        return policy, wdl

    def _ensure_model_loaded(self) -> None:
        """Lazily load the net via the selected backend (heavy import inside)."""
        if self._model is not None:
            return
        if self._backend == NetBackend.LCZEROLENS:
            from lczerolens import LczeroModel  # lazy import

            if not self._weights_path.endswith(".onnx"):
                raise ValueError(
                    "LCZEROLENS backend needs an .onnx net. Convert the Leela "
                    ".pb.gz once with `lc0 leela2onnx --input=<pb.gz> "
                    "--output=<onnx>` and pass the .onnx path."
                )
            model = LczeroModel.from_path(self._weights_path)
            # eval() is essential: the onnx2torch graph carries BatchNorm, and in
            # train mode it normalizes with per-batch statistics, making the
            # output depend on which other positions share the batch (and wrong).
            # eval() switches to the net's running statistics -> deterministic,
            # batch-independent, and matching lc0.
            model.eval()
            self._model = model.to(self._device) if self._device != "cpu" else model
            return
        raise NotImplementedError(
            f"NetEvaluator backend {self._backend.value!r} is not yet wired; "
            "use NetBackend.LCZEROLENS."
        )

    def _policy_logits_to_move_priors(
        self, board, policy_logits, temperature: float
    ) -> Dict[str, float]:
        """Map the net's 1858-wide policy onto a prior per legal UCI move.

        lc0's verbose-move-stats prior ``P`` is the policy head softmaxed over
        *legal* moves at ``PolicyTemperature``, so we gather each legal move's
        logit via lczerolens's ``board.encode_move`` index map, softmax over just
        those at ``temperature``, and key by UCI. With ``re_baseline=False`` each
        probability is then snapped to the 2-dp-% text grid
        (:func:`quantize_prior_to_uci_text_resolution`) to match the lc0-UCI
        baseline. Returned values are the (possibly quantized) per-move
        probabilities; the search loop renormalizes over the children it keeps
        via ``normalize_prior_scores``.
        """
        import numpy as np  # lazy

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return {}
        # encode_move needs the side to move; the policy index is perspective-
        # relative (lc0 flips the board for black), so pass board.turn.
        us = board.turn
        logits = np.array(
            [float(policy_logits[board.encode_move(move, us)]) for move in legal_moves],
            dtype=np.float64,
        )
        # Softmax over legal moves only, at lc0's PolicyTemperature (mirrors P).
        scaled = logits / temperature
        weights = np.exp(scaled - scaled.max())
        probabilities = weights / weights.sum()
        priors = {move.uci(): float(p) for move, p in zip(legal_moves, probabilities)}
        if not self._re_baseline:
            priors = {
                uci: quantize_prior_to_uci_text_resolution(p)
                for uci, p in priors.items()
            }
        return priors

    def clear_caches(self) -> None:
        self._eval_cache.clear()


__all__ = [
    "NetBackend",
    "NetEvaluator",
    "quantize_prior_to_uci_text_resolution",
    "quantize_wdl_to_uci_permille",
]
