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
            priority) replicates the lc0-UCI quirks bit-for-bit — priors rounded
            to the verbose-move-stats text resolution (see
            :func:`quantize_prior_to_uci_text_resolution`) and lc0's bare-FEN
            history fill (:func:`~.encoding.history_fill_for`). ``True`` keeps
            the net's full-precision priors and the canonical history fill, and
            requires re-running the A0 oracle-direction checks (re-baseline).
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

        Structure (the batching skeleton; the forward pass is the Phase-2 TODO):

        1. partition ``fens`` into cache hits and unique misses;
        2. encode the misses (``encoding.encode_board_planes_batch``);
        3. run the net on the encoded batch in ``max_batch_size`` chunks;
        4. decode policy/value/WDL heads into ``PositionEval`` per FEN;
        5. populate the cache and return results in input order.
        """
        missing = [fen for fen in dict.fromkeys(fens) if fen not in self._eval_cache]
        if missing:
            for fen, position_eval in zip(missing, self._evaluate_uncached(missing)):
                self._eval_cache[fen] = position_eval
        return [self._eval_cache[fen] for fen in fens]

    def _evaluate_uncached(self, fens: Sequence[str]) -> List[PositionEval]:
        """Run the net over a list of (already-unique, uncached) FENs.

        TODO(Phase 2.1-2.3): the real forward pass. Sketch::

            import numpy as np                      # lazy import
            from .encoding import encode_board_planes_batch, history_fill_for

            self._ensure_model_loaded()
            history_fill = history_fill_for(self._re_baseline)
            results: List[PositionEval] = []
            for start in range(0, len(fens), self._max_batch_size):
                chunk = fens[start : start + self._max_batch_size]
                planes = encode_board_planes_batch(chunk, history_fill=history_fill)  # [B,112,8,8]
                policy_logits, value, wdl = self._forward(planes) # net heads
                for i, fen in enumerate(chunk):
                    priors = self._policy_logits_to_move_priors(fen, policy_logits[i])
                    results.append(PositionEval(
                        priors=priors,
                        value=float(value[i]),
                        wdl=tuple(float(x) for x in wdl[i]),
                    ))
            return results

        Notes:
          - ``priors`` are RAW policy-head scores per legal ``move_uci`` (the
            search loop normalizes them) — emit one entry per legal move only,
            mapping the 1858-wide lc0 policy vector onto legal UCIs (lczerolens
            exposes this move<->index mapping; with ONNX it must be reproduced).
          - ``value``/``wdl`` are side-to-move perspective and ``wdl`` sums to 1.
          - fp32 keeps L2 numeric parity tight (report §2, §9).
        """
        raise NotImplementedError(
            "NetEvaluator._evaluate_uncached is a Phase-2 scaffold; implement "
            "the batched forward pass (encode -> net -> decode heads) and gate "
            "with the T-fwd numeric-parity test against lc0."
        )

    def _ensure_model_loaded(self) -> None:
        """Lazily load the net via the selected backend (heavy import inside).

        TODO(Phase 2.1): for ``LCZEROLENS``::

            from lczerolens import LczeroModel       # lazy import
            self._model = LczeroModel.from_path(self._weights_path).to(self._device)

        for ``ONNX``::

            import onnxruntime as ort                # lazy import
            # (export the .pb.gz via `lc0 leela2onnx` first, then:)
            self._model = ort.InferenceSession(onnx_path, providers=[...])
        """
        if self._model is not None:
            return
        raise NotImplementedError(
            "NetEvaluator._ensure_model_loaded is a Phase-2 scaffold; load the "
            f"net for backend={self._backend.value!r} with a lazy import."
        )

    def _policy_logits_to_move_priors(self, fen: str, policy_logits) -> Dict[str, float]:
        """Map the net's policy vector onto a raw prior per legal UCI move.

        TODO(Phase 2.2): use the backend's move<->policy-index mapping
        (lczerolens provides it; ONNX requires reproducing lc0's index table)
        and emit one raw score per legal move. Return RAW scores; normalization
        happens in the search loop via ``normalize_prior_scores``.

        When ``re_baseline=False`` (default), apply
        :func:`quantize_prior_to_uci_text_resolution` to each per-move policy
        probability before returning, reproducing the lc0-UCI text quirk so
        trees match the baseline bit-for-bit::

            if not self._re_baseline:
                priors = {m: quantize_prior_to_uci_text_resolution(p)
                          for m, p in priors.items()}
        """
        raise NotImplementedError(
            "NetEvaluator._policy_logits_to_move_priors is a Phase-2 scaffold."
        )

    def clear_caches(self) -> None:
        self._eval_cache.clear()


__all__ = ["NetBackend", "NetEvaluator", "quantize_prior_to_uci_text_resolution"]
