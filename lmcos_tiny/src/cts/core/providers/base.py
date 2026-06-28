"""Abstract provider interface that any tree-expansion backend implements.

A provider knows how to produce, for a given FEN, the per-node feature dict
and the list of child ``ExpansionChild`` records (priors, WDLs, terminal
flags). The rest of CTS only ever calls into this interface — generation,
oracles, and tests can swap in test stubs or alternative engines without
touching the consumers. Concrete production implementation is
``cts.core.providers.lc0.Lc0DirectEvalProvider``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, Optional, Sequence

from ..tree import ExpansionChild


class TreeExpansionProvider(ABC):
    """Abstract interface for any backend that can expand a FEN into children.

    Implementations own engine state and translate engine output into
    ``ExpansionChild`` records carrying priors, values, and WDL features.
    Everything in CTS that builds trees drives expansion strictly through
    this interface.
    """

    @abstractmethod
    def root_features(self, fen: str) -> Mapping[str, float]:
        """Return the per-node scalar feature dict for the root FEN."""
        raise NotImplementedError

    @abstractmethod
    def expand_node(
        self,
        fen: str,
        depth: int,
        max_children: Optional[int] = None,
    ) -> Sequence[ExpansionChild]:
        """Produce the children of ``fen``. Backends may cap to ``max_children``."""
        raise NotImplementedError

    def root_metadata(self, fen: str) -> Mapping[str, Any]:
        """Optional free-form metadata stamped onto the root node."""
        return {}

    def provider_metadata(self) -> Mapping[str, Any]:
        """Optional free-form metadata about the provider (engine name, version, etc.)."""
        return {}

    def clear_caches(self) -> None:
        """Release any per-position caches the provider holds. Default: no-op."""
        pass
