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
