"""Provider layer — bridges the UCI/lc0 engine subprocess to the rest of CTS.

``Lc0DirectEvalProvider`` is the production provider implementation. The
``TreeExpansionProvider`` ABC it satisfies lives in ``cts.core.providers.base``
so anyone writing an alternative provider (e.g. for tests or different
engines) can import it without dragging in the lc0-specific code.

The names re-exported at this package level are the public surface most
callers want: the ABC, the concrete lc0 provider, and the engine
config/process types used to construct it.
"""

from .base import TreeExpansionProvider
from .lc0 import Lc0DirectEvalProvider
from .process import UciEngineConfig, UciEngineProcess

__all__ = [
    "Lc0DirectEvalProvider",
    "TreeExpansionProvider",
    "UciEngineConfig",
    "UciEngineProcess",
]
