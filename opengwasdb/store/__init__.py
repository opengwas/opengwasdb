"""Store opening, the array-creation seam, and validation.

The names from :mod:`opengwasdb.store.open` are resolved lazily (PEP 562)
rather than imported eagerly.  The array-creation seam lives in this package,
and :mod:`opengwasdb.encoding.planes` imports it; ``open`` imports
:mod:`opengwasdb.model.manifest`, which imports :mod:`opengwasdb.encoding`.
Importing ``open`` eagerly here would therefore close a cycle that starts at
``opengwasdb.encoding`` before ``StoreManifest`` exists.  Each name below is
imported on first access instead, and behaves exactly as before.

The seam itself is imported by its users as :mod:`opengwasdb.store.arrays`
(``create_array``, ``ArrayRole``, ``compressor``, ...); re-exporting it here
would add nothing but another place to keep in step.
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "CURRENT_FORMAT_VERSION",
    "PRE_RESET_FORMAT_VERSIONS",
    "SUPPORTED_FORMAT_VERSIONS",
    "MalformedFormatVersion",
    "OpenGWASDBStore",
    "StagedRelease",
    "UnsupportedFormatVersion",
    "check_format_version",
    "check_writable_format_version",
    "open_store",
    "parse_format_version",
    "split_format_version",
]

#: Same names, as a set for the lazy resolver below.  A set membership test,
#: not a second list: `__all__` above stays the one place the surface is named.
_OPEN_EXPORTS = frozenset(__all__)


def __getattr__(name: str) -> Any:
    """Resolve an ``open`` re-export on first use (see the module docstring)."""
    if name in _OPEN_EXPORTS:
        return getattr(importlib.import_module("opengwasdb.store.open"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
