"""SQLite metadata and lookup indexes."""

from __future__ import annotations

from typing import TYPE_CHECKING

from opengwasdb.lazy_exports import exported_names, lazy_hooks

if TYPE_CHECKING:
    from opengwasdb.index.sqlite import connect as connect
    from opengwasdb.index.sqlite import get_metadata as get_metadata
    from opengwasdb.index.sqlite import initialise_schema as initialise_schema
    from opengwasdb.index.sqlite import set_metadata as set_metadata
    from opengwasdb.store_reader.analyses import AnalysesIndex as AnalysesIndex

#: Each re-exported name, by the module it comes from; imported on first use
#: (`opengwasdb.lazy_exports`, #464).
_EXPORTS = {
    "opengwasdb.store_reader.analyses": (
        "AnalysesIndex",
    ),
    "opengwasdb.index.sqlite": (
        "connect",
        "get_metadata",
        "initialise_schema",
        "set_metadata",
    ),
}

__all__ = exported_names(_EXPORTS)
__getattr__, __dir__ = lazy_hooks(__name__, _EXPORTS)
