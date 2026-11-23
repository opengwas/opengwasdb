"""Layout-independent query API.

The query facade and result resolution are the Store Reader's
(`opengwasdb.store_reader.facade`, `.resolve`, #464). This package keeps the
names opengwasdb's CLI, ogdb-api and opengwasdb-stores import, and opens a
release the way opengwasdb does: with its reader of the 0.2.x Store Variant
Table (`opengwasdb.variants.axis.open_legacy_variant_table`).
"""

from __future__ import annotations

from pathlib import Path

from opengwasdb.store.open import open_store
from opengwasdb.store_reader.facade import (
    HybridStoreQuery,
    RaggedStoreQuery,
    StoreQuery,
    query_release,
)
from opengwasdb.store_reader.resolve import resolve_rows
from opengwasdb.store_reader.stats import log10_p_two_sided

__all__ = ["StoreQuery", "log10_p_two_sided", "query_store", "resolve_rows"]


def query_store(path: str | Path) -> StoreQuery | RaggedStoreQuery | HybridStoreQuery:
    """Open a store and return the layout-independent query facade."""
    return query_release(open_store(path))
