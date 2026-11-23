"""OpenGWASDB storage and query engine.

Every name here is imported on first use (`opengwasdb.lazy_exports`, #464):
importing any `opengwasdb.*` module runs this file first, and an eager import of
`store.open` here made every one of them, the Store Reader's included, load the
builders, pandas, scipy and pysam.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from opengwasdb.lazy_exports import exported_names, lazy_hooks

if TYPE_CHECKING:
    from opengwasdb.store.open import OpenGWASDBStore as OpenGWASDBStore
    from opengwasdb.store.open import open_store as open_store
    from opengwasdb.store_reader.manifest import StoreManifest as StoreManifest
    from opengwasdb.version import __version__ as __version__

_EXPORTS = {
    "opengwasdb.store_reader.manifest": ("StoreManifest",),
    "opengwasdb.store.open": ("OpenGWASDBStore", "open_store"),
    "opengwasdb.version": ("__version__",),
}

__all__ = exported_names(_EXPORTS)
__getattr__, __dir__ = lazy_hooks(__name__, _EXPORTS)
