"""Opening, reading, writing and validating the Zarr arrays of an LD Store (layer 1; ogld#4).

The read side -- `StoreReader`, `VariantTable` and `Variants` -- is ogquery's
(`ogquery.ld.store`; opengwas/ogquery#2): queries read through the `ogquery.ld.Reader` protocol,
which `StoreReader` satisfies. opengwasdb keeps the write side: the builder writes through
`StoreWriter`, `assembly` derives a store's components, and `validate` runs the spec's named
rules. Each name is imported on first use, so importing a submodule loads none of the others.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from opengwasdb.lazy_exports import exported_names, lazy_hooks

if TYPE_CHECKING:
    from ogquery.ld.store import StoreReader as StoreReader
    from ogquery.ld.store import Variants as Variants
    from ogquery.ld.store import VariantTable as VariantTable

    from opengwasdb.ld.store.assembly import Cut as Cut
    from opengwasdb.ld.store.tables import Partners as Partners
    from opengwasdb.ld.store.validation import Finding as Finding
    from opengwasdb.ld.store.validation import ValidationFailed as ValidationFailed
    from opengwasdb.ld.store.validation import validate as validate
    from opengwasdb.ld.store.writer import StoreWriter as StoreWriter

#: Each name, by the module it comes from; imported on first use (`opengwasdb.lazy_exports`),
#: so reading one LD read name never imports the build, the writer or validation (ogquery#2).
_EXPORTS = {
    "ogquery.ld.store": (
        "StoreReader",
        "VariantTable",
        "Variants",
    ),
    "opengwasdb.ld.store.assembly": ("Cut",),
    "opengwasdb.ld.store.tables": ("Partners",),
    "opengwasdb.ld.store.validation": (
        "Finding",
        "ValidationFailed",
        "validate",
    ),
    "opengwasdb.ld.store.writer": ("StoreWriter",),
}
__all__ = exported_names(_EXPORTS)
__getattr__, __dir__ = lazy_hooks(__name__, _EXPORTS)
