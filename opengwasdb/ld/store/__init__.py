"""Opening, reading, writing and validating the Zarr arrays of an LD Store (layer 1; ogld#4).

The store layer knows Zarr and the codes; nothing above it does. Queries read through the
`opengwasdb.ld.query.reader.Reader` interface, which `StoreReader` satisfies; the builder writes
through
`StoreWriter`, and `validate` runs the spec's
named rules. `VariantTable` is the reader's variant half — the Variant Set's rows and the ALID
lookup — which any reader composes (#94). It imports `opengwasdb.ld.format` and
`opengwasdb.ld.identity` only.
"""

from opengwasdb.ld.store.assembly import Cut
from opengwasdb.ld.store.reader import StoreReader
from opengwasdb.ld.store.tables import Partners, Variants
from opengwasdb.ld.store.validation import Finding, ValidationFailed, validate
from opengwasdb.ld.store.variants import VariantTable
from opengwasdb.ld.store.writer import StoreWriter

__all__ = [
    "Cut",
    "Finding",
    "Partners",
    "StoreReader",
    "StoreWriter",
    "ValidationFailed",
    "VariantTable",
    "Variants",
    "validate",
]
