"""Opening, reading, writing and validating the Zarr arrays of an LD Store (layer 1; ogld#4).

The store layer knows Zarr and the codes; nothing above it does. Queries read through the
`ogld.query.reader.Reader` interface, which `StoreReader` satisfies; the builder writes through
`StoreWriter`, and `validate` runs the spec's
named rules. `VariantTable` is the reader's variant half — the Variant Set's rows and the ALID
lookup — which any reader composes (#94). It imports `ogld.format` and `ogld.identity` only.
"""

from ogld.store.assembly import Cut
from ogld.store.reader import StoreReader
from ogld.store.tables import Partners, Variants
from ogld.store.validation import Finding, ValidationFailed, validate
from ogld.store.variants import VariantTable
from ogld.store.writer import StoreWriter

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
