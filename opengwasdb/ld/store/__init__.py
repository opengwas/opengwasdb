"""Opening, reading, writing and validating the Zarr arrays of an LD Store (layer 1; ogld#4).

The store layer knows Zarr and the codes; nothing above it does. Queries read through
`StoreReader`, the builder writes through `StoreWriter`, and `validate` runs the spec's
named rules. It imports `ogld.format` and `ogld.identity` only.
"""

from ogld.store.reader import StoreReader
from ogld.store.tables import Partners, Variants
from ogld.store.validation import Finding, ValidationFailed, validate
from ogld.store.writer import StoreWriter

__all__ = [
    "Finding",
    "Partners",
    "StoreReader",
    "StoreWriter",
    "ValidationFailed",
    "Variants",
    "validate",
]
