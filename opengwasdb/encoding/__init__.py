"""Statistic array encodings: the plan, the codec, and the decoded planes.

ADR 0037 / issue #119. `StoreEncoding` is decided once at build time from
measured properties of the data and recorded in `manifest.json`; `StoreCodec`
is the only place a store's stored bytes become physical values.
"""

from opengwasdb.encoding.codec import (
    Z_OVERFLOW_INDEX,
    Z_OVERFLOW_VALUE,
    StoreCodec,
    ZOverflowBuilder,
    ZOverflowTable,
    positions_at,
    positions_flat,
    positions_pairs,
    positions_row_band,
    positions_rows_cols,
)
from opengwasdb.encoding.plan import (
    DEFAULT_Z_SCALE,
    ENCODING_VERSION,
    Z_MISSING,
    Z_OVERFLOW,
    EncodingMeasurements,
    SeEncoding,
    StoreEncoding,
    UnsupportedEncoding,
    ZEncoding,
)
from opengwasdb.encoding.planes import DenseZPlane

__all__ = [
    "DEFAULT_Z_SCALE",
    "ENCODING_VERSION",
    "DenseZPlane",
    "EncodingMeasurements",
    "SeEncoding",
    "StoreCodec",
    "StoreEncoding",
    "UnsupportedEncoding",
    "ZEncoding",
    "ZOverflowBuilder",
    "ZOverflowTable",
    "Z_MISSING",
    "Z_OVERFLOW",
    "Z_OVERFLOW_INDEX",
    "Z_OVERFLOW_VALUE",
    "positions_at",
    "positions_flat",
    "positions_pairs",
    "positions_row_band",
    "positions_rows_cols",
]
