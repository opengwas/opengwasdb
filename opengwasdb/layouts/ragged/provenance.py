"""The `provenance.ragged` block every Ragged builder records.

Both Ragged builders (BESD and SSF) describe their component the same way: the
statistic arrays it stores, its SE dtype, its variant-axis sidecar format, and
-- since ADR 0060 -- whether it carries the variant-centric index.  Sharing the
one function keeps the two builds' provenance from drifting, and keeps the
`by_variant` block derived from the index that was actually written rather than
threaded through each builder.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from opengwasdb.encoding import StoreEncoding
from opengwasdb.layouts.ragged.by_variant import with_variant_index
from opengwasdb.variants.axis import (
    VARIANT_AXIS_FORMAT,
    VARIANT_TABIX_FILENAME,
    VARIANT_TABLE_FILENAME,
)


def ragged_provenance(staged_path: str | Path, encoding: StoreEncoding) -> dict[str, Any]:
    """The `provenance.ragged` block for a Ragged component written at `staged_path`."""
    return with_variant_index(
        staged_path,
        {
            "statistic_arrays": ["z", "se"],
            "se_dtype": encoding.se.dtype,
            "variant_axis": {
                "format": VARIANT_AXIS_FORMAT,
                "table": VARIANT_TABLE_FILENAME,
                "tabix_index": VARIANT_TABIX_FILENAME,
            },
        },
    )
