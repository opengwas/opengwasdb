"""A manifest's Variant Reference, extracted from the sources its rows name.

`extract_variant_reference` takes sources, not a manifest: reading one is the
build manifest's job. The tests that start from a manifest read it here, with
the same per-release defaults the `extract-variant-reference` command passes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from opengwasdb.layouts.build_manifest import read_manifest
from opengwasdb.variants.routing import (
    UnionOptions,
    VariantReferenceExtraction,
    extract_variant_reference,
)

#: The `UnionOptions` fields a test may set by keyword.
_UNION_FIELDS = ("window_size_mb", "reduction_batch_size", "map_spill_records")


def extract_from_manifest(
    manifest: Path,
    output: Path,
    *,
    source_reader_capability: str | None = None,
    source_assembly: str | None = None,
    column_map: Any = None,
    liftover_failure_threshold: float = 0.01,
    **options: Any,
) -> VariantReferenceExtraction:
    """Extract ``manifest``'s Variant Reference to ``output``.

    ``options`` are `extract_variant_reference`'s own keywords, plus the
    `UnionOptions` fields, which are gathered into one.
    """
    union = UnionOptions(**{name: options.pop(name) for name in _UNION_FIELDS if name in options})
    rows = read_manifest(
        manifest,
        default_source_reader_capability=source_reader_capability,
        default_source_assembly=source_assembly,
        column_map=column_map,
    )
    sources = [row.source for row in rows]
    return extract_variant_reference(
        sources, output, failure_threshold=liftover_failure_threshold, union=union, **options
    )
