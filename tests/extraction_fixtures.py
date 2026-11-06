"""A manifest's Variant Reference, extracted from the sources its rows name.

`extract_variant_reference` takes sources, not a manifest: reading one is the
build manifest's job. The tests that start from a manifest read it here, with
the same per-release defaults the `extract-variant-reference` command passes.
A test that needs an artifact no manifest yields writes one with
`write_hg38_artifact`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from opengwasdb.layouts.build_manifest import read_manifest
from opengwasdb.variants.reference import write_variant_reference
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


def write_hg38_artifact(
    path: Path, alids: Sequence[str], rsid_by_alid: Mapping[str, str]
) -> Path:
    """A Variant Reference artifact over GRCh38 sources: each ALID routed by its
    own coordinate, in its own allele order, and nothing lifted or re-oriented.
    ``rsid_by_alid`` names the ALIDs it holds a name for."""
    source_lookup = {
        (chrom, int(position), a1, a2): alid
        for alid in alids
        for chrom, position, a1, a2 in [alid.split(":")]
    }
    write_variant_reference(path, alids, source_lookup, rsid_by_alid, reoriented=())
    return path
