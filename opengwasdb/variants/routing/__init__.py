"""Variant Routing: which Variant Index place each source coordinate reaches.

One module behind the builders' two passes. `route_variants` reads the
sources once (Pass 1), places each coordinate on GRCh38 -- through the
hg19->hg38 lift, or through a Variant Reference's source keys -- and names
each routed ALID. Its `VariantRouting.table` is the fork-safe lookup a Pass 2
worker routes one Analysis's associations through. `extract_variant_reference`
freezes a lifted union as a Variant Reference artifact.

Not imported by `opengwasdb.variants`, which the Source Readers import.
"""

from __future__ import annotations

from opengwasdb.variants.routing.extract import (
    EmptyReferenceError,
    ReverseStrandLiftError,
    VariantReferenceExtraction,
    extract_variant_reference,
)
from opengwasdb.variants.routing.keys import RoutingKey, SourceKey
from opengwasdb.variants.routing.names import RoutedNames
from opengwasdb.variants.routing.route import (
    FromReference,
    LiftedUnion,
    VariantRouting,
    route_variants,
)
from opengwasdb.variants.routing.table import (
    Admit,
    AssociationBatch,
    Placed,
    RoutedColumn,
    RouteTable,
    Unmatched,
    last_wins,
)
from opengwasdb.variants.routing.union import Pass1Stats, UnionOptions

__all__ = [
    "Admit",
    "AssociationBatch",
    "EmptyReferenceError",
    "FromReference",
    "LiftedUnion",
    "Pass1Stats",
    "Placed",
    "ReverseStrandLiftError",
    "RouteTable",
    "RoutedColumn",
    "RoutedNames",
    "RoutingKey",
    "SourceKey",
    "UnionOptions",
    "Unmatched",
    "VariantReferenceExtraction",
    "VariantRouting",
    "extract_variant_reference",
    "last_wins",
    "route_variants",
]
