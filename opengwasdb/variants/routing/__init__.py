"""Variant Routing: which Variant Index place each source coordinate reaches (ADR 0071).

One module behind the builders' two passes. `route_variants` reads the
sources once (Pass 1), places each coordinate on GRCh38 -- through the
hg19->hg38 lift, or through a Variant Reference's source keys -- and names
each routed ALID. Its `VariantRouting.table` is the fork-safe lookup a Pass 2
worker routes one Analysis's associations through. `place_sites` is the one
lift, which Pass 1 and the Hybrid fold both use, and `extract_variant_reference`
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
from opengwasdb.variants.routing.lift import SitePlacement, place_sites
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
    Duplicates,
    Placed,
    RoutedColumn,
    RouteTable,
    Unmatched,
    collapse_duplicates,
    last_wins,
)
from opengwasdb.variants.routing.union import Pass1Stats, UnionOptions

__all__ = [
    "Admit",
    "AssociationBatch",
    "Duplicates",
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
    "SitePlacement",
    "SourceKey",
    "UnionOptions",
    "Unmatched",
    "VariantReferenceExtraction",
    "VariantRouting",
    "collapse_duplicates",
    "extract_variant_reference",
    "last_wins",
    "place_sites",
    "route_variants",
]
