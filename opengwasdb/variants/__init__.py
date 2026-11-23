"""Variant identity, normalisation, and lookup helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from opengwasdb.lazy_exports import exported_names, lazy_hooks

if TYPE_CHECKING:
    from opengwasdb.store_reader.alid import is_indexable_alid as is_indexable_alid
    from opengwasdb.store_reader.alid import is_indexable_rsid as is_indexable_rsid
    from opengwasdb.store_reader.alid import parse_canonical_alid as parse_canonical_alid
    from opengwasdb.store_reader.identity import (
        VariantNormalisationError as VariantNormalisationError,
    )
    from opengwasdb.store_reader.identity import normalise_allele as normalise_allele
    from opengwasdb.store_reader.identity import normalise_chromosome as normalise_chromosome
    from opengwasdb.store_reader.variant_axis import (
        VARIANT_ALID_BYTES_FILENAME as VARIANT_ALID_BYTES_FILENAME,
    )
    from opengwasdb.store_reader.variant_axis import (
        VARIANT_ALID_ROWS_FILENAME as VARIANT_ALID_ROWS_FILENAME,
    )
    from opengwasdb.store_reader.variant_axis import (
        VARIANT_OFFSETS_FILENAME as VARIANT_OFFSETS_FILENAME,
    )
    from opengwasdb.store_reader.variant_axis import (
        VARIANT_TABIX_FILENAME as VARIANT_TABIX_FILENAME,
    )
    from opengwasdb.store_reader.variant_axis import (
        VARIANT_TABLE_FILENAME as VARIANT_TABLE_FILENAME,
    )
    from opengwasdb.store_reader.variant_axis import VariantRecord as VariantRecord
    from opengwasdb.store_reader.variant_axis import (
        variant_alid_bytes_path as variant_alid_bytes_path,
    )
    from opengwasdb.store_reader.variant_axis import (
        variant_alid_rows_path as variant_alid_rows_path,
    )
    from opengwasdb.store_reader.variant_axis import variant_offsets_path as variant_offsets_path
    from opengwasdb.store_reader.variant_axis import (
        variant_rsid_bytes_path as variant_rsid_bytes_path,
    )
    from opengwasdb.store_reader.variant_axis import (
        variant_rsid_rows_path as variant_rsid_rows_path,
    )
    from opengwasdb.store_reader.variant_axis import variant_tabix_path as variant_tabix_path
    from opengwasdb.store_reader.variant_axis import variant_table_path as variant_table_path
    from opengwasdb.variants.axis import VARIANT_AXIS_FORMAT as VARIANT_AXIS_FORMAT
    from opengwasdb.variants.axis import VariantAxis as VariantAxis
    from opengwasdb.variants.axis import iter_variant_records as iter_variant_records
    from opengwasdb.variants.axis import write_variant_axis as write_variant_axis
    from opengwasdb.variants.normalise import CanonicalVariant as CanonicalVariant
    from opengwasdb.variants.normalise import Orientation as Orientation
    from opengwasdb.variants.normalise import chromosome_sort_key as chromosome_sort_key
    from opengwasdb.variants.normalise import orient_to_canonical as orient_to_canonical
    from opengwasdb.variants.reference import UnmarkedArtifactError as UnmarkedArtifactError
    from opengwasdb.variants.reference import VariantReference as VariantReference
    from opengwasdb.variants.reference import read_variant_reference as read_variant_reference
    from opengwasdb.variants.reference import write_variant_reference as write_variant_reference

#: Each re-exported name, by the module it comes from; imported on first use
#: (`opengwasdb.lazy_exports`, #464).
_EXPORTS = {
    "opengwasdb.store_reader.alid": (
        "is_indexable_alid",
        "is_indexable_rsid",
        "parse_canonical_alid",
    ),
    "opengwasdb.store_reader.identity": (
        "normalise_allele",
        "normalise_chromosome",
        "VariantNormalisationError",
    ),
    "opengwasdb.variants.axis": (
        "VARIANT_AXIS_FORMAT",
        "VariantAxis",
        "iter_variant_records",
        "write_variant_axis",
    ),
    "opengwasdb.store_reader.variant_axis": (
        "VARIANT_OFFSETS_FILENAME",
        "VARIANT_TABIX_FILENAME",
        "VARIANT_TABLE_FILENAME",
        "VARIANT_ALID_BYTES_FILENAME",
        "VARIANT_ALID_ROWS_FILENAME",
        "VariantRecord",
        "variant_offsets_path",
        "variant_rsid_bytes_path",
        "variant_rsid_rows_path",
        "variant_tabix_path",
        "variant_table_path",
        "variant_alid_bytes_path",
        "variant_alid_rows_path",
    ),
    "opengwasdb.variants.normalise": (
        "CanonicalVariant",
        "Orientation",
        "chromosome_sort_key",
        "orient_to_canonical",
    ),
    "opengwasdb.variants.reference": (
        "UnmarkedArtifactError",
        "VariantReference",
        "read_variant_reference",
        "write_variant_reference",
    ),
}

__all__ = exported_names(_EXPORTS)
__getattr__, __dir__ = lazy_hooks(__name__, _EXPORTS)
