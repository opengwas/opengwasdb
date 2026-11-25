"""Store opening, the array-creation seam, and validation.

The names below are resolved lazily (`opengwasdb.lazy_exports`, PEP 562) rather
than imported eagerly: the package's introspection surface is what it was with
eager imports, and importing a submodule -- the array-creation seam,
:mod:`opengwasdb.store.arrays`, which the encoding imports -- does not drag in
:mod:`opengwasdb.store.open` and the builders behind it. The format-version
tables and the read check are the Store Reader's
(`opengwasdb.store_reader.format`, #464); opening and staging a release is
:mod:`opengwasdb.store.open`'s.

The seam itself is imported by its users as :mod:`opengwasdb.store.arrays`
(``create_array``, ``ArrayRole``, ``compressor``, ...); re-exporting it here
would add nothing but another place to keep in step.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from opengwasdb.lazy_exports import exported_names, lazy_hooks

if TYPE_CHECKING:
    from opengwasdb.store.open import DestinationExistsError as DestinationExistsError
    from opengwasdb.store.open import OpenGWASDBStore as OpenGWASDBStore
    from opengwasdb.store.open import StagedRelease as StagedRelease
    from opengwasdb.store.open import check_format_version_stamp as check_format_version_stamp
    from opengwasdb.store.open import check_writable_format_version as check_writable_format_version
    from opengwasdb.store.open import check_writable_zarr_layout as check_writable_zarr_layout
    from opengwasdb.store.open import completed_format_version as completed_format_version
    from opengwasdb.store.open import destination_lock as destination_lock
    from opengwasdb.store.open import directory_lock as directory_lock
    from opengwasdb.store.open import open_store as open_store
    from opengwasdb.store.open import staged_named_group as staged_named_group
    from opengwasdb.store_reader.format import CURRENT_FORMAT_VERSION as CURRENT_FORMAT_VERSION
    from opengwasdb.store_reader.format import (
        PRE_RESET_FORMAT_VERSIONS as PRE_RESET_FORMAT_VERSIONS,
    )
    from opengwasdb.store_reader.format import (
        SUPPORTED_FORMAT_VERSIONS as SUPPORTED_FORMAT_VERSIONS,
    )
    from opengwasdb.store_reader.format import WRITTEN_FORMAT_VERSIONS as WRITTEN_FORMAT_VERSIONS
    from opengwasdb.store_reader.format import ZARR_FORMAT_BY_SERIES as ZARR_FORMAT_BY_SERIES
    from opengwasdb.store_reader.format import MalformedFormatVersion as MalformedFormatVersion
    from opengwasdb.store_reader.format import UnsupportedFormatVersion as UnsupportedFormatVersion
    from opengwasdb.store_reader.format import check_format_version as check_format_version
    from opengwasdb.store_reader.format import format_version_for as format_version_for
    from opengwasdb.store_reader.format import parse_format_version as parse_format_version
    from opengwasdb.store_reader.format import split_format_version as split_format_version
    from opengwasdb.store_reader.format import zarr_format_for_version as zarr_format_for_version

_EXPORTS = {
    "opengwasdb.store_reader.format": (
        "CURRENT_FORMAT_VERSION",
        "PRE_RESET_FORMAT_VERSIONS",
        "SUPPORTED_FORMAT_VERSIONS",
        "WRITTEN_FORMAT_VERSIONS",
        "ZARR_FORMAT_BY_SERIES",
        "MalformedFormatVersion",
        "UnsupportedFormatVersion",
        "check_format_version",
        "format_version_for",
        "parse_format_version",
        "split_format_version",
        "zarr_format_for_version",
    ),
    "opengwasdb.store.open": (
        "DestinationExistsError",
        "OpenGWASDBStore",
        "StagedRelease",
        "check_format_version_stamp",
        "check_writable_format_version",
        "check_writable_zarr_layout",
        "completed_format_version",
        "destination_lock",
        "directory_lock",
        "open_store",
        "staged_named_group",
    ),
}

__all__ = exported_names(_EXPORTS)
__getattr__, __dir__ = lazy_hooks(__name__, _EXPORTS)
