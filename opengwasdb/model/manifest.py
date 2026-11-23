"""Store manifest model.

The model itself -- `StoreManifest` and `RegionsEntry`, and reading a manifest
into them -- is the Store Reader's (`opengwasdb.store_reader.manifest`, #464),
re-exported here because opengwasdb-stores imports `StoreManifest` from this
path. `normalise_coverage`, which the writers that edit a manifest as JSON call,
is this module's.
"""

from __future__ import annotations

from typing import Any

from opengwasdb.store_reader.enums import AssociationCoverage
from opengwasdb.store_reader.manifest import StoreManifest as StoreManifest


def normalise_coverage(data: dict[str, Any]) -> dict[str, Any]:
    """Rewrite a raw manifest dict's `association_coverage` to the written value.

    The writers that edit `manifest.json` as JSON -- the converter, the restamp,
    the Provenance Amendment, the variant-index build -- never construct a
    `StoreManifest`, so `AssociationCoverage`'s legacy read never runs on their
    path: a source saying `full` was republished still saying `full`. Nothing
    writes the retired spelling (issue #298), so each of them normalises through
    here, and the dict is returned for chaining.

    A manifest with no `association_coverage` is left as it is: the field is
    required (spec §2), and inventing one would hide a release that never
    declared it.
    """
    if "association_coverage" in data:
        data["association_coverage"] = AssociationCoverage(
            data["association_coverage"]
        ).value
    return data


