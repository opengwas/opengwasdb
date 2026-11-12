"""One construction of an Observed-Only Store Release's manifest, per layout.

Each layout's ``manifest.json`` is the same contract whichever builder writes
it: same layout, same completion state, and -- for the Dense and Ragged cases
here -- the same cast of fields. The Hybrid release's own manifest differs
enough to stay with the builder that writes it.

Three copies of one literal is three places for a field to be added to two of
them, which has already happened in this repository (the `eaf_orientation`
provenance block, issue #115). Each layout's coverage is a parameter, not a
constant: it is a build input since issue #298 (ADR 0064).
"""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from opengwasdb.encoding import StoreEncoding
from opengwasdb.model.enums import (
    AssociationCoverage,
    CompletionState,
    PrimaryStorageLayout,
)
from opengwasdb.model.manifest import RegionsEntry, StoreManifest
from opengwasdb.store.open import format_version_for


def observed_only_release_manifest(
    layout: PrimaryStorageLayout,
    *,
    store_id: str,
    release_id: str,
    encoding: StoreEncoding,
    association_coverage: AssociationCoverage,
    reference_assembly: str,
    provenance: dict[str, Any],
    regions: RegionsEntry | None = None,
) -> StoreManifest:
    """The manifest of an Observed-Only Store Release in ``layout`` (spec §2), declaring
    ``regions``, the region table it carries, where it carries one (spec §17), and
    stamped the lowest ``format_version`` its features use (spec §21.3)."""
    return StoreManifest(
        encoding=encoding,
        store_id=store_id,
        release_id=release_id,
        # Every build spells an indel's ALID reference first (ADR 0079).
        format_version=format_version_for(
            carries_regions=regions is not None, reference_first_indels=True
        ),
        primary_layout=layout,
        association_coverage=association_coverage,
        completion_state=CompletionState.OBSERVED_ONLY,
        reference_assembly=reference_assembly,
        created_at=datetime.now(UTC).isoformat(),
        provenance=provenance,
        regions=regions,
    )


__all__ = ["observed_only_release_manifest"]
