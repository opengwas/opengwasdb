"""Store manifest model."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from opengwasdb.encoding import StoreEncoding, UnsupportedEncoding
from opengwasdb.model.enums import (
    AssociationCoverage,
    CompletionState,
    PrimaryStorageLayout,
)


def _encoding_from_dict(data: dict[str, Any]) -> StoreEncoding:
    """The release's declared plan, or the legacy one -- but only where an
    absent declaration is what "legacy" means.

    A release at `format_version` 1.0 or above MUST declare its encoding (spec
    §6a). Falling back to the legacy plan there would decode an `int16` plane
    as `float16` and hand back plausible, wrong z-scores -- the exact failure
    an explicit declaration exists to prevent -- so a missing block is refused
    rather than guessed at.
    """
    declared = data.get("encoding")
    if declared is not None:
        return StoreEncoding.from_manifest(declared)
    # Deferred import: `opengwasdb.store.open` imports this module, and the
    # version parser belongs to the reader contract that lives there. One
    # parser, imported at call time, rather than a second copy of it here.
    from opengwasdb.store.open import parse_format_version

    major, _ = parse_format_version(str(data["format_version"]))
    if major >= 1:
        raise UnsupportedEncoding(
            f"release declares format_version={data['format_version']!r} but no `encoding` "
            "block; from format_version 1.0 the encoding is required (spec §6a), and a "
            "release that does not declare one cannot be decoded"
        )
    return StoreEncoding.legacy()


@dataclass(frozen=True)
class StoreManifest:
    """Minimal manifest required to identify and open a Store Release."""

    store_id: str
    release_id: str
    format_version: str
    primary_layout: PrimaryStorageLayout
    association_coverage: AssociationCoverage
    completion_state: CompletionState
    reference_assembly: str
    created_at: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    #: How this release's statistic planes are encoded (ADR 0037, issue #119).
    #: A release that declares none is in the `legacy` plan -- `float16`
    #: throughout, which is every release up to `format_version` 0.1. The plan
    #: is read, never re-derived: re-running the decision tree on read would
    #: mean a later threshold change silently altered how existing stores
    #: decode.
    encoding: StoreEncoding = field(default_factory=StoreEncoding.legacy)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StoreManifest:
        return cls(
            encoding=_encoding_from_dict(data),
            store_id=str(data["store_id"]),
            release_id=str(data["release_id"]),
            format_version=str(data["format_version"]),
            primary_layout=PrimaryStorageLayout(data["primary_layout"]),
            association_coverage=AssociationCoverage(data["association_coverage"]),
            completion_state=CompletionState(data["completion_state"]),
            reference_assembly=str(data["reference_assembly"]),
            created_at=data.get("created_at"),
            provenance=dict(data.get("provenance", {})),
        )

    @classmethod
    def load(cls, path: str | Path) -> StoreManifest:
        manifest_path = Path(path)
        if manifest_path.is_dir():
            manifest_path = manifest_path / "manifest.json"
        with manifest_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "store_id": self.store_id,
            "release_id": self.release_id,
            "format_version": self.format_version,
            "primary_layout": self.primary_layout.value,
            "association_coverage": self.association_coverage.value,
            "completion_state": self.completion_state.value,
            "reference_assembly": self.reference_assembly,
            "created_at": self.created_at,
            "provenance": self.provenance,
        }
        # A legacy plan is the *absence* of a declaration, not a declaration of
        # `float16`: writing one out would claim a pre-#114 store had decided
        # something it never did.
        if not self.encoding.is_legacy:
            data["encoding"] = self.encoding.to_manifest()
        return data

