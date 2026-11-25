"""A Store Release's `manifest.json`, as the model every reader opens it into.

Reading it checks the release's `format_version` first and parses its declared
encoding (spec §2, §6a). Rewriting a manifest's coverage to the written spelling
is a writer's job, `opengwasdb.model.manifest.normalise_coverage`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from opengwasdb.store_reader.encoding.plan import StoreEncoding, UnsupportedEncoding
from opengwasdb.store_reader.enums import (
    AssociationCoverage,
    CompletionState,
    PrimaryStorageLayout,
)
from opengwasdb.store_reader.format import check_format_version


def _encoding_from_dict(data: dict[str, Any]) -> StoreEncoding:
    """The release's declared plan. Never inferred, and never defaulted.

    The version is parsed first, because a release this build cannot read must
    be refused by version rather than by whatever its `encoding` block happens
    to say (spec §6a). Past that, every readable release declares a block:
    there is one format, so there is no version at which an absent block means
    something (issue #143).

    Falling back to a plan the release did not state would decode an `int16`
    plane as `float16` and hand back plausible, wrong z-scores -- the exact
    failure an explicit declaration exists to prevent -- so a missing block is
    refused rather than guessed at.

    Issue #157's per-kind version gate is gone with the versions it gated: it
    stopped a release stamped 1.0 or 2.0 declaring a kind only a later format
    admitted, and this build reads exactly one format, whose block may declare
    any kind the parser implements.
    """
    version = str(data["format_version"])
    check_format_version(version, source="release")
    declared = data.get("encoding")
    if declared is None:
        raise UnsupportedEncoding(
            f"release declares format_version={version!r} but no `encoding` block; "
            "the encoding is required (spec §6a), and a release that does not "
            "declare one cannot be decoded"
        )
    return StoreEncoding.from_manifest(declared)


@dataclass(frozen=True)
class RegionsEntry:
    """The manifest's `regions` entry: the region table a Ragged Store Release carries
    (`regions.tsv`, spec §17), by its path in the release and the SHA-256 of its bytes.

    It is how a reader finds the table (spec §21.2's manifest-based feature discovery),
    as `encoding` declares the planes: a release carrying no table has no entry, and its
    manifest is the one it would have been without this field.
    """

    path: str
    sha256: str

    def to_manifest(self) -> dict[str, str]:
        return {"path": self.path, "sha256": self.sha256}

    @classmethod
    def from_manifest(cls, data: Any) -> RegionsEntry:
        """Read a declared entry, refusing one that is not `{path, sha256}`."""
        if (
            not isinstance(data, dict)
            or set(data) != {"path", "sha256"}
            or not all(isinstance(value, str) for value in data.values())
        ):
            raise ValueError(
                f"manifest.json's regions entry {data!r} is not {{path, sha256}} (spec §17)"
            )
        return cls(path=data["path"], sha256=data["sha256"])


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
    #: How this release's statistic planes are encoded (ADR 0037, issue #119).
    #: Required, with no default: a plan nobody stated is a plan nobody can be
    #: held to, and every release this build reads or writes declares one
    #: (issue #143). The plan is read, never re-derived -- re-running the
    #: decision tree on read would mean a later threshold change silently
    #: altered how existing stores decode.
    encoding: StoreEncoding
    created_at: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    #: The region table the release carries (`RegionsEntry`); `None` when it carries
    #: none, and then omitted from the manifest entirely, as an absent `encoding.n` is.
    regions: RegionsEntry | None = None
    #: The `clumps` entry (spec §2): which tier's Clumps the release declares, None when it
    #: declares none. Read, never written: `to_dict` leaves it out, so a manifest written
    #: from this object -- a completion's, which rewrites the tier the Clumps were parallel
    #: to -- never declares Clumps it does not carry (ADR 0106 §10). Only the step that
    #: writes them declares them, in place (`OpenGWASDBStore.set_manifest_entry`).
    clumps: Any = None

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
            regions=(
                RegionsEntry.from_manifest(data["regions"]) if "regions" in data else None
            ),
            clumps=data.get("clumps"),
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
            "encoding": self.encoding.to_manifest(),
        }
        if self.regions is not None:
            data["regions"] = self.regions.to_manifest()
        return data
