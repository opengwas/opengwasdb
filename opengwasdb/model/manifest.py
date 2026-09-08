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

#: The first `format_version` MAJOR that admits each encoding kind. A release
#: stamped with an earlier major must not declare a kind its own format does
#: not know: a conforming reader of that major is entitled to read the plane
#: as the encoding that major defines -- `se` below 3.0 is `float16`, `eaf`
#: below 2.0 is ADR 0036's `float32` plane -- and a manifest that says
#: otherwise hands it bytes it will decode wrong (issue #157, spec §6a). The
#: boundary below format 1.0 is sterner still and needs no row: below 1.0 a
#: release never declared a block at all, so any block there is refused whole
#: (the fixed-point `z` of 1.0 cannot be declared by a 0.x release because a
#: 0.x release cannot declare anything). Kinds absent from this table decode
#: the same in every major that declares a block and are gated by nothing but
#: the block's own parser; the next version-gated kind to land adds a row
#: rather than a branch.
#:
#: The majors are the format changes that introduced the kinds: 2.0
#: residual-coded `eaf` (#116), 3.0 residual-coded `se` (#118).
_ENCODING_KIND_ADMITTED_MAJOR: dict[tuple[str, str], int] = {
    ("eaf", "int8_residual"): 2,
    ("se", "int8_residual"): 3,
}


def _refuse_kinds_the_version_does_not_admit(
    encoding: StoreEncoding, version: str, major: int
) -> None:
    """Refuse an `encoding` block whose kinds postdate the release's version.

    Each declared (plane, kind) must have existed by the release's own
    `format_version` major. Without the check, a release stamped 1.0 or 2.0
    can declare the format-3 residual `se` and this package decodes it --
    while a conforming 1.0/2.0 reader, entitled to `float16` there, reads the
    `int8` codes as `float16` and returns plausible, wrong standard errors.
    That is precisely the failure the explicit `encoding` block was added to
    prevent, in the same direction the parser already guards when it refuses a
    1.0-or-above release with no block at all.
    """
    for (plane, kind), admitted_major in _ENCODING_KIND_ADMITTED_MAJOR.items():
        if major >= admitted_major:
            continue
        if getattr(encoding, plane).kind == kind:
            raise UnsupportedEncoding(
                f"release declares format_version={version!r} but its `encoding` block "
                f"declares {plane} kind {kind!r}, which only format_version "
                f"{admitted_major}.0 and above admit (spec §6a); a reader of {version} "
                "would not know this representation, and this release cannot be read"
            )


def _encoding_from_dict(data: dict[str, Any]) -> StoreEncoding:
    """The release's declared plan, or the legacy one -- but only where an
    absent declaration is what "legacy" means, and only a plan the release's
    own `format_version` admits.

    The version is parsed first and decides both directions of the contract
    (issue #157, spec §6a):

    - A release at `format_version` 1.0 or above MUST declare its encoding.
      Falling back to the legacy plan there would decode an `int16` plane as
      `float16` and hand back plausible, wrong z-scores -- the exact failure
      an explicit declaration exists to prevent -- so a missing block is
      refused rather than guessed at.
    - A release below `format_version` 1.0 MUST NOT declare one: "`float16`
      throughout" *is* the absence of a block, which is what every release up
      to 0.1 is in. A block there claims an encoding the format cannot carry.
    - A declared kind must have existed by the release's major version; the
      format-3 residual `se` on a 1.0 or 2.0 manifest is refused rather than
      decoded, because an older reader would read its `int8` bytes as
      `float16`.
    """
    # Deferred import: `opengwasdb.store.open` imports this module, and the
    # version parser belongs to the reader contract that lives there. One
    # parser, imported at call time, rather than a second copy of it here.
    from opengwasdb.store.open import parse_format_version

    version = str(data["format_version"])
    major, _ = parse_format_version(version)
    declared = data.get("encoding")
    if declared is not None:
        if major < 1:
            raise UnsupportedEncoding(
                f"release declares format_version={version!r} and an `encoding` block, "
                "but below format_version 1.0 a release never declared one -- its planes "
                "are float16 throughout, which is what the absence of a block means "
                "(spec §6a). A block here cannot describe this release."
            )
        encoding = StoreEncoding.from_manifest(declared)
        _refuse_kinds_the_version_does_not_admit(encoding, version, major)
        return encoding
    if major >= 1:
        raise UnsupportedEncoding(
            f"release declares format_version={version!r} but no `encoding` "
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
