"""Rules that read a Store Release's directory and manifest (ADR 0074).

The Zarr format, the release's required entries, its closed envelope, the coverage
its layout can carry, its `format_version` stamp, and the retired SQL `analyses`
table. One rule each, for
every kind of release: a Dense release, a Ragged one, a Hybrid's outer release and
its Dense Component differ only in the entries and envelope their layout names.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from opengwasdb.index.release_index import ReleaseIndex
from opengwasdb.model.enums import COVERAGES_BY_LAYOUT
from opengwasdb.store.open import (
    check_format_version_stamp,
    envelope_at,
    newer_than_written,
    newer_than_written_text,
    ships_index,
)
from opengwasdb.store_reader.alid_index import alid_index_kind
from opengwasdb.store_reader.enums import PrimaryStorageLayout
from opengwasdb.store_reader.format import (
    WRITTEN_FORMAT_VERSIONS,
    UnsupportedFormatVersion,
    carries_index_metadata,
    carries_position_sidecar,
    carries_source_alid_sidecar,
    carries_variant_offsets,
    indels_reference_first,
    split_format_version,
    zarr_format_for_version,
)
from opengwasdb.store_reader.hybrid import dense_component_path, dense_to_shared_path
from opengwasdb.store_reader.position_sidecar import POSITION_SIDECAR_FILENAMES
from opengwasdb.store_reader.rsid_index import (
    VARIANT_RSID_BYTES_FILENAME,
    VARIANT_RSID_ROWS_FILENAME,
)
from opengwasdb.store_reader.source_alid_sidecar import SOURCE_ALID_SIDECAR_FILENAMES
from opengwasdb.store_reader.variant_axis import (
    variant_offsets_path,
    variant_tabix_path,
    variant_table_path,
)
from opengwasdb.validation.rule import Check, Finding, Kind, Rule, Severity, Target

#: The rule every rule that opens a release's entries runs after.
REQUIRED = "envelope.required"


def current_builders_wrote(target: Target) -> bool:
    """Whether the release is of a format this package's builders write: 0.2.2 on.

    0.2.2 is the first format every build and every completion of this package
    stamps (spec §21.3, ADR 0079). Such a release spells an indel's ALID
    reference first (#381), holds an SE exception at the source's `float32`
    (#355) and has dropped and counted every lift onto a non-primary contig
    (#376). The rules judging those fixes ask this one question, so they move
    together: a release before it was written by an earlier package, and nothing
    rewrites it (#367).
    """
    return indels_reference_first(target.store.manifest.format_version)


def severity_owed(target: Target, severity: Severity) -> Severity:
    """``severity`` on a release `current_builders_wrote`; a warning on an earlier
    one, which says what the release is without failing a Store nobody can fix in
    place (#355, #376, #367)."""
    return severity if current_builders_wrote(target) else Severity.WARNING


@dataclass(frozen=True, kw_only=True)
class SeverityByFormat(Check):
    """A check whose severity says what the release's *format* owes, not what a reader saw.

    An error fails the release, so only one written by a builder that had to hold
    the value exactly can be failed: a release of an earlier format, package
    0.9.0's 0.2.1 among them, may hold a rounded table and is reported as a
    warning (`severity_owed`), which says what it is without failing a Store
    nobody can fix in place (issue #355, #367). Every other rule's severity is
    its own. `side_tables.se_exceptions_rounded` and `analyses.top_hit_counts` (#509)
    are judged so.
    """

    def findings(self, target: Target) -> Iterator[Finding]:
        owed = severity_owed(target, self.severity)
        for message in self.run(target):
            # A release whose format owes the rule's own severity is judged as any rule
            # is, so a deviation its conversion recorded is lowered there (#492).
            yield self.finding(target, message, None if owed is self.severity else owed)


def zarr_format(target: Target) -> list[str]:
    """The release's Zarr tree is in the format its `format_version` names."""
    errors: list[str] = []
    store = target.store
    label = "dense/data.zarr" if target.parent is not None else "data.zarr"
    _validate_zarr_format(store.data_path, store.manifest.format_version, errors, label=label)
    return errors


def _validate_zarr_format(
    directory: Path, format_version: str, errors: list[str], *, label: str
) -> None:
    """The Zarr on-disk format must be the one `format_version` names.

    0.1.0 is Zarr v2: every group has a `.zgroup` and every array a `.zarray`,
    each declaring ``zarr_format: 2``, and no v3 `zarr.json` exists.  0.2.0 is
    Zarr v3 with sharding: nothing carries v2 metadata, every node has a
    `zarr.json` declaring ``zarr_format: 3``, and every array uses the
    ``sharding_indexed`` codec.  A half-converted release -- one manifest, two
    formats under it -- is the failure this rule exists to catch, and it is
    invisible to any check that opens the store through zarr (issue #245).
    """
    if not directory.is_dir():
        return
    # The expected layout comes from the version-to-layout seam, which is the
    # same table the reader checks, rather than a second reading of the version
    # string here (#264 review).
    try:
        expected_zarr_format = zarr_format_for_version(format_version, source=label)
    except UnsupportedFormatVersion as exc:
        errors.append(str(exc))
        return
    expected_v3 = expected_zarr_format == 3
    nodes = 0
    for node_path, present in _zarr_nodes(directory):
        nodes += 1
        relative = f"{label}/{node_path.relative_to(directory)}"
        if expected_v3:
            errors.extend(_zarr_v3_node_errors(relative, node_path, present, format_version))
        else:
            errors.extend(_zarr_v2_node_errors(relative, node_path, present, format_version))
    if nodes == 0:
        errors.append(f"{label}: no Zarr metadata found; the array tree is missing")


def _zarr_v2_node_errors(
    relative: str, node_path: Path, present: set[str], format_version: str
) -> list[str]:
    """One node's errors under a 0.1.0 (Zarr v2) manifest."""
    if _V3_NODE_METADATA in present:
        return [
            f"{relative}: Zarr v3 metadata under format_version {format_version}; a "
            f"{format_version} release is Zarr v2 (issue #245)"
        ]
    if len(present) > 1:
        return [
            f"{relative}: carries {sorted(present)}; a Zarr node has exactly one metadata "
            "file"
        ]
    metadata = next(iter(present))
    declared = json.loads((node_path / metadata).read_text(encoding="utf-8")).get(
        "zarr_format"
    )
    if declared != 2:
        return [
            f"{relative}/{metadata}: zarr_format is {declared!r}, not 2, but the release "
            f"declares format_version {format_version}"
        ]
    return []


def _zarr_v3_node_errors(
    relative: str, node_path: Path, present: set[str], format_version: str
) -> list[str]:
    """One node's errors under a 0.2.0 (Zarr v3, sharded) manifest."""
    v2_metadata = sorted(present & set(_V2_NODE_METADATA))
    if v2_metadata:
        return [
            f"{relative}: Zarr v2 metadata {v2_metadata} under format_version "
            f"{format_version}; a {format_version} release is Zarr v3, and a "
            "half-converted release is invalid (issue #245)"
        ]
    metadata = json.loads((node_path / _V3_NODE_METADATA).read_text(encoding="utf-8"))
    errors: list[str] = []
    if metadata.get("zarr_format") != 3:
        errors.append(
            f"{relative}/zarr.json: zarr_format is {metadata.get('zarr_format')!r}, not 3, "
            f"but the release declares format_version {format_version}"
        )
    if metadata.get("node_type") == "array":
        names = [
            codec.get("name")
            for codec in metadata.get("codecs", [])
            if isinstance(codec, dict)
        ]
        if "sharding_indexed" not in names:
            errors.append(
                f"{relative}/zarr.json: a {format_version} array must use the sharding "
                f"codec; its codecs are {names}"
            )
    return errors


#: Zarr metadata files that name one node, keyed by the Zarr format they belong
#: to.  A node carries exactly one of these; two is a half-converted store.
_V2_NODE_METADATA = (".zarray", ".zgroup")
_V3_NODE_METADATA = "zarr.json"


def _zarr_nodes(directory: Path) -> Iterator[tuple[Path, set[str]]]:
    """Every Zarr node directory under `directory`, in path order, with its metadata.

    Read from the filesystem, not through zarr: the check is precisely that the
    stored metadata and the manifest agree, and asking zarr would let it hide a
    node behind a consolidated record or an auto-detected format. The walk's order is
    the filesystem's, so the nodes are sorted once it ends: a tree with two bad nodes
    reports them in the same order on every run.
    """
    found: list[tuple[Path, set[str]]] = []
    for dirpath, _dirnames, filenames in os.walk(directory):
        present = {name for name in filenames if name in (*_V2_NODE_METADATA, _V3_NODE_METADATA)}
        if present:
            found.append((Path(dirpath), present))
    yield from sorted(found, key=lambda node: str(node[0]))


def _rebuild_alid_index(name: str) -> str:
    # The ALID search index has no reader fallback, so its absence names the
    # rebuild rather than reading as an internal inconsistency.
    return f"missing {name} — rebuild the store to generate the ALID search index"


def _alid_index_entries(target: Target, *, rebuild: bool = False) -> list[tuple[str, Path]]:
    """The files of the ALID index a release of its `format_version` carries
    (`alid_index_kind`): from 0.3.0 the variable-length sidecar (ADR 0105), before it the
    fixed-width pair."""
    kind = alid_index_kind(target.store.manifest.format_version)
    path = target.store.path
    return [
        (_rebuild_alid_index(name) if rebuild else f"missing {name}", path / name)
        for name in kind.FILENAMES
    ]


def _index_entry(target: Target) -> list[tuple[str, Path]]:
    """`index.sqlite`, where a release of its layout, version and completion state
    carries one (`ships_index`): every 0.2 release; from 0.3.0 only one that holds its
    own `completion_quality` (ADR 0102)."""
    if not target_ships_index(target):
        return []
    return [("missing index.sqlite", target.store.index_path)]


def target_ships_index(target: Target) -> bool:
    """Whether the release carries an `index.sqlite` (`ships_index`)."""
    manifest = target.store.manifest
    return ships_index(
        target.layout,
        manifest.format_version,
        manifest.completion_state,
        dense_component=_is_dense_component(target),
    )


def _position_sidecar_entries(target: Target) -> list[tuple[str, Path]]:
    """The position sidecar's files, from 0.3.0 (ADR 0105 §6): a 0.3.0 release answers
    every range from them, and one without them is refused, never read by tabix."""
    if not carries_position_sidecar(target.store.manifest.format_version):
        return []
    path = target.store.path
    return [
        (f"missing {name} — rebuild the store to generate the position sidecar", path / name)
        for name in POSITION_SIDECAR_FILENAMES
    ]


def _source_alid_sidecar_entries(target: Target) -> list[tuple[str, Path]]:
    """The `source_alid` sidecar's files, from 0.3.0 (#470): a 0.3.0 release names each
    row's `source_alid` from them, and one without them is refused, never read as none."""
    if not carries_source_alid_sidecar(target.store.manifest.format_version):
        return []
    path = target.store.path
    return [
        (f"missing {name} — rebuild the store to generate the source_alid sidecar", path / name)
        for name in SOURCE_ALID_SIDECAR_FILENAMES
    ]


def _variant_offsets_entries(target: Target) -> list[tuple[str, Path]]:
    """`variant_offsets.npy`, which a Dense release of the 0.2 and 0.1 series carries and
    a 0.3.0 release does not (#469)."""
    if not carries_variant_offsets(target.store.manifest.format_version):
        return []
    return [("missing variant_offsets.npy", variant_offsets_path(target.store.path))]


def _dense_entries(target: Target) -> list[tuple[str, Path]]:
    store = target.store
    path = store.path
    if target.shares_variant_axis:
        # A 0.3.0 Dense Component: its grid, the map the Hybrid's entries name, and the
        # shared axis it reads its variants from, without which none of its rules can
        # name a row (#472).
        return [
            *_index_entry(target),
            ("missing data.zarr", store.data_path),
            *_shared_axis_entries(target),
        ]
    return [
        *_index_entry(target),
        ("missing data.zarr", store.data_path),
        ("missing analyses.tsv", store.analyses_path),
        ("missing variants.tsv.gz", variant_table_path(path)),
        ("missing variants.tsv.gz.tbi", variant_tabix_path(path)),
        *_variant_offsets_entries(target),
        *_alid_index_entries(target, rebuild=True),
        *_position_sidecar_entries(target),
        *_source_alid_sidecar_entries(target),
    ]


def _shared_axis_entries(target: Target) -> list[tuple[str, Path]]:
    """The Hybrid release's search indexes and sidecars a 0.3.0 Dense Component reads its
    variants from: those the Hybrid's own `format_version` carries, as its own axis opens
    them, the `source_alid` sidecar included (#470)."""
    root = target.store.shared_root
    assert root is not None and target.parent is not None, "only a nested Dense Component"
    version = target.parent.store.manifest.format_version
    names = (
        *alid_index_kind(version).FILENAMES,
        *(POSITION_SIDECAR_FILENAMES if carries_position_sidecar(version) else ()),
        *(SOURCE_ALID_SIDECAR_FILENAMES if carries_source_alid_sidecar(version) else ()),
        VARIANT_RSID_BYTES_FILENAME,
        VARIANT_RSID_ROWS_FILENAME,
    )
    return [
        (f"missing {name} in the Hybrid release, whose variant axis the Dense Component "
         "reads", root / name)
        for name in names
    ]


def _ragged_entries(target: Target) -> list[tuple[str, Path]]:
    store = target.store
    path = store.path
    return [
        *_index_entry(target),
        ("missing data.zarr", store.data_path),
        ("missing data.zarr/ragged", store.data_path / "ragged"),
        ("missing variants.tsv.gz", variant_table_path(path)),
        ("missing variants.tsv.gz.tbi", variant_tabix_path(path)),
        *_alid_index_entries(target),
        *_position_sidecar_entries(target),
        *_source_alid_sidecar_entries(target),
        ("missing analyses.tsv", store.analyses_path),
    ]


def _hybrid_entries(target: Target) -> list[tuple[str, Path]]:
    path = target.store.path
    dense_dir = dense_component_path(path)
    return [
        ("missing dense/ (Dense Component)", dense_dir),
        ("missing dense/manifest.json", dense_dir / "manifest.json"),
        ("missing dense/dense_to_shared.npy", dense_to_shared_path(path)),
        *_index_entry(target),
        ("missing variants.tsv.gz", variant_table_path(path)),
        ("missing variants.tsv.gz.tbi", variant_tabix_path(path)),
        *_alid_index_entries(target),
        *_position_sidecar_entries(target),
        *_source_alid_sidecar_entries(target),
        ("missing data.zarr/ragged (Ragged Overflow)", path / "data.zarr" / "ragged"),
    ]


_ENTRIES = {
    PrimaryStorageLayout.DENSE: _dense_entries,
    PrimaryStorageLayout.RAGGED: _ragged_entries,
    PrimaryStorageLayout.HYBRID: _hybrid_entries,
}


def required_entries(target: Target) -> Iterator[str]:
    """Every entry the release's layout requires is present (spec §1, §10, §11, §16).

    Every rule that opens one of them runs after this, so a release that lacks one
    is reported once, here, rather than as a crash in each rule that reads it.
    """
    for message, path in _ENTRIES[target.layout](target):
        if not path.exists():
            yield message


def _is_dense_component(target: Target) -> bool:
    """Whether the release is a Hybrid's nested Dense Component, which additionally
    carries dense_to_shared.npy, as a standalone Dense release never does."""
    return target.parent is not None


def _envelope(target: Target) -> frozenset[str]:
    manifest = target.store.manifest
    return envelope_at(
        target.layout,
        manifest.format_version,
        manifest.completion_state,
        dense_component=_is_dense_component(target),
    )


def closed_envelope(target: Target) -> list[str]:
    errors: list[str] = []
    _validate_closed_envelope(target.store.path, _envelope(target), errors)
    return errors


def _validate_closed_envelope(store_path: Path, allowed: frozenset[str], errors: list[str]) -> None:
    """store-format spec §1/§20 (issue #80): a Store Release's top-level
    directory is closed -- no file or directory beyond what its layout
    legitimately produces. Every other check here is a *presence* check
    ("required entry X exists"); this is the one *closure* check, catching
    drift like the long-unnoticed Ragged `traits.tsv.gz` side-file (retired
    in issue #69) that no presence check could ever have flagged."""
    for name in sorted(p.name for p in store_path.iterdir() if p.name not in allowed):
        errors.append(
            f"unexpected store entry {name!r} in {store_path} is not part of the "
            "documented store envelope for this layout (store-format spec §1)"
        )


def coverage(target: Target) -> list[str]:
    errors: list[str] = []
    _require_layout_coverage(target.store.manifest, errors)
    return errors


def _require_layout_coverage(manifest: Any, errors: list[str]) -> None:
    """The coverage this release declares must be one its layout can carry.

    A Dense or Hybrid release has a cell for every (variant, Analysis), so a
    coverage narrower than `gwas` would leave cells whose absence means
    "filtered out" while every other missing cell means "the source did not
    report it" -- two answers with one representation (issue #298). The set of
    values each layout can carry is `COVERAGES_BY_LAYOUT`, the same one every
    builder is held to, so a release no builder could have written is refused
    rather than reported as a Store of a different kind.
    """
    accepted = COVERAGES_BY_LAYOUT[manifest.primary_layout]
    if manifest.association_coverage not in accepted:
        errors.append(
            f"a {manifest.primary_layout.value} store cannot carry "
            f"association_coverage={manifest.association_coverage.value} "
            f"(it accepts {', '.join(sorted(c.value for c in accepted))})"
        )


def stray_analyses_table(target: Target) -> list[str]:
    errors: list[str] = []
    with ReleaseIndex(target.store.path) as index:
        _reject_stray_analyses_table(index, errors)
    return errors


def _reject_stray_analyses_table(index: ReleaseIndex, errors: list[str]) -> None:
    """Every layout shares this rule (ADR 0034, issue #72): analyses.tsv is
    the sole source of truth for Analytical Metadata, so a leftover SQL
    ``analyses`` table -- Dense/Hybrid retired theirs in issue #68, Ragged in
    issue #69 -- is a validation failure, not a harmless relic."""
    if _ANALYSES_TABLE in index.names(tables_only=True):
        errors.append(
            "index.sqlite still contains an analyses table -- analyses.tsv is the sole "
            "source of truth for Analytical Metadata (ADR 0030, issue #22)"
        )


#: The one table a 0.3.0 `index.sqlite` holds (ADR 0102).
_PROVENANCE_TABLES = frozenset({"completion_quality"})

#: A table `sqlite.analyses_table` reports, on every format, so not reported again.
_ANALYSES_TABLE = "analyses"


def provenance_only_index(target: Target) -> list[str]:
    """A 0.3.0 `index.sqlite` holds `completion_quality` and nothing else (ADR 0102).

    The `metadata` table and `variant_aliases` a 0.2 release carried are dropped from
    0.3.0 (#471); one still present is a writer that did not move with the format.
    """
    with ReleaseIndex(target.store.path) as index:
        tables = index.names(tables_only=True)
    stray = sorted(
        name
        for name in tables - _PROVENANCE_TABLES - {_ANALYSES_TABLE}
        if not name.startswith("sqlite_")
    )
    if not stray:
        return []
    return [
        f"index.sqlite holds {', '.join(stray)}, which format "
        f"{target.store.manifest.format_version} does not carry: from 0.3.0 it holds only "
        "completion_quality (ADR 0102)"
    ]


def _provenance_only(target: Target) -> bool:
    """A release whose `index.sqlite` ships and holds provenance alone: 0.3.0 on."""
    version = target.store.manifest.format_version
    return target_ships_index(target) and not carries_index_metadata(version)


def format_stamp(target: Target) -> list[str]:
    """The release is stamped the lowest `format_version` its features use (spec §21.3):
    0.2.2 when its indels are spelt reference first, which its stamp declares, the table
    or not; otherwise 0.2.1 when its manifest declares `regions.tsv`, 0.2.0 when it does
    not. One stamped otherwise disagrees with itself: an older reader takes a 0.2.0 one as
    fully understood, and no writer stamps a 0.2.1 one without the table. A release
    newer than this build writes is not judged here (`_FormatStamp`)."""
    manifest = target.store.manifest
    try:
        check_format_version_stamp(
            manifest.format_version, carries_regions=manifest.regions is not None
        )
    except UnsupportedFormatVersion as exc:
        return [str(exc)]
    return []


@dataclass(frozen=True, kw_only=True)
class _FormatStamp(Check):
    """`format_stamp`, except on a release newer than every version this build writes.

    Such a release is a later package's format, and which stamp its features give is
    that package's rule: judged by this one's, a 0.2.3 release would be told it "is
    0.2.2", and re-stamped so it is a release this build completes, whatever 0.2.3
    changed -- what package 0.9.0's rule tells a 0.2.2 release. It is read with the
    newer-remainder warning, and this rule warns that the package must be upgraded.
    """

    def findings(self, target: Target) -> Iterator[Finding]:
        version = target.store.manifest.format_version
        if newer_than_written(version):
            yield self.finding(target, newer_than_written_text(version), Severity.WARNING)
            return
        yield from super().findings(target)


def converted_from(target: Target) -> list[str]:
    """Every deviation `provenance.converted_from` records is one a conversion records,
    for the reason the record gives (`legacy_conversion.judge_converted_from`, #492):
    `validate` lowers only those, so any other would silence a rule."""
    from opengwasdb.store.legacy_conversion import judge_converted_from

    _accepted, faults = judge_converted_from(target.store.manifest.provenance["converted_from"])
    return faults


def _converted(target: Target) -> bool:
    provenance = target.store.manifest.provenance
    return isinstance(provenance, dict) and "converted_from" in provenance


def _written_series(target: Target) -> bool:
    """A release of the series this build writes; a 0.1.0 one predates the stamp rule."""
    series = {split_format_version(version)[0] for version in WRITTEN_FORMAT_VERSIONS}
    return split_format_version(target.store.manifest.format_version)[0] in series


def _not_ragged(target: Target) -> bool:
    return target.layout is not PrimaryStorageLayout.RAGGED


RULES: tuple[Rule, ...] = (
    Check(name="zarr.format", kinds=Kind.RELEASE, run=zarr_format),
    Check(name=REQUIRED, kinds=Kind.RELEASE, run=required_entries),
    Check(name="envelope.closed", kinds=Kind.RELEASE, run=closed_envelope),
    Check(name="manifest.coverage", kinds=Kind.RELEASE, when=_not_ragged, run=coverage),
    _FormatStamp(
        name="manifest.format_stamp", kinds=Kind.RELEASE, when=_written_series, run=format_stamp
    ),
    Check(name="manifest.converted_from", kinds=Kind.RELEASE, when=_converted, run=converted_from),
    Check(
        name="sqlite.analyses_table",
        kinds=Kind.RELEASE,
        when=target_ships_index,
        after=(REQUIRED,),
        run=stray_analyses_table,
    ),
    Check(
        name="sqlite.provenance_only",
        kinds=Kind.RELEASE,
        when=_provenance_only,
        after=(REQUIRED,),
        run=provenance_only_index,
    ),
)
