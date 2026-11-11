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
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from opengwasdb.layouts.hybrid.layout import dense_component_path, dense_to_shared_path
from opengwasdb.model.enums import COVERAGES_BY_LAYOUT, PrimaryStorageLayout
from opengwasdb.store.open import (
    DENSE_ENVELOPE,
    HYBRID_DENSE_COMPONENT_ENVELOPE,
    HYBRID_ENVELOPE,
    RAGGED_ENVELOPE,
    WRITTEN_FORMAT_VERSIONS,
    UnsupportedFormatVersion,
    check_format_version_stamp,
    split_format_version,
    zarr_format_for_version,
)
from opengwasdb.validation.rule import Check, Kind, Rule, Target
from opengwasdb.variants import (
    variant_alid_bytes_path,
    variant_alid_rows_path,
    variant_offsets_path,
    variant_tabix_path,
    variant_table_path,
)

#: The rule every rule that opens a release's entries runs after.
REQUIRED = "envelope.required"


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


def _dense_entries(target: Target) -> list[tuple[str, Path]]:
    store = target.store
    path = store.path
    return [
        ("missing index.sqlite", store.index_path),
        ("missing data.zarr", store.data_path),
        ("missing analyses.tsv", store.analyses_path),
        ("missing variants.tsv.gz", variant_table_path(path)),
        ("missing variants.tsv.gz.tbi", variant_tabix_path(path)),
        ("missing variant_offsets.npy", variant_offsets_path(path)),
        (_rebuild_alid_index("variant_alid_bytes.npy"), path / "variant_alid_bytes.npy"),
        (_rebuild_alid_index("variant_alid_rows.npy"), path / "variant_alid_rows.npy"),
    ]


def _ragged_entries(target: Target) -> list[tuple[str, Path]]:
    store = target.store
    path = store.path
    return [
        ("missing index.sqlite", store.index_path),
        ("missing data.zarr", store.data_path),
        ("missing data.zarr/ragged", store.data_path / "ragged"),
        ("missing variants.tsv.gz", variant_table_path(path)),
        ("missing variants.tsv.gz.tbi", variant_tabix_path(path)),
        ("missing variant_alid_bytes.npy", variant_alid_bytes_path(path)),
        ("missing variant_alid_rows.npy", variant_alid_rows_path(path)),
        ("missing analyses.tsv", store.analyses_path),
    ]


def _hybrid_entries(target: Target) -> list[tuple[str, Path]]:
    path = target.store.path
    dense_dir = dense_component_path(path)
    return [
        ("missing dense/ (Dense Component)", dense_dir),
        ("missing dense/manifest.json", dense_dir / "manifest.json"),
        ("missing dense/dense_to_shared.npy", dense_to_shared_path(path)),
        ("missing index.sqlite", path / "index.sqlite"),
        ("missing variants.tsv.gz", variant_table_path(path)),
        ("missing variants.tsv.gz.tbi", variant_tabix_path(path)),
        ("missing variant_alid_bytes.npy", variant_alid_bytes_path(path)),
        ("missing variant_alid_rows.npy", variant_alid_rows_path(path)),
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


def _envelope(target: Target) -> frozenset[str]:
    if target.parent is not None:
        # A Hybrid release's Dense Component additionally carries dense_to_shared.npy,
        # which a standalone Dense release never does.
        return HYBRID_DENSE_COMPONENT_ENVELOPE
    return {
        PrimaryStorageLayout.DENSE: DENSE_ENVELOPE,
        PrimaryStorageLayout.RAGGED: RAGGED_ENVELOPE,
        PrimaryStorageLayout.HYBRID: HYBRID_ENVELOPE,
    }[target.layout]


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
    with target.store.index_connection() as connection:
        _reject_stray_analyses_table(connection, errors)
    return errors


def _reject_stray_analyses_table(connection: sqlite3.Connection, errors: list[str]) -> None:
    """Every layout shares this rule (ADR 0034, issue #72): analyses.tsv is
    the sole source of truth for Analytical Metadata, so a leftover SQL
    ``analyses`` table -- Dense/Hybrid retired theirs in issue #68, Ragged in
    issue #69 -- is a validation failure, not a harmless relic."""
    tables = {r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "analyses" in tables:
        errors.append(
            "index.sqlite still contains an analyses table -- analyses.tsv is the sole "
            "source of truth for Analytical Metadata (ADR 0030, issue #22)"
        )


def format_stamp(target: Target) -> list[str]:
    """The release is stamped the lowest `format_version` its features use (spec §21.3):
    0.2.1 when its manifest declares `regions.tsv`, 0.2.0 otherwise. One stamped otherwise
    disagrees with itself: an older reader takes a 0.2.0 one as fully understood, and no
    writer stamps a 0.2.1 one without the table."""
    manifest = target.store.manifest
    try:
        check_format_version_stamp(
            manifest.format_version, carries_regions=manifest.regions is not None
        )
    except UnsupportedFormatVersion as exc:
        return [str(exc)]
    return []


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
    Check(
        name="manifest.format_stamp", kinds=Kind.RELEASE, when=_written_series, run=format_stamp
    ),
    Check(
        name="sqlite.analyses_table",
        kinds=Kind.RELEASE,
        after=(REQUIRED,),
        run=stray_analyses_table,
    ),
)
