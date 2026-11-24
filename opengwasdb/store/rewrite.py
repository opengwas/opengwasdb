"""Rewrite a 0.2.3 Store Release into a new 0.3.0 one (`ogdb rewrite-release`, #467).

Format 0.3.0 changes how a release stores what it answers, never what it answers
(epic #450, ADR 0105). An existing 0.2.3 release reaches it by this command, not by a
rebuild: it derives a **new** Staged Release (ADR 0038) and leaves its source as it
was. The rewrite:

1. refuses a destination that exists, is the source or lies inside it, a source that
   is not 0.2.3 -- including one already 0.3.0 -- and a source any step refuses
   (`RewriteStep.refuse`), before a byte is written;
2. stages a copy of the source (`cp -a --reflink=auto`, so an unchanged array costs
   no space where the filesystem can share it), leaving out every path a step drops
   (`RewriteStep.leaves_out`): what 0.3.0 no longer carries is never copied;
3. runs its **steps** on the staged copy, in order. Each is one named change the
   format makes; #468-#472 each add theirs to `STEPS` (`RewriteStep` says how). A
   step that reads a component another drops reads it from the source;
4. stamps every manifest 0.3.0 with a fresh `release_id` and `created_at`, keeps
   `store_id`, and records a `format_rewrite` provenance block naming the source
   `release_id` and the steps (ADR 0038 §5, as the 0.2.0 converter does);
5. asks the source and the staged release the same deterministic set of questions
   (`opengwasdb.store.query_digests`) and refuses on any difference;
6. validates the staged release, and publishes it by rename only if it is clean.

Any failure discards the staging directory; nothing is published and the source is
untouched. The staging, the destination refusals and the validation gate are the
0.2.0 converter's (`opengwasdb.store.convert`).

Only a 0.2.3 source is rewritten. 0.3.0 keeps every feature 0.2.3 added on the 0.2
series: an indel's ALID spelt reference first (0.2.2, ADR 0079) and the count of
associations left out for an SE above 65,504 (0.2.3, ADR 0068). An earlier release
lacks one or both and no mechanical step can supply them, so it is rebuilt.
"""

from __future__ import annotations

import json
import shutil
import time
import uuid
from collections.abc import Callable, Iterable, Iterator
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from opengwasdb.index.release_index import ReleaseIndex
from opengwasdb.overview import write_overview_html
from opengwasdb.store import arrays as store_arrays
from opengwasdb.store.convert import (
    copy_path,
    installed_commit,
    manifest_paths,
    refuse_destination,
    require_valid_staged_release,
)
from opengwasdb.store.open import (
    DENSE_COMPONENT_COPIES,
    INDEX_FILENAME,
    OpenGWASDBStore,
    connect_index,
    open_store,
    ships_index,
)
from opengwasdb.store.query_digests import differences, release_digests
from opengwasdb.store_reader import top_hits as reader_top_hits
from opengwasdb.store_reader.alid_sidecar import AlidSidecar, write_alid_sidecar
from opengwasdb.store_reader.analyses import read_analyses
from opengwasdb.store_reader.enums import PrimaryStorageLayout
from opengwasdb.store_reader.format import (
    CURRENT_FORMAT_VERSION,
    LAST_0_2_FORMAT_VERSION,
    UnsupportedFormatVersion,
    carries_alid_sidecar,
    check_format_version,
    dense_component_shares_axis,
    format_version_for,
)
from opengwasdb.store_reader.hybrid import dense_to_shared_path
from opengwasdb.store_reader.position_sidecar import PositionSidecar, write_position_sidecar
from opengwasdb.store_reader.release import release_arrays
from opengwasdb.store_reader.variant_axis import (
    VARIANT_OFFSETS_FILENAME,
    VariantRecord,
    variant_alid_bytes_path,
    variant_alid_rows_path,
    variant_offsets_path,
    variant_table_path,
)
from opengwasdb.top_hits import STORED_P_ARRAYS, stored_p_agrees
from opengwasdb.variants.legacy_table import iter_variant_records
from opengwasdb.variants.normalise import chromosome_sort_key

#: The one `format_version` this rewrite reads: every other is refused (module docstring).
SOURCE_FORMAT_VERSION = LAST_0_2_FORMAT_VERSION

#: The `format_version` it writes: 0.2.3's features with every step's change.
TARGET_FORMAT_VERSION = CURRENT_FORMAT_VERSION

#: How many differences a refused proof prints; it counts them all.
_SHOWN_DIFFERENCES = 20


class RewriteError(Exception):
    """A refusal. An `Exception`, so `OpenGWASDBStore.staging` discards the staging directory."""


class RewriteVerificationError(RewriteError):
    """The staged release answers some question differently from its source.

    `differences` names every one (`query_digests.differences`); the message shows the first.
    """

    def __init__(self, message: str, differences: list[str]) -> None:
        super().__init__(message)
        self.differences = differences


@dataclass(frozen=True)
class Component:
    """One Store Release the source is made of: its root, or a Hybrid's nested Dense
    Component (spec §16), opened read only, as its manifest describes it."""

    #: Its directory, relative to the release's root (``"."`` for the root).
    relative: str
    store: OpenGWASDBStore
    #: Whether it is a Hybrid's nested Dense Component, which the root's layout says.
    dense_component: bool

    @property
    def path(self) -> Path:
        return self.store.path

    @property
    def carries_axis(self) -> bool:
        """Whether the 0.3.0 release made of it carries a Variant Axis of its own: every
        component but a Hybrid's Dense Component, which reads the shared one (#472)."""
        return not (self.dense_component and dense_component_shares_axis(TARGET_FORMAT_VERSION))


@dataclass(frozen=True)
class RewriteContext:
    """What a step works on: the source, read only, and the staged copy it changes.

    `components` are the source's Store Releases (`Component`): its root, and a Hybrid's
    nested Dense Component, whose files a step rewrites as it does the root's.
    """

    source: Path
    staged: Path
    components: tuple[Component, ...]

    def staged_components(self) -> Iterator[Path]:
        for component in self.components:
            yield self.staged / component.relative

    def axis_components(self) -> Iterator[Component]:
        """The components whose Variant Axis the steps rewrite (`Component.carries_axis`)."""
        return (component for component in self.components if component.carries_axis)


def _nothing_to_refuse(_component: Component) -> None:
    return None


def _nothing_left_out(_component: Component) -> Iterable[str]:
    return ()


@dataclass(frozen=True)
class RewriteStep:
    """One named change the rewrite makes. Each part is optional, and runs in this order:

    1. `refuse(component)`, for each source component, before anything is staged:
       raise `RewriteError` for a source the step cannot rewrite without changing an
       answer or losing a record;
    2. `leaves_out(component)`: the paths, relative to that component, the step drops.
       The copy never writes them, so nothing is copied to be deleted, and a step that
       needs one reads it from `RewriteContext.source`;
    3. `run(context)`, on the staged copy, after every earlier step's `run`: what the
       step writes. A step that only drops has none.

    **Adding a step** (#470, #472): write its parts, append it to `STEPS`, and give the
    proof (`opengwasdb.store.query_digests`) any question only the change could answer
    differently. The step's name is recorded in each manifest's `format_rewrite`.
    """

    name: str
    summary: str
    run: Callable[[RewriteContext], None] | None = None
    leaves_out: Callable[[Component], Iterable[str]] = _nothing_left_out
    refuse: Callable[[Component], None] = _nothing_to_refuse


def _write_alid_sidecar(context: RewriteContext) -> None:
    """Write each component's ALID sidecar from its Variant Table; drop the `|S32` pair.

    The ALIDs come from the table, not from the fixed-width index, which leaves out
    every ALID over 32 bytes (#127). The table must hold rows 0..n-1 in order, one per
    entry of `variant_offsets.npy`; any other table is refused, never indexed. The
    sidecar is written by its one writer, the codec's `write_alid_sidecar` (a build
    reaches it through `write_variant_axis`, which writes the whole axis; this step
    rewrites only the ALID index). Before the step ends, every ALID is searched for
    and must name its own row: the proof's questions reach only some of them. A
    Hybrid's Dense Component gets none: `dense_shared_axis` drops its axis (#472).
    """
    for component in context.axis_components():
        directory = context.staged / component.relative
        alids = [
            record.alid for record in _records_in_row_order(directory, _offset_rows(component))
        ]
        write_alid_sidecar(directory, alids)
        _require_every_alid_resolves(directory, alids)
        variant_alid_bytes_path(directory).unlink()
        variant_alid_rows_path(directory).unlink()


def _require_every_alid_resolves(directory: Path, alids: list[str]) -> None:
    """Refuse a sidecar in which some ALID does not find its own row."""
    import numpy as np

    sidecar = AlidSidecar.open(directory)
    if sidecar is None:
        raise RewriteError(f"{directory}: the ALID sidecar was not written")
    rows = sidecar.search(alids)
    wrong = np.flatnonzero(rows != np.arange(len(alids)))
    if wrong.size:
        first = int(wrong[0])
        raise RewriteError(
            f"{directory}: {wrong.size} ALID(s) of {len(alids)} do not resolve to their own "
            f"row in the written sidecar; first, {alids[first]!r} at row {first} resolves to "
            f"{int(rows[first])}"
        )


def _write_position_sidecar(context: RewriteContext) -> None:
    """Write each component's position sidecar from its Variant Table (ADR 0105 §6).

    The rows come from the table in row order, refused as for the ALID sidecar when
    out of order. The writer, the codec's `write_position_sidecar` (which a build
    reaches through `write_variant_axis`), refuses rows not in genomic order. Before
    the step ends, the sidecar read back must place every row where the table does:
    the proof's range questions reach only some of them. The tabix index stays, a
    convenience no 0.3.0 query reads (spec §1.1). A Hybrid's Dense Component gets
    none (#472).
    """
    for component in context.axis_components():
        directory = context.staged / component.relative
        chromosomes: list[str] = []
        positions: list[int] = []
        for record in _records_in_row_order(directory, _offset_rows(component)):
            chromosomes.append(record.chromosome)
            positions.append(record.position)
        write_position_sidecar(
            directory, chromosomes, positions, chromosome_key=chromosome_sort_key
        )
        _require_every_row_placed(directory, chromosomes, positions)


def _require_every_row_placed(
    directory: Path, chromosomes: list[str], positions: list[int]
) -> None:
    """Refuse a position sidecar that places some row elsewhere than the table does
    (`PositionSidecar.misplaced`, the comparison `variant_table.position_sidecar` makes)."""
    sidecar = PositionSidecar.open(directory)
    if sidecar is None:
        raise RewriteError(f"{directory}: the position sidecar was not written")
    wrong = sidecar.misplaced(0, chromosomes, positions)
    if wrong or sidecar.n_rows != len(chromosomes):
        first = wrong[0] if wrong else min(sidecar.n_rows, len(chromosomes))
        raise RewriteError(
            f"{directory}: the written position sidecar holds {sidecar.n_rows} rows and places "
            f"{len(wrong)} of the table's {len(chromosomes)} elsewhere; first, row {first}"
        )


def _records_in_row_order(directory: Path, n_rows: int) -> Iterator[VariantRecord]:
    """Every row of the Variant Table, in row order, read without its search indexes.

    Each step rewrites some of those indexes, so none of them is opened here: the
    table is streamed, and its rows are checked against the `n_rows` the source's
    `variant_offsets.npy` counts (`_offset_rows`; the copy leaves the file out).
    """
    table = variant_table_path(directory)
    row = -1
    for row, record in enumerate(iter_variant_records(table)):
        if record.variant_index != row:
            raise RewriteError(
                f"{table}: row {row} has variant_index {record.variant_index}; a table "
                "out of order is not rewritten"
            )
        yield record
    if row + 1 != n_rows:
        raise RewriteError(
            f"{table} has {row + 1} rows but variant_offsets.npy has {n_rows}; a table that "
            "disagrees with its offsets is not rewritten"
        )


def _offset_rows(component: Component) -> int:
    """How many rows the source component's `variant_offsets.npy` counts (#469).

    The copy leaves the file out (`_variant_offsets_left_out`), so the steps that stream
    the staged table check its rows against the source's offsets.
    """
    import numpy as np

    return len(np.load(variant_offsets_path(component.path), mmap_mode="r"))


def _variant_offsets_left_out(_component: Component) -> Iterable[str]:
    """`variant_offsets.npy`, each row's BGZF offset, which no 0.3.0 query reads (#469)."""
    return (VARIANT_OFFSETS_FILENAME,)


def _check_sidecars_count_offsets(context: RewriteContext) -> None:
    """Refuse unless each staged component's ALID and position sidecars count the rows
    its source's `variant_offsets.npy` did, and drop the manifest's note of the file.

    A 0.3.0 Store Reader never reads the Variant Table, so nothing seeks a row's BGZF
    offset: the ALID sidecar counts the variants. The copy left the file out; this
    checks the sidecars that replace it, after the steps that wrote them. A Dense
    build's `provenance.dense.variant_axis.row_offsets`, which names it, goes too. A
    Hybrid's Dense Component has no sidecars to count: `dense_shared_axis` checked its
    rows against the map instead (#472).
    """
    for component in context.axis_components():
        directory = context.staged / component.relative
        n_offsets = _offset_rows(component)
        counted = {
            "ALID": _rows_of(AlidSidecar.open(directory)),
            "position": _rows_of(PositionSidecar.open(directory)),
        }
        wrong = {name: n for name, n in counted.items() if n != n_offsets}
        if wrong:
            raise RewriteError(
                f"{directory}: variant_offsets.npy counts {n_offsets} rows, but the "
                + ", ".join(f"{name} sidecar {n}" for name, n in wrong.items())
                + "; the offsets are not dropped while the sidecars disagree"
            )
        _forget_row_offsets(directory / "manifest.json")


def _rows_of(sidecar: AlidSidecar | PositionSidecar | None) -> int | None:
    return None if sidecar is None else sidecar.n_rows


def _forget_row_offsets(manifest: Path) -> None:
    """Remove `provenance.dense.variant_axis.row_offsets` from `manifest`, if it names it."""
    data: dict[str, Any] = json.loads(manifest.read_text(encoding="utf-8"))
    axis = data.get("provenance", {}).get("dense", {}).get("variant_axis", {})
    if axis.pop("row_offsets", None) is not None:
        manifest.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _keeps_index(component: Component) -> bool:
    """Whether the 0.3.0 release made of `component` carries `index.sqlite`: only to hold
    its `completion_quality` (`ships_index`, ADR 0102)."""
    manifest = component.store.manifest
    return ships_index(
        manifest.primary_layout,
        TARGET_FORMAT_VERSION,
        manifest.completion_state,
        dense_component=component.dense_component,
    )


#: The `index.sqlite` tables a 0.2.3 release carries, by what 0.3.0 does with them.
_DROPPED_TABLES = frozenset({"metadata", "variant_aliases"})
_KEPT_TABLES = frozenset({"completion_quality"})


def _index_tables(component: Component) -> set[str]:
    """The source index's tables; none where it has no index, or an empty file."""
    index = component.path / INDEX_FILENAME
    if not index.exists() or not index.stat().st_size:
        return set()
    with ReleaseIndex(component.path) as opened:
        return opened.names(tables_only=True)


def _refuse_index_drop(component: Component) -> None:
    """Refuse a source whose `index.sqlite` cannot lose its `metadata` table and
    `variant_aliases` without changing an answer or losing a record (ADR 0102)."""
    tables = _index_tables(component)
    _refuse_unaccounted_tables(component, tables)
    if "variant_aliases" in tables:
        _refuse_unindexed_aliases(component)
    if "metadata" in tables:
        _refuse_disagreeing_dense_blob(component)
    if _keeps_index(component) and "completion_quality" not in tables:
        raise RewriteError(
            f"{component.path / INDEX_FILENAME} has no completion_quality table, which a "
            "Reference-Completed release carries and 0.3.0 keeps"
        )


def _index_left_out(component: Component) -> Iterable[str]:
    """`index.sqlite`, wherever the source has one: an Observed-Only release at 0.3.0 has
    none, and a Reference-Completed one gets a new one holding only `completion_quality`
    (`_write_provenance_index`)."""
    return (INDEX_FILENAME,) if (component.path / INDEX_FILENAME).exists() else ()


def _refuse_unaccounted_tables(component: Component, tables: set[str]) -> None:
    """Refuse an index holding a table 0.3.0 neither keeps nor drops by name."""
    unknown = sorted(
        name
        for name in tables - _DROPPED_TABLES - _KEPT_TABLES
        if not name.startswith("sqlite_")
    )
    if unknown:
        raise RewriteError(
            f"{component.path / INDEX_FILENAME} holds {', '.join(unknown)}, which a 0.3.0 "
            "release does not carry and the rewrite does not know how to drop; it is not "
            "rewritten"
        )


def _refuse_unindexed_aliases(component: Component) -> None:
    """Refuse `variant_aliases` naming an rsID the rsID index does not hold.

    A 0.2 release answers such an rsID from `variant_aliases`; a 0.3.0 one answers it
    as a miss (ADR 0102). Dropping the table would change that answer, and the proof's
    questions name only a few rsIDs, so every alias is checked here.
    """
    with ReleaseIndex(component.path) as index:
        aliases = sorted({str(alias) for alias, _ in index.variant_aliases()})
    if not aliases:
        return
    with closing(component.store.variant_axis(aliases=False)) as axis:
        unindexed = [alias for alias in aliases if not len(axis.indices_by_alias(alias))]
    if unindexed:
        raise RewriteError(
            f"{component.path / INDEX_FILENAME}: {len(unindexed)} variant_aliases rsID(s) are "
            f"not in the rsID index (first, {unindexed[0]!r}); a 0.3.0 release would answer "
            "them as misses, so the release is not rewritten"
        )


#: The `dense` blob's keys a manifest's layout record also holds (spec §10a).
_LAYOUT_KEYS = ("chunk_shape", "shard_shape", "compressor", "zarr_format")


def _manifest_layout(component: Component) -> dict[str, Any]:
    """The layout the component's manifest records (spec §10a): a Hybrid's outer
    `provenance.hybrid`, every other Dense-shaped release's `provenance.dense`."""
    manifest = component.store.manifest
    outer = manifest.primary_layout is PrimaryStorageLayout.HYBRID and not component.dense_component
    layout: dict[str, Any] = manifest.provenance.get("hybrid" if outer else "dense") or {}
    return layout


def _refuse_disagreeing_dense_blob(component: Component) -> None:
    """Refuse a `dense` blob that disagrees with the manifest's layout record.

    0.3.0 keeps the manifest as the one copy (ADR 0102); dropping a blob that said
    otherwise would drop the evidence that one of the two is wrong.
    """
    with ReleaseIndex(component.path) as index:
        blob = index.metadata("dense", default=None)
    if not isinstance(blob, dict):
        return
    recorded = _manifest_layout(component)
    differing = [
        key for key in _LAYOUT_KEYS if key in blob and blob[key] != recorded.get(key)
    ]
    if differing:
        raise RewriteError(
            f"{component.path / INDEX_FILENAME}: its dense blob records "
            f"{', '.join(differing)} otherwise than manifest.json; 0.3.0 keeps the manifest "
            "as the one copy, so a release whose two copies disagree is not rewritten"
        )


def _write_provenance_index(context: RewriteContext) -> None:
    """Give each Reference-Completed component an `index.sqlite` holding only its
    `completion_quality`, copied with its schema from the source's (ADR 0102).

    The source's index was left out of the copy (`_index_left_out`), so its `metadata`
    table and `variant_aliases` are never written to the staged release.
    """
    for component in context.components:
        if not _keeps_index(component):
            continue
        with ReleaseIndex(component.path) as index:
            schema, rows = index.completion_quality_copy()
        staged_index = context.staged / component.relative / INDEX_FILENAME
        with closing(connect_index(staged_index)) as staged:
            for sql in schema:
                staged.execute(sql)
            if rows:
                marks = ", ".join("?" * len(rows[0]))
                staged.executemany(f"INSERT INTO completion_quality VALUES ({marks})", rows)
            staged.commit()


def _tiers_storing_p(component: Component) -> Iterator[tuple[str, Any, list[str]]]:
    """Each tier of the component's Top-Hit Index that stores any of `STORED_P_ARRAYS`:
    its key, its group, and the ones it stores."""
    root = release_arrays(component.path)
    if not reader_top_hits.has_index(root):
        return
    index = reader_top_hits.index_group(root)
    for key in sorted(index):
        stored = sorted(STORED_P_ARRAYS.intersection(index[key].keys()))
        if stored:
            yield key, index[key], stored


def _refuse_stored_p_drop(component: Component) -> None:
    """Refuse a tier whose `abs_z` and `p_value` cannot be dropped: it stores one without
    the other, its stored values are not what its `z` implies, or a consolidated record
    describes it, which would go on naming the dropped arrays."""
    for key, tier, stored in _tiers_storing_p(component):
        where = component.path / "data.zarr" / reader_top_hits.TOP_HIT_GROUP / key
        try:
            store_arrays.refuse_under_consolidated_metadata(where, "dropping abs_z/p_value in")
        except store_arrays.ConsolidatedMetadataError as exc:
            raise RewriteError(str(exc)) from exc
        if stored != sorted(STORED_P_ARRAYS):
            raise RewriteError(
                f"{where}: stores {', '.join(stored)} without "
                f"{', '.join(sorted(STORED_P_ARRAYS - set(stored)))}; a 0.2 tier stores both "
                "(spec §18), so the tier is not rewritten"
            )
        if not stored_p_agrees(tier["abs_z"][:], tier["p_value"][:], tier["z"][:]):
            raise RewriteError(
                f"{where}: its stored abs_z and p_value are not what its z implies; 0.3.0 "
                "recomputes them from z, so a tier that disagrees is not rewritten"
            )


def _stored_p_left_out(component: Component) -> Iterable[str]:
    """Each tier's `abs_z` and `p_value`."""
    return [
        f"data.zarr/{reader_top_hits.TOP_HIT_GROUP}/{key}/{name}"
        for key, _tier, stored in _tiers_storing_p(component)
        for name in stored
    ]


def _refuse_unmapped_dense_rows(component: Component) -> None:
    """Refuse a Hybrid whose `dense_to_shared.npy` does not map every Dense Component
    row to the shared row holding that row's own variant (#472).

    The 0.3.0 Dense Component names its rows through the map alone, so the map must be
    strictly ascending, name a shared row for each of the component's rows and no
    other, and reach, for each, a shared row whose ALID is the dropped copy's. Both
    Variant Tables are streamed once, side by side, from the source. The copies' rsID
    and `source_alid` columns are not compared: a completed release's Dense copy lost
    some of each, and the shared table, which every query already reads, keeps them
    (#481).
    """
    if not component.dense_component:
        return
    root = component.path.parent
    mapped = np.load(dense_to_shared_path(root), mmap_mode="r")
    n_dense = _offset_rows(component)
    n_shared = len(np.load(variant_offsets_path(root), mmap_mode="r"))
    where = dense_to_shared_path(root)
    if len(mapped) != n_dense:
        raise RewriteError(
            f"{where} maps {len(mapped)} rows, but the Dense Component holds {n_dense}; its "
            "copy of the variants is not dropped while the map does not name each row"
        )
    if len(mapped) and (int(mapped.min()) < 0 or int(mapped.max()) >= n_shared):
        raise RewriteError(
            f"{where} names shared rows outside 0..{n_shared - 1}; the Dense Component's "
            "copy of the variants is not dropped"
        )
    if (np.diff(mapped.astype(np.int64)) <= 0).any():
        raise RewriteError(
            f"{where} is not strictly ascending; the Dense Component's copy of the variants "
            "is not dropped while the map cannot be searched"
        )
    _refuse_other_variants(component, root, mapped)


def _refuse_other_variants(component: Component, root: Path, mapped: np.ndarray) -> None:
    """Refuse unless each Dense Component row's mapped shared row holds its ALID.

    Both Variant Axes of the source are streamed (`VariantAxis.records`), side by side.
    """
    wanted = mapped.tolist()
    with (
        closing(component.store.variant_axis(aliases=False)) as dense_axis,
        closing(open_store(root).variant_axis(aliases=False)) as shared_axis,
    ):
        checked = _check_each_row(
            component, root, wanted, dense_axis.records(), shared_axis.records()
        )
    if checked != len(wanted):
        raise RewriteError(
            f"{variant_table_path(component.path)} holds {checked} rows, but "
            f"{dense_to_shared_path(root)} maps {len(wanted)}"
        )


def _check_each_row(
    component: Component,
    root: Path,
    wanted: list[int],
    dense: Iterator[VariantRecord],
    shared: Iterator[VariantRecord],
) -> int:
    """Walk the Dense rows and the shared rows they map to; return the Dense rows seen."""
    checked = 0
    for row, record in enumerate(dense):
        _refuse_out_of_order(component, row, record)
        target = wanted[row] if row < len(wanted) else -1
        held = _alid_at(shared, target)
        if held != record.alid:
            raise RewriteError(
                f"{dense_to_shared_path(root)} maps Dense Component row {row} "
                f"({record.alid}) to shared row {target}, which holds {held or 'no row'}; "
                "the Dense Component's copy of the variants is not dropped"
            )
        checked += 1
    return checked


def _alid_at(shared: Iterator[VariantRecord], target: int) -> str | None:
    """The ALID of shared row `target`, read on from where the walk is; None if no row."""
    found = next((s for s in shared if s.variant_index >= target), None)
    return None if found is None or found.variant_index != target else found.alid


def _refuse_out_of_order(component: Component, row: int, record: VariantRecord) -> None:
    if record.variant_index != row:
        raise RewriteError(
            f"{variant_table_path(component.path)}: row {row} has variant_index "
            f"{record.variant_index}; a table out of order is not rewritten"
        )


def _dense_copies_left_out(component: Component) -> Iterable[str]:
    """A Hybrid Dense Component's own Variant Table, `.tbi`, ALID index (the 0.2
    fixed-width pair), `variant_offsets.npy`, rsID index, `analyses.tsv` and
    `overview.html` (#472): it reads the shared ones through `dense_to_shared.npy`."""
    if component.carries_axis:
        return ()
    names = (
        DENSE_COMPONENT_COPIES
        | {VARIANT_OFFSETS_FILENAME}
        | {variant_alid_bytes_path(".").name, variant_alid_rows_path(".").name}
    )
    return sorted(name for name in names if (component.path / name).exists())


#: The rewrite's steps, in the order they run (`RewriteStep`). #468-#472 each add one.
STEPS: tuple[RewriteStep, ...] = (
    RewriteStep(
        "alid_sidecar",
        "write the variable-length ALID sidecar from the Variant Table and drop the "
        "fixed-width variant_alid_bytes.npy/variant_alid_rows.npy (ADR 0105)",
        _write_alid_sidecar,
    ),
    RewriteStep(
        "position_sidecar",
        "write the position sidecar, each row's position and each chromosome's rows, "
        "from the Variant Table, so a range is a binary search and not a tabix fetch "
        "(ADR 0105 §6)",
        _write_position_sidecar,
    ),
    RewriteStep(
        "provenance_index",
        "drop index.sqlite's metadata table and variant_aliases, and with them the file "
        "from an Observed-Only release; a Reference-Completed one keeps completion_quality "
        "(ADR 0102)",
        run=_write_provenance_index,
        leaves_out=_index_left_out,
        refuse=_refuse_index_drop,
    ),
    RewriteStep(
        "tier_stored_p",
        "drop each Top-Hit tier's abs_z and p_value, which the validator recomputes from z "
        "(ADR 0102)",
        leaves_out=_stored_p_left_out,
        refuse=_refuse_stored_p_drop,
    ),
    RewriteStep(
        "dense_shared_axis",
        "drop a Hybrid Dense Component's own Variant Table, its .tbi, ALID, position and "
        "rsID indexes, analyses.tsv and overview.html; it reads the shared ones through "
        "dense_to_shared.npy, which must map each of its rows to the shared row of its "
        "own variant (#472)",
        leaves_out=_dense_copies_left_out,
        refuse=_refuse_unmapped_dense_rows,
    ),
    RewriteStep(
        "variant_offsets",
        "drop variant_offsets.npy, each row's BGZF offset in the Variant Table, which no "
        "0.3.0 query reads: the ALID sidecar counts the variants (#469)",
        run=_check_sidecars_count_offsets,
        leaves_out=_variant_offsets_left_out,
    ),
)


def rewrite_release(
    source: str | Path,
    destination: str | Path,
    *,
    steps: tuple[RewriteStep, ...] = STEPS,
) -> Path:
    """Derive a 0.3.0 release at `destination` from the 0.2.3 one at `source`."""
    source = Path(source).resolve()
    destination = Path(destination).resolve()
    refuse_destination(source, destination, error=RewriteError, named="DESTINATION")
    components = _refuse_unrewritable(source)
    for step in steps:
        for component in components:
            step.refuse(component)
    started = time.perf_counter()
    reference = release_digests(source)
    _log_phase(
        f"asked the source {len(reference['shapes'])} questions, "
        f"{len(reference['alid_probes'])} ALID probes, "
        f"{len(reference['range_probes'])} range probes and "
        f"{len(reference['misses'])} rsID misses; "
        + _unposed_said(reference["unposed_range_questions"]),
        started,
    )
    with OpenGWASDBStore.staging(destination) as staged:
        context = RewriteContext(source=source, staged=staged.path, components=components)
        started = time.perf_counter()
        left_out = _left_out(components, steps)
        _copy_release(source, staged.path, left_out)
        _log_phase(f"copied the source release, leaving out {len(left_out)} path(s)", started)
        for step in steps:
            started = time.perf_counter()
            if step.run is not None:
                step.run(context)
            _log_phase(f"step {step.name}: {step.summary}", started)
        _stamp(
            context,
            steps,
            release_id=str(uuid.uuid4()),
            now=datetime.now(UTC).isoformat(),
            unposed=reference["unposed_range_questions"],
        )
        _refresh_overviews(context)
        started = time.perf_counter()
        _require_same_answers(reference, release_digests(staged.path), destination)
        _log_phase("asked the staged release the same questions: every answer matches", started)
        started = time.perf_counter()
        require_valid_staged_release(
            staged.path, destination, error=RewriteError, what="rewritten"
        )
        _log_phase("validated the staged release", started)
    print(f"Published {destination} as {TARGET_FORMAT_VERSION}", flush=True)
    return destination


def _refuse_unrewritable(source: Path) -> tuple[Component, ...]:
    """The release's components, after refusing a source this rewrite does not take.

    Read from the manifests alone, before anything is staged. A Hybrid is two Store
    Releases, and both must be 0.2.3: a half-rewritten one is not a source.
    """
    if not (source / "manifest.json").is_file():
        raise RewriteError(f"{source}: no manifest.json; this is not a Store Release")
    top = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if top.get("primary_layout") == "hybrid" and not (source / "dense" / "manifest.json").exists():
        raise RewriteError(
            f"{source}: primary_layout is 'hybrid' but there is no nested Dense Component "
            "manifest at dense/manifest.json (spec §16)"
        )
    for path in manifest_paths(source):
        _refuse_version(path, str(json.loads(path.read_text(encoding="utf-8"))["format_version"]))
    root = open_store(source)
    if root.manifest.primary_layout is not PrimaryStorageLayout.HYBRID:
        return (Component(".", root, dense_component=False),)
    nested = root.dense_component()
    relative = str(nested.path.relative_to(root.path))
    return (
        Component(".", root, dense_component=False),
        Component(relative, nested, dense_component=True),
    )


def _refuse_version(manifest: Path, version: str) -> None:
    try:
        check_format_version(version, source=str(manifest))
    except UnsupportedFormatVersion as exc:
        raise RewriteError(str(exc)) from exc
    if carries_alid_sidecar(version):
        raise RewriteError(
            f"{manifest}: format_version is already {version}; there is nothing to rewrite"
        )
    if version != SOURCE_FORMAT_VERSION:
        raise RewriteError(
            f"{manifest}: format_version is {version}, and only a {SOURCE_FORMAT_VERSION} "
            f"release is rewritten to {TARGET_FORMAT_VERSION}. {TARGET_FORMAT_VERSION} keeps "
            "0.2.3's features: an indel's ALID spelt reference first (0.2.2, ADR 0079) and "
            "the count of associations left out for an SE above 65,504 (0.2.3, ADR 0068). "
            f"A {version} release lacks at least one, and no rewrite can add it: rebuild it "
            "from source (spec §21.4)."
        )


def _left_out(
    components: tuple[Component, ...], steps: tuple[RewriteStep, ...]
) -> frozenset[Path]:
    """Every path, relative to the release, a step drops: the copy leaves them out."""
    return frozenset(
        Path(component.relative) / relative
        for component in components
        for step in steps
        for relative in step.leaves_out(component)
    )


def _left_out_within(name: str, left_out: frozenset[Path]) -> frozenset[Path]:
    """The left-out paths beneath the entry `name`, relative to it."""
    return frozenset(
        path.relative_to(name) for path in left_out if len(path.parts) > 1 and path.parts[0] == name
    )


def _copy_release(source: Path, staged: Path, left_out: frozenset[Path] = frozenset()) -> None:
    """Copy every entry but `left_out` (paths relative to `source`); a reflink where the
    filesystem can share the blocks. A directory holding a left-out path is recreated,
    its entries copied one by one and then its mode and times, as `cp -a` does; every
    other entry is copied whole."""
    staged.mkdir(parents=True, exist_ok=True)
    for entry in sorted(source.iterdir()):
        if Path(entry.name) in left_out:
            continue
        nested = _left_out_within(entry.name, left_out)
        if nested and entry.is_dir() and not entry.is_symlink():
            _copy_release(entry, staged / entry.name, nested)
            shutil.copystat(entry, staged / entry.name)
        else:
            copy_path(entry, staged / entry.name)


def _unposed_said(unposed: list[str]) -> str:
    """The log's account of the range questions the release cannot pose."""
    if not unposed:
        return "it poses every range question"
    return f"it cannot pose {', '.join(unposed)}: its rows hold no such case"


def _stamp(
    context: RewriteContext,
    steps: tuple[RewriteStep, ...],
    *,
    release_id: str,
    now: str,
    unposed: list[str],
) -> None:
    """Stamp every manifest as one new 0.3.0 release, recording what the rewrite did.

    Every manifest gets the same fresh `release_id`, so a Hybrid's nested Dense
    Component still names the release that nests it, as the 0.2.0 converter does.
    `unposed` names the range questions the proof could not ask of the release's rows
    (`query_digests.unposed_range_questions`): there the proof is weaker, and says so.
    """
    commit = installed_commit()
    for path in manifest_paths(context.staged):
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        source_release_id = str(data["release_id"])
        data["release_id"] = release_id
        data["created_at"] = now
        data["format_version"] = format_version_for(
            carries_regions="regions" in data, reference_first_indels=True
        )
        data["provenance"] = {
            **data.get("provenance", {}),
            "format_rewrite": {
                "source_release_id": source_release_id,
                "from_format_version": SOURCE_FORMAT_VERSION,
                "to_format_version": data["format_version"],
                "steps": [step.name for step in steps],
                "range_questions_not_posed": list(unposed),
                "tool": "ogdb rewrite-release",
                "at": now,
                "opengwasdb_git_hash": commit,
                "note": (
                    "Derived by ogdb rewrite-release: each step rewrote how the source "
                    "stores what it answers, and the source and this release answered "
                    "the same questions identically before it was published. The "
                    "source release was not modified."
                ),
            },
        }
        path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _refresh_overviews(context: RewriteContext) -> None:
    """Regenerate each `overview.html` the copy carried: its header names the release."""
    for directory in context.staged_components():
        if (directory / "overview.html").exists():
            write_overview_html(directory, read_analyses(directory / "analyses.tsv"))


def _require_same_answers(
    reference: dict[str, Any], candidate: dict[str, Any], destination: Path
) -> None:
    found = differences(reference, candidate)
    if found:
        shown = "\n  ".join(found[:_SHOWN_DIFFERENCES])
        raise RewriteVerificationError(
            f"the rewritten release answers {len(found)} question(s) differently from its "
            f"source, so it is not published to {destination}; the staging directory was "
            f"removed and the source is unchanged. First differences:\n  {shown}",
            found,
        )


def _log_phase(label: str, started: float) -> None:
    print(f"  {label}: {time.perf_counter() - started:.1f}s", flush=True)


__all__ = [
    "STEPS",
    "RewriteContext",
    "RewriteError",
    "RewriteStep",
    "RewriteVerificationError",
    "rewrite_release",
]
