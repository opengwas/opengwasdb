"""Rules that read a release's Store Variant Table, in one stream (ADR 0074).

`variants.tsv.gz` is streamed once, a block of rows at a time, and every rule
about it folds over that one read (issue #254: the table is never
materialised). The checks it feeds: the row count against the offset sidecar,
the row sequence and duplicate canonical variants, the ALID and rsid search
indexes, and the offset and tabix spot checks at a bounded set of
representative rows -- which used to stream the table a second time to reach
its last row. The alias rule reads `index.sqlite` beside it.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from opengwasdb.model.enums import PrimaryStorageLayout
from opengwasdb.validation.rule import Check, Fold, Kind, Rule, Scan, Skip, Target
from opengwasdb.validation.rules.release import REQUIRED
from opengwasdb.variants import (
    VariantRecord,
    is_indexable_alid,
    is_indexable_rsid,
    iter_variant_records,
    variant_alid_bytes_path,
    variant_table_path,
)

#: Rows one block of the stream holds: enough to amortise the per-block work,
#: few enough that a block's records are a small, bounded allocation.
_BLOCK_ROWS = 65_536

#: The fold every rule that needs the table and the offsets to agree runs after.
ROWS = "variant_table.rows"


@dataclass(frozen=True)
class TableBlock:
    """Consecutive rows of the Store Variant Table, and the row before them."""

    start: int
    records: list[VariantRecord]
    previous: VariantRecord | None


def _block_limits(n_rows: int | None) -> Iterator[tuple[int, int]]:
    """Every block of a table, by row: the one place the size of one is set.

    `n_rows` is how long the table claims to be, or None while the stream is the only
    thing that knows. The count passes the offsets sidecar's length; the read passes
    None, so a sidecar that is wrong -- or unreadable -- moves no block boundary and
    changes nothing about what the folds measure (issue #308).
    """
    start = 0
    while n_rows is None or start < n_rows:
        stop = start + _BLOCK_ROWS if n_rows is None else min(start + _BLOCK_ROWS, n_rows)
        yield start, stop
        start = stop


def _n_blocks(target: Target) -> int:
    """How many blocks `_blocks` reads for a table as long as its offsets sidecar.

    The count is the sidecar's, because how long the stream is is what the read is
    finding out: a table that disagrees with the sidecar -- which
    `variant_table.rows` reports -- runs past this count, and its progress line then
    never reaches `n/N`. The alternative, counting rows first, would read the table
    twice (issue #308).
    """
    return sum(1 for _ in _block_limits(target.variant_axis.n_variants))


def _blocks(target: Target) -> Iterator[TableBlock]:
    """Consecutive rows of the table, `_block_limits` at a time, until the stream ends."""
    limits = _block_limits(None)
    limit = next(limits)
    previous: VariantRecord | None = None
    start = 0
    records: list[VariantRecord] = []
    for record in iter_variant_records(variant_table_path(target.store.path)):
        records.append(record)
        if start + len(records) >= limit[1]:
            yield TableBlock(start, records, previous)
            previous, start, records = records[-1], start + len(records), []
            limit = next(limits)
    if records:
        yield TableBlock(start, records, previous)


def _read_failure(target: Target, exc: Exception) -> str:
    if target.layout is PrimaryStorageLayout.HYBRID:
        return f"hybrid shared-table validation failed: {exc}"
    return f"validation failed: {exc}"


VARIANT_TABLE: Scan[TableBlock] = Scan("variant_table", _blocks, _read_failure, _n_blocks)


def _rows(_target: Target, block: TableBlock) -> int:
    return len(block.records)


def _judge_rows(target: Target, n_rows: int) -> Iterator[str]:
    n_offsets = target.variant_axis.n_variants
    if n_offsets != n_rows:
        yield f"variant_offsets.npy has {n_offsets} rows but variants.tsv.gz has {n_rows} rows"


def _indexable(_target: Target, block: TableBlock) -> Counter[str]:
    return Counter(
        rows=len(block.records),
        indexable=sum(1 for record in block.records if is_indexable_alid(record.alid)),
    )


def _judge_alid_index(target: Target, counts: Counter[str]) -> Iterator[str]:
    """The ALID search index covers exactly the indexable ALIDs, collision-free."""
    keys = target.variant_axis._alid_bytes
    if keys is None:
        return
    # One entry per *indexable* ALID, not per variant: an ALID too wide for the
    # slot is left out rather than truncated into a key it shares with another
    # variant (issue #127). Those resolve by exact scan.
    if len(keys) != counts["indexable"]:
        yield (
            f"variant_alid_bytes.npy has {len(keys)} entries but "
            f"variants.tsv.gz has {counts['indexable']} ALID(s) narrow enough to index "
            f"(of {counts['rows']} rows)"
        )
    collisions = _adjacent_repeats(variant_alid_bytes_path(target.store.path))
    if collisions:
        yield (
            f"variant_alid_bytes.npy holds {collisions} key(s) shared by more than "
            "one variant: a lookup by one variant's ALID returns another's row. "
            "This is what silently truncating an over-wide ALID produced before "
            "issue #127 — rebuild the store"
        )


#: Keys one collision window compares: 1,000,000 32-byte keys is 32 MB.
_KEY_WINDOW = 1_000_000


def _adjacent_repeats(path: Path) -> int:
    """How many keys of the sorted ALID index equal the key before them.

    Read a window at a time, each window mapped and released on its own, so the
    index -- 387 MB on OGS-00006 -- is never held whole; one key of overlap carries
    the comparison across a window boundary.
    """
    n_keys = len(np.load(path, mmap_mode="r"))
    repeats = 0
    for lo in range(0, max(n_keys - 1, 0), _KEY_WINDOW):
        window = np.array(np.load(path, mmap_mode="r")[lo : lo + _KEY_WINDOW + 1])
        repeats += int(np.count_nonzero(window[1:] == window[:-1]))
    return repeats


def _named_rsids(_target: Target, block: TableBlock) -> int:
    return sum(1 for record in block.records if is_indexable_rsid(record.rsid))


def _judge_rsid_index(target: Target, named_rsids: int) -> Iterator[str]:
    """Every rsid in `variants.tsv.gz` must be resolvable (issue #109).

    The failure this exists to catch is silent by construction: a store whose
    variant table is full of rsids but whose index is empty answers every rsid
    lookup with "nothing here", which reads exactly like a real negative. A store
    with no rsids at all is fine -- not every source names variants -- and a
    *missing* index is only an error when there are rsids to index: stores built
    before #109 have none of these sidecars, and an rsid-less one is not wrong,
    merely older. `named_rsids` is counted through the same predicate the index is
    written by, so a correctly built store cannot fail over a deliberately
    omitted identifier.
    """
    indexed = target.variant_axis._rsid_bytes
    if indexed is None:
        if named_rsids:
            yield (
                f"variants.tsv.gz names {named_rsids} rsid(s) but the store has no rsid "
                "search index — rebuild the store, or every rsid lookup against it "
                "returns nothing"
            )
        return
    if len(indexed) != named_rsids:
        yield (
            f"variant_rsid_bytes.npy has {len(indexed)} entries but variants.tsv.gz "
            f"names {named_rsids} rsid(s)"
        )


@dataclass(frozen=True)
class _FirstFault:
    """The first row out of sequence, or the first repeated canonical variant."""

    row: int | None = None
    message: str = ""

    def __add__(self, other: _FirstFault) -> _FirstFault:
        return self if self.row is not None else other


def _sequence(_target: Target, block: TableBlock) -> _FirstFault:
    """The table is in `variant_index` order and holds each canonical variant once.

    It is written ALID-sorted, so a duplicate canonical variant sits beside its
    twin; the row before the block carries the comparison across its start.
    """
    previous_alid = block.previous.alid if block.previous is not None else None
    for offset, record in enumerate(block.records):
        expected = block.start + offset
        if record.variant_index != expected:
            return _FirstFault(
                expected, f"variant table row {expected} has variant_index {record.variant_index}"
            )
        if record.alid == previous_alid:
            return _FirstFault(expected, "duplicate canonical variants in variant table")
        previous_alid = record.alid
    return _FirstFault()


def _judge_sequence(_target: Target, fault: _FirstFault) -> Iterator[str]:
    if fault.row is not None:
        yield fault.message


@lru_cache(maxsize=8)
def representative_variant_indices(n_variants: int) -> tuple[int, ...]:
    """A bounded, spread set of rows: every row up to 1,000, else about 1,000 of them."""
    if n_variants <= 0:
        return ()
    if n_variants <= 1000:
        return tuple(range(n_variants))
    anchors = {0, n_variants // 2, n_variants - 1}
    step = max(1, n_variants // 997)
    anchors.update(range(0, n_variants, step))
    return tuple(sorted(index for index in anchors if 0 <= index < n_variants))


@dataclass(frozen=True)
class _Picked:
    """The table's records at the representative rows."""

    records: dict[int, VariantRecord] = field(default_factory=dict)

    def __add__(self, other: _Picked) -> _Picked:
        return _Picked({**self.records, **other.records})


def _pick(target: Target, block: TableBlock) -> _Picked:
    wanted = representative_variant_indices(target.variant_axis.n_variants)
    lo = int(np.searchsorted(wanted, block.start))
    hi = int(np.searchsorted(wanted, block.start + len(block.records)))
    picked = {index: block.records[index - block.start] for index in wanted[lo:hi]}
    return _Picked(picked)


def _judge_representatives(target: Target, picked: _Picked) -> Iterator[str]:
    """The offset sidecar and the tabix index each find the row the table holds.

    The rows are chosen from the offsets, so the check means something only when
    the table holds the same number of rows. It runs after `variant_table.rows`,
    which judged that from the same read, so a mismatch skips it behind that rule.
    """
    axis = target.variant_axis
    for index in representative_variant_indices(axis.n_variants):
        record = picked.records[index]
        try:
            offset_record = axis.by_index(index)
        except ValueError as exc:
            yield str(exc)
            return
        if offset_record != record:
            yield f"variant offset for row {index} points to a different row"
            return
        if record not in axis.range(record.chromosome, record.position, record.position):
            yield f"tabix index cannot fetch variant {record.alid}"
            return


def variant_aliases(target: Target) -> Iterator[str]:
    """Every alias names a variant the table holds.

    SQLite holds no variant rows (issue #128), so this is the one variant check
    against `index.sqlite`; a release whose index has no alias table has nothing
    for it to check, which is not the same as passing it.
    """
    with target.store.index_connection() as connection:
        tables = {r[0] for r in connection.execute("SELECT name FROM sqlite_master")}
        if "variant_aliases" not in tables:
            raise Skip("index.sqlite has no variant_aliases table")
        rows = connection.execute("SELECT alias, variant_index FROM variant_aliases").fetchall()
    n_variants = target.variant_axis.n_variants
    for row in rows:
        variant_index = int(row["variant_index"])
        if variant_index < 0 or variant_index >= n_variants:
            yield f"alias {row['alias']!r} points to missing variant {variant_index}"


RULES: tuple[Rule, ...] = (
    Fold(
        name=ROWS,
        kinds=Kind.RELEASE,
        after=(REQUIRED,),
        scan=VARIANT_TABLE,
        zero=0,
        measure=_rows,
        judge=_judge_rows,
    ),
    Fold(
        name="variant_table.alid_index",
        kinds=Kind.RELEASE,
        after=(REQUIRED,),
        scan=VARIANT_TABLE,
        zero=Counter[str](),
        measure=_indexable,
        judge=_judge_alid_index,
    ),
    Fold(
        name="variant_table.rsid_index",
        kinds=Kind.RELEASE,
        after=(REQUIRED,),
        scan=VARIANT_TABLE,
        zero=0,
        measure=_named_rsids,
        judge=_judge_rsid_index,
    ),
    Fold(
        name="variant_table.sequence",
        kinds=Kind.RELEASE,
        after=(REQUIRED,),
        scan=VARIANT_TABLE,
        zero=_FirstFault(),
        measure=_sequence,
        judge=_judge_sequence,
    ),
    Fold(
        name="variant_table.representatives",
        kinds=Kind.RELEASE,
        after=(REQUIRED, ROWS),
        scan=VARIANT_TABLE,
        zero=_Picked(),
        measure=_pick,
        judge=_judge_representatives,
    ),
    Check(
        name="sqlite.variant_aliases",
        kinds=Kind.RELEASE,
        after=(ROWS,),
        run=variant_aliases,
    ),
)
