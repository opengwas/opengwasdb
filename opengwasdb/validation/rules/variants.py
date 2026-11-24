"""Rules that read a release's Store Variant Table, in one stream (ADR 0074).

`variants.tsv.gz` is streamed once, a block of rows at a time, and every rule
about it folds over that one read (issue #254: the table is never
materialised). The checks it feeds: the row count against the offset sidecar,
the row sequence and duplicate canonical variants, the ALID and rsid search
indexes, the position sidecar (format 0.3.0), and the offset and tabix spot
checks at a bounded set of representative rows -- which used to stream the table
a second time to reach its last row. The alias rule reads a 0.2 release's
`index.sqlite` beside it; a 0.3.0 release carries no `variant_aliases` (ADR 0102,
#471).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from opengwasdb.index.release_index import ReleaseIndex
from opengwasdb.store_reader.alid import RSID_WIDTH, is_indexable_alid, is_indexable_rsid
from opengwasdb.store_reader.enums import PrimaryStorageLayout
from opengwasdb.store_reader.format import (
    carries_alid_sidecar,
    carries_position_sidecar,
    carries_variant_aliases,
    carries_variant_offsets,
)
from opengwasdb.store_reader.identity import is_primary_contig
from opengwasdb.store_reader.position_sidecar import PositionSidecar
from opengwasdb.store_reader.variant_axis import (
    VariantRecord,
)
from opengwasdb.validation import rule as validation_rule
from opengwasdb.validation.rule import Check, Fold, Kind, Rule, Scan, Skip, Target
from opengwasdb.validation.rules.release import REQUIRED, current_builders_wrote, severity_owed
from opengwasdb.variants.normalise import chromosome_sort_key

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
    for record in target.variant_axis.records():
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
    return validation_rule.failed_words(exc)


VARIANT_TABLE: Scan[TableBlock] = Scan("variant_table", _blocks, _read_failure, _n_blocks)


def _rows(_target: Target, block: TableBlock) -> int:
    return len(block.records)


def _judge_rows(target: Target, n_rows: int) -> Iterator[str]:
    """The table holds one row per variant the axis counts: `variant_offsets.npy`'s
    entries on the 0.2 series, the ALID sidecar's rows on 0.3.0 (#469)."""
    axis = target.variant_axis
    counted_by = "variant_offsets.npy" if axis.reads_table else "the ALID sidecar"
    if axis.n_variants != n_rows:
        yield f"{counted_by} has {axis.n_variants} rows but variants.tsv.gz has {n_rows} rows"


def _indexable(_target: Target, block: TableBlock) -> Counter[str]:
    return Counter(
        rows=len(block.records),
        indexable=sum(1 for record in block.records if is_indexable_alid(record.alid)),
    )


def _judge_alid_index(target: Target, counts: Counter[str]) -> Iterator[str]:
    """The 0.2 series' ALID index covers exactly the indexable ALIDs, collision-free."""
    index = target.variant_axis.alid_index
    yield from index.problems(counts["rows"], counts["indexable"])


@dataclass(frozen=True)
class _SidecarRows:
    """How many table rows a block held, and the first whose sidecar ALID differs."""

    rows: int = 0
    differing: int = 0
    first: str = ""

    def __add__(self, other: _SidecarRows) -> _SidecarRows:
        return _SidecarRows(
            self.rows + other.rows, self.differing + other.differing, self.first or other.first
        )


def _sidecar_rows(target: Target, block: TableBlock) -> _SidecarRows:
    """Each row's ALID in the index against the table's; a row past the index's end
    names none, and is counted by its row count (`_judge_alid_sidecar`)."""
    n_rows = len(block.records)
    rows = np.arange(block.start, block.start + n_rows, dtype=np.int64)
    named = target.variant_axis.alid_index.alids(rows)
    differing = [
        k
        for k in range(n_rows)
        if named[k] is not None and named[k] != block.records[k].alid
    ]
    first = ""
    if differing:
        k = differing[0]
        first = (
            f"row {block.start + k}: the sidecar names {named[k]!r}, "
            f"variants.tsv.gz {block.records[k].alid!r}"
        )
    return _SidecarRows(n_rows, len(differing), first)


def _judge_alid_sidecar(target: Target, total: _SidecarRows) -> Iterator[str]:
    """The ALID sidecar names every row's ALID and indexes every one (format 0.3.0, ADR 0105)."""
    problems = list(target.variant_axis.alid_index.problems(total.rows, total.rows))
    yield from problems
    if not problems and total.differing:
        yield (
            f"the ALID sidecar names {total.differing} row(s) by another ALID than "
            f"variants.tsv.gz does; first, {total.first}"
        )


def _carries_alid_sidecar(target: Target) -> bool:
    return carries_alid_sidecar(target.store.manifest.format_version)


def _carries_fixed_width_alid_index(target: Target) -> bool:
    return not _carries_alid_sidecar(target)


def _carries_variant_aliases(target: Target) -> bool:
    return carries_variant_aliases(target.store.manifest.format_version)


@dataclass(frozen=True)
class _PositionRows:
    """How many table rows a block held, and the first whose sidecar chromosome or
    position differs from the table's."""

    rows: int = 0
    differing: int = 0
    first: str = ""

    def __add__(self, other: _PositionRows) -> _PositionRows:
        return _PositionRows(
            self.rows + other.rows, self.differing + other.differing, self.first or other.first
        )


def _position_sidecar(target: Target) -> PositionSidecar:
    sidecar = target.variant_axis.position_sidecar
    if sidecar is None:
        raise ValueError("a 0.3.0 Variant Axis opened without its position sidecar")
    return sidecar


def _position_rows(target: Target, block: TableBlock) -> _PositionRows:
    """Each row's chromosome and position in the sidecar against the table's
    (`PositionSidecar.misplaced`); a row past the sidecar's end is counted by its row
    count (`_judge_position_sidecar`)."""
    sidecar = _position_sidecar(target)
    records = block.records
    misplaced = sidecar.misplaced(
        block.start, [r.chromosome for r in records], [r.position for r in records]
    )
    first = ""
    if misplaced:
        row = misplaced[0]
        record = records[row - block.start]
        [(chromosome, position)] = sidecar.placements([row])
        first = (
            f"row {row}: the sidecar places it at {chromosome}:{position}, "
            f"variants.tsv.gz at {record.chromosome}:{record.position}"
        )
    return _PositionRows(len(records), len(misplaced), first)


def _judge_position_sidecar(target: Target, total: _PositionRows) -> Iterator[str]:
    """The position sidecar places every row where the table does, in genomic order
    (format 0.3.0, ADR 0105 §6): a range is two binary searches in it."""
    sidecar = _position_sidecar(target)
    problems = list(sidecar.problems(total.rows, chromosome_key=chromosome_sort_key))
    yield from problems
    if not problems and total.differing:
        yield (
            f"the position sidecar places {total.differing} row(s) elsewhere than "
            f"variants.tsv.gz does; first, {total.first}"
        )


def _carries_position_sidecar(target: Target) -> bool:
    return carries_position_sidecar(target.store.manifest.format_version)


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
    rsid_index = target.variant_axis.rsid_index
    indexed = None if rsid_index is None else rsid_index.names
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
class _RsidRows:
    """How many table rows a block held, and the first whose rsID the rsID index names
    otherwise than the table does."""

    rows: int = 0
    differing: int = 0
    first: str = ""

    def __add__(self, other: _RsidRows) -> _RsidRows:
        return _RsidRows(
            self.rows + other.rows, self.differing + other.differing, self.first or other.first
        )


def _rsid_rows(target: Target, block: TableBlock) -> _RsidRows:
    """Each row's rsID as the inverted rsID index names it against the table's cell.

    A row past the axis's end is not compared: `variant_table.rows` reports it.
    """
    axis = target.variant_axis
    held = max(min(len(block.records), axis.n_variants - block.start), 0)
    rows = np.arange(block.start, block.start + held, dtype=np.int64)
    named = axis.rsids_by_indices(rows)
    differing = [k for k in range(held) if named[k] != block.records[k].rsid]
    first = ""
    if differing:
        k = differing[0]
        cell = block.records[k].rsid
        why = "" if cell is None or is_indexable_rsid(cell) else (
            f", an rsID wider than {RSID_WIDTH} bytes, which the index cannot hold"
        )
        first = (
            f"row {block.start + k}: the index names {named[k]!r}, "
            f"variants.tsv.gz {cell!r}{why}"
        )
    return _RsidRows(len(block.records), len(differing), first)


def _judge_rsid_rows(_target: Target, total: _RsidRows) -> Iterator[str]:
    """The rsID index names every row's rsID as the table does (format 0.3.0, #469): a
    0.3.0 Store Reader names a row's rsID from the index alone, never the table."""
    if total.differing:
        yield (
            f"the rsID index names {total.differing} row(s) otherwise than variants.tsv.gz "
            f"does, so variants_table() and resolved rows would give another rsID; first, "
            f"{total.first}"
        )


def _reads_rsids_from_index(target: Target) -> bool:
    return not carries_variant_offsets(target.store.manifest.format_version)


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
    """The tabix index finds the row the table holds, and on the 0.2 series so does the
    row's offset in `variant_offsets.npy`.

    No 0.3.0 query reads the table or its tabix index, but both still ship as
    conveniences restating the sidecars (spec §1.1), so the tabix index is checked here.
    A 0.3.0 release has no offsets (#469): its rows are placed by the position sidecar,
    which `variant_table.position_sidecar` judges row by row.
    The rows are chosen from the axis's count, so the check means something only when
    the table holds the same number of rows. It runs after `variant_table.rows`,
    which judged that from the same read, so a mismatch skips it behind that rule.
    """
    axis = target.variant_axis
    for index in representative_variant_indices(axis.n_variants):
        record = picked.records[index]
        if axis.reads_table:
            try:
                offset_record = axis.by_index(index)
            except ValueError as exc:
                yield str(exc)
                return
            if offset_record != record:
                yield f"variant offset for row {index} points to a different row"
                return
        if record not in axis.tabix_range(record.chromosome, record.position, record.position):
            yield f"tabix index cannot fetch variant {record.alid}"
            return


@dataclass(frozen=True)
class _IndelOrder:
    """How many indel rows spell their ALID against the table's own orientation,
    and the first of them."""

    n_rows: int = 0
    first: str = ""

    def __add__(self, other: _IndelOrder) -> _IndelOrder:
        return _IndelOrder(self.n_rows + other.n_rows, self.first or other.first)


def _indel_order(_target: Target, block: TableBlock) -> _IndelOrder:
    """Each indel row whose ALID order the table's other columns contradict.

    An indel's ALID spells its reference allele first (ADR 0079), and the table
    records that order twice more: the row's `effect_allele` and
    `other_allele`, which a statistic is signed for, and, for a lifted variant,
    its `source_alid`, the same variant spelt on the source's assembly, where
    a lift complements an allele but never changes its length.
    """
    n_rows, first = 0, ""
    for offset, record in enumerate(block.records):
        _chrom, _pos, a1, a2 = record.alid.split(":")
        if len(a1) == len(a2):
            continue
        fault = _order_fault(record, a1, a2)
        if fault:
            n_rows += 1
            first = first or f"row {block.start + offset}: {fault}"
    return _IndelOrder(n_rows, first)


def _order_fault(record: VariantRecord, a1: str, a2: str) -> str:
    """Why one indel row's ALID order contradicts the row, or ``""``."""
    alid, effect, other = record.alid, record.effect_allele, record.other_allele
    if (a1, a2) != (effect, other):
        return f"ALID {alid} but effect_allele {effect!r} and other_allele {other!r}"
    source = (record.source_alid or "").split(":")
    if len(source) == 4 and len(source[2]) != len(a1):
        return f"ALID {alid} but source_alid {record.source_alid}, the other allele first"
    return ""


def _judge_indel_order(_target: Target, found: _IndelOrder) -> Iterator[str]:
    if found.n_rows:
        yield (
            f"{found.n_rows} indel row(s) of variants.tsv.gz spell their ALID in an order "
            f"the table's own orientation contradicts ({found.first}): an indel's ALID "
            "spells its reference allele first (ADR 0079), and a statistic is signed for "
            "its effect_allele"
        )


@dataclass(frozen=True)
class _LiftedOffPrimary:
    """How many table rows name a lifted variant on a non-primary contig, the first,
    and how many non-primary rows have no ``source_alid`` to judge them by."""

    n_rows: int = 0
    first: str = ""
    n_unjudged: int = 0

    def __add__(self, other: _LiftedOffPrimary) -> _LiftedOffPrimary:
        return _LiftedOffPrimary(
            self.n_rows + other.n_rows,
            self.first or other.first,
            self.n_unjudged + other.n_unjudged,
        )


def _lifted_off_primary(_target: Target, block: TableBlock) -> _LiftedOffPrimary:
    """Each row on an alt, random or unplaced contig whose ``source_alid`` is on another.

    A source on the release's assembly keeps its alt contig, so its ``source_alid``
    is the ALID itself; a lift's is the source's own coordinate, on another contig.
    A row on such a contig with no ``source_alid`` (no lift, or a collision blanked
    it) cannot be judged, and is counted so the message can say so.
    """
    n_rows, first, n_unjudged = 0, "", 0
    for offset, record in enumerate(block.records):
        if is_primary_contig(record.chromosome):
            continue
        if record.source_alid is None:
            n_unjudged += 1
        elif record.source_alid.split(":", 1)[0] != record.chromosome:
            n_rows += 1
            first = first or f"row {block.start + offset}: {record.alid} from {record.source_alid}"
    return _LiftedOffPrimary(n_rows, first, n_unjudged)


def _unjudged(found: _LiftedOffPrimary) -> str:
    return f"{found.n_unjudged} not judged (no source ALID)"


def _judge_lifted_off_primary(_target: Target, found: _LiftedOffPrimary) -> Iterator[str]:
    if found.n_rows:
        yield (
            f"{found.n_rows} variant(s) of variants.tsv.gz were lifted onto an alt, random or "
            f"unplaced contig ({found.first}), {_unjudged(found)}: a build leaves such a "
            "variant out and counts its rows as lift_off_primary (#376), so this release was "
            "built before that, or routed by a Variant Reference extracted before it -- "
            "rebuild it (an earlier format's release is #367's)"
        )


@dataclass(frozen=True, kw_only=True)
class _ErrorFromFormat(Fold[TableBlock, _LiftedOffPrimary]):
    """A fold that fails a release of a format whose builders cannot write what it finds
    (`current_builders_wrote`) and warns on an earlier one (`severity_owed`), which says
    what the release is without failing a Store nobody can fix in place (#376, #367).

    Where it finds nothing but cannot judge some rows, it warns that it did not."""

    def judged(self, target: Target, total: _LiftedOffPrimary) -> Iterator[validation_rule.Finding]:
        severity = validation_rule.Severity
        level = severity_owed(target, severity.ERROR)
        for message in self.judge(target, total):
            yield self.finding(target, message, level)
        if not total.n_rows and total.n_unjudged:
            yield self.finding(
                target,
                f"{_unjudged(total)}: variants.tsv.gz has non-primary-contig variants with no "
                "source_alid, which cannot be told from a lift's",
                severity.WARNING,
            )


@dataclass(frozen=True)
class _SourceAlidRows:
    """How many table rows there are, and how many of them carry a ``source_alid``."""

    n_rows: int = 0
    n_sourced: int = 0

    def __add__(self, other: _SourceAlidRows) -> _SourceAlidRows:
        return _SourceAlidRows(self.n_rows + other.n_rows, self.n_sourced + other.n_sourced)


def _judges_source_alid(target: Target) -> bool:
    """Whether the release records a lift its own table must then show, and is not a
    Hybrid's Dense Component: that copy of the table is gone at 0.3.0 (#472), so a
    Hybrid is judged on its shared table alone.

    A lift is recorded as a ``provenance.lift`` block that lifted a variant (Ragged), or a
    ``provenance.chain_file`` (Dense, Hybrid). The second is a weak signal: a build writes
    a default chain label whether or not it lifted anything, and no Dense or Hybrid
    manifest records its Analyses' source assemblies. It cannot wrongly fail a release,
    though: every build now writes an unlifted row's ``source_alid`` as its ALID, so a
    release that lifted nothing still holds sourced rows. A ``lift`` block that does not
    say how many it lifted is judged, and refused by `_judge_source_alid_rows`.
    """
    provenance = target.store.manifest.provenance
    if target.is_component or not isinstance(provenance, dict):
        return False
    lift = provenance.get("lift")
    if isinstance(lift, dict):
        n_lifted = lift.get("n_lifted")
        return not isinstance(n_lifted, int) or n_lifted > 0
    return bool(provenance.get("chain_file"))


def _unreadable_lift(target: Target) -> str | None:
    """Why a ``provenance.lift`` block cannot say whether it lifted, or ``None``."""
    lift = target.store.manifest.provenance.get("lift")
    if isinstance(lift, dict) and not isinstance(lift.get("n_lifted"), int):
        return (
            f"manifest.json provenance.lift records n_lifted={lift.get('n_lifted')!r}, not a "
            "count: the release cannot say whether it lifted, so source_alid cannot be judged"
        )
    return None


def _source_alid_rows(_target: Target, block: TableBlock) -> _SourceAlidRows:
    sourced = sum(1 for record in block.records if record.source_alid)
    return _SourceAlidRows(len(block.records), sourced)


def _judge_source_alid_rows(target: Target, found: _SourceAlidRows) -> Iterator[str]:
    """A lifted release has an observed row with a ``source_alid``, or none was kept.

    Rows Reference Completion adds from the panel are blank by design and a collision
    blanks its row, so one blank row is no fault. A table with *no* ``source_alid`` at
    all holds none on any observed row: the provenance was dropped (#449).
    """
    unreadable = _unreadable_lift(target)
    if unreadable:
        yield unreadable
    elif found.n_rows and not found.n_sourced:
        yield (
            f"manifest.json records a liftover but none of the {found.n_rows} rows of "
            "variants.tsv.gz carries a source_alid: a build or Reference Completion dropped "
            "the lift's provenance (#449) -- rebuild the release from source"
        )


def variant_aliases(target: Target) -> Iterator[str]:
    """Every alias names a variant the table holds.

    SQLite holds no variant rows (issue #128), so this is the one variant check
    against `index.sqlite`; a release whose index has no alias table has nothing
    for it to check, which is not the same as passing it.
    """
    with ReleaseIndex(target.store.path) as index:
        if "variant_aliases" not in index.names(tables_only=False):
            raise Skip("index.sqlite has no variant_aliases table")
        rows = index.variant_aliases()
    n_variants = target.variant_axis.n_variants
    for alias, stored_index in rows:
        variant_index = int(stored_index)
        if variant_index < 0 or variant_index >= n_variants:
            yield f"alias {alias!r} points to missing variant {variant_index}"


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
        when=_carries_fixed_width_alid_index,
        scan=VARIANT_TABLE,
        zero=Counter[str](),
        measure=_indexable,
        judge=_judge_alid_index,
    ),
    Fold(
        name="variant_table.alid_sidecar",
        kinds=Kind.RELEASE,
        after=(REQUIRED,),
        when=_carries_alid_sidecar,
        scan=VARIANT_TABLE,
        zero=_SidecarRows(),
        measure=_sidecar_rows,
        judge=_judge_alid_sidecar,
    ),
    Fold(
        name="variant_table.position_sidecar",
        kinds=Kind.RELEASE,
        after=(REQUIRED,),
        when=_carries_position_sidecar,
        scan=VARIANT_TABLE,
        zero=_PositionRows(),
        measure=_position_rows,
        judge=_judge_position_sidecar,
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
        name="variant_table.rsid_rows",
        kinds=Kind.RELEASE,
        after=(REQUIRED,),
        when=_reads_rsids_from_index,
        scan=VARIANT_TABLE,
        zero=_RsidRows(),
        measure=_rsid_rows,
        judge=_judge_rsid_rows,
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
    Fold(
        name="variant_table.indel_order",
        kinds=Kind.RELEASE,
        after=(REQUIRED,),
        # A Store before 0.2.2, 0.2.1 included, sorts its indels (ADR 0079).
        when=current_builders_wrote,
        scan=VARIANT_TABLE,
        zero=_IndelOrder(),
        measure=_indel_order,
        judge=_judge_indel_order,
    ),
    _ErrorFromFormat(
        name="variant_table.lifted_contig",
        kinds=Kind.RELEASE,
        after=(REQUIRED,),
        scan=VARIANT_TABLE,
        zero=_LiftedOffPrimary(),
        measure=_lifted_off_primary,
        judge=_judge_lifted_off_primary,
    ),
    Fold(
        name="variant_table.source_alid",
        kinds=Kind.RELEASE,
        when=_judges_source_alid,
        after=(REQUIRED,),
        scan=VARIANT_TABLE,
        zero=_SourceAlidRows(),
        measure=_source_alid_rows,
        judge=_judge_source_alid_rows,
    ),
    Check(
        name="sqlite.variant_aliases",
        kinds=Kind.RELEASE,
        when=_carries_variant_aliases,
        after=(ROWS,),
        run=variant_aliases,
    ),
)
