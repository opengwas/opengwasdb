"""The Store Variant Axis: a Store Release's variants, by index, identifier and position.

A Store's variants are its Store Variant Table, one row per Store-local Variant
Index, and the mmap'd search indexes beside it: the ALID index
(`opengwasdb.store_reader.alid_index`: format 0.3.0's sidecar or the 0.2 series'
fixed-width pair), the sorted rsID index (`opengwasdb.store_reader.rsid_index`, #109,
#127) and, from format 0.3.0, the position sidecar
(`opengwasdb.store_reader.position_sidecar`), which answers a range. `VariantAxis`
answers every variant question a query asks from them.

**From format 0.3.0 the Store Reader never reads the Variant Table** (#469, ADR 0105
§7). A row's chromosome, position and alleles are its ALID's, which the ALID sidecar
names; its rsID is the rsID index's, inverted; the variant count is the ALID sidecar's.
`variants_table()`, `identity_by_indices` and `rsids_by_indices` answer from those, and
`variant_offsets.npy` is not written. The table and its `.tbi` still ship, a convenience
for tabix, bedtools and R that the validator checks against the sidecars (spec §1.1).

In format 0.2.x the table itself is read: a BGZF file whose rows `variant_offsets.npy`
locates and whose tabix index answers a 0.2 release's ranges. A Store built before the
rsID index resolves rsIDs through `index.sqlite`'s `variant_aliases` table. Neither
belongs in the Store Reader, which may import neither pysam nor sqlite3 (#464): the
axis asks a `LegacyVariantTable` for those reads, and opengwasdb supplies one
(`opengwasdb.variants.legacy_table.TabixVariantTable`). It opens its files on first use
(#465). The record-returning methods (`by_index`, `by_indices`, `all`, `records`,
`range`, `by_identifier`) read the table on every version, because only the table
holds a row's `source_alid`: the builders, completion and the validator ask them, and
no query does. #476 deletes the legacy seam.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from opengwasdb.store_reader.alid import ParsedAlid, parse_canonical_alid
from opengwasdb.store_reader.alid_index import ABSENT, UNINDEXED, AlidIndex, open_alid_index
from opengwasdb.store_reader.alid_sidecar import AlidSidecar
from opengwasdb.store_reader.format import carries_position_sidecar, carries_variant_offsets
from opengwasdb.store_reader.identity import normalise_chromosome
from opengwasdb.store_reader.position_sidecar import MissingPositionSidecar, PositionSidecar
from opengwasdb.store_reader.rsid_index import (
    VARIANT_RSID_BYTES_FILENAME,
    VARIANT_RSID_ROWS_FILENAME,
    MissingRsidIndex,
    RsidIndex,
)

VARIANT_TABLE_FILENAME = "variants.tsv.gz"
VARIANT_TABIX_FILENAME = "variants.tsv.gz.tbi"
VARIANT_OFFSETS_FILENAME = "variant_offsets.npy"
VARIANT_ALID_BYTES_FILENAME = "variant_alid_bytes.npy"
VARIANT_ALID_ROWS_FILENAME = "variant_alid_rows.npy"

class LegacyVariantTable(Protocol):
    """The 0.2.x Store Variant Table and `variant_aliases` reads, which need pysam and SQLite.

    What a `VariantAxis` asks of a Store's BGZF/tabix Variant Table and of its
    `index.sqlite`, and the only reads of either on the query path. opengwasdb
    implements it (`opengwasdb.variants.axis.LegacyVariantTable`); the Store
    Reader never imports that, and a format without these files needs none.
    Lines are the table's rows as text, header lines excluded, newline kept.
    """

    def fetch(self, chromosome: str, start: int, end: int) -> Iterable[str]:
        """The rows in the half-open, 0-based region; none for a chromosome the table lacks."""
        ...

    def line_at(self, offset: int) -> str:
        """The one row at a BGZF virtual offset from `variant_offsets.npy`."""
        ...

    def rows(self) -> Iterator[str]:
        """Every row, in `variant_index` order."""
        ...

    def alias_index(self, alias: str) -> int | None:
        """The lowest Variant Index `index.sqlite`'s `variant_aliases` gives `alias`, if any."""
        ...

    def close(self) -> None: ...


class LegacyTableOpener(Protocol):
    """Opens a Store's `LegacyVariantTable`; `aliases` asks for its `variant_aliases` too."""

    def __call__(self, store_path: Path, *, aliases: bool) -> LegacyVariantTable: ...


# `by_indices()` switchover point between random-access and full-scan
# resolution -- see that method's docstring for the measured costs behind it.
_BULK_SCAN_FRACTION = 0.01

@dataclass(frozen=True)
class VariantRecord(Mapping[str, Any]):
    """One Store Variant Table row."""

    variant_index: int
    #: The row's canonical identity, ``chromosome:position:effect_allele:other_allele``.
    alid: str
    chromosome: str
    position: int
    effect_allele: str
    other_allele: str
    #: None where the table's cell is blank (``.``).
    rsid: str | None
    # Canonical ALID of the source association on the *build* (pre-liftover)
    # assembly that occupies this row — the store's provenance link back to the
    # original dataset. None for same-assembly builds (no liftover) and for rows
    # where several source variants collided onto one lifted ALID (ambiguous).
    source_alid: str | None = None

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return iter(
            (
                "variant_index",
                "alid",
                "chromosome",
                "position",
                "effect_allele",
                "other_allele",
                "rsid",
                "source_alid",
            )
        )

    def __len__(self) -> int:
        return 8


def variant_table_path(store_path: str | Path) -> Path:
    return Path(store_path) / VARIANT_TABLE_FILENAME


def variant_tabix_path(store_path: str | Path) -> Path:
    return Path(store_path) / VARIANT_TABIX_FILENAME


def variant_offsets_path(store_path: str | Path) -> Path:
    return Path(store_path) / VARIANT_OFFSETS_FILENAME


def variant_alid_bytes_path(store_path: str | Path) -> Path:
    return Path(store_path) / VARIANT_ALID_BYTES_FILENAME


def variant_alid_rows_path(store_path: str | Path) -> Path:
    return Path(store_path) / VARIANT_ALID_ROWS_FILENAME


def variant_rsid_bytes_path(store_path: str | Path) -> Path:
    return Path(store_path) / VARIANT_RSID_BYTES_FILENAME


def variant_rsid_rows_path(store_path: str | Path) -> Path:
    return Path(store_path) / VARIANT_RSID_ROWS_FILENAME


class VariantAxis:
    """Read a Store's variants: its search indexes here, its Variant Table through `legacy`."""

    def __init__(
        self, store_path: str | Path, legacy: LegacyVariantTable, *, format_version: str
    ):
        self.store_path = Path(store_path)
        self.table_path = variant_table_path(self.store_path)
        self.tabix_path = variant_tabix_path(self.store_path)
        self.offsets_path = variant_offsets_path(self.store_path)
        # Each row's BGZF offset in the table: the 0.2 series' only. A 0.3.0 release
        # carries none, and its Store Reader never reads the table (#469).
        self._offsets: np.ndarray | None = (
            np.load(self.offsets_path, mmap_mode="r")
            if carries_variant_offsets(format_version)
            else None
        )
        self._legacy = legacy
        # The ALID index the release's `format_version` carries, and no other
        # (`open_alid_index`): format 0.3.0's sidecar or the 0.2 series' fixed-width pair.
        # A release missing it is refused here, never read by scanning the table.
        self._alid_index = open_alid_index(self.store_path, format_version)
        self._n_variants = (
            len(self._offsets) if self._offsets is not None else _sidecar_rows(self._alid_index)
        )
        # Whether ranges are answered by format 0.3.0's position sidecar (ADR 0105 §6)
        # or, on the 0.2 series, by a tabix fetch through `legacy`. A 0.3.0 release
        # missing the sidecar's files is refused here; the sidecar is read on first use
        # (`position_sidecar`), so a malformed one fails only what reads it.
        self._ranges_by_sidecar = carries_position_sidecar_files(self.store_path, format_version)
        self._position_sidecar: PositionSidecar | None = None
        # The rsID index (`rsid_index`), present on stores built after issue #109.
        # Absent on older 0.2 stores, where rsid lookups find nothing (as they did
        # before that index existed). On 0.3.0 it is also how a row's rsID is named,
        # so a 0.3.0 release without it is refused rather than read as naming none.
        self._rsid_index = RsidIndex.open(self.store_path)
        if self._rsid_index is None and not self.reads_table:
            raise MissingRsidIndex(
                f"{self.store_path}: format_version {format_version} names each row's rsID "
                f"from {VARIANT_RSID_BYTES_FILENAME} and {VARIANT_RSID_ROWS_FILENAME}, which "
                "are missing; "
                "the release is incomplete. Rebuild it (spec §1, §4)"
            )

    @property
    def n_variants(self) -> int:
        """How many variants the release holds: the ALID sidecar's rows on 0.3.0, the
        entries of `variant_offsets.npy` on the 0.2 series."""
        return int(self._n_variants)

    @property
    def rsid_index(self) -> RsidIndex | None:
        """The release's rsID index (`rsid_index`); None on a 0.2 release built before it."""
        return self._rsid_index

    @property
    def reads_table(self) -> bool:
        """Whether this release's queries read its Variant Table: the 0.2 series' do,
        0.3.0's never do (#469)."""
        return self._offsets is not None

    @property
    def alid_index(self) -> AlidIndex:
        """The release's ALID index (`opengwasdb.store_reader.alid_index`)."""
        return self._alid_index

    @property
    def position_sidecar(self) -> PositionSidecar | None:
        """The release's position sidecar (format 0.3.0), opened on first use; None on
        the 0.2 series. One whose chromosomes cannot slice its rows is refused
        (`position_sidecar.MalformedPositionSidecar`)."""
        if self._ranges_by_sidecar and self._position_sidecar is None:
            self._position_sidecar = PositionSidecar.open(self.store_path)
            if self._position_sidecar is None:  # removed since this axis was opened
                raise MissingPositionSidecar(f"{self.store_path}: the position sidecar is gone")
        return self._position_sidecar

    def close(self) -> None:
        self._legacy.close()

    def by_identifier(self, identifier: str) -> VariantRecord | None:
        """Resolve one ALID or rsid to a single Store Variant Table row.

        An rsid can legitimately name several rows -- a multi-allelic site, or
        one position stored under both allele orders. This method's contract is
        one record, so it returns the lowest `variant_index` among them, the
        same choice the `variant_aliases` fallback below has always made
        (``ORDER BY variant_index LIMIT 1``). Callers that need every row an
        rsid names want `indices_by_identifiers`, which returns all of them.
        A caller that needs only the row's index wants `index_of`, which reads
        no Variant Table row.
        """
        parsed = parse_canonical_alid(identifier)
        if parsed is not None:
            return self.by_alid(parsed)
        index = self.index_of(identifier)
        return None if index is None else self.by_index(index)

    def index_of(self, identifier: str) -> int | None:
        """The one Variant Index `by_identifier` resolves `identifier` to, or None.

        The lowest row among those `indices_by_identifiers` finds, so it reads no
        Variant Table row a search index could answer.
        """
        indices = self.indices_by_identifiers([identifier])
        return int(indices.min()) if len(indices) else None

    def indices_by_alias(self, alias: str) -> np.ndarray:
        """Every Store-local Variant Index the rsid `alias` names, ascending.

        Empty when the store has no rsid index (built before issue #109), when
        the rsid is unknown, or when `alias` is the table's blank marker `.` --
        a row with no rsid is not reachable by asking for one.
        """
        # An identifier too long to encode was never indexed (see
        # `_write_rsid_index`), so a truncated query could only ever produce a
        # false match for a different variant: `rows_of` asks no such name.
        if self._rsid_index is None:
            return np.empty(0, dtype="int32")
        return self._rsid_index.rows_of(alias)

    def indices_by_identifiers(self, identifiers: Sequence[str]) -> np.ndarray:
        """Resolve many identifiers to Store-local Variant Indices in one
        batched pass, without materialising a `VariantRecord` (and its
        per-call `variants.tsv.gz` BGZF open/seek, see `by_index`) for each
        one -- the fast path dense `lookup()` needs (issue #3).

        Canonical ALIDs are resolved with a single vectorised `searchsorted`
        over the mmap'd ALID index; rsids through the equivalent rsid index
        (issue #109), falling back to the `variant_aliases` SQLite table for
        stores built before it. Unlike `by_identifier`, an rsid contributes
        *every* row it names, not only the lowest-indexed one. Identifiers that
        don't resolve are dropped, matching `by_identifier()`'s None-on-miss
        semantics.

        The order: the ALIDs' rows first, in the order the ALIDs were asked, then
        the rsids' rows, each rsid's rows ascending, in the order the rsids were
        asked. A Dense `lookup` keeps this order; a Ragged or Hybrid `lookup`
        returns its associations in row order whatever it was asked.
        """
        parsed_pairs, alias_identifiers = _split_identifiers(identifiers)
        indices = self._indices_of_alids(parsed_pairs)
        indices.extend(self._indices_of_aliases(alias_identifiers))
        return np.array(indices, dtype="int32")

    def _indices_of_alids(self, parsed_pairs: list[tuple[str, ParsedAlid]]) -> list[int]:
        """The rows the ALIDs name, in the order asked: the ALID index first, the exact
        scan for any it cannot be asked about (one too wide for the 0.2 series' slot,
        #127).

        In the order asked on either index: a 0.2 release's scanned rows once came after
        its indexed ones, so a Dense lookup naming a long ALID listed its rows in another
        order than the same lookup of the 0.3.0 release rewritten from it (#467).
        """
        rows = self._alid_index.search([query for query, _ in parsed_pairs])
        indices: list[int] = []
        for (_, parsed), row in zip(parsed_pairs, rows.tolist(), strict=True):
            if row >= 0:
                indices.append(int(row))
            elif row == UNINDEXED:
                record = self._scanned_record(parsed)
                if record is not None:
                    indices.append(record.variant_index)
        return indices

    def _indices_of_aliases(self, aliases: list[str]) -> list[int]:
        """Every row each rsid names, not just the first: a phewas over rs123 at
        a multi-allelic site wants both stored rows.

        That holds for an rsid the sidecar rsid index holds. On a release without
        that index (built before issue #109) the `variant_aliases` fallback
        answers, and it is `LIMIT 1`: one row, the lowest.
        """
        indices: list[int] = []
        for identifier in aliases:
            matched = self.indices_by_alias(identifier)
            if len(matched):
                indices.extend(int(index) for index in matched)
                continue
            index = self._legacy.alias_index(identifier)
            if index is not None:
                indices.append(index)
        return indices

    def by_alid(self, parsed: ParsedAlid) -> VariantRecord | None:
        row = int(self._alid_index.search([_canonical(parsed)])[0])
        if row == UNINDEXED:
            return self._scanned_record(parsed)
        return None if row == ABSENT else self.by_index(row)

    def _scanned_record(self, parsed: ParsedAlid) -> VariantRecord | None:
        """The exact scan of the table at the ALID's position, for one the index lacks."""
        for record in self.range(parsed.chromosome, parsed.position, parsed.position):
            if (
                record.effect_allele == parsed.effect_allele
                and record.other_allele == parsed.other_allele
            ):
                return record
        return None

    def range(self, chromosome: str, start: int, end: int) -> list[VariantRecord]:
        """The rows at a position in ``[start, end]`` on `chromosome`, in row order, every
        column of each.

        Reads the Variant Table on every version, so no query calls it: a query asks
        `range_indices`. On 0.3.0 the position sidecar finds the rows and `by_indices`
        reads them, one tabix fetch a row; on the 0.2 series they are one tabix fetch
        (`tabix_range`).
        """
        if not self._ranges_by_sidecar:
            return self.tabix_range(chromosome, start, end)
        rows = self.range_indices(chromosome, start, end)
        records = self.by_indices(rows.tolist())
        return [records[int(row)] for row in rows]

    def tabix_range(self, chromosome: str, start: int, end: int) -> list[VariantRecord]:
        """`range`, read through the Variant Table's tabix index on any format.

        A 0.2 release's ranges are this. The validator asks it of a 0.3.0 release
        too, whose `variants.tsv.gz.tbi` is a convenience it still checks (spec §1.1).
        """
        chrom = normalise_chromosome(chromosome)
        lines = self._legacy.fetch(chrom, max(0, int(start) - 1), int(end))
        return [parse_variant_line(line) for line in lines]

    def range_indices(self, chromosome: str, start: int, end: int) -> np.ndarray:
        """The Variant Indices at a position in ``[start, end]`` on `chromosome`, ascending,
        as int32, without reading a row.

        On 0.3.0, two binary searches in the chromosome's slice of the position sidecar;
        on the 0.2 series, a tabix fetch. The chromosome is normalised either way, and
        either answers nothing for a range whose end is before its start, or for a
        chromosome the release lacks (`TabixVariantTable.fetch` answers tabix's refusal
        of both with no rows).
        """
        chrom = normalise_chromosome(chromosome)
        sidecar = self.position_sidecar
        if sidecar is not None:
            return sidecar.rows_in(chrom, start, end).astype("int32")
        lines = self._legacy.fetch(chrom, max(0, int(start) - 1), int(end))
        indices = []
        for line in lines:
            # variant_index is the third tab-separated field (index 2)
            fields = line.split("\t", 3)
            indices.append(int(fields[2]))
        return np.array(indices, dtype="int32")

    def by_index(self, variant_index: int) -> VariantRecord | None:
        """The table's row `variant_index`, every column of it; None outside the axis.

        Read from the table on every version, because only the table holds `source_alid`;
        no query asks it. The 0.2 series seeks the row's BGZF offset. A 0.3.0 release has
        no offsets: the position sidecar places the row, and the tabix index fetches it.
        """
        if variant_index < 0 or variant_index >= self.n_variants:
            return None
        if self._offsets is None:  # `not self.reads_table`, spelt so mypy narrows `_offsets`
            return self._fetched_record(int(variant_index))
        line = self._legacy.line_at(int(self._offsets[int(variant_index)]))
        record = parse_variant_line(line)
        if record.variant_index != variant_index:
            raise ValueError(
                f"variant offset for row {variant_index} points to row {record.variant_index}"
            )
        return record

    def _fetched_record(self, variant_index: int) -> VariantRecord:
        """Row `variant_index` of a 0.3.0 table, fetched by tabix where the sidecar places it."""
        sidecar = self.position_sidecar
        assert sidecar is not None, "a release without offsets carries the position sidecar"
        [(chromosome, position)] = sidecar.placements([variant_index])
        if chromosome is None or position is None:
            raise ValueError(
                f"the position sidecar holds {sidecar.n_rows} rows and places no row "
                f"{variant_index}, of the {self.n_variants} the ALID sidecar names"
            )
        for line in self._legacy.fetch(chromosome, position - 1, position):
            record = parse_variant_line(line)
            if record.variant_index == variant_index:
                return record
        raise ValueError(
            f"the position sidecar places row {variant_index} at {chromosome}:{position}, "
            "where the tabix index of variants.tsv.gz finds no such row"
        )

    def rsids_by_indices(self, variant_index: np.ndarray) -> np.ndarray:
        """Each row's rsID, as an object array in `variant_index`'s order; None for a row
        that names none (the table's blank, ``.``) or that the axis lacks.

        On 0.3.0 the rsID index, inverted (`RsidIndex.rsids`): the table is not read. On
        the 0.2 series the table's rows (`by_indices`), as it always was.
        """
        vi = np.asarray(variant_index, dtype=np.int64).reshape(-1)
        if not self.reads_table:
            assert self._rsid_index is not None, "a 0.3.0 axis is refused without one"
            return self._rsid_index.rsids(vi, self.n_variants)
        records = self.by_indices(vi.tolist())
        out = np.full(vi.size, None, dtype=object)
        for slot, row in enumerate(vi.tolist()):
            record = records.get(int(row))
            out[slot] = None if record is None else record.rsid
        return out

    def variants_table(self) -> dict[int, dict[str, Any]]:
        """Every variant keyed by `variant_index`: its ALID, chromosome, position, alleles
        and rsID (None where it names none).

        On 0.3.0 from the sidecars: the ALID sidecar names each row, its ALID spells the
        chromosome, position and alleles (`identity_by_indices`), and the rsID index names
        its rsID. On the 0.2 series from the table's rows. The two agree column by column
        (`tests/test_variant_table_off_read_path.py`).
        """
        if self.reads_table:
            return {r.variant_index: _table_row(r) for r in self.all()}
        rows = np.arange(self.n_variants, dtype=np.int64)
        columns = {name: values.tolist() for name, values in self.identity_by_indices(rows).items()}
        columns["rsid"] = self.rsids_by_indices(rows).tolist()
        return {
            row: {name: columns[name][row] for name in VARIANTS_TABLE_COLUMNS}
            for row in rows.tolist()
        }

    def by_indices(self, indices: Iterable[int]) -> dict[int, VariantRecord]:
        """Resolve many Store-local Variant Indices to `VariantRecord`s, every column.

        Reads the Variant Table on every version, so no query calls it: a query names a
        row by `identity_by_indices` and `rsids_by_indices`, which on 0.3.0 read the
        sidecars. On 0.3.0 a seek is one tabix fetch a row (`by_index`).

        Adaptively picks between two strategies (issue #104 follow-up,
        measured on the UKB pilot store's ~9.85M-row Store Variant Table):
        a random-access `by_index()` seek costs ~300us/row regardless of
        whether the underlying BGZF handle is reused (the cost is the
        seek-and-decompress-from-the-nearest-block itself, not file-open
        overhead), while one sequential `all()` scan costs ~1-2us/row. The
        two cross over well under 1% of the table, so once a caller wants
        more than `_BULK_SCAN_FRACTION` of it, one full scan beats N random
        seeks by orders of magnitude -- the difference between a
        sub-second dense "query-analysis" resolution and one that never
        finishes in practice.
        """
        wanted = sorted(set(int(item) for item in indices))
        if not wanted:
            return {}
        if len(wanted) > self.n_variants * _BULK_SCAN_FRACTION:
            wanted_set = set(wanted)
            return {
                record.variant_index: record
                for record in self.all()
                if record.variant_index in wanted_set
            }
        records: dict[int, VariantRecord] = {}
        for index in wanted:
            if (record := self.by_index(index)) is not None:
                records[index] = record
        return records

    def identity_by_indices(self, variant_index: np.ndarray) -> dict[str, np.ndarray]:
        """chromosome/position/effect_allele/other_allele/alid for `variant_index`,
        resolved without a table scan.

        An ALID *is* `chromosome:position:effect_allele:other_allele`
        (`parse_canonical_alid`), so the ALID index's row -> ALID answers all five.
        On 0.3.0 the sidecar names every row by two reads. On a 0.2 release the
        fixed-width index inverts its permutation once and leaves out each ALID
        wider than its slot (#127: OGS-00010 carries 115,043); those rows are
        sought one at a time with `by_index()`, bounded by how many a query
        returns. `rsid` is the one identity field this can't produce: it isn't
        part of the ALID, so it's the only reason a caller still needs `by_indices()`.
        """
        vi = np.asarray(variant_index, dtype="int64")
        alids = self._alid_index.alids(vi)
        for slot in [k for k, alid in enumerate(alids) if alid is None]:
            record = self.by_index(int(vi[slot]))
            if record is None:
                raise ValueError(f"no variant row at index {int(vi[slot])}")
            alids[slot] = record.alid
        return _identity_of(alids)

    def all(self) -> list[VariantRecord]:
        """Every row of the Variant Table, every column, in ``variant_index`` order.

        Reads the whole table on every version, so no query calls it: `variants_table()`
        reads the sidecars on 0.3.0. The builders, completion and the validator ask it,
        for `source_alid` (`records`, which streams the same rows).
        """
        return list(self.records())

    def records(self) -> Iterator[VariantRecord]:
        """Every row, streamed in ``variant_index`` order: what `all` materialises.

        Reads the whole table on every version, so no query calls it (`all`).

        A caller that needs one field per row -- the EAF orientation audit, the
        rsid-index rule -- would otherwise hold tens of millions of records at
        once on the stores it most needs to run on (#473).
        """
        return (parse_variant_line(line) for line in self._legacy.rows())

    def indices_by_source_alid(self, wanted: Iterable[str]) -> dict[str, int] | None:
        """The Variant Index of each `wanted` ALID, as the build assembly spelt it.

        Matched against each row's ``source_alid`` as `parse_variant_line` reads it
        (a blank, ``.``, is none, and matches nothing), in one pass that stops once
        every wanted ALID is found; one the table does not hold is absent from the
        result.
        None when the table predates the ``source_alid`` column, so the caller
        can lift the ALIDs over itself rather than read no match as no variant (#473).
        """
        found: dict[str, int] = {}
        remaining = set(wanted)
        for line in self._legacy.rows():
            if not carries_source_alid(line):
                return None
            record = parse_variant_line(line)
            if record.source_alid in remaining:
                found[record.source_alid] = record.variant_index
                remaining.discard(record.source_alid)
                if not remaining:
                    break
        return found


def _sidecar_rows(index: AlidIndex) -> int:
    """The variant count of a release without `variant_offsets.npy`: its ALID sidecar's
    rows, one per row of the table (`variant_table.alid_sidecar` holds it to that)."""
    if not isinstance(index, AlidSidecar):
        raise TypeError(
            "a release without variant_offsets.npy counts its variants from the ALID sidecar, "
            f"but its ALID index is a {type(index).__name__}"
        )
    return index.n_rows


#: A `variants_table()` row's columns, in order: a table record's but `variant_index`
#: and `source_alid`.
VARIANTS_TABLE_COLUMNS = ("alid", "chromosome", "position", "effect_allele", "other_allele", "rsid")


def _table_row(record: VariantRecord) -> dict[str, Any]:
    """One `variants_table()` row from a table record (`VARIANTS_TABLE_COLUMNS`)."""
    return {name: record[name] for name in VARIANTS_TABLE_COLUMNS}


def carries_position_sidecar_files(store_path: str | Path, format_version: str) -> bool:
    """Whether a release of `format_version` answers ranges from its position sidecar:
    True from 0.3.0, False on the 0.2 series. A 0.3.0 release missing the sidecar's
    files is refused, never read by tabix: the version, not the files present, says how
    a range is answered."""
    if not carries_position_sidecar(format_version):
        return False
    if not PositionSidecar.held_in(store_path):
        raise MissingPositionSidecar(
            f"{store_path}: format_version {format_version} carries a position sidecar in "
            f"{', '.join(PositionSidecar.FILENAMES)}, which is missing; the release is "
            "incomplete. Rebuild it, or rewrite it from an intact source (spec §1, §4)"
        )
    return True


#: A Store Variant Table row's fields before the trailing ``source_alid`` column,
#: which a table written before that column ends at.
_FIELDS_BEFORE_SOURCE_ALID = 7


def carries_source_alid(line: str) -> bool:
    """Whether a Store Variant Table row has the ``source_alid`` column."""
    return line.rstrip("\n").count("\t") + 1 > _FIELDS_BEFORE_SOURCE_ALID


def parse_variant_line(line: str) -> VariantRecord:
    """One Store Variant Table row, as the table spells it, as a `VariantRecord`."""
    fields = line.rstrip("\n").split("\t")
    # 7 fields = pre-source_alid stores; 8 = current (trailing source_alid column).
    if len(fields) == _FIELDS_BEFORE_SOURCE_ALID:
        chromosome, position, variant_index, effect_allele, other_allele, alid, rsid = fields
        source_alid = "."
    elif len(fields) == 8:
        (
            chromosome, position, variant_index, effect_allele, other_allele,
            alid, rsid, source_alid,
        ) = fields
    else:
        raise ValueError(f"variant row has {len(fields)} fields, expected 7 or 8")
    return VariantRecord(
        variant_index=int(variant_index),
        alid=alid,
        chromosome=chromosome,
        position=int(position),
        effect_allele=effect_allele,
        other_allele=other_allele,
        rsid=None if rsid in {"", "."} else rsid,
        source_alid=None if source_alid in {"", "."} else source_alid,
    )


def _identity_of(alids: np.ndarray) -> dict[str, np.ndarray]:
    """The four identity fields an ALID spells, and the ALID itself, per row."""
    if not alids.size:
        empty = np.empty(0, dtype=object)
        return {
            "chromosome": empty,
            "position": np.empty(0, dtype="int64"),
            "effect_allele": empty,
            "other_allele": empty,
            "alid": empty,
        }
    parts = np.array([a.split(":") for a in alids], dtype=object)
    return {
        "chromosome": parts[:, 0],
        "position": parts[:, 1].astype("int64"),
        "effect_allele": parts[:, 2],
        "other_allele": parts[:, 3],
        "alid": alids,
    }


def _canonical(parsed: ParsedAlid) -> str:
    """The ALID as the table and the indexes spell it."""
    return f"{parsed.chromosome}:{parsed.position}:{parsed.effect_allele}:{parsed.other_allele}"


def _split_identifiers(
    identifiers: Sequence[str],
) -> tuple[list[tuple[str, ParsedAlid]], list[str]]:
    """The canonical ALIDs among `identifiers`, each with its canonical spelling,
    and the rest, which are asked of the rsid indexes."""
    parsed_pairs: list[tuple[str, ParsedAlid]] = []
    alias_identifiers: list[str] = []
    for identifier in identifiers:
        parsed = parse_canonical_alid(identifier)
        if parsed is None:
            alias_identifiers.append(identifier)
        else:
            query = (
                f"{parsed.chromosome}:{parsed.position}:"
                f"{parsed.effect_allele}:{parsed.other_allele}"
            )
            parsed_pairs.append((query, parsed))
    return parsed_pairs, alias_identifiers
