"""The Store Variant Axis: a Store Release's variants, by index, identifier and position.

A Store's variants are its Store Variant Table, one row per Store-local Variant
Index, and the mmap'd search indexes beside it: `variant_offsets.npy` (each row's
place in the table, which also fixes how many variants the Store holds), the
sorted ALID index and the sorted rsid index (#109, #127). `VariantAxis` answers
every variant question a query asks from them.

In format 0.2.x the table itself is a BGZF file with a tabix index, which only
pysam reads, and a Store built before the rsid index resolves rsids through
`index.sqlite`'s `variant_aliases` table. Neither belongs in the Store Reader,
which may import neither pysam nor sqlite3 (#464): the axis asks a
`LegacyVariantTable` for those reads, and opengwasdb supplies one
(`opengwasdb.variants.legacy_table.TabixVariantTable`). It opens its files on
first use (#465): an identifier a search index holds resolves, and every result
resolves, without the table; #476 deletes it.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from opengwasdb.store_reader.alid import (
    ALID_DTYPE,
    RSID_DTYPE,
    ParsedAlid,
    is_indexable_alid,
    is_indexable_rsid,
    parse_canonical_alid,
)
from opengwasdb.store_reader.identity import normalise_chromosome

VARIANT_TABLE_FILENAME = "variants.tsv.gz"
VARIANT_TABIX_FILENAME = "variants.tsv.gz.tbi"
VARIANT_OFFSETS_FILENAME = "variant_offsets.npy"
VARIANT_ALID_BYTES_FILENAME = "variant_alid_bytes.npy"
VARIANT_ALID_ROWS_FILENAME = "variant_alid_rows.npy"
VARIANT_RSID_BYTES_FILENAME = "variant_rsid_bytes.npy"
VARIANT_RSID_ROWS_FILENAME = "variant_rsid_rows.npy"

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

    def __init__(self, store_path: str | Path, legacy: LegacyVariantTable):
        self.store_path = Path(store_path)
        self.table_path = variant_table_path(self.store_path)
        self.tabix_path = variant_tabix_path(self.store_path)
        self.offsets_path = variant_offsets_path(self.store_path)
        self._offsets = np.load(self.offsets_path, mmap_mode="r")
        self._legacy = legacy
        # mmap'd ALID search index — present on stores built after issue 029.
        # Falls back gracefully to tabix-per-call if absent (older stores).
        _bytes_path = variant_alid_bytes_path(self.store_path)
        _rows_path = variant_alid_rows_path(self.store_path)
        if _bytes_path.exists() and _rows_path.exists():
            self._alid_bytes: np.ndarray | None = np.load(_bytes_path, mmap_mode="r")
            self._alid_rows: np.ndarray | None = np.load(_rows_path, mmap_mode="r")
        else:
            self._alid_bytes = None
            self._alid_rows = None
        # mmap'd rsid search index — present on stores built after issue #109.
        # Absent on older stores, where rsid lookups find nothing (as they did
        # before that index existed).
        _rsid_bytes_path = variant_rsid_bytes_path(self.store_path)
        _rsid_rows_path = variant_rsid_rows_path(self.store_path)
        if _rsid_bytes_path.exists() and _rsid_rows_path.exists():
            self._rsid_bytes: np.ndarray | None = np.load(_rsid_bytes_path, mmap_mode="r")
            self._rsid_rows: np.ndarray | None = np.load(_rsid_rows_path, mmap_mode="r")
        else:
            self._rsid_bytes = None
            self._rsid_rows = None
        self._alid_inverse: np.ndarray | None = None  # built lazily, see identity_by_indices()

    @property
    def n_variants(self) -> int:
        return int(len(self._offsets))

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
        index = self.index_of(identifier)
        return None if index is None else self.by_index(index)

    def index_of(self, identifier: str) -> int | None:
        """The Store-local Variant Index one ALID or rsid names, or None.

        What `by_identifier` resolves before it reads the row. An identifier a
        search index holds is answered from the mmap'd index alone; only a long
        ALID (over `ALID_WIDTH` bytes, in no index), or an rsid the rsid index
        lacks and `variant_aliases` may name, reaches the legacy Variant Table.
        """
        parsed = parse_canonical_alid(identifier)
        if parsed is not None:
            return self._index_of_alid(parsed)
        indices = self.indices_by_alias(identifier)
        if len(indices):
            return int(indices[0])
        return self._legacy.alias_index(identifier)

    def alid_of(self, variant_index: int) -> str | None:
        """The canonical ALID of one row, from the ALID index when it holds the row."""
        identity = self.identity_by_indices(np.array([variant_index], dtype="int64"))
        if identity is not None:
            return str(identity["alid"][0])
        record = self.by_index(variant_index)
        return None if record is None else record.alid

    def indices_by_alias(self, alias: str) -> np.ndarray:
        """Every Store-local Variant Index the rsid `alias` names, ascending.

        Empty when the store has no rsid index (built before issue #109), when
        the rsid is unknown, or when `alias` is the table's blank marker `.` --
        a row with no rsid is not reachable by asking for one.
        """
        rsid_bytes, rsid_rows = self._rsid_bytes, self._rsid_rows
        # An identifier too long to encode was never indexed (see
        # `_write_rsid_index`), so a truncated query could only ever produce a
        # false match for a different variant.
        if rsid_bytes is None or rsid_rows is None or not is_indexable_rsid(alias):
            return np.empty(0, dtype="int32")
        query = np.array(alias, dtype=RSID_DTYPE)
        lo = int(np.searchsorted(rsid_bytes, query, side="left"))
        hi = int(np.searchsorted(rsid_bytes, query, side="right"))
        if lo == hi:
            return np.empty(0, dtype="int32")
        return np.asarray(rsid_rows[lo:hi], dtype="int32")

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
        semantics. The returned order is not guaranteed to match the input
        order.
        """
        parsed_pairs, alias_identifiers = _split_identifiers(identifiers)
        indices = self._indices_of_alids(parsed_pairs)
        indices.extend(self._indices_of_aliases(alias_identifiers))
        return np.array(indices, dtype="int32")

    def _indices_of_alids(self, parsed_pairs: list[tuple[str, ParsedAlid]]) -> list[int]:
        """The rows the ALIDs name: the mmap'd index first, the exact scan otherwise."""
        alid_bytes, alid_rows = self._alid_bytes, self._alid_rows
        indices: list[int] = []
        if alid_bytes is not None and alid_rows is not None:
            # Over-wide ALIDs are not in the index and must not be cast into a
            # query against it: the cast truncates, and a truncated query
            # matches a different variant's key (issue #127). They take the
            # exact scan, the same split `by_alid` makes for one identifier.
            indexable, scanned = _by_indexability(parsed_pairs)
            indices.extend(_indexed_rows(alid_bytes, alid_rows, indexable))
        else:
            scanned = [p for _, p in parsed_pairs]
        for parsed in scanned:
            record = self.by_alid(parsed)
            if record is not None:
                indices.append(record.variant_index)
        return indices

    def _indices_of_aliases(self, aliases: list[str]) -> list[int]:
        """Every row each rsid names, not just the first: a phewas over rs123 at
        a multi-allelic site wants both stored rows."""
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
        index = self._index_of_alid(parsed)
        return None if index is None else self.by_index(index)

    def _index_of_alid(self, parsed: ParsedAlid) -> int | None:
        alid = (
            f"{parsed.chromosome}:{parsed.position}:"
            f"{parsed.effect_allele}:{parsed.other_allele}"
        )
        # Only ask the index about ALIDs it could hold. A wider one was never
        # indexed (issue #127), and casting it to the slot width to ask would
        # match a *different* variant's truncated key -- the defect itself.
        if self._alid_bytes is not None and is_indexable_alid(alid):
            query = np.array([alid], dtype=ALID_DTYPE)
            idx = int(np.searchsorted(self._alid_bytes, query[0]))
            if idx < len(self._alid_bytes) and self._alid_bytes[idx] == query[0]:
                return int(self._alid_rows[idx])
            return None
        for record in self.range(parsed.chromosome, parsed.position, parsed.position):
            if (
                record.effect_allele == parsed.effect_allele
                and record.other_allele == parsed.other_allele
            ):
                return record.variant_index
        return None

    def range(self, chromosome: str, start: int, end: int) -> list[VariantRecord]:
        chrom = normalise_chromosome(chromosome)
        lines = self._legacy.fetch(chrom, max(0, int(start) - 1), int(end))
        return [parse_variant_line(line) for line in lines]

    def range_indices(self, chromosome: str, start: int, end: int) -> np.ndarray:
        """Return variant indices for a genomic range as int32 array, without object allocation."""
        chrom = normalise_chromosome(chromosome)
        lines = self._legacy.fetch(chrom, max(0, int(start) - 1), int(end))
        indices = []
        for line in lines:
            # variant_index is the third tab-separated field (index 2)
            fields = line.split("\t", 3)
            indices.append(int(fields[2]))
        return np.array(indices, dtype="int32")

    def by_index(self, variant_index: int) -> VariantRecord | None:
        if variant_index < 0 or variant_index >= self.n_variants:
            return None
        line = self._legacy.line_at(int(self._offsets[int(variant_index)]))
        record = parse_variant_line(line)
        if record.variant_index != variant_index:
            raise ValueError(
                f"variant offset for row {variant_index} points to row {record.variant_index}"
            )
        return record

    def by_indices(self, indices: Iterable[int]) -> dict[int, VariantRecord]:
        """Resolve many Store-local Variant Indices to `VariantRecord`s.

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

    def _alids_by_indices(self, positions: np.ndarray, vi: np.ndarray) -> np.ndarray:
        """Decode the ALIDs of requested rows from the mmap'd index.

        `positions` is each row's slot in the sorted index, or -1 for rows
        whose long-allele ALID was left out of it (#127); those few are
        sought from the table by `by_index()`, so the fallback is correct
        and bounded by how many unindexed rows a query actually returns.
        """
        alid_bytes = np.asarray(self._alid_bytes)
        indexed = positions >= 0
        if indexed.all():
            return np.array(
                [b.decode("utf-8") for b in alid_bytes[positions]], dtype=object
            )
        alids: np.ndarray = np.empty(vi.size, dtype=object)
        alids[indexed] = [b.decode("utf-8") for b in alid_bytes[positions[indexed]]]
        for slot in np.flatnonzero(~indexed):
            record = self.by_index(int(vi[slot]))
            if record is None:
                raise ValueError(f"no variant row at index {int(vi[slot])}")
            alids[slot] = record.alid
        return alids

    def identity_by_indices(self, variant_index: np.ndarray) -> dict[str, np.ndarray] | None:
        """chromosome/position/effect_allele/other_allele/alid for `variant_index`,
        resolved without a table scan.

        An ALID *is* `chromosome:position:effect_allele:other_allele`
        (`parse_canonical_alid`), and the mmap'd ALID search index
        (`_alid_bytes`/`_alid_rows`, already loaded at construction for
        `by_alid()`) already holds the reverse of what's needed here:
        `_alid_rows[k]` is the variant_index of the k-th alid in sorted
        order. Inverting it once (cached on this instance) gives
        `variant_index -> sorted position` and, from there, alid-by-index
        via pure numpy fancy indexing. When the index covers every row
        (issue 029 stores whose ALIDs all fit the fixed width), that is the
        whole resolution: zero `by_index()`/`by_indices()` seeks or scans.

        It is not the whole resolution when ALIDs exceed the fixed index
        width (#127): those rows are absent from the index — a *partial*
        permutation — and are recovered one seek per row via `by_index()`.
        A production store can carry 100k+ such rows (OGS-00010 carries
        115,043 long-allele ALIDs); an earlier revision of this method
        assumed fullness and raised `ValueError: shape mismatch` for any
        request spanning an unindexed row.
        `rsid` is the one identity field this can't produce: it isn't part
        of the alid, so it's the only reason a caller still needs
        `by_indices()`.

        Returns `None` when the mmap'd ALID index isn't present (stores
        built before issue 029) -- callers should fall back to
        `by_indices()` there.
        """
        if self._alid_bytes is None or self._alid_rows is None:
            return None
        vi = np.asarray(variant_index, dtype="int64")
        if vi.size == 0:
            empty = np.empty(0, dtype=object)
            return {
                "chromosome": empty,
                "position": np.empty(0, dtype="int64"),
                "effect_allele": empty,
                "other_allele": empty,
                "alid": empty,
            }
        if self._alid_inverse is None:
            rows = np.asarray(self._alid_rows, dtype="int64")
            inverse = np.full(self.n_variants, -1, dtype="int64")
            inverse[rows] = np.arange(rows.size, dtype="int64")
            self._alid_inverse = inverse
        alids = self._alids_by_indices(self._alid_inverse[vi], vi)
        parts = np.array([a.split(":") for a in alids], dtype=object)
        return {
            "chromosome": parts[:, 0],
            "position": parts[:, 1].astype("int64"),
            "effect_allele": parts[:, 2],
            "other_allele": parts[:, 3],
            "alid": alids,
        }

    def all(self) -> list[VariantRecord]:
        return [parse_variant_line(line) for line in self._legacy.rows()]


def parse_variant_line(line: str) -> VariantRecord:
    """One Store Variant Table row, as the table spells it, as a `VariantRecord`."""
    fields = line.rstrip("\n").split("\t")
    # 7 fields = pre-source_alid stores; 8 = current (trailing source_alid column).
    if len(fields) == 7:
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


def _indexed_rows(alid_bytes: np.ndarray, alid_rows: np.ndarray, queries: list[str]) -> list[int]:
    """The rows of the queries the sorted ALID index holds, by one vectorised search."""
    if not queries:
        return []
    keys = np.array(queries, dtype=ALID_DTYPE)
    positions = np.searchsorted(alid_bytes, keys)
    in_bounds = positions < len(alid_bytes)
    hit = np.zeros(len(keys), dtype=bool)
    hit[in_bounds] = alid_bytes[positions[in_bounds]] == keys[in_bounds]
    return [int(row) for row in alid_rows[positions[hit]]]


def _by_indexability(
    parsed_pairs: list[tuple[str, ParsedAlid]],
) -> tuple[list[str], list[ParsedAlid]]:
    """The ALIDs the fixed-width index can hold, and the rest, which take the exact scan."""
    indexable = [query for query, _ in parsed_pairs if is_indexable_alid(query)]
    scanned = [parsed for query, parsed in parsed_pairs if not is_indexable_alid(query)]
    return indexable, scanned
