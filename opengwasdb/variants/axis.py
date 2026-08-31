"""Tabix-backed Store Variant Axis."""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pysam

from opengwasdb.variants.normalise import (
    CanonicalVariant,
    VariantNormalisationError,
    normalise_allele,
    normalise_chromosome,
)

log = logging.getLogger(__name__)

VARIANT_TABLE_FILENAME = "variants.tsv.gz"
VARIANT_TABIX_FILENAME = "variants.tsv.gz.tbi"
VARIANT_OFFSETS_FILENAME = "variant_offsets.npy"
VARIANT_ALID_BYTES_FILENAME = "variant_alid_bytes.npy"
VARIANT_ALID_ROWS_FILENAME = "variant_alid_rows.npy"
VARIANT_RSID_BYTES_FILENAME = "variant_rsid_bytes.npy"
VARIANT_RSID_ROWS_FILENAME = "variant_rsid_rows.npy"
VARIANT_AXIS_FORMAT = "tabix_tsv_v1"
VARIANT_HEADER = (
    "#chromosome\tposition\tvariant_index\teffect_allele\tother_allele\talid\trsid"
    "\tsource_alid\n"
)
# Fixed byte-width for ALID encoding in the mmap'd search index. Fixed, because
# that is what lets `np.searchsorted` run over the array as an mmap without
# parsing it. Measured across the FinnGen, GWAS Catalog and metabolome pilots,
# ALID length is mean 14.9 and p99 21; the slack is for indels.
#
# An ALID wider than this is *left out* of the index, never cast into it:
# `np.array(..., dtype="|S32")` truncates silently, and two indels at one
# position agreeing over the slot width then collapse to a single key, so a
# lookup by either full ALID returned whichever row sorted first (issue #127 --
# on the published finngen-r13 release, 342 variants answering to another's
# name). Excluded ALIDs stay reachable: `by_alid` resolves them by exact scan
# over the position instead, which is slower and right.
_ALID_WIDTH = 32
_ALID_DTYPE = f"|S{_ALID_WIDTH}"

# Fixed byte-width for rsid encoding in the mmap'd rsid search index. An rsid is
# "rs" + digits, far inside this; the width exists so a malformed or non-rs
# identifier in a source's rsid column cannot silently truncate into a *false
# match* for a different variant. Anything wider is left out of the index (and
# counted in a warning) rather than stored truncated -- see `_write_rsid_index`.
_RSID_WIDTH = 24
_RSID_DTYPE = f"|S{_RSID_WIDTH}"


def is_indexable_alid(alid: str) -> bool:
    """Whether `alid` fits the fixed-width ALID search index (issue #127).

    The one place the rule lives: `write_variant_axis` decides what to index by
    it, `VariantAxis.by_alid` decides whether a miss means "absent" or "ask the
    exact scan" by it, and `opengwasdb.validation.validate` checks the index
    holds no key shared by two variants -- which is what this rule prevents.
    """
    return len(alid.encode("utf-8")) <= _ALID_WIDTH


def is_indexable_rsid(rsid: str | None) -> bool:
    """Whether `rsid` can be resolved by name in a store (issue #109).

    The one place the rule lives: `_write_rsid_index` decides what to index by
    it, `indices_by_alias` decides what is worth searching for by it, and
    `opengwasdb.validation.validate` counts what it should find by it. Three
    copies of a width check would be three chances for a correctly-built store
    to fail its own validator.
    """
    if not rsid or rsid == ".":
        return False
    return len(rsid.encode("utf-8")) <= _RSID_WIDTH

# `by_indices()` switchover point between random-access and full-scan
# resolution -- see that method's docstring for the measured costs behind it.
_BULK_SCAN_FRACTION = 0.01


@dataclass(frozen=True)
class VariantRecord(Mapping[str, Any]):
    """One Store Variant Table row."""

    variant_index: int
    alid: str
    chromosome: str
    position: int
    effect_allele: str
    other_allele: str
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


@dataclass(frozen=True)
class ParsedAlid:
    chromosome: str
    position: int
    effect_allele: str
    other_allele: str


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


def parse_canonical_alid(identifier: str) -> ParsedAlid | None:
    """Parse a canonical ALID-like identifier, returning None for aliases."""

    parts = str(identifier).split(":")
    if len(parts) != 4:
        return None
    chromosome, position_text, effect_allele, other_allele = parts
    try:
        position = int(position_text)
        if position <= 0:
            return None
        return ParsedAlid(
            chromosome=normalise_chromosome(chromosome),
            position=position,
            effect_allele=normalise_allele(effect_allele),
            other_allele=normalise_allele(other_allele),
        )
    except (TypeError, ValueError, VariantNormalisationError):
        return None


def write_variant_axis(
    store_path: str | Path,
    variants: list[CanonicalVariant],
    rsid_by_alid: Mapping[str, str],
    source_alids: Sequence[str | None] | None = None,
) -> None:
    """Write the Store Variant Table, row-offset sidecar, and tabix index.

    ``source_alids`` (optional, parallel to ``variants``) records the source-
    assembly canonical ALID that each row was lifted from — provenance back to
    the original dataset. Pass ``None`` (or per-row ``None``/``""``) for
    same-assembly builds or ambiguous (collided) rows; the column is written as
    ``"."`` there.
    """

    if source_alids is not None and len(source_alids) != len(variants):
        raise ValueError(
            f"source_alids has {len(source_alids)} entries but there are {len(variants)} variants"
        )
    store = Path(store_path)
    table_path = variant_table_path(store)
    offsets: list[int] = []
    with pysam.BGZFile(str(table_path), "w") as handle:  # type: ignore[call-arg]
        handle.write(VARIANT_HEADER.encode("utf-8"))
        for variant_index, variant in enumerate(variants):
            offsets.append(int(handle.tell()))
            rsid = rsid_by_alid.get(variant.alid) or "."
            source_alid = (source_alids[variant_index] if source_alids is not None else None) or "."
            line = (
                f"{variant.chromosome}\t{variant.position}\t{variant_index}\t"
                f"{variant.effect_allele}\t{variant.other_allele}\t{variant.alid}\t{rsid}\t"
                f"{source_alid}\n"
            )
            handle.write(line.encode("utf-8"))
    np.save(variant_offsets_path(store), np.asarray(offsets, dtype=np.uint64))
    pysam.tabix_index(
        str(table_path),
        seq_col=0,
        start_col=1,
        end_col=1,
        meta_char="#",
        zerobased=False,
        force=True,
    )
    # Write mmap'd ALID search index: two parallel arrays sorted by ALID bytes.
    # Over-wide ALIDs are left out rather than truncated in -- see `_ALID_WIDTH`.
    indexable = [(v.alid, i) for i, v in enumerate(variants) if is_indexable_alid(v.alid)]
    n_too_long = len(variants) - len(indexable)
    if n_too_long:
        log.warning(
            "%d ALID(s) longer than %d bytes left out of the ALID search index: "
            "they resolve by exact scan instead, which is slower and correct",
            n_too_long,
            _ALID_WIDTH,
        )
    alid_bytes = np.array([alid for alid, _ in indexable], dtype=_ALID_DTYPE)
    row_indices = np.array([i for _, i in indexable], dtype="int32")
    sort_order = np.argsort(alid_bytes, kind="stable")
    np.save(variant_alid_bytes_path(store), alid_bytes[sort_order])
    np.save(variant_alid_rows_path(store), row_indices[sort_order])
    _write_rsid_index(store, variants, rsid_by_alid)


def _write_rsid_index(
    store: Path, variants: list[CanonicalVariant], rsid_by_alid: Mapping[str, str]
) -> None:
    """Write the mmap'd rsid search index alongside the ALID one (issue #109).

    Written here, in the one function every builder and every completion path
    already calls, rather than in each builder: rsids reached
    `variants.tsv.gz` for years while nothing indexed them, so `by_identifier`
    returned None for every rsid -- an empty result indistinguishable from a
    real "no association." Deriving the index from the same `rsid_by_alid`
    the table rows are written from makes the two impossible to disagree.

    Rows whose rsid does not fit `_RSID_WIDTH` are left out rather than stored
    truncated, which would make them resolve *another* variant's lookup.

    Rows with no rsid are omitted, so a store whose source carries none writes
    a pair of empty arrays -- present but matching nothing, which is the
    honest answer to "does this store know rs123?" Sorted by rsid bytes with
    row order preserved within a run of equal rsids (stable sort), so the
    several rows one rsid can name (multi-allelic site, or one position stored
    under both allele orders) come back in ascending variant_index.
    """
    rows = []
    n_too_long = 0
    for index, variant in enumerate(variants):
        rsid = rsid_by_alid.get(variant.alid)
        if not rsid:
            continue
        if not is_indexable_rsid(rsid):
            n_too_long += 1
            continue
        rows.append((rsid, index))
    if n_too_long:
        log.warning(
            "%d variant identifier(s) longer than %d bytes left out of the rsid index: "
            "they are not resolvable by name in this store",
            n_too_long,
            _RSID_WIDTH,
        )
    rsid_bytes = np.array([rsid for rsid, _ in rows], dtype=_RSID_DTYPE)
    rsid_rows = np.array([index for _, index in rows], dtype="int32")
    order = np.argsort(rsid_bytes, kind="stable")
    np.save(variant_rsid_bytes_path(store), rsid_bytes[order])
    np.save(variant_rsid_rows_path(store), rsid_rows[order])


class VariantAxis:
    """Read variants from a tabix-indexed Store Variant Table."""

    def __init__(self, store_path: str | Path, aliases: sqlite3.Connection | None = None):
        self.store_path = Path(store_path)
        self.aliases = aliases
        self.table_path = variant_table_path(self.store_path)
        self.tabix_path = variant_tabix_path(self.store_path)
        self.offsets_path = variant_offsets_path(self.store_path)
        self._offsets = np.load(self.offsets_path, mmap_mode="r")
        self._tabix = pysam.TabixFile(str(self.table_path))
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
        self._tabix.close()

    def by_identifier(self, identifier: str) -> VariantRecord | None:
        """Resolve one ALID or rsid to a single Store Variant Table row.

        An rsid can legitimately name several rows -- a multi-allelic site, or
        one position stored under both allele orders. This method's contract is
        one record, so it returns the lowest `variant_index` among them, the
        same choice the `variant_aliases` fallback below has always made
        (``ORDER BY variant_index LIMIT 1``). Callers that need every row an
        rsid names want `indices_by_identifiers`, which returns all of them.
        """
        parsed = parse_canonical_alid(identifier)
        if parsed is not None:
            return self.by_alid(parsed)
        indices = self.indices_by_alias(identifier)
        if len(indices):
            return self.by_index(int(indices[0]))
        if self.aliases is None:
            return None
        row = self.aliases.execute(
            """
            SELECT variant_index
            FROM variant_aliases
            WHERE alias = ?
            ORDER BY variant_index
            LIMIT 1
            """,
            (identifier,),
        ).fetchone()
        if row is None:
            return None
        return self.by_index(int(row["variant_index"]))

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
        query = np.array(alias, dtype=_RSID_DTYPE)
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

        indices: list[int] = []
        if parsed_pairs:
            alid_bytes, alid_rows = self._alid_bytes, self._alid_rows
            # Over-wide ALIDs are not in the index and must not be cast into a
            # query against it: the cast truncates, and a truncated query
            # matches a different variant's key (issue #127). They take the
            # exact scan, the same split `by_alid` makes for one identifier.
            indexable = [(q, p) for q, p in parsed_pairs if is_indexable_alid(q)]
            scanned = [p for q, p in parsed_pairs if not is_indexable_alid(q)]
            if alid_bytes is not None and alid_rows is not None:
                if indexable:
                    queries = np.array([q for q, _ in indexable], dtype=_ALID_DTYPE)
                    positions = np.searchsorted(alid_bytes, queries)
                    in_bounds = positions < len(alid_bytes)
                    hit = np.zeros(len(queries), dtype=bool)
                    hit[in_bounds] = alid_bytes[positions[in_bounds]] == queries[in_bounds]
                    indices.extend(int(row) for row in alid_rows[positions[hit]])
            else:
                scanned = [p for _, p in parsed_pairs]
            for parsed in scanned:
                record = self.by_alid(parsed)
                if record is not None:
                    indices.append(record.variant_index)

        for identifier in alias_identifiers:
            # Every row the rsid names, not just the first: a phewas over
            # rs123 at a multi-allelic site wants both stored rows.
            matched = self.indices_by_alias(identifier)
            if len(matched):
                indices.extend(int(index) for index in matched)
                continue
            record = self.by_identifier(identifier)
            if record is not None:
                indices.append(record.variant_index)

        return np.array(indices, dtype="int32")

    def by_alid(self, parsed: ParsedAlid) -> VariantRecord | None:
        alid = (
            f"{parsed.chromosome}:{parsed.position}:"
            f"{parsed.effect_allele}:{parsed.other_allele}"
        )
        # Only ask the index about ALIDs it could hold. A wider one was never
        # indexed (issue #127), and casting it to the slot width to ask would
        # match a *different* variant's truncated key -- the defect itself.
        if self._alid_bytes is not None and is_indexable_alid(alid):
            query = np.array([alid], dtype=_ALID_DTYPE)
            idx = int(np.searchsorted(self._alid_bytes, query[0]))
            if idx < len(self._alid_bytes) and self._alid_bytes[idx] == query[0]:
                return self.by_index(int(self._alid_rows[idx]))
            return None
        for record in self.range(parsed.chromosome, parsed.position, parsed.position):
            if (
                record.effect_allele == parsed.effect_allele
                and record.other_allele == parsed.other_allele
            ):
                return record
        return None

    def range(self, chromosome: str, start: int, end: int) -> list[VariantRecord]:
        chrom = normalise_chromosome(chromosome)
        try:
            lines = self._tabix.fetch(chrom, max(0, int(start) - 1), int(end))
        except ValueError:
            return []
        return [_parse_variant_line(line) for line in lines]

    def range_indices(self, chromosome: str, start: int, end: int) -> np.ndarray:
        """Return variant indices for a genomic range as int32 array, without object allocation."""
        chrom = normalise_chromosome(chromosome)
        try:
            lines = self._tabix.fetch(chrom, max(0, int(start) - 1), int(end))
        except ValueError:
            return np.empty(0, dtype="int32")
        indices = []
        for line in lines:
            # variant_index is the third tab-separated field (index 2)
            fields = line.split("\t", 3)
            indices.append(int(fields[2]))
        return np.array(indices, dtype="int32")

    def by_index(self, variant_index: int) -> VariantRecord | None:
        if variant_index < 0 or variant_index >= self.n_variants:
            return None
        with pysam.BGZFile(str(self.table_path), "r") as handle:  # type: ignore[call-arg]
            handle.seek(int(self._offsets[int(variant_index)]))
            line = handle.readline().decode("utf-8")
        record = _parse_variant_line(line)
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

    def identity_by_indices(self, variant_index: np.ndarray) -> dict[str, np.ndarray] | None:
        """chromosome/position/effect_allele/other_allele/alid for `variant_index`,
        resolved with zero `variants.tsv.gz` I/O -- not even a scan.

        An ALID *is* `chromosome:position:effect_allele:other_allele`
        (`parse_canonical_alid`), and the mmap'd ALID search index
        (`_alid_bytes`/`_alid_rows`, already loaded at construction for
        `by_alid()`) already holds the reverse of what's needed here:
        `_alid_rows[k]` is the variant_index of the k-th alid in sorted
        order. That's a permutation of `0..n_variants-1`, so inverting it
        once (cached on this instance) gives `variant_index -> sorted
        position` and, from there, alid-by-index via pure numpy fancy
        indexing -- no `by_index()`/`by_indices()` seek or scan at all.
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
        if self._alid_inverse is None:
            rows = np.asarray(self._alid_rows, dtype="int64")
            inverse = np.empty(self.n_variants, dtype="int64")
            inverse[rows] = np.arange(self.n_variants, dtype="int64")
            self._alid_inverse = inverse
        positions = self._alid_inverse[vi]
        alid_bytes = np.asarray(self._alid_bytes)[positions]
        alids = np.array([b.decode("utf-8") for b in alid_bytes], dtype=object)
        if len(alids) == 0:
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

    def all(self) -> list[VariantRecord]:
        return list(iter_variant_records(self.table_path))


def iter_variant_records(table_path: str | Path) -> Iterator[VariantRecord]:
    """Stream a Store Variant Table row by row, in `variant_index` order.

    What `VariantAxis.all` materialises. A caller that only needs one field per
    row -- the EAF orientation audit needs `variant_index` and `alid` -- would
    otherwise pay tens of millions of `VariantRecord` objects to get at them,
    on stores where that is gigabytes.
    """
    with pysam.BGZFile(str(table_path), "r") as handle:  # type: ignore[call-arg]
        for raw_line in handle:
            line = raw_line.decode("utf-8")
            if line.startswith("#"):
                continue
            yield _parse_variant_line(line)


def _parse_variant_line(line: str) -> VariantRecord:
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
