"""Writing the tabix-backed Store Variant Axis.

`VariantAxis` -- every variant question a query asks -- is the Store Reader's
(`opengwasdb.store_reader.variant_axis`, #464). It reads the 0.2.x BGZF/tabix
Variant Table through `opengwasdb.variants.legacy_table`. This module's
`VariantAxis` is that one, opened with the table, for the opengwasdb code (and
opengwasdb-stores) that constructs an axis from a Store path.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np
import pysam

from opengwasdb.store_reader import variant_axis as reader_axis
from opengwasdb.store_reader.alid import (
    ALID_DTYPE,
    ALID_WIDTH,
    RSID_DTYPE,
    RSID_WIDTH,
    is_indexable_alid,
    is_indexable_rsid,
)

# Re-exported: opengwasdb-stores imports it from here.
from opengwasdb.store_reader.alid import parse_canonical_alid as parse_canonical_alid
from opengwasdb.store_reader.identity import normalise_chromosome

# Re-exported: opengwasdb-stores imports these three from here.
from opengwasdb.store_reader.variant_axis import variant_alid_bytes_path as variant_alid_bytes_path
from opengwasdb.store_reader.variant_axis import variant_alid_rows_path as variant_alid_rows_path
from opengwasdb.store_reader.variant_axis import (
    variant_offsets_path,
    variant_rsid_bytes_path,
    variant_rsid_rows_path,
)
from opengwasdb.store_reader.variant_axis import variant_table_path as variant_table_path
from opengwasdb.variants.legacy_table import TabixVariantTable
from opengwasdb.variants.normalise import CanonicalVariant, chromosome_sort_key

log = logging.getLogger(__name__)

VARIANT_AXIS_FORMAT = "tabix_tsv_v1"
VARIANT_HEADER = (
    "#chromosome\tposition\tvariant_index\teffect_allele\tother_allele\talid\trsid"
    "\tsource_alid\n"
)
def alid_sort_key(alid: str) -> tuple[tuple[int, str], int, str, str]:
    """The order of a Variant Index: chromosome rank, position, then the two alleles."""
    chrom, pos_str, a1, a2 = alid.split(":")
    return (chromosome_sort_key(chrom), int(pos_str), a1, a2)


def sorted_alids(alids: Iterable[str]) -> list[str]:
    """Sort unique ALIDs by ``(chromosome_sort_key, position, a1, a2)``.

    Exactly equivalent to ``sorted(set(alids), key=alid_sort_key)`` but built
    for genome-scale axis construction: the ALIDs are parsed once and each one
    is re-encoded as a sortable byte key, then ``np.argsort`` orders the byte
    keys in C rather than calling ``alid_sort_key`` once per element (~21M
    Python calls on a dense pilot build).

    The sort key is ``fixed-width rank \\x00 chrom \\x00 zero-padded-position
    \\x00 a1 \\x00 a2``. ``\\x00`` is smaller than every character that can
    appear in a chromosome, position or allele, so it terminates the variable-
    length fields and makes lexicographic byte order equal to the Python tuple
    order ``alid_sort_key`` produces.
    """
    unique = list(set(alids))
    if not unique:
        return []

    n = len(unique)
    ranks = np.empty(n, dtype=np.int64)
    chroms: list[str] = [""] * n
    positions: list[str] = [""] * n
    a1s: list[str] = [""] * n
    a2s: list[str] = [""] * n
    pos_width = 1
    for i, alid in enumerate(unique):
        chrom, pos_str, a1, a2 = alid.split(":")
        canonical_chrom = normalise_chromosome(chrom)
        chroms[i] = canonical_chrom
        positions[i] = pos_str
        a1s[i] = a1
        a2s[i] = a2
        ranks[i] = chromosome_sort_key(canonical_chrom)[0]
        if len(pos_str) > pos_width:
            pos_width = len(pos_str)

    keys = np.empty(n, dtype=object)
    rank_width = len(str(int(ranks.max())))
    for i in range(n):
        keys[i] = (
            f"{ranks[i]:0{rank_width}d}".encode()
            + b"\x00"
            + chroms[i].encode()
            + b"\x00"
            + positions[i].zfill(pos_width).encode()
            + b"\x00"
            + a1s[i].encode()
            + b"\x00"
            + a2s[i].encode()
        )
    del chroms, positions, a1s, a2s
    order = np.argsort(keys, kind="stable")
    del keys
    return [unique[i] for i in order]


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
    # Over-wide ALIDs are left out rather than truncated in -- see `ALID_WIDTH`.
    indexable = [(v.alid, i) for i, v in enumerate(variants) if is_indexable_alid(v.alid)]
    n_too_long = len(variants) - len(indexable)
    if n_too_long:
        log.warning(
            "%d ALID(s) longer than %d bytes left out of the ALID search index: "
            "they resolve by exact scan instead, which is slower and correct",
            n_too_long,
            ALID_WIDTH,
        )
    alid_bytes = np.array([alid for alid, _ in indexable], dtype=ALID_DTYPE)
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

    Rows whose rsid does not fit `RSID_WIDTH` are left out rather than stored
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
            RSID_WIDTH,
        )
    rsid_bytes = np.array([rsid for rsid, _ in rows], dtype=RSID_DTYPE)
    rsid_rows = np.array([index for _, index in rows], dtype="int32")
    order = np.argsort(rsid_bytes, kind="stable")
    np.save(variant_rsid_bytes_path(store), rsid_bytes[order])
    np.save(variant_rsid_rows_path(store), rsid_rows[order])


class VariantAxis(reader_axis.VariantAxis):
    """The Store Reader's `VariantAxis`, constructed from a Store path alone.

    What it adds is the construction: it opens the Store's 0.2.x Variant Table
    (`TabixVariantTable`) itself, so a caller -- a builder, completion,
    validation, opengwasdb-stores' audit -- writes ``VariantAxis(path)`` as it
    always has. `aliases` is an open `index.sqlite` connection the caller keeps,
    for a Store whose rsids may only be named in `variant_aliases`.
    """

    def __init__(self, store_path: str | Path, aliases: sqlite3.Connection | None = None):
        super().__init__(store_path, TabixVariantTable(store_path, aliases))
