"""Writing the tabix-backed Store Variant Axis.

`VariantAxis` -- every variant question a query asks -- is the Store Reader's
(`opengwasdb.store_reader.variant_axis`, #464). It reads the 0.2.x BGZF/tabix
Variant Table through `opengwasdb.variants.legacy_table`. This module's
`VariantAxis` is that one, opened with the table, for the opengwasdb code (and
opengwasdb-stores) that constructs an axis from a Store path.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np

from opengwasdb.store_reader import variant_axis as reader_axis
from opengwasdb.store_reader.alid import (
    RSID_DTYPE,
    RSID_WIDTH,
    is_indexable_rsid,
)

# Re-exported: opengwasdb-stores imports it from here.
from opengwasdb.store_reader.alid import parse_canonical_alid as parse_canonical_alid
from opengwasdb.store_reader.alid_sidecar import write_alid_sidecar
from opengwasdb.store_reader.identity import normalise_chromosome
from opengwasdb.store_reader.position_sidecar import write_position_sidecar
from opengwasdb.store_reader.source_alid_sidecar import write_source_alid_sidecar

# Re-exported: opengwasdb-stores imports these three from here.
from opengwasdb.store_reader.variant_axis import variant_alid_bytes_path as variant_alid_bytes_path
from opengwasdb.store_reader.variant_axis import variant_alid_rows_path as variant_alid_rows_path
from opengwasdb.store_reader.variant_axis import (
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
    """Write the Store Variant Table, its tabix index, and the sidecars beside it: the
    ALID sidecar (`opengwasdb.store_reader.alid_sidecar`), the position sidecar
    (`opengwasdb.store_reader.position_sidecar`), the `source_alid` sidecar
    (`opengwasdb.store_reader.source_alid_sidecar`) and the rsid index: the axis's one
    writer, which every build and completion path calls, a Hybrid's nested Dense
    Component included. Each sidecar's own one writer is its codec's
    (`write_alid_sidecar`, `write_position_sidecar`, `write_source_alid_sidecar`),
    which this calls and the rewrite to 0.3.0 calls directly.

    ``source_alids`` (optional, parallel to ``variants``) records the source-
    assembly canonical ALID that each row was lifted from — provenance back to
    the original dataset. Pass ``None`` (or per-row ``None``/``""``) for
    ambiguous (collided) rows and rows no source observed; the table's column is
    written as ``"."`` there, and the sidecar's value is empty.
    """

    if source_alids is not None and len(source_alids) != len(variants):
        raise ValueError(
            f"source_alids has {len(source_alids)} entries but there are {len(variants)} variants"
        )
    store = Path(store_path)
    # Before any file is written: a 0.3.0 Store Reader names a row's rsID from the rsID
    # index alone (#469), so an rsID the index cannot hold would be lost from every
    # answer while the table still showed it.
    _refuse_unindexable_rsids(variants, rsid_by_alid)
    # The `source_alid` sidecar (format 0.3.0, #470, ADR 0105 §8): each row's pre-liftover
    # ALID, which the Store Reader returns without reading the table; a row's state says
    # whether it is none, its own ALID or another, held whole. Written first: it refuses a
    # value the table's column could not hold as the sidecar does.
    write_source_alid_sidecar(
        store,
        list(source_alids) if source_alids is not None else [None] * len(variants),
        [variant.alid for variant in variants],
    )
    _write_table(store, variants, rsid_by_alid, source_alids)
    # The variable-length ALID sidecar (format 0.3.0, #467, ADR 0105): every ALID,
    # however long, by row and by search. The 0.2 series' fixed-width pair
    # (`variant_alid_bytes.npy`/`variant_alid_rows.npy`) left out each ALID wider than
    # 32 bytes and is no longer written.
    write_alid_sidecar(store, [variant.alid for variant in variants])
    # The position sidecar (format 0.3.0, #468, ADR 0105 §6): every row's position and
    # each chromosome's rows, which answer a range without tabix. It refuses rows out
    # of genomic order, as tabix does, and chromosomes out of the canonical order.
    write_position_sidecar(
        store,
        [variant.chromosome for variant in variants],
        [variant.position for variant in variants],
        chromosome_key=chromosome_sort_key,
    )
    _write_rsid_index(store, variants, rsid_by_alid)


def carried_source_alids(
    source_variants: Iterable[reader_axis.VariantRecord], variants: Sequence[CanonicalVariant]
) -> list[str | None]:
    """Each of ``variants``' `source_alid`, carried from the source release's rows.

    The one owner of how Reference Completion keeps provenance (#449): an observed row
    keeps the ALID its build recorded, and a row added from the panel (which no source
    observed) or one the build left blank stays ``None``, written as ``.``. Hybrid
    completion builds the same list inline and could call this.
    """
    by_alid = {row.alid: row.source_alid for row in source_variants if row.source_alid}
    return [by_alid.get(variant.alid) for variant in variants]


def _write_table(
    store: Path,
    variants: list[CanonicalVariant],
    rsid_by_alid: Mapping[str, str],
    source_alids: Sequence[str | None] | None,
) -> None:
    """`variants.tsv.gz` and its tabix index, both written with htslib.

    From format 0.3.0 no query reads either (#469): they are a convenience restating the
    sidecars, for tabix, bedtools and R (spec §1.1). The 0.2 series' `variant_offsets.npy`,
    each row's BGZF offset, is not written.
    """
    import pysam  # on use: importing the axis module, as every query does, loads no htslib

    table_path = variant_table_path(store)
    with pysam.BGZFile(str(table_path), "w") as handle:  # type: ignore[call-arg]
        handle.write(VARIANT_HEADER.encode("utf-8"))
        for variant_index, variant in enumerate(variants):
            rsid = rsid_by_alid.get(variant.alid) or "."
            source_alid = (source_alids[variant_index] if source_alids is not None else None) or "."
            line = (
                f"{variant.chromosome}\t{variant.position}\t{variant_index}\t"
                f"{variant.effect_allele}\t{variant.other_allele}\t{variant.alid}\t{rsid}\t"
                f"{source_alid}\n"
            )
            handle.write(line.encode("utf-8"))
    pysam.tabix_index(
        str(table_path),
        seq_col=0,
        start_col=1,
        end_col=1,
        meta_char="#",
        zerobased=False,
        force=True,
    )


class UnindexableRsidError(ValueError):
    """A variant's rsID is too wide for the rsID index, which names a 0.3.0 row's rsID."""


def _refuse_unindexable_rsids(
    variants: list[CanonicalVariant], rsid_by_alid: Mapping[str, str]
) -> None:
    """Refuse a variant whose rsID the rsID index cannot hold (`is_indexable_rsid`).

    The 0.2 series left such an rsID out of the index and kept it in the table, so it
    was shown but not found by name. From 0.3.0 the Store Reader names every row's rsID
    from the index (#469), so it would be shown nowhere, while the table kept it: a
    wrong answer that looks right. None of the four releases measured holds one
    (OGS-00003, -00004, -00007, -00025: 41,789,333 named rows, all `rs` and digits).
    """
    wide = [
        (variant.alid, rsid)
        for variant in variants
        if (rsid := rsid_by_alid.get(variant.alid)) and not is_indexable_rsid(rsid)
    ]
    if wide:
        alid, rsid = wide[0]
        raise UnindexableRsidError(
            f"{len(wide)} variant(s) name an rsID wider than the rsID index's {RSID_WIDTH} "
            f"bytes, which a 0.3.0 release cannot name a row by; first, {alid} named "
            f"{rsid!r}. Shorten or drop such names at the source"
        )


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

    An rsid that does not fit `RSID_WIDTH` is never stored truncated, which would make
    it resolve *another* variant's lookup; `write_variant_axis` refuses one before this
    runs (`_refuse_unindexable_rsids`).

    Rows with no rsid are omitted, so a store whose source carries none writes
    a pair of empty arrays -- present but matching nothing, which is the
    honest answer to "does this store know rs123?" Sorted by rsid bytes with
    row order preserved within a run of equal rsids (stable sort), so the
    several rows one rsid can name (multi-allelic site, or one position stored
    under both allele orders) come back in ascending variant_index.
    """
    rows = [
        (rsid, index)
        for index, variant in enumerate(variants)
        if (rsid := rsid_by_alid.get(variant.alid))
    ]
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

    The ALID index it opens is the one the release's `format_version` carries
    (`alid_index_kind`), read from its `manifest.json`. A directory with no manifest
    -- an axis a test or a build step has just written -- must say which version it
    is in `format_version`; nothing is guessed from the files present.
    """

    def __init__(
        self,
        store_path: str | Path,
        aliases: sqlite3.Connection | None = None,
        *,
        format_version: str | None = None,
    ):
        super().__init__(
            store_path,
            TabixVariantTable(store_path, aliases),
            format_version=format_version or _declared_format_version(Path(store_path)),
        )


def _declared_format_version(store: Path) -> str:
    """The `format_version` `store`'s manifest declares; refused when it has none."""
    manifest = store / "manifest.json"
    if not manifest.exists():
        raise ValueError(
            f"{store} has no manifest.json to say which ALID index it carries; "
            "pass format_version= for a bare Variant Axis directory"
        )
    return str(json.loads(manifest.read_text(encoding="utf-8"))["format_version"])
