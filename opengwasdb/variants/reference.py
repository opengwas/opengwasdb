"""Variant reference artifacts for single-pass builds (issue #185).

A variant reference is the precomputed variant axis a build stores against,
plus the mapping from a source's own ``(chrom, pos, ref, alt)`` coordinate to
the canonical hg38 ALID whose row it populates. Supplying one to a Dense build
bypasses Pass 1 (the variant union and liftover) entirely: the axis and the
fork-safe Pass 2 lookup are composed from the reference alone.

Three inputs are accepted, distinguished by content rather than by suffix:

* ``*.variant-ref.tsv.gz`` -- the artifact the standalone
  ``extract-variant-reference`` stage writes (issue #187). It carries explicit
  ``source_keys``, so an hg19 source variant resolves to its lifted hg38 ALID
  without re-running Pass 1.
* a plain ALID list -- one canonical GRCh38 ALID per line. Source keys are the
  ALID's own coordinate in either allele order (identity).
* a store ``variants.tsv.gz`` -- the Store Variant Table. Source keys are
  identity, as for a plain list.

Anything that cannot be read as one of these fails loudly. A build must not
proceed against an empty or corrupt axis and silently store nothing -- an empty
result indistinguishable from a real answer is the failure mode this package
exists to prevent.
"""

from __future__ import annotations

import gzip
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO, TYPE_CHECKING

from opengwasdb.variants.axis import parse_canonical_alid
from opengwasdb.variants.windows import (
    DEFAULT_MAP_SPILL_RECORDS,
    DEFAULT_REDUCTION_BATCH_SIZE,
    DEFAULT_WINDOW_SIZE_MB,
    window_key,
    window_size_bp,
)

if TYPE_CHECKING:
    from opengwasdb.layouts.dense.build_vcf import _Pass1Stats

__all__ = [
    "VariantReference",
    "VariantReferenceExtraction",
    "extract_variant_reference",
    "read_variant_reference",
    "write_variant_reference",
]

#: A source's own variant identity, exactly as a SourceReader streams it.
SourceKey = tuple[str, int, str, str]

#: Columns the artifact writer emits; the reader requires every one but
#: ``rsid`` (a source that named no rsid writes it blank).
_ARTIFACT_COLUMNS = ("alid", "chromosome", "position", "a1", "a2", "source_keys")


@dataclass(frozen=True)
class VariantReference:
    """A precomputed variant axis and the source coordinates resolving to it.

    ``source_lookup`` maps a source's raw ``(chrom, pos, ref, alt)`` (the exact
    tuple a ``SourceReader`` streams) to the canonical hg38 ALID whose row it
    populates. ``explicit_source_keys`` records whether that mapping came from
    an artifact's ``source_keys`` column or was inferred by identity, so a
    caller can tell a genuinely-hg38 panel from one carrying hg19 provenance.
    """

    alids: list[str]
    source_lookup: dict[SourceKey, str]
    rsid_by_alid: dict[str, str]
    explicit_source_keys: bool


@dataclass(frozen=True)
class VariantReferenceExtraction:
    """What :func:`extract_variant_reference` wrote, for its caller's summary.

    The phase timings (issue #191) let a caller -- the scaling benchmark, or a
    build log -- see *where* an extraction spent its time instead of one total:
    ``map_seconds`` is the parallel per-source extraction, ``reduce_seconds``
    the windowed tree-merge and ``write_seconds`` the artifact assembly. The
    serial path has no windowed split, so it reports ``reduce_seconds == 0``
    and zero window counts. ``n_reduced_windows`` counts windows that held more
    than one worker shard and therefore actually ran the tree reduce.
    ``reduce_levels`` is the number of tree-reduce levels; more than one means a
    window held more shards than ``reduction_batch_size`` -- the many-spills
    case of issue #194.
    """

    output_path: Path
    n_variants: int
    n_source_keys: int
    n_rsids: int
    map_seconds: float = 0.0
    reduce_seconds: float = 0.0
    write_seconds: float = 0.0
    n_windows: int = 0
    n_window_shards: int = 0
    n_reduced_windows: int = 0
    reduce_levels: int = 0


def extract_variant_reference(
    manifest_path: str | Path,
    output_path: str | Path,
    *,
    chain_file: str | Path | None = None,
    liftover_failure_threshold: float = 0.01,
    n_workers: int = 1,
    source_reader_capability: str | None = None,
    source_assembly: str | None = None,
    window_size_mb: float = DEFAULT_WINDOW_SIZE_MB,
    reduction_batch_size: int = DEFAULT_REDUCTION_BATCH_SIZE,
    map_spill_records: int = DEFAULT_MAP_SPILL_RECORDS,
) -> VariantReferenceExtraction:
    """Extract, lift and canonicalise a manifest's variant axis into an artifact.
    The standalone front end to the build's Pass 1 (issue #187): every source is
    read once through ``resolve_reader``, hg19 rows are lifted to GRCh38, and the
    union is written as the ``*.variant-ref.tsv.gz`` artifact
    ``--variant-reference`` consumes. First-named rsids and the union follow the
    build's rule, so the two-stage store matches the one-command build; an empty
    manifest, or one resolving no variants, fails loudly. ``window_size_mb`` and
    ``reduction_batch_size`` shape the windowed tree-reduce (issue #188) and
    never change the artifact; phase timings and shard counts ride on the result.
    ``map_spill_records`` bounds how many variants a map worker buffers before
    spilling to disk, so a worker's peak memory tracks that threshold rather
    than the size of its manifest slice (issue #194). A non-positive value
    fails loudly before any source is read.
    """
    from opengwasdb.layouts.dense.build_vcf import (
        _lift_manifest_variants,
        _Pass1Stats,
        _read_manifest,
    )

    manifest_rows = _read_manifest(
        manifest_path,
        default_source_reader_capability=source_reader_capability,
        default_source_assembly=source_assembly,
    )
    if len(manifest_rows) == 0:
        raise ValueError(f"manifest {manifest_path} contains no rows to extract a reference from")
    stats = _Pass1Stats()
    source_lookup, rsid_by_alid = _lift_manifest_variants(
        manifest_rows,
        chain_file=chain_file,
        liftover_failure_threshold=liftover_failure_threshold,
        n_workers=n_workers,
        window_size_mb=window_size_mb,
        reduction_batch_size=reduction_batch_size,
        map_spill_records=map_spill_records,
        stats=stats,
    )
    unique_alids = set(source_lookup.values())
    if not unique_alids:
        raise ValueError(f"manifest {manifest_path} yielded no hg38 variants to reference")
    write_start = time.monotonic()
    write_variant_reference(
        output_path, list(unique_alids), source_lookup, rsid_by_alid, window_size_mb=window_size_mb
    )
    write_seconds = time.monotonic() - write_start
    return _extraction_summary(output_path, source_lookup, rsid_by_alid, stats, write_seconds)


def _extraction_summary(
    output_path: str | Path,
    source_lookup: Mapping[SourceKey, str],
    rsid_by_alid: Mapping[str, str],
    stats: _Pass1Stats,
    write_seconds: float,
) -> VariantReferenceExtraction:
    """Assemble the extraction's result from its lookups and phase timings."""
    unique_alids = set(source_lookup.values())
    return VariantReferenceExtraction(
        output_path=Path(output_path),
        n_variants=len(unique_alids),
        n_source_keys=len(source_lookup),
        n_rsids=len(rsid_by_alid),
        map_seconds=stats.map_seconds,
        reduce_seconds=stats.reduce_seconds,
        write_seconds=write_seconds,
        n_windows=stats.n_windows,
        n_window_shards=stats.n_window_shards,
        n_reduced_windows=stats.n_reduced_windows,
        reduce_levels=stats.reduce_levels,
    )


def read_variant_reference(path: str | Path) -> VariantReference:
    """Read a variant reference, failing loudly on anything unreadable.

    The format is detected from the first line: a tab-separated header selects
    the artifact or Store Variant Table reader by its columns; anything else is
    a plain ALID list.
    """
    reference_path = Path(path)
    if not reference_path.is_file():
        raise ValueError(f"variant reference {reference_path} does not exist")
    header, lines = _read_header_and_lines(reference_path)
    if header is None:
        return _reference_from_alid_lines(reference_path, lines)
    if "source_keys" in header or ("a1" in header and "a2" in header):
        return _reference_from_artifact(reference_path, header, lines)
    if "alid" in header:
        return _reference_from_store_table(reference_path, header, lines)
    raise ValueError(
        f"variant reference {reference_path} has neither an 'alid' nor a "
        "'source_keys' column"
    )


def write_variant_reference(
    path: str | Path,
    alids: Sequence[str],
    source_lookup: Mapping[SourceKey, str],
    rsid_by_alid: Mapping[str, str] | None = None,
    *,
    window_size_mb: float = DEFAULT_WINDOW_SIZE_MB,
) -> None:
    """Write the artifact the inline two-pass Pass 1 would otherwise recompute.

    ``source_lookup`` is exactly the ``(source coord) -> hg38 ALID`` map
    ``_lift_manifest_variants`` returns, so a store built through a written
    artifact is identical to one built from the same manifest in two passes.

    Rows are assembled window by window (issue #188): each genomic window's
    ALIDs are sorted and the windows concatenate in genomic order, so the axis
    is never re-sorted as a whole -- only one window at a time.
    """
    # Local import: build_vcf imports this module, so a module-level import
    # would be a cycle. By call time build_vcf is fully loaded.
    from opengwasdb.layouts.dense.build_vcf import _alid_sort_key

    size_bp = window_size_bp(window_size_mb)
    keys_by_alid = _source_keys_by_alid(source_lookup)
    rsids = rsid_by_alid or {}
    windows: dict[tuple[tuple[int, str], int], list[str]] = {}
    for alid in set(alids):
        chrom, position, _a1, _a2 = alid.split(":")
        windows.setdefault(window_key(chrom, int(position), size_bp), []).append(alid)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wt", encoding="utf-8") as handle:
        handle.write("#alid\tchromosome\tposition\ta1\ta2\trsid\tsource_keys\n")
        for window in sorted(windows):
            for alid in sorted(windows[window], key=_alid_sort_key):
                handle.write(_artifact_line(alid, keys_by_alid.get(alid, ()), rsids.get(alid, "")))


def _artifact_line(alid: str, source_keys: Sequence[str], rsid: str) -> str:
    chrom, position, a1, a2 = alid.split(":")
    return (
        f"{alid}\t{chrom}\t{position}\t{a1}\t{a2}\t{rsid}\t{';'.join(source_keys)}\n"
    )


# ── reading ──────────────────────────────────────────────────────────────────


def _open_text(path: Path) -> IO[str]:
    if str(path).lower().endswith((".gz", ".bgz", ".bgzf")):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, encoding="utf-8")


def _read_header_and_lines(path: Path) -> tuple[list[str] | None, list[str]]:
    """The header (when the file is tab-separated) and every data line."""
    with _open_text(path) as handle:
        first = handle.readline()
        if not first.strip():
            raise ValueError(f"variant reference {path} is empty")
        if "\t" in first:
            header = [field.lstrip("#").strip() for field in first.rstrip("\n").split("\t")]
            return header, list(handle)
        return None, [first, *handle]


def _canonical_alid(token: str, path: Path) -> str:
    """Normalise an ALID to ``chrom:pos:a1:a2`` or fail loudly."""
    parsed = parse_canonical_alid(token)
    if parsed is None:
        raise ValueError(f"variant reference {path} contains invalid ALID {token!r}")
    a1, a2 = sorted((parsed.effect_allele, parsed.other_allele))
    return f"{parsed.chromosome}:{parsed.position}:{a1}:{a2}"


def _identity_keys(alid: str) -> Iterator[SourceKey]:
    """The two raw source tuples a canonical ALID can be reported as."""
    chrom, position, a1, a2 = alid.split(":")
    pos = int(position)
    yield (chrom, pos, a1, a2)
    yield (chrom, pos, a2, a1)


def _identity_lookup(alids: Sequence[str]) -> dict[SourceKey, str]:
    """Identity source keys -- for panels whose sources are already hg38."""
    lookup: dict[SourceKey, str] = {}
    for alid in alids:
        for key in _identity_keys(alid):
            lookup[key] = alid
    return lookup


def _require_alids(alids: Sequence[str], path: Path) -> None:
    if not alids:
        raise ValueError(f"variant reference {path} contained no ALIDs")


def _reference_from_alid_lines(path: Path, lines: Sequence[str]) -> VariantReference:
    alids: list[str] = []
    seen: set[str] = set()
    for token in _alid_tokens(lines):
        _append_alid(path, _canonical_alid(token, path), seen, alids)
    _require_alids(alids, path)
    return VariantReference(
        alids=alids,
        source_lookup=_identity_lookup(alids),
        rsid_by_alid={},
        explicit_source_keys=False,
    )


def _alid_tokens(lines: Sequence[str]) -> Iterator[str]:
    for line in lines:
        token = line.strip().split()[0] if line.strip() else ""
        if token and not token.startswith("#"):
            yield token


def _data_rows(lines: Sequence[str]) -> list[list[str]]:
    return [
        line.rstrip("\n").split("\t")
        for line in lines
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _append_alid(path: Path, alid: str, seen: set[str], alids: list[str]) -> None:
    if alid in seen:
        raise ValueError(f"variant reference {path} contains duplicate ALID {alid!r}")
    seen.add(alid)
    alids.append(alid)


def _field(fields: Sequence[str], index: int, path: Path) -> str:
    if index >= len(fields):
        raise ValueError(
            f"variant reference {path} row has {len(fields)} fields, expected at least {index + 1}"
        )
    return fields[index]


def _record_rsid(
    path: Path,
    fields: Sequence[str],
    rsid_col: int | None,
    alid: str,
    rsid_by_alid: dict[str, str],
) -> None:
    rsid = _field(fields, rsid_col, path).strip() if rsid_col is not None else ""
    if rsid and rsid != ".":
        rsid_by_alid[alid] = rsid


def _reference_from_store_table(
    path: Path, header: Sequence[str], lines: Sequence[str]
) -> VariantReference:
    alid_col = header.index("alid")
    rsid_col = header.index("rsid") if "rsid" in header else None
    alids: list[str] = []
    rsid_by_alid: dict[str, str] = {}
    seen: set[str] = set()
    for fields in _data_rows(lines):
        alid = _canonical_alid(_field(fields, alid_col, path), path)
        _append_alid(path, alid, seen, alids)
        _record_rsid(path, fields, rsid_col, alid, rsid_by_alid)
    _require_alids(alids, path)
    return VariantReference(
        alids=alids,
        source_lookup=_identity_lookup(alids),
        rsid_by_alid=rsid_by_alid,
        explicit_source_keys=False,
    )


def _reference_from_artifact(
    path: Path, header: Sequence[str], lines: Sequence[str]
) -> VariantReference:
    missing = [name for name in _ARTIFACT_COLUMNS if name not in header]
    if missing:
        raise ValueError(
            f"variant reference {path} is missing the column(s) {', '.join(missing)}"
        )
    alids: list[str] = []
    rsid_by_alid: dict[str, str] = {}
    source_lookup: dict[SourceKey, str] = {}
    seen: set[str] = set()
    rsid_col = header.index("rsid") if "rsid" in header else None
    for fields in _data_rows(lines):
        alid = _artifact_alid(path, header, fields)
        _append_alid(path, alid, seen, alids)
        source_keys = _field(fields, header.index("source_keys"), path)
        _add_source_keys(path, source_keys, alid, source_lookup)
        _record_rsid(path, fields, rsid_col, alid, rsid_by_alid)
    _require_alids(alids, path)
    return VariantReference(
        alids=alids,
        source_lookup=source_lookup,
        rsid_by_alid=rsid_by_alid,
        explicit_source_keys=True,
    )


def _artifact_alid(path: Path, header: Sequence[str], fields: Sequence[str]) -> str:
    """The row's ALID, cross-checked against its own coordinate columns."""
    alid = _canonical_alid(_field(fields, header.index("alid"), path), path)
    described = ":".join(
        _field(fields, header.index(name), path)
        for name in ("chromosome", "position", "a1", "a2")
    )
    constructed = _canonical_alid(described, path)
    if constructed != alid:
        raise ValueError(
            f"variant reference {path}: row columns describe {constructed!r} but "
            f"the alid column says {alid!r}"
        )
    return alid


def _add_source_keys(
    path: Path, raw_keys: str, alid: str, lookup: dict[SourceKey, str]
) -> None:
    tokens = [token.strip() for token in raw_keys.split(";") if token.strip()]
    if not tokens:
        # An external panel row carries no source mapping; identity is the only
        # honest reading (its sources are already on the reference build).
        for key in _identity_keys(alid):
            _bind_source_key(path, key, alid, lookup)
        return
    for token in tokens:
        _bind_source_key(path, _parse_source_key(path, token), alid, lookup)


def _bind_source_key(path: Path, key: SourceKey, alid: str, lookup: dict[SourceKey, str]) -> None:
    existing = lookup.get(key)
    if existing is not None and existing != alid:
        raise ValueError(
            f"variant reference {path}: source key {key!r} maps to both "
            f"{existing!r} and {alid!r}"
        )
    lookup[key] = alid


def _parse_source_key(path: Path, token: str) -> SourceKey:
    parts = token.split(":")
    if len(parts) != 4:
        raise ValueError(f"variant reference {path} contains invalid source key {token!r}")
    chrom, position, ref, alt = parts
    try:
        pos = int(position)
    except ValueError as exc:
        raise ValueError(
            f"variant reference {path} contains invalid source key {token!r}"
        ) from exc
    if pos <= 0 or not chrom or not ref or not alt:
        raise ValueError(f"variant reference {path} contains invalid source key {token!r}")
    return (chrom, pos, ref, alt)


def _source_keys_by_alid(
    source_lookup: Mapping[SourceKey, str],
) -> dict[str, list[str]]:
    by_alid: dict[str, list[str]] = {}
    for (chrom, pos, ref, alt), alid in source_lookup.items():
        by_alid.setdefault(alid, []).append(f"{chrom}:{pos}:{ref}:{alt}")
    for keys in by_alid.values():
        keys.sort()
    return by_alid
