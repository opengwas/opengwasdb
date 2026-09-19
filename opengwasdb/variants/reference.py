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
import logging
import shutil
import tempfile
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import IO, TYPE_CHECKING, TypeVar

from opengwasdb.variants.axis import parse_canonical_alid
from opengwasdb.variants.windows import (
    DEFAULT_MAP_SPILL_RECORDS,
    DEFAULT_REDUCTION_BATCH_SIZE,
    DEFAULT_WINDOW_SIZE_MB,
    window_key,
    window_size_bp,
)

if TYPE_CHECKING:
    from opengwasdb.layouts.dense.build_vcf import (
        _ManifestRow,
        _Pass1Stats,
        _ShardSpec,
        _WindowShards,
    )
    from opengwasdb.variants.windows import WindowKey

log = logging.getLogger(__name__)

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

#: The artifact's first line, written as the first gzip member on the streaming
#: all-hg38 path (issue #196).
_ARTIFACT_HEADER = "#alid\tchromosome\tposition\ta1\ta2\trsid\tsource_keys\n"


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


@dataclass(frozen=True)
class _WindowArtifact:
    """One window's compressed artifact member and the counts it contributed.

    Returned across the process boundary by :func:`_stream_window_artifact`;
    the member path and integer counts are all that leave the worker, never a
    row or an ALID (issue #196).
    """

    window: WindowKey
    path: Path
    n_alids: int
    n_source_keys: int
    n_rsids: int


@dataclass(frozen=True)
class _StreamedArtifact:
    """The streaming all-hg38 artifact writer's result (issue #196)."""

    n_variants: int
    n_source_keys: int
    n_rsids: int
    write_seconds: float


@dataclass(frozen=True)
class _UnionOptions:
    """The map/tree-reduce knobs shared by both extraction arms (issues #188-#196)."""

    n_workers: int
    window_size_mb: float
    reduction_batch_size: int
    map_spill_records: int


@dataclass(frozen=True)
class _LiftedBucket:
    """One pre-lift window's intermediate sub-shard for one post-lift window."""

    post_window: WindowKey
    path: Path


@dataclass(frozen=True)
class _StagedWindow:
    """One pre-lift window's step-A result: sub-shards plus lift and drop counts."""

    buckets: tuple[_LiftedBucket, ...]
    attempts: int
    failures: int
    collisions: int


@dataclass(frozen=True)
class _LiftStageTask:
    """Step A input: one pre-lift window's final shards and the lift settings.

    ``shards`` is ``(assembly, path, chunk, spill)`` per final shard; the chunk
    and spill are the shard's rank, which orders equal sites during step B.
    """

    index: int
    window: WindowKey
    shards: tuple[tuple[str, str, int, int], ...]
    size_bp: int
    members_dir: str
    chain_file: str | None


@dataclass(frozen=True)
class _LiftMergeTask:
    """Step B input: every intermediate sub-shard for one post-lift window."""

    window: WindowKey
    sub_shards: tuple[str, ...]
    member_path: str


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
    from opengwasdb.layouts.dense.build_vcf import _Pass1Stats, _read_manifest

    manifest_rows = _read_manifest(
        manifest_path,
        default_source_reader_capability=source_reader_capability,
        default_source_assembly=source_assembly,
    )
    if len(manifest_rows) == 0:
        raise ValueError(f"manifest {manifest_path} contains no rows to extract a reference from")
    stats = _Pass1Stats()
    options = _UnionOptions(
        n_workers=n_workers,
        window_size_mb=window_size_mb,
        reduction_batch_size=reduction_batch_size,
        map_spill_records=map_spill_records,
    )
    if _all_hg38(manifest_rows):
        return _extract_streaming_reference(
            manifest_rows, manifest_path, output_path, options, stats
        )
    return _extract_lifted_reference(
        manifest_rows,
        manifest_path,
        output_path,
        options,
        stats,
        chain_file=chain_file,
        liftover_failure_threshold=liftover_failure_threshold,
    )


def _all_hg38(manifest_rows: Sequence[_ManifestRow]) -> bool:
    """Whether every row is already GRCh38, so no liftover is needed (issue #196).

    Rows that omit ``source_assembly`` default to hg19 (`_read_manifest`), so a
    manifest reaches the streaming artifact path only when every row declares
    hg38 (explicitly, or through the caller's default).
    """
    return bool(manifest_rows) and all(row.source_assembly == "hg38" for row in manifest_rows)


def _streaming_extraction(
    manifest_rows: list[_ManifestRow],
    manifest_path: str | Path,
    output_path: str | Path,
    options: _UnionOptions,
    stats: _Pass1Stats,
    consume: Callable[[_WindowShards], _StreamedArtifact],
) -> VariantReferenceExtraction:
    """Run the map/reduce core and hand the shards to a streaming consumer.

    Both streaming arms share this: the consumer writes the artifact and returns
    only counts and its own phase time, so the parent never materialises the
    union (issues #196/#197).
    """
    from opengwasdb.layouts.dense.build_vcf import _consume_manifest_shards

    streamed = _consume_manifest_shards(
        manifest_rows,
        n_workers=options.n_workers,
        window_size_mb=options.window_size_mb,
        reduction_batch_size=options.reduction_batch_size,
        map_spill_records=options.map_spill_records,
        stats=stats,
        consume=consume,
    )
    if streamed.n_variants == 0:
        raise ValueError(f"manifest {manifest_path} yielded no hg38 variants to reference")
    return _extraction_summary(
        output_path,
        streamed.n_variants,
        streamed.n_source_keys,
        streamed.n_rsids,
        stats,
        streamed.write_seconds,
    )


def _extract_streaming_reference(
    manifest_rows: list[_ManifestRow],
    manifest_path: str | Path,
    output_path: str | Path,
    options: _UnionOptions,
    stats: _Pass1Stats,
) -> VariantReferenceExtraction:
    """All-hg38 arm: per-window parallel compression, then ordered concatenation.

    The parent never materialises the union; ``_write_streaming_artifact``
    returns only counts and its own phase time (issue #196).
    """
    writer = partial(
        _write_streaming_artifact, output_path=output_path, n_workers=options.n_workers
    )
    return _streaming_extraction(
        manifest_rows, manifest_path, output_path, options, stats, writer
    )


def _extract_lifted_reference(
    manifest_rows: list[_ManifestRow],
    manifest_path: str | Path,
    output_path: str | Path,
    options: _UnionOptions,
    stats: _Pass1Stats,
    *,
    chain_file: str | Path | None,
    liftover_failure_threshold: float,
) -> VariantReferenceExtraction:
    """hg19 or mixed arm: lift per pre-lift window, re-window and concatenate.

    The parent never materialises the union; ``_write_lifted_streaming_artifact``
    lifts each pre-lift window in a worker, re-buckets every surviving record by
    post-lift window, merges those buckets per post-lift window and concatenates
    the members in genomic order (issue #197).
    """
    writer = partial(
        _write_lifted_streaming_artifact,
        output_path=output_path,
        n_workers=options.n_workers,
        window_size_mb=options.window_size_mb,
        chain_file=chain_file,
        liftover_failure_threshold=liftover_failure_threshold,
    )
    return _streaming_extraction(
        manifest_rows, manifest_path, output_path, options, stats, writer
    )


def _extraction_summary(
    output_path: str | Path,
    n_variants: int,
    n_source_keys: int,
    n_rsids: int,
    stats: _Pass1Stats,
    write_seconds: float,
) -> VariantReferenceExtraction:
    """Assemble the extraction's result from its counts and phase timings."""
    return VariantReferenceExtraction(
        output_path=Path(output_path),
        n_variants=n_variants,
        n_source_keys=n_source_keys,
        n_rsids=n_rsids,
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
        handle.write(_ARTIFACT_HEADER)
        for window in sorted(windows):
            for alid in sorted(windows[window], key=_alid_sort_key):
                handle.write(_artifact_line(alid, keys_by_alid.get(alid, ()), rsids.get(alid, "")))


def _artifact_line(alid: str, source_keys: Sequence[str], rsid: str) -> str:
    chrom, position, a1, a2 = alid.split(":")
    return (
        f"{alid}\t{chrom}\t{position}\t{a1}\t{a2}\t{rsid}\t{';'.join(source_keys)}\n"
    )


def _write_member(
    member_path: str | Path,
    rsid_by_alid: Mapping[str, str],
    source_keys_by_alid: Mapping[str, list[str]],
) -> None:
    """Write one window's ALIDs, sorted by ``_alid_sort_key``, as a gzip member."""
    # Local import: build_vcf imports this module, so a module-level import
    # would be a cycle. By call time build_vcf is fully loaded.
    from opengwasdb.layouts.dense.build_vcf import _alid_sort_key

    with gzip.open(member_path, "wt", encoding="utf-8") as handle:
        for alid in sorted(source_keys_by_alid, key=_alid_sort_key):
            source_keys = source_keys_by_alid[alid]
            source_keys.sort()
            handle.write(_artifact_line(alid, source_keys, rsid_by_alid.get(alid, "")))


# ── streaming all-hg38 artifact (issue #196) ─────────────────────────────────


def _stream_window_artifact(task: tuple[WindowKey, str, str]) -> _WindowArtifact:
    """Compress one final window shard into a standalone gzip member.

    The shard is already sorted by site and holds one record per site, with its
    first-named rsid resolved in rank order by the reduce (issue #109). Sites
    are canonicalised to ALIDs; colliding ALIDs are grouped with their source
    keys sorted, and the ALID's rsid is the first non-empty in site order --
    which is exactly what the materialising writer's ``(rank, site)`` insertion
    produces within one window. Runs in a worker process, so only the member
    path and integer counts cross back (issue #196).
    """
    from opengwasdb.layouts.dense.build_vcf import _iter_pass1_shard

    window, shard_path, member_path = task
    rsid_by_alid: dict[str, str] = {}
    source_keys_by_alid: dict[str, list[str]] = {}
    n_source_keys = 0
    for (chrom, pos, ref, alt), rsid in _iter_pass1_shard(Path(shard_path)):
        a1, a2 = sorted((ref, alt))
        alid = f"{chrom}:{pos}:{a1}:{a2}"
        if rsid and alid not in rsid_by_alid:
            rsid_by_alid[alid] = rsid
        source_keys_by_alid.setdefault(alid, []).append(f"{chrom}:{pos}:{ref}:{alt}")
        n_source_keys += 1
    _write_member(member_path, rsid_by_alid, source_keys_by_alid)
    return _WindowArtifact(
        window, Path(member_path), len(source_keys_by_alid), n_source_keys, len(rsid_by_alid)
    )


_T = TypeVar("_T")
_R = TypeVar("_R")


def _run_pooled(tasks: Sequence[_T], worker: Callable[[_T], _R], n_workers: int) -> list[_R]:
    """Run ``worker`` over ``tasks`` in a fork pool, preserving task order."""
    from opengwasdb.layouts.dense.build_vcf import _fork_pool

    workers = min(n_workers, len(tasks))
    if workers <= 1:
        return [worker(task) for task in tasks]
    with _fork_pool(workers) as pool:
        futures = [pool.submit(worker, task) for task in tasks]
        return [future.result() for future in futures]


def _concatenate_window_members(
    output_path: Path, artifacts: Sequence[_WindowArtifact]
) -> None:
    """Append each window's gzip member to ``output_path`` in genomic order."""
    with open(output_path, "ab") as target:
        for artifact in sorted(artifacts, key=lambda artifact: artifact.window):
            with open(artifact.path, "rb") as member:
                shutil.copyfileobj(member, target)


def _streaming_output_path(output_path: str | Path) -> Path:
    """The artifact path, with its parent directory ensured to exist."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def _finish_members(
    out: Path, artifacts: Sequence[_WindowArtifact], write_start: float
) -> _StreamedArtifact:
    """Write the header member, append the window members, summarise the counts.

    An empty result writes no file at all: a header-only artifact reads back as
    an empty reference, indistinguishable from a real answer, and the caller's
    ``n_variants == 0`` guard must see no artifact on disk (issue #197 review).
    """
    if not artifacts:
        return _StreamedArtifact(0, 0, 0, time.monotonic() - write_start)
    with gzip.open(out, "wt", encoding="utf-8") as handle:
        handle.write(_ARTIFACT_HEADER)
    _concatenate_window_members(out, artifacts)
    return _StreamedArtifact(
        n_variants=sum(artifact.n_alids for artifact in artifacts),
        n_source_keys=sum(artifact.n_source_keys for artifact in artifacts),
        n_rsids=sum(artifact.n_rsids for artifact in artifacts),
        write_seconds=time.monotonic() - write_start,
    )


def _write_streaming_artifact(
    window_shards: _WindowShards,
    *,
    output_path: str | Path,
    n_workers: int,
) -> _StreamedArtifact:
    """Compress each final window in parallel, then concatenate the members.

    The all-hg38 artifact path (issue #196). Each worker reads one already
    sorted final window shard and writes a standalone gzip member; the parent
    writes the header as the first gzip member and appends each window's raw
    bytes in genomic order. The parent never reads a shard row or builds a
    global site set or lookup.
    """
    write_start = time.monotonic()
    shards = window_shards.shards
    if not shards:
        return _StreamedArtifact(0, 0, 0, time.monotonic() - write_start)
    out = _streaming_output_path(output_path)
    ordered = sorted(shards.items(), key=lambda item: item[0])
    with tempfile.TemporaryDirectory(prefix=".variantrefmembers.") as members_dir_str:
        members_dir = Path(members_dir_str)
        tasks = [
            (window, str(spec.path), str(members_dir / f"{index:06d}.window.tsv.gz"))
            for index, ((_assembly, window), spec) in enumerate(ordered)
        ]
        artifacts = _run_pooled(tasks, _stream_window_artifact, n_workers)
        return _finish_members(out, artifacts, write_start)


# ── streaming hg19/mixed artifact (issue #197) ──────────────────────────────


def _lifted_line(
    rank: tuple[int, int],
    pre_site: tuple[str, int, str, str],
    post_chrom: str,
    post_pos: int,
    rsid: str,
) -> str:
    """One intermediate record: rank, pre-lift site, post-lift locus and rsid."""
    chunk, spill = rank
    chrom, pos, ref, alt = pre_site
    return f"{chunk}\t{spill}\t{chrom}\t{pos}\t{ref}\t{alt}\t{post_chrom}\t{post_pos}\t{rsid}\n"


def _read_assembly_records(
    shards: Sequence[tuple[str, str, int, int]],
) -> tuple[
    dict[tuple[str, int, str, str], tuple[tuple[int, int], str]],
    dict[tuple[str, int, str, str], tuple[tuple[int, int], str]],
]:
    """Read one pre-lift window's final shards into per-assembly site records."""
    from opengwasdb.layouts.dense.build_vcf import _iter_pass1_shard

    hg38_records: dict[tuple[str, int, str, str], tuple[tuple[int, int], str]] = {}
    hg19_records: dict[tuple[str, int, str, str], tuple[tuple[int, int], str]] = {}
    for assembly, path, chunk, spill in shards:
        if assembly == "hg38":
            target = hg38_records
        elif assembly == "hg19":
            target = hg19_records
        else:
            raise ValueError(f"unhandled source_assembly value {assembly!r} in window shards")
        rank = (chunk, spill)
        for site, rsid in _iter_pass1_shard(Path(path)):
            target[site] = (rank, rsid)
    return hg38_records, hg19_records


def _bucket_lifted_records(
    hg38_records: Mapping[tuple[str, int, str, str], tuple[tuple[int, int], str]],
    hg19_records: Mapping[tuple[str, int, str, str], tuple[tuple[int, int], str]],
    lifted: Mapping[tuple[str, int, str, str], str],
    size_bp: int,
) -> tuple[dict[WindowKey, list[str]], int]:
    """Drop cross-assembly ambiguous tuples, then bucket survivors by post-lift window.

    A raw tuple in the hg38 group that also successfully lifted from hg19 is
    dropped from both: because a raw tuple fixes its pre-lift window, the union
    of these window-local intersections is exactly the global rule (issue #197).
    Returns the buckets and the number of dropped tuples.
    """
    collisions = hg38_records.keys() & lifted.keys()
    buckets: dict[WindowKey, list[str]] = {}
    for site, (rank, rsid) in hg38_records.items():
        if site in collisions:
            continue
        buckets.setdefault(window_key(site[0], site[1], size_bp), []).append(
            _lifted_line(rank, site, site[0], site[1], rsid)
        )
    for site, (rank, rsid) in hg19_records.items():
        alid = lifted.get(site)
        if alid is None or site in collisions:
            continue
        post_chrom, post_pos, _a1, _a2 = alid.split(":")
        buckets.setdefault(window_key(post_chrom, int(post_pos), size_bp), []).append(
            _lifted_line(rank, site, post_chrom, int(post_pos), rsid)
        )
    return buckets, len(collisions)


def _stage_liftover_window(task: _LiftStageTask) -> _StagedWindow:
    """Lift one pre-lift window and re-bucket the survivors by post-lift window.

    Both assemblies' final shards are read once, the hg19 rows are lifted with a
    per-worker cached LiftOver, ambiguous raw tuples are dropped from both, and
    each survivor is written to the intermediate sub-shard of the post-lift
    window its hg38 locus falls in. Lift attempts and failures are counted; only
    bucket paths and counts cross back (issue #197).
    """
    from opengwasdb.build.liftover import liftover_batch

    hg38_records, hg19_records = _read_assembly_records(task.shards)
    lifted, attempts, failures = liftover_batch(
        hg19_records, chain_file=task.chain_file, from_build="hg19", to_build="hg38"
    )
    buckets, collisions = _bucket_lifted_records(
        hg38_records, hg19_records, lifted, task.size_bp
    )
    members_dir = Path(task.members_dir)
    specs: list[_LiftedBucket] = []
    for index, (post_window, lines) in enumerate(buckets.items()):
        sub_shard_path = members_dir / f"{task.index:06d}.{index:03d}.lift.tsv"
        with open(sub_shard_path, "w", encoding="utf-8") as handle:
            handle.writelines(lines)
        specs.append(_LiftedBucket(post_window, sub_shard_path))
    return _StagedWindow(tuple(specs), attempts, failures, collisions)


def _read_lifted_records(
    paths: Sequence[str],
) -> list[tuple[int, int, str, int, str, str, str, int, str]]:
    """Read step A's intermediate sub-shards back into typed records."""
    records: list[tuple[int, int, str, int, str, str, str, int, str]] = []
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                (
                    raw_chunk,
                    raw_spill,
                    raw_pre_chrom,
                    raw_pre_pos,
                    raw_pre_ref,
                    raw_pre_alt,
                    raw_post_chrom,
                    raw_post_pos,
                    raw_rsid,
                ) = line.rstrip("\n").split("\t")
                records.append(
                    (
                        int(raw_chunk),
                        int(raw_spill),
                        raw_pre_chrom,
                        int(raw_pre_pos),
                        raw_pre_ref,
                        raw_pre_alt,
                        raw_post_chrom,
                        int(raw_post_pos),
                        raw_rsid,
                    )
                )
    return records


def _compress_lifted_window(task: _LiftMergeTask) -> _WindowArtifact:
    """Merge one post-lift window's intermediate sub-shards into a gzip member.

    Records are ordered by ``(rank, pre-lift site)``, so an ALID several records
    resolve to takes the first non-empty rsid exactly as the materialising
    writer's global ``(rank, site)`` insertion did. Source keys are combined and
    sorted and the window's ALIDs are sorted by ``_alid_sort_key``.
    """
    records = _read_lifted_records(task.sub_shards)
    records.sort(key=lambda record: record[:6])
    rsid_by_alid: dict[str, str] = {}
    source_keys_by_alid: dict[str, list[str]] = {}
    for _chunk, _spill, pre_chrom, pre_pos, pre_ref, pre_alt, post_chrom, post_pos, rsid in records:
        a1, a2 = sorted((pre_ref, pre_alt))
        alid = f"{post_chrom}:{post_pos}:{a1}:{a2}"
        if rsid and alid not in rsid_by_alid:
            rsid_by_alid[alid] = rsid
        source_keys_by_alid.setdefault(alid, []).append(
            f"{pre_chrom}:{pre_pos}:{pre_ref}:{pre_alt}"
        )
    _write_member(task.member_path, rsid_by_alid, source_keys_by_alid)
    return _WindowArtifact(
        task.window,
        Path(task.member_path),
        len(source_keys_by_alid),
        sum(len(keys) for keys in source_keys_by_alid.values()),
        len(rsid_by_alid),
    )


def _stage_all_windows(
    shards: Mapping[tuple[str, WindowKey], _ShardSpec],
    size_bp: int,
    n_workers: int,
    chain_file: str | None,
    work_dir: str,
) -> list[_StagedWindow]:
    """Step A: one lift-and-rebucket task per pre-lift window, in parallel."""
    by_window: dict[WindowKey, list[tuple[str, str, int, int]]] = {}
    for (assembly, window), spec in sorted(shards.items()):
        by_window.setdefault(window, []).append(
            (assembly, str(spec.path), spec.rank[0], spec.rank[1])
        )
    tasks = [
        _LiftStageTask(
            index=index,
            window=window,
            shards=tuple(specs),
            size_bp=size_bp,
            members_dir=work_dir,
            chain_file=chain_file,
        )
        for index, (window, specs) in enumerate(sorted(by_window.items()))
    ]
    return _run_pooled(tasks, _stage_liftover_window, n_workers)


def _enforce_liftover_threshold(staged: Sequence[_StagedWindow], threshold: float) -> None:
    """Aggregate per-window lift counts and fail before writing any artifact."""
    from opengwasdb.build.liftover import LiftoverFailureError

    attempts = sum(stage.attempts for stage in staged)
    failures = sum(stage.failures for stage in staged)
    if not failures:
        return
    rate = failures / attempts
    log.warning(
        "Liftover hg19→hg38: %d/%d variants failed (%.1f%%) across %d window(s)",
        failures,
        attempts,
        rate * 100,
        len(staged),
    )
    if rate > threshold:
        raise LiftoverFailureError(
            f"Liftover failure rate {rate:.1%} ({failures}/{attempts}) exceeds "
            f"threshold {threshold:.1%}"
        )


def _merge_lifted_windows(
    staged: Sequence[_StagedWindow], n_workers: int, work_dir: str
) -> list[_WindowArtifact]:
    """Step B: merge each post-lift window's sub-shards and compress it."""
    buckets_by_post: dict[WindowKey, list[str]] = {}
    for stage in staged:
        for bucket in stage.buckets:
            buckets_by_post.setdefault(bucket.post_window, []).append(str(bucket.path))
    tasks = [
        _LiftMergeTask(
            window=window,
            sub_shards=tuple(paths),
            member_path=str(Path(work_dir) / f"member.{index:06d}.tsv.gz"),
        )
        for index, (window, paths) in enumerate(sorted(buckets_by_post.items()))
    ]
    return _run_pooled(tasks, _compress_lifted_window, n_workers)


def _write_lifted_streaming_artifact(
    window_shards: _WindowShards,
    *,
    output_path: str | Path,
    n_workers: int,
    window_size_mb: float,
    chain_file: str | Path | None,
    liftover_failure_threshold: float,
) -> _StreamedArtifact:
    """Lift, re-window and concatenate every pre-lift window's records.

    Step A lifts each pre-lift window; the parent aggregates attempt/failure
    counts and enforces the threshold before any post-lift member is written.
    Step B merges the intermediate sub-shards per post-lift window and compresses
    each; step C writes the header and appends the post-lift members in genomic
    order. No step reads a row or builds a site set in the parent (issue #197).
    """
    size_bp = window_size_bp(window_size_mb)
    write_start = time.monotonic()
    shards = window_shards.shards
    if not shards:
        return _StreamedArtifact(0, 0, 0, time.monotonic() - write_start)
    out = _streaming_output_path(output_path)
    with tempfile.TemporaryDirectory(prefix=".variantreflift.") as work_dir:
        chain_file_str = str(chain_file) if chain_file is not None else None
        staged = _stage_all_windows(shards, size_bp, n_workers, chain_file_str, work_dir)
        collisions = sum(stage.collisions for stage in staged)
        if collisions:
            log.warning(
                "%d raw variant tuple(s) declared both hg38 and hg19 in this manifest "
                "(same chrom/pos/ref/alt string, two different builds -> two different "
                "physical loci) -- dropped from both rather than guessed which one owns "
                "the stored row",
                collisions,
            )
        _enforce_liftover_threshold(staged, liftover_failure_threshold)
        artifacts = _merge_lifted_windows(staged, n_workers, work_dir)
        return _finish_members(out, artifacts, write_start)


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
