"""Variant Reference extraction: a lifted union, frozen as an artifact (issue #187).

`extract_variant_reference` runs Pass 1 over a build's sources and writes the
routing a `LiftedUnion` would build -- each ALID with its source keys and its
first-named rsid -- as the ``*.variant-ref.tsv.gz`` artifact a single-pass
build reads back as a `FromReference` authority. It shares Pass 1's union core
and never holds the union: on an all-hg38 union each final window shard is
compressed into its own gzip member by a worker (issue #196); otherwise each
pre-lift window is lifted in a worker, re-bucketed by post-lift window, merged
and compressed (issue #197). The members concatenate in genomic order behind
one header member.

Each window writer compares the names a site's spellings carry as the union's
harvest does, and refuses a key its sources name two ways: an artifact a
build uses as given must not freeze the name of whichever source came first
(issue #379).

The lift here is the window's own, so its failures are counted over the whole
union before anything is written. It projects and orients by the rules
`place_sites` applies: a key's ALID is formed from its alleles as they read on
GRCh38 (`LiftedRow.alid`), and a key whose statistics that projection
re-orients (`statistics_swap`) is written to the artifact's
``reoriented_keys``, so a build reading it re-signs exactly what Pass 1 would
(issue #354).
"""

from __future__ import annotations

import gzip
import logging
import shutil
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TypeVar

from opengwasdb.build.liftover import LiftoverFailureError, lifted_rows, statistics_swap
from opengwasdb.build.ordered_pool import fork_pool
from opengwasdb.readers.registry import SourceSpec, open_source
from opengwasdb.variants.axis import alid_sort_key
from opengwasdb.variants.normalise import canonical_alid, is_primary_contig
from opengwasdb.variants.reference import ARTIFACT_HEADER, artifact_line
from opengwasdb.variants.routing.keys import SourceKey, routing_key
from opengwasdb.variants.routing.lift import cross_assembly_collisions, log_cross_assembly_drop
from opengwasdb.variants.routing.names import refuse_disputed_names
from opengwasdb.variants.routing.union import (
    Pass1Stats,
    ReaderFactory,
    ShardSpec,
    UnionOptions,
    WindowShards,
    consume_shards,
    iter_shard,
)
from opengwasdb.variants.windows import WindowKey, window_key, window_size_bp

log = logging.getLogger(__name__)

#: What extraction's refusal of a disputed name calls it.
EXTRACTION_LABEL = "Variant Reference extraction"


class EmptyReferenceError(ValueError):
    """The sources resolved no variant, so there is no Variant Reference to write.

    Raised before anything is written: an artifact with no rows would read back
    as an axis with nothing on it, an answer indistinguishable from a real one.
    A caller that knows where the sources came from names that instead.
    """


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
class _LiftedBucket:
    """One pre-lift window's intermediate sub-shard for one post-lift window."""

    post_window: WindowKey
    path: Path


@dataclass(frozen=True)
class _StagedWindow:
    """One pre-lift window's step-A result: sub-shards plus lift and drop counts.

    ``reoriented`` counts the window's hg19 keys whose statistics the lift
    re-oriented and which survived the cross-assembly drop, which the parent
    logs (issue #354). ``off_primary`` counts its hg19 keys the lift placed on an
    alt, random or unplaced contig, which a reference does not hold (#376).
    """

    buckets: tuple[_LiftedBucket, ...]
    attempts: int
    failures: int
    collisions: int
    reoriented: int
    off_primary: int = 0


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
    sources: Sequence[SourceSpec],
    output_path: str | Path,
    *,
    chain_file: str | Path | None = None,
    failure_threshold: float = 0.01,
    n_workers: int = 1,
    union: UnionOptions | None = None,
    readers: ReaderFactory = open_source,
) -> VariantReferenceExtraction:
    """Freeze the sources' lifted union as a ``*.variant-ref.tsv.gz`` artifact.

    The standalone form of Pass 1 (issue #187): every source is read once, hg19
    sites are lifted to GRCh38, and the union -- each ALID with its source keys
    and first-named rsid, by the rules a `LiftedUnion` routing follows -- is
    written as the artifact ``--variant-reference`` reads back as a
    `FromReference` authority, so the two-stage Store matches the one-command
    build. The union is never held in memory: each window is written by a
    worker and the members are concatenated in genomic order (issues
    #196/#197).

    ``union`` shapes the windowed tree-reduce and never changes the artifact
    (issues #188, #194); an unusable value fails before any source is read.
    Phase timings and shard counts ride on the result. A key whose statistics
    the lift re-oriented is written under the ALID its projected alleles name
    and listed in its row's ``reoriented_keys`` (issue #354). Raises
    ``ValueError`` when there are no sources, `EmptyReferenceError` when they
    resolve no variant, and `LiftoverFailureError` when the hg19 group fails
    the lift beyond ``failure_threshold``.
    """
    if not sources:
        raise ValueError("no sources to extract a Variant Reference from")
    run = _Extraction(
        sources,
        Path(output_path),
        n_workers,
        union if union is not None else UnionOptions(),
        Pass1Stats(),
        readers,
    )
    if all(source.assembly == "hg38" for source in sources):
        # Already GRCh38: per-window compression, then ordered concatenation.
        return run.write(
            partial(_write_streaming_artifact, output_path=output_path, n_workers=n_workers)
        )
    # hg19 or mixed: lift per pre-lift window, re-window and concatenate.
    return run.write(
        partial(
            _write_lifted_streaming_artifact,
            output_path=output_path,
            n_workers=n_workers,
            window_size_mb=run.union.window_size_mb,
            chain_file=chain_file,
            liftover_failure_threshold=failure_threshold,
        )
    )


@dataclass(frozen=True)
class _Extraction:
    """One extraction: the sources, where the artifact goes, and how to run Pass 1."""

    sources: Sequence[SourceSpec]
    output_path: Path
    n_workers: int
    union: UnionOptions
    stats: Pass1Stats
    readers: ReaderFactory

    def write(
        self, consume: Callable[[WindowShards], _StreamedArtifact]
    ) -> VariantReferenceExtraction:
        """Run the union core and hand its shards to a streaming writer.

        The writer returns only counts and its own phase time, so the parent
        never materialises the union (issues #196/#197).
        """
        streamed: _StreamedArtifact = consume_shards(
            self.sources,
            consume,
            n_workers=self.n_workers,
            options=self.union,
            stats=self.stats,
            readers=self.readers,
        )
        if streamed.n_variants == 0:
            raise EmptyReferenceError("the sources yielded no hg38 variants to reference")
        stats = self.stats
        return VariantReferenceExtraction(
            output_path=self.output_path,
            n_variants=streamed.n_variants,
            n_source_keys=streamed.n_source_keys,
            n_rsids=streamed.n_rsids,
            map_seconds=stats.map_seconds,
            reduce_seconds=stats.reduce_seconds,
            write_seconds=streamed.write_seconds,
            n_windows=stats.n_windows,
            n_window_shards=stats.n_window_shards,
            n_reduced_windows=stats.n_reduced_windows,
            reduce_levels=stats.reduce_levels,
        )


def _write_member(
    member_path: str | Path,
    rsid_by_alid: Mapping[str, str],
    source_keys_by_alid: Mapping[str, list[str]],
    reoriented_by_alid: Mapping[str, list[str]],
) -> None:
    """Write one window's ALIDs, sorted by ``alid_sort_key``, as a gzip member."""
    with gzip.open(member_path, "wt", encoding="utf-8") as handle:
        for alid in sorted(source_keys_by_alid, key=alid_sort_key):
            source_keys = source_keys_by_alid[alid]
            source_keys.sort()
            reoriented = sorted(reoriented_by_alid.get(alid, ()))
            handle.write(
                artifact_line(alid, source_keys, rsid_by_alid.get(alid, ""), reoriented)
            )


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
    window, shard_path, member_path = task
    rsid_by_alid: dict[str, str] = {}
    source_keys_by_alid: dict[str, list[str]] = {}
    named: dict[SourceKey, str] = {}
    n_source_keys = 0
    for (chrom, pos, ref, alt), rsid in iter_shard(Path(shard_path)):
        alid = canonical_alid(chrom, pos, ref, alt)
        if rsid:
            named[(chrom, pos, ref, alt)] = rsid
            rsid_by_alid.setdefault(alid, rsid)
        source_keys_by_alid.setdefault(alid, []).append(f"{chrom}:{pos}:{ref}:{alt}")
        n_source_keys += 1
    refuse_disputed_names(named, label=EXTRACTION_LABEL)
    # Nothing on GRCh38 is lifted, so nothing is re-oriented.
    _write_member(member_path, rsid_by_alid, source_keys_by_alid, {})
    return _WindowArtifact(
        window, Path(member_path), len(source_keys_by_alid), n_source_keys, len(rsid_by_alid)
    )


_T = TypeVar("_T")


_R = TypeVar("_R")


def _run_pooled(tasks: Sequence[_T], worker: Callable[[_T], _R], n_workers: int) -> list[_R]:
    """Run ``worker`` over ``tasks`` in a fork pool, preserving task order."""
    workers = min(n_workers, len(tasks))
    if workers <= 1:
        return [worker(task) for task in tasks]
    with fork_pool(workers) as pool:
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
        handle.write(ARTIFACT_HEADER)
    _concatenate_window_members(out, artifacts)
    return _StreamedArtifact(
        n_variants=sum(artifact.n_alids for artifact in artifacts),
        n_source_keys=sum(artifact.n_source_keys for artifact in artifacts),
        n_rsids=sum(artifact.n_rsids for artifact in artifacts),
        write_seconds=time.monotonic() - write_start,
    )


def _write_streaming_artifact(
    window_shards: WindowShards,
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


def _lifted_line(
    rank: tuple[int, int],
    pre_site: tuple[str, int, str, str],
    alid: str,
    reoriented: bool,
    rsid: str,
) -> str:
    """One intermediate record: rank, pre-lift site, post-lift ALID, whether the
    lift re-oriented the site, and its rsid."""
    chunk, spill = rank
    chrom, pos, ref, alt = pre_site
    return f"{chunk}\t{spill}\t{chrom}\t{pos}\t{ref}\t{alt}\t{alid}\t{int(reoriented)}\t{rsid}\n"


def _read_assembly_records(
    shards: Sequence[tuple[str, str, int, int]],
) -> tuple[
    dict[tuple[str, int, str, str], tuple[tuple[int, int], str]],
    dict[tuple[str, int, str, str], tuple[tuple[int, int], str]],
]:
    """Read one pre-lift window's final shards into per-assembly site records."""
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
        for site, rsid in iter_shard(Path(path)):
            target[site] = (rank, rsid)
    return hg38_records, hg19_records


def _bucket_lifted_records(
    hg38_records: Mapping[tuple[str, int, str, str], tuple[tuple[int, int], str]],
    hg19_records: Mapping[tuple[str, int, str, str], tuple[tuple[int, int], str]],
    lifted: Mapping[tuple[str, int, str, str], str],
    reoriented: AbstractSet[SourceKey],
    size_bp: int,
    off_primary: AbstractSet[SourceKey] = frozenset(),
) -> tuple[dict[WindowKey, list[str]], int, int, int]:
    """Drop cross-assembly ambiguous tuples, then bucket survivors by post-lift window.

    The union's rule, `cross_assembly_collisions`, over the hg38 group and the
    hg19 sites that lifted: every spelling of an identity both declare is
    dropped from both. A routing key fixes its pre-lift window, so the union
    of these window-local rulings is exactly the global one (issue #197).
    ``lifted`` is each lifted site's ALID and ``reoriented`` the sites whose
    statistics the lift re-oriented (issue #354). ``off_primary`` is the hg19
    sites the lift placed off the primary assembly: no bucket holds them, but
    each is a lifted locus for the collision rule (#376). Returns the buckets,
    the number of identities both groups declared, the number of re-oriented
    sites the buckets hold, and the number of off-primary sites left out.
    """
    n_tuples, collisions = _colliding_sites(hg38_records.keys(), lifted.keys() | off_primary)
    buckets: dict[WindowKey, list[str]] = {}
    n_reoriented = 0
    n_off_primary = len(off_primary - collisions)
    for site, (rank, rsid) in hg38_records.items():
        if site in collisions:
            continue
        buckets.setdefault(window_key(site[0], site[1], size_bp), []).append(
            _lifted_line(rank, site, canonical_alid(*site), False, rsid)
        )
    for site, (rank, rsid) in hg19_records.items():
        alid = lifted.get(site)
        if alid is None or site in collisions:
            continue
        post_chrom, post_pos, _a1, _a2 = alid.split(":")
        n_reoriented += site in reoriented
        buckets.setdefault(window_key(post_chrom, int(post_pos), size_bp), []).append(
            _lifted_line(rank, site, alid, site in reoriented, rsid)
        )
    return buckets, n_tuples, n_reoriented, n_off_primary


def _colliding_sites(
    hg38: AbstractSet[SourceKey], hg19: AbstractSet[SourceKey]
) -> tuple[int, set[SourceKey]]:
    """`cross_assembly_collisions` over one window's two groups: the number of
    identities both declare, and every site that spells one of them."""
    dropped = cross_assembly_collisions(hg38, hg19)
    if not dropped:
        return 0, set()
    return len(dropped), {
        site for group in (hg38, hg19) for site in group if routing_key(site) in dropped
    }


def _stage_liftover_window(task: _LiftStageTask) -> _StagedWindow:
    """Lift one pre-lift window and re-bucket the survivors by post-lift window.

    Both assemblies' final shards are read once, the hg19 rows are lifted with a
    per-worker cached LiftOver, ambiguous raw tuples are dropped from both, and
    each survivor is written to the intermediate sub-shard of the post-lift
    window its hg38 locus falls in, under the ALID its projected alleles name
    and with whether the lift re-oriented it. Lift attempts and failures are
    counted; only bucket paths and counts cross back (issues #197, #354).
    """
    hg38_records, hg19_records = _read_assembly_records(task.shards)
    rows, attempts, failures = lifted_rows(
        hg19_records, chain_file=task.chain_file, from_build="hg19", to_build="hg38"
    )
    on_primary = {key: row for key, row in rows.items() if is_primary_contig(row.chromosome)}
    off_primary = rows.keys() - on_primary.keys()
    lifted = {key: row.alid for key, row in on_primary.items()}
    reoriented = {key for key, row in on_primary.items() if statistics_swap(key, row)}
    buckets, collisions, n_reoriented, n_off_primary = _bucket_lifted_records(
        hg38_records, hg19_records, lifted, reoriented, task.size_bp, off_primary
    )
    members_dir = Path(task.members_dir)
    specs: list[_LiftedBucket] = []
    for index, (post_window, lines) in enumerate(buckets.items()):
        sub_shard_path = members_dir / f"{task.index:06d}.{index:03d}.lift.tsv"
        with open(sub_shard_path, "w", encoding="utf-8") as handle:
            handle.writelines(lines)
        specs.append(_LiftedBucket(post_window, sub_shard_path))
    return _StagedWindow(
        tuple(specs), attempts, failures, collisions, n_reoriented, n_off_primary
    )


#: One intermediate record, typed: chunk, spill, the pre-lift site, its
#: post-lift ALID, whether the lift re-oriented it, and its rsid.
_LiftedRecord = tuple[int, int, str, int, str, str, str, bool, str]


def _read_lifted_records(paths: Sequence[str]) -> list[_LiftedRecord]:
    """Read step A's intermediate sub-shards back into typed records."""
    records: list[_LiftedRecord] = []
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
                    raw_alid,
                    raw_reoriented,
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
                        raw_alid,
                        raw_reoriented == "1",
                        raw_rsid,
                    )
                )
    return records


def _compress_lifted_window(task: _LiftMergeTask) -> _WindowArtifact:
    """Merge one post-lift window's intermediate sub-shards into a gzip member.

    Records are ordered by ``(rank, pre-lift site)``, so an ALID several records
    resolve to takes the first non-empty rsid exactly as the materialising
    writer's global ``(rank, site)`` insertion did. Source keys are combined and
    sorted, the ones the lift re-oriented are listed again apart, and the
    window's ALIDs are sorted by ``alid_sort_key``.
    """
    records = _read_lifted_records(task.sub_shards)
    records.sort(key=lambda record: record[:6])
    rsid_by_alid: dict[str, str] = {}
    source_keys_by_alid: dict[str, list[str]] = {}
    reoriented_by_alid: dict[str, list[str]] = {}
    named: dict[SourceKey, str] = {}
    for _chunk, _spill, pre_chrom, pre_pos, pre_ref, pre_alt, alid, reoriented, rsid in records:
        if rsid:
            named.setdefault((pre_chrom, pre_pos, pre_ref, pre_alt), rsid)
            rsid_by_alid.setdefault(alid, rsid)
        source_key = f"{pre_chrom}:{pre_pos}:{pre_ref}:{pre_alt}"
        source_keys_by_alid.setdefault(alid, []).append(source_key)
        if reoriented:
            reoriented_by_alid.setdefault(alid, []).append(source_key)
    refuse_disputed_names(named, label=EXTRACTION_LABEL)
    _write_member(task.member_path, rsid_by_alid, source_keys_by_alid, reoriented_by_alid)
    return _WindowArtifact(
        task.window,
        Path(task.member_path),
        len(source_keys_by_alid),
        sum(len(keys) for keys in source_keys_by_alid.values()),
        len(rsid_by_alid),
    )


def _stage_all_windows(
    shards: Mapping[tuple[str, WindowKey], ShardSpec],
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
    window_shards: WindowShards,
    *,
    output_path: str | Path,
    n_workers: int,
    window_size_mb: float,
    chain_file: str | Path | None,
    liftover_failure_threshold: float,
) -> _StreamedArtifact:
    """Lift, re-window and concatenate every pre-lift window's records.

    Step A lifts each pre-lift window; the parent aggregates attempt/failure
    counts and enforces the threshold before any post-lift member is written,
    and logs how many keys the lift re-oriented (issue #354). Step B merges the
    intermediate sub-shards per post-lift window and compresses each; step C
    writes the header and appends the post-lift members in genomic order. No
    step reads a row or builds a site set in the parent (issue #197).
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
        log_cross_assembly_drop(sum(stage.collisions for stage in staged))
        _enforce_liftover_threshold(staged, liftover_failure_threshold)
        log.info(
            "Liftover re-oriented %d source key(s), recorded in reoriented_keys",
            sum(stage.reoriented for stage in staged),
        )
        if n_off_primary := sum(stage.off_primary for stage in staged):
            log.warning(
                "%d source key(s) lifted onto an alt, random or unplaced contig, which the "
                "Variant Reference does not hold -- left out",
                n_off_primary,
            )
        artifacts = _merge_lifted_windows(staged, n_workers, work_dir)
        return _finish_members(out, artifacts, write_start)

