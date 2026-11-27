"""Pass 1's union: every source read once into sorted, windowed shards (issues #5, #188).

Each source's variant sites are mapped into per-window buffers by contiguous,
size-balanced chunks of the source list, spilled as sorted shards, and every
window's shards are tree-reduced in rank order, so the first-named rsid of a
site is the one the serial read would keep (issues #109, #192, #194, #195).
What the core exposes is the final shard per ``(assembly, window)``, not an
in-memory union, so each consumer -- routing's site union, the streaming
Variant Reference writers -- decides what to build from them (issue #193).
"""

from __future__ import annotations

import heapq
import itertools
import logging
import tempfile
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from opengwasdb.build.ordered_pool import fork_pool
from opengwasdb.encoding.timing import format_duration
from opengwasdb.readers.interface import SourceReader, SourceVariant
from opengwasdb.readers.registry import SourceSpec, open_source
from opengwasdb.variants.windows import (
    DEFAULT_MAP_SPILL_RECORDS,
    DEFAULT_REDUCTION_BATCH_SIZE,
    DEFAULT_WINDOW_SIZE_MB,
    WindowKey,
    window_key,
    window_size_bp,
)

log = logging.getLogger(__name__)

#: How Pass 1 opens one source: `open_source`, or a test's fake.
ReaderFactory = Callable[[SourceSpec], SourceReader]


_Pass1Record = tuple[tuple[str, int, str, str], str]


@dataclass(frozen=True)
class ShardSpec:
    """One intermediate variant shard on disk, tagged with its window and rank.

    ``rank`` is ``(chunk_idx, spill_idx)`` -- the manifest-order chunk that
    produced the shard, then the buffer spill sequence within that chunk.
    Merging shards in rank order emits equal sites lowest-rank-first, which is
    what keeps "first named rsid wins" deterministic across the reduction tree
    (issue #109). A tuple rather than the old bare chunk index is what lets a
    later spill from one chunk sort strictly after an earlier one (issue #194):
    Python compares tuples element by element, so manifest order and first-named
    selection are unchanged by spilling.
    """

    rank: tuple[int, int]
    assembly: str
    window: WindowKey
    path: Path


@dataclass(frozen=True)
class WindowShards:
    """The union pass's result: one final sorted shard per ``(assembly, window)``.

    The shards are the seam (issue #193): the map + tree reduce core exposes
    them, and a consumer decides what to build from them. Their paths are only
    valid while the context that yielded this object is open.
    """

    shards: Mapping[tuple[str, WindowKey], ShardSpec]


@dataclass
class Pass1Stats:
    """Structured measurements from the windowed Pass 1 (issue #191).

    Filled in place by the phases so the build log and the scaling benchmark can
    report map, reduce and shard-count figures without decoding log lines. The
    artifact write is timed by the caller that owns it. ``n_reduced_windows``
    counts windows that held more than one shard and therefore actually ran the
    tree reduce -- a single-shard window is already final. ``reduce_levels`` is
    the number of tree-reduce levels that ran; more than one means a window held
    more shards than ``reduction_batch_size`` (issue #194).
    """

    map_seconds: float = 0.0
    reduce_seconds: float = 0.0
    n_windows: int = 0
    n_window_shards: int = 0
    n_reduced_windows: int = 0
    reduce_levels: int = 0


def _record_map_stats(
    stats: Pass1Stats, groups: Mapping[tuple[str, WindowKey], list[ShardSpec]], seconds: float
) -> None:
    """Record the map phase's wall time and the shard shape of its windows."""
    stats.map_seconds = seconds
    stats.n_windows = len(groups)
    stats.n_window_shards = sum(len(shards) for shards in groups.values())
    stats.n_reduced_windows = sum(1 for shards in groups.values() if len(shards) > 1)


def _pass1_record_site(record: _Pass1Record) -> tuple[str, int, str, str]:
    """The variant identity a shard record is sorted and grouped by.

    Both the worker-side sort and the parent-side ``heapq.merge`` compare
    sites only, never the rsid -- the rsid must not influence ordering, or a
    site's rsids would be emitted in rsid-string order instead of manifest
    order and the wrong "first named" would win (issue #109).
    """
    return record[0]


def _source_file_size(source: SourceSpec) -> int:
    """A source's size in bytes, used only to balance the map split.

    A path the split cannot stat counts as one byte rather than zero: the split
    is a scheduling heuristic, and an unreadable source still fails loudly when
    its worker's reader opens it. Zero-weighting it here would let a chunk that
    is not actually free look balanced (issue #195).
    """
    try:
        return Path(source.path).stat().st_size
    except OSError:
        return 1


def _chunks_needed(sizes: Sequence[int], ceiling: int) -> int:
    """Greedy contiguous chunks needed to keep every chunk at or below ``ceiling``."""
    chunks = 1
    current = 0
    for size in sizes:
        if current > 0 and current + size > ceiling:
            chunks += 1
            current = size
        else:
            current += size
    return chunks


def _balanced_ceiling(sizes: Sequence[int], target_chunks: int) -> int:
    """Smallest chunk-weight ceiling a contiguous partition into ``target_chunks`` meets.

    ``_chunks_needed`` is monotone in the ceiling, so the linear-partition
    bottleneck is exact by binary search (issue #195).
    """
    low, high = max(sizes, default=0), sum(sizes)
    while low < high:
        middle = (low + high) // 2
        if _chunks_needed(sizes, middle) <= target_chunks:
            high = middle
        else:
            low = middle + 1
    return low


def _greedy_chunk_bounds(sizes: Sequence[int], ceiling: int) -> list[int]:
    """Start indices of the greedy contiguous partition under ``ceiling``."""
    bounds = [0]
    current = 0
    for index, size in enumerate(sizes):
        if current > 0 and current + size > ceiling:
            bounds.append(index)
            current = size
        else:
            current += size
    bounds.append(len(sizes))
    return bounds


def _heaviest_splittable_chunk(bounds: Sequence[int], prefix: Sequence[int]) -> int:
    """Index of the heaviest chunk holding more than one row, or -1 when none can split."""
    best, best_weight = -1, -1
    for chunk in range(len(bounds) - 1):
        start, end = bounds[chunk], bounds[chunk + 1]
        if end - start > 1 and prefix[end] - prefix[start] > best_weight:
            best = chunk
            best_weight = prefix[end] - prefix[start]
    return best


def _chunk_midpoint(prefix: Sequence[int], start: int, end: int) -> int:
    """The element boundary inside ``(start, end)`` nearest the chunk's half-weight."""
    half = (prefix[end] - prefix[start]) / 2
    return min(range(start + 1, end), key=lambda i: abs(prefix[i] - prefix[start] - half))


def _split_to_target(bounds: list[int], prefix: Sequence[int], target_chunks: int) -> list[int]:
    """Split the heaviest chunks until the contiguous partition has ``target_chunks`` parts.

    The greedy partition holds at most ``target_chunks`` chunks, so this only
    adds boundaries. It runs only while rows outnumber chunks, so a chunk with
    more than one row always exists to split.
    """
    while len(bounds) - 1 < target_chunks:
        chunk = _heaviest_splittable_chunk(bounds, prefix)
        bounds.insert(chunk + 1, _chunk_midpoint(prefix, bounds[chunk], bounds[chunk + 1]))
    return bounds


def _prefix_sums(sizes: Sequence[int]) -> list[int]:
    """``prefix[i]`` is the cumulative size of ``sizes[:i]``."""
    prefix = [0]
    for size in sizes:
        prefix.append(prefix[-1] + size)
    return prefix


def _partition_exactly(
    sizes: Sequence[int], start: int, end: int, chunks: int
) -> list[int]:
    """Boundaries splitting ``sizes[start:end]`` into exactly ``chunks`` contiguous parts.

    Absolute indices within ``sizes``. The local min-max ceiling is found on the
    slice alone, so a dominating source outside it cannot inflate the ceiling and
    leave the slice's own chunks unbalanced (issue #195).
    """
    sub = sizes[start:end]
    bounds = _greedy_chunk_bounds(sub, _balanced_ceiling(sub, chunks))
    return [start + bound for bound in _split_to_target(bounds, _prefix_sums(sub), chunks)]


def _allocate_subchunks(weights: Sequence[int], limits: Sequence[int], extra: int) -> list[int]:
    """Hand ``extra`` additional parts to the chunks with the largest weight-per-part.

    ``limits`` caps each chunk at its row count. The caller guarantees the extra
    parts fit across all chunks; a failure to place one means the split is
    internally inconsistent, so it fails loudly rather than returning a short
    partition (issue #195).
    """
    allocations = [1] * len(weights)
    for _ in range(extra):
        target, target_load = -1, -1.0
        for index in range(len(weights)):
            if allocations[index] >= limits[index]:
                continue
            load = weights[index] / allocations[index]
            if load > target_load:
                target, target_load = index, load
        if target < 0:
            raise ValueError(
                f"cannot allocate {extra} extra chunk(s) across {len(weights)} chunk(s)"
            )
        allocations[target] += 1
    return allocations


def _chunks_from_bounds(
    sources: Sequence[SourceSpec], bounds: Sequence[int]
) -> list[list[SourceSpec]]:
    """Slice ``sources`` between consecutive ``bounds``."""
    return [list(sources[bounds[i] : bounds[i + 1]]) for i in range(len(bounds) - 1)]


def _balanced_bounds(sizes: Sequence[int], target_chunks: int) -> list[int]:
    """Boundaries for exactly ``target_chunks`` contiguous, size-balanced parts.

    The min-max greedy gives at most ``target_chunks`` parts under a ceiling a
    dominating source forces high; the extra parts are then allocated to the
    heaviest chunks and each is subdivided with its own local ceiling, so the
    non-dominating sources are balanced among themselves (issue #195).
    """
    prefix = _prefix_sums(sizes)
    bounds = _greedy_chunk_bounds(sizes, _balanced_ceiling(sizes, target_chunks))
    initial_chunks = len(bounds) - 1
    weights = [prefix[bounds[i + 1]] - prefix[bounds[i]] for i in range(initial_chunks)]
    limits = [bounds[i + 1] - bounds[i] for i in range(initial_chunks)]
    allocations = _allocate_subchunks(weights, limits, target_chunks - initial_chunks)
    final = [0]
    for index, allocation in enumerate(allocations):
        final.extend(
            _partition_exactly(sizes, bounds[index], bounds[index + 1], allocation)[1:]
        )
    return final


def _split_sources(sources: Sequence[SourceSpec], n_workers: int) -> list[list[SourceSpec]]:
    """Contiguous, size-balanced slices of the sources, in their given order.

    Contiguity is what keeps "first named rsid wins" deterministic: chunk
    ``i`` always owns the sources before chunk ``i + 1``, and the reduction merges
    shards by that chunk rank (issue #109). Balancing by on-disk source size
    rather than row count stops one oversized source setting the map phase's
    makespan (issue #195): the partition targets ``min(n, 4 * n_workers)``
    contiguous chunks of near-equal cumulative size, so a dominating source
    lands alone while the rest spread across many chunks the pool can balance.
    Fewer sources than workers leaves each source its own chunk.
    """
    n = len(sources)
    if n == 0:
        return []
    if n <= n_workers:
        return [[source] for source in sources]
    target_chunks = min(n, max(n_workers, 4 * n_workers))
    sizes = [_source_file_size(source) for source in sources]
    return _chunks_from_bounds(sources, _balanced_bounds(sizes, target_chunks))


def _write_pass1_shard(
    shard_dir: Path, worker_idx: int, shard_idx: int, records: list[_Pass1Record]
) -> str:
    """Write one worker's sorted, deduplicated variants for one window.

    One tab-separated record per variant (``chrom pos ref alt rsid``); the
    empty rsid means the worker's slice never named one, so a later shard can
    still supply it at merge time. Sorted here so a merge task streams its
    batch instead of holding it whole.

    The filename is keyed only by ``(worker, per-worker shard index)``, never
    by a chromosome sort rank: ranks are not unique -- ``M`` and ``MT`` are
    both 25, and every unrecognised contig is 1000 -- so a rank-keyed name
    would let one chromosome's shard silently overwrite another's. The window
    is carried in the returned spec, not the name.
    """
    records.sort(key=_pass1_record_site)
    path = shard_dir / f"{worker_idx:06d}.{shard_idx:06d}.pass1.variants.tsv"
    with open(path, "w", encoding="utf-8") as fh:
        for (chrom, pos, ref, alt), rsid in records:
            fh.write(f"{chrom}\t{pos}\t{ref}\t{alt}\t{rsid}\n")
    return str(path)


def iter_shard(path: Path) -> Iterator[_Pass1Record]:
    """Stream one shard's records back as ``(site, rsid)`` tuples."""
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            chrom, pos_str, ref, alt, rsid = line.rstrip("\n").split("\t")
            yield (chrom, int(pos_str), ref, alt), rsid


def _first_named_rsid(group: Iterator[_Pass1Record]) -> str:
    """The first non-empty rsid in one shard-group, in merged (manifest) order."""
    for _site, rsid in group:
        if rsid:
            return rsid
    return ""


def _reduce_worker(task: tuple[str, tuple[str, ...]]) -> str:
    """Merge a batch of rank-ordered sorted shards into one.

    ``heapq.merge`` emits equal sites rank-by-rank and ``_first_named_rsid``
    keeps the earliest non-empty rsid -- exactly the serial read's rule. Memory
    is one record per input shard plus the output buffer, not the window's
    whole union.
    """
    out_path_str, shard_paths = task
    merged = heapq.merge(
        *(iter_shard(Path(path)) for path in shard_paths), key=_pass1_record_site
    )
    with open(out_path_str, "w", encoding="utf-8") as fh:
        for site, group in itertools.groupby(merged, key=_pass1_record_site):
            rsid = _first_named_rsid(group)
            fh.write(f"{site[0]}\t{site[1]}\t{site[2]}\t{site[3]}\t{rsid}\n")
    return out_path_str


def _schedule_reduction_batches(
    groups: dict[tuple[str, WindowKey], list[ShardSpec]],
    batch_size: int,
    tmp_dir: Path,
    level: int,
) -> tuple[list[tuple[str, tuple[str, ...]]], list[tuple[tuple[str, WindowKey], list[ShardSpec]]]]:
    """Build one reduction level's merge tasks and the batch each belongs to."""
    tasks: list[tuple[str, tuple[str, ...]]] = []
    batches: list[tuple[tuple[str, WindowKey], list[ShardSpec]]] = []
    for key, shards in groups.items():
        if len(shards) <= 1:
            continue
        ordered = sorted(shards, key=lambda spec: spec.rank)
        for i in range(0, len(ordered), batch_size):
            batch = ordered[i : i + batch_size]
            target = tmp_dir / f"{level:03d}.{len(tasks):06d}.pass1merge.tsv"
            tasks.append((str(target), tuple(str(spec.path) for spec in batch)))
            batches.append((key, batch))
    return tasks, batches


def _apply_reduction_results(
    groups: dict[tuple[str, WindowKey], list[ShardSpec]],
    batches: list[tuple[tuple[str, WindowKey], list[ShardSpec]]],
    results: list[str],
) -> None:
    """Replace each merged batch with its single output shard."""
    for (key, batch), path in zip(batches, results, strict=True):
        merged = ShardSpec(
            rank=min(spec.rank for spec in batch),
            assembly=batch[0].assembly,
            window=batch[0].window,
            path=Path(path),
        )
        groups[key] = [spec for spec in groups[key] if spec not in batch] + [merged]


def _merge_shards_serial(tasks: list[tuple[str, tuple[str, ...]]]) -> list[str]:
    """Run each merge task in this process (the serial arm's tree-reduce)."""
    return [_reduce_worker(task) for task in tasks]


def _merge_shards_parallel(
    tasks: list[tuple[str, tuple[str, ...]]], pool: ProcessPoolExecutor
) -> list[str]:
    """Submit every merge task to the pool, then collect in task order."""
    futures = [pool.submit(_reduce_worker, task) for task in tasks]
    return [future.result() for future in futures]


def _reduce_shard_groups(
    groups: dict[tuple[str, WindowKey], list[ShardSpec]],
    merge: Callable[[list[tuple[str, tuple[str, ...]]]], list[str]],
    batch_size: int,
    tmp_dir: Path,
    stats: Pass1Stats | None = None,
) -> dict[tuple[str, WindowKey], ShardSpec]:
    """Tree-reduce every window's shards in batches.

    One level at a time: each window's shards are sorted by ``(chunk, spill)``
    rank, batched by ``batch_size`` and merged. Repeats until every window is
    one shard, so a window holding more shards than ``batch_size`` -- many
    spills, say -- is reduced over more than one level (issue #194). A window
    with a single shard is already final. ``merge`` decides whether the batches
    run on a process pool or in this process; the reduction shape is identical.
    """
    level = 0
    while any(len(shards) > 1 for shards in groups.values()):
        tasks, batches = _schedule_reduction_batches(groups, batch_size, tmp_dir, level)
        _apply_reduction_results(groups, batches, merge(tasks))
        level += 1
    if stats is not None:
        stats.reduce_levels = level
    return {key: shards[0] for key, shards in groups.items()}


def _flush_pass1_buffers(
    sites_by_window: dict[tuple[str, WindowKey], dict[tuple[str, int, str, str], str]],
    shard_dir: Path,
    worker_idx: int,
    spill_idx: int,
    shard_idx: int,
    specs: list[ShardSpec],
) -> tuple[int, int]:
    """Write every buffered window as a shard, then clear the buffers.

    Returns the next ``(spill_idx, shard_idx)``. Every shard written here is
    tagged with the same ``(worker_idx, spill_idx)`` rank, so a later spill of
    this chunk can never merge ahead of an earlier one (issue #194).
    """
    for (assembly, window), sites in sites_by_window.items():
        path = _write_pass1_shard(shard_dir, worker_idx, shard_idx, list(sites.items()))
        shard_idx += 1
        specs.append(ShardSpec((worker_idx, spill_idx), assembly, window, Path(path)))
    sites_by_window.clear()
    return spill_idx + 1, shard_idx


def _buffer_pass1_variant(
    sites: dict[tuple[str, int, str, str], str], variant: SourceVariant
) -> bool:
    """Add one variant to a window buffer; return whether its site is new.

    The first non-empty rsid for a site wins and a later empty rsid never
    clears it, matching the serial read. The bool lets ``_pass1_worker`` count
    distinct sites to enforce its spill bound without a second lookup.
    """
    existing = sites.get(variant.site)
    if existing is None:
        sites[variant.site] = variant.rsid or ""
        return True
    if not existing and variant.rsid:
        sites[variant.site] = variant.rsid
    return False


#: One map task: chunk rank, its sources, the shard directory, the window size,
#: the spill threshold, and how to open a source.
Pass1Task = tuple[int, list[SourceSpec], str, int, int, ReaderFactory]


def _pass1_worker(task: Pass1Task) -> list[ShardSpec]:
    """Extract one worker's slice of the sources to sorted, windowed shards.

    Returns only shard metadata -- never the variant sets themselves -- so no
    large object crosses the process pipe (the same rule that governs the
    Pass 2 spills). Each variant is routed to its ``(assembly, window)`` buffer
    and the worker's first non-empty rsid per site is kept, matching the
    serial read.

    The buffers hold at most ``map_spill_records`` distinct sites: once that
    many are buffered every window buffer is written to disk and cleared, so
    peak memory tracks the threshold rather than the number of rows in the
    slice (issue #194). ``spill_idx`` rises with reading order and a site seen
    again in a later spill becomes a later-ranked shard, so the reduction still
    resolves it to the first non-empty rsid in source order.
    """
    worker_idx, sources, shard_dir_str, size_bp, map_spill_records, readers = task
    shard_dir = Path(shard_dir_str)
    sites_by_window: dict[tuple[str, WindowKey], dict[tuple[str, int, str, str], str]] = {}
    specs: list[ShardSpec] = []
    buffered = 0
    spill_idx = 0
    shard_idx = 0
    for source in sources:
        for variant in readers(source).stream_variants():
            key = (source.assembly, window_key(variant.chromosome, variant.position, size_bp))
            if _buffer_pass1_variant(sites_by_window.setdefault(key, {}), variant):
                buffered += 1
            if buffered >= map_spill_records:
                spill_idx, shard_idx = _flush_pass1_buffers(
                    sites_by_window, shard_dir, worker_idx, spill_idx, shard_idx, specs
                )
                buffered = 0
    if sites_by_window:
        _flush_pass1_buffers(sites_by_window, shard_dir, worker_idx, spill_idx, shard_idx, specs)
    return specs


@dataclass(frozen=True)
class UnionOptions:
    """How Pass 1 windows and reduces the union; none of them changes its result.

    ``window_size_mb`` sizes the genomic windows the shards are cut by (issue
    #188), ``reduction_batch_size`` how many shards one merge task takes, and
    ``map_spill_records`` how many distinct sites a map worker buffers before
    spilling, so its peak memory tracks that threshold rather than the size of
    its slice (issue #194). An unusable value fails here, before any source is
    read.
    """

    window_size_mb: float = DEFAULT_WINDOW_SIZE_MB
    reduction_batch_size: int = DEFAULT_REDUCTION_BATCH_SIZE
    map_spill_records: int = DEFAULT_MAP_SPILL_RECORDS

    def __post_init__(self) -> None:
        window_size_bp(self.window_size_mb)
        if self.reduction_batch_size < 2:
            raise ValueError(
                f"reduction batch size must be at least 2, got {self.reduction_batch_size}"
            )
        if self.map_spill_records < 1:
            raise ValueError(
                f"map spill record count must be at least 1, got {self.map_spill_records}"
            )


def consume_shards(
    sources: Sequence[SourceSpec],
    consume: Callable[[WindowShards], Any],
    *,
    n_workers: int,
    options: UnionOptions,
    stats: Pass1Stats | None,
    readers: ReaderFactory = open_source,
) -> Any:
    """Run the union core and hand its final per-window shards to ``consume``.

    This is the assembly seam (issue #193): routing's site union and the
    streaming Variant Reference writers (issues #196/#197) are consumers of the
    same core. The shards exist only while ``consume`` runs.
    """
    with _reduce_windows(
        sources, n_workers=n_workers, options=options, stats=stats, readers=readers
    ) as window_shards:
        return consume(window_shards)


def _finalise_serial_groups(
    groups: dict[tuple[str, WindowKey], list[ShardSpec]],
    reduction_batch_size: int,
    tmp_dir: Path,
    stats: Pass1Stats | None,
) -> dict[tuple[str, WindowKey], ShardSpec]:
    """Reduce the serial map's shards, or return the single-shard windows.

    With one shard per window there is nothing to reduce and the stats report
    no window split; when a low spill threshold left several shards in a
    window the tree reduce runs in this process (issue #194).
    """
    if all(len(shards) == 1 for shards in groups.values()):
        return {key: shards[0] for key, shards in groups.items()}
    if stats is not None:
        _record_map_stats(stats, groups, stats.map_seconds)
    reduce_start = time.monotonic()
    final = _reduce_shard_groups(groups, _merge_shards_serial, reduction_batch_size, tmp_dir, stats)
    if stats is not None:
        stats.reduce_seconds = time.monotonic() - reduce_start
    return final


@dataclass(frozen=True)
class _MapRun:
    """One union pass: the sources, how to window and read them, where the
    shards go, and the stats it fills in."""

    sources: Sequence[SourceSpec]
    size_bp: int
    options: UnionOptions
    tmp_dir: Path
    readers: ReaderFactory
    stats: Pass1Stats | None


def _map_windows_serial(run: _MapRun) -> dict[tuple[str, WindowKey], ShardSpec]:
    """Read every source in this process, then reduce any window that spilled.

    The ``n_workers <= 1`` arm of the core. It runs the same per-window map a
    worker process would, so the serial and parallel arms hand the consumer the
    same shard shape; ``_finalise_serial_groups`` then reduces only when a low
    spill threshold actually split a window (issue #194).
    """
    log.info("Pass 1: collecting source variants from %d files (serial)", len(run.sources))
    start = time.monotonic()
    groups: dict[tuple[str, WindowKey], list[ShardSpec]] = {}
    task: Pass1Task = (
        0, list(run.sources), str(run.tmp_dir), run.size_bp, run.options.map_spill_records,
        run.readers,
    )
    for spec in _pass1_worker(task):
        groups.setdefault((spec.assembly, spec.window), []).append(spec)
    if run.stats is not None:
        run.stats.map_seconds = time.monotonic() - start
    return _finalise_serial_groups(
        groups, run.options.reduction_batch_size, run.tmp_dir, run.stats
    )


def _map_reduce_windows(run: _MapRun, workers: int) -> dict[tuple[str, WindowKey], ShardSpec]:
    """Map sources to window shards, then tree-reduce every window in parallel.

    The sources are split into more contiguous, size-balanced chunks than there
    are workers (issue #195). Each chunk's rank is its index in source order,
    fixed here before any task is submitted, and results are collected in
    completion order -- so which chunk finishes first can never affect shard
    sorting or first-named-rsid resolution.
    """
    tasks: list[Pass1Task] = [
        (chunk_idx, chunk, str(run.tmp_dir), run.size_bp, run.options.map_spill_records,
         run.readers)
        for chunk_idx, chunk in enumerate(_split_sources(run.sources, workers))
    ]
    log.info(
        "Pass 1: %d source(s) → %d size-balanced chunk(s) across %d worker(s)",
        len(run.sources),
        len(tasks),
        workers,
    )
    groups: dict[tuple[str, WindowKey], list[ShardSpec]] = {}
    map_start = time.monotonic()
    with fork_pool(workers) as pool:
        for future in as_completed([pool.submit(_pass1_worker, task) for task in tasks]):
            for spec in future.result():
                groups.setdefault((spec.assembly, spec.window), []).append(spec)
        if run.stats is not None:
            _record_map_stats(run.stats, groups, time.monotonic() - map_start)
        log.info(
            "Pass 1 extraction: %d files → %d window group(s)", len(run.sources), len(groups)
        )
        reduce_start = time.monotonic()
        final = _reduce_shard_groups(
            groups,
            partial(_merge_shards_parallel, pool=pool),
            run.options.reduction_batch_size,
            run.tmp_dir,
            run.stats,
        )
        if run.stats is not None:
            run.stats.reduce_seconds = time.monotonic() - reduce_start
    return final


def materialize_site_union(
    window_shards: WindowShards,
) -> tuple[dict[str, set[tuple[str, int, str, str]]], dict[tuple[str, int, str, str], str]]:
    """Read the final per-window shards into the in-memory site union.

    ``rsid_by_site`` is inserted in ``(rank, site)`` order (issue #192): the
    final shard's manifest-order rank is the primary key and its records are
    already sorted by site. Reading the shards here -- rather than letting the
    union set's hash order decide -- is what makes the rsid an ALID carries
    deterministic.
    """
    tuples_by_assembly: dict[str, set[tuple[str, int, str, str]]] = {}
    rsid_by_site: dict[tuple[str, int, str, str], str] = {}
    ranked: list[tuple[tuple[int, int], tuple[str, int, str, str], str]] = []
    for key in sorted(window_shards.shards):
        spec = window_shards.shards[key]
        sites = tuples_by_assembly.setdefault(key[0], set())
        for site, rsid in iter_shard(spec.path):
            sites.add(site)
            if rsid:
                ranked.append((spec.rank, site, rsid))
    ranked.sort()
    for _rank, site, rsid in ranked:
        rsid_by_site.setdefault(site, rsid)
    return tuples_by_assembly, rsid_by_site


@contextmanager
def _reduce_windows(
    sources: Sequence[SourceSpec],
    *,
    n_workers: int,
    options: UnionOptions,
    stats: Pass1Stats | None,
    readers: ReaderFactory,
) -> Iterator[WindowShards]:
    """Map every source into window shards and tree-reduce each window.

    The union pass core (issues #5, #188): each source is read once and routed
    into sorted per-window shards, and each window's shards are tree-reduced in
    parallel batches. A worker spills its buffers whenever ``map_spill_records``
    variants are buffered, so peak memory tracks that threshold rather than the
    size of its slice (issue #194). What it exposes is the shards -- not an
    in-memory union -- so a consumer can choose what to build from them (issue
    #193).
    """
    size_bp = window_size_bp(options.window_size_mb)
    workers = min(n_workers, len(sources)) if sources else 0
    with tempfile.TemporaryDirectory(prefix=".pass1windows.") as tmp_dir_str:
        run = _MapRun(sources, size_bp, options, Path(tmp_dir_str), readers, stats)
        if workers <= 1:
            yield WindowShards(shards=_map_windows_serial(run))
            return
        log.info(
            "Pass 1: collecting source variants from %d files (parallel, %d workers, "
            "%g Mb windows, batch %d)",
            len(sources),
            workers,
            options.window_size_mb,
            options.reduction_batch_size,
        )
        start = time.monotonic()
        final = _map_reduce_windows(run, workers)
        log.info(
            "Pass 1 merge: %d window(s) in %s",
            len(final),
            format_duration(time.monotonic() - start),
        )
        yield WindowShards(shards=final)
