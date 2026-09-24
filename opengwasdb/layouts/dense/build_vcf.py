"""Two-pass Dense Observed-Only writer from GWAS-VCF manifests with inline liftover.

Association streaming and the union-variant pass go through a ``SourceReader``
resolved from each row's ``source_reader_capability`` (issue #20) rather than
importing ``opengwasdb.build.vcf_source`` directly -- GWAS-VCF remains the only
implementation, so nothing about the build changes, but no bcftools-specific
assumption is left in this module.
"""

from __future__ import annotations

import csv
import heapq
import itertools
import logging
import multiprocessing
import shutil
import tempfile
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from numcodecs import Blosc

from opengwasdb.build.eaf_orientation import (
    DEFAULT_SAMPLE_SITES,
    EafOrientationReport,
    apply_orientation_evidence,
    sample_column_rows,
    site_hashes,
    verify_eaf_orientation,
)
from opengwasdb.build.liftover import LiftoverFailureError, build_liftover_lookup, normalise_build
from opengwasdb.build.ordered_pool import ordered_map
from opengwasdb.encoding import (
    EafExceptionBuilder,
    EafMeasurements,
    EncodingMeasurements,
    StoreCodec,
    StoreEncoding,
    ZOverflowBuilder,
    eaf_baseline_from_grid,
    measure_eaf_sample,
    optimise_dense_se,
    positions_row_band,
    write_eaf_baseline,
)
from opengwasdb.index import initialise_schema, set_metadata
from opengwasdb.layouts.dense.build import (
    DenseBuildResult,
    add_hit_counts,
    write_analyses_tsv,
)
from opengwasdb.layouts.dense.constants import (
    DEFAULT_CHUNK_SHAPE,
    DEFAULT_COMPRESSOR,
    DEFAULT_DTYPE,
    TOP_HIT_THRESHOLDS,
    dense_index_metadata,
)
from opengwasdb.layouts.dense.top_hits import (
    write_top_hit_indexes_for_store,
    z_critical,
)
from opengwasdb.model.analyses import Analysis, PassthroughMetadata
from opengwasdb.model.enums import (
    AssociationCoverage,
    CompletionState,
    EafScope,
    OriginalSdMethod,
    PrimaryStorageLayout,
    StoredEffectScale,
)
from opengwasdb.model.manifest import StoreManifest
from opengwasdb.model.manifest_columns import (
    ManifestColumns,
    manifest_n,
    manifest_trait_name,
    require_columns,
    resolve_manifest_columns,
)
from opengwasdb.readers.gwas_vcf import GWAS_VCF_CAPABILITY
from opengwasdb.readers.interface import SourceVariant
from opengwasdb.readers.registry import known_capabilities, resolve_reader
from opengwasdb.store.open import CURRENT_FORMAT_VERSION, OpenGWASDBStore, StagedRelease
from opengwasdb.variants import CanonicalVariant, write_variant_axis
from opengwasdb.variants.normalise import chromosome_sort_key, normalise_chromosome
from opengwasdb.variants.reference import VariantReference, read_variant_reference
from opengwasdb.variants.windows import (
    DEFAULT_MAP_SPILL_RECORDS,
    DEFAULT_REDUCTION_BATCH_SIZE,
    DEFAULT_WINDOW_SIZE_MB,
    WindowKey,
    window_key,
    window_size_bp,
)

log = logging.getLogger(__name__)

# One compressor for every dense statistic array (z/se/eaf), so a new array
# cannot quietly ship with different settings from the ones beside it.
_DENSE_COMPRESSOR = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)

__all__ = ["build_dense_from_vcf_manifest", "LiftoverFailureError"]


@dataclass(frozen=True)
class _ManifestRow:
    trait_id: str
    file_path: str
    trait_name: str
    n: int
    stored_effect_scale: str
    se_divisor: float  # divides original_se to standardise to SD units (issue #18); 1.0 = no-op
    source_reader_capability: str  # resolves to a SourceReader (issue #20); GWAS_VCF_CAPABILITY
    # when the manifest omits the column -- the only format this builder has ever supported.
    source_assembly: str  # normalised "hg19"/"hg38" (issue #85); "hg19" when the manifest omits
    # the column -- every source this builder read before GWAS-SSF (#84) was hg19 GWAS-VCF.
    original_sd: str  # raw manifest value ("" when the sd_method tier carries no magnitude);
    # not part of PassthroughMetadata because this builder *uses* it (se_divisor), rather
    # than only copying it.
    assigned_ancestry: str  # optional manifest column (issue #22); "" when the manifest omits it
    # or carries no assignment for this row -- e.g. a Catalogue subset's kept rows (already
    # filtered to one target ancestry) stamp this in verbatim; a bare manifest has no column.
    # Shared-core analyses.tsv columns the manifest supplies and this builder only copies
    # (issues #86, #83) -- sample-size interpretation and counts, Original Effect Scale,
    # ancestry-assignment method and proportions, Attribution. Blank when absent; the shared
    # builder must not infer their meaning. One type, shared with Ragged, so a column added
    # to the contract reaches both layouts.
    metadata: PassthroughMetadata = field(default_factory=PassthroughMetadata)
    # Trait-ontology columns (ADR 0034, issue #68): optional manifest columns, resolved
    # directly rather than via a phenotype_id/phenotype_label intermediary. "" when the
    # manifest omits them -- never fabricated. Not passthrough-shared: Ragged resolves its
    # own (a gene ID, ADR 0035) rather than copying a manifest column.
    trait_ontology_id: str = ""
    trait_ontology_label: str = ""


def _manifest_row_to_analysis(row: _ManifestRow) -> Analysis:
    """A manifest row's Analytical + Attribution Metadata (issue #22, ADR
    0034), shared by the dense and hybrid manifest builders.

    Optional fields are passed through verbatim and remain blank when absent.
    In particular, the shared builder cannot infer how ancestry was assigned:
    the manifest producer owns that fact (issue #86).
    """
    return row.metadata.applied_to(
        Analysis(
            analysis_id=row.trait_id,
            analysis_label=row.trait_name,
            trait_ontology_id=row.trait_ontology_id,
            trait_ontology_label=row.trait_ontology_label,
            stored_effect_scale=row.stored_effect_scale,
            assigned_ancestry=row.assigned_ancestry,
            sample_size=str(row.n) if row.n else "",
            original_sd=row.original_sd,
        )
    )


# original_sd_method tiers that carry an actual phenotype-SD magnitude to divide
# by (ADR-0029 methods 2-5). declared_standardised implies sd=1 (no magnitude
# recorded); binary_trait is not SD-scale at all -- neither rescales.
_SD_RESCALE_METHODS = frozenset(
    {
        OriginalSdMethod.SOURCE_PROVIDED,
        OriginalSdMethod.ESTIMATED_FROM_SOURCE_MAF,
        OriginalSdMethod.ESTIMATED_FROM_REFERENCE_MAF,
        OriginalSdMethod.ESTIMATED_FROM_BETA_DISTRIBUTION,
    }
)


# Pass 2 fork-safe lookup + spill dir, set in the parent immediately before the
# process pool is created. Forked workers (fork start method — see _fork_pool())
# inherit these; the lookup is a pair of numpy arrays — a sorted object array of
# Python bytes keys (variable-length, so one rare 546-byte indel no longer pads
# every key to 546 bytes) and an int32 row array — rather than Python dicts. A
# worker binary-searches the key array without chaining ~n_variants dict pages,
# so it avoids the refcount-COW that forced issue 043 item 2.
#
# Why disk-spill and not return arrays: an earlier design returned each file's
# result over IPC. At genome-wide scale (~9.85M rows/file × thousands of files)
# that pipe traffic deadlocked the pool. Workers write a compact per-file .npz to
# _pass2_spill_dir and return only col_idx, so no large object crosses the pipe.
_pass2_keys_sorted: np.ndarray | None = None  # sorted object array of bytes keys
_pass2_rows_sorted: np.ndarray | None = None  # int32 row per key, same order
_pass2_spill_dir: Path | None = None

# Top hits are harvested inline during Pass 2 rather than by a post-hoc scan of
# the full matrix (which had to reload ~200 GB of float32 and compute a p-value
# for every finite cell). Each column emits only the cells clearing the loosest
# threshold's |z| cutoff; the band-write phase accumulates them for the index.
_TOP_HIT_Z_CRIT = z_critical(max(TOP_HIT_THRESHOLDS))


def _fork_pool(n_workers: int) -> ProcessPoolExecutor:
    """A ProcessPoolExecutor pinned to fork start — required for _pass2_worker to
    inherit the numpy lookup arrays without re-pickling them per task. Only
    correct on platforms with fork (Linux)."""
    fork_ctx = multiprocessing.get_context("fork")
    return ProcessPoolExecutor(max_workers=n_workers, mp_context=fork_ctx)


def _encode_variant_keys(chrom: object, pos: object, ref: object, alt: object) -> np.ndarray:
    """Encode variant fields as ``chrom:pos:ref:alt`` byte-string keys, vectorised.

    Used identically for the parent's sorted key table and each worker's query
    keys, so the two encodings match exactly. ``pos`` int->bytes via numpy astype
    (``np.int64(10).astype('S') == b'10'``) avoids per-element Python string work.
    """
    chrom_b = np.asarray(chrom, dtype="S")
    pos_b = np.asarray(pos, dtype=np.int64).astype("S")
    ref_b = np.asarray(ref, dtype="S")
    alt_b = np.asarray(alt, dtype="S")
    key = np.char.add(chrom_b, b":")
    key = np.char.add(key, pos_b)
    key = np.char.add(key, b":")
    key = np.char.add(key, ref_b)
    key = np.char.add(key, b":")
    key = np.char.add(key, alt_b)
    return key


def _build_variant_key_index(
    source_lookup: dict[tuple[str, int, str, str], str],
    variant_index: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Compose the two Pass 2 dicts into a fork-safe sorted numpy lookup.

    Returns ``(keys_sorted, rows_sorted)``: a sorted array of byte keys for every
    source variant that maps to a stored row, and the matching int32 row indices.
    Workers binary-search this instead of chaining two Python dicts.
    """
    chroms: list[str] = []
    poss: list[int] = []
    refs: list[str] = []
    alts: list[str] = []
    rows: list[int] = []
    for (chrom, pos, ref, alt), alid in source_lookup.items():
        row = variant_index.get(alid)
        if row is None:
            continue
        chroms.append(chrom)
        poss.append(pos)
        refs.append(ref)
        alts.append(alt)
        rows.append(row)
    # Encode as Python bytes and sort the object array directly. ``_encode_variant_keys``
    # returns an ``S`` array padded to the longest key in the batch; with a rare
    # 546-byte indel present that pads every one of ~21.3M keys to 546 bytes
    # (~11.6 GB) and makes ``np.argsort`` walk the whole padded buffer. Python bytes
    # objects stay at their real length, so the sort and the fork-inherited key
    # table cost memory proportional to the actual key bytes, not the maximum.
    keys_list = [
        f"{chrom}:{pos}:{ref}:{alt}".encode()
        for chrom, pos, ref, alt in zip(chroms, poss, refs, alts, strict=True)
    ]
    keys = np.array(keys_list, dtype=object)
    del keys_list, chroms, poss, refs, alts
    rows_arr = np.array(rows, dtype=np.int32)
    del rows
    order = np.argsort(keys, kind="stable")
    return keys[order], rows_arr[order]


# Streamed in fixed-size batches so a worker never materialises a whole
# (genome-wide) VCF as Python lists at once — peak per-worker memory is one batch
# of columns plus the accumulated matched rows, not the entire file (issue 043).
_RESOLVE_BATCH = 250_000


def _match_batch(
    chroms: list[str],
    poss: list[int],
    refs: list[str],
    alts: list[str],
    zs: list[float],
    ses: list[float],
    eafs: list[float],
    keys_sorted: np.ndarray,
    rows_sorted: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Resolve one batch of associations to (rows, z, se, eaf) for cells whose
    variant is present in the panel (exact key match). Order-preserving.
    `eaf` is NaN for associations whose source reported none (ADR 0036)."""
    if len(keys_sorted) == 0:
        empty_i = np.empty(0, dtype=np.int64)
        empty_f = np.empty(0, dtype=np.float32)
        return empty_i, empty_f, empty_f, empty_f
    query = _encode_variant_keys(chroms, poss, refs, alts)
    idx = np.searchsorted(keys_sorted, query)
    idx_clip = np.minimum(idx, len(keys_sorted) - 1)
    matched = keys_sorted[idx_clip] == query
    rows = rows_sorted[idx_clip[matched]].astype(np.int64)
    z_arr = np.array(zs, dtype=np.float32)[matched]
    se_arr = np.array(ses, dtype=np.float32)[matched]
    eaf_arr = np.array(eafs, dtype=np.float32)[matched]
    return rows, z_arr, se_arr, eaf_arr


def _apply_se_divisor(se: np.ndarray, se_divisor: float) -> np.ndarray:
    """Continuous-trait phenotype-SD standardisation (issue #18):
    ``stored_se = original_se / sd``. Shared by the dense and hybrid builders,
    both of which resolve an association stream to ``(index, z, se)`` and need
    the same no-op-when-1.0 divide applied to ``se`` before spilling."""
    if se_divisor == 1.0:
        return se
    return se / np.float32(se_divisor)


def _resolve_column(
    file_path: str,
    keys_sorted: np.ndarray,
    rows_sorted: np.ndarray,
    se_divisor: float = 1.0,
    *,
    capability: str = GWAS_VCF_CAPABILITY,
    stored_effect_scale: str = StoredEffectScale.SD.value,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stream one source file, resolve each association's variant to a row via
    the sorted key lookup, and return deduped
    ``(rows int64, z f32, se f32, eaf f32)`` for the column. ``eaf`` is NaN
    where the source reports no frequency (ADR 0036).

    The stream is processed in ``_RESOLVE_BATCH``-sized batches (vectorised
    searchsorted per batch), so worker peak memory is bounded by one batch rather
    than the whole file. Batches are matched in stream order and concatenated, so
    the final last-wins dedup by row is identical to processing the whole file at
    once: when two source variants lift to the same row, the later stream
    occurrence wins, making the scattered matrix cell deterministic.

    ``capability`` resolves a ``SourceReader`` (issue #20) rather than this
    module streaming a VCF itself -- ``stored_effect_scale`` is required to
    construct one but unused past construction here (it is a Pass-2 concern
    only for readers that attach it to each yielded association).

    ``se_divisor`` divides the returned ``se`` (continuous-trait phenotype-SD
    standardisation, issue #18: ``stored_se = original_se / sd``). ``z`` is left
    untouched -- ``z = beta/se`` is invariant to dividing both by the same
    constant, so only ``se`` needs rescaling. Defaults to 1.0 (no-op) for
    binary-trait analyses and callers that pre-date issue #18.
    """
    reader = resolve_reader(capability, file_path, StoredEffectScale(stored_effect_scale))
    rows_parts: list[np.ndarray] = []
    z_parts: list[np.ndarray] = []
    se_parts: list[np.ndarray] = []
    eaf_parts: list[np.ndarray] = []
    chroms: list[str] = []
    poss: list[int] = []
    refs: list[str] = []
    alts: list[str] = []
    zs: list[float] = []
    ses: list[float] = []
    eafs: list[float] = []

    def _flush() -> None:
        if not zs:
            return
        r, z, se, eaf = _match_batch(
            chroms, poss, refs, alts, zs, ses, eafs, keys_sorted, rows_sorted
        )
        rows_parts.append(r)
        z_parts.append(z)
        se_parts.append(se)
        eaf_parts.append(eaf)
        chroms.clear()
        poss.clear()
        refs.clear()
        alts.clear()
        zs.clear()
        ses.clear()
        eafs.clear()

    for assoc in reader.stream_associations():
        chroms.append(assoc.chromosome)
        poss.append(assoc.position)
        refs.append(assoc.ref)
        alts.append(assoc.alt)
        zs.append(assoc.z)
        ses.append(assoc.se)
        eafs.append(float("nan") if assoc.eaf is None else assoc.eaf)
        if len(zs) >= _RESOLVE_BATCH:
            _flush()
    _flush()

    if not rows_parts:
        empty_i = np.empty(0, dtype=np.int64)
        empty_f = np.empty(0, dtype=np.float32)
        return empty_i, empty_f, empty_f, empty_f

    rows = np.concatenate(rows_parts)
    z_arr = np.concatenate(z_parts)
    se_arr = np.concatenate(se_parts)
    eaf_arr = np.concatenate(eaf_parts)

    if len(rows):
        # last-wins dedup by row: unique on the reversed rows returns the first
        # index in reversed order == the last occurrence in original order.
        _, first_in_rev = np.unique(rows[::-1], return_index=True)
        keep = np.sort(len(rows) - 1 - first_in_rev)
        rows, z_arr, se_arr, eaf_arr = rows[keep], z_arr[keep], se_arr[keep], eaf_arr[keep]
    se_arr = _apply_se_divisor(se_arr, se_divisor)
    return rows, z_arr, se_arr, eaf_arr


def _spill_column(
    spill_dir: Path,
    col_idx: int,
    rows: np.ndarray,
    z: np.ndarray,
    se: np.ndarray,
    eaf: np.ndarray,
) -> None:
    """Atomically spill one resolved column to ``{spill_dir}/{col_idx}.npz``
    (temp-then-rename; both names end in .npz because np.savez appends that suffix
    unless already present).

    Top hits are NOT harvested here — they are harvested during the band-write
    phase from the *stored* float16 values, so the index matches exactly what a
    query reads back from the ``z`` array (issue 046)."""
    final = spill_dir / f"{col_idx}.npz"
    tmp = spill_dir / f"{col_idx}.tmp.npz"
    np.savez(tmp, rows=rows, z=z, se=se, eaf=eaf)
    tmp.replace(final)


def _pass2_worker(task: tuple[int, str, float, str, str]) -> int:
    """Resolve one column against the fork-inherited numpy lookup and spill it.
    Returns col_idx only — the compact result stays on disk, never in a pipe."""
    assert _pass2_keys_sorted is not None
    assert _pass2_rows_sorted is not None
    assert _pass2_spill_dir is not None
    col_idx, file_path, se_divisor, capability, stored_effect_scale = task
    rows, z, se, eaf = _resolve_column(
        file_path,
        _pass2_keys_sorted,
        _pass2_rows_sorted,
        se_divisor,
        capability=capability,
        stored_effect_scale=stored_effect_scale,
    )
    _spill_column(_pass2_spill_dir, col_idx, rows, z, se, eaf)
    return col_idx


def _pass2_worker_tasks(
    manifest_rows: Sequence[_ManifestRow],
    analysis_index: Mapping[str, int],
) -> list[tuple[int, str, float, str, str]]:
    """One fork-pool task per manifest row, in manifest order: the column
    index from ``analysis_index`` followed by the four source fields
    ``_pass2_worker`` needs to resolve the row's file.

    The Dense and Hybrid builders submit identical task tuples to their own
    ``_pass2_worker``s, so the list is built here once rather than at each
    call site.
    """
    return [
        (
            analysis_index[row.trait_id],
            row.file_path,
            row.se_divisor,
            row.source_reader_capability,
            row.stored_effect_scale,
        )
        for row in manifest_rows
    ]


def _log_progress(
    label: str, completed: int, total: int, start_time: float, extra: str, every: int
) -> None:
    if completed % every != 0 and completed != total:
        return
    elapsed = time.monotonic() - start_time
    eta = (elapsed / completed) * (total - completed) if completed else 0.0
    log.info(
        "%s: %d/%d done (%s) — elapsed %s, ETA %s",
        label,
        completed,
        total,
        extra,
        _fmt_duration(elapsed),
        _fmt_duration(eta),
    )


_Pass1Record = tuple[tuple[str, int, str, str], str]


@dataclass(frozen=True)
class _ShardSpec:
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
class _WindowShards:
    """The union pass's result: one final sorted shard per ``(assembly, window)``.

    The shards are the seam (issue #193): the map + tree reduce core exposes
    them, and a consumer decides what to build from them. Their paths are only
    valid while the context that yielded this object is open.
    """

    shards: Mapping[tuple[str, WindowKey], _ShardSpec]


@dataclass
class _Pass1Stats:
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
    stats: _Pass1Stats, groups: Mapping[tuple[str, WindowKey], list[_ShardSpec]], seconds: float
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


def _source_file_size(row: _ManifestRow) -> int:
    """A manifest row's source size in bytes, used only to balance the map split.

    A path the split cannot stat counts as one byte rather than zero: the split
    is a scheduling heuristic, and an unreadable source still fails loudly when
    its worker's reader opens it. Zero-weighting it here would let a chunk that
    is not actually free look balanced (issue #195).
    """
    try:
        return Path(row.file_path).stat().st_size
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
    manifest_rows: list[_ManifestRow], bounds: Sequence[int]
) -> list[list[_ManifestRow]]:
    """Slice ``manifest_rows`` between consecutive ``bounds``."""
    return [manifest_rows[bounds[i] : bounds[i + 1]] for i in range(len(bounds) - 1)]


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


def _split_manifest_rows(
    manifest_rows: list[_ManifestRow], n_workers: int
) -> list[list[_ManifestRow]]:
    """Contiguous, size-balanced, manifest-order slices of the input rows.

    Contiguity is what keeps "first named rsid wins" deterministic: chunk
    ``i`` always owns the rows before chunk ``i + 1``, and the reduction merges
    shards by that chunk rank (issue #109). Balancing by on-disk source size
    rather than row count stops one oversized source setting the map phase's
    makespan (issue #195): the partition targets ``min(n, 4 * n_workers)``
    contiguous chunks of near-equal cumulative size, so a dominating source
    lands alone while the rest spread across many chunks the pool can balance.
    Fewer sources than workers leaves each source its own chunk.
    """
    n = len(manifest_rows)
    if n == 0:
        return []
    if n <= n_workers:
        return [[row] for row in manifest_rows]
    target_chunks = min(n, max(n_workers, 4 * n_workers))
    sizes = [_source_file_size(row) for row in manifest_rows]
    return _chunks_from_bounds(manifest_rows, _balanced_bounds(sizes, target_chunks))


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


def _iter_pass1_shard(path: Path) -> Iterator[_Pass1Record]:
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
        *(_iter_pass1_shard(Path(path)) for path in shard_paths), key=_pass1_record_site
    )
    with open(out_path_str, "w", encoding="utf-8") as fh:
        for site, group in itertools.groupby(merged, key=_pass1_record_site):
            rsid = _first_named_rsid(group)
            fh.write(f"{site[0]}\t{site[1]}\t{site[2]}\t{site[3]}\t{rsid}\n")
    return out_path_str


def _schedule_reduction_batches(
    groups: dict[tuple[str, WindowKey], list[_ShardSpec]],
    batch_size: int,
    tmp_dir: Path,
    level: int,
) -> tuple[list[tuple[str, tuple[str, ...]]], list[tuple[tuple[str, WindowKey], list[_ShardSpec]]]]:
    """Build one reduction level's merge tasks and the batch each belongs to."""
    tasks: list[tuple[str, tuple[str, ...]]] = []
    batches: list[tuple[tuple[str, WindowKey], list[_ShardSpec]]] = []
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
    groups: dict[tuple[str, WindowKey], list[_ShardSpec]],
    batches: list[tuple[tuple[str, WindowKey], list[_ShardSpec]]],
    results: list[str],
) -> None:
    """Replace each merged batch with its single output shard."""
    for (key, batch), path in zip(batches, results, strict=True):
        merged = _ShardSpec(
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
    groups: dict[tuple[str, WindowKey], list[_ShardSpec]],
    merge: Callable[[list[tuple[str, tuple[str, ...]]]], list[str]],
    batch_size: int,
    tmp_dir: Path,
    stats: _Pass1Stats | None = None,
) -> dict[tuple[str, WindowKey], _ShardSpec]:
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
    specs: list[_ShardSpec],
) -> tuple[int, int]:
    """Write every buffered window as a shard, then clear the buffers.

    Returns the next ``(spill_idx, shard_idx)``. Every shard written here is
    tagged with the same ``(worker_idx, spill_idx)`` rank, so a later spill of
    this chunk can never merge ahead of an earlier one (issue #194).
    """
    for (assembly, window), sites in sites_by_window.items():
        path = _write_pass1_shard(shard_dir, worker_idx, shard_idx, list(sites.items()))
        shard_idx += 1
        specs.append(_ShardSpec((worker_idx, spill_idx), assembly, window, Path(path)))
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


def _pass1_worker(task: tuple[int, list[_ManifestRow], str, int, int]) -> list[_ShardSpec]:
    """Extract one worker's slice of manifest rows to sorted, windowed shards.

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
    resolves it to the first non-empty rsid in manifest order.
    """
    worker_idx, rows, shard_dir_str, size_bp, map_spill_records = task
    shard_dir = Path(shard_dir_str)
    sites_by_window: dict[tuple[str, WindowKey], dict[tuple[str, int, str, str], str]] = {}
    specs: list[_ShardSpec] = []
    buffered = 0
    spill_idx = 0
    shard_idx = 0
    for row in rows:
        reader = resolve_reader(
            row.source_reader_capability, row.file_path, StoredEffectScale(row.stored_effect_scale)
        )
        for variant in reader.stream_variants():
            key = (row.source_assembly, window_key(variant.chromosome, variant.position, size_bp))
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


def _validate_union_options(
    window_size_mb: float, reduction_batch_size: int, map_spill_records: int
) -> None:
    """Fail loudly on an unusable window, batch or spill setting before any I/O."""
    window_size_bp(window_size_mb)
    if reduction_batch_size < 2:
        raise ValueError(f"reduction batch size must be at least 2, got {reduction_batch_size}")
    if map_spill_records < 1:
        raise ValueError(
            f"map spill record count must be at least 1, got {map_spill_records}"
        )


def _consume_manifest_shards(
    manifest_rows: list[_ManifestRow],
    *,
    n_workers: int,
    window_size_mb: float,
    reduction_batch_size: int,
    map_spill_records: int,
    stats: _Pass1Stats | None,
    consume: Callable[[_WindowShards], Any],
) -> Any:
    """Run the union core and hand its final per-window shards to ``consume``.

    This is the assembly seam (issue #193): the in-memory site union and Dense /
    Hybrid's materialised lookup are both consumers of the same core, and a
    later streaming artifact writer (issues #196/#197) plugs in the same way.
    """
    _validate_union_options(window_size_mb, reduction_batch_size, map_spill_records)
    with _reduce_manifest_windows(
        manifest_rows,
        n_workers=n_workers,
        window_size_mb=window_size_mb,
        reduction_batch_size=reduction_batch_size,
        map_spill_records=map_spill_records,
        stats=stats,
    ) as window_shards:
        return consume(window_shards)


def _collect_manifest_variant_sites(
    manifest_rows: list[_ManifestRow],
    *,
    n_workers: int = 1,
    window_size_mb: float = DEFAULT_WINDOW_SIZE_MB,
    reduction_batch_size: int = DEFAULT_REDUCTION_BATCH_SIZE,
    map_spill_records: int = DEFAULT_MAP_SPILL_RECORDS,
    stats: _Pass1Stats | None = None,
) -> tuple[dict[str, set[tuple[str, int, str, str]]], dict[tuple[str, int, str, str], str]]:
    """Pass 1: read every manifest source once for its variant sites.

    Returns the ``tuples_by_assembly`` union and ``rsid_by_site`` (first named
    rsid wins, issue #109). The map + tree-reduce core
    (`_reduce_manifest_windows`) exposes the final per-window shards, and this
    consumer materialises them into memory (issue #193). ``stats``, when given,
    is filled in place with the map/reduce timings and shard counts (#191).
    """
    return _consume_manifest_shards(
        manifest_rows, n_workers=n_workers, window_size_mb=window_size_mb,
        reduction_batch_size=reduction_batch_size, map_spill_records=map_spill_records,
        stats=stats, consume=_materialize_site_union,
    )


def _finalise_serial_groups(
    groups: dict[tuple[str, WindowKey], list[_ShardSpec]],
    reduction_batch_size: int,
    tmp_dir: Path,
    stats: _Pass1Stats | None,
) -> dict[tuple[str, WindowKey], _ShardSpec]:
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


def _map_manifest_windows_serial(
    manifest_rows: list[_ManifestRow],
    size_bp: int,
    reduction_batch_size: int,
    map_spill_records: int,
    tmp_dir: Path,
    stats: _Pass1Stats | None,
) -> dict[tuple[str, WindowKey], _ShardSpec]:
    """Read every source in this process, then reduce any window that spilled.

    The ``n_workers <= 1`` arm of the core. It runs the same per-window map a
    worker process would, so the serial and parallel arms hand the consumer the
    same shard shape; ``_finalise_serial_groups`` then reduces only when a low
    spill threshold actually split a window (issue #194).
    """
    log.info("Pass 1: collecting source variants from %d files (serial)", len(manifest_rows))
    start = time.monotonic()
    groups: dict[tuple[str, WindowKey], list[_ShardSpec]] = {}
    for spec in _pass1_worker((0, manifest_rows, str(tmp_dir), size_bp, map_spill_records)):
        groups.setdefault((spec.assembly, spec.window), []).append(spec)
    if stats is not None:
        stats.map_seconds = time.monotonic() - start
    return _finalise_serial_groups(groups, reduction_batch_size, tmp_dir, stats)


def _map_reduce_windows(
    manifest_rows: list[_ManifestRow],
    workers: int,
    size_bp: int,
    reduction_batch_size: int,
    map_spill_records: int,
    tmp_dir: Path,
    stats: _Pass1Stats | None = None,
) -> dict[tuple[str, WindowKey], _ShardSpec]:
    """Map rows to window shards, then tree-reduce every window in parallel.

    The manifest is split into more contiguous, size-balanced chunks than there
    are workers (issue #195). Each chunk's rank is its manifest-order index,
    fixed here before any task is submitted, and results are collected in
    completion order -- so which chunk finishes first can never affect shard
    sorting or first-named-rsid resolution.
    """
    tasks = [
        (chunk_idx, chunk, str(tmp_dir), size_bp, map_spill_records)
        for chunk_idx, chunk in enumerate(_split_manifest_rows(manifest_rows, workers))
    ]
    log.info(
        "Pass 1: %d source(s) → %d size-balanced chunk(s) across %d worker(s)",
        len(manifest_rows),
        len(tasks),
        workers,
    )
    groups: dict[tuple[str, WindowKey], list[_ShardSpec]] = {}
    map_start = time.monotonic()
    with _fork_pool(workers) as pool:
        for future in as_completed([pool.submit(_pass1_worker, task) for task in tasks]):
            for spec in future.result():
                groups.setdefault((spec.assembly, spec.window), []).append(spec)
        if stats is not None:
            _record_map_stats(stats, groups, time.monotonic() - map_start)
        log.info(
            "Pass 1 extraction: %d files → %d window group(s)",
            len(manifest_rows),
            len(groups),
        )
        reduce_start = time.monotonic()
        final = _reduce_shard_groups(
            groups,
            partial(_merge_shards_parallel, pool=pool),
            reduction_batch_size,
            tmp_dir,
            stats,
        )
        if stats is not None:
            stats.reduce_seconds = time.monotonic() - reduce_start
    return final


def _materialize_site_union(
    window_shards: _WindowShards,
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
        for site, rsid in _iter_pass1_shard(spec.path):
            sites.add(site)
            if rsid:
                ranked.append((spec.rank, site, rsid))
    ranked.sort()
    for _rank, site, rsid in ranked:
        rsid_by_site.setdefault(site, rsid)
    return tuples_by_assembly, rsid_by_site


def _materialize_manifest_lookup(
    window_shards: _WindowShards,
    *,
    chain_file: str | Path | None,
    liftover_failure_threshold: float,
) -> tuple[dict[tuple[str, int, str, str], str], dict[str, str]]:
    """The Dense and Hybrid consumer: final window shards -> the hg38 lookup.

    This is the assembly seam (issue #193). The union pass hands over final
    per-window shards; this consumer materialises the whole site union and lifts
    it, which is what Dense and Hybrid need. A later streaming artifact writer
    (issues #196/#197) can consume the same shards per window without building
    either dict.
    """
    tuples_by_assembly, rsid_by_site = _materialize_site_union(window_shards)
    return _resolve_manifest_variants_to_alids(
        tuples_by_assembly,
        rsid_by_site,
        chain_file=chain_file,
        liftover_failure_threshold=liftover_failure_threshold,
    )


@contextmanager
def _reduce_manifest_windows(
    manifest_rows: list[_ManifestRow],
    *,
    n_workers: int,
    window_size_mb: float,
    reduction_batch_size: int,
    map_spill_records: int = DEFAULT_MAP_SPILL_RECORDS,
    stats: _Pass1Stats | None = None,
) -> Iterator[_WindowShards]:
    """Map every manifest source into window shards and tree-reduce each window.

    The union pass core (issues #5, #188): each source is read once and routed
    into sorted per-window shards, and each window's shards are tree-reduced in
    parallel batches. A worker spills its buffers whenever ``map_spill_records``
    variants are buffered, so peak memory tracks that threshold rather than the
    row count of its manifest slice (issue #194). What it exposes is the shards
    -- not an in-memory union -- so a consumer can choose what to build from
    them (issue #193).
    """
    size_bp = window_size_bp(window_size_mb)
    workers = min(n_workers, len(manifest_rows)) if manifest_rows else 0
    with tempfile.TemporaryDirectory(prefix=".pass1windows.") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        if workers <= 1:
            yield _WindowShards(
                shards=_map_manifest_windows_serial(
                    manifest_rows, size_bp, reduction_batch_size, map_spill_records, tmp_dir, stats
                )
            )
            return
        log.info(
            "Pass 1: collecting source variants from %d files (parallel, %d workers, "
            "%g Mb windows, batch %d)",
            len(manifest_rows),
            workers,
            window_size_mb,
            reduction_batch_size,
        )
        start = time.monotonic()
        final = _map_reduce_windows(
            manifest_rows,
            workers,
            size_bp,
            reduction_batch_size,
            map_spill_records,
            tmp_dir,
            stats,
        )
        log.info(
            "Pass 1 merge: %d window(s) in %s",
            len(final),
            _fmt_duration(time.monotonic() - start),
        )
        yield _WindowShards(shards=final)


def _lift_manifest_variants(
    manifest_rows: list[_ManifestRow],
    *,
    chain_file: str | Path | None,
    liftover_failure_threshold: float,
    n_workers: int = 1,
    window_size_mb: float = DEFAULT_WINDOW_SIZE_MB,
    reduction_batch_size: int = DEFAULT_REDUCTION_BATCH_SIZE,
    map_spill_records: int = DEFAULT_MAP_SPILL_RECORDS,
    stats: _Pass1Stats | None = None,
) -> tuple[dict[tuple[str, int, str, str], str], dict[str, str]]:
    """Resolve every manifest row's union of source variants to hg38 ALIDs
    (issue #85; the dense and hybrid builders' shared Pass 1).

    Each row declares its own ``source_assembly`` (`_read_manifest`). Rows
    already on hg38 -- a harmonised GWAS-SSF source, say -- map straight to
    their ALID with no liftover: running an already-hg38 coordinate through
    the hg19->hg38 chain a second time silently shifts it to the wrong
    position (the issue #85 bug), and pyliftover has no way to detect that
    from the coordinate alone. Rows declaring hg19 (the only other build this
    package knows) go through one shared ``LiftOver`` object for that group,
    same as when GWAS-VCF was the only source this builder ever saw.
    ``liftover_failure_threshold`` therefore applies only to the hg19 group's
    own failure rate, not diluted by (or inflated against) hg38 rows that
    were never at risk of failing. ``n_workers`` controls the union pass's
    parallelism (issue 5); the serial read is kept for ``n_workers <= 1``.

    Returns one merged ``{(chrom, pos, ref, alt): hg38_alid}`` lookup -- the
    shape the fork-safe Pass 2 key index (`_build_variant_key_index` /
    `_build_routing_index`) is already built from, so nothing downstream of
    this function changes -- plus ``{hg38_alid: rsid}`` for the rows whose
    source named one (issue #109). The rsid map rides along on this pass
    rather than a second one: Pass 1 is already the serial read of every
    source file, and re-reading them just to recover identifiers the first
    read saw would double the most expensive part of a genome-scale build.
    Where two sources name one variant differently, the first wins; where a
    source names none, the ALID is simply absent from the map and its row's
    rsid column is written blank. A raw tuple present in *both* groups is not a
    same-locus dedup the way a tuple shared by two same-assembly files is:
    the hg38 group's tuple is a literal coordinate, the hg19 group's
    identical-looking tuple is a *pre-lift* coordinate bound for a different
    hg38 position, so the two groups agreeing on a raw tuple means two
    physically different loci coincidentally share one string, not one real
    variant reported twice. Binding both to a single stored row would
    misattribute one row's association to the other's variant -- exactly the
    kind of silent corruption this fix exists to remove -- so any such tuple
    is dropped from both groups (never guessed) before returning.

    The union pass core (`_reduce_manifest_windows`) exposes final per-window
    shards; this function hands them to the `_materialize_manifest_lookup`
    consumer, which is the Dense and Hybrid assembly seam (issue #193).
    """
    return _consume_manifest_shards(
        manifest_rows, n_workers=n_workers, window_size_mb=window_size_mb,
        reduction_batch_size=reduction_batch_size, map_spill_records=map_spill_records,
        stats=stats,
        consume=partial(
            _materialize_manifest_lookup,
            chain_file=chain_file,
            liftover_failure_threshold=liftover_failure_threshold,
        ),
    )


def _resolve_manifest_variants_to_alids(
    tuples_by_assembly: dict[str, set[tuple[str, int, str, str]]],
    rsid_by_site: dict[tuple[str, int, str, str], str],
    *,
    chain_file: str | Path | None,
    liftover_failure_threshold: float,
) -> tuple[dict[tuple[str, int, str, str], str], dict[str, str]]:
    """Lift the union to hg38 ALIDs and re-key the rsids onto them (issues #85, #109)."""
    finalise_start = time.monotonic()
    passthrough_lookup: dict[tuple[str, int, str, str], str] = {}
    passthrough = tuples_by_assembly.pop("hg38", set())
    if passthrough:
        log.info("%d variants already GRCh38 -- no liftover needed", len(passthrough))
        for chrom, pos, ref, alt in passthrough:
            a1, a2 = sorted((ref, alt))
            passthrough_lookup[(chrom, pos, ref, alt)] = f"{chrom}:{pos}:{a1}:{a2}"
    lifted_lookup: dict[tuple[str, int, str, str], str] = {}
    hg19_tuples = tuples_by_assembly.pop("hg19", set())
    if hg19_tuples:
        log.info("Running liftover hg19 → hg38 (%d variants)", len(hg19_tuples))
        lifted_lookup = build_liftover_lookup(
            hg19_tuples,
            from_build="hg19",
            to_build="hg38",
            failure_threshold=liftover_failure_threshold,
            chain_file=chain_file,
        )
        log.info("Liftover complete: %d variants mapped", len(lifted_lookup))
    assert not tuples_by_assembly, f"unhandled source_assembly values: {sorted(tuples_by_assembly)}"
    ambiguous = passthrough_lookup.keys() & lifted_lookup.keys()
    if ambiguous:
        log.warning(
            "%d raw variant tuple(s) declared both hg38 and hg19 in this manifest "
            "(same chrom/pos/ref/alt string, two different builds -> two different "
            "physical loci) -- dropped from both rather than guessed which one owns "
            "the stored row",
            len(ambiguous),
        )
        for key in ambiguous:
            del passthrough_lookup[key]
            del lifted_lookup[key]
    source_lookup = {**passthrough_lookup, **lifted_lookup}
    rsid_by_alid = _rekey_rsids_to_alids(source_lookup, rsid_by_site)
    log.info(
        "Pass 1 liftover/finalisation: %d source variants → %d hg38 ALIDs in %s",
        len(source_lookup),
        len(set(source_lookup.values())),
        _fmt_duration(time.monotonic() - finalise_start),
    )
    return source_lookup, rsid_by_alid


def _rekey_rsids_to_alids(
    source_lookup: Mapping[tuple[str, int, str, str], str],
    rsid_by_site: Mapping[tuple[str, int, str, str], str],
) -> dict[str, str]:
    """The rsid each ALID carries: the first non-empty in ``(rank, site)`` order.

    ``rsid_by_site`` is produced in that order (issue #192), so iterating it
    directly pins the winner for an ALID several source keys resolve to.
    Iterating ``source_lookup`` would fall back to the union set's per-process
    string hash order and let the answer move between runs (issue #192).
    """
    by_alid: dict[str, str] = {}
    for site, rsid in rsid_by_site.items():
        alid = source_lookup.get(site)
        if alid and rsid:
            by_alid.setdefault(alid, rsid)
    return by_alid


def build_dense_from_vcf_manifest(
    manifest_path: str | Path, output_path: str | Path, *,
    chain_file: str | Path | None = None, store_id: str, release_id: str,
    liftover_failure_threshold: float = 0.01,
    chunk_shape: tuple[int, int] = DEFAULT_CHUNK_SHAPE,
    dtype: str = DEFAULT_DTYPE, overwrite: bool = False, n_workers: int = 1,
    eaf_reference: str | Path | None = None,
    eaf_reference_ancestry: str | None = None, allow_unverified_eaf: bool = False,
    source_reader_capability: str | None = None, source_assembly: str | None = None,
    variant_reference: str | Path | None = None,
) -> DenseBuildResult:
    """Build a Dense Observed-Only Store from a manifest of GWAS-VCF files.

    A thin orchestrator over the private phases below (issue #130): the build
    is staged atomically at ``output_path``, its two streaming passes never
    materialise the full association matrix (issue 043), and every phase runs
    before the store is committed. Keyword semantics live with the phase that
    consumes each -- the manifest columns and per-release defaults (#174) with
    `_read_manifest`; ``chain_file`` and ``liftover_failure_threshold`` with the
    hg19→hg38 lift (issue #85); ``eaf_reference``/``eaf_reference_ancestry``/
    ``allow_unverified_eaf`` with EAF orientation verification (issue #115, ADR 0037 §6).
    ``variant_reference`` (issue #185) supplies a precomputed axis
    (``*.variant-ref.tsv.gz``, an ALID list, or a store ``variants.tsv.gz``);
    Pass 1 and liftover are then bypassed.
    """
    manifest_rows = _read_manifest(
        manifest_path,
        default_source_reader_capability=source_reader_capability,
        default_source_assembly=source_assembly,
    )
    if not manifest_rows:
        raise ValueError(f"manifest {manifest_path} contains no rows")

    out = Path(output_path)
    with OpenGWASDBStore.staging(out, overwrite=overwrite) as staged:
        prepared = _prepare_axis(
            staged, out, manifest_rows, chain_file, liftover_failure_threshold, chunk_shape,
            n_workers, variant_reference,
        )
        # Phases 5-7 run inside one spill-dir lifetime: every spill is removed
        # even when a phase fails, keeping the staged release atomic.
        eaf_report, encoded = _spill_verify_and_encode(
            staged, out, manifest_rows, prepared, n_workers, chunk_shape, dtype,
            eaf_reference, eaf_reference_ancestry, allow_unverified_eaf,
        )
        # Phase 8: top-hit indexes, manifest and analyses.tsv metadata.
        _finalize_store(
            staged, prepared, encoded, eaf_report, store_id, release_id, chain_file,
            chunk_shape, dtype, allow_unverified_eaf, n_workers,
        )
        log.info(
            "Build complete: %d variants × %d analyses",
            len(prepared.axis.alids), len(prepared.axis.analyses),
        )

    return DenseBuildResult(
        output_path=out,
        n_variants=len(prepared.axis.alids),
        n_analyses=len(prepared.axis.analyses),
    )


@dataclass(frozen=True)
class _AxisMetadata:
    """The store's axis: sorted hg38 ALIDs, the row maps, and Analysis records.

    ``analyses`` are the manifest's Analytical Metadata (issue #17, #86) --
    derived from each row, never from a VCF header.
    """

    alids: list[str]
    analysis_index: dict[str, int]
    analyses: list[Analysis]


@dataclass(frozen=True)
class _HitCandidates:
    """The cells the band-write's z pass harvested above the loosest threshold.

    Harvested from the *stored* values so the top-hit index matches exactly
    what a query reads back from the ``z`` array (issue 046)."""

    rows: np.ndarray
    cols: np.ndarray
    z: np.ndarray
    se: np.ndarray


@dataclass(frozen=True)
class _EncodedBands:
    """Phase-7 outcome the index and metadata phases consume."""

    encoding: StoreEncoding
    hits: _HitCandidates
    column_has_eaf: np.ndarray


@dataclass(frozen=True)
class _PreparedBuild:
    """What the preparation phase hands the spill phases: the store's axis and
    the fork-safe Pass 2 lookup arrays resolved against it.

    The parent-side ``source_lookup``/``variant_index`` dicts are dropped when
    the preparation phase returns, before any worker forks (issue 043).
    """

    axis: _AxisMetadata
    keys_sorted: np.ndarray
    rows_sorted: np.ndarray
    #: The ``--variant-reference`` a single-pass build was given, or None for
    #: the inline two-pass build. Recorded in the store manifest's provenance
    #: so the axis's origin is auditable (issue #185).
    variant_reference: str | None = None


def _axis_metadata(
    source_lookup: Mapping[tuple[str, int, str, str], str],
    manifest_rows: Sequence[_ManifestRow],
    alids: Sequence[str] | None = None,
) -> tuple[_AxisMetadata, dict[str, int]]:
    """Resolve the store's axis metadata.

    Sorting hg38 ALIDs by (chromosome, position, a1, a2) gives the variant
    axis a stable, position-ordered row index; the manifest's own row order
    fixes the analysis column order. ``alids`` overrides the source lookup as
    the authority when a precomputed variant reference supplied the axis
    (issue #185). The variant index is returned alongside the axis so the
    caller can free it (it is the one ~n_variants-sized dict) once the Pass 2
    lookup arrays are built.
    """
    hg38_alids = _sorted_alids(alids if alids is not None else source_lookup.values())
    variant_index: dict[str, int] = {alid: i for i, alid in enumerate(hg38_alids)}
    analysis_index: dict[str, int] = {row.trait_id: i for i, row in enumerate(manifest_rows)}
    analyses: list[Analysis] = [_manifest_row_to_analysis(row) for row in manifest_rows]
    return (
        _AxisMetadata(
            alids=hg38_alids,
            analysis_index=analysis_index,
            analyses=analyses,
        ),
        variant_index,
    )


def _source_alids_by_alid(
    source_lookup: Mapping[tuple[str, int, str, str], str],
    hg38_alids: Sequence[str],
) -> list[str | None]:
    """The source-build canonical ALID each stored row was resolved from.

    Provenance (the variant axis's ``source_alid`` column): the hg19 tuple's
    own canonical ALID, or the passed-through hg38 tuple's, so a reader sees
    which build coordinate a row's associations came from. A single hg38 ALID
    can be the target of several source variants (a liftover collapse); when
    they disagree on the origin the row is ambiguous, so it is left blank --
    never guessed (a guess would silently misattribute provenance).
    """
    hg38_to_source: dict[str, str | None] = {}
    for (chrom, pos, ref, alt), hg38_alid in source_lookup.items():
        a1, a2 = sorted((ref, alt))
        origin = f"{chrom}:{pos}:{a1}:{a2}"
        if hg38_alid in hg38_to_source:
            if hg38_to_source[hg38_alid] != origin:
                hg38_to_source[hg38_alid] = None  # collision → ambiguous
        else:
            hg38_to_source[hg38_alid] = origin
    return [hg38_to_source.get(alid) for alid in hg38_alids]


def _write_axis_and_index(
    staged: StagedRelease,
    source_lookup: Mapping[tuple[str, int, str, str], str],
    axis: _AxisMetadata,
    rsid_by_alid: Mapping[str, str],
    chunk_shape: tuple[int, int],
) -> None:
    """Write the SQLite index and the tabix variant axis.

    The variant axis carries one canonical variant per stored row with its
    rsid (first named wins) and the source-build provenance each row resolved
    from (``_source_alids_by_alid`` -- collisions stay blank, never guessed).
    """
    _write_index(staged, axis.alids, axis.analyses, chunk_shape)
    canonical_variants = [
        CanonicalVariant(
            chromosome=chrom,
            position=int(pos_str),
            effect_allele=a1,
            other_allele=a2,
        )
        for alid in axis.alids
        for chrom, pos_str, a1, a2 in [alid.split(":")]
    ]
    source_alids = _source_alids_by_alid(source_lookup, axis.alids)
    write_variant_axis(staged.path, canonical_variants, rsid_by_alid, source_alids)


def _build_pass2_lookup(
    source_lookup: dict[tuple[str, int, str, str], str],
    variant_index: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Compose the Pass 2 dicts into the sorted numpy lookup workers search.

    Fork-safe by construction: the returned arrays are a contiguous C buffer
    and an int32 row array, so a worker reading them via searchsorted never
    touches per-element refcounts (issue 043 item 2). The caller frees the
    dicts before the pool is created.
    """
    keys_sorted, rows_sorted = _build_variant_key_index(source_lookup, variant_index)
    max_key_len = max((len(key) for key in keys_sorted), default=0)
    log.info(
        "Pass 2 lookup: %d variant keys, max key length %d bytes",
        len(keys_sorted),
        max_key_len,
    )
    return keys_sorted, rows_sorted


def _resolve_axis_source(
    manifest_rows: list[_ManifestRow],
    chain_file: str | Path | None,
    liftover_failure_threshold: float,
    n_workers: int,
    variant_reference: str | Path | None,
) -> tuple[list[str] | None, dict[tuple[str, int, str, str], str], dict[str, str]]:
    """The hg38 axis, its source-coordinate lookup and its rsids.

    With a precomputed ``variant_reference`` the reference is the authority and
    Pass 1 never runs (single-pass build, issue #185); otherwise the manifest's
    sources are read once and lifted (the inline two-pass build). ``alids`` is
    None in the two-pass case so the axis is derived from the lifted union.
    """
    if variant_reference is not None:
        reference = _load_variant_reference(variant_reference, manifest_rows)
        return reference.alids, reference.source_lookup, reference.rsid_by_alid
    source_lookup, rsid_by_alid = _lift_manifest_variants(
        manifest_rows,
        chain_file=chain_file,
        liftover_failure_threshold=liftover_failure_threshold,
        n_workers=n_workers,
    )
    return None, source_lookup, rsid_by_alid


def _load_variant_reference(
    variant_reference: str | Path, manifest_rows: Sequence[_ManifestRow]
) -> VariantReference:
    """Read the reference and note when it cannot resolve the manifest's assembly.

    An identity reference (a plain ALID list or a store ``variants.tsv.gz``)
    carries no source-key mapping, so it resolves only rows already on hg38.
    The reference is the axis authority and no liftover runs in Pass 2, so a
    manifest declaring hg19 against one keeps only coordinates that happen to
    coincide; that is the documented off-reference behaviour, but it is worth
    saying so rather than letting the shortfall pass unremarked.
    """
    reference = read_variant_reference(variant_reference)
    log.info(
        "Single-pass build: variant axis loaded from %s (%d variants); "
        "Pass 1 variant union and liftover bypassed",
        variant_reference,
        len(reference.alids),
    )
    if not reference.explicit_source_keys:
        non_hg38 = sorted(
            {row.source_assembly for row in manifest_rows if row.source_assembly != "hg38"}
        )
        if non_hg38:
            log.warning(
                "Variant reference %s carries no source-key mapping and resolves only hg38 "
                "coordinates, but the manifest declares %s; variants whose source coordinate "
                "differs from the reference will be dropped",
                variant_reference,
                ", ".join(non_hg38),
            )
    return reference


def _prepare_axis(
    staged: StagedRelease,
    out: Path,
    manifest_rows: list[_ManifestRow],
    chain_file: str | Path | None,
    liftover_failure_threshold: float,
    chunk_shape: tuple[int, int],
    n_workers: int,
    variant_reference: str | Path | None,
) -> _PreparedBuild:
    """Phases 1-4: axis source, axis metadata, index + axis, Pass 2 lookup.

    hg19-declared rows are lifted to GRCh38 once (issue #85), or a precomputed
    ``variant_reference`` supplies the axis directly and Pass 1 is skipped
    (single-pass build, issue #185). Either way the axis metadata and
    provenance-carrying variant axis are written, and the Pass 2 lookup arrays
    are composed -- the parent ``source_lookup``/``variant_index`` dicts then
    drop out of scope before any worker forks (issue 043).
    """
    alids, source_lookup, rsid_by_alid = _resolve_axis_source(
        manifest_rows, chain_file, liftover_failure_threshold, n_workers, variant_reference,
    )
    axis, variant_index = _axis_metadata(source_lookup, manifest_rows, alids=alids)
    _write_axis_and_index(staged, source_lookup, axis, rsid_by_alid, chunk_shape)
    keys_sorted, rows_sorted = _build_pass2_lookup(source_lookup, variant_index)
    return _PreparedBuild(
        axis=axis,
        keys_sorted=keys_sorted,
        rows_sorted=rows_sorted,
        variant_reference=str(variant_reference) if variant_reference is not None else None,
    )


def _spill_columns_serial(
    manifest_rows: Sequence[_ManifestRow],
    analysis_index: Mapping[str, int],
    spill_dir: Path,
    keys_sorted: np.ndarray,
    rows_sorted: np.ndarray,
    axis: _AxisMetadata,
    pass2_start: float,
) -> None:
    """Pass 2 for ``n_workers <= 1``: resolve and spill each column in this
    process, one Analysis at a time, so peak memory is one resolved column."""
    log.info(
        "Pass 2: resolving %d analyses × %d variants (n_workers=1)",
        len(axis.analyses),
        len(axis.alids),
    )
    for i, row in enumerate(manifest_rows):
        col_idx = analysis_index[row.trait_id]
        rows, z, se, eaf = _resolve_column(
            row.file_path,
            keys_sorted,
            rows_sorted,
            row.se_divisor,
            capability=row.source_reader_capability,
            stored_effect_scale=row.stored_effect_scale,
        )
        _spill_column(spill_dir, col_idx, rows, z, se, eaf)
        _log_progress(
            "Pass 2", i + 1, len(axis.analyses), pass2_start, f"last: {row.trait_id}", every=25
        )


def _spill_columns_parallel(
    manifest_rows: Sequence[_ManifestRow],
    analysis_index: Mapping[str, int],
    spill_dir: Path,
    keys_sorted: np.ndarray,
    rows_sorted: np.ndarray,
    axis: _AxisMetadata,
    n_workers: int,
    pass2_start: float,
) -> None:
    """Pass 2 for ``n_workers > 1``: resolve each column in a fork pool.

    Workers inherit the numpy lookup arrays at fork (see the module note on
    ``_pass2_keys_sorted``) and write only their own ``.npz``, so nothing large
    crosses the pipe; the parent waits on completion. The globals are reset
    whether or not every task succeeded.
    """
    log.info(
        "Pass 2: resolving %d analyses × %d variants (n_workers=%d)",
        len(axis.analyses),
        len(axis.alids),
        n_workers,
    )
    global _pass2_keys_sorted, _pass2_rows_sorted, _pass2_spill_dir
    _pass2_keys_sorted = keys_sorted
    _pass2_rows_sorted = rows_sorted
    _pass2_spill_dir = spill_dir
    try:
        with _fork_pool(n_workers) as pool:
            id_by_col = {analysis_index[row.trait_id]: row.trait_id for row in manifest_rows}
            tasks = _pass2_worker_tasks(manifest_rows, analysis_index)
            futures = [pool.submit(_pass2_worker, t) for t in tasks]
            for i, fut in enumerate(as_completed(futures)):
                col_idx = fut.result()
                _log_progress(
                    "Pass 2",
                    i + 1,
                    len(axis.analyses),
                    pass2_start,
                    f"last: {id_by_col[col_idx]}",
                    every=25,
                )
    finally:
        _pass2_keys_sorted = None
        _pass2_rows_sorted = None
        _pass2_spill_dir = None


def _survey_and_verify_eaf(
    spill_dir: Path,
    manifest_rows: Sequence[_ManifestRow],
    axis: _AxisMetadata,
    *,
    eaf_reference: str | Path | None,
    eaf_reference_ancestry: str | None,
    allow_unverified_eaf: bool,
    n_workers: int,
) -> tuple[EafSpillSurvey, EafOrientationReport]:
    """Sample each Analysis's stored frequencies and verify their orientation.

    The check (issue #115, ADR 0037 §6) runs off the Pass 2 spills -- the exact
    values about to be stored -- and before any statistic array is written, so
    a build that would store a frequency column against the wrong allele fails
    here rather than after an hour of band-writing. The survey also measures
    the frequency spread the encoding plan reads, so both answer in one pass.
    ``allow_unverified_eaf`` records Analyses the supplied reference could not
    verify in the store's provenance instead of rejecting them.
    """
    id_by_col = {axis.analysis_index[row.trait_id]: row.trait_id for row in manifest_rows}
    eaf_survey = survey_eaf_spills(
        spill_dir, id_by_col, axis.alids, site_hashes(axis.alids), n_workers=n_workers
    )
    eaf_report = verify_eaf_orientation(
        eaf_survey.observations,
        eaf_reference=eaf_reference,
        eaf_reference_ancestry=eaf_reference_ancestry,
        allow_unverified=allow_unverified_eaf,
    )
    return eaf_survey, eaf_report


def _write_encoded_bands(
    staged: StagedRelease,
    spill_dir: Path,
    eaf_survey: EafSpillSurvey,
    axis: _AxisMetadata,
    chunk_shape: tuple[int, int],
    dtype: str,
    pass2_start: float,
    n_workers: int,
) -> _EncodedBands:
    """Create the statistic arrays and fill them from the spills.

    One encoding plan per build is decided here -- after Pass 2 and the EAF
    survey, because the ``eaf`` rules read the frequencies the sources
    actually carried -- and recorded in manifest.json (ADR 0037, issue #119).
    The z/se planes are then filled in chunk-column bands so peak memory is
    one band (issue 043), the eaf plane is written under the plan, and
    ``optimise_dense_se`` rewrites ``se`` under the plan's codec. The
    top-hit candidates are harvested in the same pass (issue 046).
    """
    n_variants, n_analyses = len(axis.alids), len(axis.analyses)
    encoding = StoreEncoding.decide(
        EncodingMeasurements(
            n_analyses=n_analyses,
            eaf=eaf_survey.measurements(
                n_cells=n_variants * n_analyses, n_variants=n_variants
            ),
        )
    )
    log.info("Encoding plan: %s", encoding.to_manifest())
    effective_chunks = _create_dense_zarr(
        staged, n_variants, n_analyses, chunk_shape, dtype, encoding
    )
    rows, cols, z, se, column_has_eaf = _write_dense_bands(
        staged,
        spill_dir,
        n_variants,
        n_analyses,
        effective_chunks,
        dtype,
        pass2_start,
        encoding,
    )
    encoding = optimise_dense_se(staged.arrays(mode="a"), encoding, n_workers=n_workers)
    return _EncodedBands(
        encoding=encoding,
        hits=_HitCandidates(rows=rows, cols=cols, z=z, se=se),
        column_has_eaf=column_has_eaf,
    )


def _spill_verify_and_encode(
    staged: StagedRelease,
    out: Path,
    manifest_rows: Sequence[_ManifestRow],
    prepared: _PreparedBuild,
    n_workers: int,
    chunk_shape: tuple[int, int],
    dtype: str,
    eaf_reference: str | Path | None,
    eaf_reference_ancestry: str | None,
    allow_unverified_eaf: bool,
) -> tuple[EafOrientationReport, _EncodedBands]:
    """Phases 5-7: spill each column, verify EAF, and write the encoded planes.

    Runs inside the spill dir's lifecycle: every spill is removed when this
    returns or raises, so a phase failure leaves nothing behind and the staged
    release stays atomic. ``n_workers`` > 1 resolves through the fork pool;
    EAF orientation is verified against ``eaf_reference`` before any array is
    written (issue #115, ADR 0037 §6).
    """
    spill_dir = Path(
        tempfile.mkdtemp(prefix=f".{out.name}.pass2spill.", dir=staged.path.parent)
    )
    try:
        pass2_start = time.monotonic()
        if n_workers <= 1:
            _spill_columns_serial(
                manifest_rows, prepared.axis.analysis_index, spill_dir,
                prepared.keys_sorted, prepared.rows_sorted, prepared.axis, pass2_start,
            )
        else:
            _spill_columns_parallel(
                manifest_rows, prepared.axis.analysis_index, spill_dir,
                prepared.keys_sorted, prepared.rows_sorted,
                prepared.axis, n_workers, pass2_start,
            )
        eaf_survey, eaf_report = _survey_and_verify_eaf(
            spill_dir,
            manifest_rows,
            prepared.axis,
            eaf_reference=eaf_reference,
            eaf_reference_ancestry=eaf_reference_ancestry,
            allow_unverified_eaf=allow_unverified_eaf,
            n_workers=n_workers,
        )
        encoded = _write_encoded_bands(
            staged,
            spill_dir,
            eaf_survey,
            prepared.axis,
            chunk_shape,
            dtype,
            pass2_start,
            n_workers,
        )
    finally:
        shutil.rmtree(spill_dir, ignore_errors=True)
    return eaf_report, encoded


def _finalize_store(
    staged: StagedRelease,
    prepared: _PreparedBuild,
    encoded: _EncodedBands,
    eaf_report: EafOrientationReport,
    store_id: str,
    release_id: str,
    chain_file: str | Path | None,
    chunk_shape: tuple[int, int],
    dtype: str,
    allow_unverified_eaf: bool,
    n_workers: int,
) -> None:
    """Phase 8: write the store's final metadata.

    The manifest names the encoding and the EAF-orientation evidence; the
    top-hit indexes are written from the band-write's stored-value harvest
    (issue 046); analyses.tsv is written last, with each Analysis's
    ``eaf_scope`` stamped from what was actually stored (ADR 0036) and the
    orientation evidence applied (issue #115).
    """
    axis = prepared.axis
    _write_manifest(
        staged,
        store_id,
        release_id,
        len(axis.alids),
        len(axis.analyses),
        chain_file,
        chunk_shape,
        dtype,
        encoding=encoded.encoding,
        eaf_orientation=eaf_report.provenance(allow_unverified=allow_unverified_eaf),
        variant_reference=prepared.variant_reference,
    )
    write_top_hit_indexes_for_store(
        staged.path, encoded.hits.rows, encoded.hits.cols, encoded.hits.z, encoded.hits.se,
        encoded.encoding, n_workers=n_workers,
    )
    analyses = apply_orientation_evidence(
        _apply_eaf_scope(axis.analyses, encoded.column_has_eaf), eaf_report
    )
    write_analyses_tsv(staged.path, add_hit_counts(staged.path, analyses))


def _fmt_duration(seconds: float) -> str:
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _parse_source_assembly(
    row: Mapping[str, str], fallback: str, trait_id: str, manifest_path: str | Path
) -> str:
    """Resolve and validate source_assembly for one manifest row."""
    raw = row.get("source_assembly") or fallback
    try:
        return normalise_build(raw)
    except ValueError as exc:
        raise ValueError(
            f"manifest {manifest_path}: analysis {trait_id!r} has invalid "
            f"source_assembly {raw!r}"
        ) from exc


def _parse_reader_capability(
    row: Mapping[str, str], fallback: str, trait_id: str, manifest_path: str | Path
) -> str:
    """Resolve and validate source_reader_capability for one manifest row."""
    capability = row.get("source_reader_capability") or fallback
    if capability not in known_capabilities():
        known = ", ".join(known_capabilities()) or "(none registered)"
        raise ValueError(
            f"manifest {manifest_path}: analysis {trait_id!r} has unknown "
            f"source_reader_capability {capability!r}; known: {known}"
        )
    return capability


def _parse_original_sd(
    row: Mapping[str, str], trait_id: str, manifest_path: str | Path
) -> float:
    """Validate original_sd_method and original_sd, returning se_divisor."""
    sd_method_raw = row["original_sd_method"]
    try:
        sd_method = OriginalSdMethod(sd_method_raw)
    except ValueError as exc:
        raise ValueError(
            f"manifest {manifest_path}: analysis {trait_id!r} has invalid "
            f"original_sd_method {sd_method_raw!r}"
        ) from exc
    if sd_method is OriginalSdMethod.UNAVAILABLE:
        raise ValueError(
            f"manifest {manifest_path}: analysis {trait_id!r} has "
            "original_sd_method='unavailable' -- its phenotype SD could not be "
            "established upstream, so the build cannot standardise its effects (issue #18)"
        )
    se_divisor = 1.0
    original_sd_raw = row.get("original_sd", "")
    if sd_method in _SD_RESCALE_METHODS:
        try:
            se_divisor = float(original_sd_raw)
        except ValueError as exc:
            raise ValueError(
                f"manifest {manifest_path}: analysis {trait_id!r} has "
                f"original_sd_method={sd_method.value!r} but original_sd "
                f"{original_sd_raw!r} is not a valid number"
            ) from exc
        if not se_divisor > 0:
            raise ValueError(
                f"manifest {manifest_path}: analysis {trait_id!r} has "
                f"non-positive original_sd {original_sd_raw!r}"
            )
    elif original_sd_raw:
        raise ValueError(
            f"manifest {manifest_path}: analysis {trait_id!r} has "
            f"original_sd_method={sd_method.value!r}, which carries no SD "
            f"magnitude, but original_sd={original_sd_raw!r} was supplied"
        )
    return se_divisor


def _parse_manifest_row(
    row: Mapping[str, str],
    cols: ManifestColumns,
    manifest_path: str | Path,
    fallback_assembly: str,
    fallback_capability: str,
) -> _ManifestRow:
    """Parse and validate one manifest row into a _ManifestRow."""
    trait_id = row[cols.analysis_id]
    scale = row["stored_effect_scale"]
    try:
        StoredEffectScale(scale)
    except ValueError as exc:
        raise ValueError(
            f"manifest {manifest_path}: analysis {trait_id!r} has invalid "
            f"stored_effect_scale {scale!r}"
        ) from exc
    assembly = _parse_source_assembly(row, fallback_assembly, trait_id, manifest_path)
    capability = _parse_reader_capability(row, fallback_capability, trait_id, manifest_path)
    se_divisor = _parse_original_sd(row, trait_id, manifest_path)
    return _ManifestRow(
        trait_id=trait_id,
        file_path=row[cols.source_file],
        trait_name=manifest_trait_name(row, cols.analysis_label, trait_id),
        n=manifest_n(row, cols.sample_size),
        stored_effect_scale=scale,
        se_divisor=se_divisor,
        source_reader_capability=capability,
        source_assembly=assembly,
        original_sd=row.get("original_sd", ""),
        assigned_ancestry=row.get("assigned_ancestry") or "",
        metadata=PassthroughMetadata.from_manifest_row(row),
        trait_ontology_id=row.get("trait_ontology_id") or "",
        trait_ontology_label=(
            row.get("trait_ontology_label") or row.get("trait_ontology_name") or ""
        ),
    )


def _resolve_manifest_defaults(
    default_capability: str | None, default_assembly: str | None
) -> tuple[str, str]:
    """Validate and resolve fallback capability and assembly."""
    if default_capability is not None:
        if not default_capability or default_capability not in known_capabilities():
            known = ", ".join(known_capabilities()) or "(none registered)"
            raise ValueError(
                f"unknown source reader capability {default_capability!r}; known: {known}"
            )
    if default_assembly is not None:
        default_assembly = normalise_build(default_assembly)
    return default_assembly or "hg19", default_capability or GWAS_VCF_CAPABILITY


def _read_manifest(
    manifest_path: str | Path,
    *,
    default_source_reader_capability: str | None = None,
    default_source_assembly: str | None = None,
) -> list[_ManifestRow]:
    """Read the build manifest with canonical names, aliases, and option defaults.

    Accepts canonical ``analyses.tsv`` names (``analysis_id``/``source_file``/
    ``analysis_label``/``sample_size``, ADR 0034, #170) or legacy aliases
    (``trait_id``/``file_path``/``trait_name``/``n``). Requires ``stored_effect_scale``
    (#17) and ``original_sd_method`` (#18; ``original_sd`` required for rescaling tiers).

    Optional ``source_reader_capability`` (#20, #174) and ``source_assembly`` (#85, #174)
    fall back to CLI defaults (or GWAS-VCF/hg19). Note harmonised SSF is already hg38 and
    must declare so or coordinates liftover twice (issue #85 double-liftover warning).
    """
    fallback_assembly, fallback_capability = _resolve_manifest_defaults(
        default_source_reader_capability, default_source_assembly
    )
    with open(manifest_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    require_columns(fieldnames, manifest_path, "stored_effect_scale", "original_sd_method")
    cols = resolve_manifest_columns(fieldnames, manifest_path)
    return [
        _parse_manifest_row(row, cols, manifest_path, fallback_assembly, fallback_capability)
        for row in rows
    ]


def _alid_sort_key(alid: str) -> tuple[tuple[int, str], int, str, str]:
    chrom, pos_str, a1, a2 = alid.split(":")
    return (chromosome_sort_key(chrom), int(pos_str), a1, a2)


def _sorted_alids(alids: Iterable[str]) -> list[str]:
    """Sort unique ALIDs by ``(chromosome_sort_key, position, a1, a2)``.

    Exactly equivalent to ``sorted(set(alids), key=_alid_sort_key)`` but built
    for genome-scale axis construction: the ALIDs are parsed once and each one
    is re-encoded as a sortable byte key, then ``np.argsort`` orders the byte
    keys in C rather than calling ``_alid_sort_key`` once per element (~21M
    Python calls on a dense pilot build).

    The sort key is ``fixed-width rank \\x00 chrom \\x00 zero-padded-position
    \\x00 a1 \\x00 a2``. ``\\x00`` is smaller than every character that can
    appear in a chromosome, position or allele, so it terminates the variable-
    length fields and makes lexicographic byte order equal to the Python tuple
    order ``_alid_sort_key`` produces.
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


def _write_index(
    staged: StagedRelease,
    hg38_alids: list[str],
    analyses: list[Analysis],
    chunk_shape: tuple[int, int],
) -> None:
    with staged.index_connection() as connection:
        initialise_schema(connection)
        set_metadata(connection, "schema_version", 1)
        set_metadata(connection, "n_variants", len(hg38_alids))
        set_metadata(connection, "n_analyses", len(analyses))
        set_metadata(connection, "dense", dense_index_metadata(chunk_shape))
        connection.commit()


def _apply_eaf_scope(analyses: list[Analysis], column_has_eaf: np.ndarray) -> list[Analysis]:
    """Stamp each Analysis's `eaf_scope` from what the build actually stored.

    Derived, never copied from the manifest (ADR 0036): only the build knows
    whether the source file turned out to carry a usable frequency, so a
    manifest claiming EAF for an Analysis whose file has none must not be able
    to make the store declare it. `column_has_eaf` is positional, matching
    `analysis_index`.
    """
    return [
        replace(
            analysis,
            eaf_scope=(
                EafScope.ASSOCIATION.value if bool(column_has_eaf[index]) else EafScope.ABSENT.value
            ),
        )
        for index, analysis in enumerate(analyses)
    ]


def _eaf_row_band(effective_chunks: tuple[int, int], n_analyses: int) -> int:
    """Rows per band for the re-encode pass: whole chunk rows, ~50M cells."""
    chunk_rows = max(int(effective_chunks[0]), 1)
    target = max(50_000_000 // max(n_analyses, 1), 1)
    return max(chunk_rows, (target // chunk_rows) * chunk_rows)


def _write_dense_eaf(
    staged: StagedRelease,
    spill_dir: Path,
    n_variants: int,
    n_analyses: int,
    effective_chunks: tuple[int, int],
    band_cols: int,
    codec: StoreCodec,
    pass2_start: float,
) -> None:
    """Write the `eaf` plane the store's plan declares.

    A `float32` plane is written straight from the spills, one chunk-column
    band at a time, exactly as ADR 0036 shipped it.

    An `int8_residual` plane cannot be, and this is the one place in the build
    where that costs a pass. The baseline is per *variant*, so encoding a cell
    needs every Analysis's frequency at that variant; the spills are per
    *Analysis*. So the frequencies are staged as `float32` in column bands, and
    then read back in row bands -- where a whole variant is resident -- to
    compute each baseline and encode against it. The staging array is deleted
    afterwards, and peak disk is 5 bytes per cell against the 4 the `float32`
    plane occupied on its own.
    """
    residual = codec.encoding.eaf.is_residual
    staging = _EAF_STAGING if residual else "eaf"
    _create_eaf_array(staged, n_variants, n_analyses, effective_chunks, name=staging)
    root = staged.arrays(mode="a")
    eaf_zarr = root[staging]
    eaf_band = np.empty((n_variants, band_cols), dtype="float32")
    for c0 in range(0, n_analyses, band_cols):
        c1 = min(c0 + band_cols, n_analyses)
        w = c1 - c0
        eaf_band[:, :w] = np.nan
        for c in range(c0, c1):
            local = c - c0
            with np.load(spill_dir / f"{c}.npz") as data:
                eaf_band[data["rows"], local] = data["eaf"]
        eaf_zarr[:, c0:c1] = eaf_band[:, :w]
        _log_progress(
            "Band-write eaf", c1, n_analyses, pass2_start, f"cols {c0}:{c1}", every=band_cols
        )
    del eaf_band
    if not residual:
        return
    _encode_residual_dense_eaf(
        staged, n_variants, n_analyses, effective_chunks, codec, pass2_start
    )


def _encode_residual_dense_eaf(
    staged: StagedRelease,
    n_variants: int,
    n_analyses: int,
    effective_chunks: tuple[int, int],
    codec: StoreCodec,
    pass2_start: float,
) -> None:
    """Residual re-encode: transpose the staged float32 grid, then delete it.

    The staging array is only deleted once the baseline and exception tables it
    produced are both written -- a failed encode leaves the staging plane in
    place so a resume can tell the pass never finished. Full-float staging and
    exact exception values are preserved; nothing is encoded by column or
    clipped.
    """
    root = staged.arrays(mode="a")
    encoded = _create_eaf_array(
        staged,
        n_variants,
        n_analyses,
        effective_chunks,
        dtype=codec.eaf_dtype,
        fill_value=codec.eaf_fill_value,
    )
    baseline = np.full(n_variants, np.nan, dtype=np.float32)
    exceptions = EafExceptionBuilder()
    band_rows = _eaf_row_band(effective_chunks, n_analyses)
    for r0 in range(0, n_variants, band_rows):
        r1 = min(r0 + band_rows, n_variants)
        block = np.asarray(root[_EAF_STAGING][r0:r1], dtype=np.float32)
        rows_baseline = eaf_baseline_from_grid(block)
        baseline[r0:r1] = rows_baseline
        encoded[r0:r1] = codec.encode_eaf(
            block,
            baseline=np.repeat(rows_baseline[:, None], n_analyses, axis=1),
            positions=positions_row_band(r0, n_analyses),
            exceptions=exceptions,
        )
        _log_progress("Encode eaf", r1, n_variants, pass2_start, f"rows {r0}:{r1}", every=band_rows)
    write_eaf_baseline(root, baseline, compressor=_DENSE_COMPRESSOR)
    exceptions.table().write(root)
    log.info(
        "eaf: int8 residual at +/-%.1f, %d exception cell(s)",
        codec.encoding.eaf.residual_range,
        len(exceptions),
    )
    del root[_EAF_STAGING]


#: Name of the transient `float32` plane a residual-coded build stages into.
#: The spills are per Analysis and the baseline is per variant, so the grid has
#: to be transposed through something; staging it is the honest way to say so.
_EAF_STAGING = "eaf_source"


def _create_eaf_array(
    staged: StagedRelease,
    n_variants: int,
    n_analyses: int,
    effective_chunks: tuple[int, int],
    *,
    name: str = "eaf",
    dtype: str = "float32",
    fill_value: Any = float("nan"),
) -> Any:
    """Create the missing-filled `eaf` array (ADR 0036, ADR 0037 §2).

    Created after Pass 2, not alongside `z`/`se`, because how -- and whether --
    a build stores frequencies is only known once the sources have been read.
    A build whose plan says `absent` writes no array at all, so its store is
    byte-identical in shape to one built before EAF existed.

    Never `float16`: its spacing near 1.0 is 0.00049, and the canonical A1 is
    the lexicographically smaller allele rather than the minor one, so EAF near
    1 is ordinary here -- `float16` would silently round "MAF 1e-4" to "MAF 0"
    for half the ALID space.
    """
    root = staged.arrays(mode="a")
    if name in root:
        del root[name]
    return root.create_dataset(
        name,
        shape=(n_variants, n_analyses),
        chunks=effective_chunks,
        compressor=_DENSE_COMPRESSOR,
        dtype=dtype,
        fill_value=fill_value,
    )


def _create_dense_zarr(
    staged: StagedRelease,
    n_variants: int,
    n_analyses: int,
    chunk_shape: tuple[int, int],
    dtype: str,
    encoding: StoreEncoding,
) -> tuple[int, int]:
    """Create the empty missing-filled z/se datasets and return the effective chunks.

    Chunk shape is clipped to the array dimensions (ADR-0021) so zarr's declared
    chunks match what is physically stored. The arrays are created without data;
    ``_write_dense_bands`` fills them by chunk-column band afterwards. Each plane
    is filled with **its own** missing marker (spec §15): NaN for `se`, the
    reserved sentinel for a fixed-point `z`, so an untouched cell reads as
    missing under either encoding.
    """
    compressor = _DENSE_COMPRESSOR
    codec = StoreCodec(encoding)
    effective_chunks = (min(chunk_shape[0], n_variants), min(chunk_shape[1], n_analyses))
    root = staged.arrays(mode="w")
    for name, plane_dtype, fill in (
        ("z", codec.z_dtype, codec.z_fill_value),
        # Scratch in float32, as dense.complete does: the band-writer fills
        # this before the SE encoding is decided, and an exact residual
        # exception must be the source's own value, not one already rounded
        # to the dtype the plane happened to start in (spec §6a).
        ("se", "float32", float("nan")),
    ):
        root.create_dataset(
            name,
            shape=(n_variants, n_analyses),
            chunks=effective_chunks,
            compressor=compressor,
            dtype=plane_dtype,
            fill_value=fill,
        )
    root.attrs["layout"] = "dense"
    root.attrs["completion_state"] = "observed_only"
    root.attrs["compressor"] = DEFAULT_COMPRESSOR
    root.attrs["chunk_shape"] = list(effective_chunks)
    return effective_chunks


# ── EAF orientation (issue #115, ADR 0037 §6) ────────────────────────────────
#
# Shared with the Hybrid builder, which spills the same per-column arrays. The
# check runs off the spills rather than re-reading the sources: the spilled
# `eaf` is exactly what the band-write is about to store, so what is verified
# is what a query will read back, and it costs no extra pass over the source
# files.


@dataclass(frozen=True)
class EafSpillSurvey:
    """One pass over the column spills, answering both EAF questions at once.

    The orientation check (§9.1) needs a deterministic per-Analysis sample of
    frequencies; the encoding tree (ADR 0037 §2) needs the residual spread over
    the same sample plus the build's exact cell counts. They are read together
    because the spills are large and the sample is the same one.
    """

    observations: dict[str, dict[str, float]]
    #: Cells the spills hold, whether or not they carry a frequency. For a
    #: Dense component this is not the plane's size -- the plane is the whole
    #: grid -- so the caller says which count the bytes should be reckoned on.
    n_spill_cells: int
    n_eaf_cells: int
    sample_rows: np.ndarray
    sample_values: np.ndarray

    def measurements(self, *, n_cells: int, n_variants: int) -> EafMeasurements:
        return measure_eaf_sample(
            self.sample_rows,
            self.sample_values,
            n_variants=n_variants,
            n_cells=n_cells,
            n_eaf_cells=self.n_eaf_cells,
        )


@dataclass(frozen=True)
class _SurveyContext:
    """Immutable per-call state a forked survey worker reads.

    Set as a module global immediately before `ordered_map` forks, so the
    workers inherit `hashes`/`row_map` without pickling them once per column.
    Cleared once the map is drained.
    """

    spill_dir: Path
    hashes: np.ndarray
    k: int
    suffix: str
    index_key: str
    row_map: np.ndarray | None


@dataclass(frozen=True)
class _SurveyColumn:
    """One Analysis's contribution to the survey: its sample and its counts.

    Carries the Analysis id so the parent can key `observations` and order the
    concatenated samples by Analysis, not by whichever column finished first.
    """

    analysis_id: str
    selected: np.ndarray
    values: np.ndarray
    n_spill_cells: int
    n_eaf_cells: int


_survey_context: _SurveyContext | None = None


def _survey_column(task: tuple[int, str]) -> _SurveyColumn | None:
    """Sample one Analysis's spill column. Runs in a forked worker.

    Returns `None` for a column with no spill file, which the parent reads as
    "this Analysis contributes nothing" rather than as a zero-valued sample.
    """
    col, analysis_id = task
    context = _survey_context
    if context is None:
        raise RuntimeError("survey context is not set; call survey_eaf_spills")
    path = context.spill_dir / f"{col}{context.suffix}.npz"
    if not path.exists():
        return None
    with np.load(path) as data:
        rows = data[context.index_key].astype(np.int64)
        if context.row_map is not None:
            rows = context.row_map[rows].astype(np.int64)
        eaf = np.asarray(data["eaf"], dtype=np.float64)
        n_spill_cells = int(eaf.size)
        n_eaf_cells = int(np.count_nonzero(np.isfinite(eaf)))
        selected, values = sample_column_rows(rows, eaf, context.hashes, k=context.k)
    return _SurveyColumn(
        analysis_id=analysis_id,
        selected=selected,
        values=values,
        n_spill_cells=n_spill_cells,
        n_eaf_cells=n_eaf_cells,
    )


@dataclass
class _SurveyTotals:
    """The running totals a survey accumulates across its columns."""

    n_spill_cells: int = 0
    n_eaf_cells: int = 0
    sample_rows: list[np.ndarray] = field(default_factory=list)
    sample_values: list[np.ndarray] = field(default_factory=list)


def _concatenate_or_empty(parts: list[np.ndarray], dtype: Any) -> np.ndarray:
    """The sample arrays concatenated in order, or an empty array of `dtype`."""
    return np.concatenate(parts) if parts else np.empty(0, dtype=dtype)


def _accumulate_survey_column(
    result: _SurveyColumn | None,
    alids: Sequence[str],
    observations: dict[str, dict[str, float]],
    totals: _SurveyTotals,
) -> None:
    """Fold one column's sample into the survey, in the order it is yielded."""
    if result is None:
        return
    totals.n_spill_cells += result.n_spill_cells
    totals.n_eaf_cells += result.n_eaf_cells
    totals.sample_rows.append(result.selected)
    totals.sample_values.append(result.values)
    observations[result.analysis_id].update(
        {
            alids[row]: float(value)
            for row, value in zip(result.selected.tolist(), result.values.tolist(), strict=True)
        }
    )


def survey_eaf_spills(
    spill_dir: Path,
    id_by_col: Mapping[int, str],
    alids: Sequence[str],
    hashes: np.ndarray,
    *,
    k: int = DEFAULT_SAMPLE_SITES,
    suffix: str = "",
    index_key: str = "rows",
    row_map: np.ndarray | None = None,
    n_workers: int = 1,
) -> EafSpillSurvey:
    """Sample each Analysis's frequencies and count them, from the spills.

    Every Analysis appears in `observations`, including ones whose source
    carried no frequency: an absent key would read as "not checked yet"
    downstream, where the honest answer is "checked, and there was nothing to
    check."

    `row_map` translates a spill's own row indices onto the axis `alids` and
    `hashes` describe — the Hybrid builder's Dense Component spills are indexed
    by dense row, while both of its components are sampled on the shared axis
    so that an Analysis living mostly off-panel is checked on the same footing
    as one sitting on it.

    The columns are independent, so `n_workers` > 1 surveys them through
    `ordered_map`: each worker loads, samples and counts one column, and the
    parent concatenates the samples and fills `observations` in Analysis order,
    never in completion order. `n_workers <= 1` is the serial path.
    """
    global _survey_context
    observations: dict[str, dict[str, float]] = {aid: {} for aid in id_by_col.values()}
    totals = _SurveyTotals()
    try:
        _survey_context = _SurveyContext(
            spill_dir=spill_dir,
            hashes=hashes,
            k=k,
            suffix=suffix,
            index_key=index_key,
            row_map=row_map,
        )
        for result in ordered_map(_survey_column, list(id_by_col.items()), n_workers):
            _accumulate_survey_column(result, alids, observations, totals)
    finally:
        _survey_context = None
    return EafSpillSurvey(
        observations=observations,
        n_spill_cells=totals.n_spill_cells,
        n_eaf_cells=totals.n_eaf_cells,
        sample_rows=_concatenate_or_empty(totals.sample_rows, np.int64),
        sample_values=_concatenate_or_empty(totals.sample_values, np.float64),
    )


def _write_dense_z_bands(
    root: Any,
    spill_dir: Path,
    n_variants: int,
    n_analyses: int,
    band_cols: int,
    codec: StoreCodec,
    dtype: str,
    pass2_start: float,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray], np.ndarray]:
    """z pass: encode, write, and harvest stored-value top hits.

    One ``(n_variants × band_cols)`` buffer is resident at a time. The top-hit
    harvest thresholds on the **quantised/decoded stored** z -- what a query
    reads back -- not the source z, and reads each hit's se straight from the
    spill rounded to the stored dtype. The overflow table is part of the z
    plane, not an addendum: it is written in the same pass that finished
    writing z. Returns the per-band hit parts and the per-\Analysis
    ``column_has_eaf`` survey for the coordinator.
    """
    z_arr = root["z"]
    band = np.empty((n_variants, band_cols), dtype=codec.z_dtype)
    overflow = ZOverflowBuilder()
    hit_rows_parts: list[np.ndarray] = []
    hit_cols_parts: list[np.ndarray] = []
    hit_z_parts: list[np.ndarray] = []
    hit_se_parts: list[np.ndarray] = []
    column_has_eaf = np.zeros(n_analyses, dtype=bool)

    log.info("Band-write z: %d analyses in bands of %d", n_analyses, band_cols)
    for c0 in range(0, n_analyses, band_cols):
        c1 = min(c0 + band_cols, n_analyses)
        w = c1 - c0
        band[:, :w] = codec.z_fill_value
        for c in range(c0, c1):
            local = c - c0
            with np.load(spill_dir / f"{c}.npz") as data:
                rows = data["rows"]
                column_has_eaf[c] = bool(np.isfinite(data["eaf"]).any())
                band[rows, local] = codec.encode_z(
                    data["z"],
                    positions=rows.astype(np.int64) * n_analyses + c,
                    overflow=overflow,
                )
                zc = codec.quantise_z(data["z"])  # what a query will read back
                hit = np.abs(zc) >= _TOP_HIT_Z_CRIT
                if np.any(hit):
                    hit_rows_parts.append(rows[hit])
                    hit_cols_parts.append(np.full(int(np.count_nonzero(hit)), c, dtype=np.int64))
                    hit_z_parts.append(zc[hit])
                    hit_se_parts.append(data["se"][hit].astype(dtype).astype(np.float32))
        z_arr[:, c0:c1] = band[:, :w]
        _log_progress(
            "Band-write z", c1, n_analyses, pass2_start, f"cols {c0}:{c1}", every=band_cols
        )
    overflow.table().write(root)
    return hit_rows_parts, hit_cols_parts, hit_z_parts, hit_se_parts, column_has_eaf


def _write_dense_se_bands(
    root: Any,
    spill_dir: Path,
    n_variants: int,
    n_analyses: int,
    band_cols: int,
    dtype: str,
    pass2_start: float,
) -> None:
    """se pass: the independent float-scratch write over one band at a time.

    ``z`` and ``se`` no longer share a dtype (ADR 0037), so this pass owns a
    fresh buffer of its own rather than reusing the z pass's.
    """
    se_arr = root["se"]
    band = np.empty((n_variants, band_cols), dtype=dtype)
    for c0 in range(0, n_analyses, band_cols):
        c1 = min(c0 + band_cols, n_analyses)
        w = c1 - c0
        band[:, :w] = np.nan
        for c in range(c0, c1):
            local = c - c0
            with np.load(spill_dir / f"{c}.npz") as data:
                band[data["rows"], local] = data["se"]
        se_arr[:, c0:c1] = band[:, :w]
        _log_progress(
            "Band-write se", c1, n_analyses, pass2_start, f"cols {c0}:{c1}", every=band_cols
        )


def _write_dense_bands(
    staged: StagedRelease,
    spill_dir: Path,
    n_variants: int,
    n_analyses: int,
    effective_chunks: tuple[int, int],
    dtype: str,
    pass2_start: float,
    encoding: StoreEncoding,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stream the retained per-column spills into the zarr in chunk-column bands.

    ``z`` and ``se`` are written in two separate passes so only one band is
    ever resident, and the top-hit harvest runs in the z-pass on the stored z.
    Spills are retained through the se pass and the EAF write, and are only
    unlinked once both consumed them. Returns the concatenated top-hit
    candidate arrays ``(rows, cols, z, se)`` plus the per-\Analysis
    ``column_has_eaf`` survey for the index build and EAF decision.
    """
    root = staged.arrays(mode="a")
    band_cols = effective_chunks[1]
    codec = StoreCodec(encoding)
    hit_rows_parts, hit_cols_parts, hit_z_parts, hit_se_parts, column_has_eaf = (
        _write_dense_z_bands(
            root, spill_dir, n_variants, n_analyses, band_cols, codec, dtype, pass2_start
        )
    )
    _write_dense_se_bands(
        root, spill_dir, n_variants, n_analyses, band_cols, dtype, pass2_start
    )

    # Pass 3 -- eaf. Its own float32 buffer, since eaf cannot share z/se's
    # float16 (see `_create_eaf_array`). What is written is decided by the
    # plan, not by whether the array happens to be wanted here.
    if column_has_eaf.any() and encoding.eaf.is_absent:
        raise ValueError(
            "the encoding plan declares no eaf plane, but "
            f"{int(column_has_eaf.sum())} of {n_analyses} Analyses carried a frequency; "
            "the plan and the data disagree (ADR 0037 §2)"
        )
    if not encoding.eaf.is_absent:
        # Written whenever the plan says so, even if *this* component carries
        # no frequency: a Hybrid release's two components share one plan, and
        # a Dense Component with no EAF where the Ragged Overflow has some
        # would otherwise declare a plane it does not have. An all-absent
        # `int8` plane costs essentially nothing compressed.
        _write_dense_eaf(
            staged,
            spill_dir,
            n_variants,
            n_analyses,
            effective_chunks,
            band_cols,
            codec,
            pass2_start,
        )
    for c in range(n_analyses):
        (spill_dir / f"{c}.npz").unlink(missing_ok=True)

    if hit_rows_parts:
        return (
            np.concatenate(hit_rows_parts),
            np.concatenate(hit_cols_parts),
            np.concatenate(hit_z_parts),
            np.concatenate(hit_se_parts),
            column_has_eaf,
        )
    return (
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.float32),
        np.empty(0, dtype=np.float32),
        column_has_eaf,
    )


def _write_manifest(
    staged: StagedRelease,
    store_id: str,
    release_id: str,
    n_variants: int,
    n_analyses: int,
    chain_file: str | Path | None,
    chunk_shape: tuple[int, int],
    dtype: str,
    encoding: StoreEncoding,
    eaf_orientation: dict[str, Any] | None = None,
    variant_reference: str | None = None,
) -> None:
    manifest = StoreManifest(
        encoding=encoding,
        store_id=store_id,
        release_id=release_id,
        format_version=CURRENT_FORMAT_VERSION,
        primary_layout=PrimaryStorageLayout.DENSE,
        association_coverage=AssociationCoverage.FULL,
        completion_state=CompletionState.OBSERVED_ONLY,
        reference_assembly="GRCh38",
        created_at=datetime.now(UTC).isoformat(),
        provenance={
            "builder": (
                "opengwasdb.v0.1_dense_vcf_single_pass"
                if variant_reference is not None
                else "opengwasdb.v0.1_dense_vcf_two_pass"
            ),
            "chain_file": str(chain_file) if chain_file else "pyliftover_builtin_hg19_hg38",
            "n_variants": n_variants,
            "n_analyses": n_analyses,
            "dense": {
                "statistic_arrays": ["z", "se"],
                "se_dtype": encoding.se.dtype,
                "chunk_shape": list(chunk_shape),
                "compressor": DEFAULT_COMPRESSOR,
                "top_hit_thresholds": [5e-8, 5e-6, 5e-4],
            },
            **({"eaf_orientation": eaf_orientation} if eaf_orientation is not None else {}),
            **(
                {"variant_reference": variant_reference}
                if variant_reference is not None
                else {}
            ),
        },
    )
    staged.write_manifest(manifest)
