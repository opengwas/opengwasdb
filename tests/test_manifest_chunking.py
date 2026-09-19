"""Tests for the size-balanced manifest split that feeds the map phase (#195).

The map phase used to divide the manifest into exactly ``n_workers`` contiguous
chunks of equal row count, so a single oversized source set the makespan of the
one worker that happened to own it. ``_split_manifest_rows`` now creates several
times more chunks than workers and balances them by on-disk source size, keeping
chunks contiguous and manifest-ordered so first-named-rsid selection (#109) is
unchanged.
"""

from __future__ import annotations

import heapq
from pathlib import Path

from opengwasdb.layouts.dense.build_vcf import (
    _ManifestRow,
    _source_file_size,
    _split_manifest_rows,
)


def _manifest_row(path: Path) -> _ManifestRow:
    return _ManifestRow(
        trait_id=path.stem,
        file_path=str(path),
        trait_name=path.stem,
        n=1_000,
        stored_effect_scale="sd",
        se_divisor=1.0,
        source_reader_capability="",
        source_assembly="hg38",
        original_sd="",
        assigned_ancestry="",
    )


def _rows_with_sizes(tmp_path: Path, sizes: list[int]) -> list[_ManifestRow]:
    rows: list[_ManifestRow] = []
    for index, size in enumerate(sizes):
        path = tmp_path / f"source_{index:03d}.tsv.gz"
        path.write_bytes(b"x" * size)
        rows.append(_manifest_row(path))
    return rows


def _chunk_weights(chunks: list[list[_ManifestRow]]) -> list[int]:
    return [sum(_source_file_size(row) for row in chunk) for chunk in chunks]


def _equal_row_split(
    rows: list[_ManifestRow], n_chunks: int
) -> list[list[_ManifestRow]]:
    """The pre-#195 split: ``n_chunks`` contiguous chunks of equal row count."""
    n = len(rows)
    n_chunks = min(n_chunks, n)
    base, extra = divmod(n, n_chunks)
    chunks: list[list[_ManifestRow]] = []
    start = 0
    for i in range(n_chunks):
        size = base + (1 if i < extra else 0)
        chunks.append(rows[start : start + size])
        start += size
    return chunks


def _simulated_makespan(chunks: list[list[_ManifestRow]], workers: int) -> int:
    """FIFO list-scheduling makespan for ``workers`` identical processes.

    The fork pool pulls tasks in submission order, so this is the wall-clock
    makespan a pool of ``workers`` would see for these chunk weights, computed
    exactly rather than by timing anything.
    """
    free = [0] * workers
    heapq.heapify(free)
    for chunk in chunks:
        heapq.heappush(free, heapq.heappop(free) + sum(_source_file_size(r) for r in chunk))
    return max(free)


# ── contiguous, size-balanced partitioning ───────────────────────────────────


def test_split_is_contiguous_and_targets_more_chunks_than_workers(tmp_path):
    """Issue #195 AC: contiguous in manifest order, and several chunks per
    worker so the pool has work to balance."""
    rows = _rows_with_sizes(tmp_path, [10] * 10)

    chunks = _split_manifest_rows(rows, n_workers=2)

    assert [row for chunk in chunks for row in chunk] == rows, "not contiguous"
    assert len(chunks) == min(len(rows), 4 * 2) == 8
    assert all(chunk for chunk in chunks), "empty chunks are not submitted"


def test_split_balances_equal_sources_by_cumulative_size(tmp_path):
    """10 equal sources into 8 chunks: the two doubled chunks set the max."""
    rows = _rows_with_sizes(tmp_path, [4] * 10)

    chunks = _split_manifest_rows(rows, n_workers=2)
    weights = _chunk_weights(chunks)

    assert len(chunks) == 8
    assert sum(weights) == 40
    assert max(weights) == 8


def test_split_isolates_a_dominant_source(tmp_path):
    """Issue #195 AC: a source far larger than the rest stands alone while the
    remaining sources spread across the remaining chunks."""
    rows = _rows_with_sizes(tmp_path, [10_000] + [100] * 39)

    chunks = _split_manifest_rows(rows, n_workers=4)

    assert [row for chunk in chunks for row in chunk] == rows
    assert len(chunks) == 16
    assert chunks[0] == [rows[0]], "the dominant source must stand alone"
    rest = _chunk_weights(chunks[1:])
    assert sum(rest) == 39 * 100
    assert max(rest) <= 1.5 * (39 * 100 / len(rest))


def test_fewer_sources_than_workers_is_one_chunk_per_source(tmp_path):
    """Issue #195 AC: fewer sources than workers never splits a source."""
    rows = _rows_with_sizes(tmp_path, [10, 20, 30])

    chunks = _split_manifest_rows(rows, n_workers=5)

    assert chunks == [[rows[0]], [rows[1]], [rows[2]]]


def test_unstatable_source_weighs_one_not_zero(tmp_path):
    """A source the split cannot stat keeps unit weight, so a chunk holding it
    never looks free; the worker's reader is what fails loudly (issue #195)."""
    missing = _manifest_row(tmp_path / "missing.tsv.gz")

    assert _source_file_size(missing) == 1


# ── makespan ─────────────────────────────────────────────────────────────────


def test_balanced_split_beats_equal_rows_on_an_imbalanced_manifest(tmp_path):
    """Issue #195 AC, with the numbers recorded in the commit message.

    One 10,000-byte source plus 39 100-byte sources, four workers. The old
    equal-row split puts the dominant source with nine neighbours (10,900); the
    balanced split isolates it (10,000) and spreads the rest.
    """
    workers = 4
    rows = _rows_with_sizes(tmp_path, [10_000] + [100] * 39)

    balanced = _split_manifest_rows(rows, n_workers=workers)
    equal = _equal_row_split(rows, workers)
    balanced_makespan = _simulated_makespan(balanced, workers)
    equal_makespan = _simulated_makespan(equal, workers)

    assert equal_makespan == 10_900
    assert balanced_makespan == 10_000
    assert balanced_makespan < equal_makespan
