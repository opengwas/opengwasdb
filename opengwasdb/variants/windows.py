"""Genomic windows for the hierarchical variant-union reduction (issue #188).

The union of 1,000+ source files is reduced as a tree of bounded merges rather
than one parent-process k-way merge. Variants are partitioned into fixed-size,
non-overlapping windows of ``(chromosome, floor(position / window_size))``;
each window's shards reduce independently, and the final window shards
concatenate in genomic order, so the reference is assembled without a global
re-sort. The window key is comparable, so sorting windows sorts the genome.
"""

from __future__ import annotations

from opengwasdb.variants.normalise import chromosome_sort_key

__all__ = [
    "DEFAULT_MAP_SPILL_RECORDS",
    "DEFAULT_REDUCTION_BATCH_SIZE",
    "DEFAULT_WINDOW_SIZE_MB",
    "WindowKey",
    "window_key",
    "window_size_bp",
]

#: Base pairs per genomic window when no caller chooses one.
DEFAULT_WINDOW_SIZE_MB = 20.0
#: Shards merged per tree-reduction task when no caller chooses one.
DEFAULT_REDUCTION_BATCH_SIZE = 16
#: Buffered variants a map worker may hold before spilling every window buffer
#: to a shard (issue #194). At the default, a worker's peak memory tracks this
#: constant rather than the number of rows in its manifest slice.
DEFAULT_MAP_SPILL_RECORDS = 5_000_000

#: ``(chromosome_sort_key, window index)`` -- comparable, and genomic-ordered.
WindowKey = tuple[tuple[int, str], int]


def window_size_bp(window_size_mb: float) -> int:
    """Base pairs per genomic window, failing loudly on a non-positive size."""
    if not window_size_mb > 0:
        raise ValueError(f"window size must be positive, got {window_size_mb!r} Mb")
    return int(window_size_mb * 1_000_000)


def window_key(chromosome: str, position: int, size_bp: int) -> WindowKey:
    """The non-overlapping ``(chromosome, floor(position / window_size))`` window."""
    return chromosome_sort_key(chromosome), position // size_bp
