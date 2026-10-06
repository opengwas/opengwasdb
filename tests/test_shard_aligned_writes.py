"""No build path writes a partial shard (#247).

A shard is the unit a 0.2.0 release stores as one file (ADR 0057), and a write
that covers part of one turns it into a read-modify-write of the whole shard.
That is correct but loses the throughput the shard exists for, and it is
invisible: the store validates either way, and a small fixture's shard is the
whole array, so nothing else in the suite can show it.

`opengwasdb.store.arrays.require_whole_shard_writes` is the test-time hook.  It
patches `zarr.Array.__setitem__` and refuses a write that does not start and end
on a shard boundary of a 2-D Dense plane (the 1-D arrays whose shard policy is
"one shard holds the whole array", like the exception tables, are written
incrementally by design and are not judged).  Production pays nothing: the hook
is entered only by tests or when `OPEN_GWASDB_REQUIRE_WHOLE_SHARD_WRITES=1`,
which the real-data pilot sets so a genuinely multi-shard build proves its
writers are aligned.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from opengwasdb.encoding.se import _row_block_of
from opengwasdb.layouts.dense.build_vcf import _eaf_row_band, _flush_band, _shard_columns
from opengwasdb.layouts.dense.complete import _completion_band_rows
from opengwasdb.store.arrays import (
    ArrayRole,
    PartialShardWriteError,
    create_array,
    open_group_for_write,
    require_whole_shard_writes,
    sharded_compressor,
)

#: A plane small enough to build in a test but with **two** shards on the
#: Analysis axis and two on the variant axis, so an unaligned selection is
#: observably different from an aligned one.
SHAPE = (20, 16)
INNER = (10, 4)
SHARD = (20, 8)


@pytest.fixture
def plane(tmp_path: Path) -> tuple[object, object]:
    """A v3 Dense plane with a 2x2 grid of shards, and a band buffer for it."""
    root = open_group_for_write(tmp_path / "data.zarr", "w", zarr_format=3)
    array = create_array(
        root,
        "z",
        ArrayRole.DENSE_STATISTIC_PLANE,
        shape=SHAPE,
        dtype="float32",
        fill_value=np.nan,
        compressor=sharded_compressor(),
        inner_chunk=INNER,
        shards=SHARD,
    )
    assert tuple(int(size) for size in array.shards) == SHARD
    assert SHAPE[1] > SHARD[1]  # more than one shard on the Analysis axis
    return array, np.zeros((SHAPE[0], SHARD[1]), dtype="float32")


def test_an_aligned_band_write_passes_the_guard(plane: tuple[object, object]) -> None:
    array, band = plane
    with require_whole_shard_writes():
        _flush_band(array, (0, SHARD[1], band), "test", SHAPE[1], time.monotonic(), SHARD[1])
    assert np.isfinite(np.asarray(array[:, : SHARD[1]])).all()


def test_a_misaligned_band_write_fails_loudly(plane: tuple[object, object]) -> None:
    """The deliberately misaligned writer: a band of one inner chunk, not a shard.

    Before #247 this write is exactly what the Dense VCF band writer did (it
    used the inner chunk as the band width), so this is the regression the guard
    exists to catch, not a hypothetical one.
    """
    array, band = plane
    with require_whole_shard_writes(), pytest.raises(PartialShardWriteError, match="whole shard"):
        _flush_band(
            array,
            (0, INNER[1], band[:, : INNER[1]]),
            "test",
            SHAPE[1],
            time.monotonic(),
            INNER[1],
        )


def test_a_write_that_starts_off_a_shard_boundary_fails(plane: tuple[object, object]) -> None:
    array, band = plane
    with require_whole_shard_writes(), pytest.raises(PartialShardWriteError):
        # Starts at column 4, inside the first shard.
        array[:, INNER[1] : SHARD[1]] = band[:, INNER[1] : SHARD[1]]


def test_a_whole_array_write_is_aligned(plane: tuple[object, object]) -> None:
    """The whole-array write `create_array(data=...)` makes must stay allowed."""
    array, _band = plane
    with require_whole_shard_writes():
        array[...] = np.arange(SHAPE[0] * SHAPE[1], dtype="float32").reshape(SHAPE)
    assert np.isfinite(np.asarray(array[:])).all()


def test_a_partial_shard_write_without_the_guard_is_silent(plane: tuple[object, object]) -> None:
    """The guard is what makes the defect visible; without it, nothing objects.

    This is the "observed failing" half: with the hook absent the misaligned
    write that `test_a_misaligned_band_write_fails_loudly` refuses simply
    succeeds, which is the silent read-modify-write #247 exists to prevent.
    """
    array, band = plane
    array[0 : INNER[0], 0 : INNER[1]] = band[: INNER[0], : INNER[1]]
    assert np.isfinite(np.asarray(array[:INNER[0], :INNER[1]])).all()


# ── the row-block writers choose whole-shard multiples ───────────────────────


def test_the_eaf_row_band_is_a_whole_number_of_shards() -> None:
    for shard_rows in (1, 7, 100_000):
        band = _eaf_row_band(shard_rows, 10)
        assert band % shard_rows == 0, (shard_rows, band)
        assert band >= shard_rows


def test_the_completion_band_rows_are_a_whole_number_of_shards() -> None:
    for shard_rows in (1, 7, 100_000):
        band = _completion_band_rows(shard_rows)
        assert band % shard_rows == 0, (shard_rows, band)
        assert band >= shard_rows


def test_the_dense_vcf_band_width_is_the_shard_analysis_width(plane: tuple[object, object]) -> None:
    """The band writer's width is the shard's Analysis extent, not the inner chunk.

    A band one inner chunk wide is the pre-#247 behaviour and turns every band
    write into a read-modify-write of the shard it ends inside; this pins the
    fix at the function the band writer calls.
    """
    array, _band = plane
    assert _shard_columns(array, INNER) == SHARD[1]
    assert _shard_columns(array, INNER) != INNER[1]


def test_the_se_rewrite_row_block_is_the_shard_rows(plane: tuple[object, object]) -> None:
    array, _band = plane
    assert _row_block_of(array) == SHARD[0]


def test_the_se_rewrite_row_block_falls_back_to_the_inner_chunk_without_a_shard(
    tmp_path: Path,
) -> None:
    root = open_group_for_write(tmp_path / "v2.zarr", "w", zarr_format=2)
    array = create_array(
        root, "se", ArrayRole.DENSE_STATISTIC_PLANE, shape=(7, 3), hint=(7, 3), dtype="float32"
    )
    assert array.shards is None
    assert _row_block_of(array) == 7
