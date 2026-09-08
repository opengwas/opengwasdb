"""The SE size measurement charges edge chunks at zarr's padded size (#158).

`_packed_chunks` in `opengwasdb/encoding/measure.py` compresses each measured
slice at the size it happens to be, but zarr stores every chunk at its declared
shape: an edge chunk -- one whose span runs past the array's extent -- is padded
out to the full chunk with the array's fill value *before* it is compressed. A
plane whose extent is not a multiple of its chunk was therefore being measured
smaller than the array that will actually be written, biasing the compressed-
bytes gate toward the residual coding on exactly the small or awkwardly-shaped
planes where the margin is narrowest.

Every test here compares a measurement against a real zarr array written the way
the builders write theirs (`create_dataset(data=..., chunks=..., fill_value=...)`)
and sums the bytes actually stored on disk, so a measurement that omitted the
edge padding fails the comparison.
"""

from __future__ import annotations

import numpy as np
import pytest
import zarr
from numcodecs import Blosc

from opengwasdb.encoding.measure import _packed_chunks, fit_se_grid

_COMPRESSOR = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)


def _real_zarr_bytes(
    data: np.ndarray, chunks: tuple[int, ...], dtype: str, fill_value: object
) -> int:
    """Sum of the compressed chunk bytes zarr actually stores for `data`.

    Mirrors the builders' `create_dataset(data=..., chunks=..., compressor=...)`
    call, with the fill value stated explicitly so the test controls the padding
    content. An array whose extent is not a multiple of its chunk therefore
    includes one padded edge chunk, exactly as on a real store.
    """
    with zarr.TempStore() as store:
        group = zarr.open_group(store, mode="w")
        group.create_dataset(
            "measured",
            data=data,
            chunks=chunks,
            compressor=_COMPRESSOR,
            dtype=dtype,
            fill_value=fill_value,
        )
        return sum(
            len(store[k])
            for k in store.keys()
            if k.startswith("measured/") and not k.endswith((".zarray", ".zattrs"))
        )


def _unpadded_sum(data: np.ndarray, chunk_shape: tuple[int, ...]) -> int:
    """What the pre-#158 measurement charged: each slice at its own size."""
    shape = (chunk_shape,) if isinstance(chunk_shape, int) else chunk_shape
    if data.ndim == 1:
        chunk = shape[0]
        return sum(
            len(_COMPRESSOR.encode(np.ascontiguousarray(data[start : start + chunk])))
            for start in range(0, len(data), chunk)
        )
    assert data.ndim == 2
    cr, cc = shape
    return sum(
        len(_COMPRESSOR.encode(np.ascontiguousarray(data[r0 : r0 + cr, c0 : c0 + cc])))
        for r0 in range(0, data.shape[0], cr)
        for c0 in range(0, data.shape[1], cc)
    )


#: Codes that do not all land in the last chunk as a compressible run, so the
#: padded edge chunk genuinely costs more than the unpadded data would.
_1D_DATA = np.array([5, -12, 30, 0, 7, -25, 11, 3, -9, 44, 0, 18, -30, 6], dtype=np.int8)
_2D_DATA = np.array(
    [
        [5, -12, 30],
        [0, 7, -25],
        [11, 3, -9],
        [44, 0, 18],
        [-30, 6, 2],
        [8, -5, 21],
        [0, 13, -2],
        [-17, 4, 9],
        [26, -8, 0],
    ],
    dtype=np.int8,
)


def test_packed_1d_charges_a_partial_plane_at_its_padded_size() -> None:
    """1-D plane: 14 cells in chunks of 4 end in a 2-cell edge chunk.

    zarr stores that last chunk padded to 4 cells with the array's fill value,
    so the measured cost must be the cost of the padded chunk -- which is what a
    real array of the same extent, chunk and fill actually occupies on disk.
    """
    chunk = 4
    data = _1D_DATA
    assert len(data) % chunk != 0  # the fixture only tests something if partial

    measured = _packed_chunks(_COMPRESSOR, data, chunk, fill_value=0)
    real = _real_zarr_bytes(data, (chunk,), "int8", 0)
    assert measured == real
    # The padded measurement is strictly larger than the old unpadded one: the
    # fixture only tests the fix if the two differ.
    assert measured > _unpadded_sum(data, (chunk,))


@pytest.mark.parametrize("fill_value", [0, -128], ids=["zarr-default", "se-missing"])
def test_packed_2d_charges_a_partial_plane_at_its_padded_size(fill_value: int) -> None:
    """2-D plane partial in *both* dimensions: 9x3 cells, 4x2 chunks.

    The bottom and right edge chunks are each padded to the full 4x2 chunk, with
    the array's own declared fill value (issue #158: "each array's own fill").
    """
    data = _2D_DATA
    chunk = (4, 2)
    assert data.shape[0] % chunk[0] != 0 and data.shape[1] % chunk[1] != 0

    measured = _packed_chunks(_COMPRESSOR, data, chunk, fill_value=fill_value)
    real = _real_zarr_bytes(data, chunk, "int8", fill_value)
    assert measured == real
    assert measured > _unpadded_sum(data, chunk)


def test_packed_chunks_matches_zarr_for_each_measured_side_array() -> None:
    """The coefficient array and both side tables are padded too.

    They are charged in `_candidate_bytes` at chunk shapes their extents do not
    divide: an `int64` index and `float32` value table of 450_007 entries in
    200_000-cell chunks, and a per-Analysis coefficient grid whose row count is
    not a multiple of its 1024-row chunk.
    """
    rng = np.random.default_rng(158)
    cases: list[tuple[np.ndarray, tuple[int, ...], str, object]] = [
        # se_exception_index: int64 flat, chunked at EXACT_TABLE_CHUNK (200_000)
        ((rng.integers(0, 10**9, size=450_007, dtype=np.int64) * 2), (200_000,), "int64", 0),
        # se_exception_value: float32 flat
        ((rng.standard_normal(450_007) * 0.1).astype(np.float32), (200_000,), "float32", 0.0),
        # se_coefficients: (n_analyses, 2) in (1024, 2) chunks
        ((rng.standard_normal((2_500, 2)) * 0.5).astype(np.float32), (1024, 2), "float32", 0.0),
    ]
    for data, chunk, dtype, fill in cases:
        # Each fixture's extent must leave an edge chunk for the test to mean
        # anything: partial along the flat axis (or the coefficient grid's rows).
        assert data.shape[0] % chunk[0] != 0
        measured = _packed_chunks(_COMPRESSOR, data, chunk, fill_value=fill)
        assert measured == _real_zarr_bytes(data, chunk, dtype, fill)


def test_packed_chunks_is_unchanged_when_the_shape_divides_evenly() -> None:
    """A plane whose extent is a multiple of its chunk has no edge chunk.

    Padding must not change what an evenly-divided plane costs: every chunk is
    full already, so the measured total is exactly the pre-#158 slice sum (and
    exactly what zarr stores).
    """
    data = np.tile(_2D_DATA, (8, 6))  # 72 x 18: a multiple of the 4x2 chunk
    chunk = (4, 2)
    assert data.shape[0] % chunk[0] == 0 and data.shape[1] % chunk[1] == 0

    assert _packed_chunks(_COMPRESSOR, data, chunk, fill_value=0) == _unpadded_sum(data, chunk)
    assert _packed_chunks(_COMPRESSOR, data, chunk, fill_value=0) == _real_zarr_bytes(
        data, chunk, "int8", 0
    )


def test_fit_se_grid_float16_cost_includes_the_edge_chunk_padding() -> None:
    """The number `decide()` compares against candidates is the padded one.

    `float16_compressed_bytes` is the baseline of the compressed-bytes gate. On
    a grid whose extent is not a multiple of its chunk it must equal the bytes a
    real `float16` plane of that grid would occupy on disk -- the plane the
    store would actually contain if the residual coding lost.
    """
    n_rows, n_cols = 613, 3  # not a multiple of the (100, 2) chunk
    chunk = (100, 2)
    rng = np.random.default_rng(5)
    frequencies = np.linspace(0.05, 0.95, n_rows, dtype=np.float32)[:, None]
    eaf = np.repeat(frequencies, n_cols, axis=1)
    coefficients = np.array([[-3.0, -0.5], [-2.7, -0.45], [-2.4, -0.4]], dtype=np.float32)
    se = np.exp(
        coefficients[None, :, 0]
        + coefficients[None, :, 1] * np.log(2 * eaf * (1 - eaf))
        + 0.05 * np.sin(np.arange(n_rows)[:, None] * 0.1)
    ).astype(np.float32)
    # A few genuinely missing cells keep the fixture honest about NaNs.
    se[rng.integers(0, n_rows, 7), rng.integers(0, n_cols, 7)] = np.nan

    measured = fit_se_grid(se, eaf, compressor=_COMPRESSOR, chunks=chunk)[1]
    float16_data = se.astype(np.float16)
    assert measured.float16_compressed_bytes == _real_zarr_bytes(
        float16_data, chunk, "float16", 0.0
    )
    # The fixture only tests the fix if the padded charge differs from the old
    # unpadded one -- it does not have to be larger, because an edge pad of
    # zeros can compress *better* than the ragged slice it replaces; it has to
    # be the charge of what zarr actually stores.
    assert measured.float16_compressed_bytes != _unpadded_sum(float16_data, chunk)
