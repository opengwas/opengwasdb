"""The streamed SE optimiser charges edge chunks at zarr's padded size (#158).

`1fe066d` corrected the whole-grid fits in `encoding/measure.py`; the streamed
Dense / Hybrid / migration optimiser in `encoding/se.py` measured the same
arrays with its own cost code and kept compressing partial plane, coefficient
and side-table slices at the size they happened to be. zarr never stores a
partial chunk: an edge chunk is padded out to the declared chunk shape with the
array's fill value *before* it is compressed, so those measurements
undercharged exactly the small or awkwardly-shaped planes where the
compressed-bytes gate has the narrowest margin.

Every cost the optimiser compares now goes through the same
`measure.packed_chunk_bytes` the whole-grid fits use, with the fill each array's
own writer declares: the Dense codes plane `_rewrite_dense` writes declares
`SE_MISSING`, the `float16` a float32 scratch is narrowed to declares NaN, and
the Overflow's whole-written planes, the coefficient array and both side tables
keep the numeric default fill of 0. These tests compare each measured charge
against the bytes a real zarr array of the same extent, chunk and fill
occupies on disk, so a measurement that omits the edge padding fails the
comparison; each one fails against the pre-#158 streamed accounting.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import zarr
from test_se_measurement_edge_padding import _COMPRESSOR, _real_zarr_bytes

from opengwasdb.encoding import se as se_module
from opengwasdb.encoding.codec import se_residual_codes
from opengwasdb.encoding.plan import EafEncoding, SeEncoding, StoreEncoding, ZEncoding
from opengwasdb.encoding.planes import DenseSePlane
from opengwasdb.encoding.se import OverflowCells, optimise_dense_se_joint


def _stored_bytes(group: Any, name: str) -> int:
    """Compressed chunk bytes one array of a live store physically occupies."""
    prefix = f"{name}/"
    return sum(
        len(group.store[k])
        for k in group.store.keys()
        if k.startswith(prefix) and not k.endswith((".zarray", ".zattrs"))
    )


def _dense_group(path, n_rows: int, n_cols: int) -> tuple[Any, np.ndarray]:
    """A float32 Dense scratch plane that fits the MAF model cleanly, partial
    in both dimensions when `n_rows` and `n_cols` do not divide the chunk."""
    group = zarr.open_group(str(path), mode="w")
    frequencies = np.linspace(0.05, 0.95, n_rows, dtype=np.float32)[:, None]
    eaf = np.repeat(frequencies, n_cols, axis=1)
    intercepts = np.linspace(-3.0, -2.4, n_cols, dtype=np.float32)
    slopes = np.full(n_cols, -0.5, dtype=np.float32)
    se = np.exp(
        intercepts[None, :]
        + slopes[None, :] * np.log(2 * eaf * (1 - eaf))
        + 0.03 * np.sin(np.arange(n_rows)[:, None] * 0.05)
    ).astype(np.float32)
    group.create_dataset("eaf", data=eaf, chunks=(100, 2), compressor=_COMPRESSOR, dtype="float32")
    group.create_dataset(
        "se", data=se, chunks=(100, 2), compressor=_COMPRESSOR, dtype="float32"
    )
    group.create_dataset(
        "z",
        data=np.ones_like(se, dtype=np.float16),
        chunks=(100, 2),
        compressor=_COMPRESSOR,
        dtype="float16",
    )
    return group, se


def _run_optimise_capturing_measurements(
    group: Any, monkeypatch: pytest.MonkeyPatch
) -> tuple[StoreEncoding, np.ndarray | None, dict[str, Any]]:
    """Run the streamed optimiser and capture the `SeMeasurements` it decides on."""
    preliminary = StoreEncoding(
        z=ZEncoding("float16"),
        se=SeEncoding("float16"),
        eaf=EafEncoding("float32"),
    )
    captured: dict[str, Any] = {}
    real_decide = StoreEncoding.decide

    def spy(cls: Any, measurements: Any) -> StoreEncoding:
        captured["se"] = measurements.se
        return real_decide(measurements)

    monkeypatch.setattr(StoreEncoding, "decide", classmethod(spy))
    selected, coefficients = optimise_dense_se_joint(group, preliminary)
    monkeypatch.setattr(StoreEncoding, "decide", real_decide)
    return selected, coefficients, captured


def test_streamed_dense_charges_a_partial_plane_at_zarr_padded_size(tmp_path, monkeypatch) -> None:
    """A Dense plane partial in both dimensions, the streamed path's decision.

    The float16 baseline must be what a real float16 plane of the same grid,
    chunk and NaN fill occupies, and the selected residual candidate's cost
    must be what the codes plane, coefficient array and both side tables the
    rewrite actually writes occupy on disk.
    """
    n_rows, n_cols = 613, 3
    chunk = (100, 2)
    assert n_rows % chunk[0] != 0 and n_cols % chunk[1] != 0
    group, se = _dense_group(tmp_path / "dense.zarr", n_rows, n_cols)
    selected, coefficients, captured = _run_optimise_capturing_measurements(group, monkeypatch)
    measured = captured["se"]

    assert selected.se.is_residual, "fixture only tests the fix if the coding is chosen"
    assert measured.float16_compressed_bytes == _real_zarr_bytes(
        se.astype(np.float16), chunk, "float16", np.nan
    )
    candidate = selected.se.residual_range
    stored = sum(
        _stored_bytes(group, name)
        for name in ("se", "se_coefficients", "se_exception_index", "se_exception_value")
    )
    assert measured.compressed_bytes[candidate] == stored
    assert coefficients is not None
    decoded = DenseSePlane.open(group, selected).band(0, n_rows)
    np.testing.assert_allclose(decoded, se.astype(np.float16).astype(np.float32), rtol=0.01)


def test_evenly_divided_dense_shapes_keep_the_unpadded_cost(tmp_path, monkeypatch) -> None:
    """A plane whose extent divides its chunk has no edge chunks to pad.

    The padded measurement must equal the old unpadded slice sum exactly, so a
    shape with no partial chunks neither moves the measured cost nor the
    selected encoding.
    """
    n_rows, n_cols = 600, 2
    chunk = (100, 2)
    assert n_rows % chunk[0] == 0 and n_cols % chunk[1] == 0
    group, se = _dense_group(tmp_path / "dense.zarr", n_rows, n_cols)
    selected, _, captured = _run_optimise_capturing_measurements(group, monkeypatch)
    measured = captured["se"]

    assert selected.se.is_residual
    float16_data = se.astype(np.float16)
    unpadded = sum(
        len(
            _COMPRESSOR.encode(
                np.ascontiguousarray(float16_data[r0 : r0 + 100, c0 : c0 + 2])
            )
        )
        for r0 in range(0, n_rows, 100)
        for c0 in range(0, n_cols, 2)
    )
    assert measured.float16_compressed_bytes == unpadded
    assert measured.float16_compressed_bytes == _real_zarr_bytes(
        float16_data, chunk, "float16", np.nan
    )


def test_coefficient_arrays_are_charged_with_their_padded_edge_chunk() -> None:
    """The review's partial coefficient array: rows that do not divide 1024.

    `write_se_coefficients` declares a (min(n_analyses, 1024), 2) chunk, so a
    2,500-Analysis store stores its last 452 rows padded out to 1024. The
    charge must be what that array occupies on disk, not the unpadded slice
    sum (2,500 rows measured 517 bytes against zarr's 535 in the #161 review).
    """
    rng = np.random.default_rng(158)
    coefficients = (rng.standard_normal((2_500, 2)) * 0.5).astype(np.float32)
    chunk_rows = min(len(coefficients), 1024)
    assert len(coefficients) % chunk_rows != 0

    assert se_module._packed_coefficients(_COMPRESSOR, coefficients) == _real_zarr_bytes(
        coefficients, (chunk_rows, 2), "float32", 0.0
    )


def test_overflow_measurement_charges_its_partial_flat_plane_at_padded_size() -> None:
    """A Hybrid Overflow flat plane whose length does not divide its chunk.

    The Overflow store writes its planes whole in `chunk`-sized chunks with the
    numeric default fill, so the final chunk is padded out to `chunk` with 0.
    The float16 alternative and every candidate's codes must be charged at that
    padded size.
    """
    n = 450_007
    chunk = 200_000
    assert n % chunk != 0
    rng = np.random.default_rng(158)
    eaf = np.linspace(0.05, 0.95, n, dtype=np.float32)
    analyses = rng.integers(0, 2, size=n)
    coefficients = np.array([[-3.0, -0.5], [-2.7, -0.45]], dtype=np.float32)
    eaf64 = np.asarray(eaf, dtype=np.float64)
    x = np.log(2 * eaf64 * (1 - eaf64))
    prediction = coefficients[analyses, 0] + coefficients[analyses, 1] * x
    se = np.exp(prediction + 0.02 * np.sin(np.arange(n) * 0.001)).astype(np.float32)

    cells = OverflowCells(
        se_values=se,
        eaf_values=eaf,
        analysis_indices=analyses,
        n_analyses=2,
    )
    cost = se_module._measure_overflow(cells, coefficients, _COMPRESSOR, chunk, 2)

    assert cost.float_bytes == _real_zarr_bytes(
        se.astype(np.float16), (chunk,), "float16", 0.0
    )
    # Pick the candidate the overflow gates would accept, so the side table is
    # the one the whole plane would store.
    accepted = min(
        (c for c in (0.5, 1.0, 2.0) if cost.exception_counts[c].sum() == 0),
        default=None,
    )
    assert accepted is not None, "fixture data must code cleanly at some candidate"
    residual = np.log(np.asarray(se, dtype=np.float64)) - prediction
    codes, _ = se_residual_codes(
        se, residual, accepted / 127
    )  # the same single-site quantiser the codec uses
    honest = _real_zarr_bytes(codes, (chunk,), "int8", 0)
    assert cost.candidate_bytes[accepted] == honest


def test_side_tables_charge_a_table_that_outgrows_one_chunk_at_padded_size() -> None:
    """A side table longer than EXACT_TABLE_CHUNK ends in a padded edge chunk.

    The rewrite pre-sizes both arrays at ``max(1, min(count, EXACT_TABLE_CHUNK))``
    and writes them slot by slot, so a 450,007-row table is stored in 200,000-row
    chunks and its last 50,007 rows are padded out to 200,000 with the arrays'
    default fill before compression. The final partial flush must be charged at
    that size, while a table small enough to fit one chunk must be charged at
    its own length.
    """
    rng = np.random.default_rng(158)
    for count in (450_007, 45_007):
        index = rng.integers(0, 10**9, size=count) * 2
        index.sort()
        value = (rng.standard_normal(count) * 0.1).astype(np.float32)
        cost = se_module._SideTableCost(_COMPRESSOR, 1)
        cost.add(index, value, np.zeros(count, dtype=np.int64))
        _, charged = cost.finish()
        chunk = min(count, 200_000)
        honest = _real_zarr_bytes(index, (chunk,), "int64", 0) + _real_zarr_bytes(
            value, (chunk,), "float32", 0.0
        )
        assert charged == honest, f"side table of {count} rows charged at the wrong size"
