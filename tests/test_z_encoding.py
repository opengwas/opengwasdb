"""Fixed-point `z` encoding: the plan, the codec, and the overflow table.

Issue #114 / ADR 0037 §1. The unit-level half of the change -- the layout
round-trips live in `test_z_encoding_layouts.py`.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from opengwasdb.encoding import (
    Z_MISSING,
    Z_OVERFLOW,
    EncodingMeasurements,
    StoreCodec,
    StoreEncoding,
    UnsupportedEncoding,
    ZOverflowBuilder,
    ZOverflowTable,
)
from opengwasdb.stats import log10_p_two_sided


def _codec(values: np.ndarray) -> tuple[StoreCodec, np.ndarray]:
    """Encode `values` at flat positions 0..n and return a decoding codec + codes."""
    encoding = StoreEncoding.decide(EncodingMeasurements(n_analyses=1))
    builder = ZOverflowBuilder()
    codes = StoreCodec(encoding).encode_z(
        values, positions=np.arange(len(values)), overflow=builder
    )
    return StoreCodec(encoding, z_overflow=builder.table()), codes


# ── The plan ────────────────────────────────────────────────────────────────


def test_decide_encodes_z_as_int16_fixed_point_at_1024():
    encoding = StoreEncoding.decide(EncodingMeasurements(n_analyses=8))
    assert encoding.z.kind == "int16_fixed"
    assert encoding.z.scale == 1024
    assert encoding.se.kind == "float16"


def test_plan_round_trips_through_the_manifest():
    encoding = StoreEncoding.decide(EncodingMeasurements(n_analyses=8))
    payload = json.loads(json.dumps(encoding.to_manifest()))
    assert StoreEncoding.from_manifest(payload) == encoding


def test_reader_rejects_an_encoding_kind_it_does_not_implement():
    payload = StoreEncoding.decide(EncodingMeasurements(n_analyses=1)).to_manifest()
    payload["z"]["kind"] = "int8_fixed"
    with pytest.raises(UnsupportedEncoding, match="int8_fixed"):
        StoreEncoding.from_manifest(payload)


def test_legacy_plan_is_float16_for_stores_that_declare_none():
    assert StoreEncoding.legacy().z.kind == "float16"


def test_legacy_codec_decodes_a_float16_plane_unchanged():
    raw = np.array([2.5, np.nan, -6.0], dtype=np.float16)
    decoded = StoreCodec(StoreEncoding.legacy()).decode_z(raw)
    assert decoded.dtype == np.float32
    assert decoded[0] == pytest.approx(2.5)
    assert np.isnan(decoded[1])
    assert decoded[2] == pytest.approx(-6.0)


# ── Round trip ──────────────────────────────────────────────────────────────


def test_in_range_values_round_trip_to_within_a_thousandth():
    values = np.array(
        [0.0, 1e-4, -1e-4, 1.959964, -5.451, 8.0, 29.999, 31.9, -31.9], dtype=np.float64
    )
    codec, codes = _codec(values)
    assert codes.dtype == np.int16
    decoded = codec.decode_z(codes, positions=np.arange(len(values)))
    assert np.max(np.abs(decoded - values)) < 0.001


def test_missing_cells_encode_to_the_reserved_sentinel_and_decode_to_nan():
    values = np.array([np.nan, 3.0, np.nan], dtype=np.float64)
    codec, codes = _codec(values)
    assert codes[0] == Z_MISSING
    assert codes[2] == Z_MISSING
    decoded = codec.decode_z(codes, positions=np.arange(3))
    assert np.isnan(decoded[[0, 2]]).all()
    assert decoded[1] == pytest.approx(3.0, abs=0.001)


def test_out_of_range_values_are_held_exactly_in_the_overflow_table():
    # 47.8 is the FADS1/FADS2 pilot hit; 137.5 the largest |z| in ukb-b (ADR 0037).
    values = np.array([47.8, -137.5, 32.0, 5.0], dtype=np.float64)
    codec, codes = _codec(values)
    assert list(codes[:3]) == [Z_OVERFLOW, Z_OVERFLOW, Z_OVERFLOW]
    decoded = codec.decode_z(codes, positions=np.arange(len(values)))
    assert decoded[0] == pytest.approx(47.8, abs=1e-5)
    assert decoded[1] == pytest.approx(-137.5, abs=1e-5)
    assert decoded[2] == pytest.approx(32.0, abs=1e-5)


def test_the_representable_edge_is_stated_exactly_not_as_plus_or_minus_32():
    encoding = StoreEncoding.decide(EncodingMeasurements(n_analyses=1))
    assert encoding.z.max_representable == pytest.approx(32767 / 1024)
    assert encoding.z.min_representable == pytest.approx(-32766 / 1024)
    values = np.array(
        [encoding.z.max_representable, encoding.z.min_representable], dtype=np.float64
    )
    _, codes = _codec(values)
    assert list(codes) == [32767, -32766]


def test_a_non_finite_statistic_fails_loudly_rather_than_being_stored():
    with pytest.raises(ValueError, match="non-finite"):
        _codec(np.array([1.0, np.inf], dtype=np.float64))


def test_decoding_an_out_of_range_cell_without_its_table_raises():
    encoding = StoreEncoding.decide(EncodingMeasurements(n_analyses=1))
    codes = np.array([Z_OVERFLOW], dtype=np.int16)
    with pytest.raises(ValueError, match="overflow"):
        StoreCodec(encoding).decode_z(codes, positions=np.arange(1))


def test_an_overflow_table_missing_the_cell_it_is_asked_for_raises():
    codec = StoreCodec(
        StoreEncoding.decide(EncodingMeasurements(n_analyses=1)),
        z_overflow=ZOverflowTable(
            index=np.array([7], dtype=np.int64), value=np.array([47.8], dtype=np.float32)
        ),
    )
    with pytest.raises(ValueError, match="overflow"):
        codec.decode_z(np.array([Z_OVERFLOW], dtype=np.int16), positions=np.array([3]))


def test_quantise_reports_what_a_query_will_read_back():
    values = np.array([5.4511, 47.8, np.nan, -0.31337], dtype=np.float64)
    encoding = StoreEncoding.decide(EncodingMeasurements(n_analyses=1))
    quantised = StoreCodec(encoding).quantise_z(values)
    codec, codes = _codec(values)
    decoded = codec.decode_z(codes, positions=np.arange(len(values)))
    assert np.array_equal(quantised, decoded, equal_nan=True)


# ── What the encoding is for ────────────────────────────────────────────────


def test_p_value_at_the_largest_pilot_z_is_accurate_to_two_percent():
    """float16 gets z=47.8's p wrong by 1.82x (ADR 0037); fixed point does not."""
    codec, codes = _codec(np.array([47.8], dtype=np.float64))
    stored = float(codec.decode_z(codes, positions=np.arange(1))[0])
    exact = log10_p_two_sided(np.array([47.8]))[0]
    assert abs(log10_p_two_sided(np.array([stored]))[0] - exact) < np.log10(1.02)


def test_worst_case_p_error_across_the_representable_range_stays_under_two_percent():
    encoding = StoreEncoding.decide(EncodingMeasurements(n_analyses=1))
    # Half a quantisation step is the worst a round-to-nearest can do.
    edge = np.array([5.0, 10.0, 20.0, encoding.z.max_representable])
    shifted = edge + 0.5 / encoding.z.scale
    ratio = np.abs(log10_p_two_sided(edge) - log10_p_two_sided(shifted))
    assert np.max(ratio) < np.log10(1.02)


# ── The overflow table ──────────────────────────────────────────────────────


def test_the_overflow_table_is_sorted_by_position_for_binary_search():
    builder = ZOverflowBuilder()
    builder.add(np.array([9, 2]), np.array([40.0, -50.0]))
    builder.add(np.array([5]), np.array([33.0]))
    table = builder.table()
    assert list(table.index) == [2, 5, 9]
    assert list(table.value) == [-50.0, 33.0, 40.0]


def test_the_overflow_table_round_trips_through_zarr(tmp_path):
    import zarr

    root = zarr.open_group(str(tmp_path / "data.zarr"), mode="w")
    builder = ZOverflowBuilder()
    builder.add(np.array([4, 1]), np.array([137.5, -47.8]))
    builder.table().write(root)
    read = ZOverflowTable.read(root)
    assert list(read.index) == [1, 4]
    assert read.value[1] == pytest.approx(137.5)


def test_an_absent_overflow_table_reads_as_empty(tmp_path):
    import zarr

    root = zarr.open_group(str(tmp_path / "data.zarr"), mode="w")
    assert len(ZOverflowTable.read(root).index) == 0
