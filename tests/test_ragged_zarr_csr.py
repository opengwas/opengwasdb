"""Tests for zarr CSR ragged storage (issue 035)."""

import numpy as np
import pytest

from opengwasdb.encoding import (
    EafEncoding,
    EncodingMeasurements,
    SeEncoding,
    StoreEncoding,
    ZEncoding,
)
from opengwasdb.layouts.ragged.zarr_csr import RaggedCSRReader, RaggedCSRWriter
from opengwasdb.store.arrays import ArrayRole, open_group, shard_layout

# These tests exercise the CSR arrays directly, in a bare directory with no
# manifest.json, so they pass the plan explicitly where a real store's reader
# would read it from the release (ADR 0037).
ENCODING = StoreEncoding.decide(EncodingMeasurements(n_analyses=1))
# The variant axis these fixtures draw indices from. The writer needs it to
# size the per-variant `eaf_baseline` array (ADR 0037 §2); a bare CSR with no
# variant table has no other way to know how long the axis is.
_N_VARIANTS = 10_000_000
# Half a fixed-point step at scale 1/1024, which is the most a round trip can
# move a z-score.
Z_TOL = 1.0 / 2048


def _make_writer(n_analyses: int, seed: int = 42) -> tuple[RaggedCSRWriter, list[tuple]]:
    rng = np.random.default_rng(seed)
    writer = RaggedCSRWriter(_N_VARIANTS)
    expected = []
    for i in range(n_analyses):
        count = int(rng.integers(0, 500))
        vi = np.sort(rng.integers(0, 10_000_000, size=count)).astype(np.int32)
        z = rng.standard_normal(count).astype(np.float32)
        se = np.abs(rng.standard_normal(count)).astype(np.float16)
        writer.add_analysis(vi, z, se)
        expected.append((vi, z, se))
    return writer, expected


def test_round_trip_small(tmp_path):
    writer, expected = _make_writer(20)
    writer.flush(tmp_path, ENCODING)

    reader = RaggedCSRReader(tmp_path, ENCODING)
    assert reader.n_analyses == 20

    for i, (vi, z, se) in enumerate(expected):
        result = reader.get_analysis(i)
        np.testing.assert_array_equal(result.variant_index, vi)
        np.testing.assert_allclose(result.z, z, atol=Z_TOL)
        np.testing.assert_array_equal(result.se, se)


def test_round_trip_large(tmp_path):
    writer, expected = _make_writer(1000)
    writer.flush(tmp_path, ENCODING)

    reader = RaggedCSRReader(tmp_path, ENCODING)
    assert reader.n_analyses == 1000
    assert reader.n_associations == sum(len(v) for v, _, _ in expected)

    # Spot-check a few analyses
    for i in [0, 1, 500, 999]:
        vi, z, se = expected[i]
        result = reader.get_analysis(i)
        np.testing.assert_array_equal(result.variant_index, vi)
        np.testing.assert_allclose(result.z, z, atol=Z_TOL)
        np.testing.assert_array_equal(result.se, se)


def test_empty_analysis(tmp_path):
    writer = RaggedCSRWriter(_N_VARIANTS)
    writer.add_analysis(
        np.array([1, 2, 3], dtype=np.int32),
        np.array([1.0, -2.0, 0.5], dtype=np.float32),
        np.array([0.1, 0.2, 0.3], dtype=np.float16),
    )
    writer.add_analysis(
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.float16),
        np.empty(0, dtype=np.float16),
    )
    writer.add_analysis(
        np.array([100], dtype=np.int32),
        np.array([3.0], dtype=np.float16),
        np.array([0.5], dtype=np.float16),
    )
    writer.flush(tmp_path, ENCODING)

    reader = RaggedCSRReader(tmp_path, ENCODING)
    assert reader.n_analyses == 3

    empty = reader.get_analysis(1)
    assert len(empty.variant_index) == 0
    assert len(empty.z) == 0
    assert len(empty.se) == 0

    last = reader.get_analysis(2)
    assert len(last.variant_index) == 1
    assert int(last.variant_index[0]) == 100


def test_all_empty_analyses(tmp_path):
    writer = RaggedCSRWriter(_N_VARIANTS)
    for _ in range(5):
        writer.add_analysis(
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float16),
            np.empty(0, dtype=np.float16),
        )
    writer.flush(tmp_path, ENCODING)

    reader = RaggedCSRReader(tmp_path, ENCODING)
    assert reader.n_analyses == 5
    assert reader.n_associations == 0
    for i in range(5):
        result = reader.get_analysis(i)
        assert len(result.variant_index) == 0


def test_n_associations_counter(tmp_path):
    writer = RaggedCSRWriter(_N_VARIANTS)
    writer.add_analysis(
        np.array([1, 2], dtype=np.int32),
        np.array([1.0, 2.0], dtype=np.float16),
        np.array([0.1, 0.2], dtype=np.float16),
    )
    writer.add_analysis(
        np.array([3, 4, 5], dtype=np.int32),
        np.array([3.0, 4.0, 5.0], dtype=np.float16),
        np.array([0.3, 0.4, 0.5], dtype=np.float16),
    )
    assert writer.n_associations == 5
    writer.flush(tmp_path, ENCODING)

    reader = RaggedCSRReader(tmp_path, ENCODING)
    assert reader.n_associations == 5


def test_get_analyses_batch(tmp_path):
    writer, expected = _make_writer(10)
    writer.flush(tmp_path, ENCODING)

    reader = RaggedCSRReader(tmp_path, ENCODING)
    results = reader.get_analyses([0, 5, 9])
    assert len(results) == 3
    for result, idx in zip(results, [0, 5, 9]):
        vi, z, se = expected[idx]
        np.testing.assert_array_equal(result.variant_index, vi)


# The Dense and Ragged per-variant shard policies differ above 1,000,000
# elements: `PER_VARIANT` caps a shard at 1,000,000 and `RAGGED_PER_VARIANT` at
# 10,000,000 (ADR 0059). 2,000,000 variants is the smallest length where a
# directly-opened `data.zarr/ragged` group (whose `path` is "") would show a
# wrong role.
_ROLE_N_VARIANTS = 2_000_000


def test_the_ragged_eaf_baseline_takes_the_ragged_shard_policy(tmp_path):
    """A big Ragged per-variant array must not get the Dense shard (#247 r1).

    `RaggedCSRWriter` opens `<store>/data.zarr/ragged` as the group **root**, so
    Zarr reports `group.path == ""`; a role inferred from the path would select
    the Dense `PER_VARIANT` policy (shard 1,000,000) instead of
    `RAGGED_PER_VARIANT` (shard 2,000,000), and the release would not match the
    converter's layout.
    """
    encoding = StoreEncoding(
        z=ZEncoding(kind="int16_fixed", scale=1024),
        se=SeEncoding(kind="float16"),
        eaf=EafEncoding(kind="int8_residual", residual_range=0.5),
    )
    writer = RaggedCSRWriter(_ROLE_N_VARIANTS)
    writer.add_analysis(
        np.array([0, 1, 2], dtype=np.int32),
        np.array([1.0, 2.0, 3.0], dtype=np.float32),
        np.array([0.1, 0.2, 0.3], dtype=np.float32),
        eaf=np.array([0.2, 0.3, 0.4], dtype=np.float32),
    )
    writer.flush(
        tmp_path,
        encoding,
        eaf_baseline=np.zeros(_ROLE_N_VARIANTS, dtype=np.float32),
    )

    root = open_group(tmp_path / "data.zarr" / "ragged", "r")
    baseline = root["eaf_baseline"]
    assert tuple(int(size) for size in baseline.chunks) == (200_000,)
    assert tuple(int(size) for size in baseline.shards) == (2_000_000,)
    # The two policies really differ at this length, or the assertion above would
    # hold for either role.
    dense_policy = shard_layout(
        ArrayRole.PER_VARIANT, (2_000_000,), inner_chunk=(200_000,)
    )
    ragged_policy = shard_layout(
        ArrayRole.RAGGED_PER_VARIANT, (2_000_000,), inner_chunk=(200_000,)
    )
    assert dense_policy == (1_000_000,)
    assert ragged_policy == (2_000_000,)
