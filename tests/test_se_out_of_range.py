"""An SE above float16's range is missing, and no encoder writes it (issue #338).

The rule's seams: the helper, and the codec / plane writers that refuse an SE
the rule should already have removed.  Builders and completion are tested in
`test_se_out_of_range_builds.py`.
"""

from __future__ import annotations

import numpy as np
import pytest
import zarr

from opengwasdb.encoding.codec import StoreCodec
from opengwasdb.encoding.plan import EafEncoding, SeEncoding, StoreEncoding, ZEncoding
from opengwasdb.encoding.planes import write_se_csr, write_se_dense
from opengwasdb.encoding.se_range import (
    SE_MAX_STORED,
    SeOutOfRangeError,
    mask_se_out_of_range,
)

EDGE = 65504.0


def _plan(kind: str) -> StoreEncoding:
    return StoreEncoding(
        z=ZEncoding("float16"),
        se=SeEncoding(kind, 0.5) if kind == "int8_residual" else SeEncoding(kind),
        eaf=EafEncoding("float32"),
    )


def test_bound_is_float16_max() -> None:
    assert SE_MAX_STORED == EDGE


def test_mask_sets_both_markers_and_counts() -> None:
    z = np.array([1.0, 2.0, 3.0, 4.0, np.nan], dtype=np.float32)
    se = np.array([0.1, 65505.0, np.inf, EDGE, 70000.0], dtype=np.float32)
    z_out, se_out, n = mask_se_out_of_range(z, se)
    assert n == 3
    # 65,504 itself is kept; +inf and anything above is missing; z agrees with se.
    np.testing.assert_array_equal(np.isnan(z_out), [False, True, True, False, True])
    np.testing.assert_array_equal(np.isnan(se_out), [False, True, True, False, True])
    assert se_out[3] == EDGE
    assert se[1] == 65505.0  # inputs untouched


def test_mask_with_nothing_to_do_returns_zero() -> None:
    z = np.ones(3, dtype=np.float32)
    _, _, n = mask_se_out_of_range(z, np.array([0.1, np.nan, EDGE], dtype=np.float32))
    assert n == 0


@pytest.mark.parametrize("kind", ["float16", "int8_residual"])
@pytest.mark.parametrize("bad", [65505.0, 1e9, np.inf])
def test_encode_se_refuses_out_of_range(kind: str, bad: float) -> None:
    codec = StoreCodec(_plan(kind))
    with pytest.raises(SeOutOfRangeError, match=r"1 standard error.*first: "):
        codec.encode_se(
            np.array([0.1, bad], dtype=np.float32),
            eaf=np.array([0.3, 0.3], dtype=np.float32),
            analysis_index=np.zeros(2, dtype=np.int64),
            coefficients=np.array([[np.log(0.1), -0.5]], dtype=np.float32),
            positions=np.arange(2),
        )


def test_encode_se_writes_exactly_the_bound_as_float16() -> None:
    codec = StoreCodec(_plan("float16"))
    raw = codec.encode_se(
        np.array([EDGE], dtype=np.float32),
        eaf=np.array([0.3], dtype=np.float32),
        analysis_index=np.zeros(1, dtype=np.int64),
        coefficients=np.zeros((1, 2), dtype=np.float32),
    )
    assert np.isfinite(raw[0]) and float(raw[0]) == EDGE


def test_dense_and_csr_writers_refuse_and_accept_the_edge(tmp_path) -> None:
    codec = StoreCodec(_plan("float16"))
    eaf = np.full((2, 2), 0.3, dtype=np.float32)
    for name, bad, good in (("dense", 70000.0, EDGE),):
        group = zarr.open_group(str(tmp_path / f"{name}.zarr"), mode="w", zarr_format=2)
        values = np.array([[0.1, bad], [0.2, 0.3]], dtype=np.float32)
        with pytest.raises(SeOutOfRangeError):
            write_se_dense(group, codec, values, eaf, None, chunks=(2, 2))
        values[0, 1] = good
        write_se_dense(group, codec, values, eaf, None, chunks=(2, 2))
        assert float(group["se"][0, 1]) == EDGE
    group = zarr.open_group(str(tmp_path / "csr.zarr"), mode="w", zarr_format=2)
    flat = np.array([0.1, 70000.0, 0.3], dtype=np.float32)
    with pytest.raises(SeOutOfRangeError):
        write_se_csr(
            group, codec, flat, np.full(3, 0.3, np.float32), np.zeros(3, np.int64), None
        )
