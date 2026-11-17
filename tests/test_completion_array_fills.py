"""A block prepares fills as arrays, preserving the scalar cap's exact values."""
from __future__ import annotations

import numpy as np
import pytest

from opengwasdb.completion import block


def test_prepared_fills_are_arrays_with_exact_region_caps(monkeypatch: pytest.MonkeyPatch):
    # Unsorted positions, duplicate positions, inclusive edges and an empty window.
    positions = np.array([100, 0, 200, 100, 300, 1000, 50, 150])
    observed = np.array([3., -2., np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
    predicted = np.array([3., -2., -9., 9., 8., -7., -0., 2.5])
    monkeypatch.setattr(block, "impute_z_block", lambda *a, **kw: (predicted, 0.9))
    result = block.complete_block_for_analysis(
        observed, np.where(np.isnan(observed), np.nan, 0.2), np.full(8, 0.3),
        positions, np.eye(8), np.ones(8), min_cor=0.0, region_cap_bp=100,
    )
    assert isinstance(result.fills, np.ndarray), "no Python object per prepared fill"
    assert result.fills.dtype.names == ("local_index", "z", "se")
    np.testing.assert_array_equal(result.fills["local_index"], [2, 3, 4, 5, 6, 7])
    expected = np.array([-3., 3., 8., -7., -0., 2.5])
    assert result.fills["z"].tobytes() == expected.tobytes()
    np.testing.assert_array_equal(result.fills["se"], np.full(6, 0.19999999999999998))
    assert result.n_missing == 0


def test_block_checkpoint_uses_owned_array_rows_and_keeps_long_alids(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
):
    from ld_blocks import write_ld_block

    from opengwasdb.completion.checkpoint import write_block_checkpoint

    long_alid = "1:300:" + "A" * 70 + ":" + "C" * 70
    unfilled_long_alid = "1:100:" + "A" * 90 + ":" + "C" * 90
    alids = [unfilled_long_alid, "1:200:A:G", long_alid, "1:400:A:G"]
    write_ld_block(tmp_path / "EUR" / "1", "100-400", [
        (alid, 0.3, pos) for alid, pos in zip(alids, [100, 200, 300, 400], strict=True)
    ])
    observed = np.array([2., -3., np.nan, np.nan])
    monkeypatch.setattr(
        block, "impute_z_block", lambda *a, **kw: (np.array([2., -3., 1.5, -2.5]), 0.9)
    )
    def reader(loaded, canonical):
        assert canonical == alids
        return [2, 7], lambda ai: (observed, np.where(np.isnan(observed), np.nan, 0.2))
    result = block.run_block(
        tmp_path / "EUR" / "1" / "100-400.tsv", 0.9, 0.0, None, reader,
        ceded=lambda ai: frozenset({alids[3]}) if ai == 2 else frozenset(),
    )
    assert result is not None
    assert isinstance(result.fills, np.ndarray), "ownership must not form FillRow objects"
    assert result.fills.dtype.names == ("local_index", "analysis_index", "z", "se")
    checkpoint = tmp_path / "block.npz"
    write_block_checkpoint(checkpoint, result)
    with np.load(checkpoint, allow_pickle=False) as written:
        assert written["format"].tolist() == [3]
        assert written["f_alid"].tolist() == [b"", b"", b"1:400:A:G"]
        assert written["f_long_at"].tolist() == [0, 1]
        assert written["f_long_alid"].tolist() == [long_alid.encode()] * 2
        assert written["f_long_alid"].dtype.itemsize == len(long_alid)
        assert written["f_ai"].tolist() == [2, 7, 7]
        assert written["f_z"].tobytes() == np.array([1.5, 1.5, -2.5], np.float32).tobytes()
        assert written["f_se"].tobytes() == np.full(3, 0.2, np.float32).tobytes()
        assert written["q_ai"].tolist() == [2, 7]
        assert written["q_nimp"].tolist() == [1, 2]
        assert written["q_nmiss"].tolist() == [0, 0]
        assert written["q_pearson"].tolist() == [0.9, 0.9]


@pytest.mark.parametrize("window", [None, 0, 1, 1_000_000, 10_000_000])
def test_array_cap_matches_scalar_window_reduction_bit_for_bit(monkeypatch, window):
    rng = np.random.default_rng(422)
    positions = rng.integers(0, 10_000_000, 257)
    observed = rng.normal(size=257)
    observed[100:] = np.nan
    predicted = rng.normal(size=257) * 10
    predicted[100:102] = [-0., 0.]
    monkeypatch.setattr(block, "impute_z_block", lambda *a, **kw: (predicted, 0.8))
    result = block.complete_block_for_analysis(
        observed, np.where(np.isnan(observed), np.nan, 0.25), np.full(257, 0.5),
        positions, np.eye(257), np.ones(257), min_cor=0.0, region_cap_bp=window,
    )
    expected = predicted[100:].copy()
    if window is not None:
        for i, pos in enumerate(positions[100:]):
            inside = np.abs(positions[:100] - pos) <= window
            if inside.any():
                maximum = float(np.abs(observed[:100][inside]).max())
                if abs(expected[i]) > maximum:
                    expected[i] = maximum if expected[i] >= 0 else -maximum
    assert len(result.fills) == 157, "exercise many overlapping windows"
    assert result.fills["z"].tobytes() == expected.tobytes()


@pytest.mark.parametrize("observed_count,failed", [(0, False), (1, False), (2, True), (4, False)])
def test_empty_prepared_fills_keep_quality_and_array_dtype(monkeypatch, observed_count, failed):
    z = np.full(4, np.nan)
    z[:observed_count] = 1.0
    monkeypatch.setattr(
        block, "impute_z_block", lambda *a, **kw: (None, 0.2) if failed else (z, 1.0)
    )
    result = block.complete_block_for_analysis(
        z, np.where(np.isnan(z), np.nan, 0.2), np.full(4, 0.5), np.arange(4),
        np.eye(4), np.ones(4), min_cor=0.7,
    )
    assert isinstance(result.fills, np.ndarray)
    assert result.fills.shape == (0,)
    assert result.fills.dtype == block.PREPARED_FILL_DTYPE
    assert result.n_missing == 4 - observed_count
    assert result.pearson_r == (None if observed_count < 2 else 0.2 if failed else 1.0)


def test_checkpoint_memory_depends_on_filled_alid_width(tmp_path):
    import tracemalloc

    from opengwasdb.completion.checkpoint import (
        BLOCK_FILL_DTYPE,
        BlockCompletionResult,
        write_block_checkpoint,
    )

    filled_alid = "1:100:A:" + "C" * 60
    unfilled_alid = "1:200:A:" + "C" * 20_000
    fills = np.zeros(2_000, dtype=BLOCK_FILL_DTYPE)
    fills["z"], fills["se"] = 1.0, 0.25
    result = BlockCompletionResult("1/1-200", [], fills, [filled_alid, unfilled_alid])
    path = tmp_path / "block.npz"
    tracemalloc.start()
    try:
        write_block_checkpoint(path, result)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    with np.load(path, allow_pickle=False) as written:
        assert written["f_long_alid"].shape == (2_000,)
        assert written["f_long_alid"].dtype.itemsize == len(filled_alid)
        assert np.all(written["f_long_alid"] == filled_alid.encode())
    # The compact checkpoint is <300 KiB. A 2 MiB allowance accommodates ZIP
    # buffers but rejects allocating 40 MB at the unfilled panel ALID's width.
    assert peak < 2 * 1024**2, peak
