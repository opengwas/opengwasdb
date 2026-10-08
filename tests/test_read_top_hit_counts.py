"""Tests for opengwasdb.layouts.dense.top_hits.read_top_hit_counts (issue #34/#35).

Covers the fallback path for real pre-issue-#22-era stores (e.g. ukb-b) whose
top-hit index predates the `analysis_offsets` array `read_top_hit_counts`
otherwise reads directly.
"""

from __future__ import annotations

import numpy as np
import zarr

from opengwasdb.layouts.dense.top_hits import read_top_hit_counts, threshold_key


def _write_group(store_path, threshold, analysis_index, *, with_offsets: bool, n_analyses: int):
    root = zarr.open_group(str(store_path / "data.zarr"), mode="a", zarr_format=3)
    top = root.require_group("top_hits")
    group = top.require_group(threshold_key(threshold))
    analysis_index = np.asarray(analysis_index, dtype="uint32")
    group.create_array("analysis_index", data=np.asarray(analysis_index, dtype="uint32"))
    if with_offsets:
        order = np.argsort(analysis_index, kind="stable")
        sorted_ai = analysis_index[order]
        offsets = np.empty(n_analyses + 1, dtype="uint64")
        offsets[0] = 0
        np.cumsum(np.bincount(sorted_ai, minlength=n_analyses), dtype=np.uint64, out=offsets[1:])
        group.create_array("analysis_offsets", data=np.asarray(offsets, dtype="uint64"))


def test_read_top_hit_counts_uses_analysis_offsets_when_present(tmp_path):
    _write_group(tmp_path, 5e-8, [1, 1, 0], with_offsets=True, n_analyses=3)
    counts = read_top_hit_counts(tmp_path, 3, thresholds=(5e-8,))
    assert counts["n_hits_5e8"] == [1, 2, 0]


def test_read_top_hit_counts_falls_back_to_analysis_index_when_offsets_missing(tmp_path):
    # Simulates a real pre-issue-#22 store (e.g. ukb-b): only the flat
    # analysis_index array exists, no analysis_offsets.
    _write_group(tmp_path, 5e-8, [1, 1, 0], with_offsets=False, n_analyses=3)
    counts = read_top_hit_counts(tmp_path, 3, thresholds=(5e-8,))
    assert counts["n_hits_5e8"] == [1, 2, 0]


def test_read_top_hit_counts_fallback_handles_trailing_zero_hit_analyses(tmp_path):
    # analysis 2 (the last, n_analyses=3) has zero hits and never appears in
    # analysis_index -- minlength must still produce a full-length result.
    _write_group(tmp_path, 5e-8, [0, 1], with_offsets=False, n_analyses=3)
    counts = read_top_hit_counts(tmp_path, 3, thresholds=(5e-8,))
    assert counts["n_hits_5e8"] == [1, 1, 0]
