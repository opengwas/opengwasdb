"""Contracts for the #267 Indexed Variant Subset benchmark harness.

The harness publishes the production measurement, so the tests here pin the two
things a plausible-but-wrong result would hide: the equivalence comparison that
gates publication, and the artifact shape that must record every field #267
names. Byte accounting is tested on a synthetic tree because the physical count
is what the storage decision rests on.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from benchmarks import benchmark_indexed_subset as bench


def _expect_system_exit(action, needle: str) -> None:
    """Assert `action` raises SystemExit whose message names `needle`."""
    try:
        action()
    except SystemExit as exc:
        assert needle in str(exc), str(exc)
    else:
        raise AssertionError(f"expected SystemExit naming {needle!r}")


def _ordinary() -> dict[str, np.ndarray]:
    return {
        "variant_index": np.array([1, 4, 7, 9], dtype="int32"),
        "analysis_index": np.full(4, 3, dtype="int32"),
        "z": np.array([0.5, -1.25, np.nan, 2.0], dtype="float32"),
        "se": np.array([0.1, 0.2, 0.3, 0.4], dtype="float32"),
        "eaf": np.array([0.11, np.nan, 0.33, 0.44], dtype="float32"),
    }


def _indexed() -> dict[str, np.ndarray]:
    return {
        "variant_index": np.array([1, 7], dtype="int32"),
        "analysis_index": np.full(2, 3, dtype="int32"),
        "z": np.array([0.5, np.nan], dtype="float32"),
        "se": np.array([0.1, 0.3], dtype="float32"),
        "eaf": np.array([0.11, 0.33], dtype="float32"),
    }


def test_subset_ordinary_rows_filters_to_the_subset_in_store_order() -> None:
    subset = np.array([1, 7, 8], dtype="int64")
    filtered = bench.subset_ordinary_rows(_ordinary(), subset)

    assert filtered["variant_index"].tolist() == [1, 7]
    assert filtered["z"][0] == np.float32(0.5)
    assert np.isnan(filtered["z"][1])


def test_compare_indexed_to_ordinary_accepts_exact_nan_equality() -> None:
    result = bench.compare_indexed_to_ordinary(_ordinary(), _indexed(), np.array([1, 7]))

    assert result["exact"] is True
    assert result["expected_count"] == 2
    assert result["indexed_count"] == 2
    assert all(result["fields"].values())


def test_compare_indexed_to_ordinary_rejects_one_changed_value() -> None:
    indexed = _indexed()
    indexed["z"] = indexed["z"].copy()
    indexed["z"][0] = np.float32(0.5001)

    result = bench.compare_indexed_to_ordinary(_ordinary(), indexed, np.array([1, 7]))

    assert result["exact"] is False
    assert result["fields"]["z"] is False
    assert result["fields"]["se"] is True


def test_compare_indexed_to_ordinary_rejects_a_dropped_row() -> None:
    indexed = {name: values[:1] for name, values in _indexed().items()}

    result = bench.compare_indexed_to_ordinary(_ordinary(), indexed, np.array([1, 7]))

    assert result["exact"] is False
    assert result["indexed_count"] == 1
    assert result["expected_count"] == 2


def _write_bytes(path: Path, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)


def test_logical_bytes_sums_file_sizes_not_directories(tmp_path: Path) -> None:
    _write_bytes(tmp_path / "z" / "0", 100)
    _write_bytes(tmp_path / "se" / "0", 50)
    _write_bytes(tmp_path / "top", 7)

    assert bench.logical_bytes(tmp_path) == 157
    assert bench.physical_bytes(tmp_path) > 0


def test_plane_bytes_buckets_every_top_level_entry(tmp_path: Path) -> None:
    _write_bytes(tmp_path / "z" / "0", 100)
    _write_bytes(tmp_path / "se" / "0", 50)
    _write_bytes(tmp_path / ".zgroup", 1)

    buckets = bench.plane_bytes(tmp_path)

    assert set(buckets) == {"z", "se", ".zgroup", "total"}
    assert buckets["z"] == bench.physical_bytes(tmp_path / "z")
    assert buckets["se"] == bench.physical_bytes(tmp_path / "se")
    # total is the whole group, including its own directory blocks, so it can
    # exceed the sum of the children.
    assert buckets["total"] == bench.physical_bytes(tmp_path)
    assert buckets["total"] >= buckets["z"] + buckets["se"] + buckets[".zgroup"]
    assert buckets["z"] >= 100  # allocated, not apparent, bytes


def _complete_artifact() -> dict:
    return {
        "commit": "abcdef0",
        "measured_at": "2026-10-10T00:00:00+00:00",
        "opengwasdb_path": "/repo/opengwasdb",
        "opengwasdb_fingerprint": "f" * 64,
        "store": {
            "path": "/data/store",
            "store_id": "OGS-00009",
            "release_id": "OGS-00009",
            "format_version": "0.1.0",
            "reference_assembly": "hg38",
            "completion_state": "observed_only",
            "n_analyses": 2024,
        "n_variants": 9847701,
        "encoding": {"version": 3},
        },
        "input": {
            "variant_list_path": "/work/hm3.grch38.alid.txt",
            "variant_list_sha256": "a" * 64,
            "hapmap3_source_path": "/work/w_hm3.snplist.gz",
            "hapmap3_source_md5": "153ecc2bcfa740afafe656e6a384d769",
            "reference_assembly": "hg38",
            "requested": 3,
            "resolved": 2,
            "absent": 1,
            "hapmap3_resolution": {
                "rsids_requested": 10,
                "resolved_rsids": 8,
                "absent_from_store_axis": 1,
                "allele_incompatible": 1,
                "multiple_store_rows": 0,
                "duplicate_alids": 0,
            },
        },
        "storage": {
            "baseline_logical_bytes": 1,
            "baseline_physical_bytes": 1,
            "index_physical_bytes": 1,
            "index_logical_bytes": 1,
            "increase_percent": 100.0,
            "planes": {"z": 1, "se": 1, "eaf": 1, "total": 3},
        },
        "build": {
            "total_seconds": 1.0,
            "peak_rss_mb": 1.0,
            "band_cells": 4_000_000,
            "rss_bound_mb": 1.0,
            "rss_within_bound": True,
            "phases": {"prepare_seconds": 0.1, "publish_seconds": 0.9},
            "peak_rss_source": "/usr/bin/time -v",
        },
        "timings": {
            "ordinary_before": {
                "first_ms": 1.0,
                "median_ms": 2.0,
                "p95_ms": 3.0,
                "repetitions": 5,
                "result_count": 10,
            },
            "ordinary_after": {
                "first_ms": 1.0,
                "median_ms": 2.1,
                "p95_ms": 3.0,
                "repetitions": 5,
                "result_count": 10,
            },
            "indexed": {
                "first_ms": 0.5,
                "median_ms": 0.6,
                "p95_ms": 0.7,
                "repetitions": 5,
                "result_count": 2,
            },
        },
        "equivalence": {
            "exact": True,
            "fields": {"z": True, "se": True},
            "expected_count": 2,
            "indexed_count": 2,
        },
        "targets": {
            "warm_median_under_1s": True,
            "exact_equivalence": True,
            "ordinary_unchanged_within_envelope": True,
            "original_store_unchanged": True,
        },
        "publication": {"published": True, "reason": ""},
    }


def test_assert_artifact_complete_accepts_every_field_issue_267_names() -> None:
    bench.assert_artifact_complete(_complete_artifact())


def test_assert_artifact_complete_rejects_a_missing_field() -> None:
    artifact = _complete_artifact()
    del artifact["timings"]["ordinary_after"]

    _expect_system_exit(lambda: bench.assert_artifact_complete(artifact), "ordinary_after")


def test_assert_artifact_complete_rejects_a_missing_nested_field() -> None:
    artifact = _complete_artifact()
    del artifact["build"]["peak_rss_mb"]

    _expect_system_exit(lambda: bench.assert_artifact_complete(artifact), "peak_rss_mb")


def test_committed_production_artifact_records_a_complete_exact_measurement() -> None:
    """The committed #267 measurement satisfies the schema and the hard targets.

    The RSS bound is deliberately *not* asserted true: the production run measured
    a peak above the declared conservative bound and the artifact records that
    failure. Asserting it here would be the tuning the harness refuses.
    """
    path = (
        Path(__file__).resolve().parents[1]
        / "docs/benchmark-output/opengwasdb_267_indexed_subset_benchmark.json"
    )
    artifact = json.loads(path.read_text(encoding="utf-8"))

    bench.assert_artifact_complete(artifact)
    assert artifact["equivalence"]["exact"] is True
    assert artifact["targets"]["warm_median_under_1s"] is True
    assert artifact["targets"]["original_store_unchanged"] is True
    assert artifact["targets"]["ordinary_unchanged_within_envelope"] is True
    assert artifact["publication"]["published"] is True
