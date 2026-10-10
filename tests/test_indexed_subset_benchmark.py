"""Contracts for the #267 Indexed Variant Subset benchmark harness.

The harness publishes the production measurement, so the tests here pin the
things a plausible-but-wrong result would hide: the six-field equivalence gate,
the run-bracketing that makes before/after provable, the writer-vs-resolution
count reconciliation, and an artifact schema whose zeros and omissions fail
rather than publish. Byte accounting is tested on a synthetic tree, and the
committed QMD/HTML are searched for unevaluated inline expressions.
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
        "association_status": np.array(["observed"] * 4),
    }


def _indexed() -> dict[str, np.ndarray]:
    return {
        "variant_index": np.array([1, 7], dtype="int32"),
        "analysis_index": np.full(2, 3, dtype="int32"),
        "z": np.array([0.5, np.nan], dtype="float32"),
        "se": np.array([0.1, 0.3], dtype="float32"),
        "eaf": np.array([0.11, 0.33], dtype="float32"),
        "association_status": np.array(["observed"] * 2),
    }


def test_subset_ordinary_rows_filters_to_the_subset_in_store_order() -> None:
    subset = np.array([1, 7, 8], dtype="int64")
    filtered = bench.subset_ordinary_rows(_ordinary(), subset)

    assert filtered["variant_index"].tolist() == [1, 7]
    assert filtered["z"][0] == np.float32(0.5)
    assert np.isnan(filtered["z"][1])


def test_compare_indexed_to_ordinary_requires_all_six_fields() -> None:
    indexed = _indexed()
    del indexed["association_status"]

    result = bench.compare_indexed_to_ordinary(_ordinary(), indexed, np.array([1, 7]))

    assert result["exact"] is False
    assert result["required_fields_present"] is False
    assert result["required_fields"] == list(bench.REQUIRED_RESULT_FIELDS)


def test_compare_indexed_to_ordinary_accepts_exact_nan_equality() -> None:
    result = bench.compare_indexed_to_ordinary(_ordinary(), _indexed(), np.array([1, 7]))

    assert result["exact"] is True
    assert result["required_fields_present"] is True
    assert result["dtypes_match"] is True
    assert result["lengths_match"] is True
    assert result["non_empty"] is True
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


def test_compare_indexed_to_ordinary_rejects_a_dtype_change() -> None:
    indexed = _indexed()
    indexed["z"] = indexed["z"].astype("float64")

    result = bench.compare_indexed_to_ordinary(_ordinary(), indexed, np.array([1, 7]))

    assert result["exact"] is False
    assert result["dtypes_match"] is False


def test_compare_indexed_to_ordinary_rejects_a_dropped_row() -> None:
    indexed = {name: values[:1] for name, values in _indexed().items()}

    result = bench.compare_indexed_to_ordinary(_ordinary(), indexed, np.array([1, 7]))

    assert result["exact"] is False
    assert result["indexed_count"] == 1
    assert result["expected_count"] == 2


def test_compare_indexed_to_ordinary_rejects_an_empty_result() -> None:
    result = bench.compare_indexed_to_ordinary(_ordinary(), _indexed(), np.array([100]))

    assert result["exact"] is False
    assert result["non_empty"] is False


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


def test_scratch_copy_reports_the_verified_method(tmp_path: Path) -> None:
    """The copy helper names the method it actually used, not an assumption.

    On a filesystem that supports reflink this returns `reflink`; elsewhere it
    returns `full_copy`. Either way the destination is a real copy and an
    existing destination is refused rather than merged into.
    """
    from benchmarks._artifact import scratch_copy

    source = tmp_path / "source"
    source.mkdir()
    (source / "chunk").write_bytes(b"payload")
    destination = tmp_path / "scratch"

    method = scratch_copy(source, destination)

    assert method in {"reflink", "full_copy"}
    assert (destination / "chunk").read_bytes() == b"payload"
    _expect_system_exit(lambda: scratch_copy(source, destination), "already exists")


def _timing(median: float, count: int) -> dict:
    return {
        "first_ms": median,
        "first_read_semantics": bench.FIRST_READ_SEMANTICS,
        "median_ms": median,
        "p95_ms": median,
        "p95_method": bench.p95_method(5),
        "repetitions": 5,
        "result_count": count,
    }


def _complete_artifact() -> dict:
    fields = list(bench.REQUIRED_RESULT_FIELDS)
    reconciliation = {
        "writer_requested_equals_alids": True,
        "writer_resolved_equals_alids": True,
        "writer_absent_zero": True,
        "resolution_resolved_equals_alids": True,
        "rsid_budget_balances": True,
        "variant_list_sha256_equals_writer": True,
    }
    return {
        "artifact_schema_version": bench.ARTIFACT_SCHEMA_VERSION,
        "commit": "abcdef0",
        "measured_at": "2026-10-10T00:00:00+00:00",
        "opengwasdb_path": "/repo/opengwasdb",
        "opengwasdb_fingerprint": "f" * 64,
        "analysis_id": "ukb-b-17805",
        "subset_name": "hm3",
        "store": {
            "path": "/scratch/store-copy.opengwasdb",
            "authoritative_path": "/data/store.opengwasdb",
            "copy_kind": "reflink",
            "store_id": "OGS-00009",
            "release_id": "OGS-00009",
            "format_version": "0.1.0",
            "reference_assembly": "GRCh38",
            "completion_state": "observed_only",
            "layout": "dense",
            "n_analyses": 2024,
            "n_variants": 9847701,
            "encoding": {"version": 3},
        },
        "input": {
            "variant_list_path": "/work/hm3.grch38.alid.txt",
            "variant_list_sha256": "a" * 64,
            "hapmap3_source_path": "/work/w_hm3.snplist.gz",
            "hapmap3_source_md5": "153ecc2bcfa740afafe656e6a384d769",
            "reference_assembly": "GRCh38",
            "requested": 3,
            "resolved": 2,
            "absent": 1,
            "alids_derived": 2,
            "counts_reconciled": True,
            "counts_reconciliation": reconciliation,
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
            "run_mode": "fresh",
            "reused_from": "",
            "total_seconds": 1.0,
            "in_process_seconds": 1.0,
            "peak_rss_mb": 1.0,
            "band_cells": 4_000_000,
            "rss_bound_mb": 1.0,
            "rss_within_bound": True,
            "phases": {"prepare_seconds": 0.1, "publish_seconds": 0.9},
            "peak_rss_source": "/usr/bin/time -v",
            "requested": 3,
            "resolved": 2,
            "absent": 1,
            "input_sha256": "a" * 64,
        },
        "timings": {
            "method": {
                "first_read": bench.FIRST_READ_SEMANTICS,
                "p95": bench.p95_method(5),
            },
            "ordinary_before": _timing(2.0, 10),
            "ordinary_after": _timing(2.1, 10),
            "indexed": _timing(0.6, 2),
        },
        "equivalence": {
            "exact": True,
            "required_fields": fields,
            "required_fields_present": True,
            "dtypes_match": True,
            "lengths_match": True,
            "non_empty": True,
            "fields": {name: True for name in fields},
            "expected_count": 2,
            "indexed_count": 2,
        },
        "run": {
            "mode": "fresh",
            "started_at": "2026-10-10T00:00:00+00:00",
            "finished_at": "2026-10-10T00:01:00+00:00",
            "subset_present_during_ordinary_before": False,
            "subset_published_by_this_run": True,
            "ordinary_bracketed": True,
            "reset_subset": True,
        },
        "targets": {
            "warm_median_under_1s": True,
            "exact_equivalence": True,
            "ordinary_unchanged_within_envelope": True,
            "original_store_unchanged": True,
            "peak_rss_within_bound": True,
            "ordinary_bracketed": True,
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


def test_assert_artifact_complete_rejects_a_fabricated_zero_build() -> None:
    artifact = _complete_artifact()
    artifact["build"]["total_seconds"] = 0.0

    _expect_system_exit(lambda: bench.assert_artifact_complete(artifact), "build.total_seconds")


def test_assert_artifact_complete_rejects_unbalanced_writer_counts() -> None:
    artifact = _complete_artifact()
    artifact["input"]["requested"] = 4

    _expect_system_exit(lambda: bench.assert_artifact_complete(artifact), "input counts")


def test_assert_artifact_complete_rejects_an_unsix_field_equivalence_block() -> None:
    artifact = _complete_artifact()
    artifact["equivalence"]["required_fields"] = ["z"]

    _expect_system_exit(lambda: bench.assert_artifact_complete(artifact), "required_fields")


def test_assert_artifact_complete_rejects_bracketing_disagreement() -> None:
    artifact = _complete_artifact()
    artifact["run"]["ordinary_bracketed"] = False

    _expect_system_exit(lambda: bench.assert_artifact_complete(artifact), "ordinary_bracketed")


def test_assert_artifact_complete_rejects_fresh_run_with_subset_present() -> None:
    artifact = _complete_artifact()
    artifact["run"]["subset_present_during_ordinary_before"] = True

    _expect_system_exit(lambda: bench.assert_artifact_complete(artifact), "ordinary_before")


def test_assert_artifact_complete_rejects_rss_target_disagreement() -> None:
    artifact = _complete_artifact()
    artifact["targets"]["peak_rss_within_bound"] = False

    _expect_system_exit(
        lambda: bench.assert_artifact_complete(artifact), "peak_rss_within_bound"
    )


def test_reuse_without_build_stats_is_refused() -> None:
    assert bench.reuse_requires_build_stats(False, None) is None
    _expect_system_exit(
        lambda: bench.reuse_requires_build_stats(True, None), "build-stats"
    )
    assert bench.reuse_requires_build_stats(True, Path("/tmp/a.json")) == Path("/tmp/a.json")


def test_benchmark_measures_ordinary_before_before_the_build() -> None:
    """The pre-index baseline must run before the index is built.

    The artifact's `run.subset_present_during_ordinary_before` field documents
    the state, but only source order guarantees the measurement happened first;
    this pins that order so a future reshuffle cannot silently measure it after
    the build.
    """
    import inspect

    source = inspect.getsource(bench.benchmark)
    assert source.index("ordinary_before = timed") < source.index("run_build(copy")


def test_load_prior_build_rejects_a_zero_measurement(tmp_path: Path) -> None:
    art = _complete_artifact()
    art["build"]["total_seconds"] = 0.0
    path = tmp_path / "prior.json"
    path.write_text(json.dumps(art), encoding="utf-8")

    _expect_system_exit(lambda: bench.load_prior_build(path), "non-positive")


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
    assert artifact["targets"]["ordinary_bracketed"] is True
    assert artifact["publication"]["published"] is True
    assert artifact["analysis_id"] == "ukb-b-17805"
    assert artifact["subset_name"] == "hm3"


def _report_paths() -> tuple[Path, Path]:
    base = (
        Path(__file__).resolve().parents[1]
        / "docs/benchmark-output/opengwasdb_267_indexed_subset_benchmark"
    )
    return base.with_suffix(".qmd"), base.with_suffix(".html")


def test_report_source_has_no_inline_python_expressions() -> None:
    qmd, _ = _report_paths()
    source = qmd.read_text(encoding="utf-8").replace("```{python}", "")
    assert "{python}" not in source


def test_rendered_report_has_no_unevaluated_expressions() -> None:
    _, html_path = _report_paths()
    html = html_path.read_text(encoding="utf-8")
    assert "{python}" not in html
    # Quarto turned a straight-quoted f-string into a smart-quoted literal when
    # an inline expression failed to parse; neither marker may survive.
    assert "f”" not in html
    assert "f‘" not in html
