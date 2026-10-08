"""Tests for the store-comparison harness's identity check and footprint walker.

The harness is the instrument epic #240 measures with, so its two silent-failure
surfaces get unit coverage: the identity comparison must fail on a changed
value, a moved NaN and a reordered result (and not on an equal one), and the
footprint walker must separate Zarr array bytes from the non-Zarr envelope on
both the v2 (`.zarray`) and v3 (`zarr.json`) layouts.
"""

from __future__ import annotations

import argparse
import json
import signal
import time

import numpy as np
import pytest

from benchmarks.benchmark_store_comparison import (
    DEFAULT_EXPOSURE,
    DEFAULT_REGION,
    _check_memory_counts,
    _common_shapes,
    _dense_plane_root,
    _parse_region,
    _parse_store,
    _parser,
    _ShapeTimeout,
    _timed_shape,
    _wait_for_quiet,
    differing_arrays,
    differing_shapes,
    digest_array,
    effective_reader_settings,
    footprint,
    result_digests,
)
from opengwasdb.store.arrays import ArrayRole, create_array, open_group_for_write


def _result(
    z: list[float], eaf: list[float] | None = None
) -> dict[str, np.ndarray]:
    n = len(z)
    return {
        "variant_index": np.arange(n, dtype="int32"),
        "analysis_index": np.zeros(n, dtype="int32"),
        "z": np.asarray(z, dtype="float32"),
        "se": np.full(n, 0.5, dtype="float32"),
        "eaf": np.asarray(eaf if eaf is not None else [0.3] * n, dtype="float32"),
        "association_status": np.array(["observed"] * n, dtype=object),
    }


def test_identical_results_differ_in_nothing():
    reference = {"bulk": result_digests(_result([1.0, 2.0, 3.0]))}
    candidate = {"bulk": result_digests(_result([1.0, 2.0, 3.0]))}

    assert differing_arrays(reference["bulk"], candidate["bulk"]) == []
    assert differing_shapes(reference, candidate) == {}


def test_changed_value_is_detected():
    reference = result_digests(_result([1.0, 2.0, 3.0]))
    candidate = result_digests(_result([1.0, 2.0, 4.0]))

    assert differing_arrays(reference, candidate) == ["z"]


def test_moved_nan_is_detected():
    reference = result_digests(_result([1.0, 2.0, 3.0], eaf=[0.1, np.nan, 0.3]))
    candidate = result_digests(_result([1.0, 2.0, 3.0], eaf=[0.1, 0.3, np.nan]))

    assert differing_arrays(reference, candidate) == ["eaf"]


def test_reordered_values_are_detected():
    reference = result_digests(_result([1.0, 2.0, 3.0]))
    candidate = result_digests(_result([1.0, 3.0, 2.0]))

    assert differing_arrays(reference, candidate) == ["z"]


def test_differing_shapes_names_the_shape_and_arrays():
    reference = {
        "bulk": result_digests(_result([1.0, 2.0])),
        "phewas": result_digests(_result([5.0])),
    }
    candidate = {
        "bulk": result_digests(_result([1.0, 9.0])),
        "phewas": result_digests(_result([5.0])),
    }

    assert differing_shapes(reference, candidate) == {"bulk": ["z"]}


def test_array_present_on_one_side_only_is_detected():
    reference = result_digests(_result([1.0, 2.0]))
    extra = _result([1.0, 2.0])
    extra["future_array"] = np.array([7], dtype="int64")
    candidate = result_digests(extra)

    assert "future_array" in candidate  # every returned array is hashed
    assert differing_arrays(reference, candidate) == ["future_array"]


def test_differing_extra_array_is_detected():
    reference_result = _result([1.0])
    reference_result["future_array"] = np.array([1, 2], dtype="int64")
    candidate_result = _result([1.0])
    candidate_result["future_array"] = np.array([1, 3], dtype="int64")

    reference = result_digests(reference_result)
    candidate = result_digests(candidate_result)

    assert differing_arrays(reference, candidate) == ["future_array"]


def test_equal_nans_with_different_payloads_are_equal():
    # A NaN's payload bits are not its meaning; two stores may decode the same
    # missing cell to different NaN bit patterns. Only the positions must agree.
    plain = np.array([np.nan, 1.0], dtype="float32")
    payload = np.frombuffer(bytes.fromhex("0100c07f"), dtype="float32")  # NaN payload 1
    disguised = np.array([payload[0], 1.0], dtype="float32")

    assert plain.tobytes() != disguised.tobytes()
    assert digest_array(plain) == digest_array(disguised)


def test_result_digests_rejects_a_result_missing_an_array():
    result = _result([1.0])
    del result["eaf"]

    with pytest.raises(SystemExit, match="missing array"):
        result_digests(result)


def test_footprint_separates_v2_arrays_from_the_envelope(tmp_path):
    store = tmp_path / "store.opengwasdb"
    array = store / "data.zarr" / "z"
    array.mkdir(parents=True)
    (store / "data.zarr" / ".zgroup").write_text('{"zarr_format": 2}')
    (array / ".zarray").write_text("{}")
    (array / "0.0").write_bytes(b"x" * 1000)
    (store / "variants.tsv.gz").write_bytes(b"v" * 500)
    (store / "index.sqlite").write_bytes(b"s" * 100)

    result = footprint(store)

    assert result["n_files"] == 5
    assert [row["path"] for row in result["envelope"]["files"]] == [
        "index.sqlite",
        "variants.tsv.gz",
    ]
    nodes = {row["node"]: row for row in result["arrays"]}
    assert nodes["data.zarr/z"]["n_files"] == 2  # .zarray plus one chunk
    assert nodes["data.zarr/z"]["apparent_bytes"] >= 1000
    # The largest file is the chunk, not the 2-byte .zarray: this is the shard
    # size #246 reports, and an average over the array could hide a large one.
    assert nodes["data.zarr/z"]["largest_file_bytes"] == 1000
    assert result["zarr_metadata"]["n_files"] == 1  # data.zarr/.zgroup
    assert result["apparent_bytes"] == sum(
        row["apparent_bytes"] for row in result["arrays"]
    ) + result["zarr_metadata"]["apparent_bytes"] + result["envelope"]["apparent_bytes"]


def test_footprint_reads_zarr_v3_node_types(tmp_path):
    store = tmp_path / "v3-store"
    array = store / "data.zarr" / "z"
    array.mkdir(parents=True)
    (store / "data.zarr" / "zarr.json").write_text(json.dumps({"node_type": "group"}))
    (array / "zarr.json").write_text(json.dumps({"node_type": "array"}))
    (array / "c").mkdir()
    (array / "c" / "0.0").write_bytes(b"y" * 1000)
    (store / "manifest.json").write_bytes(b"m" * 100)

    result = footprint(store)

    nodes = {row["node"]: row for row in result["arrays"]}
    assert list(nodes) == ["data.zarr/z"]
    assert nodes["data.zarr/z"]["n_files"] == 2  # array zarr.json plus inner shard
    assert nodes["data.zarr/z"]["largest_file_bytes"] == 1000  # the shard, not zarr.json
    assert [row["path"] for row in result["envelope"]["files"]] == ["manifest.json"]
    # The group's own zarr.json is Zarr metadata, not envelope.
    assert result["zarr_metadata"]["n_files"] == 1


def test_effective_reader_settings_reports_the_pinned_configuration(tmp_path):
    """The artifact must record what ran, not what the code intended.

    A benchmark under any other reader configuration is not comparable (#244,
    #253), so the values are asserted, not merely present.
    """
    root = open_group_for_write(tmp_path / "data.zarr", "w")
    create_array(
        root,
        "z",
        ArrayRole.DENSE_STATISTIC_PLANE,
        shape=(4, 4),
        dtype="int16",
        fill_value=-1,
    )
    assert effective_reader_settings(root) == {
        "use_threads": True,
        "pipeline": "FusedCodecPipeline",
        "max_workers": 1,
    }


def test_store_arg_requires_a_label_and_a_path():
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_store("/data/opengwasdb/stores/OGS-00009/store.opengwasdb")


def test_parse_region_accepts_chrom_start_end():
    assert _parse_region("10:112500000-113500000") == ("10", 112_500_000, 113_500_000)


@pytest.mark.parametrize("text", ["10", "10:1", "10:abc-2", "10:5-5", "10:9-3"])
def test_parse_region_refuses_a_malformed_or_empty_window(text):
    """A malformed region must fail, not silently time an empty window (#250)."""
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_region(text)


def test_anchors_default_to_the_ogs00009_selection():
    """No override must reproduce the committed OGS-00009 anchors exactly."""
    args = _parser().parse_args(
        ["--store", "v2=/data/opengwasdb/stores/OGS-00009/store.opengwasdb"]
    )
    assert args.exposure == DEFAULT_EXPOSURE == "ukb-b-17805"
    assert args.phewas_alid is None
    assert args.region == DEFAULT_REGION == ("19", 44_500_000, 45_500_000)


def test_parser_accepts_a_store_specific_anchor_override():
    """OGS-00016/OGS-00011 cannot resolve the ukb-b anchors, so they override them.

    Without these options the harness refuses those stores; the parser is the
    user-facing half of that fix (#250).
    """
    args = _parser().parse_args(
        [
            "--store",
            "v2=/data/opengwasdb/stores/OGS-00016/store.opengwasdb",
            "--exposure",
            "finngen-r13-T2D",
            "--phewas-alid",
            "10:112998590:C:T",
            "--region",
            "10:112500000-113500000",
        ]
    )
    assert args.exposure == "finngen-r13-T2D"
    assert args.phewas_alid == "10:112998590:C:T"
    assert args.region == ("10", 112_500_000, 113_500_000)


def test_timed_shape_records_a_limit_hit_without_a_digest(monkeypatch):
    """A shape over the limit must be recorded as a hit, never as a timing (#250).

    `setitimer` raises `_ShapeTimeout` on the arm call and does nothing on the
    clear, so the test exercises the handler and its restoration without waiting
    on a clock or delivering a real signal.
    """
    calls = {"n": 0}

    def arm_then_raise(_which: int, _seconds: float) -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            raise _ShapeTimeout

    monkeypatch.setattr(signal, "setitimer", arm_then_raise)

    timed = _timed_shape(lambda: _result([1.0]), 5, limit_s=1.0, slow_shape_s=0.0)

    assert timed["timed_out"] is True
    assert timed["digests"] == {}
    assert timed["result_count"] is None


def test_timed_shape_times_a_slow_shape_once_after_its_warm_up(monkeypatch):
    """#252 timed an over-threshold shape once; the harness must match it.

    The clock is stubbed rather than slept on: the warm-up spans 5 s, so the
    shape is timed once even though `reps` is 5.
    """
    calls: list[int] = []
    ticks = iter([0.0, 5.0, 10.0, 20.0])
    monkeypatch.setattr(time, "perf_counter", lambda: next(ticks))

    def shape() -> dict[str, np.ndarray]:
        calls.append(1)
        return _result([1.0, 2.0])

    timed = _timed_shape(shape, 5, limit_s=0.0, slow_shape_s=1.0)

    assert timed["timed_out"] is False
    assert timed["repetitions"] == 1
    assert len(calls) == 2  # one warm-up plus one timed call
    assert timed["result_count"] == 2


def test_timed_shape_defaults_to_reps_and_digests_the_warm_up():
    """No limit and no slow threshold must keep the committed `_median_ms` shape."""
    calls: list[int] = []

    def shape() -> dict[str, np.ndarray]:
        calls.append(1)
        return _result([1.0, 2.0])

    timed = _timed_shape(shape, 3, limit_s=0.0, slow_shape_s=0.0)

    assert timed["timed_out"] is False
    assert timed["repetitions"] == 3
    assert len(calls) == 4  # one warm-up plus three timed calls
    assert timed["digests"]["z"] == digest_array(np.asarray([1.0, 2.0], dtype="float32"))


def test_common_shapes_excludes_a_shape_one_store_did_not_measure():
    """A timed-out shape must drop out of the identity set, not compare as equal.

    This is the narrowing that lets an artifact say `identical: true` only about
    the shapes every store actually measured (#250 review r1, minor 11).
    """
    common, measured = _common_shapes([{"a": {}, "b": {}}, {"a": {}, "c": {}}])

    assert common == {"a"}
    assert measured == {"a", "b", "c"}
    assert _common_shapes([]) == (set(), set())


def test_check_memory_counts_skips_a_timed_out_probe_but_catches_a_mismatch():
    _check_memory_counts(
        [{"query": "a", "timed_out": True}], [{"query": "a", "timed_out": False, "result_count": 5}]
    )
    with pytest.raises(SystemExit):
        _check_memory_counts(
            [{"query": "a", "timed_out": False, "result_count": 2}],
            [{"query": "a", "timed_out": False, "result_count": 1}],
        )


def test_dense_plane_root_reads_dense_and_hybrid_and_refuses_nothing():
    dense_root = {"z": object()}
    assert _dense_plane_root(type("Q", (), {"_root": dense_root})()) is dense_root

    inner = {"z": object()}
    hybrid = type("H", (), {"_dense": type("D", (), {"_root": inner})()})()
    assert _dense_plane_root(hybrid) is inner

    with pytest.raises(SystemExit):
        _dense_plane_root(type("N", (), {})())


def test_effective_reader_settings_tolerates_a_zarr2_plane():
    """zarr-python 2 has no `_async_array`; the 2.18 column must not crash (#250)."""

    class _Zarr2Plane:
        chunks = (1,)

    record = effective_reader_settings({"z": _Zarr2Plane()})

    assert record["pipeline"] is None
    assert record["max_workers"] is None


def test_wait_for_quiet_returns_immediately_when_disabled_or_below_the_limit():
    assert _wait_for_quiet(0) == 0.0
    assert _wait_for_quiet(10_000) < 1.0
