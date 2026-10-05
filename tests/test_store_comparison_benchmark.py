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

import numpy as np
import pytest

from benchmarks.benchmark_store_comparison import (
    _parse_store,
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
