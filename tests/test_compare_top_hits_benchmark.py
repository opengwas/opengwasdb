"""The recovered #312 harness compares arrays, attributes and all three column bands."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from shutil import copytree

import numpy as np
import pytest
import zarr
from dense_sources import write_release_manifest

from benchmarks.compare_top_hits import main


def _index_store(path):
    root = zarr.open_group(path / "data.zarr", mode="w")
    root.create_array("z", shape=(1, 2050), chunks=(1, 1), shards=(1, 1024), dtype="int16")
    index = root.create_group("top_hits")
    index.attrs["thresholds"] = [5e-8, 5e-6, 5e-4]
    for tier, threshold in (("p_5e_08", 5e-8), ("p_5e_06", 5e-6), ("p_5e_04", 5e-4)):
        group = index.create_group(tier)
        group.attrs.update({"threshold": threshold, "order": "analysis_index,variant_index"})
        arrays = {
            "analysis_index": np.array([0, 1024, 2048], dtype=np.uint32),
            "analysis_offsets": np.searchsorted([0, 1024, 2048], np.arange(2051)),
            "variant_index": np.zeros(3, dtype=np.uint32),
            "z": np.array([6.0, 7.0, -8.0], dtype=np.float32),
            "abs_z": np.array([6.0, 7.0, 8.0], dtype=np.float32),
            "se": np.array([0.1, 0.2, 0.3], dtype=np.float32),
            "eaf": np.array([0.2, 0.4, 0.6], dtype=np.float32),
            "p_value": np.array([1e-9, 1e-10, 1e-12], dtype=np.float64),
        }
        for name, values in arrays.items():
            group.create_array(name, data=values)
    write_release_manifest(path, store_id="index-proof")
    return path


def _bytes(path):
    return {
        str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in path.rglob("*")
        if p.is_file()
    }


def test_read_only_harness_reproduces_attributes_arrays_and_three_band_counts(tmp_path):
    inline = _index_store(tmp_path / "inline")
    standalone = copytree(inline, tmp_path / "standalone")
    before = (_bytes(inline), _bytes(standalone))
    output = tmp_path / "proof.json"
    args = [str(inline), str(standalone), "--output", str(output)]

    assert main(args) == 0

    report = json.loads(output.read_text())
    assert report["identical"] and report["group_attrs_equal"]
    assert report["n_arrays"] == {"inline": 24, "standalone": 24}
    assert report["arrays_differing"] == []
    assert report["generator"] == "benchmarks/compare_top_hits.py"
    assert report["argv"] == args
    assert len(report["tiers"]) == 3
    for tier in report["tiers"].values():
        assert tier["hits"] == 3
        assert tier["hits_by_column_band"] == [1, 1, 1]
        assert tier["attrs_equal"]
    assert before == (_bytes(inline), _bytes(standalone))


@pytest.mark.parametrize("change", ["array", "group", "tier", "extra_tier"])
def test_harness_fails_for_each_kind_of_index_difference(tmp_path, change):
    inline = _index_store(tmp_path / "inline")
    standalone = copytree(inline, tmp_path / "standalone")
    root = zarr.open_group(standalone / "data.zarr", mode="r+")
    if change == "array":
        root["top_hits/p_5e_04/z"][0] = 9.0
    elif change == "group":
        root["top_hits"].attrs["thresholds"] = [5e-8]
    elif change == "tier":
        root["top_hits/p_5e_04"].attrs["order"] = "variant_index,analysis_index"
    else:
        root["top_hits"].create_group("extra_empty_tier")
    output = tmp_path / "proof.json"

    assert main([str(inline), str(standalone), "--output", str(output)]) == 1
    assert not json.loads(output.read_text())["identical"]


def test_full_size_evidence_identifies_the_recovered_harness_and_exact_invocation():
    repo = Path(__file__).resolve().parents[1]
    evidence = repo / "docs/benchmark-output/epic307_322"
    original = evidence / "top-hits-comparison.json"
    provenance_path = evidence / "top-hits-comparison-provenance.json"
    provenance = json.loads(provenance_path.read_text())
    harness = repo / provenance["committed_harness"]["path"]
    harness_sha = hashlib.sha256(harness.read_bytes()).hexdigest()
    artifact_sha = hashlib.sha256(original.read_bytes()).hexdigest()
    assert harness_sha == provenance["committed_harness"]["sha256"]
    assert artifact_sha == provenance["original_artifact"]["sha256"]
    report = json.loads(original.read_text())
    args = provenance["reproduction_argv"]
    assert args[:6] == [
        "pixi", "run", "-e", "dev", "python", "benchmarks/compare_top_hits.py"
    ]
    assert args[6:8] == [report["inline"], report["standalone"]]
    assert args[8] == "--output"
    assert not any(Path(args[9]).is_relative_to(Path(report[side]))
                   for side in ("inline", "standalone"))


def test_a_comparison_report_cannot_be_written_inside_either_input_store(tmp_path):
    inline = _index_store(tmp_path / "inline")
    standalone = copytree(inline, tmp_path / "standalone")
    with pytest.raises(SystemExit, match="outside both input Stores"):
        main([str(inline), str(standalone), "--output", str(standalone / "proof.json")])
