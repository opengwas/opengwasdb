"""Carried reader samples retain their provenance; probes never become a new revision A/B."""

from __future__ import annotations

import json

import pytest

from benchmarks._epic307_reader_cache import load_reader_evidence


def _inputs(tmp_path, rows=10):
    artifact = tmp_path / "old.json"
    readers = [
        {
            "label": label,
            "n_rows": 10,
            "samples": [
                {"side": side, "seconds": seconds}
                for side, seconds in (("parse_floor", 2), ("reader_base", 8))
            ],
            "medians": {"parse_floor": 2, "reader_base": 8},
        }
        for label in ("ukb-b-10001", "finngen_R13_AB1_ACTINOMYCOSIS")
    ]
    artifact.write_text(json.dumps({"commit": "old-head", "reader_ab": readers}))
    probes = tmp_path / "probes"
    probes.mkdir()
    (probes / "floors.txt").write_text(
        "".join(
            f"{i} {cache} {cap} load=7 {seconds} {rows}\n"
            for i, cache, seconds in ((1, "cold", 4), (1, "warm", 2), (2, "cold", 8))
            for cap in ("opengwasdb.gwas-vcf", "opengwasdb.finngen-r13")
        )
    )
    for name in ("control-ukb.txt", "control-finngen.txt"):
        (probes / name).write_text(
            f"1 cold reader_base source load=8 10 {rows}\n"
            f"1 warm reader_base source load=9 8 {rows}\n"
            "Fri 13 Nov 2026\n[exited with code 0]\n"
        )
    return artifact, probes, readers


def test_original_samples_are_unchanged_and_cache_samples_are_labelled_separately(tmp_path):
    artifact, probes, old = _inputs(tmp_path)
    before = artifact.read_bytes()

    readers, origin = load_reader_evidence(artifact, probes)

    assert artifact.read_bytes() == before
    assert origin["commit"] == "old-head"
    assert len(origin["sha256"]) == 64
    assert len(origin["cache_probe_files"]) == 3
    for reader, original in zip(readers, old, strict=True):
        assert reader["samples"] == original["samples"]
        assert reader["medians"] == original["medians"]
        assert reader["cache_probe"]["medians"]["parse_floor_cold"] == 6
        assert "causality is unresolved" in reader["cache_probe"]["note"]


def test_a_probe_of_a_different_source_is_refused(tmp_path):
    artifact, probes, _ = _inputs(tmp_path, rows=11)
    with pytest.raises(ValueError, match="row count differs"):
        load_reader_evidence(artifact, probes)


def test_the_harness_attaches_provenance_without_running_a_reader(tmp_path, monkeypatch):
    from benchmarks import _reader_ab
    from benchmarks.measure_epic307_evidence import main

    artifact, probes, _ = _inputs(tmp_path)
    output = tmp_path / "after.json"
    records = tmp_path / "store/records"
    records.mkdir(parents=True)
    (records / "build.json").write_text(
        json.dumps(
            {
                "step": "build",
                "store_id": "OGS-00009",
                "elapsed_seconds": 1,
            }
        )
    )

    def unexpected_sample(*args, **kwargs):
        pytest.fail("a carried artifact must not remeasure the source")

    monkeypatch.setattr(_reader_ab, "measure_source", unexpected_sample)
    assert (
        main(
            [
                "--reader-evidence",
                str(artifact),
                "--reader-cache-probes",
                str(probes),
                "--store",
                f"OGS-00009={records.parent}",
                "--tz",
                "Europe/London",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    payload = json.loads(output.read_text())
    assert payload["reader_evidence_origin"]["commit"] == "old-head"
    assert payload["reader_ab"][0]["cache_probe"]["medians"]["parse_floor_cold"] == 6
    floor = next(
        row for row in payload["comparison"]["figures"] if row["key"] == "reader_ukb_floor"
    )
    assert floor["detail"]["cold_samples"] == [4, 8]


def test_no_probe_is_not_an_empty_measurement(tmp_path):
    artifact, _, old = _inputs(tmp_path)
    readers, origin = load_reader_evidence(artifact)
    assert readers == old
    assert "cache_probe_files" not in origin
