"""A build's Association Admission record (issue #357).

The record is written beside the Store Release: one part per Analysis while
the build runs, then `admission.json`. These tests cover the record on its own
and as a Hybrid build writes it: the same at one worker and at two, and its
counts the ones the Store's `info_score` provenance reports.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import test_info_score_hybrid_build as info_fixture

from opengwasdb.admission import (
    ADMISSION_RULES,
    AdmissionRecord,
    AnalysisRules,
    Reason,
    RecordIncomplete,
    SourceRows,
    admission_record_for,
)
from opengwasdb.layouts.hybrid.build import build_hybrid_from_vcf_manifest
from opengwasdb.model.maf_policy import MafPolicy, MafState
from opengwasdb.readers.interface import ImputationScoreStatus

FILTERED_MAF = MafPolicy(0.05, MafState.FILTERED)


def _rows(af: list[float]) -> SourceRows:
    return SourceRows(
        eaf=np.asarray(af, dtype=np.float64),
        imputation_score=np.full(len(af), np.nan),
        imputation_status=np.array([ImputationScoreStatus.UNDECLARED] * len(af), dtype=object),
    )


def _judge(record: AdmissionRecord, analysis_id: str, af: list[float]) -> None:
    with record.gate(AnalysisRules(analysis_id, maf=FILTERED_MAF)) as gate:
        gate.source_rows(_rows(af))


def test_finish_writes_each_analysis_in_the_order_the_record_was_opened(tmp_path: Path):
    record = AdmissionRecord.open(tmp_path / "rec", ["B", "A"], command="build-test")
    _judge(record, "A", [0.3, 0.01])
    _judge(record, "B", [0.3, np.nan, 0.2])

    summary = record.finish()

    written = json.loads((tmp_path / "rec" / "admission.json").read_text(encoding="utf-8"))
    assert written["rules"] == ADMISSION_RULES
    assert written["command"] == "build-test"
    assert [a["analysis_id"] for a in written["analyses"]] == ["B", "A"]
    assert written["analyses"][0]["dropped"] == {"maf_missing": 1}
    assert written["analyses"][1]["dropped"] == {"maf_below_threshold": 1}
    assert summary.counts["A"].admitted == 1
    assert summary.counts["B"].dropped == {Reason.MAF_MISSING: 1}


def test_finish_refuses_an_analysis_without_a_part(tmp_path: Path):
    record = AdmissionRecord.open(tmp_path / "rec", ["A", "B"], command="build-test")
    _judge(record, "A", [0.3])

    with pytest.raises(RecordIncomplete, match=r"1 Analysis.*\['B'\]"):
        record.finish()
    assert not (tmp_path / "rec" / "admission.json").exists()


def test_a_gate_that_raises_writes_no_part(tmp_path: Path):
    record = AdmissionRecord.open(tmp_path / "rec", ["A"], command="build-test")
    with pytest.raises(RuntimeError, match="source failed"):
        with record.gate(AnalysisRules("A")) as gate:
            gate.source_rows(_rows([0.3]))
            raise RuntimeError("source failed")

    with pytest.raises(RecordIncomplete):
        record.finish()


def test_resume_keeps_the_parts_and_a_fresh_open_discards_them(tmp_path: Path):
    first = AdmissionRecord.open(tmp_path / "rec", ["A", "B"], command="build-test")
    _judge(first, "A", [0.3])
    _judge(first, "B", [0.3])

    resumed = AdmissionRecord.open(tmp_path / "rec", ["A", "B"], command="build-test", resume=True)
    assert resumed.finish().counts["B"].examined == 1

    fresh = AdmissionRecord.open(tmp_path / "rec", ["A", "B"], command="build-test")
    assert not (tmp_path / "rec" / "admission.json").exists()
    with pytest.raises(RecordIncomplete):
        fresh.finish()


def test_a_part_written_for_another_manifest_is_refused(tmp_path: Path):
    first = AdmissionRecord.open(tmp_path / "rec", ["A"], command="build-test")
    _judge(first, "A", [0.3])

    other = AdmissionRecord.open(tmp_path / "rec", ["Z"], command="build-test", resume=True)
    with pytest.raises(RecordIncomplete, match="another manifest"):
        other.finish()


def _three_analysis_manifest(tmp_path: Path) -> Path:
    """Three Analyses with an INFO threshold that drops rows in each."""
    lines: list[str] = []
    for i, analysis_id in enumerate(("GCST_A", "GCST_B", "GCST_C")):
        directory = tmp_path / analysis_id
        directory.mkdir()
        rows = [*info_fixture.SCORED_ROWS[i:], info_fixture.NO_EFFECT_ROW]
        source = info_fixture._write_source(directory / f"{analysis_id}.tsv.gz", rows)
        manifest = info_fixture._manifest(directory, source, "0.7", analysis_id)
        text = manifest.read_text(encoding="utf-8")
        header, row = text.splitlines()
        lines = lines or [header]
        lines.append(row)
    manifest = tmp_path / "analyses.tsv"
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def _hybrid(tmp_path: Path, manifest: Path, n_workers: int) -> Path:
    store = tmp_path / f"store-{n_workers}.opengwasdb"
    build_hybrid_from_vcf_manifest(
        manifest,
        store,
        reference_panel=info_fixture._panel(tmp_path),
        store_id="s",
        release_id="r",
        n_workers=n_workers,
    )
    return store


def test_a_hybrid_record_is_the_same_at_one_and_two_workers(tmp_path: Path):
    manifest = _three_analysis_manifest(tmp_path)
    one, two = _hybrid(tmp_path, manifest, 1), _hybrid(tmp_path, manifest, 2)

    record = admission_record_for(one) / "admission.json"
    written = json.loads(record.read_text(encoding="utf-8"))
    # Fixture first: every Analysis dropped rows, so equal records are not two
    # records of nothing.
    assert all(a["dropped"].get("info_below_threshold", 0) > 0 for a in written["analyses"])
    assert record.read_bytes() == (admission_record_for(two) / "admission.json").read_bytes()


def test_a_hybrid_record_holds_the_counts_its_provenance_reports(tmp_path: Path):
    store = _hybrid(tmp_path, _three_analysis_manifest(tmp_path), 2)
    written = json.loads(
        (admission_record_for(store) / "admission.json").read_text(encoding="utf-8")
    )

    provenances = info_fixture._info_analyses(store)
    for counts, provenance in zip(written["analyses"], provenances, strict=True):
        assert counts["analysis_id"] == provenance["analysis_id"]
        assert counts["examined"] == provenance["associations_observed"]
        assert counts["admitted"] == provenance["associations_retained"]
        assert counts["dropped"]["info_below_threshold"] == (
            provenance["associations_below_threshold"]
        )
