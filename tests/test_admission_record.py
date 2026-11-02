"""A build's Association Admission record (issue #357).

The record is written beside the Store Release: one part per Analysis while
the build runs, then `admission.json`. These tests cover the record on its own
and as each builder writes it: the same at one worker and at two, and its
counts the ones the Store's `info_score` provenance reports.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import besd_fixtures
import numpy as np
import pytest
import test_dense_vcf_build as dense_fixture
import test_info_score_hybrid_build as info_fixture
import test_ragged_build_ssf as ssf_fixture

from opengwasdb.admission import (
    ADMISSION_RULES,
    AdmissionRecord,
    AnalysisRules,
    Reason,
    RecordIncomplete,
    SourceRows,
    admission_record_for,
)
from opengwasdb.layouts.dense.build_vcf import build_dense_from_vcf_manifest
from opengwasdb.layouts.hybrid.build import build_hybrid_from_vcf_manifest
from opengwasdb.layouts.ragged.build_besd import build_ragged_from_besd
from opengwasdb.layouts.ragged.build_ssf import build_ragged_from_ssf
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


def _dense_vcf(tmp_path: Path, n_workers: int) -> Path:
    """Two Analyses' GWAS-VCF sources, three rows and two, built Dense."""
    rows = [
        f"1\t{position}\t.\t{ref}\t{alt}\t.\tPASS\t.\tES:SE\t{beta}:{se}\n"
        for position, ref, alt, beta, se in (
            (dense_fixture.HG19_POS_1, "A", "G", 2.0, 0.5),
            (dense_fixture.HG19_POS_2, "C", "T", 1.5, 0.3),
            (dense_fixture.HG19_POS_3, "G", "A", 0.6, 0.2),
        )
    ]
    sources = tmp_path / "sources"
    sources.mkdir(exist_ok=True)
    manifest = dense_fixture._make_manifest(
        sources,
        [
            ("trait_a", dense_fixture._make_vcf(sources, "trait_a", rows), "Trait A"),
            ("trait_b", dense_fixture._make_vcf(sources, "trait_b", rows[1:]), "Trait B"),
        ],
    )
    store = tmp_path / f"dense-{n_workers}.opengwasdb"
    build_dense_from_vcf_manifest(
        manifest, store, store_id="s", release_id="r", n_workers=n_workers
    )
    return store


def test_a_dense_vcf_build_writes_its_record(tmp_path: Path):
    one, two = _dense_vcf(tmp_path, 1), _dense_vcf(tmp_path, 2)

    record = admission_record_for(one) / "admission.json"
    written = json.loads(record.read_text(encoding="utf-8"))
    assert written["command"] == "build-dense-vcf"
    assert [(a["analysis_id"], a["examined"], a["admitted"]) for a in written["analyses"]] == [
        ("trait_a", 3, 3),
        ("trait_b", 2, 2),
    ]
    assert record.read_bytes() == (admission_record_for(two) / "admission.json").read_bytes()


def _dropped(record: Path, index: int) -> list[list[str]]:
    """One Analysis's logged drops, header first."""
    with gzip.open(record / "dropped" / f"{index}.tsv.gz", "rt", encoding="utf-8") as fh:
        return [line.rstrip("\n").split("\t") for line in fh]


def test_a_ragged_ssf_build_logs_each_unusable_se(tmp_path: Path):
    """The SE of 1:300000 does not parse and that of 1:400000 is 0: both are
    dropped, counted and logged by ALID. 1:600000's beta is not finite, so it
    has no effect at all and is not an association Admission judges."""
    manifest, filtered = ssf_fixture._make_fixture(tmp_path)
    store = tmp_path / "ssf.opengwasdb"
    build_ragged_from_ssf(
        manifest, filtered, store, store_id="s", release_id="r",
        association_coverage="signals_only",
    )

    record = admission_record_for(store)
    # The gzip header's mtime (bytes 4-8) is zero, so the same drops give the
    # same bytes whenever they are written.
    assert (record / "dropped" / "0.tsv.gz").read_bytes()[4:8] == bytes(4)
    written = json.loads((record / "admission.json").read_text(encoding="utf-8"))
    assert written["command"] == "build-ragged-ssf"
    first, second = written["analyses"]
    assert (first["examined"], first["admitted"], first["dropped"]) == (4, 2, {"se_unusable": 2})
    assert (second["examined"], second["admitted"], second["dropped"]) == (2, 2, {})
    assert _dropped(record, 0) == [
        ["source_key", "reason"],
        ["1:300000:C:T", "se_unusable"],
        ["1:400000:C:T", "se_unusable"],
    ]
    assert not (record / "dropped" / "1.tsv.gz").exists()


def test_a_ragged_besd_build_logs_each_unusable_se_by_esi_row(tmp_path: Path):
    """Every raw BESD row is judged before the ESI map: the SE of ESI row 1 is
    negative and that of row 4 is not finite."""
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    snps: list[besd_fixtures.EsiRow] = [
        {"chr": "1", "snp_id": f"rs{i}", "bp": 1_000_000 + 100_000 * i, "a1": "A", "a2": "G"}
        for i in range(5)
    ]
    besd_fixtures.write_esi(fixture / "test.esi", snps)
    besd_fixtures.write_epi(
        fixture / "test.epi",
        [{"chr": "1", "probe_id": f"ENSG{i}", "bp": 1_050_000} for i in range(2)],
    )
    besd_fixtures.write_besd_sparse_3f(
        fixture / "test.besd",
        2,
        [[(0, 0.1, 0.02), (1, -0.2, -0.03)], [(2, 0.5, 0.05), (4, 0.3, float("inf"))]],
    )
    store = tmp_path / "besd.opengwasdb"
    result = build_ragged_from_besd(
        fixture / "test", store, store_id="s", release_id="r",
        association_coverage="cis_and_signals",
    )

    assert result.n_associations == 2
    record = admission_record_for(store)
    written = json.loads((record / "admission.json").read_text(encoding="utf-8"))
    assert [(a["examined"], a["admitted"]) for a in written["analyses"]] == [(2, 1), (2, 1)]
    assert _dropped(record, 0)[1:] == [["esi:1", "se_unusable"]]
    assert _dropped(record, 1)[1:] == [["esi:4", "se_unusable"]]
