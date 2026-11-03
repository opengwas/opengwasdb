"""A build's Association Admission record (issue #357).

The record is written beside the Store Release: one part per Analysis while
the build runs, then `admission.json`. These tests cover the record on its own
and as each builder writes it: the same at one worker and at two, and its
counts the ones the Store's `info_score` provenance reports.
"""

from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

import besd_fixtures
import numpy as np
import pytest
import test_dense_vcf_build as dense_fixture
import test_info_score_hybrid_build as info_fixture
import test_ragged_build_ssf as ssf_fixture
from admission_helpers import ragged_ssf_store, vcf_store

from opengwasdb.admission import (
    ADMISSION_RULES,
    AdmissionRecord,
    AnalysisRules,
    Reason,
    RecordIncomplete,
    SourceRows,
    admission_record_for,
    counting_gate,
)
from opengwasdb.layouts.dense import DenseBuildResult
from opengwasdb.layouts.dense.build_vcf import build_dense_from_vcf_manifest
from opengwasdb.layouts.hybrid.build import build_hybrid_from_vcf_manifest
from opengwasdb.layouts.ragged import RaggedBuildResult
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
        # The source's one row with no effect is examined and dropped before the
        # declared filter, which counts the associations alone.
        assert counts["dropped"]["effect_missing"] == 1
        assert counts["examined"] == provenance["associations_observed"] + 1
        assert counts["info"]["observed"] == provenance["associations_observed"]
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
    dropped, counted and logged by ALID. 1:600000's beta is infinite, so its z
    is too, and it is dropped and logged as well (D4)."""
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
    assert (first["examined"], first["admitted"]) == (5, 2)
    assert first["dropped"] == {"se_unusable": 2, "z_nonfinite": 1}
    assert (second["examined"], second["admitted"], second["dropped"]) == (2, 2, {})
    assert _dropped(record, 0) == [
        ["source_key", "reason"],
        ["1:300000:C:T", "se_unusable"],
        ["1:400000:C:T", "se_unusable"],
        ["1:600000:C:T", "z_nonfinite"],
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


def test_a_vcf_record_with_no_effect_is_logged_by_its_source_coordinate(tmp_path: Path):
    """A GWAS-VCF record with neither EZ nor ES has no effect: the Dense build
    drops it, and its record names it as the reader spelt it."""
    rows = [
        f"1\t{dense_fixture.HG19_POS_1}\t.\tA\tG\t.\tPASS\t.\tES:SE\t2.0:0.5\n",
        f"1\t{dense_fixture.HG19_POS_2}\t.\tC\tT\t.\tPASS\t.\tES:SE\t.:0.3\n",
    ]
    store = _one_vcf_analysis(tmp_path, rows)
    record = admission_record_for(store)
    (analysis,) = _analyses(store)
    assert (analysis["examined"], analysis["admitted"]) == (2, 1)
    assert _dropped(record, 0)[1:] == [
        [f"1:{dense_fixture.HG19_POS_2}:C:T", "effect_missing"]
    ]


def test_a_dense_build_from_records_writes_its_record(tmp_path: Path):
    """Each Analysis's records pass its gate, in the order the Analyses first
    appear, which is the order of the Store's columns."""

    source = tmp_path / "records.tsv"
    source.write_text(
        "analysis_id\tchromosome\tposition\teffect_allele\tother_allele\tbeta\tse\n"
        "B\t1\t100\tA\tG\t0.2\t0.1\n"
        "A\t1\t100\tA\tG\t0.3\t0.1\n"
        "B\t1\t200\tC\tT\t0.1\t0.05\n",
        encoding="utf-8",
    )
    store, _ = _dense_from_records(tmp_path, source)

    written = json.loads(
        (admission_record_for(store) / "admission.json").read_text(encoding="utf-8")
    )
    assert written["command"] == "build-dense"
    assert [(a["analysis_id"], a["examined"], a["admitted"]) for a in written["analyses"]] == [
        ("B", 2, 2),
        ("A", 1, 1),
    ]


def test_a_vcf_record_with_an_unusable_se_is_dropped_and_logged(tmp_path: Path):
    """D3: a GWAS-VCF record whose SE is NaN or infinite used to be stored, and
    one whose SE is 0 or missing skipped in silence. Each is now dropped and
    logged as `se_unusable`."""
    positions = (
        dense_fixture.HG19_POS_1,
        dense_fixture.HG19_POS_2,
        dense_fixture.HG19_POS_3,
    )
    rows = [
        f"1\t{positions[0]}\t.\tA\tG\t.\tPASS\t.\tEZ:SE\t2.0:nan\n",
        f"1\t{positions[1]}\t.\tC\tT\t.\tPASS\t.\tEZ:SE\t1.5:inf\n",
        f"1\t{positions[2]}\t.\tG\tA\t.\tPASS\t.\tEZ:SE\t1.0:0\n",
    ]
    vcf = dense_fixture._make_vcf(tmp_path, "trait_a", rows)
    good = dense_fixture._make_vcf(
        tmp_path, "trait_b", [f"1\t{positions[0]}\t.\tA\tG\t.\tPASS\t.\tES:SE\t2.0:0.5\n"]
    )
    manifest = dense_fixture._make_manifest(
        tmp_path, [("trait_a", vcf, "Trait A"), ("trait_b", good, "Trait B")]
    )
    store = tmp_path / "dense.opengwasdb"
    build_dense_from_vcf_manifest(manifest, store, store_id="s", release_id="r")

    record = admission_record_for(store)
    first, _ = json.loads((record / "admission.json").read_text(encoding="utf-8"))["analyses"]
    assert (first["examined"], first["admitted"], first["dropped"]) == (3, 0, {"se_unusable": 3})
    assert [row[1] for row in _dropped(record, 0)[1:]] == ["se_unusable"] * 3


def test_a_dense_record_with_an_unusable_se_is_dropped_not_raised(tmp_path: Path):
    """D3 for `build-dense`: an SE of 0 beside a given z used to be stored, and
    a missing or NaN SE raised; each is now dropped and logged."""

    source = tmp_path / "records.tsv"
    source.write_text(
        "analysis_id\tchromosome\tposition\teffect_allele\tother_allele\tz\tse\n"
        "A\t1\t100\tA\tG\t2.0\t0\n"
        "A\t1\t200\tA\tG\t2.0\tNA\n"
        "A\t1\t300\tA\tG\t2.0\t0.1\n",
        encoding="utf-8",
    )
    store, result = _dense_from_records(tmp_path, source)
    record = admission_record_for(store)
    (analysis,) = _analyses(store)
    assert (analysis["examined"], analysis["admitted"]) == (3, 1)
    assert _dropped(record, 0)[1:] == [
        ["1:100:A:G", "se_unusable"],
        ["1:200:A:G", "se_unusable"],
    ]
    assert result.n_variants == 1


def test_a_reader_association_keeps_the_se_its_source_gave():
    """An unusable SE is the source's, as read: Admission, not the reader's
    record type, decides the row is not an association."""
    from opengwasdb.model.enums import StoredEffectScale
    from opengwasdb.readers.interface import ReaderAssociation

    association = ReaderAssociation("1", 100, "A", "G", 1.0, -0.5, StoredEffectScale.SD)
    assert association.se == -0.5


def test_source_fidelity_compares_only_the_rows_admission_stored(tmp_path: Path):
    """A source row Admission dropped is not in the Store. Validating the Store
    against its source skips the cell, missing in the Store, rather than
    report a disagreement: B holds the variant A's dropped row names."""
    from opengwasdb.validation import validate_store

    source = tmp_path / "records.tsv"
    source.write_text(
        "analysis_id\tchromosome\tposition\teffect_allele\tother_allele\tz\tse\n"
        "A\t1\t100\tA\tG\t2.0\tNA\n"
        "A\t1\t200\tA\tG\t2.0\t0.1\n"
        "B\t1\t100\tA\tG\t1.0\t0.2\n",
        encoding="utf-8",
    )
    store, _ = _dense_from_records(tmp_path, source)

    result = validate_store(store, source=source)
    assert result.ok, result.errors


def test_a_row_with_an_infinite_z_is_dropped_and_logged_not_refused(tmp_path: Path):
    """D4: an infinite z used to stop `build-dense` (and the encoder, on a
    tabular source); one row's statistic does not refuse its Analysis, so it is
    dropped and logged."""

    source = tmp_path / "records.tsv"
    source.write_text(
        "analysis_id\tchromosome\tposition\teffect_allele\tother_allele\tz\tse\n"
        "A\t1\t100\tA\tG\tinf\t0.1\n"
        "A\t1\t200\tA\tG\t-inf\t0.1\n"
        "A\t1\t300\tA\tG\t2.0\t0.1\n",
        encoding="utf-8",
    )
    store, _ = _dense_from_records(tmp_path, source)

    record = admission_record_for(store)
    assert _dropped(record, 0)[1:] == [
        ["1:100:A:G", "z_nonfinite"],
        ["1:200:A:G", "z_nonfinite"],
    ]


def _gate_admitting_everything(monkeypatch: pytest.MonkeyPatch) -> None:
    """A gate with a defect: it keeps every row it is handed."""
    from opengwasdb.admission import Gate

    monkeypatch.setattr(Gate, "source_rows", lambda self, rows: np.ones(len(rows), dtype=bool))


def test_a_writer_refuses_a_half_missing_pair_that_passed_its_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """D1: spec §15 forbids a cell with one of z and SE missing. A source row
    cannot become one through a working gate, so one at the writer is a defect
    and refused, naming where."""
    from opengwasdb.admission import NotAdmitted
    from opengwasdb.build.observed import build_dense_observed_from_sources

    source = tmp_path / "records.tsv"
    source.write_text(
        "analysis_id\tchromosome\tposition\teffect_allele\tother_allele\tz\tse\n"
        "A\t1\t100\tA\tG\t2.0\tNA\n",
        encoding="utf-8",
    )
    _gate_admitting_everything(monkeypatch)

    with pytest.raises(NotAdmitted, match=r"build-dense: .*one of z and SE missing: 1"):
        build_dense_observed_from_sources(
            [source], tmp_path / "dense.opengwasdb", store_id="s", release_id="r",
            reference_assembly="GRCh38",
        )


def test_a_dense_vcf_writer_refuses_an_unusable_se_that_passed_its_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from opengwasdb.admission import NotAdmitted

    rows = [f"1\t{dense_fixture.HG19_POS_1}\t.\tA\tG\t.\tPASS\t.\tEZ:SE\t2.0:nan\n"]
    vcf = dense_fixture._make_vcf(tmp_path, "trait_a", rows)
    manifest = dense_fixture._make_manifest(tmp_path, [("trait_a", vcf, "Trait A")])
    _gate_admitting_everything(monkeypatch)

    with pytest.raises(NotAdmitted, match=r"build-dense-vcf, Analysis trait_a: .*z and SE"):
        build_dense_from_vcf_manifest(
            manifest, tmp_path / "dense.opengwasdb", store_id="s", release_id="r"
        )


def test_a_dense_vcf_build_collapses_agreeing_duplicates_and_drops_conflicting_ones(tmp_path: Path):
    """D5: two source keys reaching one row (a liftover collision) are one
    association when they agree, collapsed to the first; when they disagree,
    neither is stored, and both are logged by the variant's ALID."""
    hg19 = dense_fixture.HG19_POS_1
    rows = [
        f"1\t{hg19}\t.\tA\tG\t.\tPASS\t.\tES:SE\t2.0:0.5\n",
        f"1\t{hg19}\t.\tA\tG\t.\tPASS\t.\tES:SE\t2.0:0.5\n",
        f"1\t{dense_fixture.HG19_POS_2}\t.\tC\tT\t.\tPASS\t.\tES:SE\t1.5:0.3\n",
        f"1\t{dense_fixture.HG19_POS_2}\t.\tC\tT\t.\tPASS\t.\tES:SE\t-1.5:0.3\n",
    ]
    store = _one_vcf_analysis(tmp_path, rows)
    record = admission_record_for(store)
    (analysis,) = _analyses(store)
    assert analysis["dropped"] == {"duplicate_collapsed": 1, "duplicate_conflicting": 2}
    assert (analysis["examined"], analysis["admitted"]) == (4, 1)
    (conflict,) = {row[0] for row in _dropped(record, 0)[1:]}
    assert conflict.endswith(":C:T")


def test_a_dense_build_resolves_duplicate_records_rather_than_raising(tmp_path: Path):
    """D5 for `build-dense`, which used to refuse any duplicate."""

    source = tmp_path / "records.tsv"
    source.write_text(
        "analysis_id\tchromosome\tposition\teffect_allele\tother_allele\tz\tse\n"
        "A\t1\t100\tA\tG\t2.0\t0.1\n"
        "A\t1\t100\tA\tG\t2.0\t0.1\n"
        "A\t1\t200\tA\tG\t1.0\t0.1\n"
        "A\t1\t200\tA\tG\t3.0\t0.1\n",
        encoding="utf-8",
    )
    store, result = _dense_from_records(tmp_path, source)
    (analysis,) = _analyses(store)
    assert analysis["dropped"] == {"duplicate_collapsed": 1, "duplicate_conflicting": 2}
    assert result.n_variants == 1


def test_the_hybrid_off_reference_spill_resolves_its_duplicates():
    """D5 for associations no axis holds: a raw key's agreeing rows collapse to
    the first, and a conflicting key's rows are all dropped and logged."""
    from opengwasdb.layouts.hybrid.build import _off_reference_spill
    from opengwasdb.variants.routing import Unmatched

    unmatched = Unmatched(
        ["1:5:A:G", "1:5:A:G", "1:9:C:T", "1:9:C:T", "2:4:A:C"],
        np.array([1.0, 1.0, 2.0, -2.0, 3.0], dtype=np.float32),
        np.full(5, 0.1, dtype=np.float32),
        np.full(5, np.nan, dtype=np.float32),
    )
    gate = counting_gate(AnalysisRules("A"))
    keys, z, *_ = _off_reference_spill(unmatched, 1.0, gate)

    assert z.tolist() == [1.0, 3.0]
    assert len(keys) == 2
    assert gate.counts.dropped == {Reason.DUPLICATE_COLLAPSED: 1, Reason.DUPLICATE_CONFLICTING: 2}
    assert gate.dropped == [
        ("1:9:C:T", Reason.DUPLICATE_CONFLICTING),
        ("1:9:C:T", Reason.DUPLICATE_CONFLICTING),
    ]


def _analyses(store: Path) -> list[dict]:
    """A Store's Admission record, one entry per Analysis."""
    text = (admission_record_for(store) / "admission.json").read_text(encoding="utf-8")
    analyses: list[dict] = json.loads(text)["analyses"]
    return analyses


def _one_vcf_analysis(tmp_path: Path, rows: list[str]) -> Path:
    """A Dense VCF Store of one Analysis, `trait_a`, from these VCF records."""
    vcf = dense_fixture._make_vcf(tmp_path, "trait_a", rows)
    manifest = dense_fixture._make_manifest(tmp_path, [("trait_a", vcf, "Trait A")])
    store = tmp_path / "dense.opengwasdb"
    build_dense_from_vcf_manifest(manifest, store, store_id="s", release_id="r")
    return store


def _dense_from_records(tmp_path: Path, source: Path) -> tuple[Path, DenseBuildResult]:
    """A `build-dense` Store from one records file, and the build's result."""
    from opengwasdb.build.observed import build_dense_observed_from_sources

    store = tmp_path / "dense.opengwasdb"
    result = build_dense_observed_from_sources(
        [source], store, store_id="s", release_id="r", reference_assembly="GRCh38"
    )
    return store, result


def test_a_single_pass_dense_build_logs_the_rows_its_reference_lacks(tmp_path: Path):
    """D6: a row whose variant the Variant Reference does not hold reaches no
    row of the Store; it used to be counted nowhere, and is now logged."""
    vcf = dense_fixture._make_vcf(
        tmp_path,
        "trait_a",
        [
            "1\t1000\t.\tA\tG\t.\tPASS\t.\tES:SE\t2.0:0.5\n",
            "1\t2000\t.\tC\tT\t.\tPASS\t.\tES:SE\t1.0:0.5\n",
        ],
    )
    manifest = dense_fixture._make_manifest(tmp_path, [("trait_a", vcf, "Trait A")])
    reference = tmp_path / "reference.txt"
    reference.write_text("1:1000:A:G\n", encoding="utf-8")
    store = tmp_path / "dense.opengwasdb"
    build_dense_from_vcf_manifest(
        manifest, store, store_id="s", release_id="r",
        source_assembly="hg38", variant_reference=reference,
    )

    (analysis,) = _analyses(store)
    assert (analysis["admitted"], analysis["dropped"]) == (1, {"off_axis": 1})
    assert _dropped(admission_record_for(store), 0)[1:] == [["1:2000:C:T", "off_axis"]]


@pytest.mark.parametrize("folds", [True, False])
def test_a_hybrid_drops_an_unmatched_row_only_where_nothing_folds_it(
    monkeypatch: pytest.MonkeyPatch, folds: bool
):
    """D6: a single-pass Hybrid folds an unmatched row onto its axis later, so
    it keeps the row; a two-pass one has nowhere to put it, so it counts and
    logs it rather than spill it for nobody to read."""
    from types import SimpleNamespace

    from opengwasdb.layouts.hybrid import build as hybrid_build
    from opengwasdb.model.enums import StoredEffectScale
    from opengwasdb.readers.fake import FakeReader
    from opengwasdb.readers.interface import ReaderAssociation
    from opengwasdb.variants.routing import RouteTable, RoutingKey

    rows = [
        ReaderAssociation("1", 100, "A", "G", 2.0, 0.1, StoredEffectScale.SD),
        ReaderAssociation("1", 900, "C", "T", 1.0, 0.1, StoredEffectScale.SD),
    ]
    monkeypatch.setattr(
        hybrid_build, "open_source", lambda *args, **kwargs: FakeReader(associations=rows)
    )
    table = RouteTable.build(
        {("1", 100, "A", "G"): "1:100:A:G"}, (), [["1:100:A:G"], ["1:100:A:G"]],
        key=RoutingKey.CASE_FOLDED, keep_unmatched=True,
    )
    gate = counting_gate(AnalysisRules("A"))
    row = SimpleNamespace(source=None, se_divisor=1.0, trait_id="A")

    dense, _, off_reference = hybrid_build._resolve_column_hybrid(
        table, row, gate, (["1:100:A:G"], ["1:100:A:G"]), folds
    )

    assert dense[0].tolist() == [0]
    assert len(off_reference[0]) == (1 if folds else 0)
    assert gate.counts.dropped == ({} if folds else {Reason.OFF_AXIS: 1})
    assert gate.dropped == ([] if folds else [("1:900:C:T", Reason.OFF_AXIS)])


def test_a_dense_vcf_build_applies_a_declared_info_filter(tmp_path: Path):
    """D2: the declared INFO threshold used to filter only a Hybrid build; a
    Dense build of the same GWAS-SSF source now drops the same row."""
    source = info_fixture._write_source(
        tmp_path / "GCST_INFO.tsv.gz", [*info_fixture.SCORED_ROWS, info_fixture.NO_EFFECT_ROW]
    )
    manifest = info_fixture._manifest(tmp_path, source, "0.7")
    store = tmp_path / "dense.opengwasdb"
    build_dense_from_vcf_manifest(manifest, store, store_id="s", release_id="r")

    (analysis,) = _analyses(store)
    assert analysis["dropped"] == {"effect_missing": 1, "info_below_threshold": 1}
    assert analysis["info"]["below_threshold"] == 1


def _ragged_ssf_with_frequencies(
    tmp_path: Path, frequencies: list[str], maf_threshold: str | None = None
) -> tuple[Path, RaggedBuildResult]:
    """A Ragged SSF Store of one Analysis, `trait_a`, one row per frequency
    cell at 1:100000, 1:200000, ..., under a declared MAF floor if one is given."""
    rows = [
        {"chromosome": "1", "base_pair_location": 100_000 * (i + 1), "effect_allele": "A",
         "other_allele": "G", "beta": 1.0, "standard_error": 0.5,
         "effect_allele_frequency": af}
        for i, af in enumerate(frequencies)
    ]
    return ragged_ssf_store(tmp_path, rows, maf_threshold)


def test_a_ragged_ssf_build_applies_a_declared_maf_filter(tmp_path: Path):
    """D2 for `build-ragged-ssf`, which used to ignore a declared MAF floor."""
    store, result = _ragged_ssf_with_frequencies(tmp_path, ["0.3", "0.001"], "0.01")

    (analysis,) = _analyses(store)
    assert analysis["dropped"] == {"maf_below_threshold": 1}
    assert result.n_associations == 1


@pytest.mark.parametrize(
    ("column", "value", "match"),
    [
        ("maf_threshold", "0.01", "--store-esi-eaf"),
        ("info_score_threshold", "0.6", "INFO policy"),
    ],
)
def test_a_besd_build_refuses_a_filter_it_cannot_judge(
    tmp_path: Path, column: str, value: str, match: str
):
    """D2: a BESD source has no imputation score, and its only frequency is the
    `.esi`'s, which the build reads only with `store_esi_eaf`."""
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    besd_fixtures.write_esi(
        fixture / "test.esi",
        [{"chr": "1", "snp_id": "rs1", "bp": 1_000_000, "a1": "A", "a2": "G"}],
    )
    besd_fixtures.write_epi(fixture / "test.epi", [{"chr": "1", "probe_id": "ENSG1", "bp": 1}])
    besd_fixtures.write_besd_sparse_3f(fixture / "test.besd", 1, [[(0, 0.1, 0.02)]])
    overlay = tmp_path / "analyses.tsv"
    overlay.write_text(f"analysis_id\t{column}\nENSG1\t{value}\n", encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        build_ragged_from_besd(
            fixture / "test", tmp_path / "besd.opengwasdb", store_id="s", release_id="r",
            association_coverage="cis_and_signals", analyses_path=overlay,
        )


_AF_ROWS = ["0.3", "1", "0", "1.5", "."]


def _vcf_with_frequencies(tmp_path: Path) -> Path:
    """One Analysis's VCF whose FORMAT AF is each of `_AF_ROWS`: three reported
    frequencies outside (0, 1), and one the source left missing."""
    header = dense_fixture._vcf_header("Continuous").replace(
        "##FORMAT=<ID=EZ",
        '##FORMAT=<ID=AF,Number=A,Type=Float,Description="Allele frequency">\n##FORMAT=<ID=EZ',
    )
    rows = "".join(
        f"1\t{1000 * (i + 1)}\t.\tA\tG\t.\tPASS\t.\tES:SE:AF\t1.0:0.5:{af}\n"
        for i, af in enumerate(_AF_ROWS)
    )
    vcf = tmp_path / "trait_a.vcf"
    vcf.write_text(header + rows, encoding="utf-8")
    return dense_fixture._make_manifest(tmp_path, [("trait_a", vcf, "Trait A")])


@pytest.mark.parametrize("hybrid", [False, True])
def test_a_vcf_build_counts_the_frequencies_it_makes_absent(tmp_path: Path, hybrid: bool):
    """D7: a frequency outside (0, 1) was already absent in a Store, and nothing
    said how many; the Admission record now counts them, and keeps the rows."""
    panel = [f"1:{1000 * (i + 1)}:A:G" for i in range(len(_AF_ROWS))] if hybrid else None
    store = vcf_store(tmp_path, _vcf_with_frequencies(tmp_path), panel)

    (analysis,) = _analyses(store)
    assert (analysis["examined"], analysis["admitted"]) == (5, 5)
    assert analysis["eaf_made_absent"] == 3


def test_a_ragged_ssf_build_counts_the_frequencies_it_makes_absent(tmp_path: Path):
    store, _ = _ragged_ssf_with_frequencies(tmp_path, ["0.3", "1.0", "1.5", "NA"])

    (analysis,) = _analyses(store)
    assert (analysis["admitted"], analysis["eaf_made_absent"]) == (4, 2)


def test_a_dense_build_stores_no_frequency_outside_the_interval(tmp_path: Path):
    """D7 for `build-dense`, which used to store a record's frequency of 1 or
    1.2 as given: they are absent now, and counted, and the Analysis whose only
    frequencies they were reports none."""
    source = tmp_path / "records.tsv"
    source.write_text(
        "analysis_id\tchromosome\tposition\teffect_allele\tother_allele\tz\tse\teaf\n"
        "A\t1\t100\tA\tG\t2.0\t0.1\t0.3\n"
        "A\t1\t200\tA\tG\t2.0\t0.1\t1.2\n"
        "B\t1\t100\tA\tG\t1.0\t0.1\t1\n"
        "B\t1\t200\tA\tG\t1.0\t0.1\t0\n",
        encoding="utf-8",
    )
    store, _ = _dense_from_records(tmp_path, source)

    assert [(a["admitted"], a["eaf_made_absent"]) for a in _analyses(store)] == [(2, 1), (2, 2)]
    with (store / "analyses.tsv").open(encoding="utf-8") as fh:
        rows = csv.DictReader(fh, delimiter="\t")
        scopes = {row["analysis_id"]: row["eaf_scope"] for row in rows}
    assert scopes == {"A": "association", "B": "absent"}


def test_an_esi_frequency_outside_the_interval_is_counted_once_for_the_file(
    caplog: pytest.LogCaptureFixture,
):
    """A `.esi` frequency is a variant's, shared by every probe, so D7's count
    for a BESD source is the file's, logged once, not any Analysis's."""
    from opengwasdb.layouts.ragged.besd_eaf import stored_esi_frequencies
    from opengwasdb.layouts.ragged.besd_reader import SnpRecord

    snps = [
        SnpRecord(i, "1", f"rs{i}", 100 * (i + 1), "A", "G", freq)
        for i, freq in enumerate([0.3, 1.0, 1.5, None])
    ]
    with caplog.at_level("INFO", logger="opengwasdb.layouts.ragged.besd_eaf"):
        stored = stored_esi_frequencies(snps, {i: (i, False) for i in range(4)})

    assert stored is not None
    assert np.isnan(stored[1:]).all()
    assert ".esi: 2 reported frequencies outside (0, 1) are absent" in caplog.messages


def test_a_declared_info_filter_s_drops_are_counted_and_not_logged(tmp_path: Path):
    """D8: a row a declared filter drops is reproducible from the source and the
    policy, so it is counted only. The log names the rows dropped for any other
    reason, and only those."""
    store = _hybrid(tmp_path, _three_analysis_manifest(tmp_path), 1)

    record = admission_record_for(store)
    for index, analysis in enumerate(_analyses(store)):
        assert analysis["dropped"]["info_below_threshold"] > 0
        assert [reason for _, reason in _dropped(record, index)[1:]] == ["effect_missing"]


def test_a_declared_maf_filter_s_drops_are_counted_and_not_logged(tmp_path: Path):
    """D8 for both of the MAF filter's reasons: a frequency below the floor, and
    none at all."""
    store, _ = _ragged_ssf_with_frequencies(tmp_path, ["0.3", "0.001", "NA"], "0.01")

    (analysis,) = _analyses(store)
    assert analysis["dropped"] == {"maf_missing": 1, "maf_below_threshold": 1}
    assert not (admission_record_for(store) / "dropped" / "0.tsv.gz").exists()
