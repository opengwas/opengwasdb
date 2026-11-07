"""The association TSV Source Reader, and `build-dense` reading through it."""

from __future__ import annotations

import gzip
import math
from pathlib import Path

import numpy as np
import pytest
from dense_sources import write_dense_manifest
from typer.testing import CliRunner, Result

from opengwasdb.cli.main import app
from opengwasdb.model.enums import StoredEffectScale
from opengwasdb.readers import ReferenceAllele
from opengwasdb.readers.association_tsv import (
    ASSOCIATION_TSV_CAPABILITY,
    AssociationTsvReader,
    SourceRowError,
)
from opengwasdb.readers.interface import StreamTally
from opengwasdb.readers.registry import known_capabilities, resolve_reader

#: Associations in each of the shapes the format reads: aliased columns, a flipped
#: effect allele, an effect from beta, a missing, zero and unusable SE, lower-case
#: alleles, an indel, and a frequency the source reports outside (0, 1). The
#: format declares no reference allele, so the indel names no ALID (ADR 0079).
CASES = (
    "chrom\tpos\tea\toa\tz\tbeta\tstandard_error\teffect_allele_frequency\tvariant_id\n"
    "1\t100\tA\tG\t2.0\t\t0.1\t0.3\trs1\n"
    "1\t200\tT\tC\t3.0\t\t0.2\t0.1\trs2\n"
    "2\t300\tG\tA\t\t0.6\t0.2\t0.25\trs3\n"
    "2\t400\tC\tT\tNA\t-0.4\t0.1\t.\trs4\n"
    "3\t500\ta\tg\t1.5\t\tNA\t\t\n"
    "3\t600\tG\tC\t\t0.5\t0\t0.5\trs6\n"
    "4\t700\tAT\tA\t-2.0\t\t0.3\t0.0\trs7\n"
    "X\t800\tG\tT\t4.0\t\t0.5\t1.5\trs8\n"
    "chr5\t900\tC\tG\tinf\t\t0.5\t0.2\trs9\n"
)

#: (alid, z, se, eaf, rsid): what the combined-TSV loop `build-dense` had before it
#: read through a Source Reader (00e91b1's `read_normalised_associations`) gave for
#: `CASES`, with one difference in where a rule lives: that loop passed a
#: frequency outside (0, 1) on and the build made it absent (D7), where the reader
#: now makes it absent as every reader does; and since #381 the indel `4:700`,
#: which the loop stored sorted as `4:700:A:AT`, is dropped and named.
EXPECTED = (
    ("1:100:A:G", 2.0, 0.1, 0.3, "rs1"),
    ("1:200:C:T", -3.0, 0.2, 0.9, "rs2"),
    ("2:300:A:G", -2.9999999999999996, 0.2, 0.75, "rs3"),
    ("2:400:C:T", -4.0, 0.1, None, "rs4"),
    ("3:500:A:G", 1.5, math.nan, None, ""),
    ("3:600:C:G", math.nan, 0.0, 0.5, "rs6"),
    ("X:800:G:T", 4.0, 0.5, None, "rs8"),
    ("5:900:C:G", math.inf, 0.5, 0.2, "rs9"),
)


def _same(a: float | None, b: float | None) -> bool:
    if a is None or b is None:
        return a is b
    return a == b or (math.isnan(a) and math.isnan(b))


@pytest.fixture
def cases(tmp_path: Path) -> Path:
    path = tmp_path / "cases.tsv"
    path.write_text(CASES, encoding="utf-8")
    return path


def _reader(path: Path) -> AssociationTsvReader:
    reader = resolve_reader(ASSOCIATION_TSV_CAPABILITY, path, StoredEffectScale.SD)
    assert isinstance(reader, AssociationTsvReader)
    return reader


def test_the_registry_resolves_the_association_tsv_capability(cases: Path) -> None:
    assert isinstance(_reader(cases), AssociationTsvReader)
    assert ASSOCIATION_TSV_CAPABILITY == "opengwasdb.association-tsv"
    assert ASSOCIATION_TSV_CAPABILITY in known_capabilities()


def test_known_capabilities_grew_by_this_one_and_no_other() -> None:
    assert set(known_capabilities()) == {
        "opengwasdb.association-tsv",
        "opengwasdb.besd",
        "opengwasdb.column-map",
        "opengwasdb.finngen-r13",
        "opengwasdb.gwas-ssf",
        "opengwasdb.gwas-vcf",
    }


def test_the_reader_yields_what_the_combined_loop_yielded_field_for_field(cases: Path) -> None:
    reader = _reader(cases)
    tally = StreamTally()
    (batch,) = reader.stream_batches(tally)
    assert len(batch) == len(EXPECTED)
    for i, (alid, z, se, eaf, rsid) in enumerate(EXPECTED):
        assert batch.alid[i] == alid
        assert _same(float(batch.z[i]), z), (alid, batch.z[i], z)
        assert _same(float(batch.se[i]), se), (alid, batch.se[i], se)
        assert _same(None if np.isnan(batch.eaf[i]) else float(batch.eaf[i]), eaf), alid
        assert batch.rsid[i] == rsid
    # The frequency it made absent is counted for the Admission record, and the
    # indel is named, as Admission logs it.
    assert tally.frequencies.made_absent == 1
    assert tally.faults.rows == [("4:700:A:AT", "indel_reference_undeclared")]

    associations = list(reader.stream_associations())
    assert [(a.chromosome, a.position, a.ref, a.alt) for a in associations][:3] == [
        ("1", 100, "G", "A"), ("1", 200, "C", "T"), ("2", 300, "A", "G"),
    ]
    assert {a.stored_effect_scale for a in associations} == {StoredEffectScale.SD}
    assert {a.reference for a in associations} == {ReferenceAllele.UNDECLARED}
    for association, (_alid, z, se, eaf, _rsid) in zip(associations, EXPECTED, strict=True):
        assert _same(association.z, z)
        assert _same(association.se, se)
        assert _same(association.eaf, eaf)


def test_a_gzipped_comma_delimited_file_reads_the_same(tmp_path: Path) -> None:
    path = tmp_path / "cases.csv.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(CASES.replace("\t", ","))
    (batch,) = _reader(path).stream_batches(StreamTally())
    assert batch.alid.tolist() == [alid for alid, *_ in EXPECTED]


@pytest.mark.parametrize(
    "header",
    [
        "analysis_id", "study_id", "trait_id", "analysis_label", "trait", "license",
        "stored_effect_scale", "effect_scale", "maf_threshold", "info_score_threshold",
        "first_author", "trait_ontology_id",
    ],
)
def test_a_column_the_manifest_owns_is_refused_not_ignored(tmp_path: Path, header: str) -> None:
    path = tmp_path / "declares.tsv"
    path.write_text(
        f"{header}\tchromosome\tposition\teffect_allele\tother_allele\tz\tse\n"
        "A\t1\t100\tA\tG\t2.0\t0.1\n",
        encoding="utf-8",
    )
    with pytest.raises(SourceRowError, match=f"declares {header}.*build manifest"):
        list(_reader(path).stream_associations())


def _build_dense(tmp_path: Path, source: Path, store: Path) -> Result:
    """`build-dense` through the CLI, of the Analyses ``source`` holds."""
    return CliRunner().invoke(
        app,
        [
            "build-dense", str(store), "--manifest", str(write_dense_manifest(source)),
            "--store-id", "s", "--release-id", "r", "--admission-record", str(tmp_path / "adm"),
        ],
    )


_HEADER = "chromosome\tposition\teffect_allele\tother_allele\tz\tse\n"


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ("1\t\tA\tG\t2.0\t0.1", "row 3: missing required field 'position'"),
        ("1\t100\tA\tG\t2.0\tabc", "row 3: invalid se 'abc'"),
        ("1\t100\tA\tG\t\t0.1", "row 3: missing required field 'beta'"),
        ("1\t100\tA\tA\t2\t0.1", "row 3: effect and other alleles are identical"),
        ("1\tx\tA\tG\t2\t0.1", "row 3: invalid position 'x'"),
    ],
)
def test_build_dense_refuses_a_run_with_an_unreadable_row(
    tmp_path: Path, row: str, message: str
) -> None:
    """The rows of one Analysis's file that cannot be read refuse the run, with
    the message the combined loop gave, and write no Store."""
    source = tmp_path / "associations.tsv"
    source.write_text(
        "analysis_id\t" + _HEADER + "A\t1\t50\tA\tG\t1\t0.1\n" + f"A\t{row}\n",
        encoding="utf-8",
    )
    store = tmp_path / "store.opengwasdb"
    result = _build_dense(tmp_path, source, store)
    assert result.exit_code == 1
    assert isinstance(result.exception, SourceRowError)
    assert str(result.exception) == message
    assert not store.exists()


def test_a_malformed_identity_is_dropped_and_logged_through_build_dense(tmp_path: Path) -> None:
    """A row whose identity no ALID can spell is not an unreadable row: Admission
    drops it and logs it, and the run goes on."""
    import json

    source = tmp_path / "associations.tsv"
    source.write_text(
        "analysis_id\t" + _HEADER + "A\t1\t100\tA\tG\t2.0\t0.1\nA\t1\t200\tA\tN\t2.0\t0.1\n",
        encoding="utf-8",
    )
    store = tmp_path / "store.opengwasdb"
    result = _build_dense(tmp_path, source, store)
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["n_variants"] == 1
    written = json.loads((tmp_path / "adm" / "admission.json").read_text(encoding="utf-8"))
    (analysis,) = written["analyses"]
    assert (analysis["examined"], analysis["admitted"]) == (2, 1)
    assert analysis["dropped"] == {"allele_invalid": 1}


def test_without_a_tally_a_malformed_identity_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "malformed.tsv"
    path.write_text(_HEADER + "1\t200\tA\tN\t2.0\t0.1\n", encoding="utf-8")
    with pytest.raises(SourceRowError, match="row 2: "):
        list(_reader(path).stream_associations())


def test_a_file_with_no_header_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "empty.tsv"
    path.write_text("", encoding="utf-8")
    with pytest.raises(SourceRowError, match="has no header"):
        list(_reader(path).stream_associations())
