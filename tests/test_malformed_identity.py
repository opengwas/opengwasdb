"""A malformed identity is dropped and logged, the same on every reader (#365).

A row whose chromosome is missing, or whose allele lies outside A, C, G and T,
names no variant an ALID can spell. Before #365 the GWAS-VCF build streams
validated no allele and `normalise_chromosome` kept `NA` and `.` as contigs, so
such a row became an ALID such as `NA:2000:A:G`; the tabular readers dropped
the allele rows unseen, and `build-dense` refused the source. Now every reader
leaves the row out of both of its streams and names it, and Association
Admission counts and logs it under `chromosome_missing` or `allele_invalid`.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Callable
from pathlib import Path

import besd_fixtures
import pytest
import test_dense_vcf_build as dense_fixture
from admission_helpers import ragged_ssf_store, vcf_store

from opengwasdb.admission import admission_record_for
from opengwasdb.layouts.ragged.build_besd import build_ragged_from_besd
from opengwasdb.readers.finngen import FinnGenR13Reader
from opengwasdb.readers.gwas_ssf import GwasSsfReader
from opengwasdb.readers.gwas_vcf import GwasVcfReader
from opengwasdb.readers.interface import SourceReader, StreamTally
from opengwasdb.variants.axis import iter_variant_records
from opengwasdb.variants.normalise import (
    IdentityFaults,
    InvalidAllele,
    MissingChromosome,
    allele_fault,
    identity_fault,
    normalise_allele,
    normalise_chromosome,
    orient_or_drop,
)

#: ``(chromosome, position, ref, alt)``: one well-formed row, then one of each
#: malformation.
ROWS = [
    ("1", 1000, "A", "G"),
    ("NA", 2000, "A", "G"),
    ("1", 3000, "N", "G"),
    ("1", 4000, "A", "*"),
]
#: Each malformed row's source key and reason, in source order.
DROPPED = [
    ["NA:2000:A:G", "chromosome_missing"],
    ["1:3000:N:G", "allele_invalid"],
    ["1:4000:A:*", "allele_invalid"],
]


@pytest.mark.parametrize("spelling", ["NA", "na", ".", "", "  ", "NaN", "None", "chrNA"])
def test_a_missing_value_token_is_no_chromosome(spelling: str):
    with pytest.raises(MissingChromosome):
        normalise_chromosome(spelling)


@pytest.mark.parametrize(
    ("contig", "name"),
    [
        ("11_KI270927v1_alt", "11_KI270927v1_alt"),
        ("chr1_KI270706v1_random", "1_KI270706v1_random"),
        ("chrUn_gl000220", "Un_gl000220"),
        ("XY", "XY"),
    ],
)
def test_an_alt_or_random_contig_is_a_chromosome(contig: str, name: str):
    """The hg19 lift's alt and random contigs are #376's, not this rule's."""
    assert normalise_chromosome(contig) == name


@pytest.mark.parametrize("allele", ["N", "*", ".", "<DEL>", "-", "", "AN"])
def test_an_allele_outside_the_alphabet_is_invalid(allele: str):
    with pytest.raises(InvalidAllele):
        normalise_allele(allele)


def test_a_lower_case_base_is_in_the_alphabet():
    """#360 upper-cases an allele before identity, and the rule judges that."""
    assert allele_fault("a", "g") is None
    assert allele_fault("acgt", "T") is None
    assert identity_fault("1", "a", "G", other_is_reference=False) is None
    assert isinstance(allele_fault("a", "n"), InvalidAllele)


def test_a_malformed_identity_is_judged_before_the_rest_of_the_row():
    """A row with a bad position as well as a bad allele is this rule's, on the
    readers that orient and on the GWAS-VCF one that does not; a chromosome
    refused for another reason (an unsupported number) leaves it the reader's."""
    faults = IdentityFaults()

    assert orient_or_drop("1", "not-a-position", "G", "N", faults, other_is_reference=False) is None
    assert orient_or_drop("27", 100, "G", "N", faults, other_is_reference=False) is None

    assert faults.rows == [("1:not-a-position:N:G", "allele_invalid")]
    assert isinstance(identity_fault("1", "N", "G", other_is_reference=False), InvalidAllele)
    assert identity_fault("27", "N", "G", other_is_reference=False) is None


def _vcf_text(rows: list[tuple[str, int, str, str]]) -> str:
    return dense_fixture._vcf_header("Continuous") + "".join(
        f"{chrom}\t{pos}\t.\t{ref}\t{alt}\t.\tPASS\t.\tES:SE\t1.0:0.5\n"
        for chrom, pos, ref, alt in rows
    )


def _vcf(tmp_path: Path) -> SourceReader:
    path = tmp_path / "trait_a.vcf"
    path.write_text(_vcf_text(ROWS), encoding="utf-8")
    return GwasVcfReader(path)


def _ssf(tmp_path: Path) -> SourceReader:
    path = tmp_path / "trait_a.tsv.gz"
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write("chromosome\tbase_pair_location\teffect_allele\tother_allele\tbeta\tstandard_error\n")
        fh.writelines(f"{c}\t{p}\t{alt}\t{ref}\t1.0\t0.5\n" for c, p, ref, alt in ROWS)
    return GwasSsfReader(path)


def _finngen(tmp_path: Path) -> SourceReader:
    path = tmp_path / "trait_a.finngen.tsv.gz"
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write("#chrom\tpos\tref\talt\trsids\tpval\tbeta\tsebeta\taf_alt\n")
        fh.writelines(f"{c}\t{p}\t{ref}\t{alt}\t\t0.05\t1.0\t0.5\t0.3\n" for c, p, ref, alt in ROWS)
    return FinnGenR13Reader(path)


@pytest.mark.parametrize("reader_for", [_vcf, _ssf, _finngen])
def test_every_reader_leaves_out_and_names_a_malformed_identity(
    tmp_path: Path, reader_for: Callable[[Path], SourceReader]
):
    """Both streams leave the row out, so no axis holds a variant whose
    associations Admission dropped; the association stream names it."""
    reader = reader_for(tmp_path)
    tally = StreamTally()

    yielded = [(a.chromosome, a.position) for a in reader.stream_associations(tally)]

    assert yielded == [("1", 1000)]
    assert [list(row) for row in tally.faults.rows] == DROPPED
    assert [(v.chromosome, v.position) for v in reader.stream_variants()] == [("1", 1000)]


def _record(store: Path) -> tuple[dict, list[list[str]]]:
    """A one-Analysis Store's record entry, and its logged drops."""
    record = admission_record_for(store)
    (analysis,) = json.loads((record / "admission.json").read_text(encoding="utf-8"))["analyses"]
    with gzip.open(record / "dropped" / "0.tsv.gz", "rt", encoding="utf-8") as fh:
        logged = [line.rstrip("\n").split("\t") for line in fh][1:]
    return analysis, logged


def _alids(store: Path) -> list[str]:
    return [record.alid for record in iter_variant_records(store / "variants.tsv.gz")]


@pytest.mark.parametrize("hybrid", [False, True])
def test_a_vcf_build_stores_no_malformed_alid(tmp_path: Path, hybrid: bool):
    vcf = tmp_path / "trait_a.vcf"
    vcf.write_text(_vcf_text(ROWS), encoding="utf-8")
    manifest = dense_fixture._make_manifest(tmp_path, [("trait_a", vcf, "Trait A")])
    store = vcf_store(tmp_path, manifest, ["1:1000:A:G"] if hybrid else None)

    analysis, logged = _record(store)
    assert (analysis["examined"], analysis["admitted"]) == (4, 1)
    assert analysis["dropped"] == {"chromosome_missing": 1, "allele_invalid": 2}
    assert logged == DROPPED
    assert _alids(store) == ["1:1000:A:G"]


def test_a_ragged_ssf_build_logs_a_malformed_identity(tmp_path: Path):
    store, _ = ragged_ssf_store(
        tmp_path,
        [
            {"chromosome": c, "base_pair_location": p, "effect_allele": alt,
             "other_allele": ref, "beta": 1.0, "standard_error": 0.5}
            for c, p, ref, alt in ROWS
        ],
    )

    analysis, logged = _record(store)
    assert (analysis["examined"], analysis["admitted"]) == (4, 1)
    assert logged == DROPPED
    assert _alids(store) == ["1:1000:A:G"]


def test_a_dense_build_drops_a_malformed_row_rather_than_refusing_its_source(tmp_path: Path):
    """`build-dense` used to refuse the whole source over one such row."""
    from dense_sources import build_dense_fixture

    source = tmp_path / "records.tsv"
    source.write_text(
        "analysis_id\tchromosome\tposition\teffect_allele\tother_allele\tz\tse\n"
        + "".join(f"A\t{c}\t{p}\t{alt}\t{ref}\t2.0\t0.1\n" for c, p, ref, alt in ROWS),
        encoding="utf-8",
    )
    store = tmp_path / "dense.opengwasdb"
    build_dense_fixture(
        [source], store, store_id="s", release_id="r", reference_assembly="GRCh38"
    )

    analysis, logged = _record(store)
    assert (analysis["examined"], analysis["admitted"]) == (4, 1)
    assert logged == DROPPED
    assert _alids(store) == ["1:1000:A:G"]


def test_a_besd_build_logs_each_probe_s_row_at_a_malformed_esi_row(tmp_path: Path):
    """An ESI row is every probe's, so each probe's row there is its own drop,
    named by ESI index; the row is not lifted or placed, and so not `off_axis`."""
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    snps: list[besd_fixtures.EsiRow] = [
        {"chr": c, "snp_id": f"rs{i}", "bp": p, "a1": alt, "a2": ref}
        for i, (c, p, ref, alt) in enumerate(ROWS)
    ]
    besd_fixtures.write_esi(fixture / "test.esi", snps)
    besd_fixtures.write_epi(fixture / "test.epi", [{"chr": "1", "probe_id": "ENSG0", "bp": 1000}])
    besd_fixtures.write_besd_sparse_3f(
        fixture / "test.besd", 1, [[(i, 0.1, 0.02) for i in range(len(ROWS))]]
    )
    store = tmp_path / "besd.opengwasdb"
    build_ragged_from_besd(
        fixture / "test", store, store_id="s", release_id="r",
        association_coverage="cis_and_signals",
    )

    analysis, logged = _record(store)
    assert (analysis["examined"], analysis["admitted"]) == (4, 1)
    assert logged == [["esi:1", "chromosome_missing"], ["esi:2", "allele_invalid"],
                      ["esi:3", "allele_invalid"]]
    assert _alids(store) == ["1:1000:A:G"]


def _truncated_ssf(tmp_path: Path) -> SourceReader:
    path = tmp_path / "trait_a.tsv"
    path.write_text(
        "chromosome\tbase_pair_location\teffect_allele\tother_allele\tbeta\tstandard_error\n"
        "1\t1000\tG\tA\t1.0\t0.5\n"
        "1\t2000\tG\n"
        "1\t3000\n",
        encoding="utf-8",
    )
    return GwasSsfReader(path)


def _truncated_finngen(tmp_path: Path) -> SourceReader:
    path = tmp_path / "trait_a.finngen.tsv"
    path.write_text(
        "#chrom\tpos\tref\talt\trsids\tpval\tbeta\tsebeta\taf_alt\n"
        "1\t1000\tA\tG\t\t0.05\t1.0\t0.5\t0.3\n"
        "1\t2000\tA\n"
        "1\t3000\n",
        encoding="utf-8",
    )
    return FinnGenR13Reader(path)


@pytest.mark.parametrize(
    ("reader_for", "short_key"),
    # The key spells ref then alt: GWAS-SSF's row keeps its effect (alt)
    # allele, FinnGen's its ref.
    [(_truncated_ssf, "1:2000::G"), (_truncated_finngen, "1:2000:A:")],
    ids=["gwas-ssf", "finngen"],
)
def test_a_truncated_row_s_missing_allele_is_named_not_skipped(
    tmp_path: Path, reader_for: Callable[[Path], SourceReader], short_key: str
):
    """#365: a row that stops before an allele has a missing allele, so the
    association stream names it as `allele_invalid`, where the reader used to
    skip it unseen."""
    tally = StreamTally()

    yielded = [(a.chromosome, a.position) for a in reader_for(tmp_path).stream_associations(tally)]

    assert yielded == [("1", 1000)]
    assert [list(row) for row in tally.faults.rows] == [
        [short_key, "allele_invalid"], ["1:3000::", "allele_invalid"],
    ]
