"""Tests for opengwasdb.readers (issue #19; extended by #20, #21, #84):
capability resolution, the GWAS-VCF and GWAS-SSF readers, the in-memory
fake, and the conformance suite all three share.
"""

from __future__ import annotations

import gzip
import math
import subprocess
from pathlib import Path

import numpy as np
import pytest

import opengwasdb.readers.finngen as finngen_module
import opengwasdb.readers.gwas_ssf as gwas_ssf_module
import opengwasdb.readers.tabular as tabular_module
from opengwasdb.build.phenotype_sd import estimate_phenotype_sd
from opengwasdb.model.enums import OriginalSdMethod, StoredEffectScale
from opengwasdb.readers import (
    FINNGEN_R13_CAPABILITY,
    GWAS_SSF_CAPABILITY,
    GWAS_VCF_CAPABILITY,
    EffectSource,
    EffectSourceKind,
    FakeReader,
    FinnGenR13Reader,
    GwasSsfReader,
    GwasVcfReader,
    ReaderAssociation,
    SiteMetrics,
    af_only,
    is_palindromic,
    known_capabilities,
    resolve_reader,
    site_metrics_arrays,
)


def _write_vcf(path: Path, body: str, study_type: str = "Continuous") -> None:
    """Write a minimal GWAS-VCF fixture to ``path``, including AF alongside
    the effect/precision fields ``tests/test_vcf_source.py`` already uses."""
    header = f"""\
##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##FORMAT=<ID=ES,Number=A,Type=Float,Description="Effect size relative to ALT">
##FORMAT=<ID=SE,Number=A,Type=Float,Description="Standard error of effect size">
##FORMAT=<ID=EZ,Number=A,Type=Float,Description="Z-score, if used to derive EFFECT/SE">
##FORMAT=<ID=AF,Number=A,Type=Float,Description="Alternate allele frequency">
##SAMPLE=<ID=STUDY1,StudyType={study_type}>
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSTUDY1
"""
    path.write_text(header + body, encoding="utf-8")


def _bgzip_index(vcf: Path) -> Path:
    """Compress + tabix-index a plain VCF (what extract_at_sites's `-R` needs),
    matching ``tests/test_ancestry_assignment.py``'s identical helper for
    ``extract_af_at_sites`` -- both readers hit the same bcftools -R
    requirement."""
    out = vcf.with_suffix(".vcf.gz")
    subprocess.run(
        ["bcftools", "view", str(vcf), "-Oz", "-o", str(out), "--write-index=tbi"],
        check=True,
        capture_output=True,
    )
    return out


# --- resolve_reader(): the single, documented resolution point ---


def test_resolve_reader_returns_gwas_vcf_reader_for_its_capability(tmp_path):
    vcf = tmp_path / "study.vcf"
    _write_vcf(vcf, "1\t100\t.\tA\tG\t.\tPASS\t.\tES:SE\t1.0:0.5\n")

    reader = resolve_reader(GWAS_VCF_CAPABILITY, vcf, StoredEffectScale.SD)

    assert isinstance(reader, GwasVcfReader)


_SSF_HEADER = [
    "chromosome",
    "base_pair_location",
    "effect_allele",
    "other_allele",
    "beta",
    "standard_error",
    "effect_allele_frequency",
]


def _write_ssf(path: Path, rows: list[dict]) -> None:
    """Write a filtered/harmonised GWAS-SSF ``.tsv.gz`` fixture."""
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write("\t".join(_SSF_HEADER) + "\n")
        for row in rows:
            fh.write("\t".join(str(row.get(col, "")) for col in _SSF_HEADER) + "\n")


def _variant_bytes(reader: FinnGenR13Reader | GwasSsfReader) -> bytes:
    """Serialize the reader output exactly as a builder-facing byte stream."""
    return b"".join(
        f"{variant.chromosome}\t{variant.position}\t{variant.ref}\t{variant.alt}\t{variant.rsid}\n".encode()
        for variant in reader.stream_variants()
    )


def _legacy_variant_bytes(variants) -> bytes:
    return b"".join(
        f"{variant.chromosome}\t{variant.position}\t{variant.ref}\t{variant.alt}\t{variant.rsid}\n".encode()
        for variant in variants
    )


def test_resolve_reader_returns_gwas_ssf_reader_for_its_capability(tmp_path):
    path = tmp_path / "study.tsv.gz"
    _write_ssf(
        path,
        [
            {
                "chromosome": "1", "base_pair_location": 100,
                "effect_allele": "G", "other_allele": "A",
                "beta": 1.0, "standard_error": 0.5,
            }
        ],
    )

    reader = resolve_reader(GWAS_SSF_CAPABILITY, path, StoredEffectScale.SD)

    assert isinstance(reader, GwasSsfReader)


def test_resolve_reader_returns_finngen_r13_reader_for_its_capability():
    path = Path(__file__).parent / "fixtures" / "finngen_r13.tsv"

    reader = resolve_reader(FINNGEN_R13_CAPABILITY, path, StoredEffectScale.LOG_OR)

    assert isinstance(reader, FinnGenR13Reader)


def test_known_capabilities_returns_sorted_registered_set():
    """known_capabilities() returns a sorted tuple of all registered capabilities."""
    caps = known_capabilities()
    assert isinstance(caps, tuple)
    assert caps == (FINNGEN_R13_CAPABILITY, GWAS_SSF_CAPABILITY, GWAS_VCF_CAPABILITY)


def test_resolve_reader_rejects_unknown_capability(tmp_path):
    with pytest.raises(
        ValueError,
        match=(
            r"unknown source reader capability '.*'; "
            r"known: opengwasdb\.finngen-r13, opengwasdb\.gwas-ssf, opengwasdb\.gwas-vcf"
        ),
    ):
        resolve_reader(
            "opengwasdb.some-future-format", tmp_path / "study.vcf", StoredEffectScale.SD
        )


# --- Shared conformance suite: same assertions against GwasVcfReader and FakeReader ---


def _gwas_vcf_reader(tmp_path: Path) -> GwasVcfReader:
    vcf = tmp_path / "study.vcf"
    _write_vcf(
        vcf,
        # REF=A, ALT=G -> A1=A, effect allele (ALT=G) is A2 -> z negated, AF flipped.
        "1\t100\t.\tA\tG\t.\tPASS\t.\tES:SE:AF\t1.0:0.5:0.30\n"
        # REF=G, ALT=A -> A1=A, effect allele (ALT=A) is A1 -> no flip.
        "1\t200\t.\tG\tA\t.\tPASS\t.\tES:SE:AF\t1.0:0.5:0.70\n",
    )
    # Bgzip+index: stream_associations works on a plain VCF, but
    # extract_at_sites's bcftools -R region lookup requires an index.
    return GwasVcfReader(_bgzip_index(vcf), StoredEffectScale.SD)


def _gwas_ssf_reader(tmp_path: Path) -> GwasSsfReader:
    path = tmp_path / "study.tsv.gz"
    _write_ssf(
        path,
        [
            # effect_allele=G, other_allele=A -> A1=A, effect (G) is A2 ->
            # z negated, af flipped -- same fixture shape as _gwas_vcf_reader.
            {
                "chromosome": "1", "base_pair_location": 100,
                "effect_allele": "G", "other_allele": "A",
                "beta": 1.0, "standard_error": 0.5, "effect_allele_frequency": 0.30,
            },
            # effect_allele=A, other_allele=G -> A1=A -> no flip.
            {
                "chromosome": "1", "base_pair_location": 200,
                "effect_allele": "A", "other_allele": "G",
                "beta": 1.0, "standard_error": 0.5, "effect_allele_frequency": 0.70,
            },
        ],
    )
    return GwasSsfReader(path, StoredEffectScale.SD)


def _fake_reader() -> FakeReader:
    return FakeReader(
        associations=[
            ReaderAssociation(
                chromosome="1",
                position=100,
                ref="A",
                alt="G",
                z=-2.0,
                se=0.5,
                stored_effect_scale=StoredEffectScale.SD,
            ),
            ReaderAssociation(
                chromosome="1",
                position=200,
                ref="G",
                alt="A",
                z=2.0,
                se=0.5,
                stored_effect_scale=StoredEffectScale.SD,
            ),
        ],
        sites={
            "1:100:A:G": SiteMetrics(af=0.70, se=0.5),
            "1:200:A:G": SiteMetrics(af=0.70, se=0.5),
        },
    )


def _finngen_reader() -> FinnGenR13Reader:
    # Captured verbatim from the public R13 AB1_ACTINOMYCOSIS endpoint.
    return FinnGenR13Reader(
        Path(__file__).parent / "fixtures" / "finngen_r13.tsv",
        StoredEffectScale.SD,
    )


@pytest.fixture(params=["gwas_vcf", "fake", "gwas_ssf", "finngen_r13"])
def reader(request, tmp_path):
    if request.param == "gwas_vcf":
        return _gwas_vcf_reader(tmp_path)
    if request.param == "gwas_ssf":
        return _gwas_ssf_reader(tmp_path)
    if request.param == "finngen_r13":
        return _finngen_reader()
    return _fake_reader()


def _malformed_fake_reader() -> FakeReader:
    """A negative SE violates ReaderAssociation's own invariant -- raised at
    construction, which is what lets "rejection of malformed input" be a
    genuinely shared, reader-agnostic conformance check rather than one only
    a file-parsing reader can exercise."""
    return FakeReader(
        associations=[
            ReaderAssociation(
                chromosome="1",
                position=100,
                ref="A",
                alt="G",
                z=1.0,
                se=-0.5,
                stored_effect_scale=StoredEffectScale.SD,
            )
        ]
    )


@pytest.fixture
def malformed_reader_factory():
    """Only the fake is exercised here: GWAS-VCF's own malformed-input
    rejection (an unrecognised header StudyType) no longer exists as of
    issue #17 -- `stored_effect_scale` is supplied by the caller at
    construction, not derived from the file's header, so `GwasVcfReader` has
    no remaining file-content validation of its own to reject on. A zero-arg
    callable rather than an already-built reader, so the fake's construction
    ValueError is caught by the test's own `pytest.raises`, not by fixture
    setup."""
    return _malformed_fake_reader


def test_reader_conformance_rejects_malformed_input(malformed_reader_factory):
    with pytest.raises(ValueError):
        list(malformed_reader_factory().stream_associations())


def test_reader_conformance_orients_associations_to_a1(reader):
    associations = list(reader.stream_associations())

    assert len(associations) >= 2
    by_position = {a.position: a for a in associations}
    if isinstance(reader, FinnGenR13Reader):
        assert by_position[13668].z == pytest.approx(1.99175 / 1.46977)
        assert by_position[19234].z == pytest.approx(2.16344 / 10.5777)
        return
    assert by_position[100].z == pytest.approx(-2.0)
    assert by_position[200].z == pytest.approx(2.0)


def test_reader_conformance_se_is_non_negative(reader):
    associations = list(reader.stream_associations())

    assert associations
    assert all(a.se >= 0 for a in associations)


def test_reader_conformance_extract_at_sites_returns_a1_oriented_af(reader):
    if isinstance(reader, FinnGenR13Reader):
        sites = reader.extract_at_sites(["1:13668:A:G", "1:19234:A:G"])
        assert sites["1:13668:A:G"].af == pytest.approx(0.00596897)
        assert sites["1:19234:A:G"].af == pytest.approx(1.0 - 6.76508e-05)
        assert all(metrics.se >= 0 for metrics in sites.values())
        return
    sites = reader.extract_at_sites(["1:100:A:G", "1:200:A:G"])

    assert sites["1:100:A:G"].af == pytest.approx(0.70)
    assert sites["1:200:A:G"].af == pytest.approx(0.70)
    assert all(metrics.se >= 0 for metrics in sites.values())


def test_reader_conformance_extract_at_sites_ignores_unrequested_alids(reader):
    requested = "1:13668:A:G" if isinstance(reader, FinnGenR13Reader) else "1:100:A:G"
    sites = reader.extract_at_sites([requested])

    assert set(sites) == {requested}


def test_reader_conformance_stream_variants_covers_stream_associations(reader):
    """stream_variants (issue #20) is a superset of stream_associations'
    positions -- every association's variant must appear in the variant
    stream, independent of any per-reader extra filtering."""
    assoc_positions = {
        (a.chromosome, a.position, a.ref, a.alt) for a in reader.stream_associations()
    }
    variant_positions = {v.site for v in reader.stream_variants()}

    assert assoc_positions <= variant_positions


def test_finngen_r13_drops_unusable_rows_without_fabricating_metrics(tmp_path):
    path = tmp_path / "malformed.tsv"
    path.write_text(
        "#chrom\tpos\tref\talt\tbeta\tsebeta\taf_alt\n"
        "1\t400\tA\tC\tNA\t0\tNA\n"
        "1\tbad\tA\tG\t1.0\t0.5\t0.2\n"
        "1\t500\tA\tN\t1.0\t0.5\t0.2\n",
        encoding="utf-8",
    )
    reader = FinnGenR13Reader(path, StoredEffectScale.SD)

    associations = list(reader.stream_associations())
    variants = list(reader.stream_variants())
    sites = reader.extract_at_sites(["1:400:A:C"])

    assert associations == []
    assert [v.site for v in variants] == [("1", 400, "A", "C")]
    assert sites == {}


def test_finngen_r13_normalises_chromosome_23_to_x():
    reader = _finngen_reader()

    association = next(a for a in reader.stream_associations() if a.position == 98536)
    sites = reader.extract_at_sites(["X:98536:A:C"])

    assert association.chromosome == "X"
    assert sites["X:98536:A:C"].af == pytest.approx(0.00146324)


def test_finngen_variant_projection_preserves_exact_output_without_parsing_statistics(
    tmp_path, monkeypatch
):
    path = tmp_path / "finngen.tsv"
    path.write_text(
        '"extra"\t"alt"\t"rsids"\t"pos"\t"#chrom"\t"ref"\t'
        '"beta"\t"sebeta"\t"af_alt"\t"pval"\t"mlogp"\t"nearest_genes"\n'
        "x\tA\trs100,rs101\t100\t1\tG\tbad-beta\tbad-se\tbad-af\tbad-p\tbad-m\tGENE1\n"
        "x\tG\t.\t200\t1\tA\tbad-beta\tbad-se\tbad-af\tbad-p\tbad-m\tGENE2\n"
        "x\tA\trs102\t100\t1\tG\tbad-beta\tbad-se\tbad-af\tbad-p\tbad-m\tGENE1\n"
        "x\tC\trsX\t300\t23\tA\tbad-beta\tbad-se\tbad-af\tbad-p\tbad-m\tGENEX\n"
        "x\tC\trsBadChrom\t400\t\tA\tbad-beta\tbad-se\tbad-af\tbad-p\tbad-m\tGENE\n"
        "x\tC\trsBadPos\t0\t1\tA\tbad-beta\tbad-se\tbad-af\tbad-p\tbad-m\tGENE\n"
        "x\tN\trsBadAllele\t500\t1\tA\tbad-beta\tbad-se\tbad-af\tbad-p\tbad-m\tGENE\n"
        '"x"\t"A"\t"rsQuoted"\t"600"\t"1"\t"G"\tbad-beta\tbad-se\tbad-af\tbad-p\tbad-m\tGENE\n',
        encoding="utf-8",
    )

    legacy = _legacy_variant_bytes(finngen_module.stream_full_row_variants(path))

    def fail_if_called(value):
        raise AssertionError(f"variant projection parsed association statistic {value!r}")

    monkeypatch.setattr(tabular_module, "parse_finite_float", fail_if_called)
    monkeypatch.setattr(tabular_module, "parse_positive_float", fail_if_called)
    monkeypatch.setattr(tabular_module, "parse_af", fail_if_called)

    projected = _variant_bytes(FinnGenR13Reader(path))

    assert projected == legacy == (
        b"1\t100\tG\tA\trs100\n"
        b"1\t200\tA\tG\t\n"
        b"1\t100\tG\tA\trs102\n"
        b"X\t300\tA\tC\trsX\n"
        b"1\t600\tG\tA\trsQuoted\n"
    )


def test_finngen_variant_projection_is_byte_identical_without_optional_rsid_column(tmp_path):
    path = tmp_path / "finngen.tsv.gz"
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write(
            "alt\tsebeta\tchrom\tref\tpos\tbeta\taf_alt\textra\n"
            "G\t0.5\t1\tA\t100\t1.0\t0.2\tx\n"
            "A\t0.6\t23\tC\t200\t-1.0\t0.3\ty\n"
        )

    projected = _variant_bytes(FinnGenR13Reader(path))
    legacy = _legacy_variant_bytes(finngen_module.stream_full_row_variants(path))

    assert projected == legacy == b"1\t100\tA\tG\t\nX\t200\tC\tA\t\n"


# --- GWAS-VCF manifest authority (issue #17) ---


def test_gwas_vcf_reader_uses_constructor_scale_regardless_of_header(tmp_path):
    """The ieu-a-7 scenario at the reader level: a VCF header StudyType that
    disagrees with (or is entirely absent from) the manifest-resolved scale
    must not affect what's yielded -- the constructor argument always wins."""
    vcf = tmp_path / "study.vcf"
    _write_vcf(vcf, "1\t100\t.\tA\tG\t.\tPASS\t.\tES:SE\t1.0:0.5\n", study_type="Unknown")

    reader = GwasVcfReader(vcf, StoredEffectScale.LOG_OR)
    associations = list(reader.stream_associations())

    assert len(associations) == 1
    assert associations[0].stored_effect_scale is StoredEffectScale.LOG_OR


def test_gwas_vcf_reader_extract_at_sites_with_no_sites_returns_empty(tmp_path):
    vcf = tmp_path / "study.vcf"
    _write_vcf(vcf, "1\t100\t.\tA\tG\t.\tPASS\t.\tES:SE:AF\t1.0:0.5:0.30\n")

    assert GwasVcfReader(vcf, StoredEffectScale.SD).extract_at_sites([]) == {}


def test_gwas_vcf_reader_stream_variants_includes_rows_dropped_from_associations(tmp_path):
    """A row with an invalid SE is skipped by stream_associations (issue #19)
    but still belongs on a builder's union-variant axis (issue #20)."""
    vcf = tmp_path / "study.vcf"
    _write_vcf(
        vcf,
        "1\t100\t.\tA\tG\t.\tPASS\t.\tES:SE\t1.0:0.5\n"
        "1\t200\t.\tG\tA\t.\tPASS\t.\tES:SE\t1.0:0\n",  # SE=0 -> dropped by stream_associations
    )
    reader = GwasVcfReader(vcf, StoredEffectScale.SD)

    assoc_positions = {(a.chromosome, a.position) for a in reader.stream_associations()}
    variant_positions = {(v.chromosome, v.position) for v in reader.stream_variants()}

    assert assoc_positions == {("1", 100)}
    assert variant_positions == {("1", 100), ("1", 200)}


# --- GWAS-SSF reader (issue #84) ---


def test_gwas_ssf_reader_extract_at_sites_with_no_sites_returns_empty(tmp_path):
    path = tmp_path / "study.tsv.gz"
    _write_ssf(
        path,
        [
            {
                "chromosome": "1", "base_pair_location": 100,
                "effect_allele": "G", "other_allele": "A",
                "beta": 1.0, "standard_error": 0.5, "effect_allele_frequency": 0.30,
            }
        ],
    )

    assert GwasSsfReader(path, StoredEffectScale.SD).extract_at_sites([]) == {}


def test_gwas_ssf_reader_extract_at_sites_returns_nothing_when_file_has_no_af_column(tmp_path):
    """SiteMetrics never fabricates an AF: a filtered file with no
    `effect_allele_frequency` column yields no site metrics at all, rather
    than guessing one."""
    path = tmp_path / "study.tsv.gz"
    _write_ssf(
        path,
        [
            {
                "chromosome": "1", "base_pair_location": 100,
                "effect_allele": "G", "other_allele": "A",
                "beta": 1.0, "standard_error": 0.5,
            }
        ],
    )

    assert GwasSsfReader(path, StoredEffectScale.SD).extract_at_sites(["1:100:A:G"]) == {}


def test_gwas_ssf_reader_stream_variants_includes_rows_dropped_from_associations(tmp_path):
    """A row with a non-positive SE is skipped by stream_associations but
    still belongs on a builder's union-variant axis (issue #20), mirroring
    GwasVcfReader's identical contract."""
    path = tmp_path / "study.tsv.gz"
    _write_ssf(
        path,
        [
            {
                "chromosome": "1", "base_pair_location": 100,
                "effect_allele": "G", "other_allele": "A",
                "beta": 1.0, "standard_error": 0.5,
            },
            {
                "chromosome": "1", "base_pair_location": 200,
                "effect_allele": "A", "other_allele": "G",
                "beta": 1.0, "standard_error": 0.0,  # non-positive -> dropped
            },
        ],
    )
    reader = GwasSsfReader(path, StoredEffectScale.SD)

    assoc_positions = {(a.chromosome, a.position) for a in reader.stream_associations()}
    variant_positions = {(v.chromosome, v.position) for v in reader.stream_variants()}

    assert assoc_positions == {("1", 100)}
    assert variant_positions == {("1", 100), ("1", 200)}


def test_gwas_ssf_reader_drops_rows_with_unparseable_alleles(tmp_path):
    """Malformed rows -- here a non-ACGT allele -- can't be represented as a
    variant at all, so they are dropped from every stream, not fabricated."""
    path = tmp_path / "study.tsv.gz"
    _write_ssf(
        path,
        [
            {
                "chromosome": "1", "base_pair_location": 100,
                "effect_allele": "G", "other_allele": "A",
                "beta": 1.0, "standard_error": 0.5,
            },
            {
                "chromosome": "1", "base_pair_location": 200,
                "effect_allele": "N", "other_allele": "A",  # unparseable allele
                "beta": 1.0, "standard_error": 0.5,
            },
        ],
    )
    reader = GwasSsfReader(path, StoredEffectScale.SD)

    assert {a.position for a in reader.stream_associations()} == {100}
    assert {v.position for v in reader.stream_variants()} == {100}


def test_gwas_ssf_reader_extract_at_sites_excludes_palindromic(tmp_path):
    path = tmp_path / "study.tsv.gz"
    _write_ssf(
        path,
        [
            # A/T is palindromic -- excluded, like GwasVcfReader.extract_at_sites.
            {
                "chromosome": "1", "base_pair_location": 100,
                "effect_allele": "A", "other_allele": "T",
                "beta": 1.0, "standard_error": 0.5, "effect_allele_frequency": 0.30,
            },
            {
                "chromosome": "1", "base_pair_location": 200,
                "effect_allele": "A", "other_allele": "G",
                "beta": 1.0, "standard_error": 0.5, "effect_allele_frequency": 0.30,
            },
        ],
    )

    sites = GwasSsfReader(path, StoredEffectScale.SD).extract_at_sites(
        ["1:100:A:T", "1:200:A:G"]
    )

    assert set(sites) == {"1:200:A:G"}


def test_gwas_ssf_reader_uses_constructor_scale(tmp_path):
    path = tmp_path / "study.tsv.gz"
    _write_ssf(
        path,
        [
            {
                "chromosome": "1", "base_pair_location": 100,
                "effect_allele": "G", "other_allele": "A",
                "beta": 1.0, "standard_error": 0.5,
            }
        ],
    )

    reader = GwasSsfReader(path, StoredEffectScale.LOG_OR)
    associations = list(reader.stream_associations())

    assert len(associations) == 1
    assert associations[0].stored_effect_scale is StoredEffectScale.LOG_OR


def test_gwas_ssf_variant_projection_preserves_exact_output_without_parsing_statistics(
    tmp_path, monkeypatch
):
    path = tmp_path / "study.tsv"
    path.write_text(
        "extra\tother_allele\tvariant_id\tstandard_error\tchromosome\t"
        "effect_allele_frequency\tbase_pair_location\teffect_allele\trsid\tbeta\n"
        "x\tA\trsFallback\tbad-se\t1\tbad-af\t100\tG\t\tbad-beta\n"
        "x\tG\tnot-an-rsid\tbad-se\t1\tbad-af\t200\tA\trsPrimary\tbad-beta\n"
        "x\tA\trsDuplicate\tbad-se\t1\tbad-af\t100\tG\t\tbad-beta\n"
        "x\tA\trsBadChrom\tbad-se\t\tbad-af\t300\tC\t\tbad-beta\n"
        "x\tA\trsBadPos\tbad-se\t1\tbad-af\t0\tC\t\tbad-beta\n"
        "x\tA\trsBadAllele\tbad-se\t1\tbad-af\t400\tN\t\tbad-beta\n"
        '"x"\t"A"\t"rsQuoted"\tbad-se\t"1"\tbad-af\t"500"\t"G"\t""\tbad-beta\n',
        encoding="utf-8",
    )

    legacy = _legacy_variant_bytes(gwas_ssf_module.stream_full_row_variants(path))

    def fail_if_called(value):
        raise AssertionError(f"variant projection parsed association statistic {value!r}")

    monkeypatch.setattr(gwas_ssf_module, "parse_finite_float", fail_if_called)
    monkeypatch.setattr(gwas_ssf_module, "parse_positive_float", fail_if_called)
    monkeypatch.setattr(gwas_ssf_module, "parse_af", fail_if_called)

    projected = _variant_bytes(GwasSsfReader(path))

    assert projected == legacy == (
        b"1\t100\tA\tG\trsFallback\n"
        b"1\t200\tG\tA\trsPrimary\n"
        b"1\t100\tA\tG\trsDuplicate\n"
        b"1\t500\tA\tG\trsQuoted\n"
    )


@pytest.mark.parametrize(
    ("reader_type", "header"),
    [
        (FinnGenR13Reader, "#chrom\tpos\tref\n"),
        (GwasSsfReader, "chromosome\tbase_pair_location\tother_allele\n"),
    ],
)
def test_projected_variant_reader_fails_loudly_when_identity_column_is_missing(
    tmp_path, reader_type, header
):
    path = tmp_path / "missing-required-column.tsv"
    path.write_text(header, encoding="utf-8")

    with pytest.raises(ValueError, match="missing required variant columns"):
        list(reader_type(path).stream_variants())


def test_gwas_ssf_variant_projection_is_byte_identical_without_optional_aliases(tmp_path):
    path = tmp_path / "study.tsv.gz"
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write(
            "beta\tother_allele\tchromosome\tstandard_error\teffect_allele\t"
            "base_pair_location\textra\n"
            "1.0\tA\t1\t0.5\tG\t100\tx\n"
            "-1.0\tG\t1\t0.6\tA\t200\ty\n"
        )

    projected = _variant_bytes(GwasSsfReader(path))
    legacy = _legacy_variant_bytes(gwas_ssf_module.stream_full_row_variants(path))

    assert projected == legacy == b"1\t100\tA\tG\t\n1\t200\tG\tA\t\n"


@pytest.mark.parametrize(
    ("header", "row", "expected"),
    [
        (
            "chromosome\tbase_pair_location\tother_allele\teffect_allele\n",
            '1\t100\tA\t"G"\n',
            b"1\t100\tA\tG\t\n",
        ),
        (
            "chromosome\tbase_pair_location\tother_allele\teffect_allele\trsid\n",
            '1\t100\tA\tG\t"rs1"\n',
            b"1\t100\tA\tG\trs1\n",
        ),
    ],
)
def test_gwas_ssf_projection_preserves_quoted_final_projected_column(
    tmp_path, header, row, expected
):
    path = tmp_path / "quoted-final-column.tsv"
    path.write_text(header + row, encoding="utf-8")

    projected = _variant_bytes(GwasSsfReader(path))
    legacy = _legacy_variant_bytes(gwas_ssf_module.stream_full_row_variants(path))

    assert projected == legacy == expected


# --- Site extraction: orientation, palindromes, liftover (issue #21) ---


def _af_se_vcf(tmp_path: Path, rows: list[str]) -> Path:
    header = (
        "##fileformat=VCFv4.2\n"
        '##FORMAT=<ID=AF,Number=A,Type=Float,Description="Allele frequency">\n'
        '##FORMAT=<ID=SE,Number=A,Type=Float,Description="Standard error">\n'
        "##SAMPLE=<ID=S1,StudyType=Continuous>\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\n"
    )
    path = tmp_path / "af_se_study.vcf"
    path.write_text(header + "".join(rows), encoding="utf-8")
    return path


def test_is_palindromic():
    assert is_palindromic("A", "T") and is_palindromic("C", "G")
    assert not is_palindromic("A", "C") and not is_palindromic("A", "G")


def test_extract_at_sites_orients_and_excludes_palindromic(tmp_path):
    # Variant 1: REF=C ALT=A -> A1=A=ALT (not flipped) -> af stays 0.30.
    # Variant 2: REF=A ALT=G -> A1=A=REF (ALT flipped) -> af = 1 - 0.30 = 0.70.
    # Variant 3: REF=A ALT=T -> palindromic -> excluded.
    vcf = _bgzip_index(
        _af_se_vcf(
            tmp_path,
            [
                "1\t1000\t.\tC\tA\t.\tPASS\t.\tAF:SE\t0.30:0.1\n",
                "1\t1001\t.\tA\tG\t.\tPASS\t.\tAF:SE\t0.30:0.1\n",
                "1\t1002\t.\tA\tT\t.\tPASS\t.\tAF:SE\t0.30:0.1\n",
            ],
        )
    )
    wanted = {"1:1000:A:C", "1:1001:A:G", "1:1002:A:T"}
    sites = GwasVcfReader(vcf, StoredEffectScale.SD).extract_at_sites(wanted)
    assert sites["1:1000:A:C"].af == pytest.approx(0.30)
    assert sites["1:1001:A:G"].af == pytest.approx(0.70)
    assert "1:1002:A:T" not in sites  # palindromic dropped


class _FakeLiftover:
    """Stand-in for pyliftover.LiftOver: shifts chr1 positions by +64620."""

    def convert_coordinate(self, chrom, pos0):  # noqa: D401
        if chrom != "chr1":
            return []
        return [("chr1", pos0 + 64620, "+", 0)]


def test_extract_at_sites_applies_liftover(tmp_path):
    # Study is on an older build; the reference ALIDs are on the lifted build.
    vcf = _af_se_vcf(
        tmp_path,
        ["1\t1000\t.\tC\tA\t.\tPASS\t.\tAF:SE\t0.25:0.1\n"],  # pos 1000 -> 65620 after lift
    )
    wanted = {"1:65620:A:C"}  # canonical A1=A (ALT), lifted position
    sites = GwasVcfReader(vcf, StoredEffectScale.SD, liftover=_FakeLiftover()).extract_at_sites(
        wanted
    )
    assert set(sites) == {"1:65620:A:C"}
    assert sites["1:65620:A:C"].af == pytest.approx(0.25)  # unflipped (A is canonical A1)


def test_extract_at_sites_region_restricts_the_scan(tmp_path):
    """``region`` (bcftools -r) is a cheap chromosome-level restriction, kept
    from the pre-#21 API for callers like a genome-wide dev-batch script that
    only need one chromosome (issue #21 regression safety). bcftools rejects
    -r/-R together, so this is exercised the way real callers pair it: with
    liftover, not with the default -R regions file."""
    vcf = _bgzip_index(
        _af_se_vcf(
            tmp_path,
            [
                "1\t1000\t.\tC\tA\t.\tPASS\t.\tAF:SE\t0.30:0.1\n",  # -> hg38 1:65620
                "2\t1000\t.\tC\tA\t.\tPASS\t.\tAF:SE\t0.40:0.1\n",  # chr2, excluded by region
            ],
        )
    )
    wanted = {"1:65620:A:C", "2:65620:A:C"}
    sites = GwasVcfReader(
        vcf, StoredEffectScale.SD, liftover=_FakeLiftover(), region="1"
    ).extract_at_sites(wanted)
    assert set(sites) == {"1:65620:A:C"}


def test_af_only_projects_out_se():
    sites = {"1:100:A:G": SiteMetrics(af=0.3, se=0.1), "1:200:A:G": SiteMetrics(af=0.7, se=0.2)}
    assert af_only(sites) == {"1:100:A:G": 0.3, "1:200:A:G": 0.7}


# --- site_metrics_arrays + phenotype-SD estimator, end-to-end (issue #21 AC4) ---


def test_phenotype_sd_estimator_driven_end_to_end_from_a_real_analysis(tmp_path):
    """Ancestry assignment and SD estimation share one extraction call: this
    drives `estimate_phenotype_sd` entirely from `GwasVcfReader.extract_at_sites`
    output for a real (fixture) GWAS-VCF, not synthetic se/af arrays built by
    hand as `tests/test_phenotype_sd.py` does."""
    rng = np.random.default_rng(7)
    n = 300
    sample_size = 20_000.0
    sd_true = 2.5
    afs = rng.uniform(0.05, 0.5, n)
    # se implied by the asymptotic SE model the ADR-0029 estimator inverts:
    # se = sd / sqrt(2*af*(1-af)*N).
    ses = sd_true / np.sqrt(2.0 * afs * (1.0 - afs) * sample_size)

    rows = [
        f"1\t{1000 + i}\t.\tC\tA\t.\tPASS\t.\tAF:SE\t{af:.6g}:{se:.6g}\n"
        for i, (af, se) in enumerate(zip(afs, ses, strict=True))
    ]
    vcf = _bgzip_index(_af_se_vcf(tmp_path, rows))
    wanted = {f"1:{1000 + i}:A:C" for i in range(n)}

    sites = GwasVcfReader(vcf, StoredEffectScale.SD).extract_at_sites(wanted)
    assert len(sites) == n

    se_arr, af_arr = site_metrics_arrays(sites)
    result = estimate_phenotype_sd(
        OriginalSdMethod.ESTIMATED_FROM_SOURCE_MAF, sample_size, se=se_arr, af=af_arr
    )

    assert result.method is OriginalSdMethod.ESTIMATED_FROM_SOURCE_MAF
    assert result.sd == pytest.approx(sd_true, rel=1e-2)


# --- FakeReader: usable with no bcftools/fixture VCF dependency ---


def test_fake_reader_needs_no_bcftools_or_fixture_vcf():
    reader = FakeReader(
        associations=[
            ReaderAssociation(
                chromosome="1",
                position=100,
                ref="A",
                alt="G",
                z=1.0,
                se=0.1,
                stored_effect_scale=StoredEffectScale.SD,
            )
        ]
    )

    associations = list(reader.stream_associations())

    assert len(associations) == 1
    assert associations[0].z == 1.0


# --- Metrics projection: the one-pass resolver's row stream (issue #207) ---


def _metrics_fields(rows) -> list[tuple]:
    """Rows as comparable tuples, so two streams can be compared field for field.

    `rsid` is deliberately not one of them: it is the one field the metrics seam
    does not carry, and the parity claim is about the nine it does.
    """
    return [
        (
            row.chromosome,
            row.position,
            row.ref,
            row.alt,
            row.alid,
            row.flipped,
            row.af_alt,
            row.beta,
            row.se,
        )
        for row in rows
    ]


def _metrics_rows(path: Path) -> list[tuple]:
    return _metrics_fields(GwasSsfReader(path).stream_metrics())


def _reference_metrics_rows(path: Path) -> list[tuple]:
    return _metrics_fields(gwas_ssf_module.stream_full_row_metrics(path))


def _write_metrics_fixture(path: Path, rows: list[str], header: str | None = None) -> None:
    columns = header or "\t".join(_SSF_HEADER)
    path.write_text(columns + "\n" + "".join(row + "\n" for row in rows), encoding="utf-8")


# Every row here is one the projection has to agree about: a plain row, a row
# whose effect allele is not the canonical A1 (so `flipped` is load-bearing), a
# lowercase allele, a `chr`-prefixed chromosome, an indel, and rows the
# full-row parser drops outright (bad position, identical alleles, too few
# cells). A projection tested only on well-formed rows would not have been
# tested at all.
_METRICS_FIXTURE_ROWS = [
    "1\t100\tA\tC\t0.10\t0.20\t0.30",
    "1\t200\tC\tA\t0.11\t0.21\t0.31",
    "1\t300\ta\tc\t0.12\t0.22\t0.32",
    "chr1\t400\tA\tC\t0.13\t0.23\t0.33",
    "1\t500\tA\tCG\t0.14\t0.24\t0.34",
    "1\t0\tA\tC\t0.15\t0.25\t0.35",
    "1\t600\tA\tA\t0.16\t0.26\t0.36",
    "1\t700\tA\tC\t0.17\t0.27",
    "1\t800\tA\tC\tNA\t0.28\t1.4",
    "1\t900\tA\tC\t0.19\tNA\t0.39",
]


def test_gwas_ssf_metrics_projection_agrees_with_the_full_row_parser(tmp_path):
    path = tmp_path / "metrics.tsv"
    _write_metrics_fixture(path, _METRICS_FIXTURE_ROWS)

    projected = _metrics_rows(path)
    reference = _reference_metrics_rows(path)

    # The fixture must actually reach the interesting rows before either stream
    # is compared: a fixture of dropped rows would compare two empty lists.
    assert len(reference) == 8, "fixture must keep the rows it is meant to test"
    assert any(row[5] for row in reference), "fixture must include a flipped row"
    assert any(row[0] == "1" and row[3] == "a" for row in reference), (
        "fixture must include a lowercase allele row"
    )
    assert any(row[2] == "CG" for row in reference), "fixture must include an indel"
    assert projected == reference


def test_gwas_ssf_metrics_projection_without_optional_statistics_columns(tmp_path):
    """A file with no `effect_allele_frequency` still names its variants."""
    path = tmp_path / "no-frequency.tsv"
    _write_metrics_fixture(
        path,
        ["1\t100\tA\tC\t0.10\t0.20", "1\t200\tC\tA\t0.11\t0.21"],
        header="\t".join(column for column in _SSF_HEADER if column != "effect_allele_frequency"),
    )

    projected = _metrics_rows(path)

    assert projected == _reference_metrics_rows(path)
    assert [row[6] for row in projected] == [None, None]


def test_gwas_ssf_metrics_projection_keeps_rows_the_association_stream_drops(tmp_path):
    """The reason the metrics seam exists, asserted rather than assumed.

    A row carrying a frequency and a standard error but no usable beta is
    evidence for ancestry assignment and none for the SD estimator; the
    association stream drops it, so a resolver built on that stream would lose
    ancestry evidence the source actually carries.
    """
    path = tmp_path / "no-beta-column.tsv"
    _write_metrics_fixture(
        path,
        ["1\t100\tA\tC\t0.20\t0.30"],
        header="\t".join(column for column in _SSF_HEADER if column != "beta"),
    )

    metrics = _metrics_rows(path)
    associations = list(GwasSsfReader(path).stream_associations())

    assert len(metrics) == 1, "the metrics stream must see the row"
    assert metrics[0][7] is None, "the row has no beta, and none is fabricated"
    assert metrics[0][6] == pytest.approx(0.30)
    assert associations == [], "the association stream must drop the same row"


def test_metrics_projection_fails_loudly_when_identity_column_is_missing(tmp_path):
    path = tmp_path / "missing-allele-column.tsv"
    _write_metrics_fixture(
        path,
        ["1\t100\tA\tC\t0.10\t0.20\t0.30"],
        header="\t".join(column for column in _SSF_HEADER if column != "other_allele"),
    )

    with pytest.raises(ValueError, match=r"missing required variant columns"):
        list(GwasSsfReader(path).stream_metrics())


def test_metrics_projection_preserves_a_quoted_final_projected_column(tmp_path):
    path = tmp_path / "quoted-metrics.tsv"
    _write_metrics_fixture(path, ['1\t100\tA\tC\t0.10\t0.20\t"0.30"'])

    assert _metrics_rows(path) == _reference_metrics_rows(path)


def test_metrics_projection_survives_a_row_shorter_than_the_header(tmp_path):
    """A truncated row is a row with missing cells, never a borrowed one."""
    path = tmp_path / "short-row.tsv"
    _write_metrics_fixture(path, ["1\t100\tA", "1\t200\tA\tC\t0.11\t0.21\t0.31"])

    projected = _metrics_rows(path)

    assert len(projected) == 1, "the truncated row names no usable variant"
    assert projected == _reference_metrics_rows(path)


# --- Effect source: beta or odds_ratio (issue #213) ---


def _effect_header(effect_column: str) -> str:
    """`_SSF_HEADER` with the effect column renamed, order preserved."""
    return "\t".join(effect_column if column == "beta" else column for column in _SSF_HEADER)


_ODDS_RATIO_HEADER = _effect_header("odds_ratio")


def test_gwas_ssf_reader_reports_a_beta_file_as_beta(tmp_path):
    """The rule for files that already carry `beta` is explicit, not incidental."""
    path = tmp_path / "beta.tsv"
    _write_metrics_fixture(path, ["1\t100\tA\tG\t0.25\t0.50\t0.30"])

    reader = GwasSsfReader(path, StoredEffectScale.SD)

    assert reader.effect_source == EffectSource(
        column_name="beta",
        kind=EffectSourceKind.BETA,
        is_derived=False,
        assumes_standardised=False,
    )


def test_gwas_ssf_reader_reads_odds_ratio_as_log_or_beta(tmp_path):
    """An odds-ratio file yields a usable beta on the log-OR scale."""
    path = tmp_path / "odds-ratio.tsv"
    _write_metrics_fixture(path, ["1\t100\tA\tG\t2.0\t0.50\t0.30"], header=_ODDS_RATIO_HEADER)

    reader = GwasSsfReader(path, StoredEffectScale.LOG_OR)

    assert reader.effect_source == EffectSource(
        column_name="odds_ratio",
        kind=EffectSourceKind.ODDS_RATIO,
        is_derived=True,
        assumes_standardised=False,
    )

    associations = list(reader.stream_associations())
    assert len(associations) == 1, "the odds-ratio row must reach the association stream"
    assert associations[0].z == pytest.approx(math.log(2.0) / 0.50)
    # The standard error is already on the log scale in GWAS-SSF; it is carried
    # through unchanged, never rescaled by the log transform.
    assert associations[0].se == 0.50
    assert associations[0].eaf == 0.30


def test_gwas_ssf_reader_reads_odds_ratio_below_one_as_a_negative_beta(tmp_path):
    """`log(0.5) < 0`, so the effect direction survives the transform."""
    path = tmp_path / "protective.tsv"
    _write_metrics_fixture(path, ["1\t100\tA\tG\t0.5\t0.50\t0.30"], header=_ODDS_RATIO_HEADER)

    associations = list(GwasSsfReader(path, StoredEffectScale.LOG_OR).stream_associations())

    assert len(associations) == 1
    assert associations[0].z == pytest.approx(math.log(0.5) / 0.50)
    assert associations[0].z < 0.0


def test_gwas_ssf_reader_drops_a_non_positive_or_unparseable_odds_ratio(tmp_path):
    """An unusable `odds_ratio` drops the row exactly as an unusable `beta` does."""
    path = tmp_path / "unusable.tsv"
    _write_metrics_fixture(
        path,
        [
            "1\t100\tA\tG\t0\t0.50\t0.30",  # zero: log undefined
            "1\t200\tA\tC\t-2.0\t0.50\t0.30",  # negative: log undefined
            "1\t300\tA\tC\tbad\t0.50\t0.30",  # unparseable
            "1\t400\tA\tC\tNA\t0.50\t0.30",  # missing
            "1\t500\tA\tC\t2.0\t0.50\t0.30",  # usable
        ],
        header=_ODDS_RATIO_HEADER,
    )

    associations = list(GwasSsfReader(path, StoredEffectScale.LOG_OR).stream_associations())

    assert [association.position for association in associations] == [500], (
        "only the usable odds-ratio row may reach the association stream"
    )
    # The metrics seam still sees every row, with the unusable beta absent.
    assert [row[7] for row in _metrics_rows(path)] == [
        None, None, None, None, pytest.approx(math.log(2.0))
    ]


def test_gwas_ssf_reader_prefers_beta_when_both_effect_columns_are_present(tmp_path):
    """The explicit precedence rule, tested rather than assumed.

    `odds_ratio=100.0` would give a very different beta; reading the row and
    getting `beta`'s answer is what proves the precedence is real.
    """
    path = tmp_path / "both.tsv"
    header = "\t".join(_SSF_HEADER) + "\todds_ratio"
    _write_metrics_fixture(path, ["1\t100\tA\tG\t0.25\t0.50\t0.30\t100.0"], header=header)

    reader = GwasSsfReader(path, StoredEffectScale.SD)

    assert reader.effect_source is not None
    assert reader.effect_source.kind is EffectSourceKind.BETA
    associations = list(reader.stream_associations())
    assert len(associations) == 1
    assert associations[0].z == pytest.approx(0.25 / 0.50)


def test_gwas_ssf_reader_rejects_a_duplicate_effect_column(tmp_path):
    """`GCST006329` carries `beta` twice; last-wins would be a silent wrong answer."""
    path = tmp_path / "duplicate.tsv"
    header = "\t".join(_SSF_HEADER) + "\tbeta"
    _write_metrics_fixture(path, ["1\t100\tA\tG\t0.25\t0.50\t0.30\t9.99"], header=header)

    reader = GwasSsfReader(path, StoredEffectScale.SD)

    with pytest.raises(ValueError, match=r"Duplicate effect column 'beta' in header"):
        _ = reader.effect_source
    with pytest.raises(ValueError, match=r"Duplicate effect column 'beta' in header"):
        list(reader.stream_associations())


def test_gwas_ssf_reader_reports_no_effect_source_when_the_file_names_none(tmp_path):
    """A file with neither column is reported as having none, not guessed at."""
    path = tmp_path / "no-effect.tsv"
    header = "\t".join(column for column in _SSF_HEADER if column != "beta")
    _write_metrics_fixture(path, ["1\t100\tA\tG\t0.50\t0.30"], header=header)

    reader = GwasSsfReader(path, StoredEffectScale.SD)

    assert reader.effect_source is None
    assert list(reader.stream_associations()) == []


def test_gwas_ssf_odds_ratio_metrics_projection_matches_the_full_row_parser(tmp_path):
    """The projection and the reference parser agree on the log-OR beta."""
    path = tmp_path / "or-metrics.tsv"
    _write_metrics_fixture(
        path,
        [
            "1\t100\tA\tG\t2.0\t0.50\t0.30",
            "1\t200\tA\tC\t0.5\t0.25\t0.40",
            "1\t300\tA\tC\t0\t0.25\t0.40",
        ],
        header=_ODDS_RATIO_HEADER,
    )

    projected = _metrics_rows(path)
    reference = _reference_metrics_rows(path)

    assert len(reference) == 3, "fixture must keep the rows it is meant to test"
    assert projected == reference
    assert [row[7] for row in projected] == [
        pytest.approx(math.log(2.0)),
        pytest.approx(math.log(0.5)),
        None,
    ]


def test_gwas_ssf_reader_rejects_the_gcst006329_padded_beta_duplicate(tmp_path):
    """The real `GCST006329` header: `beta ` holds the values, `beta` is all `NA`.

    Matching the two exactly made them different columns, so `row.get("beta")`
    read the empty one and every row silently dropped from the association
    stream. Whitespace is not part of a column's name; the ambiguity is
    refused instead of resolved.
    """
    path = tmp_path / "gcst006329-shaped.tsv"
    header = (
        "chromosome\tbase_pair_location\teffect_allele\tother_allele\t"
        "beta \tstandard_error\teffect_allele_frequency\tbeta"
    )
    _write_metrics_fixture(path, ["1\t100\tA\tG\t0.0458\t0.1342\t0.033\tNA"], header=header)

    reader = GwasSsfReader(path, StoredEffectScale.SD)

    with pytest.raises(ValueError, match=r"Duplicate effect column 'beta' in header"):
        _ = reader.effect_source
    with pytest.raises(ValueError, match=r"Duplicate effect column 'beta' in header"):
        list(reader.stream_associations())
    with pytest.raises(ValueError, match=r"Duplicate effect column 'beta' in header"):
        list(reader.stream_metrics())


def test_gwas_ssf_reader_reads_a_whitespace_padded_odds_ratio(tmp_path):
    """A lone padded spelling is still the column it names, and is read by it."""
    path = tmp_path / "padded-odds-ratio.tsv"
    header = (
        "chromosome\tbase_pair_location\teffect_allele\tother_allele\t"
        "odds_ratio \tstandard_error\teffect_allele_frequency"
    )
    _write_metrics_fixture(path, ["1\t100\tA\tG\t2.0\t0.50\t0.30"], header=header)

    reader = GwasSsfReader(path, StoredEffectScale.LOG_OR)

    assert reader.effect_source is not None
    assert reader.effect_source.column_name == "odds_ratio "
    associations = list(reader.stream_associations())
    assert len(associations) == 1
    assert associations[0].z == pytest.approx(math.log(2.0) / 0.50)
