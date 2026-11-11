"""The GWAS-VCF reader's one record query and the rows it parses (#56).

Every projection but the variant one reads the file through one streaming
``bcftools query``, so these drive the reader itself: its z (`EZ`, else
`ES/SE`), the flip onto A1, and what it refuses. `stored_effect_scale` is
never read from the header (issue #17); `study_type` is in the fixture only as
the realism of a ``##SAMPLE`` line.
"""

from __future__ import annotations

import itertools
import math
from pathlib import Path
from unittest.mock import patch

import pytest

from opengwasdb.model.enums import StoredEffectScale
from opengwasdb.readers.gwas_vcf import GwasVcfReader, _record_rows
from opengwasdb.readers.interface import StreamTally
from opengwasdb.variants.normalise import (
    VariantNormalisationError,
    orient_to_canonical,
    oriented_alid,
)


def _write_vcf(
    path: Path, body: str, study_type: str = "Continuous", samples: str = "STUDY1"
) -> None:
    """Write a minimal GWAS-VCF fixture to ``path``."""
    header = f"""\
##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##FORMAT=<ID=ES,Number=A,Type=Float,Description="Effect size relative to ALT">
##FORMAT=<ID=SE,Number=A,Type=Float,Description="Standard error of effect size">
##FORMAT=<ID=EZ,Number=A,Type=Float,Description="Z-score, if used to derive EFFECT/SE">
##SAMPLE=<ID=STUDY1,StudyType={study_type}>
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{samples}
"""
    path.write_text(header + body, encoding="utf-8")


def _associations(vcf: Path) -> list:
    return list(GwasVcfReader(vcf, StoredEffectScale.SD).stream_associations())


def test_a_reported_ez_is_the_stored_z_bit_for_bit(tmp_path: Path) -> None:
    # REF=A, ALT=G: the effect allele G is A2, so the stored z is -EZ, never
    # rebuilt as ES/SE (5.0).
    vcf = tmp_path / "test.vcf"
    _write_vcf(vcf, "1\t100\t.\tA\tG\t.\tPASS\t.\tEZ:ES:SE\t5.25:0.5:0.1\n")
    (association,) = _associations(vcf)
    assert association.z == -5.25


@pytest.mark.parametrize(
    ("ref", "alt", "z"), [("A", "G", -2.0), ("G", "A", 2.0), ("g", "A", 2.0)]
)
def test_es_over_se_is_signed_onto_a1(tmp_path: Path, ref: str, alt: str, z: float) -> None:
    vcf = tmp_path / "test.vcf"
    _write_vcf(vcf, f"1\t100\t.\t{ref}\t{alt}\t.\tPASS\t.\tES:SE\t1.0:0.5\n")
    (association,) = _associations(vcf)
    assert association.z == z
    assert association.se == 0.5
    # The file's own spelling is kept.
    assert (association.ref, association.alt) == (ref, alt)


def test_an_unusable_se_leaves_no_z_and_no_se_for_admission(tmp_path: Path) -> None:
    vcf = tmp_path / "test.vcf"
    _write_vcf(
        vcf,
        "1\t100\t.\tA\tG\t.\tPASS\t.\tES:SE\t1.0:0.0\n"
        "1\t200\t.\tC\tT\t.\tPASS\t.\tES:SE\t0.5:0.1\n",
    )
    first, second = _associations(vcf)
    assert math.isnan(first.z) and math.isnan(first.se)
    assert second.z == pytest.approx(-5.0)
    assert first.eaf is None  # the file declares no AF


def test_multi_allelic_records_are_skipped_and_a_chr_prefix_normalised(tmp_path: Path) -> None:
    vcf = tmp_path / "test.vcf"
    _write_vcf(
        vcf,
        "chr1\t100\trs1\tA\tG,T\t.\tPASS\t.\tES:SE\t0.5:0.1\n"
        "chr1\t200\trs2\tC\tT\t.\tPASS\t.\tES:SE\t0.3:0.15\n",
    )
    reader = GwasVcfReader(vcf)
    assert [(v.chromosome, v.position, v.rsid) for v in reader.stream_variants()] == [
        ("1", 200, "rs2")
    ]
    assert [(a.chromosome, a.position) for a in reader.stream_associations()] == [("1", 200)]


def test_the_variant_query_yields_exactly_the_rows_records(tmp_path: Path) -> None:
    """The fast variant projection and the record rows drop the same records."""
    vcf = tmp_path / "test.vcf"
    _write_vcf(
        vcf,
        "1\t100\trs1\tA\tG\t.\tPASS\t.\tES:SE\t0.5:0.1\n"
        "1\t150\t.\tA\ta\t.\tPASS\t.\tES:SE\t0.5:0.1\n"  # one allele, two spellings
        "1\t175\t.\tA\tN\t.\tPASS\t.\tES:SE\t0.5:0.1\n"  # malformed identity
        "NA\t180\t.\tA\tG\t.\tPASS\t.\tES:SE\t0.5:0.1\n"  # missing chromosome
        "2\t200\tfoo\tCT\tC\t.\tPASS\t.\tES:SE\t0.3:0.15\n",
    )
    variants = [v.site for v in GwasVcfReader(vcf).stream_variants()]
    tally = StreamTally()
    rows = list(_record_rows(vcf, tally=tally))
    assert variants == [(r.chromosome, r.position, r.ref, r.alt) for r in rows]
    assert variants == [("1", 100, "A", "G"), ("2", 200, "CT", "C")]
    assert [row.source_id for row in rows] == ["rs1", "foo"]
    assert [reason for _key, reason in tally.faults.rows] == [
        "allele_invalid", "chromosome_missing"
    ]


def test_a_vcf_with_two_samples_is_refused(tmp_path: Path) -> None:
    vcf = tmp_path / "test.vcf"
    _write_vcf(
        vcf, "1\t100\t.\tA\tG\t.\tPASS\t.\tES:SE\t0.5:0.1\t0.4:0.1\n", samples="S1\tS2"
    )
    with pytest.raises(ValueError, match="exactly one sample column, found 2"):
        _associations(vcf)


def test_a_failing_query_raises_rather_than_yield_a_short_stream(tmp_path: Path) -> None:
    vcf = tmp_path / "test.vcf"
    _write_vcf(vcf, "1\t100\t.\tA\tG\t.\tPASS\t.\tES:SE\t0.5:0.1\n")
    reader = GwasVcfReader(vcf)
    with patch("opengwasdb.readers.gwas_vcf._format_tags", return_value={"ES", "SE", "XX"}):
        with patch(
            "opengwasdb.readers.gwas_vcf._record_query_format",
            return_value="%CHROM\t%POS\t%ID\t%REF\t%ALT\t[%XX]\t.\t.\t.\n",
        ):
            with pytest.raises(OSError, match="bcftools record query failed"):
                list(reader.stream_associations())


def test_every_projection_needs_bcftools(tmp_path: Path) -> None:
    vcf = tmp_path / "test.vcf"
    _write_vcf(vcf, "1\t100\t.\tA\tG\t.\tPASS\t.\tES:SE\t1.0:0.5\n")
    reader = GwasVcfReader(vcf)
    for stream in (reader.stream_variants, reader.stream_associations):
        with patch("shutil.which", return_value=None):
            with pytest.raises(RuntimeError, match="bcftools"):
                list(stream())


def test_oriented_alid_is_orient_to_canonicals_answer() -> None:
    """The VCF reader's fast orientation, against the rule it stands in for."""
    alleles = ["A", "c", "G", "t", "AC", "Tg", "GGGGCC", "g"]
    declared_indels = 0
    for effect, other in itertools.product(alleles, repeat=2):
        for declared in (True, False):
            try:
                expected = orient_to_canonical(
                    "1", 100, effect, other, other_is_reference=declared
                )
            except VariantNormalisationError:
                assert oriented_alid("1", 100, effect, other, other_is_reference=declared) is None
                continue
            declared_indels += declared and len(effect) != len(other)
            assert oriented_alid("1", 100, effect, other, other_is_reference=declared) == (
                expected.variant.alid, expected.flipped
            )
    # The pairs must reach a declared indel, whose ALID is spelt reference first.
    assert declared_indels
    assert oriented_alid("1", 100, "G", "GGGGCC", other_is_reference=True) == (
        "1:100:GGGGCC:G", True
    )
