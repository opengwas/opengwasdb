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
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from opengwasdb.model.enums import StoredEffectScale
from opengwasdb.readers.gwas_vcf import GwasVcfReader, _record_rows
from opengwasdb.readers.interface import StreamTally
from opengwasdb.store_reader.identity import VariantNormalisationError
from opengwasdb.variants.normalise import orient_to_canonical, oriented_alid

#: The FORMAT meta-lines a GWAS-VCF declares when it carries both spellings of
#: an effect. `_write_vcf` takes them as a parameter so a test can write a file
#: whose header cannot give a record its z (issue #363).
_ES_SE_EZ = (
    '##FORMAT=<ID=ES,Number=A,Type=Float,Description="Effect size relative to ALT">\n'
    '##FORMAT=<ID=SE,Number=A,Type=Float,Description="Standard error of effect size">\n'
    '##FORMAT=<ID=EZ,Number=A,Type=Float,Description="Z-score, if used to derive EFFECT/SE">\n'
)
_ES_ONLY = '##FORMAT=<ID=ES,Number=A,Type=Float,Description="Effect size relative to ALT">\n'
_EZ_ONLY = '##FORMAT=<ID=EZ,Number=A,Type=Float,Description="Z-score">\n'
_AF_ONLY = '##FORMAT=<ID=AF,Number=A,Type=Float,Description="Alternate allele frequency">\n'


def _write_vcf(
    path: Path,
    body: str,
    study_type: str = "Continuous",
    samples: str = "STUDY1",
    formats: str = _ES_SE_EZ,
) -> None:
    """Write a minimal GWAS-VCF fixture to ``path``."""
    header = (
        "##fileformat=VCFv4.2\n"
        '##FILTER=<ID=PASS,Description="All filters passed">\n'
        f"{formats}"
        f"##SAMPLE=<ID=STUDY1,StudyType={study_type}>\n"
        f"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{samples}\n"
    )
    path.write_text(header + body, encoding="utf-8")


def _declared_format_ids(vcf: Path) -> set[str]:
    """The FORMAT tags the fixture's own text declares, read back without the reader."""
    prefix = "##FORMAT=<ID="
    return {
        line[len(prefix) :].split(",", 1)[0]
        for line in vcf.read_text(encoding="utf-8").splitlines()
        if line.startswith(prefix)
    }


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
            return_value="%CHROM\t%POS\t%ID\t%REF\t%ALT\t[%XX]\t.\t.\t.\t.\n",
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


# --- Refusals that name the file (issue #363) ---

#: Every projection that turns records into associations: the two the builders
#: store, and the resolver's metric blocks.
_RECORD_PROJECTIONS = ("associations", "batches", "metric_chunks")


def _read(reader: GwasVcfReader, projection: str) -> list:
    stream = {
        "associations": reader.stream_associations,
        "batches": reader.stream_batches,
        "metric_chunks": reader.stream_metric_chunks,
    }[projection]
    return list(stream())


@pytest.mark.parametrize("projection", _RECORD_PROJECTIONS)
def test_a_header_without_an_effect_field_is_refused_naming_the_file(
    tmp_path: Path, projection: str
) -> None:
    """No EZ and no ES/SE is not a source of 0 associations: it is a broken file."""
    vcf = tmp_path / "no-effect.vcf"
    _write_vcf(vcf, "1\t100\t.\tA\tG\t.\tPASS\t.\tAF\t0.3\n", formats=_AF_ONLY)
    # The fixture must really lack all three tags for this to mean anything.
    assert _declared_format_ids(vcf) == {"AF"}
    with pytest.raises(ValueError) as raised:
        _read(GwasVcfReader(vcf), projection)
    message = str(raised.value)
    assert str(vcf) in message
    assert "FORMAT/EZ" in message and "FORMAT/SE" in message
    assert "AF" in message  # names the tags it did find


@pytest.mark.parametrize("projection", _RECORD_PROJECTIONS)
def test_a_header_with_es_but_no_se_is_refused(tmp_path: Path, projection: str) -> None:
    """ES without SE derives no z, so it is refused as the missing pair it is."""
    vcf = tmp_path / "es-without-se.vcf"
    _write_vcf(vcf, "1\t100\t.\tA\tG\t.\tPASS\t.\tES\t0.5\n", formats=_ES_ONLY)
    assert _declared_format_ids(vcf) == {"ES"}
    assert "SE" not in _declared_format_ids(vcf)
    with pytest.raises(ValueError, match="FORMAT/ES with FORMAT/SE") as raised:
        _read(GwasVcfReader(vcf), projection)
    assert str(vcf) in str(raised.value)


def test_a_header_declaring_only_ez_reads_as_today(tmp_path: Path) -> None:
    vcf = tmp_path / "ez-only.vcf"
    _write_vcf(vcf, "1\t100\t.\tA\tG\t.\tPASS\t.\tEZ\t5.25\n", formats=_EZ_ONLY)
    assert _declared_format_ids(vcf) == {"EZ"}
    (association,) = _associations(vcf)
    assert association.z == -5.25
    assert math.isnan(association.se)  # no SE is reported, not a zero


def test_a_declared_effect_holding_a_missing_value_is_a_missing_association(
    tmp_path: Path,
) -> None:
    """A record's `.` is a missing association, not a broken file (issue #363)."""
    vcf = tmp_path / "missing-cells.vcf"
    _write_vcf(
        vcf,
        "1\t100\t.\tA\tG\t.\tPASS\t.\tEZ:ES:SE\t.:.:.\n"
        "1\t200\t.\tA\tG\t.\tPASS\t.\tEZ:ES:SE\t2.0:0.5:0.25\n",
    )
    assert {"EZ", "ES", "SE"} <= _declared_format_ids(vcf)
    first, second = _associations(vcf)
    assert math.isnan(first.z) and math.isnan(first.se)
    assert second.z == -2.0 and second.se == 0.25


_HEADER_FAILS = """\
#!{python}
import sys

sys.stderr.write("synthetic bcftools header failure\\n")
raise SystemExit(3)
"""

_RECORDS_FAIL = """\
#!{python}
import sys

if sys.argv[1] == "view":
    with open(sys.argv[-1], encoding="utf-8") as handle:
        sys.stdout.write("".join(line for line in handle if line.startswith("#")))
    raise SystemExit(0)
sys.stderr.write("synthetic bcftools record failure\\n")
raise SystemExit(7)
"""


def _install_stub_bcftools(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, script: str) -> Path:
    """Put a fake ``bcftools`` first on PATH, ahead of the real one."""
    binary = tmp_path / "stub-bin" / "bcftools"
    binary.parent.mkdir()
    binary.write_text(script.format(python=sys.executable), encoding="utf-8")
    binary.chmod(0o755)
    monkeypatch.setenv("PATH", f"{binary.parent}{os.pathsep}{os.environ['PATH']}")
    return binary


def test_a_failing_header_probe_names_the_file(tmp_path: Path, monkeypatch) -> None:
    vcf = tmp_path / "study.vcf"
    _write_vcf(vcf, "1\t100\t.\tA\tG\t.\tPASS\t.\tES:SE\t0.5:0.1\n")
    # The fixture is readable: what fails below is bcftools, not the file.
    assert len(_associations(vcf)) == 1
    _install_stub_bcftools(tmp_path, monkeypatch, _HEADER_FAILS)
    with pytest.raises(OSError) as raised:
        _associations(vcf)
    message = str(raised.value)
    assert str(vcf) in message
    assert "bcftools header query failed (3)" in message
    assert "synthetic bcftools header failure" in message


def test_a_failing_record_query_names_the_file(tmp_path: Path, monkeypatch) -> None:
    vcf = tmp_path / "study.vcf"
    _write_vcf(vcf, "1\t100\t.\tA\tG\t.\tPASS\t.\tES:SE\t0.5:0.1\n")
    assert len(_associations(vcf)) == 1
    _install_stub_bcftools(tmp_path, monkeypatch, _RECORDS_FAIL)
    with pytest.raises(OSError) as raised:
        _associations(vcf)
    message = str(raised.value)
    assert str(vcf) in message
    assert "bcftools record query failed (7)" in message
    assert "synthetic bcftools record failure" in message


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


# ── LP, the p-value the effect-consistency check compares (#510) ─────────────

_ES_SE_LP = (
    '##FORMAT=<ID=ES,Number=A,Type=Float,Description="Effect size relative to ALT">\n'
    '##FORMAT=<ID=SE,Number=A,Type=Float,Description="Standard error of effect size">\n'
    '##FORMAT=<ID=LP,Number=A,Type=Float,Description="-log10 p-value for effect estimate">\n'
)


def test_the_metric_blocks_carry_each_record_s_p_as_ten_to_the_minus_lp(tmp_path: Path) -> None:
    vcf = tmp_path / "lp.vcf"
    _write_vcf(
        vcf,
        "1\t100\t.\tA\tG\t.\tPASS\t.\tES:SE:LP\t0.5:0.1:5.5\n"
        "1\t200\t.\tA\tG\t.\tPASS\t.\tES:SE:LP\t0.1:0.1:.\n"
        "1\t300\t.\tA\tG\t.\tPASS\t.\tES:SE:LP\t0.1:0.1:0.30103\n",
        formats=_ES_SE_LP,
    )
    [chunk] = GwasVcfReader(vcf).stream_metric_chunks()
    assert chunk.p_value[0] == pytest.approx(10**-5.5)
    assert math.isnan(chunk.p_value[1])
    assert chunk.p_value[2] == pytest.approx(0.5, rel=1e-5)


def test_a_vcf_without_lp_gives_no_p_and_the_associations_read_none(tmp_path: Path) -> None:
    """No LP declared: every block's p is missing (`no_source_p_value`), and the
    association path, which never needs a p, does not query LP even where declared."""
    vcf = tmp_path / "es_se.vcf"
    _write_vcf(vcf, "1\t100\t.\tA\tG\t.\tPASS\t.\tES:SE:EZ\t0.5:0.1:5\n")
    [chunk] = GwasVcfReader(vcf).stream_metric_chunks()
    assert math.isnan(chunk.p_value[0])
    with_lp = tmp_path / "lp.vcf"
    _write_vcf(with_lp, "1\t100\t.\tA\tG\t.\tPASS\t.\tES:SE:LP\t0.5:0.1:5.5\n", formats=_ES_SE_LP)
    [row] = list(_record_rows(with_lp))
    assert row.p_value is None
