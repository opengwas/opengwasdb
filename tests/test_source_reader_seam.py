"""Every Source Reader presents every projection, through the registry (#56).

Each reader's batches are its association stream as columns; each carries its
format's declaration of the reference allele unchanged; a lift is the
registry's to apply and only a capability that can lift accepts one; and a
BESD file set is a registered reader, one probe per Analysis, its `.esi` read
once for every probe.
"""

from __future__ import annotations

import gzip
from dataclasses import replace
from pathlib import Path
from subprocess import run

import numpy as np
import pytest

from opengwasdb.model.enums import StoredEffectScale
from opengwasdb.readers import ReferenceAllele
from opengwasdb.readers.assembly import AssemblyPlan, assembly_plan
from opengwasdb.readers.association_tsv import ASSOCIATION_TSV_CAPABILITY, AssociationTsvReader
from opengwasdb.readers.besd import BESD_CAPABILITY, BesdProbeReader, load_besd_source
from opengwasdb.readers.column_map import COLUMN_MAP_CAPABILITY, load_column_map
from opengwasdb.readers.column_map_reader import ColumnMapReader
from opengwasdb.readers.finngen import FINNGEN_R13_CAPABILITY, FinnGenR13Reader
from opengwasdb.readers.gwas_ssf import GWAS_SSF_CAPABILITY, GwasSsfReader
from opengwasdb.readers.gwas_vcf import GWAS_VCF_CAPABILITY, GwasVcfReader
from opengwasdb.readers.interface import (
    ImputationScoreDeclaration,
    ImputationScoreKind,
    SourceReader,
    StreamTally,
)
from opengwasdb.readers.registry import SourceSpec, open_source, resolve_reader
from opengwasdb.readers.rows import SourceRow, stream_batches
from tests.besd_fixtures import write_besd_sparse_3f, write_epi, write_esi

_VCF_HEADER = """\
##fileformat=VCFv4.2
##FORMAT=<ID=ES,Number=A,Type=Float,Description="Effect size">
##FORMAT=<ID=SE,Number=A,Type=Float,Description="Standard error">
##FORMAT=<ID=AF,Number=A,Type=Float,Description="Alt frequency">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSTUDY1
"""


def _first(lower: bool) -> tuple[str, str]:
    """The first row's ref and alt, spelt in lower case when ``lower``."""
    return ("a", "g") if lower else ("A", "G")


def _vcf(tmp_path: Path, *, lower: bool = False) -> Path:
    ref, alt = _first(lower)
    path = tmp_path / "study.vcf"
    path.write_text(
        _VCF_HEADER
        + f"1\t100\trs1\t{ref}\t{alt}\t.\tPASS\t.\tES:SE:AF\t0.5:0.1:0.2\n"
        + "1\t200\trs2\tGGGGCC\tG\t.\tPASS\t.\tES:SE:AF\t-0.3:0.15:0.4\n"
        + "2\t300\t.\tT\tC\t.\tPASS\t.\tES:SE:AF\t0.2:0.0:0.6\n",
        encoding="utf-8",
    )
    return path


def _ssf(tmp_path: Path, *, lower: bool = False) -> Path:
    other, effect = _first(lower)
    path = tmp_path / "study.tsv.gz"
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write(
            "chromosome\tbase_pair_location\teffect_allele\tother_allele\tbeta\t"
            "standard_error\teffect_allele_frequency\trsid\n"
            f"1\t100\t{effect}\t{other}\t0.5\t0.1\t0.2\trs1\n"
            "1\t200\tG\tGGGGCC\t-0.3\t0.15\t0.4\trs2\n"
            "2\t300\tC\tT\t0.2\t0\t0.6\t.\n"
        )
    return path


def _association_tsv(tmp_path: Path, *, lower: bool = False) -> Path:
    other, effect = _first(lower)
    path = tmp_path / "associations.tsv"
    path.write_text(
        "chromosome\tposition\teffect_allele\tother_allele\tbeta\tse\teaf\trsid\n"
        f"1\t100\t{effect}\t{other}\t0.5\t0.1\t0.2\trs1\n"
        "1\t200\tG\tGGGGCC\t-0.3\t0.15\t0.4\trs2\n"
        "2\t300\tC\tT\t0.2\t0\t0.6\t.\n",
        encoding="utf-8",
    )
    return path


def _finngen(tmp_path: Path, *, lower: bool = False) -> Path:
    ref, alt = _first(lower)
    path = tmp_path / "finngen.tsv.gz"
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write(
            "#chrom\tpos\tref\talt\trsids\tbeta\tsebeta\taf_alt\n"
            f"1\t100\t{ref}\t{alt}\trs1\t0.5\t0.1\t0.2\n"
            "1\t200\tGGGGCC\tG\trs2\t-0.3\t0.15\t0.4\n"
            "2\t300\tT\tC\t\tinf\t0.2\t0.6\n"
        )
    return path


def _indexed(vcf: Path) -> Path:
    """The VCF bgzipped and indexed, which a restricted bcftools query needs."""
    out = vcf.with_suffix(".vcf.gz")
    run(
        ["bcftools", "view", str(vcf), "-Oz", "-o", str(out), "--write-index=tbi"], check=True
    )
    return out


#: The fixture's rows as a reader that declares its `ref` the reference stores
#: them: the deletion `GGGGCC>G` spelt reference first (ADR 0079).
_DECLARED_ALIDS = ["1:100:A:G", "1:200:GGGGCC:G", "2:300:C:T"]
#: And as one that declares none, which drops the deletion: it names no ALID.
_UNDECLARED_ALIDS = ["1:100:A:G", "2:300:C:T"]


def _alids(declared: ReferenceAllele) -> list[str]:
    return _UNDECLARED_ALIDS if declared is ReferenceAllele.UNDECLARED else _DECLARED_ALIDS


def _readers(
    tmp_path: Path, *, lower: bool = False
) -> dict[str, tuple[SourceReader, ReferenceAllele]]:
    return {
        "gwas-vcf": (GwasVcfReader(_indexed(_vcf(tmp_path, lower=lower))), ReferenceAllele.REF),
        "gwas-ssf": (GwasSsfReader(_ssf(tmp_path, lower=lower)), ReferenceAllele.UNDECLARED),
        "finngen-r13": (
            FinnGenR13Reader(_finngen(tmp_path, lower=lower)),
            ReferenceAllele.REF,
        ),
        "association-tsv": (
            AssociationTsvReader(_association_tsv(tmp_path, lower=lower), StoredEffectScale.SD),
            ReferenceAllele.UNDECLARED,
        ),
        "column-map": (
            ColumnMapReader(_finngen(tmp_path, lower=lower), load_column_map("finngen-r13")),
            ReferenceAllele.REF,
        ),
    }


@pytest.mark.parametrize(
    "name", ["gwas-vcf", "gwas-ssf", "finngen-r13", "association-tsv", "column-map"]
)
def test_a_readers_batches_are_its_associations_as_columns(tmp_path: Path, name: str) -> None:
    reader, reference = _readers(tmp_path)[name]
    associations = list(reader.stream_associations(StreamTally()))
    batches = list(reader.stream_batches(StreamTally()))
    z = np.concatenate([b.z for b in batches])
    se = np.concatenate([b.se for b in batches])
    eaf = np.concatenate([b.eaf for b in batches])
    assert len(z) == len(associations) == len(_alids(reference))
    np.testing.assert_array_equal(z, [a.z for a in associations])
    np.testing.assert_array_equal(se, [a.se for a in associations])
    np.testing.assert_array_equal(eaf, [np.nan if a.eaf is None else a.eaf for a in associations])
    alids = np.concatenate([b.alid for b in batches]).tolist()
    assert alids == _alids(reference)


@pytest.mark.parametrize(
    "name", ["gwas-vcf", "gwas-ssf", "finngen-r13", "association-tsv", "column-map", "besd"]
)
def test_a_batch_spells_each_rows_alleles_as_its_source_does(tmp_path: Path, name: str) -> None:
    """The rsid harvest compares the names a site's spellings carry (#379), so a
    batch keeps each row's ref and alt as its associations do (ADR 0075), while
    its ALID is spelt as identity is assigned."""
    if name == "besd":
        reader = open_source(_besd_spec(_besd(tmp_path, lower=True), "P1"))
    else:
        reader = _readers(tmp_path, lower=True)[name][0]
    associations = list(reader.stream_associations(StreamTally()))
    (batch,) = reader.stream_batches(StreamTally())
    assert sorted((associations[0].ref, associations[0].alt)) == ["a", "g"]
    assert batch.alid.tolist()[0] == "1:100:A:G"
    assert batch.ref.tolist() == [a.ref for a in associations]
    assert batch.alt.tolist() == [a.alt for a in associations]
    kept = batch.take(np.array([False, True] + [False] * (len(batch) - 2)))
    assert (kept.ref.tolist(), kept.alt.tolist()) == ([associations[1].ref], [associations[1].alt])


@pytest.mark.parametrize(
    "name", ["gwas-vcf", "gwas-ssf", "finngen-r13", "association-tsv", "column-map"]
)
def test_a_declared_deletion_spells_its_reference_first_in_a_batch(
    tmp_path: Path, name: str
) -> None:
    reader, declared = _readers(tmp_path)[name]
    (batch,) = reader.stream_batches(StreamTally())
    # A declared deletion GGGGCC>G spells its reference allele first in its
    # ALID, 1:200:GGGGCC:G (ADR 0079); an undeclared one names none.
    assert batch.alid.tolist() == _alids(declared)


def test_finngens_metric_blocks_read_an_infinite_beta_as_absent(tmp_path: Path) -> None:
    reader = FinnGenR13Reader(_finngen(tmp_path))
    (chunk,) = reader.stream_metric_chunks()
    assert chunk.alid.tolist() == _DECLARED_ALIDS
    assert np.isnan(chunk.beta[2])
    # The association keeps it, for Admission to drop as `z_nonfinite` (D4).
    assert np.isinf(list(reader.stream_associations())[2].z)


def test_site_extraction_keeps_one_chromosome_on_every_reader(tmp_path: Path) -> None:
    for reader, _reference in _readers(tmp_path).values():
        sites = reader.extract_at_sites({"1:100:A:G", "2:300:C:T"}, region="1")
        assert set(sites) == {"1:100:A:G"}
        assert sites["1:100:A:G"].af == pytest.approx(0.8)


def test_a_lift_is_refused_by_a_capability_that_cannot_lift(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot lift"):
        resolve_reader(
            GWAS_SSF_CAPABILITY, _ssf(tmp_path), StoredEffectScale.SD, liftover=object()
        )
    reader = resolve_reader(
        GWAS_VCF_CAPABILITY, _vcf(tmp_path), StoredEffectScale.SD, liftover=object()
    )
    assert isinstance(reader, GwasVcfReader) and reader.liftover is not None


def test_open_source_lifts_by_the_alignment_and_only_when_it_requires(tmp_path: Path) -> None:
    spec = SourceSpec("a", str(_vcf(tmp_path)), GWAS_VCF_CAPABILITY, "hg38", StoredEffectScale.SD)
    same = assembly_plan(
        capability=GWAS_VCF_CAPABILITY, source_build="hg38", reference_build="hg38",
        chain_file=None,
    )
    unlifted = open_source(spec, alignment=same)
    assert isinstance(unlifted, GwasVcfReader) and unlifted.liftover is None
    with pytest.raises(ValueError, match="no liftover seam"):
        assembly_plan(
            capability=FINNGEN_R13_CAPABILITY, source_build="hg19", reference_build="hg38",
            chain_file=tmp_path / "chain",
        )
    assert AssemblyPlan("hg19", "hg38", tmp_path / "x.chain").requires_liftover


def _besd(tmp_path: Path, *, lower: bool = False) -> Path:
    a2, a1 = _first(lower)
    prefix = tmp_path / "eqtl"
    write_esi(
        Path(f"{prefix}.esi"),
        [
            {"chr": "1", "snp_id": "rs1", "bp": 100, "a1": a1, "a2": a2, "freq": "0.2"},
            {"chr": "NA", "snp_id": "rs2", "bp": 200, "a1": "C", "a2": "T", "freq": "0.4"},
            {"chr": "2", "snp_id": "2:300", "bp": 300, "a1": "C", "a2": "T", "freq": "1"},
        ],
    )
    write_epi(
        Path(f"{prefix}.epi"),
        [{"chr": "1", "probe_id": "P1", "bp": 1000}, {"chr": "2", "probe_id": "P2", "bp": 2000}],
    )
    write_besd_sparse_3f(
        Path(f"{prefix}.besd"),
        2,
        [[(0, 0.5, 0.1), (1, 0.2, 0.1), (2, 0.3, 0.1)], [(2, -0.4, 0.2)]],
    )
    return prefix


def _besd_spec(prefix: Path, probe_id: str | None) -> SourceSpec:
    return SourceSpec(
        probe_id or "x", str(prefix), BESD_CAPABILITY, "hg38", StoredEffectScale.SD,
        probe_id=probe_id,
    )


def test_a_besd_source_names_its_probe_and_no_other_source_does(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must name its .epi probe"):
        _besd_spec(_besd(tmp_path), None)
    with pytest.raises(ValueError, match="names no probe"):
        SourceSpec("a", "x.vcf", GWAS_VCF_CAPABILITY, "hg38", StoredEffectScale.SD, probe_id="P1")
    reader = open_source(_besd_spec(_besd(tmp_path), "P1"))
    assert isinstance(reader, BesdProbeReader)


def test_a_besd_probes_batch_names_its_malformed_rows_and_absent_frequencies(
    tmp_path: Path,
) -> None:
    reader = open_source(_besd_spec(_besd(tmp_path), "P1"))
    tally = StreamTally()
    (batch,) = reader.stream_batches(tally)
    # The .esi's a1 is the effect allele: G against A is A2, so z is negated.
    assert batch.alid.tolist() == ["1:100:A:G", "2:300:C:T"]
    np.testing.assert_allclose(batch.z, [-5.0, 3.0])
    np.testing.assert_allclose(batch.eaf[:1], [0.8])
    assert np.isnan(batch.eaf[1])  # a frequency of exactly 1 is absent (D7)
    assert batch.source_row.tolist() == [0, 2]
    assert batch.keys_at(np.array([1])) == ["esi:2"]
    assert tally.faults.rows == [("esi:1", "chromosome_missing")]
    assert tally.frequencies.made_absent == 1
    assert [a.z for a in reader.stream_associations()] == pytest.approx([-5.0, 3.0])


def test_every_probe_shares_one_read_of_the_esi(tmp_path: Path) -> None:
    prefix = _besd(tmp_path)
    first = open_source(_besd_spec(prefix, "P1"))
    second = open_source(_besd_spec(prefix, "P2"))
    assert isinstance(first, BesdProbeReader) and isinstance(second, BesdProbeReader)
    assert first.source is second.source is load_besd_source(prefix)
    (batch,) = second.stream_batches()
    np.testing.assert_allclose(batch.z, [-2.0])


class _ReverseStrandLift:
    """A chain that maps every position 1,000 bp on, onto the reverse strand."""

    def convert_coordinate(self, chrom: str, pos0: int) -> list[tuple[str, int, str, float]]:
        return [(chrom, pos0 + 1000, "-", 1.0)]


def test_a_lifted_besd_reader_keeps_each_projections_z_on_its_own_alleles(
    tmp_path: Path,
) -> None:
    """The associations and batches stay in the `.esi`'s assembly; the metric
    blocks and sites are on the reference build. A reverse-strand lift makes
    the other allele A1, so the two orientations' z differ in sign."""
    prefix = tmp_path / "lifted"
    write_esi(Path(f"{prefix}.esi"), [
        {"chr": "1", "snp_id": "rs1", "bp": 100, "a1": "G", "a2": "A", "freq": "0.2"},
    ])
    write_epi(Path(f"{prefix}.epi"), [{"chr": "1", "probe_id": "P1", "bp": 1000}])
    write_besd_sparse_3f(Path(f"{prefix}.besd"), 1, [[(0, 0.5, 0.1)]])
    reader = resolve_reader(
        BESD_CAPABILITY, prefix, StoredEffectScale.SD, liftover=_ReverseStrandLift(),
        probe_id="P1",
    )
    # In the .esi's assembly the effect allele G is A2 of A/G: z is negated.
    (association,) = reader.stream_associations()
    assert (association.position, association.ref, association.alt) == (100, "A", "G")
    assert association.z == pytest.approx(-5.0)
    assert association.eaf == pytest.approx(0.8)
    (batch,) = reader.stream_batches()
    assert batch.alid.tolist() == ["1:100:A:G"]
    np.testing.assert_allclose(batch.z, [-5.0])
    # Lifted, G/A reads C/T on the forward strand, and the effect allele C is A1.
    (chunk,) = reader.stream_metric_chunks()
    assert chunk.alid.tolist() == ["1:1100:C:T"]
    assert chunk.flipped.tolist() == [False]
    np.testing.assert_allclose(chunk.a1_frequency(), [0.2])
    assert reader.extract_at_sites({"1:1100:C:T"})["1:1100:C:T"].af == pytest.approx(0.2)


_FIXTURES = Path(__file__).parent / "fixtures"


def _registered_readers(tmp_path: Path) -> dict[str, SourceReader]:
    """Every registered capability, and every Column Map, opened through the registry."""
    scale = StoredEffectScale.SD
    return {
        "gwas-vcf": resolve_reader(GWAS_VCF_CAPABILITY, _indexed(_vcf(tmp_path)), scale),
        "gwas-ssf": resolve_reader(GWAS_SSF_CAPABILITY, _ssf(tmp_path), scale),
        "finngen-r13": resolve_reader(FINNGEN_R13_CAPABILITY, _finngen(tmp_path), scale),
        "association-tsv": resolve_reader(
            ASSOCIATION_TSV_CAPABILITY, _association_tsv(tmp_path), scale
        ),
        "besd": resolve_reader(BESD_CAPABILITY, _besd(tmp_path), scale, probe_id="P1"),
        "column-map gwas-ssf": resolve_reader(
            COLUMN_MAP_CAPABILITY, _ssf(tmp_path), scale, column_map=load_column_map("gwas-ssf")
        ),
        "column-map finngen-r13": resolve_reader(
            COLUMN_MAP_CAPABILITY, _finngen(tmp_path), scale,
            column_map=load_column_map("finngen-r13"),
        ),
        "column-map mvp-gia": resolve_reader(
            COLUMN_MAP_CAPABILITY, _FIXTURES / "dbgap_mvp" / "raw" / "continuous.tsv", scale,
            column_map=load_column_map(_FIXTURES / "dbgap_mvp" / "mvp-gia.toml"),
        ),
    }


@pytest.mark.parametrize(
    "name",
    [
        "gwas-vcf", "gwas-ssf", "finngen-r13", "association-tsv", "besd", "column-map gwas-ssf",
        "column-map finngen-r13", "column-map mvp-gia",
    ],
)
def test_every_registered_readers_batches_hold_its_associations(
    tmp_path: Path, name: str
) -> None:
    reader = _registered_readers(tmp_path)[name]
    associations = list(reader.stream_associations(StreamTally()))
    batches = list(reader.stream_batches(StreamTally()))
    assert associations and sum(len(batch) for batch in batches) == len(associations)


@pytest.mark.parametrize("short", ["ref", "alt", "alid"])
def test_a_batch_refuses_alleles_that_do_not_line_up_with_its_rows(short: str) -> None:
    """The rsid harvest pairs each row's ALID with its ref and alt (#379). A
    column one row short would hand every later row its neighbour's spelling."""
    rows = [
        SourceRow(
            chromosome="1", position=position, ref="A", alt="G", alid=f"1:{position}:A:G",
            flipped=False, af_alt=None, beta=0.4, se=0.2,
        )
        for position in (100, 200)
    ]
    (batch,) = stream_batches(rows)
    assert len(batch) == len(batch.ref) == len(batch.alt) == 2
    with pytest.raises(ValueError, match="ref and alt must each hold one allele per row"):
        replace(batch, **{short: getattr(batch, short)[:-1]})


def test_a_besd_probe_refuses_a_column_map_and_a_score_declaration(tmp_path: Path) -> None:
    """A BESD reader reads neither, so either one is refused, not ignored."""
    prefix = _besd(tmp_path)
    with pytest.raises(ValueError, match="a Column Map reads only"):
        resolve_reader(
            BESD_CAPABILITY, prefix, StoredEffectScale.SD, probe_id="P1",
            column_map=load_column_map("gwas-ssf"),
        )
    declaration = ImputationScoreDeclaration(
        "info", ImputationScoreKind.IMPUTATION_INFO, "provider documentation"
    )
    with pytest.raises(ValueError, match="imputation score declaration unsupported"):
        resolve_reader(
            BESD_CAPABILITY, prefix, StoredEffectScale.SD, probe_id="P1",
            imputation_score_declaration=declaration,
        )
