"""A lift onto an alt or random contig drops the row and counts it (#376).

The chain is `liftover_fixtures.write_alt_target_chain`'s: hg19 chr5:1000 lands on
`11_KI270927v1_alt`, chr3:1000 on chr3:1000. Every builder that lifts a source
leaves the first out of the Store and says so in the Admission record, per
Analysis; a source already on GRCh38 keeps an alt-contig row, since there the alt
contig is a chromosome (spec §4).
"""

from __future__ import annotations

import gzip
import json
import logging
from collections.abc import Mapping
from pathlib import Path

import pytest
from liftover_fixtures import (
    ALT_LIFTED_ROW,
    ALT_TARGET_CONTIG,
    PRIMARY_LIFTED_ROW,
    write_alt_target_chain,
)
from residual_fixtures import write_gwas_vcf_with_eaf

from opengwasdb.admission import admission_record_for
from opengwasdb.layouts.dense.build_vcf import build_dense_from_vcf_manifest
from opengwasdb.variants.axis import VariantAxis
from opengwasdb.variants.normalise import is_primary_contig
from opengwasdb.variants.routing import place_sites

#: The alt-lifted row's source key as the Admission record logs it.
ALT_KEY = "{}:{}:{}:{}".format(*ALT_LIFTED_ROW[:4])


@pytest.fixture()
def chain(tmp_path: Path) -> Path:
    return write_alt_target_chain(tmp_path / "alt-target.chain")


def _vcf_row(chrom: str, pos: int, ref: str, alt: str, es: float = 0.2) -> str:
    return f"{chrom}\t{pos}\t.\t{ref}\t{alt}\t.\tPASS\t.\tES:SE:AF\t{es}:0.1:0.2\n"


def _alt_and_primary_rows() -> list[str]:
    return [_vcf_row(*ALT_LIFTED_ROW[:4]), _vcf_row(*PRIMARY_LIFTED_ROW[:4])]


def _vcf_manifest(tmp_path: Path, vcfs: dict[str, Path], assembly: str | None = None) -> Path:
    header = (
        "trait_id\tfile_path\ttrait_name\tn\tstored_effect_scale\toriginal_sd_method"
        + ("\tsource_assembly" if assembly else "")
    )
    lines = [header] + [
        f"{trait}\t{vcf}\tTrait {trait}\t1000\tsd\tdeclared_standardised"
        + (f"\t{assembly}" if assembly else "")
        for trait, vcf in vcfs.items()
    ]
    path = tmp_path / "manifest.tsv"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _admission(store: Path) -> list[Mapping[str, object]]:
    record = admission_record_for(store)
    return json.loads((record / "admission.json").read_text(encoding="utf-8"))["analyses"]


def _logged(store: Path, analysis: int, stage: str | None = None) -> list[str]:
    name = f"{analysis}.tsv.gz" if stage is None else f"{analysis}.{stage}.tsv.gz"
    with gzip.open(admission_record_for(store) / "dropped" / name, "rt") as fh:
        return fh.read().splitlines()[1:]


def _alids(store: Path) -> set[str]:
    return {v.alid for v in VariantAxis(store).all()}


def test_dense_drops_a_row_lifted_onto_an_alt_contig_and_counts_it(tmp_path, chain):
    vcfs = {
        "a": write_gwas_vcf_with_eaf(tmp_path / "a.vcf", _alt_and_primary_rows()),
        "b": write_gwas_vcf_with_eaf(tmp_path / "b.vcf", _alt_and_primary_rows()[:1]),
    }
    store = tmp_path / "store.opengwasdb"
    build_dense_from_vcf_manifest(
        _vcf_manifest(tmp_path, vcfs), store, store_id="s", release_id="r", chain_file=chain,
        liftover_failure_threshold=0.0,
    )
    assert _alids(store) == {PRIMARY_LIFTED_ROW[4]}, "the alt-contig variant is not stored"
    a, b = _admission(store)
    assert a["dropped"] == {"lift_off_primary": 1}
    assert (a["examined"], a["admitted"]) == (2, 1)
    assert b["dropped"] == {"lift_off_primary": 1}
    assert (b["examined"], b["admitted"]) == (1, 0)
    assert _logged(store, 0) == [f"{ALT_KEY}\tlift_off_primary"]
    assert _logged(store, 1) == [f"{ALT_KEY}\tlift_off_primary"]


def test_dense_keeps_a_grch38_source_row_on_an_alt_contig(tmp_path):
    """The same alt contig in a source already on GRCh38 is a chromosome."""
    alt_row = _vcf_row(ALT_TARGET_CONTIG, 1_000, "G", "A")
    vcfs = {"a": write_gwas_vcf_with_eaf(tmp_path / "a.vcf", [alt_row])}
    store = tmp_path / "store.opengwasdb"
    build_dense_from_vcf_manifest(
        _vcf_manifest(tmp_path, vcfs, assembly="hg38"), store, store_id="s", release_id="r",
    )
    assert _alids(store) == {ALT_LIFTED_ROW[4]}
    (analysis,) = _admission(store)
    assert analysis["dropped"] == {}
    assert analysis["admitted"] == 1


def test_dense_stores_a_row_lifted_onto_a_primary_contig_as_before(tmp_path, chain):
    vcfs = {"a": write_gwas_vcf_with_eaf(tmp_path / "a.vcf", [_vcf_row(*PRIMARY_LIFTED_ROW[:4])])}
    store = tmp_path / "store.opengwasdb"
    build_dense_from_vcf_manifest(
        _vcf_manifest(tmp_path, vcfs), store, store_id="s", release_id="r", chain_file=chain,
    )
    assert _alids(store) == {PRIMARY_LIFTED_ROW[4]}
    (analysis,) = _admission(store)
    assert analysis["dropped"] == {}
    assert (analysis["examined"], analysis["admitted"]) == (1, 1)


# --- GWAS-SSF -----------------------------------------------------------------

_SSF_HEADER = [
    "chromosome", "base_pair_location", "effect_allele", "other_allele", "beta",
    "standard_error", "effect_allele_frequency", "rsid",
]


def _ssf_row(chrom: str, pos: int, ref: str, alt: str, rsid: str) -> dict[str, object]:
    """An SSF row whose effect allele is the VCF-style ALT: z = 2."""
    return {
        "chromosome": chrom, "base_pair_location": pos, "effect_allele": alt,
        "other_allele": ref, "beta": 0.2, "standard_error": 0.1,
        "effect_allele_frequency": 0.2, "rsid": rsid,
    }


def _ssf_build(tmp_path: Path, analyses: list[tuple[str, str, list[dict]]], **options) -> Path:
    from opengwasdb.layouts.ragged.build_ssf import build_ragged_from_ssf

    source_dir = tmp_path / "ssf"
    source_dir.mkdir()
    lines = ["analysis_index\tanalysis_id\tfiltered_file\tsource_genome_build"]
    for index, (analysis_id, build, rows) in enumerate(analyses):
        with gzip.open(source_dir / f"{analysis_id}.tsv.gz", "wt", encoding="utf-8") as fh:
            fh.write("\t".join(_SSF_HEADER) + "\n")
            for row in rows:
                fh.write("\t".join(str(row[col]) for col in _SSF_HEADER) + "\n")
        lines.append(f"{index}\t{analysis_id}\t{analysis_id}.tsv.gz\t{build}")
    manifest = tmp_path / "ssf-manifest.tsv"
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out = tmp_path / "ssf.opengwasdb"
    build_ragged_from_ssf(
        manifest, source_dir, out, store_id="t", release_id="v1",
        association_coverage="cis_and_signals", allow_unverified_eaf=True, **options,
    )
    return out


def test_ssf_drops_a_row_lifted_onto_an_alt_contig_and_counts_it(tmp_path, chain):
    rows = [
        _ssf_row(*ALT_LIFTED_ROW[:4], "rs_alt"),
        _ssf_row(*PRIMARY_LIFTED_ROW[:4], "rs_primary"),
    ]
    store = _ssf_build(
        tmp_path, [("a", "GRCh37", rows)], chain_file=chain, liftover_failure_threshold=0.0
    )
    assert _alids(store) == {PRIMARY_LIFTED_ROW[4]}
    (analysis,) = _admission(store)
    assert analysis["dropped"] == {"lift_off_primary": 1}
    assert (analysis["examined"], analysis["admitted"]) == (2, 1)
    # A GWAS-SSF row is named by its source ALID, as `lift_unmapped`'s are.
    assert _logged(store, 0) == ["5:1000:A:G\tlift_off_primary"]
    lift = json.loads((store / "manifest.json").read_text(encoding="utf-8"))["provenance"]["lift"]
    assert (lift["n_lifted"], lift["n_unmapped"], lift["n_off_primary"]) == (1, 0, 1)


def test_ssf_keeps_a_grch38_source_row_on_an_alt_contig(tmp_path, chain):
    rows = [_ssf_row(ALT_TARGET_CONTIG, 1_000, "G", "A", "rs_alt")]
    store = _ssf_build(tmp_path, [("a", "GRCh38", rows)], chain_file=chain)
    assert _alids(store) == {ALT_LIFTED_ROW[4]}
    (analysis,) = _admission(store)
    assert analysis["dropped"] == {}


def test_ssf_stores_a_row_lifted_onto_a_primary_contig_as_before(tmp_path, chain):
    rows = [_ssf_row(*PRIMARY_LIFTED_ROW[:4], "rs_primary")]
    store = _ssf_build(tmp_path, [("a", "GRCh37", rows)], chain_file=chain)
    assert _alids(store) == {PRIMARY_LIFTED_ROW[4]}
    (analysis,) = _admission(store)
    assert (analysis["dropped"], analysis["admitted"]) == ({}, 1)


def test_ssf_refuses_a_build_whose_every_lift_landed_off_the_primary_assembly(tmp_path, chain):
    rows = [_ssf_row(*ALT_LIFTED_ROW[:4], "rs_alt")]
    with pytest.raises(ValueError, match=r"1 lift_off_primary"):
        _ssf_build(tmp_path, [("a", "GRCh37", rows)], chain_file=chain)


# --- Hybrid -------------------------------------------------------------------


def _hybrid_build(tmp_path: Path, rows: list[str], *, panel: bool = True, **options) -> Path:
    from opengwasdb.layouts.hybrid.build import build_hybrid_from_vcf_manifest

    vcf = write_gwas_vcf_with_eaf(tmp_path / "h.vcf", rows)
    manifest = _vcf_manifest(tmp_path, {"a": vcf, "b": vcf})
    store = tmp_path / "hybrid.opengwasdb"
    if panel:
        panel_file = tmp_path / "panel.txt"
        panel_file.write_text(f"{PRIMARY_LIFTED_ROW[4]}\n", encoding="utf-8")
        options["reference_panel"] = panel_file
    build_hybrid_from_vcf_manifest(
        manifest, store, store_id="h", release_id="v1", **options
    )
    return store


def test_hybrid_two_pass_drops_a_row_lifted_onto_an_alt_contig_and_counts_it(tmp_path, chain):
    store = _hybrid_build(tmp_path, _alt_and_primary_rows(), chain_file=chain)
    assert _alids(store) == {PRIMARY_LIFTED_ROW[4]}
    for analysis in _admission(store):
        assert analysis["dropped"] == {"lift_off_primary": 1}
        assert (analysis["examined"], analysis["admitted"]) == (2, 1)
    assert _logged(store, 0) == [f"{ALT_KEY}\tlift_off_primary"]


def test_hybrid_single_pass_drops_an_off_reference_row_lifted_onto_an_alt_contig(
    tmp_path, chain
):
    """The reference holds only the primary variant, so the alt row is off-reference and
    the fold lifts it."""
    reference = tmp_path / "axis.alid.txt"
    reference.write_text(f"{PRIMARY_LIFTED_ROW[4]}\n", encoding="utf-8")
    store = _hybrid_build(
        tmp_path, _alt_and_primary_rows(), chain_file=chain, panel=False,
        variant_reference=reference,
    )
    assert _alids(store) == {PRIMARY_LIFTED_ROW[4]}
    for analysis in _admission(store):
        assert analysis["dropped"] == {"lift_off_primary": 1}
        assert (analysis["examined"], analysis["admitted"]) == (2, 1)
    assert _logged(store, 0, "fold") == [f"{ALT_KEY}\tlift_off_primary"]


def test_hybrid_keeps_a_grch38_source_row_on_an_alt_contig(tmp_path):
    from opengwasdb.layouts.hybrid.build import build_hybrid_from_vcf_manifest

    vcf = write_gwas_vcf_with_eaf(
        tmp_path / "h.vcf", [_vcf_row(ALT_TARGET_CONTIG, 1_000, "G", "A")]
    )
    manifest = _vcf_manifest(tmp_path, {"a": vcf}, assembly="hg38")
    panel = tmp_path / "panel.txt"
    panel.write_text(f"{ALT_LIFTED_ROW[4]}\n", encoding="utf-8")
    store = tmp_path / "hybrid.opengwasdb"
    build_hybrid_from_vcf_manifest(
        manifest, store, store_id="h", release_id="v1", reference_panel=panel
    )
    assert _alids(store) == {ALT_LIFTED_ROW[4]}
    (analysis,) = _admission(store)
    assert analysis["dropped"] == {}


def test_hybrid_stores_a_row_lifted_onto_a_primary_contig_as_before(tmp_path, chain):
    store = _hybrid_build(tmp_path, [_vcf_row(*PRIMARY_LIFTED_ROW[:4])], chain_file=chain)
    assert _alids(store) == {PRIMARY_LIFTED_ROW[4]}
    for analysis in _admission(store):
        assert (analysis["dropped"], analysis["admitted"]) == ({}, 1)


# --- BESD ---------------------------------------------------------------------


def _besd_prefix(tmp_path: Path, chrom_a: str = "5", pos_a: int = 1_000) -> Path:
    """One probe, with an association at each of two ESI rows: row 0 is the alt-lifted
    site (or, given another contig, whatever it names), row 1 the primary one."""
    from besd_fixtures import write_besd_sparse_3f, write_epi, write_esi

    fixture = tmp_path / "besd"
    fixture.mkdir()
    snps = [
        {"chr": chrom_a, "snp_id": "rs_alt", "bp": pos_a, "a1": "A", "a2": "G"},
        {"chr": "3", "snp_id": "rs_primary", "bp": 1_000, "a1": "A", "a2": "G"},
    ]
    probes = [{"chr": "3", "probe_id": "ENSG00000000001", "bp": 500, "gene": "GENE1"}]
    write_esi(fixture / "t.esi", snps)
    write_epi(fixture / "t.epi", probes)
    write_besd_sparse_3f(fixture / "t.besd", 1, [[(0, 0.1, 0.02), (1, -0.2, 0.03)]])
    return fixture / "t"


def _besd_build(tmp_path: Path, prefix: Path, **options) -> Path:
    from opengwasdb.layouts.ragged.build_besd import build_ragged_from_besd

    out = tmp_path / "besd.opengwasdb"
    build_ragged_from_besd(
        prefix, out, store_id="t", release_id="v1", association_coverage="cis_and_signals",
        **options,
    )
    return out


def test_besd_drops_a_row_lifted_onto_an_alt_contig_and_counts_it(tmp_path, chain):
    store = _besd_build(tmp_path, _besd_prefix(tmp_path), source_build="hg19", chain_file=chain)
    assert _alids(store) == {PRIMARY_LIFTED_ROW[4]}
    (analysis,) = _admission(store)
    assert analysis["dropped"] == {"lift_off_primary": 1}
    assert (analysis["examined"], analysis["admitted"]) == (2, 1)
    assert _logged(store, 0) == ["esi:0\tlift_off_primary"]
    lift = json.loads((store / "manifest.json").read_text(encoding="utf-8"))["provenance"]["lift"]
    assert (lift["n_lifted"], lift["n_off_primary"]) == (1, 1)


def test_besd_keeps_a_grch38_source_row_on_an_alt_contig(tmp_path):
    prefix = _besd_prefix(tmp_path, chrom_a=ALT_TARGET_CONTIG)
    store = _besd_build(tmp_path, prefix, source_build="hg38")
    assert _alids(store) == {ALT_LIFTED_ROW[4], PRIMARY_LIFTED_ROW[4]}
    (analysis,) = _admission(store)
    assert analysis["dropped"] == {}


def test_besd_stores_a_row_lifted_onto_a_primary_contig_as_before(tmp_path, chain):
    prefix = _besd_prefix(tmp_path, chrom_a="3", pos_a=2_000)
    store = _besd_build(tmp_path, prefix, source_build="hg19", chain_file=chain)
    assert _alids(store) == {PRIMARY_LIFTED_ROW[4], "3:2000:A:G"}
    lift = json.loads((store / "manifest.json").read_text(encoding="utf-8"))["provenance"]["lift"]
    assert lift["n_off_primary"] == 0


# --- Variant Reference extraction and the resolver's lift ----------------------


def test_extraction_leaves_out_a_site_lifted_onto_an_alt_contig(tmp_path, chain, caplog):
    from extraction_fixtures import extract_from_manifest

    from opengwasdb.variants.reference import read_variant_reference

    vcf = write_gwas_vcf_with_eaf(tmp_path / "e.vcf", _alt_and_primary_rows())
    artifact = tmp_path / "ref.variant-ref.tsv.gz"
    with caplog.at_level(logging.WARNING):
        extract_from_manifest(
            _vcf_manifest(tmp_path, {"a": vcf}), artifact, chain_file=chain,
            liftover_failure_threshold=0.0,
        )
    reference = read_variant_reference(artifact)
    assert list(reference.alids) == [PRIMARY_LIFTED_ROW[4]]
    assert not any(ALT_TARGET_CONTIG in alid for alid in reference.source_lookup.values())
    assert "1 source key(s) lifted onto an alt, random or unplaced contig" in caplog.text


def test_extraction_keeps_a_grch38_source_site_on_an_alt_contig(tmp_path):
    from extraction_fixtures import extract_from_manifest

    from opengwasdb.variants.reference import read_variant_reference

    vcf = write_gwas_vcf_with_eaf(
        tmp_path / "e.vcf", [_vcf_row(ALT_TARGET_CONTIG, 1_000, "G", "A")]
    )
    artifact = tmp_path / "ref.variant-ref.tsv.gz"
    extract_from_manifest(_vcf_manifest(tmp_path, {"a": vcf}, assembly="hg38"), artifact)
    assert list(read_variant_reference(artifact).alids) == [ALT_LIFTED_ROW[4]]


def test_the_resolvers_lift_returns_no_variant_for_an_alt_contig_target(chain):
    from opengwasdb.readers.gwas_vcf import load_liftover, oriented_variant

    liftover = load_liftover("hg19", "hg38", str(chain))
    chrom, pos, ref, alt, _ = ALT_LIFTED_ROW
    assert oriented_variant(
        liftover, chrom, str(pos), alt, ref, other_is_reference=True
    ) is None
    chrom, pos, ref, alt, alid = PRIMARY_LIFTED_ROW
    orientation = oriented_variant(liftover, chrom, str(pos), alt, ref, other_is_reference=True)
    assert orientation is not None
    assert orientation.variant.alid == alid


# --- What a primary-assembly contig is ----------------------------------------


@pytest.mark.parametrize(
    "label",
    [*map(str, range(1, 23)), "X", "Y", "MT", "chr7", "chrX", "23", "24", "25", "26", "M", "x"],
)
def test_a_canonical_label_or_its_alias_is_a_primary_contig(label):
    assert is_primary_contig(label)


@pytest.mark.parametrize(
    "label",
    [ALT_TARGET_CONTIG, "chr11_KI270927v1_alt", "chrUn_KI270302v1", "1_KI270706v1_random",
     "HLA-A*01:01:01:01", "0", "27", "", "chrEBV"],
)
def test_an_alt_random_or_unplaced_contig_is_not(label):
    assert not is_primary_contig(label)


def test_the_alt_target_chain_lifts_onto_an_alt_and_a_primary_contig(tmp_path):
    """The fixture is meaningful: the lift itself places chr5 on the alt contig, so the
    drop below is the placement's and not the chain's failure."""
    chain = write_alt_target_chain(tmp_path / "alt.chain")
    alt, primary = ALT_LIFTED_ROW, PRIMARY_LIFTED_ROW
    placement = place_sites(
        {"hg19": [alt[:4], primary[:4]]}, chain_file=chain, failure_threshold=0.0
    )
    assert placement.alid == {primary[:4]: primary[4]}
    assert placement.off_primary == {alt[:4]}
