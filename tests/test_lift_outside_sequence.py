"""A lift that lands where GRCh38 has no sequence drops the row and counts it (#410).

Admission judges a GRCh38 row's own position; a lifted row reaches it at its source
coordinate, so the judgement of where its lift landed is the lift's (`split_lifted`),
made once for every path that lifts. A row lifted into an assembly gap (a run of
two or more `N`) is `in_assembly_gap`, and one lifted past its contig's last base
is `past_contig_end`, as a GRCh38 row there is.

The chain is built here, so where each site lands is known by construction:
hg19 chr1 lands in chr1's leading gap (1-10,000); hg19 chr2:1-2 in the 2 bp gap at
chr1:121,757,967-121,757,968, and chr2:3 on the base after it, which has sequence;
hg19 chr3 past chr1's last base, 248,956,422. Every test reads the committed table.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest
from admission_helpers import record_analyses
from liftover_fixtures import vcf_manifest, vcf_row
from residual_fixtures import write_gwas_vcf_with_eaf

from opengwasdb.admission import admission_record_for
from opengwasdb.layouts.dense.build_vcf import build_dense_from_vcf_manifest
from opengwasdb.variants.assembly_gaps import grch38
from opengwasdb.variants.axis import VariantAxis
from opengwasdb.variants.routing.lift import place_union

pytestmark = pytest.mark.assembly_gaps

#: GRCh38 chr1's length, and the size the chain gives its target.
CHR1 = 248_956_422
_TARGET_SIZE = 249_000_000

#: Source sites as ``(hg19 contig, position, REF, ALT)``, and where the chain puts them.
LEADING_GAP = ("1", 500, "G", "A")  # -> 1:4500, in 1-10,000
TWO_BP_GAP = ("2", 1, "G", "A")  # -> 1:121757967, in 121,757,967-121,757,968
PAST_END = ("3", 600, "G", "A")  # -> 1:248956600, past 248,956,422
CONTROL = ("2", 3, "G", "A")  # -> 1:121757969, which has sequence
CONTROL_ALID = "1:121757969:A:G"

#: Each dropped site's key as the Admission record logs it, with its reason.
DROPPED = {
    "1:500:G:A": "in_assembly_gap",
    "2:1:G:A": "in_assembly_gap",
    "3:600:G:A": "past_contig_end",
}


def _chain(path: Path) -> Path:
    """Three forward blocks, all onto GRCh38 chr1, at the offsets the sites name."""
    blocks = [("chr1", 4_000), ("chr2", 121_757_966), ("chr3", 248_956_000)]
    path.write_text(
        "".join(
            f"chain 1000 {source} 1000 + 0 1000 chr1 {_TARGET_SIZE} + {start} {start + 1000} "
            f"{i}\n1000\n\n"
            for i, (source, start) in enumerate(blocks, start=1)
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture()
def chain(tmp_path: Path) -> Path:
    return _chain(tmp_path / "outside.chain")


def _rows() -> list[str]:
    return [vcf_row(*site) for site in (LEADING_GAP, TWO_BP_GAP, CONTROL, PAST_END)]


def _alids(store: Path) -> set[str]:
    return {v.alid for v in VariantAxis(store).all()}


def _logged(store: Path, analysis: int, stage: str | None = None) -> dict[str, str]:
    name = f"{analysis}.tsv.gz" if stage is None else f"{analysis}.{stage}.tsv.gz"
    with gzip.open(admission_record_for(store) / "dropped" / name, "rt") as fh:
        return dict(line.split("\t") for line in fh.read().splitlines()[1:])


def test_the_chain_lands_each_site_where_its_name_says(chain):
    """The fixture is meaningful: the chain itself places three sites where the
    committed table has no sequence and the control where it has, so a drop below is
    the gap rule's and not the chain's."""
    from opengwasdb.build.liftover import lifted_rows

    rows, _attempts, failures = lifted_rows(
        [LEADING_GAP, TWO_BP_GAP, PAST_END, CONTROL], chain_file=chain
    )
    assert failures == 0
    landed = {site: (row.chromosome, row.position) for site, row in rows.items()}
    assert landed == {
        LEADING_GAP: ("1", 4_500), TWO_BP_GAP: ("1", 121_757_967),
        PAST_END: ("1", 248_956_600), CONTROL: ("1", 121_757_969),
    }
    table = grch38()
    assert table.lengths["1"] == CHR1
    assert table.reasons_at(["1"] * 4, [4_500, 121_757_967, 248_956_600, 121_757_969]) == [
        "in_assembly_gap", "in_assembly_gap", "past_contig_end", None,
    ]


def test_place_union_drops_a_lift_outside_the_sequence_with_its_reason(chain):
    placed = place_union(
        {"hg19": {LEADING_GAP, TWO_BP_GAP, PAST_END, CONTROL}},
        chain_file=chain, failure_threshold=0.0,
    )
    assert placed.lookup == {CONTROL: CONTROL_ALID}
    assert placed.away == DROPPED


def test_dense_two_pass_drops_and_counts_a_lift_outside_the_sequence(tmp_path, chain):
    vcf = write_gwas_vcf_with_eaf(tmp_path / "a.vcf", _rows())
    store = tmp_path / "dense.opengwasdb"
    build_dense_from_vcf_manifest(
        vcf_manifest(tmp_path, {"a": vcf}), store, store_id="s", release_id="r",
        chain_file=chain, liftover_failure_threshold=0.0,
    )
    assert _alids(store) == {CONTROL_ALID}
    (analysis,) = record_analyses(store)
    assert analysis["dropped"] == {"in_assembly_gap": 2, "past_contig_end": 1}
    assert (analysis["examined"], analysis["admitted"]) == (4, 1)
    assert _logged(store, 0) == DROPPED


def test_dense_single_pass_counts_an_unmatched_lift_outside_the_sequence(tmp_path, chain):
    """An artifact holding only the control leaves the other rows unmatched; the build
    lifts them to say why, and the chain, not the axis, lost them."""
    from opengwasdb.variants.reference import write_variant_reference

    reference = tmp_path / "ref.variant-ref.tsv.gz"
    write_variant_reference(reference, [CONTROL_ALID], {CONTROL: CONTROL_ALID}, {}, reoriented=())
    vcf = write_gwas_vcf_with_eaf(tmp_path / "a.vcf", _rows())
    store = tmp_path / "dense.opengwasdb"
    build_dense_from_vcf_manifest(
        vcf_manifest(tmp_path, {"a": vcf}), store, store_id="s", release_id="r",
        chain_file=chain, variant_reference=reference,
    )
    assert _alids(store) == {CONTROL_ALID}
    (analysis,) = record_analyses(store)
    assert analysis["dropped"] == {"in_assembly_gap": 2, "past_contig_end": 1}
    assert _logged(store, 0) == DROPPED


def _hybrid(tmp_path: Path, chain: Path, **options) -> Path:
    from opengwasdb.layouts.hybrid.build import build_hybrid_from_vcf_manifest

    vcf = write_gwas_vcf_with_eaf(tmp_path / "h.vcf", _rows())
    store = tmp_path / "hybrid.opengwasdb"
    build_hybrid_from_vcf_manifest(
        vcf_manifest(tmp_path, {"a": vcf}), store, store_id="h", release_id="v1",
        chain_file=chain, liftover_failure_threshold=0.0, **options,
    )
    return store


def test_hybrid_two_pass_drops_and_counts_a_lift_outside_the_sequence(tmp_path, chain):
    panel = tmp_path / "panel.txt"
    panel.write_text(f"{CONTROL_ALID}\n", encoding="utf-8")
    store = _hybrid(tmp_path, chain, reference_panel=panel)
    assert _alids(store) == {CONTROL_ALID}
    (analysis,) = record_analyses(store)
    assert analysis["dropped"] == {"in_assembly_gap": 2, "past_contig_end": 1}
    assert _logged(store, 0) == DROPPED


def test_the_hybrid_fold_drops_and_counts_an_off_reference_lift_outside_the_sequence(
    tmp_path, chain
):
    """The artifact routes the control alone, so the other rows are off-reference and the
    fold lifts them (`resolve_keys`)."""
    from opengwasdb.variants.reference import write_variant_reference

    reference = tmp_path / "axis.variant-ref.tsv.gz"
    write_variant_reference(reference, [CONTROL_ALID], {CONTROL: CONTROL_ALID}, {}, reoriented=())
    store = _hybrid(tmp_path, chain, variant_reference=reference)
    assert _alids(store) == {CONTROL_ALID}
    (analysis,) = record_analyses(store)
    assert analysis["dropped"] == {"in_assembly_gap": 2, "past_contig_end": 1}
    assert _logged(store, 0, "fold") == DROPPED


def test_extraction_leaves_out_a_lift_outside_the_sequence(tmp_path, chain, caplog):
    import logging

    from extraction_fixtures import extract_from_manifest

    from opengwasdb.variants.reference import read_variant_reference

    manifest = vcf_manifest(tmp_path, {"a": write_gwas_vcf_with_eaf(tmp_path / "e.vcf", _rows())})
    artifact = tmp_path / "ref.variant-ref.tsv.gz"
    caplog.set_level(logging.WARNING)
    extract_from_manifest(manifest, artifact, chain_file=chain, liftover_failure_threshold=0.0)
    reference = read_variant_reference(artifact)
    assert list(reference.alids) == [CONTROL_ALID]
    assert set(reference.source_lookup) == {CONTROL}
    assert "3 source key(s) lifted where no Store row can be" in caplog.text
    assert "(in_assembly_gap: 2, past_contig_end: 1)" in caplog.text


def test_ssf_drops_and_counts_a_lift_outside_the_sequence(tmp_path, chain):
    from test_lift_off_primary import _ssf_build, _ssf_row

    rows = [_ssf_row(*site, f"rs{i}") for i, site in enumerate(
        (LEADING_GAP, TWO_BP_GAP, CONTROL, PAST_END)
    )]
    store = _ssf_build(
        tmp_path, [("a", "GRCh37", rows)], chain_file=chain, liftover_failure_threshold=0.0
    )
    assert _alids(store) == {CONTROL_ALID}
    (analysis,) = record_analyses(store)
    assert analysis["dropped"] == {"in_assembly_gap": 2, "past_contig_end": 1}
    # A GWAS-SSF row is named by its source ALID.
    assert _logged(store, 0) == {
        "1:500:A:G": "in_assembly_gap", "2:1:A:G": "in_assembly_gap",
        "3:600:A:G": "past_contig_end",
    }
    lift = json.loads((store / "manifest.json").read_text(encoding="utf-8"))["provenance"]["lift"]
    assert (lift["n_lifted"], lift["n_unmapped"], lift["n_off_primary"]) == (4, 0, 0)
    assert lift["n_outside_sequence"] == 3
    # Every site the chain mapped is accounted for: the one stored, and the three left out.
    assert lift["n_lifted"] == 1 + lift["n_off_primary"] + lift["n_outside_sequence"]


def test_besd_drops_and_counts_a_lift_outside_the_sequence(tmp_path, chain):
    from besd_fixtures import write_besd_sparse_3f, write_epi, write_esi

    from opengwasdb.layouts.ragged.build_besd import build_ragged_from_besd

    fixture = tmp_path / "besd"
    fixture.mkdir()
    sites = (LEADING_GAP, CONTROL)
    write_esi(fixture / "t.esi", [
        {"chr": chrom, "snp_id": f"rs{i}", "bp": pos, "a1": "A", "a2": "G"}
        for i, (chrom, pos, _ref, _alt) in enumerate(sites)
    ])
    write_epi(fixture / "t.epi", [{"chr": "1", "probe_id": "ENSG1", "bp": 500, "gene": "G1"}])
    write_besd_sparse_3f(fixture / "t.besd", 1, [[(0, 0.1, 0.02), (1, -0.2, 0.03)]])
    store = tmp_path / "besd.opengwasdb"
    build_ragged_from_besd(
        fixture / "t", store, store_id="t", release_id="v1",
        association_coverage="cis_and_signals", source_build="hg19", chain_file=chain,
    )
    assert _alids(store) == {CONTROL_ALID}
    (analysis,) = record_analyses(store)
    assert analysis["dropped"] == {"in_assembly_gap": 1}
    assert _logged(store, 0) == {"esi:0": "in_assembly_gap"}
    lift = json.loads((store / "manifest.json").read_text(encoding="utf-8"))["provenance"]["lift"]
    assert (lift["n_lifted"], lift["n_off_primary"], lift["n_outside_sequence"]) == (2, 0, 1)
    assert lift["n_lifted"] == 1 + lift["n_off_primary"] + lift["n_outside_sequence"]
