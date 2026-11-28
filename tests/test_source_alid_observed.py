"""A row no source observed records no source coordinate: its `source_alid` is `.` (#261).

A plain reference (an ALID list, a Store Variant Table) names hg38 ALIDs and no source
keys, so a single-pass build routes a GRCh38 row by identity. It used to take the
reference itself as the source of every ALID it listed, observed or not, and wrote each
unobserved row's own ALID as its `source_alid`, as if a source had supplied it. The
oracle is the two-pass `--reference-panel` build of the same manifest and list, which
writes `.` for a panel row no Analysis observed.

The sources observe 1:1000:A:G (spelt `g`/`a`, the other way round and in lower case:
its Routing Key reaches the ALID all the same) and 1:3000:C:T. The list also holds
1:2000:A:C and 1:4000:G:T, which no source observes.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import pytest
from liftover_fixtures import vcf_manifest, vcf_row
from residual_fixtures import write_gwas_vcf_with_eaf
from store_assertions import assert_hybrid_stores_match

from opengwasdb.layouts.dense.build_vcf import build_dense_from_vcf_manifest
from opengwasdb.layouts.hybrid.build import build_hybrid_from_vcf_manifest
from opengwasdb.variants.routing import (
    SuppliedCoordinates,
    VariantRouting,
    require_observed_provenance,
    route,
)
from opengwasdb.variants.routing.keys import note_origin, source_alid

OBSERVED = ("1:1000:A:G", "1:3000:C:T")
UNOBSERVED = ("1:2000:A:C", "1:4000:G:T")
PANEL = tuple(sorted(OBSERVED + UNOBSERVED, key=lambda alid: int(alid.split(":")[1])))


def _vcfs(tmp_path: Path) -> dict[str, Path]:
    return {
        "a": write_gwas_vcf_with_eaf(
            tmp_path / "a.vcf", [vcf_row("1", 1_000, "g", "a"), vcf_row("1", 3_000, "C", "T")]
        ),
        "b": write_gwas_vcf_with_eaf(tmp_path / "b.vcf", [vcf_row("1", 3_000, "T", "C", 0.3)]),
    }


def _plain_list(tmp_path: Path) -> Path:
    path = tmp_path / "panel_alids.txt"
    path.write_text("".join(f"{alid}\n" for alid in PANEL), encoding="utf-8")
    return path


def _variant_table(tmp_path: Path) -> Path:
    """A Store Variant Table naming an rsid for every ALID, observed or not."""
    path = tmp_path / "variants.tsv.gz"
    header = (
        "#chromosome\tposition\tvariant_index\teffect_allele\tother_allele\talid\trsid"
        "\tsource_alid\n"
    )
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write(header)
        for i, alid in enumerate(PANEL):
            chrom, pos, a1, a2 = alid.split(":")
            fh.write(f"{chrom}\t{pos}\t{i}\t{a1}\t{a2}\t{alid}\trs{pos}\t.\n")
    return path


def _table(store: Path) -> list[list[str]]:
    with gzip.open(store / "variants.tsv.gz", "rt", encoding="utf-8") as fh:
        return [line.rstrip("\n").split("\t") for line in fh if not line.startswith("#")]


def _source_alids(store: Path) -> dict[str, str]:
    return {row[5]: row[7] for row in _table(store)}


def _hybrid(tmp_path: Path, name: str, **options) -> Path:
    store = tmp_path / f"{name}.opengwasdb"
    build_hybrid_from_vcf_manifest(
        vcf_manifest(tmp_path, _vcfs(tmp_path), assembly="hg38"), store,
        store_id="s", release_id="r", **options,
    )
    return store


def test_the_two_pass_oracle_writes_dot_for_a_panel_row_no_source_observed(tmp_path):
    """The fixture is meaningful: the two-pass build holds every panel row, and records a
    coordinate exactly for the observed ones."""
    two_pass = _hybrid(tmp_path, "two", reference_panel=_plain_list(tmp_path))
    assert _source_alids(two_pass) == {
        "1:1000:A:G": "1:1000:A:G", "1:2000:A:C": ".", "1:3000:C:T": "1:3000:C:T",
        "1:4000:G:T": ".",
    }


def test_a_plain_list_single_pass_hybrid_writes_the_two_pass_variant_table(tmp_path):
    """The ticket's acceptance: byte for byte the same table, `source_alid` included."""
    panel = _plain_list(tmp_path)
    two_pass = _hybrid(tmp_path, "two", reference_panel=panel)
    single_pass = _hybrid(tmp_path, "single", variant_reference=panel)
    with gzip.open(two_pass / "variants.tsv.gz", "rb") as fh:
        expected = fh.read()
    with gzip.open(single_pass / "variants.tsv.gz", "rb") as fh:
        assert fh.read() == expected
    assert_hybrid_stores_match(two_pass, single_pass)


def test_a_store_variant_table_reference_records_only_observed_coordinates(tmp_path):
    """A reference that names its rsids keeps its own names, which win over the
    harvest's (#258); its `source_alid` column is still the two-pass build's."""
    two_pass = _hybrid(tmp_path, "two", reference_panel=_plain_list(tmp_path))
    single_pass = _hybrid(tmp_path, "single", variant_reference=_variant_table(tmp_path))
    assert _source_alids(single_pass) == _source_alids(two_pass)
    assert {row[5]: row[6] for row in _table(single_pass)}["1:2000:A:C"] == "rs2000"


@pytest.mark.parametrize("reference", [_plain_list, _variant_table], ids=["list", "table"])
def test_a_plain_reference_single_pass_dense_writes_dot_for_an_unobserved_row(
    tmp_path, reference
):
    store = tmp_path / "dense.opengwasdb"
    build_dense_from_vcf_manifest(
        vcf_manifest(tmp_path, _vcfs(tmp_path), assembly="hg38"), store,
        store_id="s", release_id="r", variant_reference=reference(tmp_path),
    )
    assert _source_alids(store) == {
        "1:1000:A:G": "1:1000:A:G", "1:2000:A:C": ".", "1:3000:C:T": "1:3000:C:T",
        "1:4000:G:T": ".",
    }


def test_an_artifacts_recorded_source_coordinates_stay(tmp_path):
    """An extraction artifact's source keys are coordinates its sources supplied: each
    ALID keeps the one it recorded, whether or not this build's sources observe it."""
    from opengwasdb.variants.reference import write_variant_reference

    artifact = tmp_path / "ref.variant-ref.tsv.gz"
    write_variant_reference(
        artifact, list(PANEL),
        {("1", 1_000, "G", "A"): "1:1000:A:G", ("1", 2_000, "A", "C"): "1:2000:A:C",
         ("1", 3_000, "C", "T"): "1:3000:C:T", ("1", 4_000, "G", "T"): "1:4000:G:T"},
        {}, reoriented=(),
    )
    store = tmp_path / "dense.opengwasdb"
    build_dense_from_vcf_manifest(
        vcf_manifest(tmp_path, _vcfs(tmp_path), assembly="hg38"), store,
        store_id="s", release_id="r", variant_reference=artifact,
    )
    assert _source_alids(store) == {alid: alid for alid in PANEL}


_BUILDERS = {"dense": build_dense_from_vcf_manifest, "hybrid": build_hybrid_from_vcf_manifest}


def _plain_build(tmp_path: Path, builder: str, reference=_plain_list) -> None:
    _BUILDERS[builder](
        vcf_manifest(tmp_path, _vcfs(tmp_path), assembly="hg38"),
        tmp_path / "s.opengwasdb", store_id="s", release_id="r",
        variant_reference=reference(tmp_path),
    )


def _identity_provenance(self: VariantRouting) -> dict[str, str | None]:
    """`source_alids` as it was before #261: every site of the routing's map, so a plain
    reference's identity of every listed ALID."""
    by_alid: dict[str, str | None] = {}
    for site, alid in self._sites().items():
        note_origin(by_alid, alid, source_alid(site))
    return by_alid


#: What the guard says of the first unobserved row, and how many there are.
_FABRICATED = r"2 ALID\(s\) would record a source_alid that no source supplied \(first: 1:2000:A:C"


@pytest.mark.parametrize("builder", ["dense", "hybrid"])
def test_a_build_refuses_provenance_read_from_the_routing_map(tmp_path, monkeypatch, builder):
    """Put the pre-#261 `source_alids` back and the build refuses, naming the first
    fabricated ALID."""
    monkeypatch.setattr(VariantRouting, "source_alids", _identity_provenance)
    with pytest.raises(ValueError, match=_FABRICATED):
        _plain_build(tmp_path, builder)


@pytest.mark.parametrize("reference", [_plain_list, _variant_table], ids=["list", "table"])
@pytest.mark.parametrize("builder", ["dense", "hybrid"])
def test_a_build_refuses_when_the_observation_record_itself_regresses(
    tmp_path, monkeypatch, builder, reference
):
    """The rule is checked against the coordinates the sources supplied, recorded from
    the sites Pass 1 read, never from the routing's record of what was observed: make
    that record the reference's identity, as the bug did, and the build still refuses."""
    monkeypatch.setattr(
        route, "_observed_by", lambda reference, sites: dict(reference.source_lookup)
    )
    with pytest.raises(ValueError, match=_FABRICATED):
        _plain_build(tmp_path, builder, reference)


@pytest.mark.parametrize("builder", ["dense", "hybrid"])
def test_a_plain_reference_with_no_observation_record_is_refused(tmp_path, monkeypatch, builder):
    """A plain reference whose routing records nothing observed would fall back to its
    identity map: refused before anything is routed."""
    monkeypatch.setattr(route, "_observed_by", lambda reference, sites: None)
    with pytest.raises(ValueError, match="must say which ones a source observed"):
        _plain_build(tmp_path, builder)


def test_the_rule_counts_supplied_coordinates_and_passes_dot_and_ambiguity():
    supplied = SuppliedCoordinates.of_sites([("1", 1_000, "g", "a"), ("1", 3_000, "C", "T")])
    origin = {"1:1000:A:G": "1:1000:A:G", "1:3000:C:T": None, "1:2000:A:C": None}
    assert require_observed_provenance(origin, supplied, where="t") == 1
    with pytest.raises(ValueError, match=r"t: 1 ALID\(s\).*1:2000:A:C <- 1:2000:A:C"):
        require_observed_provenance(
            {**origin, "1:2000:A:C": "1:2000:A:C"}, supplied, where="t"
        )
