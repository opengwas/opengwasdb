"""A plain-ALID Variant Reference lifts an hg19 source's rows; it never routes them by
identity (#260).

A plain list names hg38 ALIDs and nothing else, so a single-pass build cannot read an
hg19 row's place from it: the row is lifted by the lift the two-pass build uses, and the
list then decides which ALIDs exist. The oracle is the two-pass build of the same
manifest (`--reference-panel` for Hybrid, the union for Dense), which these tests build
beside the single-pass one and compare cell for cell; the chain's arithmetic, worked in
`liftover_fixtures.write_reverse_collision_chain`, fixes the literal expectations.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from admission_helpers import lift_collision_ledger, record_analyses
from lift_collision_fixtures import LEDGER, named_row
from liftover_fixtures import (
    vcf_manifest,
    vcf_row,
    write_collision_chain,
    write_reverse_collision_chain,
)
from residual_fixtures import write_gwas_vcf_with_eaf
from store_assertions import assert_hybrid_stores_match

from opengwasdb.build.liftover import LiftoverFailureError
from opengwasdb.layouts.dense.build_vcf import build_dense_from_vcf_manifest
from opengwasdb.layouts.hybrid.build import build_hybrid_from_vcf_manifest
from opengwasdb.query import query_store
from opengwasdb.variants.reference import write_variant_reference

# The reverse-collision chain: hg19 chr1 [1, 1000] lifts forward to p + 5000, and
# [1001, 2000] lifts onto the reverse strand at 6502 - p, alleles complemented. Each
# row's z is ES/SE = 10 * ES for its ALT allele; stored, it is signed for the ALID's
# first allele, so a row whose (projected) ALT is that allele keeps its sign.
#: hg19 rows of one source, each with the hg38 ALID the chain gives it (None: dropped).
HG19_ROWS = (
    # 501 and 1001 both reach 1:5501:A:G (1001 T>C complemented): a collision.
    (named_row(501, "A", "G", "rs501", es=0.11), None),
    (named_row(1_001, "T", "C", "rs1001", es=0.12), None),
    # 601 lifts forward to 5601; the list also holds 1:601:C:T, the identity trap.
    (named_row(601, "C", "T", "rs601", es=0.13), "1:5601:C:T"),
    # 1501 lifts reverse to 5001, re-oriented.
    (named_row(1_501, "C", "T", "rs1501", es=0.14), "1:5001:A:G"),
    # Off the list: 701 forward, and 1201 reverse (G>T complemented to C>A).
    (named_row(701, "G", "A", "rs701", es=0.15), "1:5701:A:G"),
    (named_row(1_201, "G", "T", "rs1201", es=0.16), "1:5301:A:C"),
    # Outside every chain block: the lift fails (1 of 7 hg19 sites, 14%).
    (named_row(50_001, "A", "G", "rs_unlifted", es=0.17), None),
)
#: The plain list: hg38 ALIDs only. 1:501:A:G and 1:601:C:T are hg19 coordinates of
#: rows that must never land on them; 1:5501:A:G is the collided ALID.
PLAIN_LIST = (
    "1:501:A:G", "1:601:C:T", "1:1501:C:T", "1:5001:A:G", "1:5501:A:G", "1:5601:C:T",
    "1:9001:A:C",
)
#: Below the fixture's 1-in-7 failure rate, so the failure is counted, not refused.
THRESHOLD = 0.2
#: The two builders that accept `--variant-reference`.
BUILDERS = {"dense": build_dense_from_vcf_manifest, "hybrid": build_hybrid_from_vcf_manifest}


def _hg19_vcf(tmp_path: Path, name: str = "b") -> Path:
    return write_gwas_vcf_with_eaf(tmp_path / f"{name}.vcf", [row for row, _ in HG19_ROWS])


def _grch38_vcf(tmp_path: Path) -> Path:
    """A GRCh38 source: 5601 is the lifted 601's variant seen from GRCh38; 1:1501:C:T
    is the hg19 1501's tuple declared under both assemblies (two loci, placed nowhere);
    9001 is on the list and 9101 off it."""
    rows = [
        vcf_row("1", 5_601, "C", "T", 0.21),
        named_row(1_501, "C", "T", "rs_native1501", es=0.22),
        named_row(9_001, "A", "C", "rs9001", es=0.23),
        named_row(9_101, "G", "T", "rs9101", es=0.24),
    ]
    return write_gwas_vcf_with_eaf(tmp_path / "a.vcf", rows)


def _plain_list(tmp_path: Path, alids=PLAIN_LIST) -> Path:
    path = tmp_path / "panel_alids.txt"
    path.write_text("".join(f"{alid}\n" for alid in alids), encoding="utf-8")
    return path


def _analysis_ids(store: Path) -> list[str]:
    lines = (store / "analyses.tsv").read_text(encoding="utf-8").splitlines()
    column = lines[0].split("\t").index("analysis_id")
    return [line.split("\t")[column] for line in lines[1:]]


def _cells(store: Path) -> tuple[dict[str, str], dict[tuple[str, str], tuple[float, ...]]]:
    """Each stored ALID's rsid, and every stored association by (ALID, Analysis)."""
    with query_store(store) as query:
        table = query.variants_table()
        rsids = {row["alid"]: row["rsid"] or "" for row in table.values()}
        alids = [row["alid"] for row in table.values()]
        cells: dict[tuple[str, str], tuple[float, ...]] = {}
        for analysis in _analysis_ids(store):
            found = query.lookup(alids, [analysis])
            for index, z, se, eaf in zip(
                found.variant_index, found.z, found.se, found.eaf, strict=True
            ):
                if not math.isnan(float(z)):
                    cells[(table[int(index)]["alid"], analysis)] = (
                        round(float(z), 4), round(float(se), 4), round(float(eaf), 4)
                    )
    return rsids, cells


def _assert_fixture_meaningful(two_pass_cells: dict, listed: set[str]) -> None:
    """The two-pass oracle stores what the chain arithmetic says, so it can judge."""
    stored = {alid for alid, _ in two_pass_cells}
    assert "1:5601:C:T" in stored & listed, "a lifted row on the list"
    assert {"1:5701:A:G", "1:5301:A:C"} <= stored - listed, "lifted rows off the list"
    assert not {"1:501:A:G", "1:601:C:T", "1:5501:A:G"} & stored, "trap ALIDs stay empty"


def _dense_build(tmp_path: Path, vcfs: dict[str, Path], assembly, *, plain: bool) -> Path:
    """A Dense build of ``vcfs``: single-pass against the plain list, or two-pass."""
    chain = write_reverse_collision_chain(tmp_path / "reverse.chain")
    store = tmp_path / ("single.opengwasdb" if plain else "two.opengwasdb")
    build_dense_from_vcf_manifest(
        vcf_manifest(tmp_path, vcfs, assembly=assembly), store, store_id="s",
        release_id="r", chain_file=chain, liftover_failure_threshold=THRESHOLD,
        variant_reference=_plain_list(tmp_path) if plain else None,
    )
    return store


def _dense_pair(tmp_path: Path, vcfs: dict[str, Path], assembly) -> tuple[Path, Path]:
    """The two-pass and the single-pass Dense build of ``vcfs``."""
    return (
        _dense_build(tmp_path, vcfs, assembly, plain=False),
        _dense_build(tmp_path, vcfs, assembly, plain=True),
    )


def _assert_dense_matches_on_the_list(two_pass: Path, single_pass: Path) -> None:
    """Dense keeps the list's ALIDs, so the single-pass store is the two-pass store
    restricted to them: every association and rsid on the same ALID."""
    listed = set(PLAIN_LIST)
    two_rsids, two_cells = _cells(two_pass)
    one_rsids, one_cells = _cells(single_pass)
    _assert_fixture_meaningful(two_cells, listed)
    assert set(one_rsids) == listed
    assert one_cells == {key: cell for key, cell in two_cells.items() if key[0] in listed}
    assert one_rsids == {alid: two_rsids.get(alid, "") for alid in listed}


@pytest.mark.parametrize(
    ("vcfs", "assembly"),
    [
        pytest.param(("b",), "hg19", id="hg19"),
        pytest.param(("a", "b"), {"a": "hg38", "b": "hg19"}, id="mixed"),
    ],
)
def test_dense_plain_list_stores_what_the_two_pass_build_stores(tmp_path, vcfs, assembly):
    sources = {"a": _grch38_vcf, "b": _hg19_vcf}
    two_pass, single_pass = _dense_pair(
        tmp_path, {name: sources[name](tmp_path) for name in vcfs}, assembly
    )
    _assert_dense_matches_on_the_list(two_pass, single_pass)
    rsids, cells = _cells(single_pass)
    assert ("1:601:C:T", "b") not in cells, "hg19 601 is not stored at its own coordinate"
    assert cells[("1:5601:C:T", "b")][0] == pytest.approx(-1.3, abs=0.01)
    if "a" not in vcfs:  # in the mixed manifest both assemblies declare 1:1501:C:T
        assert cells[("1:5001:A:G", "b")][0] == pytest.approx(1.4, abs=0.01), "re-oriented"
        assert rsids["1:5001:A:G"] == "rs1501"


def test_mixed_manifest_drops_a_tuple_both_assemblies_declare(tmp_path):
    """hg19 1501 and GRCh38 1:1501:C:T are two loci sharing a string: neither is
    stored, as the two-pass build drops both (#255)."""
    single_pass = _dense_build(
        tmp_path, {"a": _grch38_vcf(tmp_path), "b": _hg19_vcf(tmp_path)},
        {"a": "hg38", "b": "hg19"}, plain=True,
    )
    rsids, cells = _cells(single_pass)
    assert ("1:1501:C:T", "a") not in cells
    assert ("1:5001:A:G", "b") not in cells
    assert rsids["1:1501:C:T"] == "" and rsids["1:5001:A:G"] == ""
    assert cells[("1:9001:A:C", "a")][0] == pytest.approx(-2.3, abs=0.01), "fixture: a stored"


@pytest.mark.parametrize(
    ("vcfs", "assembly"),
    [
        pytest.param(("b",), "hg19", id="hg19"),
        pytest.param(("a", "b"), {"a": "hg38", "b": "hg19"}, id="mixed"),
    ],
)
def test_hybrid_plain_list_is_the_two_pass_panel_build(tmp_path, vcfs, assembly):
    """The Overflow holds the off-list lifts, the reverse-strand one re-oriented,
    exactly as the two-pass `--reference-panel` build places them."""
    sources = {"a": _grch38_vcf, "b": _hg19_vcf}
    chain = write_reverse_collision_chain(tmp_path / "reverse.chain")
    manifest = vcf_manifest(
        tmp_path, {name: sources[name](tmp_path) for name in vcfs}, assembly=assembly
    )
    panel = _plain_list(tmp_path)
    two_pass, single_pass = tmp_path / "two.opengwasdb", tmp_path / "single.opengwasdb"
    common = {"store_id": "s", "release_id": "r", "chain_file": chain,
              "liftover_failure_threshold": THRESHOLD}
    build_hybrid_from_vcf_manifest(manifest, two_pass, reference_panel=panel, **common)
    build_hybrid_from_vcf_manifest(manifest, single_pass, variant_reference=panel, **common)

    two_rsids, two_cells = _cells(two_pass)
    _assert_fixture_meaningful(two_cells, set(PLAIN_LIST))
    assert two_cells[("1:5301:A:C", "b")][0] == pytest.approx(1.6, abs=0.01)
    assert _cells(single_pass) == (two_rsids, two_cells)
    assert_hybrid_stores_match(two_pass, single_pass)


@pytest.mark.parametrize("builder", ["dense", "hybrid"])
def test_a_failure_rate_over_the_threshold_refuses_the_plain_list_build(tmp_path, builder):
    """1 of the 7 hg19 sites fails the lift: over a 10% threshold the build refuses."""
    chain = write_reverse_collision_chain(tmp_path / "reverse.chain")
    manifest = vcf_manifest(tmp_path, {"b": _hg19_vcf(tmp_path)}, assembly="hg19")
    with pytest.raises(LiftoverFailureError):
        BUILDERS[builder](
            manifest, tmp_path / "s.opengwasdb", store_id="s", release_id="r",
            chain_file=chain, liftover_failure_threshold=0.1,
            variant_reference=_plain_list(tmp_path),
        )


def test_a_failure_rate_under_the_threshold_is_counted(tmp_path):
    """Under the threshold the unlifted row is dropped and counted, as two-pass does."""
    single_pass = _dense_build(tmp_path, {"b": _hg19_vcf(tmp_path)}, "hg19", plain=True)
    (analysis,) = record_analyses(single_pass)
    # 501/1001 collide; 701 and 1201 lift off the list; 50001 does not lift.
    assert analysis["dropped"] == {"lift_collision": 2, "off_axis": 3}
    assert (analysis["examined"], analysis["admitted"]) == (7, 2)


# #325's review reproductions: hg19 501 and 1001 both lift to 1:5501:A:G, and the plain
# list holds 1:501:A:G, the hg19 501's own coordinate.
COLLISION_LIST = ("1:501:A:G", "1:6001:C:T")


def _collision_rows() -> list[str]:
    return [
        named_row(501, "A", "G", "rs_first"),
        named_row(1_001, "A", "G", "rs_second"),
        named_row(1_501, "C", "T", "rs_clean"),
    ]


def _collision_build(tmp_path: Path, builder: str, rows=None, **options) -> Path:
    """Analyses a and b, each holding ``rows`` (the collision rows by default)."""
    chain = write_collision_chain(tmp_path / "forward.chain")
    rows = _collision_rows() if rows is None else rows
    vcfs = {name: write_gwas_vcf_with_eaf(tmp_path / f"{name}.vcf", rows) for name in ("a", "b")}
    store = tmp_path / "store.opengwasdb"
    BUILDERS[builder](
        vcf_manifest(tmp_path, vcfs, assembly="hg19"), store, store_id="s", release_id="r",
        chain_file=chain, **options,
    )
    return store


def _frozen_artifact(tmp_path: Path, rsid_by_alid: dict[str, str]) -> Path:
    """An artifact extracted before #325: hg19 501 and 1001 both route to 1:5501:A:G."""
    artifact = tmp_path / "frozen.variant-ref.tsv.gz"
    write_variant_reference(
        artifact,
        ["1:5501:A:G", "1:6001:C:T"],
        {
            ("1", 501, "A", "G"): "1:5501:A:G",
            ("1", 1_001, "A", "G"): "1:5501:A:G",
            ("1", 1_501, "C", "T"): "1:6001:C:T",
        },
        rsid_by_alid,
        reoriented=(),
    )
    return artifact


def _assert_only_the_clean_row_is_stored(store: Path, lift_collision: int) -> list:
    """Each Analysis stores 6001 alone and counts ``lift_collision`` rows; returns them."""
    _rsids, cells = _cells(store)
    assert set(cells) == {("1:6001:C:T", "a"), ("1:6001:C:T", "b")}
    analyses = record_analyses(store)
    assert len(analyses) == 2, "fixture: the record names both Analyses"
    for analysis in analyses:
        assert analysis["dropped"]["lift_collision"] == lift_collision
    return analyses


@pytest.mark.parametrize("builder", ["dense", "hybrid"])
def test_a_collided_source_is_not_admitted_at_its_own_coordinate(tmp_path, builder):
    """hg19 501 is not stored at 1:501:A:G, and 1001 is not re-admitted at 1:5501:A:G
    by a fold that met it alone: both are dropped, counted and in the ledger."""
    store = _collision_build(
        tmp_path, builder, variant_reference=_plain_list(tmp_path, COLLISION_LIST)
    )
    for analysis in _assert_only_the_clean_row_is_stored(store, 2):
        assert analysis["dropped"] == {"lift_collision": 2}
        assert (analysis["examined"], analysis["admitted"]) == (3, 1)
    assert sorted(lift_collision_ledger(store)[1:]) == sorted(LEDGER)


@pytest.mark.parametrize("builder", ["dense", "hybrid"])
def test_a_frozen_artifact_routing_two_loci_to_one_alid_drops_both(tmp_path, builder):
    """An artifact extracted before #325 routes hg19 501 and 1001 to 1:5501:A:G: both
    sources are dropped and recorded, and the ALID holds no association."""
    artifact = _frozen_artifact(tmp_path, {"1:5501:A:G": "rs_first", "1:6001:C:T": "rs_clean"})
    store = _collision_build(tmp_path, builder, variant_reference=artifact)
    for analysis in _assert_only_the_clean_row_is_stored(store, 2):
        assert analysis["dropped"] == {"lift_collision": 2}
    ledger = sorted(line.rsplit("\t", 1)[0] for line in lift_collision_ledger(store)[1:])
    assert ledger == sorted(line.rsplit("\t", 1)[0] for line in LEDGER)


@pytest.mark.parametrize("builder", ["dense", "hybrid"])
def test_a_frozen_collision_is_dropped_when_its_partner_row_is_unusable(tmp_path, builder):
    """SE is 0 at 1001, so admission drops that row before it is routed: only 501 is
    unmatched. The routing dropped both sources, so 501 is a `lift_collision` and the
    Hybrid fold never meets it alone to place it at 1:5501:A:G. The ledger names what
    the build dropped as a collision: 501."""
    rows = _collision_rows()
    rows[1] = rows[1].replace("0.2:0.1:0.2", "0.2:0:0.2")
    assert ":0:" in rows[1], "fixture: 1001 has SE 0"
    artifact = _frozen_artifact(tmp_path, {"1:6001:C:T": "rs_clean"})
    store = _collision_build(tmp_path, builder, rows, variant_reference=artifact)
    for analysis in _assert_only_the_clean_row_is_stored(store, 1):
        assert "se_unusable" in analysis["dropped"]
    assert [line.split("\t")[:2] for line in lift_collision_ledger(store)[1:]] == [
        ["1:5501:A:G", "1:501:A:G"]
    ]
