"""Builders drop the sources of a lift collision, count each row, and record each (#325).

A collision's statistics must never reach the ALID: whichever source's row was read
last would own it, and its rsid would name the other. The routing-level cases are in
`test_variant_routing.py`; these run Dense, Hybrid and Ragged SSF builds through the
real chain (`liftover_fixtures`), whose arithmetic is worked in its docstring.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from liftover_fixtures import vcf_manifest, vcf_row, write_collision_chain
from residual_fixtures import write_gwas_vcf_with_eaf
from test_lift_off_primary import _admission, _alids, _logged, _ssf_build, _ssf_row

from opengwasdb.admission import admission_record_for
from opengwasdb.layouts.dense.build_vcf import build_dense_from_vcf_manifest

COLLIDED = "1:5501:A:G"
CLEAN = "1:6001:C:T"


def _named_row(pos: int, ref: str, alt: str, rsid: str) -> str:
    """A GWAS-VCF row whose ID column is ``rsid``."""
    return vcf_row("1", pos, ref, alt).replace("\t.\t", f"\t{rsid}\t", 1)


#: hg19 501 and 1001 both lift to 1:5501:A:G; 1501 lifts to 1:6001:C:T alone.
ROWS = [
    _named_row(501, "A", "G", "rs_first"),
    _named_row(1_001, "A", "G", "rs_second"),
    _named_row(1_501, "C", "T", "rs_clean"),
]
LEDGER = [
    f"{COLLIDED}\t1:501:A:G\trs_first",
    f"{COLLIDED}\t1:1001:A:G\trs_second",
]


def _vcfs(tmp_path: Path):
    return {
        name: write_gwas_vcf_with_eaf(tmp_path / f"{name}.vcf", ROWS) for name in ("a", "b")
    }


def _collision_file(store: Path) -> list[str]:
    path = admission_record_for(store) / "lift_collisions.tsv"
    return path.read_text(encoding="utf-8").splitlines()


def _check_record(store: Path, *, stage: str | None = None):
    analyses = _admission(store)
    assert analyses, "fixture: the record names Analyses to judge"
    for index, analysis in enumerate(analyses):
        assert analysis["dropped"] == {"lift_collision": 2}
        assert (analysis["examined"], analysis["admitted"]) == (3, 1)
        assert _logged(store, index, stage) == [
            "1:501:A:G\tlift_collision", "1:1001:A:G\tlift_collision"
        ]
    assert _collision_file(store)[0] == "target_alid\tsource_key\trsid"
    assert sorted(_collision_file(store)[1:]) == sorted(LEDGER)


def test_dense_drops_both_sources_and_counts_each_row(tmp_path):
    chain = write_collision_chain(tmp_path / "forward.chain")
    store = tmp_path / "store.opengwasdb"
    build_dense_from_vcf_manifest(
        vcf_manifest(tmp_path, _vcfs(tmp_path), assembly="hg19"), store,
        store_id="s", release_id="r", chain_file=chain,
    )
    assert _alids(store) == {CLEAN}
    _check_record(store)


def test_hybrid_two_pass_drops_both_sources_and_counts_each_row(tmp_path):
    from opengwasdb.layouts.hybrid.build import build_hybrid_from_vcf_manifest

    chain = write_collision_chain(tmp_path / "forward.chain")
    panel = tmp_path / "panel.txt"
    panel.write_text(f"{CLEAN}\n", encoding="utf-8")
    store = tmp_path / "hybrid.opengwasdb"
    build_hybrid_from_vcf_manifest(
        vcf_manifest(tmp_path, _vcfs(tmp_path), assembly="hg19"), store,
        store_id="h", release_id="v1", chain_file=chain, reference_panel=panel,
    )
    assert _alids(store) == {CLEAN}
    _check_record(store)


def test_hybrid_single_pass_counts_collided_off_reference_rows_in_the_fold(tmp_path):
    store = _hybrid_single_pass(tmp_path)
    assert _alids(store) == {CLEAN}
    _check_record(store, stage="fold")


def test_ragged_ssf_drops_both_sources_and_counts_them_as_collisions(tmp_path):
    chain = write_collision_chain(tmp_path / "forward.chain")
    rows = [
        _ssf_row("1", 501, "A", "G", "rs_first"),
        _ssf_row("1", 1_001, "A", "G", "rs_second"),
        _ssf_row("1", 1_501, "C", "T", "rs_clean"),
    ]
    store = _ssf_build(tmp_path, [("a", "GRCh37", rows)], chain_file=chain)

    assert _alids(store) == {CLEAN}
    (analysis,) = _admission(store)
    assert analysis["dropped"] == {"lift_collision": 2}
    assert (analysis["examined"], analysis["admitted"]) == (3, 1)
    assert sorted(_logged(store, 0)) == [
        "1:1001:A:G\tlift_collision", "1:501:A:G\tlift_collision"
    ]
    lift = json.loads((store / "manifest.json").read_text(encoding="utf-8"))["provenance"]["lift"]
    assert (lift["n_lifted"], lift["n_unmapped"]) == (3, 0), "collided sites are not unmapped"
    assert sorted(_collision_file(store)[1:]) == sorted(LEDGER)


@pytest.mark.parametrize("native_rsid", ["rs_aaa_native", "."])
def test_dense_keeps_no_native_row_a_collided_source_spells(tmp_path, native_rsid):
    """hg19 501 and 1001 collide at 1:5501, and a GRCh38 Analysis (read first) declares
    1:501:A:G, named or not: the dropped hg19 501 spells that key, so no statistics of it
    may land on that native row, and the key both assemblies declare is dropped from both.
    The native row is `lift_ambiguous`, and the ledger names the hg19 source by the rsid
    its own source gave it, never the native's and never blank."""
    chain = write_collision_chain(tmp_path / "forward.chain")
    vcfs = {
        "a": write_gwas_vcf_with_eaf(
            tmp_path / "a.vcf", [_named_row(501, "A", "G", native_rsid)]
        ),
        "b": write_gwas_vcf_with_eaf(tmp_path / "b.vcf", ROWS),
    }
    store = tmp_path / "store.opengwasdb"
    build_dense_from_vcf_manifest(
        vcf_manifest(tmp_path, vcfs, assembly={"a": "hg38", "b": "hg19"}), store,
        store_id="s", release_id="r", chain_file=chain,
    )
    assert _alids(store) == {CLEAN}, "1:501:A:G holds no row of either assembly"
    native, lifted = _admission(store)
    assert native["dropped"] == {"lift_ambiguous": 1}
    assert lifted["dropped"] == {"lift_collision": 2}
    assert sorted(_collision_file(store)[1:]) == sorted(LEDGER)


def test_dense_names_the_row_dropped_not_its_case_folded_sibling(tmp_path):
    """Analysis a drops `1:501:A:G` and 1001 as collisions. Analysis b's `1:501:a:g`
    (SE 0, named) is dropped as `se_unusable` and never meets the lift: it shares a's
    Routing Key, but it is not a row the build dropped as a collision."""
    chain = write_collision_chain(tmp_path / "forward.chain")
    lower = _named_row(501, "a", "g", "rs_lower_zero").replace("0.2:0.1:0.2", "0.2:0:0.2")
    assert "\ta\tg\t" in lower and ":0:" in lower, "fixture: lower-case allele, SE 0"
    vcfs = {
        "a": write_gwas_vcf_with_eaf(tmp_path / "a.vcf", ROWS),
        "b": write_gwas_vcf_with_eaf(tmp_path / "b.vcf", [lower, ROWS[2]]),
    }
    store = tmp_path / "store.opengwasdb"
    build_dense_from_vcf_manifest(
        vcf_manifest(tmp_path, vcfs, assembly="hg19"), store,
        store_id="s", release_id="r", chain_file=chain,
    )
    _a, b = _admission(store)
    assert "se_unusable" in b["dropped"], "fixture: b's lower-case row is dropped for its SE"
    assert "lift_collision" not in b["dropped"]
    assert sorted(_collision_file(store)[1:]) == sorted(LEDGER)


def test_dense_with_a_plain_reference_counts_and_records_a_collision(tmp_path):
    """A reference naming no rsids is harvested; the harvest's lift collides on 5501, and
    the two sources' rows, which the reference does not route, are `lift_collision`."""
    chain = write_collision_chain(tmp_path / "forward.chain")
    reference = tmp_path / "axis.alid.txt"
    reference.write_text(f"{CLEAN}\n", encoding="utf-8")
    store = tmp_path / "store.opengwasdb"
    build_dense_from_vcf_manifest(
        vcf_manifest(tmp_path, _vcfs(tmp_path), assembly="hg19"), store,
        store_id="s", release_id="r", chain_file=chain, variant_reference=reference,
    )
    assert _alids(store) == {CLEAN}
    analyses = _admission(store)
    assert analyses, "fixture: the record names Analyses to judge"
    for analysis in analyses:
        # The plain reference holds no source key, so 1501 is `off_axis` as ever.
        assert analysis["dropped"] == {"lift_collision": 2, "off_axis": 1}
    assert sorted(_collision_file(store)[1:]) == sorted(LEDGER)


def _hybrid_single_pass(tmp_path: Path, **options) -> Path:
    from opengwasdb.layouts.hybrid.build import build_hybrid_from_vcf_manifest

    chain = write_collision_chain(tmp_path / "forward.chain")
    reference = tmp_path / "axis.alid.txt"
    reference.write_text(f"{CLEAN}\n", encoding="utf-8")
    store = tmp_path / "hybrid.opengwasdb"
    build_hybrid_from_vcf_manifest(
        vcf_manifest(tmp_path, _vcfs(tmp_path), assembly="hg19"), store,
        store_id="h", release_id="v1", chain_file=chain, variant_reference=reference, **options,
    )
    return store


def test_a_fold_interrupted_after_its_axis_is_recorded_resumes_with_the_ledger(
    tmp_path, monkeypatch
):
    """The axis is resumable once recorded; the identities of what the fold dropped must
    be recorded with it, or a resumed fold counts the drops and never names them. The
    fold itself is interrupted, and the ledger is deleted, so nothing written before can
    stand in for what the resume restores."""
    from opengwasdb.layouts.hybrid import build as hybrid_build
    from opengwasdb.layouts.hybrid.build import resume_hybrid_build
    from opengwasdb.layouts.hybrid.checkpoint import checkpoint_dir_for

    def _interrupted(col: int) -> int:
        raise RuntimeError("interrupted in the fold")

    with monkeypatch.context() as patch:
        patch.setattr(hybrid_build, "_fold_column", _interrupted)
        with pytest.raises(RuntimeError, match="interrupted in the fold"):
            _hybrid_single_pass(tmp_path, checkpoint=True)
    store = tmp_path / "hybrid.opengwasdb"
    checkpoint = checkpoint_dir_for(store)
    assert (checkpoint / "axis.npz").exists(), "fixture: the axis was recorded"
    (admission_record_for(store) / "lift_collisions.tsv").unlink()

    resume_hybrid_build(checkpoint)

    assert sorted(_collision_file(store)[1:]) == sorted(LEDGER)
    _check_record(store, stage="fold")


def _hybrid_unnamed_reference(tmp_path: Path, vcfs: dict, assembly) -> Path:
    from opengwasdb.layouts.hybrid.build import build_hybrid_from_vcf_manifest

    tmp_path.mkdir(exist_ok=True)
    chain = write_collision_chain(tmp_path / "forward.chain")
    reference = tmp_path / "axis.alid.txt"
    reference.write_text(f"{CLEAN}\n", encoding="utf-8")
    store = tmp_path / "hybrid.opengwasdb"
    build_hybrid_from_vcf_manifest(
        vcf_manifest(tmp_path, vcfs, assembly=assembly), store,
        store_id="h", release_id="v1", chain_file=chain, variant_reference=reference,
    )
    return store


def test_the_ledger_names_what_the_fold_dropped_not_what_the_harvest_found(tmp_path):
    """SE is 0 at 1001 in both Analyses, so admission drops that row before anything
    spills: the fold meets only 501, which stores at 1:5501. The harvest found both
    sources colliding, but the build dropped neither on that account. The control is
    the same input with a usable SE: it records both, so the fixture does reach the
    ledger."""
    usable = _named_row(1_001, "A", "G", "rs_second")
    zero_se = usable.replace("0.2:0.1:0.2", "0.2:0:0.2")
    assert zero_se != usable, "fixture: the SE is edited"

    def vcfs(root: Path, row_1001: str) -> dict:
        root.mkdir(exist_ok=True)
        rows = [ROWS[0], row_1001, ROWS[2]]
        return {n: write_gwas_vcf_with_eaf(root / f"{n}.vcf", rows) for n in ("a", "b")}

    root = tmp_path / "usable"
    control = _hybrid_unnamed_reference(root, vcfs(root, usable), "hg19")
    assert sorted(_collision_file(control)[1:]) == sorted(LEDGER), "control: both are recorded"
    assert COLLIDED not in _alids(control)

    root = tmp_path / "zero"
    store = _hybrid_unnamed_reference(root, vcfs(root, zero_se), "hg19")
    assert COLLIDED in _alids(store), "the fold stored 501, so nothing was dropped"
    assert not (admission_record_for(store) / "lift_collisions.tsv").exists()


@pytest.mark.parametrize(
    ("native", "native_rsid", "lifted", "lifted_rsid"),
    [
        ("A:G", "rs_aaa_native", "A:G", "rs_first"),
        ("A:G", ".", "A:G", "rs_first"),
        ("A:G", ".", "a:g", "rs_first"),
        ("A:G", "rs_aaa_native", "a:g", "rs_first"),
        ("A:G", "rs_aaa_native", "a:g", "."),
        ("a:g", "rs_native", "A:G", "."),
    ],
    ids=[
        "native-named",
        "native-unnamed",
        "lower-case-hg19",
        "lower-case-hg19-native-named",
        "unnamed-lower-case-hg19-native-named",
        "unnamed-hg19-lower-case-native-named",
    ],
)
def test_the_fold_names_a_lifted_source_by_its_own_rsid_beside_a_native_one(
    tmp_path, native, native_rsid, lifted, lifted_rsid
):
    """A GRCh38 Analysis (read first) declares 1:501 with SE 0, so admission drops it;
    the fold sees only the two hg19 sources, which collide. The ledger gives hg19 501 the
    rsid its own source named it, blank if it named none, in the spelling its source
    wrote: never the native row's rsid or spelling, whatever case either row used."""
    native_row = _named_row(501, *native.split(":"), native_rsid)
    native_row = native_row.replace("0.2:0.1:0.2", "0.2:0:0.2")
    assert ":0:" in native_row, "fixture: the native row has SE 0"
    vcfs = {
        "a": write_gwas_vcf_with_eaf(tmp_path / "a.vcf", [native_row]),
        "b": write_gwas_vcf_with_eaf(
            tmp_path / "b.vcf", [_named_row(501, *lifted.split(":"), lifted_rsid), *ROWS[1:]]
        ),
    }
    store = _hybrid_unnamed_reference(tmp_path, vcfs, {"a": "hg38", "b": "hg19"})
    named = "" if lifted_rsid == "." else lifted_rsid
    assert sorted(_collision_file(store)[1:]) == sorted(
        [f"{COLLIDED}\t1:501:{lifted}\t{named}", LEDGER[1]]
    )


@pytest.mark.parametrize("native_rsid", ["rs_aaa_native", "."])
def test_the_fold_drops_collided_hg19_sources_beside_a_usable_native_row(tmp_path, native_rsid):
    """hg19 501 and 1001 collide at 1:5501, and a GRCh38 Analysis declares a usable
    1:501:A:G. The both-assemblies rule drops key 501, but the collision is found first:
    1:5501:A:G is not stored from 1001 alone, and both hg19 sources are in the ledger.
    The native row spells a dropped lifted key, so it is `lift_ambiguous`, as Dense and
    the two-pass Hybrid count it; only the two lifted rows are `lift_collision`."""
    vcfs = {
        "a": write_gwas_vcf_with_eaf(tmp_path / "a.vcf", [_named_row(501, "A", "G", native_rsid)]),
        "b": write_gwas_vcf_with_eaf(tmp_path / "b.vcf", ROWS),
    }
    store = _hybrid_unnamed_reference(tmp_path, vcfs, {"a": "hg38", "b": "hg19"})
    assert COLLIDED not in _alids(store)
    assert sorted(_collision_file(store)[1:]) == sorted(LEDGER)
    native, lifted = _admission(store)
    assert native["dropped"] == {"lift_ambiguous": 1}
    assert lifted["dropped"] == {"lift_collision": 2}
