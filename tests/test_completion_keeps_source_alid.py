"""Reference Completion keeps each observed row's `source_alid` (#449).

`source_alid` is the pre-liftover ALID a build recorded as a row's provenance. Dense and
Ragged completion rewrote the Store Variant Table without it, so a completed release
lost which source variant any of its rows came from, and nothing failed. An observed row
keeps its source's `source_alid`; a row completion added from the panel, which no source
observed, stays `.`; a row the build left blank (a collision) stays blank.

The validator's `variant_table.source_alid` refuses a release whose manifest records a
liftover and whose table holds no `source_alid` at all.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import test_dense_completion as tdc
import test_extract_variant_reference as tev
import test_hybrid_completion as thc
from dense_sources import build_dense_fixture
from region_tables import edit_manifest

from opengwasdb.layouts.dense.complete import complete_dense_store
from opengwasdb.layouts.hybrid.complete import complete_hybrid_store
from opengwasdb.layouts.ragged.build_besd import build_ragged_from_besd
from opengwasdb.layouts.ragged.complete import complete_ragged_store
from opengwasdb.store_reader.manifest import StoreManifest
from opengwasdb.store_reader.variant_axis import VariantRecord
from opengwasdb.validation.validate import validate_store
from opengwasdb.variants.axis import VariantAxis, write_variant_axis
from opengwasdb.variants.normalise import CanonicalVariant

RULE = "variant_table.source_alid"


def _records(store: Path) -> list[VariantRecord]:
    axis = VariantAxis(store)
    try:
        return axis.all()
    finally:
        axis.close()


def _spelt_on_source(record: VariantRecord) -> str:
    """A plausible pre-liftover ALID for a row: another position, never the row's own ALID."""
    alleles = f"{record.effect_allele}:{record.other_allele}"
    return f"{record.chromosome}:{record.position - 1000}:{alleles}"


def plant_source_alids(store: Path, *, blank: int = 1) -> dict[str, str]:
    """Give every row of an observed release a distinct `source_alid` but the first
    `blank` rows (a collision leaves a row blank). Returns the planted ones by ALID."""
    records = _records(store)
    planted = {r.alid: _spelt_on_source(r) for r in records[blank:]}
    assert planted and all(s != a for a, s in planted.items()), "the fixture plants nothing"
    write_variant_axis(
        store,
        [
            CanonicalVariant(r.chromosome, r.position, r.effect_allele, r.other_allele)
            for r in records
        ],
        {r.alid: r.rsid for r in records if r.rsid},
        [planted.get(r.alid) for r in records],
    )
    return planted


def assert_source_alids_survive(
    source: Path, completed: Path, planted: dict[str, str]
) -> None:
    observed = {r.alid for r in _records(source)}
    after = {r.alid: r.source_alid for r in _records(completed)}
    added = set(after) - observed
    assert added, "completion added no panel row: the fixture does not reach the path"
    assert {a: after[a] for a in planted} == planted
    assert all(after[a] is None for a in added), "a panel row was given a source_alid"
    blanked = observed - set(planted)
    assert blanked and all(after[a] is None for a in blanked), "a blank source row was filled"


def _strand_source_alids() -> dict[str, str]:
    """The hg38 ALID of each `STRAND_ROWS` variant, and the hg19 ALID its source spelt."""
    from liftover_fixtures import STRAND_ROWS

    from opengwasdb.variants.normalise import canonical_alid

    return {row[5]: canonical_alid(row[1], row[2], row[3], row[4]) for row in STRAND_ROWS}


def _besd_source_alids(tb) -> dict[str, str]:
    """The hg38 ALID of each BESD strand-fixture variant, and the ALID its own hg19
    coordinate spells."""
    from opengwasdb.variants.normalise import orient_to_canonical

    return {
        alid: orient_to_canonical(
            snp["chr"], snp["bp"], snp["a1"], snp["a2"], other_is_reference=False
        ).variant.alid
        for alid, snp in (
            (tb._REVERSE_SNP_HG38_ALID, tb._REVERSE_SNP),
            (tb._FORWARD_SNP_HG38_ALID, tb._FORWARD_SNP),
        )
    }


def _strand_panel(root: Path) -> Path:
    """Panel blocks over the three lifted variants and one new variant on each of chr2, chr3."""
    from ld_blocks import write_ld_block

    write_ld_block(
        root / "EUR" / "2", "990000-1000000",
        [("2:995000:A:G", 0.3, 995_000), ("2:998001:A:T", 0.3, 998_001),
         ("2:999001:C:T", 0.3, 999_001)],
        seed=2,
    )
    write_ld_block(
        root / "EUR" / "3", "1-5000",
        [("3:1000:A:G", 0.3, 1_000), ("3:2000:A:G", 0.3, 2_000)], seed=3,
    )
    return root


def assert_lifted_provenance_kept(source: Path, completed: Path, alids: dict[str, str]) -> None:
    """Every observed row holds its pre-lift ALID, which is not its own ALID for a row the
    chain moved; every row completion added from the panel is `.`."""
    assert any(a != s for a, s in alids.items()), "nothing moved: the fixture proves nothing"
    observed = {r.alid for r in _records(source)}
    after = {r.alid: r.source_alid for r in _records(completed)}
    assert {a: after[a] for a in observed} == {a: alids[a] for a in observed}
    assert set(alids) <= observed
    added = set(after) - observed
    assert added, "completion added no panel row: the fixture does not reach the path"
    assert all(after[a] is None for a in added), "a panel row was given a source_alid"


def lifted_dense_completion(tmp_path: Path) -> tuple[Path, Path]:
    """A Dense release built from an hg19 VCF over a two-strand chain, and its completion
    over a panel that adds rows: ``(observed, completed)``."""
    from liftover_fixtures import strand_vcf_rows, write_two_strand_chain

    from opengwasdb.layouts.dense.build_vcf import build_dense_from_vcf_manifest

    chain = write_two_strand_chain(tmp_path / "hg19ToHg38.chain")
    vcf = tev._make_vcf(tmp_path, "strand_rows", strand_vcf_rows())
    manifest = tev._make_manifest(tmp_path, [("strand", vcf, "", "hg19")])
    observed = tmp_path / "obs.opengwasdb"
    build_dense_from_vcf_manifest(
        manifest, observed, store_id="s", release_id="r", chain_file=chain,
        liftover_failure_threshold=0.5,
    )
    completed = tmp_path / "comp.opengwasdb"
    complete_dense_store(
        observed, completed, _strand_panel(tmp_path / "panel"),
        ancestry="EUR", min_cor=0.0, release_id="comp-v1",
    )
    return observed, completed


def test_a_lifted_dense_source_keeps_its_source_alid_through_completion(tmp_path: Path) -> None:
    observed, completed = lifted_dense_completion(tmp_path)
    assert_lifted_provenance_kept(observed, completed, _strand_source_alids())
    assert validate_store(completed).failed() == frozenset()


def test_a_lifted_ragged_source_keeps_its_source_alid_through_completion(tmp_path: Path) -> None:
    import test_ragged_build_besd as tb
    from liftover_fixtures import write_two_strand_chain

    chain = write_two_strand_chain(tmp_path / "hg19ToHg38.chain")
    observed = tmp_path / "obs.opengwasdb"
    build_ragged_from_besd(
        tb._write_strand_fixture(tmp_path), observed, store_id="test", release_id="v1",
        source_build="hg19", chain_file=chain, association_coverage="cis",
    )
    completed = tmp_path / "comp.opengwasdb"
    complete_ragged_store(
        observed, completed, _strand_panel(tmp_path / "panel"),
        ancestry="EUR", cis_window_bp=500_000, min_cor=0.0, release_id="comp-v1",
    )
    alids = _besd_source_alids(tb)
    assert_lifted_provenance_kept(observed, completed, alids)


def test_dense_completion_leaves_a_blank_source_row_blank(tmp_path: Path) -> None:
    """A row the build left blank (a collision) is never filled, whatever its neighbours hold."""
    source_tsv = tmp_path / "associations.tsv"
    source_tsv.write_text(
        tdc.SOURCE_HEADER + "\n" + "\n".join(tdc.SOURCE_ROWS) + "\n", encoding="utf-8"
    )
    observed = tmp_path / "obs.opengwasdb"
    build_dense_fixture(
        [source_tsv], observed, store_id="test", release_id="obs-v1", reference_assembly="GRCh38"
    )
    planted = plant_source_alids(observed)
    completed = tmp_path / "comp.opengwasdb"
    complete_dense_store(
        observed, completed, tdc._make_ld_panel(tmp_path),
        ancestry="EUR", min_cor=0.0, release_id="comp-v1",
    )
    assert_source_alids_survive(observed, completed, planted)


def test_hybrid_completion_keeps_the_shared_tables_source_alid(tmp_path: Path) -> None:
    """A guard: the Hybrid already carries it (`complete.py`'s `source_alid_by_alid`).
    Built from an hg19 source and lifted, so the observed rows' `source_alid` is real."""
    source = thc._build_source(tmp_path)
    completed = tmp_path / "dst.opengwasdb"
    complete_hybrid_store(source, completed, thc._make_ld_panel(tmp_path), min_cor=0.0, thresh=0.9)

    before = {r.alid: r.source_alid for r in _records(source)}
    after = {r.alid: r.source_alid for r in _records(completed)}
    sourced = {a: s for a, s in before.items() if s}
    assert sourced and any(s != a for a, s in sourced.items()), "nothing was lifted"
    assert {a: after[a] for a in sourced} == sourced
    added = set(after) - set(before)
    # The shared table gains a row only where completion folds one in from the panel.
    assert all(after[a] is None for a in added)


def test_a_hybrid_variant_seen_from_two_assemblies_has_no_source_alid(tmp_path: Path) -> None:
    import test_hybrid_build as thb

    from opengwasdb.layouts.hybrid.build import build_hybrid_from_vcf_manifest

    early = thb.make_vcf(
        tmp_path, "early_hg19",
        ["1\t1000000\trsEARLY\tC\tT\t.\tPASS\t.\tES:SE\t1.5:0.3\n"],
    )
    late = thb.make_vcf(
        tmp_path, "late_hg38",
        ["1\t1064620\trsLATE\tC\tT\t.\tPASS\t.\tES:SE\t1.2:0.3\n"],
    )
    manifest = thb._manifest_with_source_assembly(
        tmp_path, [("early", early, "Early", "hg19"), ("late", late, "Late", "hg38")]
    )
    panel = tmp_path / "panel.txt"
    panel.write_text("1:1064619:A:G\n", encoding="utf-8")
    store = tmp_path / "h.opengwasdb"
    build_hybrid_from_vcf_manifest(
        manifest, store, reference_panel=panel, store_id="s", release_id="r"
    )
    # By design (#85): the routing blanks an ALID whose sites name different origins
    # (`routing/keys.note_origin`), never guessing one. Here the hg19 row lifts onto the
    # variant the hg38 row names, so its two origins differ.
    by_alid = {r.alid: r.source_alid for r in _records(store)}
    assert by_alid["1:1064620:C:T"] is None


# ── the rule ────────────────────────────────────────────────────────────────


def _blank_source_alids(store: Path) -> None:
    """Rewrite a release's table with no `source_alid` on any row."""
    records = _records(store)
    write_variant_axis(
        store,
        [
            CanonicalVariant(r.chromosome, r.position, r.effect_allele, r.other_allele)
            for r in records
        ],
        {r.alid: r.rsid for r in records if r.rsid},
    )


def _edit_provenance(store: Path, **entries: object) -> None:
    edit_manifest(store, lambda m: {**m, "provenance": {**m["provenance"], **entries}})


@pytest.fixture
def observed_dense(tmp_path: Path) -> Path:
    source_tsv = tmp_path / "associations.tsv"
    source_tsv.write_text(
        tdc.SOURCE_HEADER + "\n" + "\n".join(tdc.SOURCE_ROWS) + "\n", encoding="utf-8"
    )
    out = tmp_path / "obs.opengwasdb"
    build_dense_fixture(
        [source_tsv], out, store_id="test", release_id="obs-v1", reference_assembly="GRCh38"
    )
    return out


def record_liftover(store: Path) -> None:
    _edit_provenance(store, chain_file="pyliftover_builtin_hg19_hg38")


def test_the_rule_fails_a_lifted_release_whose_rows_all_lack_a_source_alid(
    observed_dense: Path,
) -> None:
    assert validate_store(observed_dense).failed() == frozenset()  # a valid release to begin with
    # The TSV build names each row's own ALID (#470); a release that kept none is blanked.
    assert all(r.source_alid == r.alid for r in _records(observed_dense))
    _blank_source_alids(observed_dense)
    assert all(r.source_alid is None for r in _records(observed_dense))  # meaningful fixture
    record_liftover(observed_dense)
    result = validate_store(observed_dense)
    assert result.failed() == {RULE}
    assert "records a liftover" in result.of(RULE)[0].message


def test_the_rule_passes_when_one_row_carries_a_source_alid(observed_dense: Path) -> None:
    record_liftover(observed_dense)
    plant_source_alids(observed_dense, blank=len(_records(observed_dense)) - 1)
    assert sum(1 for r in _records(observed_dense) if r.source_alid) == 1
    assert validate_store(observed_dense).failed() == frozenset()


def test_the_rule_passes_a_release_that_records_no_liftover(observed_dense: Path) -> None:
    _blank_source_alids(observed_dense)
    assert all(r.source_alid is None for r in _records(observed_dense))
    assert validate_store(observed_dense).failed() == frozenset()


def test_the_rule_passes_a_chain_recorded_where_nothing_was_lifted(observed_dense: Path) -> None:
    """OGS-00011's shape (opengwasdb-stores#499): a build records `provenance.chain_file`
    as a default label even when no variant was lifted, and its `source_alid` is the ALID."""
    record_liftover(observed_dense)
    records = _records(observed_dense)
    write_variant_axis(
        observed_dense,
        [
            CanonicalVariant(r.chromosome, r.position, r.effect_allele, r.other_allele)
            for r in records
        ],
        {r.alid: r.rsid for r in records if r.rsid},
        [r.alid for r in records],
    )
    assert validate_store(observed_dense).failed() == frozenset()


def test_the_rule_passes_a_completed_lifted_release_that_kept_them(
    observed_dense: Path, tmp_path: Path
) -> None:
    plant_source_alids(observed_dense)
    record_liftover(observed_dense)
    completed = tmp_path / "comp.opengwasdb"
    complete_dense_store(
        observed_dense, completed, tdc._make_ld_panel(tmp_path),
        ancestry="EUR", min_cor=0.0, release_id="comp-v1",
    )
    assert validate_store(completed).failed() == frozenset()


@pytest.fixture
def lifted_ragged(tmp_path: Path) -> Path:
    import test_ragged_build_besd as tb
    from liftover_fixtures import write_two_strand_chain

    out = tmp_path / "ragged.opengwasdb"
    build_ragged_from_besd(
        tb._write_strand_fixture(tmp_path), out, store_id="test", release_id="v1",
        source_build="hg19", chain_file=write_two_strand_chain(tmp_path / "x.chain"),
        association_coverage="cis_and_signals",
    )
    return out


def test_the_rule_fails_a_ragged_release_that_lifted_and_holds_no_source_alid(
    lifted_ragged: Path,
) -> None:
    import json

    lift = json.loads((lifted_ragged / "manifest.json").read_text())["provenance"]["lift"]
    assert lift["n_lifted"] > 0  # meaningful fixture
    assert validate_store(lifted_ragged).failed() == frozenset()
    _blank_source_alids(lifted_ragged)
    assert validate_store(lifted_ragged).failed() == {RULE}


def test_the_rule_does_not_judge_a_ragged_release_that_lifted_nothing(lifted_ragged: Path) -> None:
    import json

    lift = json.loads((lifted_ragged / "manifest.json").read_text())["provenance"]["lift"]
    _edit_provenance(lifted_ragged, lift={**lift, "n_lifted": 0})
    _blank_source_alids(lifted_ragged)
    result = validate_store(lifted_ragged)
    assert result.failed() == frozenset()
    assert not [f for f in result.of(RULE) if f.severity.name == "ERROR"]


def test_the_rule_refuses_a_lift_block_that_does_not_say_how_many_it_lifted(
    lifted_ragged: Path,
) -> None:
    import json

    lift = json.loads((lifted_ragged / "manifest.json").read_text())["provenance"]["lift"]
    _edit_provenance(lifted_ragged, lift={k: v for k, v in lift.items() if k != "n_lifted"})
    result = validate_store(lifted_ragged)
    assert result.failed() == {RULE}
    assert "n_lifted" in result.of(RULE)[0].message


def test_a_hybrid_is_judged_on_its_shared_table_alone(tmp_path: Path) -> None:
    """The Dense Component's copy of the table is gone at 0.3.0 (#472): there is no copy to
    blank, and its rows' `source_alid` are the shared table's, so a blank shared table is
    the finding. A 0.2.3 Hybrid still carries the copy, and a blank copy is no finding
    there either: the rule judges the shared table alone."""
    from format_0_2_3 import as_0_2_3, with_fixed_width_alid_index

    source = thc._build_source(tmp_path)
    assert StoreManifest.load(source).provenance.get("chain_file"), "the fixture records a liftover"
    assert validate_store(source).failed() == frozenset()
    assert not (source / "dense" / "variants.tsv.gz").exists(), "#472: no Dense copy"

    legacy = as_0_2_3(Path(shutil.copytree(source, tmp_path / "legacy")))
    assert (legacy / "dense" / "variants.tsv.gz").exists(), "fixture: the 0.2.3 Dense copy"
    assert validate_store(legacy).failed() == frozenset()
    _blank_source_alids(legacy / "dense")
    with_fixed_width_alid_index(legacy / "dense")  # the copy's indexes, as 0.2 lays them out
    assert all(r.source_alid is None for r in _records(legacy / "dense")), "fixture: blanked"
    assert validate_store(legacy).failed() == frozenset()

    _blank_source_alids(source)
    assert validate_store(source).failed() == {RULE}


# ── the Ragged builders record `source_alid` (#449) ─────────────────────────


def _hg19_alids_of_strand_rows() -> dict[str, str]:
    """The hg38 ALID of each `STRAND_ROWS` variant, and the hg19 ALID its source spelt."""
    from liftover_fixtures import STRAND_ROWS

    from opengwasdb.variants.normalise import canonical_alid

    return {row[5]: canonical_alid(row[1], row[2], row[3], row[4]) for row in STRAND_ROWS}


def test_a_lifted_ssf_build_records_each_rows_pre_lift_alid(tmp_path: Path) -> None:
    import test_ragged_build_ssf_lift as tl
    from liftover_fixtures import write_two_strand_chain

    chain = write_two_strand_chain(tmp_path / "two-strand.chain")
    store = tl._build(tmp_path, [("a", "GRCh37", tl._GRCH37_ROWS)], "lifted", chain_file=chain)
    expected = _hg19_alids_of_strand_rows()
    got = {r.alid: r.source_alid for r in _records(store)}
    assert got == expected
    assert any(a != s for a, s in expected.items()), "nothing moved: the fixture proves nothing"


def test_an_ssf_build_with_no_lift_records_the_alid_as_its_source(tmp_path: Path) -> None:
    import test_ragged_build_ssf_lift as tl

    store = tl._build(tmp_path, [("a", "GRCh38", tl._GRCH38_ROWS)])
    got = {r.alid: r.source_alid for r in _records(store)}
    assert got and all(source == alid for alid, source in got.items())


def test_an_ssf_variant_two_sources_reach_from_different_origins_is_blank(tmp_path: Path) -> None:
    """#85's rule: the hg19 and the hg38 Analysis name a chr2 variant from different
    coordinates, so its origin is `.`. (A chr3 variant, the chain's identity, is declared
    under both builds, which the lift drops from both.)"""
    import test_ragged_build_ssf_lift as tl
    from liftover_fixtures import write_two_strand_chain

    chain = write_two_strand_chain(tmp_path / "two-strand.chain")
    store = tl._build(
        tmp_path,
        [("a", "GRCh37", tl._GRCH37_ROWS), ("b", "GRCh38", tl._GRCH38_ROWS)],
        "mixed",
        chain_file=chain,
    )
    got = {r.alid: r.source_alid for r in _records(store)}
    chr2 = {a: s for a, s in _hg19_alids_of_strand_rows().items() if a.startswith("2:")}
    assert chr2 and all(a != s for a, s in chr2.items()) and set(got) == set(chr2)
    assert all(source is None for source in got.values())


def test_a_lifted_besd_build_records_each_rows_pre_lift_alid(tmp_path: Path) -> None:
    import test_ragged_build_besd as tb
    from liftover_fixtures import write_two_strand_chain


    chain = write_two_strand_chain(tmp_path / "hg19ToHg38.chain")
    prefix = tb._write_strand_fixture(tmp_path)
    reference = tb._write_reference_alleles(
        tmp_path, [tb._REVERSE_SNP_HG38_ALID, tb._FORWARD_SNP_HG38_ALID]
    )
    out = tmp_path / "out.opengwasdb"
    build_ragged_from_besd(
        prefix, out, store_id="test", release_id="v1", source_build="hg19", chain_file=chain,
        variant_reference=reference, association_coverage="cis_and_signals",
    )
    got = {r.alid: r.source_alid for r in _records(out)}
    expected = _besd_source_alids(tb)
    assert got == expected
    assert any(a != s for a, s in expected.items()), "nothing moved: the fixture proves nothing"


def test_a_besd_build_with_no_lift_records_the_alid_as_its_source(tmp_path: Path) -> None:
    import test_ragged_build_besd as tb

    out = tmp_path / "out.opengwasdb"
    build_ragged_from_besd(
        tb._write_strand_fixture(tmp_path), out, store_id="test", release_id="v1",
        association_coverage="cis_and_signals",
    )
    got = {r.alid: r.source_alid for r in _records(out)}
    assert got and all(source == alid for alid, source in got.items())
