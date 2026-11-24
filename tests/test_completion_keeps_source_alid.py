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

from pathlib import Path

import pytest
import test_dense_completion as tdc
import test_hybrid_completion as thc
import test_ragged_completion as trc
from dense_sources import build_dense_fixture
from region_tables import edit_manifest

from opengwasdb.layouts.dense.complete import complete_dense_store
from opengwasdb.layouts.hybrid.complete import complete_hybrid_store
from opengwasdb.layouts.ragged.build_besd import build_ragged_from_besd
from opengwasdb.layouts.ragged.complete import complete_ragged_store
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
    return f"{record.chromosome}:{record.position - 1000}:{record.effect_allele}:{record.other_allele}"


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


def test_dense_completion_keeps_the_observed_rows_source_alid(tmp_path: Path) -> None:
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


def test_ragged_completion_keeps_the_observed_rows_source_alid(tmp_path: Path) -> None:
    observed = tmp_path / "obs.opengwasdb"
    build_ragged_from_besd(
        trc._make_besd_fixture(tmp_path), observed, store_id="test", release_id="obs-v1",
        tissue="Blood", association_coverage="cis",
    )
    planted = plant_source_alids(observed)
    completed = tmp_path / "comp.opengwasdb"
    complete_ragged_store(
        observed, completed, trc._make_ld_panel(tmp_path, "1", 900_000, 1_300_000),
        ancestry="EUR", cis_window_bp=500_000, min_cor=0.0, release_id="comp-v1",
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


# ── the rule ────────────────────────────────────────────────────────────────


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
    edit_manifest(
        store,
        lambda m: {
            **m,
            "provenance": {**m["provenance"], "chain_file": "pyliftover_builtin_hg19_hg38"},
        },
    )


def source_alid_findings(store: Path) -> list[str]:
    return validate_store(store, rules={RULE}).errors


def test_the_rule_fails_a_lifted_release_whose_rows_all_lack_a_source_alid(
    observed_dense: Path,
) -> None:
    assert all(r.source_alid is None for r in _records(observed_dense))  # meaningful fixture
    record_liftover(observed_dense)
    (error,) = source_alid_findings(observed_dense)
    assert "records a liftover" in error and "source_alid" in error


def test_the_rule_passes_when_one_row_carries_a_source_alid(observed_dense: Path) -> None:
    record_liftover(observed_dense)
    plant_source_alids(observed_dense, blank=len(_records(observed_dense)) - 1)
    assert sum(1 for r in _records(observed_dense) if r.source_alid) == 1
    assert source_alid_findings(observed_dense) == []


def test_the_rule_passes_a_release_that_records_no_liftover(observed_dense: Path) -> None:
    assert all(r.source_alid is None for r in _records(observed_dense))
    assert source_alid_findings(observed_dense) == []


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
    assert source_alid_findings(observed_dense) == []


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
    assert source_alid_findings(completed) == []


@pytest.mark.xfail(
    strict=True,
    reason="#449 report: this mixed hg19+hg38 Hybrid writes every shared-table source_alid blank, "
    "the lifted row's included, where an hg19 and an hg38 row reach one ALID; whether that "
    "blank is the collision rule or a drop is for #472, which owns layouts/hybrid",
)
def test_a_hybrid_build_keeps_a_folded_rows_source_alid(tmp_path: Path) -> None:
    import test_hybrid_build as thb
    from opengwasdb.layouts.hybrid.build import build_hybrid_from_vcf_manifest

    early = thb._make_vcf(
        tmp_path, "early_hg19",
        ["1\t1000000\trsEARLY\tC\tT\t.\tPASS\t.\tES:SE\t1.5:0.3\n"],
    )
    late = thb._make_vcf(
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
    by_alid = {r.alid: r.source_alid for r in _records(store)}
    assert by_alid["1:1064620:C:T"] == "1:1000000:C:T"
